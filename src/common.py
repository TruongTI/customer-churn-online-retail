"""Hàm và cấu hình dùng chung (đường dẫn, kết nối DuckDB, ngưỡng churn, snapshot đặc trưng, biểu đồ)."""
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")           # chạy được cả khi không có màn hình
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Đường dẫn: luôn tính từ thư mục gốc repo (thư mục cha của src/), nên chạy
# script từ thư mục nào cũng được, không phụ thuộc thư mục hiện hành.
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
POWERBI_DIR = ROOT / "data" / "powerbi_data"
SQL_DIR = ROOT / "sql"
DB_PATH = PROCESSED_DIR / "retail.duckdb"
REPORT_DIR = ROOT / "reports" / "reports_churn"
FIG_DIR = REPORT_DIR / "figures"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Tham số định nghĩa churn: CHỈ khai báo ở đây, mọi script import từ đây.
# ----------------------------------------------------------------------------
CHURN_DAYS = 90       # churn = không mua trong 90 ngày sau ngày snapshot (chốt ở câu hỏi 1)
ACTIVE_DAYS = 180     # khách "đang hoạt động" = có mua trong 180 ngày trước snapshot
LOOKBACK_DAYS = 270   # cửa sổ quá khứ dùng để tính đặc trưng
RECENT_DAYS = 90      # cửa sổ "hành vi gần đây" cho đặc trưng freq_90 / monetary_90

SEED = 42             # seed chung cho mọi mô hình / chia tập

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})


# ----------------------------------------------------------------------------
# Kết nối DuckDB
# ----------------------------------------------------------------------------
class DatabaseLockedError(RuntimeError):
    """File DuckDB đang bị một tiến trình khác (thường là DBeaver) giữ khóa."""


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """
    Mở data/processed/retail.duckdb.

    DuckDB chỉ cho MỘT tiến trình mở file khi có kết nối ghi; nếu DBeaver đang
    kết nối tới file thì Python sẽ bị từ chối (kể cả read_only). Hàm này đổi
    lỗi khó hiểu đó thành hướng dẫn cụ thể.
    """
    if read_only and not DB_PATH.exists():
        raise FileNotFoundError(
            f"Chưa có {DB_PATH}. Hãy chạy `python src/clean_data.py` và "
            f"`python src/build_star_schema.py` trước.")
    try:
        return duckdb.connect(str(DB_PATH), read_only=read_only)
    except duckdb.IOException as e:
        if "lock" in str(e).lower():
            raise DatabaseLockedError(
                f"Không mở được {DB_PATH.name}: file đang bị chương trình khác giữ (thường là DBeaver).\n"
                f"-> Trong DBeaver: chuột phải vào kết nối retail.duckdb -> Disconnect (hoặc đóng DBeaver), "
                f"rồi chạy lại script.\nChi tiết gốc: {e}") from e
        raise


def check_star_schema_fresh(con) -> None:
    """
    Dừng chương trình nếu Fact_Sales không khớp clean_retail (ví dụ đã chạy lại
    clean_data.py nhưng quên chạy lại star_schema) -> tránh phân tích trên dữ liệu cũ.
    """
    tables = {t[0] for t in con.execute("SHOW TABLES").fetchall()}
    missing = {"clean_retail", "Fact_Sales", "Dim_Customer"} - tables
    if missing:
        raise RuntimeError(f"Thiếu bảng {sorted(missing)} trong {DB_PATH.name}. "
                           f"Hãy chạy `python src/build_star_schema.py`.")
    clean = con.execute("SELECT COUNT(*), ROUND(SUM(TotalPrice), 2) FROM clean_retail").fetchone()
    fact = con.execute("SELECT COUNT(*), ROUND(SUM(Revenue), 2) FROM Fact_Sales").fetchone()
    if clean != fact:
        raise RuntimeError(
            f"Fact_Sales đã cũ so với clean_retail (Fact_Sales: {fact[0]:,} dòng / {fact[1]:,.2f}; "
            f"clean_retail: {clean[0]:,} dòng / {clean[1]:,.2f}).\n"
            f"-> Hãy chạy lại `python src/build_star_schema.py` rồi chạy lại script này.")


def load_lines() -> pd.DataFrame:
    """Đọc Fact_Sales (mức dòng hàng) + quốc gia khách. Thêm cột `day` (ngày, không giờ)."""
    con = connect(read_only=True)
    try:
        check_star_schema_fresh(con)
        df = con.execute("""
            SELECT f.CustomerID, f.Invoice, f.StockCode, f.InvoiceDate, f.Revenue, c.Country
            FROM Fact_Sales f JOIN Dim_Customer c USING (CustomerID)
        """).fetchdf()
    finally:
        con.close()
    df["day"] = pd.to_datetime(df["InvoiceDate"]).dt.normalize()
    return df


def data_end(lines: pd.DataFrame) -> pd.Timestamp:
    """Ngày giao dịch cuối cùng trong dữ liệu (dùng làm ngày tham chiếu)."""
    return lines["day"].max()


# ----------------------------------------------------------------------------
# Snapshot: đặc trưng tính đến ngày `snap` + nhãn churn trong `horizon` ngày sau
# ----------------------------------------------------------------------------
FEATURES = ["recency", "frequency", "monetary", "aov", "n_products", "tenure",
            "avg_gap", "recency_ratio", "freq_90", "monetary_90", "freq_share_90", "is_uk"]


def build_snapshot(lines, snap, lookback=LOOKBACK_DAYS, active_days=ACTIVE_DAYS, horizon=CHURN_DAYS):
    """
    Trả về 1 dòng / khách hàng "đang hoạt động" tại ngày `snap`.

    - Đặc trưng chỉ dùng dữ liệu trong cửa sổ (snap - lookback, snap]  -> KHÔNG rò rỉ tương lai.
    - Khách hàng hoạt động = có mua trong `active_days` ngày cuối trước snap.
    - Nhãn churn = 1 nếu KHÔNG mua gì trong (snap, snap + horizon]; NaN nếu cửa sổ nhãn vượt quá dữ liệu.
    - rev90_est = doanh thu ước tính trong `horizon` ngày tới (cùng độ dài cửa sổ nhãn churn).
    """
    s = pd.Timestamp(snap)
    end = data_end(lines)
    w = lines[(lines["day"] > s - pd.Timedelta(days=lookback)) & (lines["day"] <= s)]

    g = w.groupby("CustomerID")
    f = pd.DataFrame({
        "last_day": g["day"].max(),
        "first_day": g["day"].min(),
        "frequency": g["Invoice"].nunique(),
        "monetary": g["Revenue"].sum(),
        "n_products": g["StockCode"].nunique(),
        "purchase_days": g["day"].nunique(),
        "is_uk": g["Country"].first().eq("United Kingdom").astype(int),
    })
    f["recency"] = (s - f["last_day"]).dt.days
    f["tenure"] = (s - f["first_day"]).dt.days
    f["aov"] = f["monetary"] / f["frequency"]

    # khoảng cách trung bình giữa các ngày mua (nếu chỉ mua 1 ngày -> điền = lookback)
    d = w[["CustomerID", "day"]].drop_duplicates().sort_values(["CustomerID", "day"])
    d["gap"] = d.groupby("CustomerID")["day"].diff().dt.days
    f["avg_gap"] = d.groupby("CustomerID")["gap"].mean().reindex(f.index).fillna(lookback)
    f["recency_ratio"] = f["recency"] / f["avg_gap"].clip(lower=1)

    # hành vi RECENT_DAYS (90) ngày gần nhất
    r = w[w["day"] > s - pd.Timedelta(days=RECENT_DAYS)].groupby("CustomerID")
    f["freq_90"] = r["Invoice"].nunique().reindex(f.index).fillna(0)
    f["monetary_90"] = r["Revenue"].sum().reindex(f.index).fillna(0)
    f["freq_share_90"] = f["freq_90"] / f["frequency"]

    # ước tính doanh thu / `horizon` ngày (sàn = horizon để khách mới không bị phóng đại)
    observed = (f["tenure"] + 1).clip(lower=horizon)
    f["rev90_est"] = f["monetary"] / observed * horizon

    f = f[f["recency"] <= active_days].copy()

    if s + pd.Timedelta(days=horizon) <= end:
        fut = lines[(lines["day"] > s) & (lines["day"] <= s + pd.Timedelta(days=horizon))]
        returned = set(fut["CustomerID"].unique())
        f["churn"] = (~f.index.isin(returned)).astype(int)
    else:
        f["churn"] = np.nan
    f["snapshot"] = s
    return f.drop(columns=["last_day", "first_day", "purchase_days"])


# ----------------------------------------------------------------------------
# Mô hình dùng chung (04c và 04d phải dùng ĐÚNG cùng cấu hình)
# ----------------------------------------------------------------------------
def make_lr():
    """Logistic Regression: log1p -> chuẩn hóa -> hồi quy logistic."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    return make_pipeline(FunctionTransformer(np.log1p), StandardScaler(),
                         LogisticRegression(max_iter=2000, random_state=SEED))


def make_rf():
    """Random Forest dùng cho dự báo churn và chấm điểm Top 500."""
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10,
                                  random_state=SEED, n_jobs=-1)
