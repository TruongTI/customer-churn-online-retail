"""Hàm dùng chung cho Phase 4 (đọc dữ liệu, tạo snapshot đặc trưng, vẽ biểu đồ)."""
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")           # chạy được cả khi không có màn hình
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DB_PATH = Path("data/processed/retail.duckdb")
REPORT_DIR = Path("reports/phase4")
FIG_DIR = REPORT_DIR / "figures"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})


def load_lines() -> pd.DataFrame:
    """Đọc Fact_Sales (mức dòng hàng) + quốc gia khách. Thêm cột `day` (ngày, không giờ)."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("""
        SELECT f.CustomerID, f.Invoice, f.StockCode, f.InvoiceDate, f.Revenue, c.Country
        FROM Fact_Sales f JOIN Dim_Customer c USING (CustomerID)
    """).fetchdf()
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


def build_snapshot(lines, snap, lookback=270, active_days=180, horizon=90):
    """
    Trả về 1 dòng / khách hàng "đang hoạt động" tại ngày `snap`.

    - Đặc trưng chỉ dùng dữ liệu trong cửa sổ (snap - lookback, snap]  -> KHÔNG rò rỉ tương lai.
    - Khách hàng hoạt động = có mua trong `active_days` ngày cuối trước snap.
    - Nhãn churn = 1 nếu KHÔNG mua gì trong (snap, snap + horizon]; NaN nếu cửa sổ nhãn vượt quá dữ liệu.
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

    # hành vi 90 ngày gần nhất
    r = w[w["day"] > s - pd.Timedelta(days=90)].groupby("CustomerID")
    f["freq_90"] = r["Invoice"].nunique().reindex(f.index).fillna(0)
    f["monetary_90"] = r["Revenue"].sum().reindex(f.index).fillna(0)
    f["freq_share_90"] = f["freq_90"] / f["frequency"]

    # ước tính doanh thu / 90 ngày (sàn 90 ngày để khách mới không bị phóng đại)
    observed = (f["tenure"] + 1).clip(lower=90)
    f["rev90_est"] = f["monetary"] / observed * 90

    f = f[f["recency"] <= active_days].copy()

    if s + pd.Timedelta(days=horizon) <= end:
        fut = lines[(lines["day"] > s) & (lines["day"] <= s + pd.Timedelta(days=horizon))]
        returned = set(fut["CustomerID"].unique())
        f["churn"] = (~f.index.isin(returned)).astype(int)
    else:
        f["churn"] = np.nan
    f["snapshot"] = s
    return f.drop(columns=["last_day", "first_day", "purchase_days"])
