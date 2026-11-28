"""
Làm sạch dữ liệu Online Retail II.

Chạy (từ thư mục nào cũng được):
    python src/clean_data.py
Bước tiếp theo: python src/build_star_schema.py

Đầu vào : data/raw/online_retail_II.csv
Đầu ra  : data/processed/online_retail_clean.parquet
          data/processed/retail.duckdb  (bảng clean_retail, cleaning_log)
          reports/cleaning_log.csv      (nhật ký số dòng bị loại ở mỗi bước)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, RAW_DIR, PROCESSED_DIR, DB_PATH, connect  # noqa: E402

# ------------------------------------------------------------------ cấu hình
RAW_CSV = RAW_DIR / "online_retail_II.csv"
OUT_PARQUET = PROCESSED_DIR / "online_retail_clean.parquet"
OUT_DUCKDB = DB_PATH
OUT_LOG = ROOT / "reports" / "cleaning_log.csv"

if not RAW_CSV.exists():
    sys.exit(f"Không tìm thấy {RAW_CSV}. Tải Online Retail II và đặt file CSV vào data/raw/ (xem README mục 3.3).")

# Kiểm tra sớm: nếu DBeaver đang giữ file DuckDB thì báo ngay, không đợi làm sạch xong
OUT_DUCKDB.parent.mkdir(parents=True, exist_ok=True)
connect().close()

# Bước bổ sung ngoài checklist (bật/tắt được, xem giải thích trong hướng dẫn)
DROP_DUPLICATES = True         # bỏ dòng trùng lặp hoàn toàn
DROP_FULLY_CANCELLED = True    # bỏ dòng bán đã bị hủy đúng số lượng về sau

# StockCode không phải sản phẩm thật (đã chuẩn hóa IN HOA)
NON_PRODUCT_CODES = {
    "POST", "DOT", "M", "C2", "D", "S", "B", "PADS",
    "BANK CHARGES", "ADJUST", "ADJUST2", "AMAZONFEE", "CRUK",
    "TEST001", "TEST002",
}
NON_PRODUCT_REGEX = r"^GIFT_0001_\d+$"   # voucher quà tặng: gift_0001_20, ...

log = []  # nhật ký các bước làm sạch


def record(step, df_before, df_after, flagged=None):
    """Ghi lại số dòng bị loại ở mỗi bước."""
    log.append({
        "step": step,
        "rows_flagged_independent": flagged,   # số dòng vi phạm quy tắc (tính trên dữ liệu gốc)
        "rows_removed": len(df_before) - len(df_after),
        "rows_remaining": len(df_after),
    })
    print(f"{step:<45} loại {len(df_before) - len(df_after):>8,}  còn {len(df_after):>9,}")


# ------------------------------------------------------------------ 1. đọc dữ liệu
df = pd.read_csv(
    RAW_CSV,
    dtype={"Invoice": "string", "StockCode": "string", "Description": "string", "Country": "string"},
    parse_dates=["InvoiceDate"],
)
df["StockCode"] = df["StockCode"].str.strip().str.upper()   # '84797b' và '84797B' là một
df["Invoice"] = df["Invoice"].str.strip()
print(f"Số dòng gốc: {len(df):,}\n")
log.append({"step": "0. Dữ liệu gốc", "rows_flagged_independent": None,
            "rows_removed": 0, "rows_remaining": len(df)})

# Cờ độc lập (đếm trên dữ liệu gốc, để đưa vào báo cáo)
is_cancel = df["Invoice"].str.startswith("C")
flag_null_cust = df["Customer ID"].isna()
flag_returns = (df["Quantity"] < 0) | is_cancel
flag_price = df["Price"] <= 0
flag_nonprod = df["StockCode"].isin(NON_PRODUCT_CODES) | df["StockCode"].str.match(NON_PRODUCT_REGEX)

# ------------------------------------------------------------------ 2. lọc theo checklist
step = df.dropna(subset=["Customer ID"])
record("1. Bỏ Customer ID rỗng", df, step, flag_null_cust.sum()); df = step

step = df[(df["Quantity"] > 0) & (~df["Invoice"].str.startswith("C"))]
record("2. Bỏ đơn trả/hủy (Quantity<0 hoặc Invoice 'C')", df, step, flag_returns.sum()); df = step

step = df[df["Price"] > 0]
record("3. Bỏ Price <= 0", df, step, flag_price.sum()); df = step

step = df[~(df["StockCode"].isin(NON_PRODUCT_CODES) | df["StockCode"].str.match(NON_PRODUCT_REGEX))]
record("4. Bỏ StockCode không phải sản phẩm", df, step, flag_nonprod.sum()); df = step

# ------------------------------------------------------------------ 3. bước bổ sung
if DROP_DUPLICATES:
    step = df.drop_duplicates()
    record("5. Bỏ dòng trùng lặp hoàn toàn", df, step); df = step

if DROP_FULLY_CANCELLED:
    # Lấy các dòng hủy của TẤT CẢ dữ liệu gốc có Customer ID.
    # Đọc đủ 8 cột và chuẩn hóa giống hệt bước 1 để tiêu chí "trùng lặp" của
    # dòng hủy khớp với tiêu chí đã dùng cho dòng bán ở bước 5.
    raw = pd.read_csv(
        RAW_CSV,
        dtype={"Invoice": "string", "StockCode": "string", "Description": "string", "Country": "string"},
        parse_dates=["InvoiceDate"],
    )
    raw["StockCode"] = raw["StockCode"].str.strip().str.upper()
    raw["Invoice"] = raw["Invoice"].str.strip()
    raw = raw.dropna(subset=["Customer ID"])
    canc = raw[raw["Invoice"].str.startswith("C")].copy()
    if DROP_DUPLICATES:
        # Dòng bán đã bỏ trùng lặp ở bước 5 -> dòng hủy cũng phải bỏ trùng lặp,
        # nếu không 1 dòng hủy bị nhân đôi sẽ "nhận" thêm 1 dòng bán hợp lệ khác.
        n_dup_canc = int(canc.duplicated().sum())
        canc = canc.drop_duplicates()
        print(f"   (bỏ {n_dup_canc:,} dòng hủy trùng lặp hoàn toàn trước khi ghép)")
    canc["Quantity"] = -canc["Quantity"]                       # đổi về số dương để khớp với dòng bán
    canc = canc.rename(columns={"InvoiceDate": "cancel_date"})[
        ["Customer ID", "StockCode", "Quantity", "cancel_date"]]

    # Ghép 1-1 theo thời gian (LIFO): mỗi dòng hủy chỉ được phép hủy ĐÚNG MỘT
    # dòng bán cùng khóa (Customer ID, StockCode, Quantity) có InvoiceDate gần
    # nhất TRƯỚC nó và CHƯA bị dòng hủy nào khác "nhận" trước đó.
    #
    # Lưu ý: nếu ghép bằng merge thông thường trên khóa (Customer ID, StockCode,
    # Quantity), MỘT dòng hủy có thể khớp nhầm với NHIỀU dòng bán trùng khóa
    # (ví dụ khách mua cùng sản phẩm, cùng số lượng ở 2 hóa đơn khác nhau)
    # -> xóa oan các đơn hợp lệ. Thuật toán dưới đây đảm bảo quan hệ 1-1.
    #
    # Đổi tên cột trước khi dùng itertuples để tránh pandas tự đổi tên field
    # (cột có khoảng trắng hoặc bắt đầu bằng "_" sẽ bị itertuples đổi thành
    # "_0", "_1"... một cách không tường minh, dễ gây lỗi thầm lặng).
    tmp = (df.reset_index()
             .rename(columns={"index": "rid", "Customer ID": "CustID"})
             [["rid", "CustID", "StockCode", "Quantity", "InvoiceDate"]])
    canc_r = canc.rename(columns={"Customer ID": "CustID"})

    from collections import defaultdict, deque

    events = []  # (thời điểm, 0=bán/1=hủy ưu tiên xử lý sau trong cùng thời điểm, khóa, rid)
    for row in tmp.itertuples(index=False):
        events.append((row.InvoiceDate, 0, (row.CustID, row.StockCode, row.Quantity), row.rid))
    for row in canc_r.itertuples(index=False):
        events.append((row.cancel_date, 1, (row.CustID, row.StockCode, row.Quantity), None))
    events.sort(key=lambda e: (e[0], e[1]))   # cùng thời điểm: xử lý "bán" (0) trước "hủy" (1)

    pending = defaultdict(deque)  # khóa -> deque các rid đang chờ, theo thứ tự thời gian bán
    cancelled_ids = set()
    for _, kind, key, rid in events:
        if kind == 0:
            pending[key].append(rid)
        elif pending[key]:
            cancelled_ids.add(pending[key].pop())    # LIFO: hủy đơn bán gần nhất còn lại

    step = df[~df.index.isin(cancelled_ids)]
    record("6. Bỏ dòng bán đã bị hủy đúng số lượng", df, step); df = step

# ------------------------------------------------------------------ 4. chuẩn hóa & tạo trường mới
df = df.copy()
df["CustomerID"] = df["Customer ID"].astype("int64")
df["TotalPrice"] = (df["Quantity"] * df["Price"]).round(2)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["InvoiceDay"] = df["InvoiceDate"].dt.normalize()            # ngày, bỏ giờ
df["Description"] = df["Description"].str.strip()
df["Country"] = df["Country"].str.strip()

df = df[["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate",
         "InvoiceDay", "Price", "TotalPrice", "CustomerID", "Country"]].reset_index(drop=True)

# ------------------------------------------------------------------ 5. kiểm tra chất lượng
assert df["CustomerID"].notna().all(), "Còn Customer ID rỗng"
assert (df["Quantity"] > 0).all(), "Còn Quantity <= 0"
assert (df["Price"] > 0).all(), "Còn Price <= 0"
assert not df["Invoice"].str.startswith("C").any(), "Còn Invoice bắt đầu bằng C"
assert (df["TotalPrice"] > 0).all(), "Còn TotalPrice <= 0"
print("\nCác kiểm tra chất lượng: ĐẠT")
print(f"Khoảng thời gian: {df['InvoiceDate'].min()}  ->  {df['InvoiceDate'].max()}")
print(f"Số khách hàng: {df['CustomerID'].nunique():,} | Số hóa đơn: {df['Invoice'].nunique():,} "
      f"| Số sản phẩm: {df['StockCode'].nunique():,} | Tổng doanh thu: {df['TotalPrice'].sum():,.0f}")

# ------------------------------------------------------------------ 6. lưu kết quả
OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
OUT_LOG.parent.mkdir(parents=True, exist_ok=True)

df.to_parquet(OUT_PARQUET, index=False)

log_df = pd.DataFrame(log)
log_df.to_csv(OUT_LOG, index=False, encoding="utf-8-sig")

con = connect()
con.execute("CREATE OR REPLACE TABLE clean_retail AS SELECT * FROM df")
con.execute("CREATE OR REPLACE TABLE cleaning_log AS SELECT * FROM log_df")
print("\nBảng trong DuckDB:", con.execute("SHOW TABLES").fetchall())
con.close()
print(f"Đã lưu: {OUT_PARQUET}, {OUT_DUCKDB}, {OUT_LOG}")
print("Bước tiếp theo: python src/build_star_schema.py")