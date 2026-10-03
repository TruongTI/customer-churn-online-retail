"""
Làm sạch dữ liệu Online Retail II.

Chạy từ thư mục gốc của repo:
    python src/clean_data.py

Đầu vào : data/raw/online_retail_II.csv
Đầu ra  : data/processed/online_retail_clean.parquet
          data/processed/retail.duckdb  (bảng clean_retail, cleaning_log)
          reports/cleaning_log.csv      (nhật ký số dòng bị loại ở mỗi bước)
"""
from pathlib import Path

import duckdb
import pandas as pd

# ------------------------------------------------------------------ cấu hình
RAW_CSV = Path("data/raw/online_retail_II.csv")
OUT_PARQUET = Path("data/processed/online_retail_clean.parquet")
OUT_DUCKDB = Path("data/processed/retail.duckdb")
OUT_LOG = Path("reports/cleaning_log.csv")

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
    # Lấy các dòng hủy của TẤT CẢ dữ liệu gốc có Customer ID
    raw = pd.read_csv(RAW_CSV, dtype={"Invoice": "string", "StockCode": "string"},
                      parse_dates=["InvoiceDate"], usecols=["Invoice", "StockCode", "Quantity",
                                                            "InvoiceDate", "Customer ID"])
    raw = raw.dropna(subset=["Customer ID"])
    raw["StockCode"] = raw["StockCode"].str.strip().str.upper()
    canc = raw[raw["Invoice"].str.startswith("C")].copy()
    canc["Quantity"] = -canc["Quantity"]                       # đổi về số dương để khớp với dòng bán
    canc = canc.rename(columns={"InvoiceDate": "cancel_date"})[
        ["Customer ID", "StockCode", "Quantity", "cancel_date"]]

    tmp = df.reset_index().rename(columns={"index": "_rid"})
    m = tmp.merge(canc, on=["Customer ID", "StockCode", "Quantity"], how="inner")
    m = m[m["cancel_date"] >= m["InvoiceDate"]]                # hủy phải xảy ra SAU khi bán
    cancelled_ids = set(m["_rid"])
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

con = duckdb.connect(str(OUT_DUCKDB))
con.execute("CREATE OR REPLACE TABLE clean_retail AS SELECT * FROM df")
con.execute("CREATE OR REPLACE TABLE cleaning_log AS SELECT * FROM log_df")
print("\nBảng trong DuckDB:", con.execute("SHOW TABLES").fetchall())
con.close()
print(f"Đã lưu: {OUT_PARQUET}, {OUT_DUCKDB}, {OUT_LOG}")
