"""
Xuất dữ liệu cho Power BI.

Power BI Desktop không có connector DuckDB có sẵn, nên ta xuất các bảng
cần thiết ra CSV (mã hóa UTF-8 có BOM để Power BI đọc đúng tiếng Việt/ký
tự đặc biệt) vào thư mục data/powerbi_data/, rồi Power BI import trực tiếp
các file CSV này.

Chạy: python src/export_for_powerbi.py
"""
from pathlib import Path

import duckdb
import pandas as pd

DB_PATH = "data/processed/retail.duckdb"
OUT_DIR = Path("data/powerbi_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

con = duckdb.connect(DB_PATH, read_only=True)

# ------------------------------------------------------------------ Fact_Sales
fact = con.execute("SELECT * FROM Fact_Sales").fetchdf()
fact.to_csv(OUT_DIR / "Fact_Sales.csv", index=False, encoding="utf-8-sig")
print(f"Fact_Sales: {len(fact):,} dòng")

# ------------------------------------------------------------------ Dim_Date
dim_date = con.execute("SELECT * FROM Dim_Date").fetchdf()
dim_date.to_csv(OUT_DIR / "Dim_Date.csv", index=False, encoding="utf-8-sig")
print(f"Dim_Date: {len(dim_date):,} dòng")

# ------------------------------------------------------------------ Dim_Product
dim_product = con.execute("SELECT * FROM Dim_Product").fetchdf()
dim_product.to_csv(OUT_DIR / "Dim_Product.csv", index=False, encoding="utf-8-sig")
print(f"Dim_Product: {len(dim_product):,} dòng")

# ------------------------------------------------------------------ Dim_Customer (gộp thêm RFM + Churn)
dim_customer = con.execute("SELECT * FROM Dim_Customer").fetchdf()

rfm = con.execute("""
    SELECT CustomerID, Segment AS RFM_Segment, Cluster AS KMeans_Cluster,
           R, F, M
    FROM rfm_segments
""").fetchdf()

churn = con.execute("""
    SELECT CustomerID,
           p_churn        AS Churn_Probability,
           rev_at_risk     AS Revenue_At_Risk,
           rank            AS Churn_Risk_Rank,
           top500          AS Is_Top500
    FROM churn_scores
""").fetchdf()

dim_customer = (dim_customer
                 .merge(rfm, on="CustomerID", how="left")
                 .merge(churn, on="CustomerID", how="left"))

# Khách không nằm trong churn_scores (không "đang hoạt động" tại ngày cuối
# dữ liệu, ví dụ chỉ mua 1 lần rất lâu rồi) -> không có điểm churn.
# Gán nhãn rõ ràng thay vì để trống, để Power BI không hiểu nhầm là lỗi.
dim_customer["Is_Top500"] = dim_customer["Is_Top500"].fillna(False)
dim_customer["Churn_Status"] = dim_customer["Churn_Probability"].apply(
    lambda x: "Không đủ điều kiện chấm điểm" if pd.isna(x) else "Đã chấm điểm"
)

dim_customer.to_csv(OUT_DIR / "Dim_Customer.csv", index=False, encoding="utf-8-sig")
print(f"Dim_Customer (đã gộp RFM + Churn): {len(dim_customer):,} dòng, {dim_customer.shape[1]} cột")
print("Cột:", list(dim_customer.columns))

con.close()
print(f"\nĐã xuất xong vào thư mục: {OUT_DIR.resolve()}")
