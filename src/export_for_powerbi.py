"""
Xuất dữ liệu cho Power BI.

Power BI Desktop không có connector DuckDB có sẵn, nên ta xuất các bảng
cần thiết ra CSV (mã hóa UTF-8 có BOM để Power BI đọc đúng tiếng Việt/ký
tự đặc biệt) vào thư mục data/powerbi_data/, rồi Power BI import trực tiếp
các file CSV này.

Chạy: python src/export_for_powerbi.py   (sau 04b và 04c)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import POWERBI_DIR, connect, check_star_schema_fresh  # noqa: E402

OUT_DIR = POWERBI_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

con = connect(read_only=True)
check_star_schema_fresh(con)       # không xuất Star Schema cũ sang Power BI
_tables = {t[0] for t in con.execute("SHOW TABLES").fetchall()}
for _t, _script in [("rfm_segments", "04b_rfm_segmentation.py"), ("churn_scores", "04c_churn_model_top500.py")]:
    if _t not in _tables:
        sys.exit(f"Thiếu bảng {_t}. Hãy chạy `python src/{_script}` trước.")

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

# ------------------------------------------------------------------ Monthly_Overview
# Nguồn cho biểu đồ xu hướng doanh thu theo tháng (trang Executive Overview).
# Giữ nguyên tên/thứ tự 8 cột mà BI.pbix đang dùng; cột IsPartialMonth thêm ở
# cuối để lọc tháng chưa đủ dữ liệu (12/2011 chỉ có đến ngày 09).
# MonthStart = ngày đầu tháng theo lịch (không phải ngày giao dịch đầu tiên).
monthly = con.execute("""
    WITH m AS (
        SELECT
            d.Year,
            d.Month,
            d.YearMonthKey,
            CAST(date_trunc('month', d.FullDate) AS DATE) AS MonthStart,
            COUNT(DISTINCT f.Invoice)                     AS SoHoaDon,
            COUNT(DISTINCT f.CustomerID)                  AS SoKhachHang,
            ROUND(SUM(f.Revenue), 0)                      AS DoanhThu,
            ROUND(SUM(f.Revenue) / COUNT(DISTINCT f.Invoice), 2) AS AOV
        FROM Fact_Sales f
        JOIN Dim_Date d ON f.DateKey = d.DateKey
        GROUP BY d.Year, d.Month, d.YearMonthKey, MonthStart
    )
    SELECT m.*,
           -- tháng chưa kết thúc tại ngày cuối dữ liệu -> đánh dấu là tháng thiếu
           (SELECT MAX(FullDate) FROM Dim_Date) < last_day(m.MonthStart) AS IsPartialMonth
    FROM m
    ORDER BY m.YearMonthKey
""").fetchdf()
monthly.to_csv(OUT_DIR / "Monthly_Overview.csv", index=False, encoding="utf-8-sig")
print(f"Monthly_Overview: {len(monthly):,} dòng "
      f"(tháng thiếu dữ liệu: {monthly.loc[monthly['IsPartialMonth'], 'YearMonthKey'].tolist()})")

# ------------------------------------------------------------------ Cohort_Retention
# Nguồn cho ma trận Cohort Retention (trang Executive Overview).
# Cohort = tháng mua đầu tiên; MonthIndex = số tháng kể từ tháng đầu (0 = tháng đầu).
# Giới hạn 12 tháng như bản dashboard hiện tại.
MAX_COHORT_MONTHS = 12
cohort = con.execute("""
    WITH cohort AS (
        SELECT CustomerID,
               CAST(date_trunc('month', FirstPurchaseDate) AS DATE) AS CohortMonth
        FROM Dim_Customer
    ),
    activity AS (
        SELECT DISTINCT f.CustomerID,
               CAST(date_trunc('month', d.FullDate) AS DATE) AS ActivityMonth
        FROM Fact_Sales f
        JOIN Dim_Date d ON f.DateKey = d.DateKey
    ),
    size AS (
        SELECT CohortMonth, COUNT(*) AS SoKhachBanDau
        FROM cohort
        GROUP BY CohortMonth
    ),
    ret AS (
        SELECT co.CohortMonth,
               CAST(date_diff('month', co.CohortMonth, a.ActivityMonth) AS INTEGER) AS MonthIndex,
               COUNT(DISTINCT a.CustomerID) AS SoKhachConHoatDong
        FROM cohort co
        JOIN activity a USING (CustomerID)
        GROUP BY co.CohortMonth, MonthIndex
    )
    SELECT r.CohortMonth,
           r.MonthIndex,
           r.SoKhachConHoatDong,
           s.SoKhachBanDau,
           ROUND(100.0 * r.SoKhachConHoatDong / s.SoKhachBanDau, 1) AS RetentionRate_Pct
    FROM ret r
    JOIN size s USING (CohortMonth)
    WHERE r.MonthIndex <= ?
    ORDER BY r.CohortMonth, r.MonthIndex
""", [MAX_COHORT_MONTHS]).fetchdf()

# Kiểm tra nhanh: tháng 0 của mọi cohort phải giữ lại 100% khách
assert (cohort.loc[cohort["MonthIndex"] == 0, "RetentionRate_Pct"] == 100.0).all(), \
    "Cohort_Retention: tháng 0 không đạt 100% -> kiểm tra lại Dim_Customer/Fact_Sales"
cohort.to_csv(OUT_DIR / "Cohort_Retention.csv", index=False, encoding="utf-8-sig")
print(f"Cohort_Retention: {len(cohort):,} dòng, {cohort['CohortMonth'].nunique()} cohort")

con.close()
print(f"\nĐã xuất xong vào thư mục: {OUT_DIR.resolve()}")
