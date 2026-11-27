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

try:
    import openpyxl  # noqa: F401  (pandas dùng để ghi file Excel Top 500)
except ImportError:
    sys.exit("Thiếu thư viện openpyxl. Hãy chạy: pip install -r requirements.txt")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, POWERBI_DIR, connect, check_star_schema_fresh  # noqa: E402

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

# ------------------------------------------------------------------ Top 500 -> Excel
# Power BI Desktop chỉ xuất dữ liệu visual ra .csv, nên danh sách Top 500 được
# xuất sẵn ra Excel cho bộ phận kinh doanh: sắp xếp theo Revenue at Risk giảm dần,
# tiêu đề tiếng Việt, có bộ lọc, cố định dòng tiêu đề, kèm sheet giải thích.
top500 = con.execute("""
    SELECT s.rank                         AS "Hạng ưu tiên",
           s.CustomerID                   AS "Mã khách hàng",
           c.Country                      AS "Quốc gia",
           s.Segment                      AS "Phân khúc RFM",
           s.p_churn                      AS "Xác suất churn",
           s.rev90_est                    AS "Doanh thu ước tính 90 ngày (GBP)",
           s.rev_at_risk                  AS "Revenue at Risk (GBP)",
           s.recency                      AS "Số ngày từ lần mua cuối",
           s.frequency                    AS "Số hóa đơn (270 ngày gần nhất)",
           CAST(c.LastPurchaseDate AS DATE)  AS "Ngày mua gần nhất",
           CAST(c.FirstPurchaseDate AS DATE) AS "Ngày mua đầu tiên",
           ROUND(c.TotalRevenue, 2)       AS "Tổng doanh thu lịch sử (GBP)"
    FROM churn_scores s
    JOIN Dim_Customer c USING (CustomerID)
    WHERE s.top500
    ORDER BY s.rank
""").fetchdf()
snapshot_date = con.execute("SELECT CAST(MAX(InvoiceDate) AS DATE) FROM Fact_Sales").fetchone()[0]
total_rar = con.execute("SELECT SUM(rev_at_risk) FROM churn_scores").fetchone()[0]
n_scored = con.execute("SELECT COUNT(*) FROM churn_scores").fetchone()[0]

assert len(top500) == 500, f"Top 500 có {len(top500)} dòng"
assert top500["Revenue at Risk (GBP)"].is_monotonic_decreasing, "Top 500 chưa sắp xếp theo Revenue at Risk"

# Gợi ý hành động theo phân khúc (là GỢI Ý, bộ phận kinh doanh điều chỉnh theo thực tế)
ACTIONS = {
    "Champions": "Gọi điện chăm sóc cá nhân, ưu đãi VIP / giao hàng ưu tiên",
    "Loyal": "Email cá nhân hóa + ưu đãi khi đặt đơn tiếp theo",
    "Potential Loyalist": "Gợi ý sản phẩm theo lịch sử mua, khuyến khích đơn tiếp theo",
    "Need Attention": "Voucher có thời hạn ngắn để kéo khách quay lại",
    "At Risk": "Liên hệ hỏi lý do ngừng mua + ưu đãi win-back",
    "Hibernating": "Email win-back chi phí thấp",
}
top500["Hành động gợi ý"] = top500["Phân khúc RFM"].map(ACTIONS).fillna("Chăm sóc theo đánh giá của bộ phận kinh doanh")

seg_summary = (top500.groupby("Phân khúc RFM")
               .agg(**{"Số khách": ("Mã khách hàng", "count"),
                       "Tổng Revenue at Risk (GBP)": ("Revenue at Risk (GBP)", "sum"),
                       "Xác suất churn trung bình": ("Xác suất churn", "mean")})
               .sort_values("Tổng Revenue at Risk (GBP)", ascending=False)
               .reset_index())

notes = pd.DataFrame({"Nội dung": [
    "Danh sách 500 khách hàng ưu tiên chăm sóc trong 90 ngày tới",
    f"Ngày tham chiếu (giao dịch cuối cùng trong dữ liệu): {snapshot_date:%d/%m/%Y}",
    f"Số khách đang hoạt động được chấm điểm: {n_scored:,} (có mua trong 180 ngày trước ngày tham chiếu)".replace(",", "."),
    f"Top 500 chiếm {100 * top500['Revenue at Risk (GBP)'].sum() / total_rar:.1f}% tổng Revenue at Risk".replace(".", ","),
    "",
    "Giải thích cột:",
    "Xác suất churn: xác suất khách KHÔNG mua trong 90 ngày tới (mô hình Random Forest huấn luyện trên các snapshot lịch sử 2010–2011)",
    "Doanh thu ước tính 90 ngày: doanh thu trung bình mỗi 90 ngày, ước từ 270 ngày lịch sử trước ngày tham chiếu",
    "Revenue at Risk = Xác suất churn × Doanh thu ước tính 90 ngày (giá trị kỳ vọng có thể mất, KHÔNG phải doanh thu chắc chắn mất)",
    "Hạng ưu tiên: xếp theo Revenue at Risk giảm dần (hạng 1 = cần chăm sóc trước nhất)",
    "Hành động gợi ý: gợi ý theo phân khúc RFM, bộ phận kinh doanh điều chỉnh theo thực tế",
    "",
    "Nguồn: src/export_for_powerbi.py (chạy lại bằng python src/run_pipeline.py)",
]})

XLSX_PATH = ROOT / "reports" / "Top500_KhachHang_ChamSoc.xlsx"
with pd.ExcelWriter(XLSX_PATH, engine="openpyxl") as xw:
    top500.to_excel(xw, sheet_name="Top 500", index=False)
    seg_summary.to_excel(xw, sheet_name="Tóm tắt theo phân khúc", index=False)
    notes.to_excel(xw, sheet_name="Ghi chú", index=False)

    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    header_fill = PatternFill("solid", fgColor="F4B183")
    formats = {
        "Xác suất churn": "0.0%", "Xác suất churn trung bình": "0.0%",
        "Doanh thu ước tính 90 ngày (GBP)": "#,##0.00", "Revenue at Risk (GBP)": "#,##0.00",
        "Tổng Revenue at Risk (GBP)": "#,##0.00", "Tổng doanh thu lịch sử (GBP)": "#,##0.00",
        "Ngày mua gần nhất": "dd/mm/yyyy", "Ngày mua đầu tiên": "dd/mm/yyyy",
    }
    for ws, df in [(xw.sheets["Top 500"], top500), (xw.sheets["Tóm tắt theo phân khúc"], seg_summary)]:
        for j, col in enumerate(df.columns, start=1):
            cell = ws.cell(row=1, column=j)
            cell.font, cell.fill = Font(bold=True), header_fill
            cell.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
            width = max(len(str(col)) * 0.8, df[col].astype(str).str.len().max() + 2)
            ws.column_dimensions[get_column_letter(j)].width = min(max(width, 10), 60)
            if col in formats:
                for r in range(2, len(df) + 2):
                    ws.cell(row=r, column=j).number_format = formats[col]
        ws.row_dimensions[1].height = 45
        ws.freeze_panes = "C2" if ws.title == "Top 500" else "A2"
        ws.auto_filter.ref = ws.dimensions
    wn = xw.sheets["Ghi chú"]
    wn.column_dimensions["A"].width = 120
    wn["A1"].font, wn["A2"].font = Font(bold=True), Font(bold=True, size=13)
print(f"Top 500 -> Excel: {XLSX_PATH} ({len(top500)} khách, sắp xếp theo Revenue at Risk giảm dần)")

con.close()
print(f"\nĐã xuất xong vào thư mục: {OUT_DIR.resolve()}")
