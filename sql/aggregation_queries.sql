-- =====================================================================
-- Phase 3 - Truy vấn tổng hợp (Aggregation queries) trên Star Schema
-- Điều kiện: đã chạy xong sql/03_star_schema.sql
-- =====================================================================


-- ---------------------------------------------------------------------
-- Truy vấn 1: Doanh thu, số khách, số hóa đơn theo THÁNG
-- Mục đích: nguồn dữ liệu cho biểu đồ "xu hướng doanh thu theo thời gian"
-- ở Phase 5 (Trang 1 - Executive Overview).
-- ---------------------------------------------------------------------
SELECT
    d.Year,
    d.Month,
    d.YearMonthKey,
    COUNT(DISTINCT f.Invoice)      AS SoHoaDon,
    COUNT(DISTINCT f.CustomerID)   AS SoKhachHang,
    ROUND(SUM(f.Revenue), 0)       AS DoanhThu,
    ROUND(SUM(f.Revenue) / COUNT(DISTINCT f.Invoice), 2) AS AOV  -- giá trị đơn hàng trung bình
FROM Fact_Sales f
JOIN Dim_Date d ON f.DateKey = d.DateKey
GROUP BY d.Year, d.Month, d.YearMonthKey
ORDER BY d.YearMonthKey;


-- ---------------------------------------------------------------------
-- Truy vấn 2: Top 20 sản phẩm bán chạy nhất theo doanh thu
-- Mục đích: trả lời nhanh "sản phẩm nào mang lại doanh thu nhiều nhất"
-- ---------------------------------------------------------------------
SELECT
    p.StockCode,
    p.StandardDescription,
    SUM(f.Quantity)            AS TongSoLuongBan,
    ROUND(SUM(f.Revenue), 0)   AS TongDoanhThu,
    COUNT(DISTINCT f.Invoice)  AS SoHoaDonChua
FROM Fact_Sales f
JOIN Dim_Product p ON f.StockCode = p.StockCode
GROUP BY p.StockCode, p.StandardDescription
ORDER BY TongDoanhThu DESC
LIMIT 20;


-- ---------------------------------------------------------------------
-- Truy vấn 3: Doanh thu và số khách theo QUỐC GIA
-- Mục đích: xem thị trường nào đóng góp doanh thu chính (dữ liệu tập
-- trung mạnh vào United Kingdom, cần nêu trong báo cáo)
-- ---------------------------------------------------------------------
SELECT
    c.Country,
    COUNT(DISTINCT f.CustomerID)   AS SoKhachHang,
    COUNT(DISTINCT f.Invoice)      AS SoHoaDon,
    ROUND(SUM(f.Revenue), 0)       AS DoanhThu,
    ROUND(100.0 * SUM(f.Revenue) / SUM(SUM(f.Revenue)) OVER (), 2) AS TyTrongPhanTram
FROM Fact_Sales f
JOIN Dim_Customer c ON f.CustomerID = c.CustomerID
GROUP BY c.Country
ORDER BY DoanhThu DESC;


-- ---------------------------------------------------------------------
-- Truy vấn 4: RFM cơ bản mức khách hàng (nền tảng cho Phase 4, câu hỏi 2)
-- Recency: số ngày kể từ lần mua gần nhất tới ngày tham chiếu
--          (ngày tham chiếu = ngày giao dịch cuối cùng trong toàn bộ dữ liệu,
--           vì đây là dữ liệu lịch sử, không phải "hôm nay")
-- Frequency: số hóa đơn riêng biệt
-- Monetary: tổng doanh thu
-- ---------------------------------------------------------------------
WITH ngay_tham_chieu AS (
    SELECT MAX(InvoiceDate) AS ref_date FROM Fact_Sales
)
SELECT
    c.CustomerID,
    c.Country,
    DATE_DIFF('day', c.LastPurchaseDate, CAST((SELECT ref_date FROM ngay_tham_chieu) AS DATE)) AS Recency_Days,
    c.TotalInvoices                                    AS Frequency,
    c.TotalRevenue                                      AS Monetary
FROM Dim_Customer c
ORDER BY Monetary DESC
LIMIT 20;


-- ---------------------------------------------------------------------
-- Truy vấn 5: Cohort Retention đơn giản theo THÁNG mua hàng đầu tiên
-- Mục đích: nguồn dữ liệu thô cho biểu đồ "Cohort Retention Rate" ở Phase 5.
-- Ý tưởng: nhóm mỗi khách theo tháng mua đầu tiên (cohort), sau đó đếm số
-- khách trong cohort đó còn quay lại mua ở các tháng tiếp theo.
-- ---------------------------------------------------------------------
WITH cohort AS (
    SELECT
        CustomerID,
        DATE_TRUNC('month', FirstPurchaseDate) AS CohortMonth
    FROM Dim_Customer
),
activity AS (
    SELECT DISTINCT
        f.CustomerID,
        DATE_TRUNC('month', d.FullDate) AS ActivityMonth
    FROM Fact_Sales f
    JOIN Dim_Date d ON f.DateKey = d.DateKey
)
SELECT
    co.CohortMonth,
    DATE_DIFF('month', co.CohortMonth, a.ActivityMonth) AS ThangThuMay,  -- 0 = tháng mua đầu tiên
    COUNT(DISTINCT a.CustomerID) AS SoKhachConHoatDong
FROM cohort co
JOIN activity a ON a.CustomerID = co.CustomerID
GROUP BY co.CohortMonth, ThangThuMay
ORDER BY co.CohortMonth, ThangThuMay;
