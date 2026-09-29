-- =====================================================================
-- Phase 2 - Kiểm tra chất lượng dữ liệu sau làm sạch (chạy trong DBeaver)
-- Kết nối tới: data/processed/retail.duckdb
-- =====================================================================

-- 1) Tổng quan: số dòng, khách hàng, hóa đơn, sản phẩm, khoảng thời gian, doanh thu
SELECT
    COUNT(*)                     AS so_dong,
    COUNT(DISTINCT CustomerID)   AS so_khach_hang,
    COUNT(DISTINCT Invoice)      AS so_hoa_don,
    COUNT(DISTINCT StockCode)    AS so_san_pham,
    MIN(InvoiceDate)             AS ngay_dau,
    MAX(InvoiceDate)             AS ngay_cuoi,
    ROUND(SUM(TotalPrice), 0)    AS tong_doanh_thu
FROM clean_retail;

-- 2) Đảm bảo không còn giá trị vi phạm (kết quả mong đợi: tất cả bằng 0)
SELECT
    SUM(CASE WHEN CustomerID IS NULL THEN 1 ELSE 0 END)   AS null_customer,
    SUM(CASE WHEN Quantity   <= 0    THEN 1 ELSE 0 END)   AS qty_khong_duong,
    SUM(CASE WHEN Price      <= 0    THEN 1 ELSE 0 END)   AS gia_khong_duong,
    SUM(CASE WHEN Invoice LIKE 'C%'  THEN 1 ELSE 0 END)   AS hoa_don_huy
FROM clean_retail;

-- 3) StockCode còn sót không có dạng 5 chữ số (để rà soát thủ công)
SELECT StockCode, MIN(Description) AS mo_ta, COUNT(*) AS so_dong
FROM clean_retail
WHERE NOT regexp_matches(StockCode, '^[0-9]{5}')
GROUP BY StockCode
ORDER BY so_dong DESC;

-- 4) Doanh thu và số khách theo tháng (nhìn nhanh xem dữ liệu có bị đứt quãng không)
SELECT
    date_trunc('month', InvoiceDay)  AS thang,
    COUNT(DISTINCT CustomerID)       AS so_khach,
    ROUND(SUM(TotalPrice), 0)        AS doanh_thu
FROM clean_retail
GROUP BY 1
ORDER BY 1;

-- 5) Nhật ký làm sạch (dùng lại cho báo cáo thực tập)
SELECT * FROM cleaning_log;
