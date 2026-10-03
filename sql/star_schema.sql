-- =====================================================================
-- Thiết kế Star Schema trong DuckDB
-- Chạy trong DBeaver, kết nối tới data/processed/retail.duckdb
-- =====================================================================


-- ---------------------------------------------------------------------
-- 1) DIM_DATE
-- Một dòng cho mỗi ngày xuất hiện trong dữ liệu (không phải lịch đầy đủ),
-- Grain: 1 dòng = 1 ngày
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE Dim_Date AS
SELECT
    CAST(strftime(d, '%Y%m%d') AS INTEGER)     AS DateKey,      -- khóa dạng số, dễ join & lọc
    d                                          AS FullDate,
    EXTRACT(YEAR    FROM d)                    AS Year,
    EXTRACT(QUARTER FROM d)                    AS Quarter,
    EXTRACT(MONTH   FROM d)                    AS Month,
    strftime(d, '%B')                          AS MonthName,
    EXTRACT(WEEK    FROM d)                    AS WeekOfYear,
    EXTRACT(DAY     FROM d)                    AS DayOfMonth,
    EXTRACT(DOW     FROM d)                    AS DayOfWeekNum, -- 0=Chủ nhật ... 6=Thứ 7
    strftime(d, '%A')                          AS DayName,
    CASE WHEN EXTRACT(DOW FROM d) IN (0, 6)
         THEN TRUE ELSE FALSE END              AS IsWeekend,
    CAST(strftime(d, '%Y%m') AS INTEGER)        AS YearMonthKey  -- tiện GROUP BY theo tháng
FROM (
    -- sinh dãy ngày liên tục từ ngày đầu đến ngày cuối của dữ liệu,
    -- để không "mất" ngày nào kể cả khi không có giao dịch
    SELECT UNNEST(generate_series(
        (SELECT MIN(InvoiceDay) FROM clean_retail),
        (SELECT MAX(InvoiceDay) FROM clean_retail),
        INTERVAL 1 DAY
    )) AS d
);

-- ---------------------------------------------------------------------
-- 2) DIM_CUSTOMER
-- Grain: 1 dòng = 1 khách hàng
-- Ngày mua đầu/gần nhất tính trực tiếp từ Fact để luôn khớp dữ liệu.
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE Dim_Customer AS
SELECT
    CustomerID,
    -- một khách có thể xuất hiện với nhiều quốc gia (đổi địa chỉ giao hàng);
    -- lấy quốc gia xuất hiện nhiều nhất làm quốc gia đại diện
    (SELECT Country
     FROM clean_retail c2
     WHERE c2.CustomerID = c1.CustomerID
     GROUP BY Country
     ORDER BY COUNT(*) DESC, Country
     LIMIT 1)                              AS Country,
    MIN(InvoiceDay)                        AS FirstPurchaseDate,
    MAX(InvoiceDay)                        AS LastPurchaseDate,
    COUNT(DISTINCT Invoice)                AS TotalInvoices,
    ROUND(SUM(TotalPrice), 2)              AS TotalRevenue
FROM clean_retail c1
GROUP BY CustomerID;

-- ---------------------------------------------------------------------
-- 3) DIM_PRODUCT
-- Grain: 1 dòng = 1 StockCode
-- Mỗi StockCode có thể có vài Description khác nhau theo thời gian
-- (lỗi nhập liệu/đổi tên) -> chọn mô tả xuất hiện nhiều nhất làm chuẩn.
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE Dim_Product AS
WITH desc_rank AS (
    SELECT
        StockCode,
        Description,
        COUNT(*) AS n,
        ROW_NUMBER() OVER (PARTITION BY StockCode ORDER BY COUNT(*) DESC, Description) AS rn
    FROM clean_retail
    WHERE Description IS NOT NULL
    GROUP BY StockCode, Description
)
SELECT
    r.StockCode,
    r.Description                          AS StandardDescription,
    ROUND(AVG(c.Price), 2)                  AS AvgUnitPrice,
    MIN(c.Price)                            AS MinUnitPrice,
    MAX(c.Price)                            AS MaxUnitPrice,
    COUNT(DISTINCT c.Invoice)               AS TimesOrdered,
    SUM(c.Quantity)                         AS TotalQuantitySold
FROM desc_rank r
JOIN clean_retail c ON c.StockCode = r.StockCode
WHERE r.rn = 1
GROUP BY r.StockCode, r.Description;

-- ---------------------------------------------------------------------
-- 4) FACT_SALES
-- Grain: 1 dòng = 1 dòng hàng trong 1 hóa đơn (line item)
-- Chứa khóa ngoại tới 3 chiều + các số đo (measures)
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE Fact_Sales AS
SELECT
    c.Invoice,
    c.StockCode,
    c.CustomerID,
    CAST(strftime(c.InvoiceDay, '%Y%m%d') AS INTEGER)  AS DateKey,
    c.InvoiceDate,
    c.Quantity,
    c.Price          AS UnitPrice,
    c.TotalPrice      AS Revenue
FROM clean_retail c;

-- ---------------------------------------------------------------------
-- 5) Khóa chính & kiểm tra toàn vẹn tham chiếu
-- DuckDB (bản dùng trong đồ án) chưa hỗ trợ ALTER TABLE ... ADD FOREIGN KEY,
-- nên chỉ khai báo PRIMARY KEY cho 3 bảng Dim; quan hệ khóa ngoại của
-- Fact_Sales -> Dim_* được đảm bảo bằng 3 câu kiểm tra "mồ côi" bên dưới
-- thay vì một ràng buộc FK cứng trong schema.
-- ---------------------------------------------------------------------
ALTER TABLE Dim_Date     ADD PRIMARY KEY (DateKey);
ALTER TABLE Dim_Customer ADD PRIMARY KEY (CustomerID);
ALTER TABLE Dim_Product  ADD PRIMARY KEY (StockCode);

-- Kiểm tra không có "mồ côi" (Fact tham chiếu tới dim không tồn tại)
-- -> cả 3 câu dưới phải trả về 0
SELECT COUNT(*) AS mo_coi_date     FROM Fact_Sales f LEFT JOIN Dim_Date d     ON f.DateKey=d.DateKey     WHERE d.DateKey IS NULL;
SELECT COUNT(*) AS mo_coi_customer FROM Fact_Sales f LEFT JOIN Dim_Customer c ON f.CustomerID=c.CustomerID WHERE c.CustomerID IS NULL;
SELECT COUNT(*) AS mo_coi_product  FROM Fact_Sales f LEFT JOIN Dim_Product p  ON f.StockCode=p.StockCode  WHERE p.StockCode IS NULL;

SHOW TABLES;
