"""
Dựng Star Schema trong DuckDB bằng Python (thay cho bước chạy tay trong DBeaver).

Chạy: python src/build_star_schema.py      (sau src/clean_data.py)

Chạy nguyên file sql/star_schema.sql trên data/processed/retail.duckdb, sau đó
tự kiểm tra (thay vì phải đọc kết quả bằng mắt trong DBeaver):
  - không còn giá trị vi phạm trong clean_retail (câu kiểm tra số 2 của quality_checks.sql)
  - 3 câu kiểm tra "mồ côi" Fact -> Dim đều bằng 0
  - Fact_Sales khớp clean_retail (số dòng + tổng doanh thu)
DBeaver vẫn dùng được để XEM dữ liệu, nhưng nhớ Disconnect trước khi chạy script Python.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SQL_DIR, connect, check_star_schema_fresh  # noqa: E402

sql = (SQL_DIR / "star_schema.sql").read_text(encoding="utf-8")

con = connect()
try:
    con.execute(sql)   # DuckDB chạy lần lượt mọi câu lệnh trong script

    bad = con.execute("""
        SELECT
            SUM(CASE WHEN CustomerID IS NULL THEN 1 ELSE 0 END),
            SUM(CASE WHEN Quantity   <= 0    THEN 1 ELSE 0 END),
            SUM(CASE WHEN Price      <= 0    THEN 1 ELSE 0 END),
            SUM(CASE WHEN Invoice LIKE 'C%'  THEN 1 ELSE 0 END)
        FROM clean_retail
    """).fetchone()
    assert all(v == 0 for v in bad), f"clean_retail còn giá trị vi phạm (null/qty/price/hủy): {bad}"

    orphans = {
        "Dim_Date": "SELECT COUNT(*) FROM Fact_Sales f LEFT JOIN Dim_Date d ON f.DateKey = d.DateKey WHERE d.DateKey IS NULL",
        "Dim_Customer": "SELECT COUNT(*) FROM Fact_Sales f LEFT JOIN Dim_Customer c ON f.CustomerID = c.CustomerID WHERE c.CustomerID IS NULL",
        "Dim_Product": "SELECT COUNT(*) FROM Fact_Sales f LEFT JOIN Dim_Product p ON f.StockCode = p.StockCode WHERE p.StockCode IS NULL",
    }
    for dim, q in orphans.items():
        n = con.execute(q).fetchone()[0]
        assert n == 0, f"Fact_Sales có {n:,} dòng mồ côi (không tìm thấy trong {dim})"

    check_star_schema_fresh(con)

    for t in ["Fact_Sales", "Dim_Customer", "Dim_Product", "Dim_Date"]:
        print(f"{t:<13} {con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]:>9,} dòng")
    print("Kiểm tra vi phạm, mồ côi, khớp clean_retail: ĐẠT")
finally:
    con.close()
