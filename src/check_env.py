import sys
import pandas, sklearn, matplotlib, seaborn, duckdb

print("Python :", sys.version.split()[0])
print("pandas :", pandas.__version__)
print("sklearn:", sklearn.__version__)
print("duckdb :", duckdb.__version__)
print(duckdb.sql("SELECT 42 AS answer").fetchall())