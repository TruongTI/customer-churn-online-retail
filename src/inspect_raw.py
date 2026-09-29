import pandas as pd

df = pd.read_csv("data/raw/online_retail_II.csv", parse_dates=["InvoiceDate"])

print(df.shape)                       # ~1.07 triệu dòng, 8 cột
print(df.dtypes)
print(df["InvoiceDate"].min(), df["InvoiceDate"].max())
print(df.isna().mean().round(3))      # tỷ lệ null mỗi cột
print((df["Quantity"] < 0).sum())     # số dòng số lượng âm
print(df["Invoice"].astype(str).str.startswith("C").sum())  # số đơn hủy