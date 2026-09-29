# %% [markdown]
# # Phase 4 - Câu hỏi 2: Phân khúc khách hàng (RFM) & độ ổn định theo quý
# Chạy: `python src/04b_rfm_segmentation.py`

# %%
import sys; sys.path.insert(0, "src")
import duckdb, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from common import load_lines, data_end, REPORT_DIR, FIG_DIR, DB_PATH

lines = load_lines()
END = data_end(lines)

# %% [markdown]
# ## 1. Tính RFM tại một ngày bất kỳ (tích lũy từ đầu dữ liệu đến ngày đó)
# - Recency = số ngày kể từ lần mua gần nhất
# - Frequency = số hóa đơn riêng biệt
# - Monetary = tổng doanh thu
# Điểm 1-5 theo phân vị (5 = tốt nhất). Dùng rank trung bình để khách bằng nhau luôn cùng điểm.

# %%
def rfm_at(snap):
    s = pd.Timestamp(snap)
    h = lines[lines["day"] <= s]
    g = h.groupby("CustomerID")
    r = pd.DataFrame({
        "Recency": (s - g["day"].max()).dt.days,
        "Frequency": g["Invoice"].nunique(),
        "Monetary": g["Revenue"].sum(),
    })
    def score(x, reverse=False):
        pct = x.rank(method="average", pct=True, ascending=not reverse)
        return np.ceil(pct * 5).clip(1, 5).astype(int)
    r["R"] = score(r["Recency"], reverse=True)      # recency càng nhỏ càng tốt
    r["F"] = score(r["Frequency"])
    r["M"] = score(r["Monetary"])
    r["FM"] = np.floor((r["F"] + r["M"]) / 2 + 0.5).astype(int)
    r["Segment"] = r.apply(segment, axis=1)
    return r

def segment(row):
    R, FM = row["R"], row["FM"]
    if R >= 4:
        return "Champions" if FM >= 4 else "Potential Loyalist"
    if R == 3:
        return "Loyal" if FM >= 3 else "Need Attention"
    return "At Risk" if FM >= 3 else "Hibernating"          # R <= 2

SEG_ORDER = ["Champions", "Loyal", "Potential Loyalist", "Need Attention", "At Risk", "Hibernating"]

# %% [markdown]
# ## 2. Phân khúc tại ngày cuối dữ liệu

# %%
now = rfm_at(END)
seg = (now.groupby("Segment")
          .agg(so_khach=("Monetary", "size"), doanh_thu=("Monetary", "sum"),
               recency_tb=("Recency", "mean"), frequency_tb=("Frequency", "mean"),
               monetary_tb=("Monetary", "mean"))
          .reindex(SEG_ORDER))
seg["ty_trong_khach_%"] = 100 * seg["so_khach"] / seg["so_khach"].sum()
seg["ty_trong_doanh_thu_%"] = 100 * seg["doanh_thu"] / seg["doanh_thu"].sum()
seg = seg.round(1)
print(seg.to_string())
seg.to_csv(REPORT_DIR / "q2_segment_profile.csv", encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(SEG_ORDER)); w = 0.38
ax.bar(x - w/2, seg["ty_trong_khach_%"], w, label="% khách hàng", color="#4C72B0")
ax.bar(x + w/2, seg["ty_trong_doanh_thu_%"], w, label="% doanh thu", color="#DD8452")
ax.set_xticks(x); ax.set_xticklabels(SEG_ORDER, rotation=20)
ax.set(title=f"Phân khúc RFM tại {END.date()}: tỷ trọng khách vs doanh thu", ylabel="%")
ax.legend(); fig.tight_layout(); fig.savefig(FIG_DIR / "q2_segments_share.png"); plt.close(fig)

# %% [markdown]
# ## 3. Độ ổn định phân khúc theo quý (Quarter-over-Quarter)
# Tại mỗi cuối quý tính lại RFM + phân khúc; với khách có mặt ở cả 2 quý liên tiếp,
# đo tỷ lệ giữ nguyên phân khúc và ma trận chuyển dịch.

# %%
snaps = [pd.Timestamp(d) for d in
         ["2010-03-31", "2010-06-30", "2010-09-30", "2010-12-31", "2011-03-31", "2011-06-30", "2011-09-30"]] + [END]
labels = ["Q1-10", "Q2-10", "Q3-10", "Q4-10", "Q1-11", "Q2-11", "Q3-11", "Q4-11*"]   # *Q4-11 chỉ đến 09/12
by_q = {lab: rfm_at(s) for lab, s in zip(labels, snaps)}

rows, pooled = [], []
for a, b in zip(labels[:-1], labels[1:]):
    both = by_q[a][["Segment"]].join(by_q[b][["Segment"]], how="inner", lsuffix="_from", rsuffix="_to")
    stay = (both["Segment_from"] == both["Segment_to"]).mean()
    rows.append((f"{a} -> {b}", len(both), 100 * stay))
    pooled.append(both)
stab = pd.DataFrame(rows, columns=["chuyen_quy", "so_khach_chung", "ty_le_giu_nguyen_%"]).round(1)
print(stab.to_string(index=False))
print("\nTrung bình (theo khách):", round(np.average(stab["ty_le_giu_nguyen_%"], weights=stab["so_khach_chung"]), 1), "%")
stab.to_csv(REPORT_DIR / "q2_stability_by_quarter.csv", index=False, encoding="utf-8-sig")

P = pd.concat(pooled).reset_index(drop=True)
trans = pd.crosstab(P["Segment_from"], P["Segment_to"], normalize="index").reindex(index=SEG_ORDER, columns=SEG_ORDER).fillna(0) * 100
print("\nMa trận chuyển dịch gộp (% theo hàng):\n", trans.round(1).to_string())
trans.round(1).to_csv(REPORT_DIR / "q2_transition_matrix.csv", encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(7.5, 5.5))
sns.heatmap(trans, annot=True, fmt=".0f", cmap="Blues", cbar_kws={"label": "%"}, ax=ax)
ax.set(title="Ma trận chuyển dịch phân khúc giữa 2 quý liên tiếp (%)", xlabel="Quý sau", ylabel="Quý trước")
fig.tight_layout(); fig.savefig(FIG_DIR / "q2_transition_matrix.png"); plt.close(fig)

# %% [markdown]
# ## 4. K-Means để đối chiếu với phân khúc theo quy tắc
# Chuẩn hóa log(R,F,M) rồi chọn k theo silhouette.

# %%
X = StandardScaler().fit_transform(np.log1p(now[["Recency", "Frequency", "Monetary"]]))
res = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    res.append((k, km.inertia_, silhouette_score(X, km.labels_)))
km_res = pd.DataFrame(res, columns=["k", "inertia", "silhouette"]).round(3)
print(km_res.to_string(index=False))
km_res.to_csv(REPORT_DIR / "q2_kmeans_selection.csv", index=False)

K = 4                                                   # chốt k=4 (xem giải thích trong báo cáo)
km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X)
now["Cluster"] = km.labels_
cl = (now.groupby("Cluster").agg(so_khach=("Monetary", "size"), recency_tb=("Recency", "mean"),
      frequency_tb=("Frequency", "mean"), monetary_tb=("Monetary", "mean"), doanh_thu=("Monetary", "sum")))
cl["ty_trong_doanh_thu_%"] = 100 * cl["doanh_thu"] / cl["doanh_thu"].sum()
cl = cl.sort_values("monetary_tb", ascending=False).round(1)
print("\nHồ sơ cụm K-Means (k=4):\n", cl.to_string())
cl.to_csv(REPORT_DIR / "q2_kmeans_profile.csv", encoding="utf-8-sig")
ari = adjusted_rand_score(now["Segment"], now["Cluster"])
print("ARI giữa phân khúc quy tắc và K-Means:", round(ari, 3))
print(pd.crosstab(now["Segment"], now["Cluster"]).reindex(SEG_ORDER).to_string())

# %% [markdown]
# ## 5. Lưu kết quả cho Phase 5 (Power BI)

# %%
out = now.reset_index().merge(lines.drop_duplicates("CustomerID")[["CustomerID", "Country"]], on="CustomerID")
out["ReferenceDate"] = END
out.to_csv("data/processed/rfm_segments.csv", index=False)
con = duckdb.connect(str(DB_PATH))
con.execute("CREATE OR REPLACE TABLE rfm_segments AS SELECT * FROM out")
con.close()
print("\nĐã lưu data/processed/rfm_segments.csv và bảng rfm_segments trong DuckDB:", len(out), "khách")
