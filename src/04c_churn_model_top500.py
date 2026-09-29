# %% [markdown]
# # Phase 4 - Câu hỏi 3 + Bài toán nghiệp vụ: Dự báo churn, Model Drift, chọn Top 500 khách hàng
# Chạy: `python src/04c_churn_model_top500.py`
#
# Định nghĩa (chốt ở câu hỏi 1): churn = KHÔNG mua trong 90 ngày sau ngày snapshot.
# Khách hàng đưa vào mô hình = khách "đang hoạt động" (có mua trong 180 ngày trước snapshot).
# Đặc trưng chỉ dùng dữ liệu TRƯỚC snapshot (cửa sổ 270 ngày) -> không rò rỉ tương lai.

# %%
import sys; sys.path.insert(0, "src")
import duckdb, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from common import load_lines, data_end, build_snapshot, FEATURES, REPORT_DIR, FIG_DIR, DB_PATH

SEED = 42
HORIZON = 90
lines = load_lines()
END = data_end(lines)

def make_lr():
    return make_pipeline(FunctionTransformer(np.log1p), StandardScaler(),
                         LogisticRegression(max_iter=2000, random_state=SEED))

def make_rf():
    return RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10,
                                  random_state=SEED, n_jobs=-1)

def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return {"Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred), "F1": f1_score(y, pred), "ROC_AUC": roc_auc_score(y, p)}

# %% [markdown]
# ## A. Huấn luyện trên 2010, kiểm thử trên 2011 -> đo Model Drift
# - Snapshot 2010: 10/09/2010 (nhãn = 10/09 -> 09/12/2010, nằm trọn trong 2010)
# - Snapshot 2011: 10/09/2011 (nhãn = 10/09 -> 09/12/2011): cùng mùa vụ, cách đúng 1 năm
# - Để so sánh công bằng, "hiệu năng 2010" đo trên tập giữ lại (30% khách hàng) chưa dùng để huấn luyện.

# %%
S2010 = END - pd.Timedelta(days=HORIZON) - pd.DateOffset(years=1)
S2011 = END - pd.Timedelta(days=HORIZON)
d10 = build_snapshot(lines, S2010); d11 = build_snapshot(lines, S2011)
print(f"Snapshot 2010: {S2010.date()} | {len(d10)} khách | tỷ lệ churn {d10['churn'].mean():.3f}")
print(f"Snapshot 2011: {S2011.date()} | {len(d11)} khách | tỷ lệ churn {d11['churn'].mean():.3f}")
overlap = len(set(d10.index) & set(d11.index))
print(f"Khách có mặt ở cả 2 tập: {overlap} ({overlap/len(d11):.0%} của tập 2011)")

tr, ho = train_test_split(d10, test_size=0.3, stratify=d10["churn"], random_state=SEED)
rows = []
fitted = {}
for name, mk in [("Logistic Regression", make_lr), ("Random Forest", make_rf)]:
    m = mk().fit(tr[FEATURES], tr["churn"])
    fitted[name] = m
    for period, data in [("2010 (tập giữ lại)", ho), ("2011", d11)]:
        p = m.predict_proba(data[FEATURES])[:, 1]
        rows.append({"Mo_hinh": name, "Giai_doan": period, **metrics(data["churn"], p)})
res = pd.DataFrame(rows).round(3)
print("\n", res.to_string(index=False))
res.to_csv(REPORT_DIR / "q3_metrics_2010_vs_2011.csv", index=False, encoding="utf-8-sig")

# mức suy giảm 2011 so với 2010 (điểm phần trăm)
drift = []
for name in fitted:
    a = res[(res.Mo_hinh == name) & (res.Giai_doan.str.startswith("2010"))].iloc[0]
    b = res[(res.Mo_hinh == name) & (res.Giai_doan == "2011")].iloc[0]
    drift.append({"Mo_hinh": name, **{k: round(b[k] - a[k], 3) for k in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]}})
drift = pd.DataFrame(drift)
print("\nThay đổi 2011 - 2010 (điểm tuyệt đối):\n", drift.to_string(index=False))
drift.to_csv(REPORT_DIR / "q3_drift_delta.csv", index=False, encoding="utf-8-sig")

# %% [markdown]
# ### A2. Chẩn đoán drift: PSI của đặc trưng và tầm quan trọng đặc trưng

# %%
def psi(expected, actual, bins=10):
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))

psi_df = pd.DataFrame({"feature": FEATURES,
                       "PSI": [psi(d10[f], d11[f]) for f in FEATURES]}).sort_values("PSI", ascending=False).round(3)
psi_df["muc_do"] = pd.cut(psi_df["PSI"], [-1, 0.1, 0.25, 99], labels=["ổn định", "thay đổi vừa", "thay đổi lớn"])
print("\nPSI (>0.25 = thay đổi lớn):\n", psi_df.to_string(index=False))
psi_df.to_csv(REPORT_DIR / "q3_feature_psi.csv", index=False, encoding="utf-8-sig")

imp = pd.Series(fitted["Random Forest"].feature_importances_, index=FEATURES).sort_values(ascending=False).round(3)
print("\nTầm quan trọng đặc trưng (Random Forest):\n", imp.to_string())
imp.rename("importance").to_csv(REPORT_DIR / "q3_feature_importance.csv", encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(7, 4.5))
m = res.melt(id_vars=["Mo_hinh", "Giai_doan"], value_vars=["Accuracy", "Precision", "Recall", "F1"])
for i, (nm, g) in enumerate(m.groupby("Mo_hinh")):
    for j, (per, gg) in enumerate(g.groupby("Giai_doan")):
        ax.bar(np.arange(4) + (i * 2 + j) * 0.2 - 0.3, gg["value"], 0.2, label=f"{nm} - {per}")
ax.set_xticks(range(4)); ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1"])
ax.set(ylim=(0, 1), title="Hiệu năng mô hình: 2010 (giữ lại) vs 2011"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG_DIR / "q3_drift_metrics.png"); plt.close(fig)

# %% [markdown]
# ## B. Bài toán nghiệp vụ: chọn Top 500 khách hàng cần chăm sóc
# Điểm ưu tiên = Xác suất churn x Doanh thu kỳ vọng 90 ngày (Revenue at Risk).
# Kiểm chứng ngược (back-test) tại snapshot 10/09/2011 với mô hình chỉ học từ dữ liệu trước đó.

# %%
K = 500
snap_dates = [END - pd.Timedelta(days=HORIZON) - pd.Timedelta(days=91 * k) for k in range(4)]   # k=0 là gần nhất
snaps = {s: build_snapshot(lines, s) for s in snap_dates}
test_date = snap_dates[0]
train_all = pd.concat([snaps[s] for s in snap_dates[1:]])          # nhãn của các snapshot này đều kết thúc trước test_date
print("Huấn luyện back-test trên", [s.date().isoformat() for s in snap_dates[1:]], "-> kiểm thử", test_date.date())

rf_bt = make_rf().fit(train_all[FEATURES], train_all["churn"])
te = snaps[test_date].copy()
te["p_churn"] = rf_bt.predict_proba(te[FEATURES])[:, 1]
te["rev_at_risk"] = te["p_churn"] * te["rev90_est"]
te["lost_actual"] = te["churn"] * te["rev90_est"]                   # doanh thu ước tính thực sự mất (khách thực sự rời bỏ)
print("AUC back-test:", round(roc_auc_score(te["churn"], te["p_churn"]), 3), "| số khách:", len(te))

total_lost = te["lost_actual"].sum()
def eval_sel(idx):
    s = te.loc[idx]
    return {"precision_churn": s["churn"].mean(), "doanh_thu_mat_bat_duoc": s["lost_actual"].sum(),
            "ty_le_bat_duoc_%": 100 * s["lost_actual"].sum() / total_lost}

rng = np.random.default_rng(SEED)
rand = pd.DataFrame([eval_sel(rng.choice(te.index, K, replace=False)) for _ in range(300)]).mean()
strategies = {
    "Ngẫu nhiên (TB 300 lần)": rand,
    "Top theo xác suất churn": pd.Series(eval_sel(te.nlargest(K, "p_churn").index)),
    "Top theo giá trị (doanh thu/90 ngày)": pd.Series(eval_sel(te.nlargest(K, "rev90_est").index)),
    "Top theo Revenue at Risk (đề xuất)": pd.Series(eval_sel(te.nlargest(K, "rev_at_risk").index)),
}
bt = pd.DataFrame(strategies).T.round(3)
print("\nBack-test Top", K, "tại", test_date.date(), f"(tổng doanh thu ước tính bị mất trong toàn bộ {len(te)} khách: {total_lost:,.0f})\n", bt.to_string())
bt.to_csv(REPORT_DIR / "q3_top500_backtest.csv", encoding="utf-8-sig")

# đường cong tích lũy: chọn top-K theo từng chiến lược thì bắt được bao nhiêu % doanh thu mất
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ks = np.arange(50, 1501, 50)
for label, col in [("Theo Revenue at Risk", "rev_at_risk"), ("Theo xác suất churn", "p_churn"), ("Theo giá trị", "rev90_est")]:
    order = te.sort_values(col, ascending=False)["lost_actual"].to_numpy()
    ax.plot(ks, [100 * order[:k].sum() / total_lost for k in ks], label=label)
ax.plot(ks, [100 * k / len(te) for k in ks], "k--", label="Ngẫu nhiên")
ax.axvline(K, color="grey", ls=":"); ax.text(K + 15, 5, "ngân sách 500", fontsize=8)
ax.set(title="% doanh thu mất bắt được khi chăm sóc top-K khách", xlabel="K khách được chọn", ylabel="%")
ax.legend(); fig.tight_layout(); fig.savefig(FIG_DIR / "q3_top500_capture.png"); plt.close(fig)

# %% [markdown]
# ## C. Chấm điểm thật tại ngày cuối dữ liệu (09/12/2011) và xuất danh sách Top 500
# Mô hình cuối huấn luyện trên cả 4 snapshot (gồm cả mùa Giáng sinh -> tháng 1 của năm trước).

# %%
final = make_rf().fit(pd.concat(snaps.values())[FEATURES], pd.concat(snaps.values())["churn"])
now = build_snapshot(lines, END)
now["p_churn"] = final.predict_proba(now[FEATURES])[:, 1]
now["rev_at_risk"] = now["p_churn"] * now["rev90_est"]
now = now.sort_values("rev_at_risk", ascending=False)
now["rank"] = np.arange(1, len(now) + 1)
now["top500"] = now["rank"] <= K

seg = pd.read_csv("data/processed/rfm_segments.csv")[["CustomerID", "Segment"]].set_index("CustomerID")
now = now.join(seg, how="left")
cols = ["rank", "top500", "Segment", "p_churn", "rev90_est", "rev_at_risk", "recency", "frequency", "monetary", "avg_gap", "is_uk"]
out = now[cols].round(3).reset_index()
top = out[out["top500"]]
print(f"\nKhách hàng active tại {END.date()}: {len(out)} | churn TB dự báo: {now['p_churn'].mean():.3f}")
print(f"Top {K}: tổng Revenue at Risk = {top['rev_at_risk'].sum():,.0f} "
      f"({100*top['rev_at_risk'].sum()/out['rev_at_risk'].sum():.1f}% tổng); "
      f"p_churn TB = {top['p_churn'].mean():.2f}; doanh thu/90 ngày TB = {top['rev90_est'].mean():,.0f}")
print("Phân bố phân khúc RFM trong Top 500:\n", top["Segment"].value_counts().to_string())
print(top.head(10).to_string(index=False))

out.to_csv("data/processed/churn_scores.csv", index=False)
top.to_csv(REPORT_DIR / "q3_top500_list.csv", index=False, encoding="utf-8-sig")
con = duckdb.connect(str(DB_PATH))
con.execute("CREATE OR REPLACE TABLE churn_scores AS SELECT * FROM out")
con.close()

# %% [markdown]
# ## D. ROI minh họa (THAM SỐ GIẢ ĐỊNH - thay bằng số thật của doanh nghiệp)
# ROI = (Doanh thu cứu được x Biên lợi nhuận - Chi phí chăm sóc) / Chi phí chăm sóc
# Doanh thu cứu được = Tỷ lệ thành công x Revenue at Risk của nhóm được chọn

# %%
COST_PER_CUSTOMER = 10.0     # GIẢ ĐỊNH: chi phí email/gọi điện/voucher mỗi khách (GBP)
SAVE_RATE = 0.15             # GIẢ ĐỊNH: 15% khách có nguy cơ được giữ lại nhờ chăm sóc
GROSS_MARGIN = 0.30          # GIẢ ĐỊNH: biên lợi nhuận gộp 30%
cost = COST_PER_CUSTOMER * K
saved = SAVE_RATE * top["rev_at_risk"].sum()
roi = (saved * GROSS_MARGIN - cost) / cost
print(f"\nROI minh họa: chi phí {cost:,.0f} | doanh thu cứu được {saved:,.0f} | ROI = {roi:.1f}x")
