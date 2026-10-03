# %% [markdown]
# # Câu hỏi 1: Định nghĩa Churn (mô hình Non-contractual)
# Chạy từ thư mục gốc repo: `python src/04a_churn_definition.py`
# (trong VS Code/Jupyter có thể chạy từng ô `# %%`)

# %%
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from common import load_lines, data_end, REPORT_DIR, FIG_DIR

lines = load_lines()
END = data_end(lines)
print("Ngày cuối dữ liệu:", END.date())

# một dòng / (khách, ngày mua)
days = lines[["CustomerID", "day"]].drop_duplicates().sort_values(["CustomerID", "day"])
days["gap"] = days.groupby("CustomerID")["day"].diff().dt.days

# %% [markdown]
# ## 1. Khoảng cách giữa 2 lần mua liên tiếp (inter-purchase gap)
# Nếu 80% các lần quay lại xảy ra trong vòng X ngày thì X là ngưỡng "im lặng" hợp lý.

# %%
gaps = days["gap"].dropna()
q = gaps.quantile([.5, .75, .8, .9, .95]).round(0)
print(q.to_string())
q.rename("so_ngay").to_csv(REPORT_DIR / "q1_gap_quantiles.csv", encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(gaps.clip(upper=400), bins=60, color="#4C72B0")
for n, c in [(60, "#DD8452"), (90, "#C44E52"), (120, "#55A868")]:
    ax.axvline(n, color=c, ls="--", label=f"{n} ngày")
ax.set(title="Phân phối khoảng cách giữa 2 lần mua liên tiếp", xlabel="Số ngày (cắt tại 400)", ylabel="Số lần")
ax.legend(); fig.tight_layout(); fig.savefig(FIG_DIR / "q1_gap_distribution.png"); plt.close(fig)

# %% [markdown]
# ## 2. Độ nhạy: tỷ lệ churn theo ngưỡng N (Sensitivity Analysis)
# Tại mỗi mốc thời gian (snapshot cuối tháng), lấy nhóm khách "đang hoạt động"
# (có mua trong 180 ngày trước đó), churn = không mua trong N ngày tiếp theo.

# %%
last_by_snap = {}
rows = []
snaps = pd.date_range("2010-06-30", "2011-09-30", freq="ME")
for N in [30, 60, 90, 120, 180]:
    for s in snaps:
        if s + pd.Timedelta(days=N) > END:       # cửa sổ tương lai vượt quá dữ liệu -> bỏ
            continue
        past = days[days["day"] <= s]
        last = past.groupby("CustomerID")["day"].max()
        active = last[(s - last).dt.days <= 180].index
        fut = days[(days["day"] > s) & (days["day"] <= s + pd.Timedelta(days=N))]["CustomerID"].unique()
        rows.append((N, s, len(active), 1 - np.isin(active, fut).mean()))
sens = pd.DataFrame(rows, columns=["N_ngay", "snapshot", "so_khach_active", "ty_le_churn"])
pivot = sens.pivot(index="snapshot", columns="N_ngay", values="ty_le_churn").round(3)
print(pivot.to_string())
summary = sens.groupby("N_ngay")["ty_le_churn"].agg(["mean", "min", "max"]).round(3)
print("\nTóm tắt theo ngưỡng:\n", summary.to_string())
pivot.to_csv(REPORT_DIR / "q1_sensitivity_by_month.csv", encoding="utf-8-sig")
summary.to_csv(REPORT_DIR / "q1_sensitivity_summary.csv", encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(8, 4.5))
for N in [60, 90, 120]:
    p = sens[sens["N_ngay"] == N]
    ax.plot(p["snapshot"], p["ty_le_churn"], marker="o", label=f"Ngưỡng {N} ngày")
ax.set(title="Độ nhạy: tỷ lệ churn theo ngưỡng và thời điểm", ylabel="Tỷ lệ churn", xlabel="Snapshot")
ax.legend(); fig.autofmt_xdate(); fig.tight_layout()
fig.savefig(FIG_DIR / "q1_sensitivity.png"); plt.close(fig)

# %% [markdown]
# ## 3. Xác suất quay lại sau khi im lặng quá N ngày
# Nếu khách im lặng quá N ngày nhưng phần lớn vẫn quay lại -> ngưỡng N quá ngắn (gán nhầm churn).
# Dùng cả các khoảng đã kết thúc (khách đã quay lại) và các khoảng còn đang mở (chưa quay lại tới ngày cuối dữ liệu).

# %%
last_day = days.groupby("CustomerID")["day"].max()
open_spell = (END - last_day).dt.days                   # khoảng im lặng còn đang mở
done = days["gap"].dropna()
rows = []
for N in [30, 60, 90, 120, 180, 270]:
    a, b = int((done > N).sum()), int((open_spell > N).sum())
    rows.append((N, a, b, a / (a + b)))
ret = pd.DataFrame(rows, columns=["N_ngay", "da_quay_lai", "chua_quay_lai", "ty_le_quay_lai"]).round(3)
print(ret.to_string(index=False))
ret.to_csv(REPORT_DIR / "q1_return_probability.csv", index=False, encoding="utf-8-sig")

# %% [markdown]
# ## 4. Kết luận
# Chốt **N = 90 ngày** (xem giải thích trong báo cáo): xấp xỉ phân vị 80% của khoảng cách giữa 2 lần mua,
# tỷ lệ churn quanh 40-50% (cân bằng cho mô hình), đủ ngắn để hành động kịp thời.
CHURN_DAYS = 90
print("\nĐịnh nghĩa chốt: khách hàng churn nếu không mua trong", CHURN_DAYS, "ngày.")
