# %% [markdown]
# # Câu hỏi 1 (bổ sung) — Kết quả nhạy đến đâu với lựa chọn ngưỡng churn?
#
# 04a chỉ cho biết TỶ LỆ CHURN thay đổi theo ngưỡng. Script này lặp lại TOÀN BỘ
# quy trình của 04c với 3 ngưỡng 60 / 90 / 120 ngày để xem các KẾT LUẬN có đổi không:
#   A. Model drift: AUC của mô hình học trên 2010, kiểm thử trên 2011
#   B. Back-test Top 500: % doanh thu ước tính bị mất mà chiến lược Revenue at Risk bắt được
#   C. Danh sách Top 500 tại ngày cuối dữ liệu: trùng bao nhiêu khách với danh sách ở ngưỡng chính (90 ngày)
#
# Với mỗi ngưỡng H, mọi thứ phụ thuộc ngưỡng đều đổi theo: nhãn churn (không mua trong H ngày),
# ngày snapshot (lùi H ngày để cửa sổ nhãn nằm trong dữ liệu) và doanh thu kỳ vọng (doanh thu / H ngày).
# Cấu hình mô hình, đặc trưng, cửa sổ đặc trưng 270 ngày và định nghĩa "đang hoạt động" giữ nguyên.
#
# Chạy: python src/04d_threshold_sensitivity.py   (sau 04c; khoảng 1-2 phút)

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent) if "__file__" in globals() else "src")
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from common import (load_lines, data_end, build_snapshot, FEATURES, REPORT_DIR, LOOKBACK_DAYS,
                    CHURN_DAYS, SEED, make_lr, make_rf)

THRESHOLDS = [60, 90, 120]
K = 500
assert CHURN_DAYS in THRESHOLDS, "Ngưỡng chính phải nằm trong danh sách để làm mốc so sánh"

lines = load_lines()
END = data_end(lines)
START = lines["day"].min()


def run(H):
    """Chạy lại quy trình 04c với ngưỡng H. Trả về (dòng kết quả, danh sách Top 500 tại END)."""
    row = {"Nguong_ngay": H}

    # ---- A. Huấn luyện 2010 -> kiểm thử 2011 (giống 04c mục A)
    s10 = END - pd.Timedelta(days=H) - pd.DateOffset(years=1)
    s11 = END - pd.Timedelta(days=H)
    d10, d11 = build_snapshot(lines, s10, horizon=H), build_snapshot(lines, s11, horizon=H)
    tr, ho = train_test_split(d10, test_size=0.3, stratify=d10["churn"], random_state=SEED)
    row.update({
        "Ngay_snapshot_2011": s11.date(),
        "Ty_le_churn_2010": d10["churn"].mean(),
        "Ty_le_churn_2011": d11["churn"].mean(),
        # cửa sổ đặc trưng 2010 bị cắt nếu snapshot quá gần ngày đầu dữ liệu
        "Lich_su_dac_trung_2010_ngay": min(LOOKBACK_DAYS, (s10 - START).days),
    })
    for name, mk in [("LR", make_lr), ("RF", make_rf)]:
        m = mk().fit(tr[FEATURES], tr["churn"])
        row[f"AUC_{name}_2010"] = roc_auc_score(ho["churn"], m.predict_proba(ho[FEATURES])[:, 1])
        row[f"AUC_{name}_2011"] = roc_auc_score(d11["churn"], m.predict_proba(d11[FEATURES])[:, 1])

    # ---- B. Back-test Top 500 (giống 04c mục B)
    snap_dates = [END - pd.Timedelta(days=H) - pd.Timedelta(days=91 * k) for k in range(4)]
    snaps = {s: build_snapshot(lines, s, horizon=H) for s in snap_dates}
    te = snaps[snap_dates[0]].copy()
    rf_bt = make_rf().fit(pd.concat([snaps[s] for s in snap_dates[1:]])[FEATURES],
                          pd.concat([snaps[s] for s in snap_dates[1:]])["churn"])
    te["p_churn"] = rf_bt.predict_proba(te[FEATURES])[:, 1]
    te["rev_at_risk"] = te["p_churn"] * te["rev90_est"]          # rev90_est = doanh thu / H ngày
    te["lost_actual"] = te["churn"] * te["rev90_est"]
    total_lost = te["lost_actual"].sum()

    def capture(col):
        return 100 * te.nlargest(K, col)["lost_actual"].sum() / total_lost

    row.update({
        "AUC_backtest": roc_auc_score(te["churn"], te["p_churn"]),
        "Bat_duoc_RevenueAtRisk_%": capture("rev_at_risk"),
        "Bat_duoc_GiaTri_%": capture("rev90_est"),
        "Bat_duoc_XacSuatChurn_%": capture("p_churn"),
        "Bat_duoc_NgauNhien_%": 100 * K / len(te),               # kỳ vọng của chọn ngẫu nhiên
    })

    # ---- C. Danh sách Top 500 tại ngày cuối dữ liệu (giống 04c mục C)
    allsnap = pd.concat(snaps.values())
    final = make_rf().fit(allsnap[FEATURES], allsnap["churn"])
    now = build_snapshot(lines, END, horizon=H)
    now["rev_at_risk"] = final.predict_proba(now[FEATURES])[:, 1] * now["rev90_est"]
    top = set(now.nlargest(K, "rev_at_risk").index)
    return row, top


rows, tops = [], {}
for H in THRESHOLDS:
    print(f"Đang chạy ngưỡng {H} ngày ...", flush=True)
    r, tops[H] = run(H)
    rows.append(r)

res = pd.DataFrame(rows)
res["Top500_trung_voi_nguong_chinh"] = [len(tops[H] & tops[CHURN_DAYS]) for H in THRESHOLDS]

num = res.select_dtypes("number").columns
res[num] = res[num].round(3)
res.to_csv(REPORT_DIR / "q1_model_sensitivity.csv", index=False, encoding="utf-8-sig")
print("\n", res.T.to_string())
print(f"\nĐã lưu {REPORT_DIR / 'q1_model_sensitivity.csv'}")
