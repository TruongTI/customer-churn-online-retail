"""
Chạy lại TOÀN BỘ pipeline theo đúng thứ tự, dừng ngay khi một bước lỗi.

Chạy: python src/run_pipeline.py
      python src/run_pipeline.py --from 04b_rfm_segmentation   (chạy từ một bước trở đi)

Trước khi chạy: Disconnect DBeaver khỏi data/processed/retail.duckdb.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
STEPS = [
    "clean_data",               # làm sạch -> clean_retail
    "build_star_schema",        # Star Schema + kiểm tra mồ côi
    "04a_churn_definition",     # câu hỏi 1
    "04b_rfm_segmentation",     # câu hỏi 2 (tạo rfm_segments, 04c cần)
    "04c_churn_model_top500",   # câu hỏi 3 + Top 500 (tạo churn_scores)
    "export_for_powerbi",       # CSV cho Power BI
]

ap = argparse.ArgumentParser()
ap.add_argument("--from", dest="start", choices=STEPS, default=STEPS[0],
                help="bước bắt đầu (mặc định: chạy từ đầu)")
args = ap.parse_args()

for step in STEPS[STEPS.index(args.start):]:
    print(f"\n{'=' * 70}\n>>> {step}.py\n{'=' * 70}", flush=True)
    t0 = time.time()
    rc = subprocess.run([sys.executable, str(SRC / f"{step}.py")]).returncode
    if rc != 0:
        sys.exit(f"\nDỪNG: {step}.py lỗi (mã {rc}). Sửa lỗi rồi chạy lại với --from {step}")
    print(f"<<< {step}.py xong sau {time.time() - t0:.0f}s")

print("\nPipeline chạy xong. Mở dashboard/BI.pbix -> Home -> Refresh.")
