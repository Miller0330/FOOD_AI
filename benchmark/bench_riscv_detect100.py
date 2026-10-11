"""Andes Voyager（RISC-V）上實測 100 類偵測速度。
用法（在 repo 根目錄）：
    python3 benchmark/bench_riscv_detect100.py <圖片或資料夾> [張數=20] [執行緒=4]
會印出：模型載入時間、單張完整處理時間（平均／p50／p95／最大）、純模型前向時間。
不設定任何環境變數也可執行；請先確認 models/uecfood_s512/best.torchscript 已用 git lfs pull 取得。
"""
import glob
import os
import statistics as st
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Riscv2025_Server_scale_Code', 'platforms', 'riscv'))
target = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
os.environ['UEC100_THREADS'] = sys.argv[3] if len(sys.argv) > 3 else '4'

import torch  # noqa: E402
import detector100 as D  # noqa: E402

files = sorted(glob.glob(os.path.join(target, '*.jpg'))) if os.path.isdir(target) else [target]
files = (files * (n // max(len(files), 1) + 1))[:n]

t0 = time.perf_counter()
det = D.UECFood100Detector(0.15, 0.45)
model, _ = D._load()
load_s = time.perf_counter() - t0
det.detect(files[0])  # 暖機

full, fwd = [], []
for f in files:
    t = time.perf_counter()
    det.detect(f)
    full.append((time.perf_counter() - t) * 1000)
x = torch.zeros(1, 3, D.IMG_SIZE, D.IMG_SIZE)
with torch.inference_mode():
    for _ in range(min(n, 10)):
        t = time.perf_counter()
        model(x)
        fwd.append((time.perf_counter() - t) * 1000)

full.sort()
print('threads', torch.get_num_threads(), '| torch', torch.__version__)
print('模型載入 %.1f s' % load_s)
print('單張完整處理 ms：平均 %.0f  p50 %.0f  p95 %.0f  最大 %.0f（n=%d）' % (st.mean(full), full[len(full) // 2], full[int(.95 * (len(full) - 1))], full[-1], len(full)))
print('純模型前向 ms：平均 %.0f（n=%d）' % (st.mean(fwd), len(fwd)))
