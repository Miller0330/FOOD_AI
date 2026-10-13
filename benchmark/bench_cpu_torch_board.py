"""Matrix board CPU baseline: same YOLO11s/512 model (float TorchScript) on the 2 A73 cores.
Run with riv-foodai stopped. Needs PYTHONPATH=/data/pylibs."""
import glob, json, statistics as st, sys, time
import cv2, numpy as np, torch

torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
m = torch.jit.load(sys.argv[1], map_location='cpu').eval()
files = sorted(glob.glob('/data/val_eval/img/*.jpg'))[:40]
fw, tot = [], []
with torch.inference_mode():
    for i, f in enumerate(files):
        t0 = time.perf_counter()
        x = cv2.cvtColor(cv2.resize(cv2.imread(f), (512, 512)), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(x).permute(2, 0, 1).float().div(255)[None]
        t1 = time.perf_counter()
        out = m(x)
        out = out[0] if isinstance(out, (list, tuple)) else out
        t2 = time.perf_counter()
        if i >= 3:
            fw.append((t2 - t1) * 1000)
            tot.append((t2 - t0) * 1000)
print('RESULT', json.dumps({'threads': torch.get_num_threads(), 'n': len(fw), 'forward_ms_mean': round(st.mean(fw), 1),
                            'forward_ms_min': round(min(fw), 1), 'forward_ms_max': round(max(fw), 1), 'forward_ms_sd': round(st.pstdev(fw), 1),
                            'with_preprocess_ms_mean': round(st.mean(tot), 1), 'out_shape': list(out.shape)}))
