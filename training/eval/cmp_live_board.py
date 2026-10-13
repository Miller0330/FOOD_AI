import sys, os, glob, json, time
sys.path.insert(0, '/data/RIV-FoodAI-Backup/Riscv2025_Server_scale_Code')
from Detector100 import UECFood100Detector, USE_LETTERBOX
d = UECFood100Detector(conf_threshold=0.15, iou_threshold=0.45)
files = sorted(glob.glob('/data/val_eval/img/*.jpg'))
d.detect(files[0])
res, ts = {}, []
for f in files:
    t = time.perf_counter()
    r = d.detect(f)
    ts.append((time.perf_counter() - t) * 1000)
    res[os.path.basename(f)] = [(x['class_name'], x['confidence'], x['bbox']) for x in r]
tag = 'letterbox' if USE_LETTERBOX else 'stretch'
json.dump(res, open('/data/val_eval/live_%s.json' % tag, 'w'))
ts.sort()
print(tag, 'images', len(files), 'mean_ms %.1f p95 %.1f' % (sum(ts) / len(ts), ts[int(.95 * len(ts))]), 'fallbacks?', d._daemon is not None)
