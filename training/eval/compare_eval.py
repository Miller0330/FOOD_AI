import json, re, glob, os, sys
names = open('labels_100.txt', encoding='utf-8').read().split()
gt = {}
for f in glob.glob('yolo_full100/labels/val/*.txt'):
    gt[os.path.basename(f)[:-4] + '.jpg'] = {int(l.split()[0]) for l in open(f) if l.strip()}
flt = {k: [tuple(x) for x in v] for k, v in json.load(open('float_preds.json')).items()}
npu, cur = {}, None
for line in open('board_out.txt', encoding='utf-8'):
    if line.startswith('IMG '):
        cur = line.split()[1]; npu[cur] = []
    else:
        m = re.search(r'class=(\d+).*score=([\d.]+)', line)
        if m and cur: npu[cur].append((int(m.group(1)), float(m.group(2))))
imgs = [k for k in npu if k in gt]
print('images compared:', len(imgs))
def stats(pred, t):
    hit = tot = rec = top1 = anyd = 0
    for k in imgs:
        d = [(c, s) for c, s in pred.get(k, []) if s >= t]
        tot += len(d); hit += sum(c in gt[k] for c, s in d)
        rec += any(c in gt[k] for c, s in d)
        anyd += bool(d)
        if d: top1 += max(d, key=lambda x: x[1])[0] in gt[k]
    n = len(imgs)
    return dict(detect_rate=anyd/n, image_recall=rec/n, det_precision=hit/max(tot,1), top1_of_detected=top1/max(anyd,1), dets_per_img=tot/n)
print(f"{'thr':>5} | {'':6} {'有偵測':>6} {'召回':>6} {'精確':>6} {'top1':>6} {'框/圖':>6}")
for t in (0.15, 0.2, 0.3, 0.4, 0.5):
    for name, p in (('float', flt), ('NPU', npu)):
        s = stats(p, t)
        print(f"{t:5.2f} | {name:6} {s['detect_rate']:6.1%} {s['image_recall']:6.1%} {s['det_precision']:6.1%} {s['top1_of_detected']:6.1%} {s['dets_per_img']:6.2f}")
# 每類 NPU 召回 (thr 0.3)
from collections import defaultdict
per = defaultdict(lambda: [0, 0, 0])
for k in imgs:
    for c in gt[k]:
        per[c][0] += 1
        per[c][1] += any(cc == c and s >= 0.3 for cc, s in flt.get(k, []))
        per[c][2] += any(cc == c and s >= 0.3 for cc, s in npu.get(k, []))
rows = sorted(((v[2]/v[0], v[1]/v[0], c, v[0]) for c, v in per.items()))
print('\nNPU召回最低10類 (thr0.3)  NPU / float / 張數')
for n, f, c, cnt in rows[:10]: print(f"  {names[c]:38} {n:5.0%} {f:5.0%} {cnt}")
print('NPU召回最高5類')
for n, f, c, cnt in rows[-5:]: print(f"  {names[c]:38} {n:5.0%} {f:5.0%} {cnt}")
