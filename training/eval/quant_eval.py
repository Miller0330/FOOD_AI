"""Float(onnx) vs NPU(.nb) box-level mAP on the 2109-image val split.
Float side replicates matrixrt_yolo exactly: stretch-resize to 512, RGB, /255,
per-anchor argmax class, score>=CONF, class-aware NMS 0.45. NPU side is parsed
from the board's matrixrt_yolo stdout (conf 0.05). Same decoder => differences
come from quantization only.
Usage: quant_eval.py float   -> writes float_stretch_preds.json
       quant_eval.py eval <npu_out.txt> [<npu_out2.txt> ...]
"""
import sys, os, re, glob, json
import numpy as np

VAL = os.environ.get('VAL', 'yolo_gpu100')
SZ = int(os.environ.get('SZ', '512'))
ONNX = os.environ.get('ONNX', '../handoff_to_convert/best.onnx')
CONF = 0.05
IOU_NMS = 0.45


def nms(boxes, scores, classes, thr):
    order = np.argsort(-scores)
    keep = []
    sup = np.zeros(len(order), bool)
    for ii, i in enumerate(order):
        if sup[ii]:
            continue
        keep.append(i)
        for jj in range(ii + 1, len(order)):
            j = order[jj]
            if sup[jj] or classes[i] != classes[j]:
                continue
            if iou1(boxes[i], boxes[j]) > thr:
                sup[jj] = True
    return keep


def iou1(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-6)


def run_float(mode='stretch', multi=0):
    import onnxruntime as ort, cv2
    sess = ort.InferenceSession(ONNX, providers=['CPUExecutionProvider'])
    name = sess.get_inputs()[0].name
    res = {}
    files = sorted(glob.glob(f'{VAL}/images/val/*.jpg'))
    for n, f in enumerate(files):
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        if mode == 'stretch':
            x = cv2.resize(img, (SZ, SZ), interpolation=cv2.INTER_LINEAR)
        else:
            r = min(SZ / h, SZ / w)
            nw, nh = round(w * r), round(h * r)
            x = np.full((SZ, SZ, 3), 114, np.uint8)
            px, py = (SZ - nw) // 2, (SZ - nh) // 2
            x[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = (x.astype(np.float32) / 255).transpose(2, 0, 1)[None]
        out = sess.run(None, {name: x})[0][0]
        cxcywh, sc = out[:4], out[4:]
        if multi:
            ci, ai = np.nonzero(sc >= CONF)
            s_ = sc[ci, ai]
            cls_ = ci
            cx, cy, w_, h_ = cxcywh[:, ai]
        else:
            cls = sc.argmax(0)
            s = sc.max(0)
            m = s >= CONF
            s_, cls_ = s[m], cls[m]
            cx, cy, w_, h_ = cxcywh[:, m]
        b = np.stack([np.maximum(0, cx - w_ / 2), np.maximum(0, cy - h_ / 2), np.minimum(SZ, cx + w_ / 2), np.minimum(SZ, cy + h_ / 2)], 1)
        k = nms(b, s_, cls_, IOU_NMS)[:300]
        res[os.path.basename(f)] = [(int(cls_[i]), float(s_[i]), *map(float, b[i])) for i in k]
        if n % 400 == 0:
            print(n, flush=True)
    json.dump(res, open(f'float_{mode}_m{multi}.json', 'w'))


def load_npu(path):
    d, cur = {}, None
    for line in open(path, encoding='utf-8'):
        if line.startswith('IMG '):
            cur = line.split()[1]
            d[cur] = []
        else:
            m = re.search(r'class=(\d+).*score=([\d.]+)\s+bbox=\[([\d.]+),([\d.]+)\s*->\s*([\d.]+),([\d.]+)\]', line)
            if m and cur:
                c, s, x1, y1, x2, y2 = m.groups()
                d[cur].append((int(c), float(s), float(x1), float(y1), float(x2), float(y2)))
    return d


def load_gt(keys):
    gt = {}
    for k in keys:
        f = f'{VAL}/labels/val/{k[:-4]}.txt'
        rows = []
        if os.path.exists(f):
            for l in open(f):
                if l.strip():
                    c, cx, cy, w, h = l.split()[:5]
                    cx, cy, w, h = float(cx) * SZ, float(cy) * SZ, float(w) * SZ, float(h) * SZ
                    rows.append((int(c), cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        gt[k] = rows
    return gt


def ap_per_class(preds, gt, thrs):
    """returns {thr: {class: AP}} using 101-point interpolation (COCO style)."""
    classes = sorted({g[0] for v in gt.values() for g in v})
    out = {t: {} for t in thrs}
    for c in classes:
        dets = [(s, k, (x1, y1, x2, y2)) for k, v in preds.items() if k in gt for (cc, s, x1, y1, x2, y2) in v if cc == c]
        dets.sort(key=lambda d: -d[0])
        gts = {k: [g[1:] for g in v if g[0] == c] for k, v in gt.items()}
        npos = sum(len(v) for v in gts.values())
        for t in thrs:
            used = {k: [False] * len(v) for k, v in gts.items()}
            tp = np.zeros(len(dets))
            for i, (s, k, b) in enumerate(dets):
                best, bj = 0, -1
                for j, g in enumerate(gts.get(k, [])):
                    if used[k][j]:
                        continue
                    o = iou1(b, g)
                    if o > best:
                        best, bj = o, j
                if best >= t:
                    tp[i] = 1
                    used[k][bj] = True
            ctp = np.cumsum(tp)
            cfp = np.cumsum(1 - tp)
            rec = ctp / max(npos, 1)
            prec = ctp / np.maximum(ctp + cfp, 1e-9)
            mpre = np.concatenate([[0], prec, [0]])
            mrec = np.concatenate([[0], rec, [1]])
            for i in range(len(mpre) - 2, -1, -1):
                mpre[i] = max(mpre[i], mpre[i + 1])
            xs = np.linspace(0, 1, 101)
            idx = np.searchsorted(mrec, xs, side='left')
            out[t][c] = float(np.mean([mpre[i] if i < len(mpre) else 0 for i in idx]))
    return out


def summarize(name, preds, gt, names):
    thrs = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    ap = ap_per_class(preds, gt, thrs)
    m50 = float(np.mean(list(ap[0.5].values())))
    m5095 = float(np.mean([np.mean(list(ap[t].values())) for t in thrs]))
    print(f'{name:28} mAP50 {m50:.4f}  mAP50-95 {m5095:.4f}  (images={len(preds)})')
    return m50, m5095, ap[0.5]


if __name__ == '__main__':
    if sys.argv[1] == 'float':
        run_float(sys.argv[2], int(sys.argv[3]))
    else:
        names = open('labels_100.txt', encoding='utf-8').read().split()
        res = {}
        srcs = [('float(stretch)', json.load(open('float_stretch_preds.json')))]
        for p in sys.argv[2:]:
            srcs.append((os.path.basename(p), load_npu(p)))
        keys = set(srcs[1][1].keys()) & set(srcs[0][1].keys())
        gt = load_gt(sorted(keys))
        for name, preds in srcs:
            preds = {k: preds[k] for k in keys if k in preds}
            m50, m5095, ap50 = summarize(name, preds, gt, names)
            res[name] = {'mAP50': m50, 'mAP50-95': m5095, 'ap50': {names[c]: v for c, v in ap50.items()}}
        json.dump(res, open('quant_eval_result.json', 'w'), ensure_ascii=False, indent=1)
        f, n = res[srcs[0][0]]['ap50'], res[srcs[1][0]]['ap50']
        d = sorted(((n[k] - f[k], k, f[k], n[k]) for k in f))
        print('\n量化後 AP50 掉最多的 8 類 (NPU-float)')
        for dd, k, a, b in d[:8]:
            print(f'  {k:36} float {a:.3f} -> NPU {b:.3f}  ({dd:+.3f})')
        print('量化後反而變好的 3 類')
        for dd, k, a, b in d[-3:]:
            print(f'  {k:36} float {a:.3f} -> NPU {b:.3f}  ({dd:+.3f})')
