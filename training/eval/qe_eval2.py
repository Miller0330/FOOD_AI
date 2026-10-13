import json, os, sys, glob
import numpy as np
import cv2
import quant_eval as q

SZ = q.SZ
names = open('labels_100.txt', encoding='utf-8').read().split()


def gt_space(mode, keys):
    gt = {}
    for k in keys:
        f = f'{q.VAL}/labels/val/{k[:-4]}.txt'
        h, w = cv2.imread(f'{q.VAL}/images/val/{k}').shape[:2]
        if mode == 'stretch':
            sx, sy, ox, oy = SZ, SZ, 0, 0
        else:
            r = min(SZ / h, SZ / w)
            nw, nh = round(w * r), round(h * r)
            sx, sy = w * r, h * r
            ox, oy = (SZ - nw) // 2, (SZ - nh) // 2
        rows = []
        if os.path.exists(f):
            for l in open(f):
                if l.strip():
                    c, cx, cy, bw, bh = l.split()[:5]
                    cx, cy, bw, bh = float(cx) * sx + ox, float(cy) * sy + oy, float(bw) * sx, float(bh) * sy
                    rows.append((int(c), cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
        gt[k] = rows
    return gt


def run(label, preds, mode):
    keys = sorted(preds.keys())
    gt = gt_cache.setdefault(mode, gt_space(mode, sorted(all_keys)))
    p = {k: preds[k] for k in keys}
    m50, m5095, ap50 = q.summarize(label, p, gt, names)
    return {'mAP50': m50, 'mAP50-95': m5095}


if __name__ == '__main__':
    _p = sys.argv[1].split('|')[2]
    all_keys = set((q.load_npu(_p) if _p.endswith('.txt') else json.load(open(_p))).keys())
    gt_cache = {}
    res = {}
    for arg in sys.argv[1:]:
        label, mode, path = arg.split('|')
        preds = q.load_npu(path) if path.endswith('.txt') else json.load(open(path))
        res[label] = run(label, preds, mode)
    json.dump(res, open('qe_eval2_result.json', 'w'), ensure_ascii=False, indent=1)
