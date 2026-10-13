"""
100類UEC-FOOD偵測器（RISC-V 版）：torch（CPU）執行 TorchScript 版 YOLO11s／512px，
與 Matrix、Genio 是同一顆模型，前處理與解碼也與它們一致：
  letterbox（等比縮放、灰邊 114 補齊置中）→ RGB／255 → 每個位置取最高分類別 → 類別內 NMS，
  最後把框座標換算回「原圖拉伸成 512×512」的座標，網頁疊圖與營養計算不需要修改。
介面與 Matrix／Genio 的 UECFood100Detector 相同。

RISC-V 主核心沒有向量擴充（rv64imafdc），純量浮點運算，速度比 Matrix／Genio 慢很多（預估每張數秒），
請用 benchmark/bench_riscv_detect100.py 在板上實測。
"""
import os

import cv2
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
MODEL_PATH = os.environ.get('UEC100_MODEL', os.path.join(_REPO, 'models', 'uecfood_s512', 'best.torchscript'))
LABELS_PATH = os.environ.get('UEC100_LABELS', os.path.join(_REPO, 'models', 'uecfood_s512', 'labels.txt'))
IMG_SIZE = 512
NUM_THREADS = int(os.environ.get('UEC100_THREADS', '4'))

_model = None
_labels = None


def _load():
    global _model, _labels
    if _model is None:
        torch.set_num_threads(NUM_THREADS)
        _model = torch.jit.load(MODEL_PATH, map_location='cpu').eval()
        with open(LABELS_PATH, encoding='utf-8') as f:
            _labels = [ln.strip() for ln in f if ln.strip()]
    return _model, _labels


def _letterbox(bgr):
    """等比縮放並以灰色（114）補邊置中，回傳 (輸入影像, 縮放比, x 偏移, y 偏移)。"""
    h, w = bgr.shape[:2]
    r = min(IMG_SIZE / w, IMG_SIZE / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    canvas = np.full((IMG_SIZE, IMG_SIZE, 3), 114, np.uint8)
    ox, oy = (IMG_SIZE - nw) // 2, (IMG_SIZE - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, r, ox, oy


def _nms(boxes, scores, iou_thr):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        a_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        a_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        order = rest[inter / (a_i + a_r - inter + 1e-9) <= iou_thr]
    return keep


class UECFood100Detector:
    def __init__(self, conf_threshold=0.15, iou_threshold=0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def detect(self, image_path, output_path=None):
        """回傳 [{'class_name': str, 'confidence': float, 'bbox': [x1,y1,x2,y2]}]（512×512 拉伸座標）"""
        model, labels = _load()
        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            return []
        h, w = bgr.shape[:2]
        inp, scale, ox, oy = _letterbox(bgr)
        x = torch.from_numpy(cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div_(255.0)[None]
        with torch.inference_mode():
            out = model(x)
        out = (out[0] if isinstance(out, (list, tuple)) else out)[0].numpy()   # [4+NC, N]
        cls_scores = out[4:]
        cls_ids = cls_scores.argmax(0)
        confs = cls_scores.max(0)
        mask = confs >= self.conf_threshold
        detections = []
        if mask.any():
            xywh = out[:4, mask].T
            confs, cls_ids = confs[mask], cls_ids[mask]
            boxes = np.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
                              xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2], 1)
            boxes = np.clip(boxes, 0, IMG_SIZE)
            sx, sy = IMG_SIZE / w, IMG_SIZE / h
            for c in np.unique(cls_ids):
                idx = np.where(cls_ids == c)[0]
                for k in _nms(boxes[idx], confs[idx], self.iou_threshold):
                    j = idx[k]
                    x1, y1, x2, y2 = boxes[j]
                    name = labels[int(c)] if int(c) < len(labels) else str(int(c))
                    detections.append({'class_name': name, 'confidence': float(confs[j]),
                                       'bbox': [float(min(IMG_SIZE, max(0.0, (x1 - ox) / scale * sx))),
                                                float(min(IMG_SIZE, max(0.0, (y1 - oy) / scale * sy))),
                                                float(min(IMG_SIZE, max(0.0, (x2 - ox) / scale * sx))),
                                                float(min(IMG_SIZE, max(0.0, (y2 - oy) / scale * sy)))]})
            detections.sort(key=lambda d: -d['confidence'])
        if output_path:
            vis = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            for d in detections:
                x1, y1, x2, y2 = [int(v) for v in d['bbox']]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(output_path, vis)
        return detections


if __name__ == '__main__':
    import sys
    det = UECFood100Detector()
    for r in det.detect(sys.argv[1]):
        print(r)
