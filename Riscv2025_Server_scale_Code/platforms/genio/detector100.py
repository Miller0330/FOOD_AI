"""
100類UEC-FOOD偵測器（Genio 版）：onnxruntime CPU 跑 YOLO11s/512（與 Matrix 同一顆模型 best.onnx）。
輸出 [1, 4+NC, N]（xywh 於 512 空間 + 各類別分數），介面與 Matrix 版 UECFood100Detector 相同。
"""
import os

import numpy as np
from PIL import Image, ImageDraw

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get('UEC100_MODEL', '/root/models/uecfood_s512/best.onnx')
LABELS_PATH = os.environ.get('UEC100_LABELS', '/root/models/uecfood_s512/labels.txt')
IMG_SIZE = 512
NUM_THREADS = int(os.environ.get('UEC100_THREADS', '8'))

_session = None
_labels = None


def _load():
    global _session, _labels
    if _session is None:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = NUM_THREADS
        _session = ort.InferenceSession(MODEL_PATH, opts, providers=['CPUExecutionProvider'])
        with open(LABELS_PATH, encoding='utf-8') as f:
            _labels = [ln.strip() for ln in f if ln.strip()]
    return _session, _labels


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
        iou = inter / (a_i + a_r - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return keep


class UECFood100Detector:
    def __init__(self, conf_threshold=0.15, iou_threshold=0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def detect(self, image_path, output_path=None):
        """回傳 [{'class_name': str, 'confidence': float, 'bbox': [x1,y1,x2,y2]}]（512 空間座標）"""
        sess, labels = _load()
        img = Image.open(image_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]  # [4+NC, N]
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
            for c in np.unique(cls_ids):
                idx = np.where(cls_ids == c)[0]
                for k in _nms(boxes[idx], confs[idx], self.iou_threshold):
                    j = idx[k]
                    name = labels[int(c)] if int(c) < len(labels) else str(int(c))
                    detections.append({'class_name': name, 'confidence': float(confs[j]),
                                       'bbox': [float(v) for v in boxes[j]]})
            detections.sort(key=lambda d: -d['confidence'])
        if output_path:
            draw = ImageDraw.Draw(img)
            for d in detections:
                draw.rectangle(d['bbox'], outline=(255, 0, 0), width=2)
                draw.text((d['bbox'][0] + 2, d['bbox'][1] + 2), '%s %.2f' % (d['class_name'], d['confidence']), fill=(255, 255, 0))
            img.save(output_path)
        return detections


if __name__ == '__main__':
    import sys
    import time
    det = UECFood100Detector()
    det.detect(sys.argv[1])
    t0 = time.time()
    res = det.detect(sys.argv[1], output_path='/tmp/detector100_test_out.jpg')
    print('latency %.0f ms' % ((time.time() - t0) * 1000))
    for r in res:
        print(r)
