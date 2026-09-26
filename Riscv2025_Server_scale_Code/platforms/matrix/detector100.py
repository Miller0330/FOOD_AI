"""
100類UEC-FOOD主食辨識器：透過subprocess呼叫已patch過的matrixrt_yolo（板端NPU demo binary），
每次請求都重新啟動一次（不像LLM引擎那樣常駐）——實測單次呼叫（含模型載入+推論）約0.3秒，
不需要常駐化。

跟原本9類的 Detector.py（tflite/matrixrt CNN detector）是獨立、平行的第二套偵測器，
方案C：兩個入口分開跑，還沒決定要不要合併，先讓兩邊都能用、互不影響。
"""
import subprocess
import re
import os

MATRIXRT_YOLO_BIN = '/opt/matrix/bin/matrixrt_yolo'
MODEL_PATH = '/data/models/uecfood_s512/network_binary.nb'  # YOLO11s/512px；舊版11n/416px在 /data/models/uecfood_full100/
LABELS_PATH = '/data/models/uecfood_s512/labels.txt'
IMG_SIZE = 512
NUM_CLASSES = 100
NPU_CORES = 2

_DETECTION_RE = re.compile(
    r'#\d+\s+class=(\d+)\s+\(([^)]+)\)\s+score=([\d.]+)\s+bbox=\[([\d.]+),([\d.]+)\s*->\s*([\d.]+),([\d.]+)\]'
)


class UECFood100Detector:
    def __init__(self, conf_threshold=0.15, iou_threshold=0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def detect(self, image_path, output_path=None):
        """回傳 [{'class_name': str, 'confidence': float, 'bbox': [x1,y1,x2,y2]}]"""
        cmd = [
            MATRIXRT_YOLO_BIN, MODEL_PATH, image_path,
            '--imgsz', str(IMG_SIZE),
            '--nc', str(NUM_CLASSES),
            '--cores', str(NPU_CORES),
            '--conf', str(self.conf_threshold),
            '--iou', str(self.iou_threshold),
            '-c', LABELS_PATH,
        ]
        if output_path:
            cmd += ['-o', output_path]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        detections = []
        for match in _DETECTION_RE.finditer(result.stdout):
            class_id, class_name, score, x1, y1, x2, y2 = match.groups()
            detections.append({
                'class_name': class_name,
                'confidence': float(score),
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
            })
        return detections


if __name__ == '__main__':
    import sys
    det = UECFood100Detector()
    results = det.detect(sys.argv[1], output_path='/tmp/detector100_test_out.jpg')
    for r in results:
        print(r)
