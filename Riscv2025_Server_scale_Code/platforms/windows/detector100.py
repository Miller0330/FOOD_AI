"""
100類UEC-FOOD偵測器（Windows 版）：沿用 Genio 版的 onnxruntime CPU 實作（YOLO11s/512），只換預設路徑。

best.onnx 不在 repo 內，請用環境變數指定：
  UEC100_MODEL   best.onnx 的路徑
  UEC100_LABELS  labels.txt 的路徑（沒設就用 repo 的 models/uecfood_s512/labels.txt）
  UEC100_THREADS onnxruntime 執行緒數（預設 8）
GPU 留給 LLM，偵測固定走 CPU。
"""
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_MODEL_DIR = os.path.join(_REPO_ROOT, 'models', 'uecfood_s512')
os.environ.setdefault('UEC100_MODEL', os.path.join(_MODEL_DIR, 'best.onnx'))
os.environ.setdefault('UEC100_LABELS', os.path.join(_MODEL_DIR, 'labels.txt'))

from platforms.genio.detector100 import UECFood100Detector as _GenioDetector   # noqa: E402
from platforms.genio.detector100 import IMG_SIZE, MODEL_PATH, LABELS_PATH      # noqa: E402,F401


class UECFood100Detector(_GenioDetector):
    def __init__(self, conf_threshold=0.15, iou_threshold=0.45):
        for what, path in (('UEC100_MODEL', MODEL_PATH), ('UEC100_LABELS', LABELS_PATH)):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"找不到 100 類偵測模型檔案：{path}（請設定環境變數 {what}）")
        super().__init__(conf_threshold, iou_threshold)
