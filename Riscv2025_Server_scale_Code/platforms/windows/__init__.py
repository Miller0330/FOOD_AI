"""
Windows（NVIDIA 獨立顯卡）平台：LLM 走 HTTP 呼叫本機 llama-server，100 類 YOLO 用 onnxruntime 跑 CPU。

- llm_advisor.py：呼叫 llama-server 的 /v1/chat/completions，換模型只要換 llama-server 載入的 GGUF
- detector100.py：沿用 Genio 的 onnxruntime CPU 偵測器（模型路徑用環境變數指定）
- run_llama_server.ps1：啟動 llama-server；serve.py：用 waitress 啟動網站（gunicorn 不能在 Windows 跑）
額外相依見 requirements/windows.txt。
"""
import os

from hal.interfaces import Platform, Meal100Config

_HERE = os.path.dirname(os.path.abspath(__file__))
_MATRIX = os.path.join(os.path.dirname(_HERE), 'matrix')


class _WindowsLLM:
    def __init__(self):
        from . import llm_advisor
        self._m = llm_advisor

    def warmup(self):
        return self._m.warmup()

    def get_advice(self, detections, nutrition, recommendations, weight, name_map=None, category_map=None):
        return self._m.get_llm_advice(detections, nutrition, recommendations, weight,
                                      name_map=name_map, category_map=category_map)

    def model_name(self):
        return self._m.model_name()

    def chat_stream(self, payload, history, user_message):
        return self._m.chat_stream(payload, history, user_message)

    def get_daily_summary(self, payload):
        return self._m.get_daily_summary(payload)


def _make_detector100():
    from .detector100 import UECFood100Detector
    return UECFood100Detector(conf_threshold=0.15, iou_threshold=0.45)


def create():
    from .detector100 import IMG_SIZE
    return Platform(
        name='windows',
        llm=_WindowsLLM(),
        features={'meal100': True, 'async_advice': True, 'diet_log': True},
        warmup_at_import=True,   # waitress 載入 main:app 時就暖機（__main__ 區塊不會執行）
        meal100=Meal100Config(
            detector_factory=_make_detector100,
            nutrition_path=os.path.join(_MATRIX, 'nutrition_data_100_uecfood.json'),
            img_size=IMG_SIZE,
            min_confidence=0.3,
        ),
    )
