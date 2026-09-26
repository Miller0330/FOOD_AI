"""
Realtek Matrix（RTD2811，Ubuntu 24.04）平台：NPU 上的 Gemma4 E4B ＋ 100 類 YOLO（matrixrt_yolo）。

- llm_advisor.py：Matrix 分支的 LLMAdvisor.py（常駐 MatriXRT-llm-cli 子程序）
- detector100.py：呼叫已修改的 /opt/matrix/bin/matrixrt_yolo
- 由 gunicorn 載入，所以模組載入時就暖機（warmup_at_import=True）；建議非同步顯示（LLM 約 20 秒）
"""
import os

from hal.interfaces import Platform, Meal100Config

_HERE = os.path.dirname(os.path.abspath(__file__))


class _MatrixLLM:
    def __init__(self):
        from . import llm_advisor
        self._m = llm_advisor

    def warmup(self):
        return self._m._ensure_engine_running()

    def get_advice(self, detections, nutrition, recommendations, weight, name_map=None, category_map=None):
        return self._m.get_llm_advice(detections, nutrition, recommendations, weight,
                                      name_map=name_map, category_map=category_map)


def _make_detector100():
    from .detector100 import UECFood100Detector
    return UECFood100Detector(conf_threshold=0.15, iou_threshold=0.45)


def create():
    from .detector100 import IMG_SIZE
    return Platform(
        name='matrix',
        llm=_MatrixLLM(),
        features={'meal100': True, 'async_advice': True},
        warmup_at_import=True,
        meal100=Meal100Config(
            detector_factory=_make_detector100,
            nutrition_path=os.path.join(_HERE, 'nutrition_data_100_uecfood.json'),
            img_size=IMG_SIZE,
            min_confidence=0.3,
        ),
    )
