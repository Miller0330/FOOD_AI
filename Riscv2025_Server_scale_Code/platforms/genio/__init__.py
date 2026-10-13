"""
MediaTek Genio 720（ARM64 Linux）平台：llama.cpp b10964（llama-completion，CPU）＋ onnxruntime CPU 100 類 YOLO11s。

- llm_advisor.py：沿用 Matrix 版提示詞（口語、具體補充食物），以 LLM_MODEL 環境變數切換 GGUF 模型
- detector100.py：與 Matrix 同一顆 best.onnx（512px），onnxruntime 推論
額外相依：onnxruntime、opencc（見 requirements/genio.txt）。
"""
import os

from hal.interfaces import Platform, Meal100Config

_HERE = os.path.dirname(os.path.abspath(__file__))
_MATRIX = os.path.join(os.path.dirname(_HERE), 'matrix')


class _GenioLLM:
    def __init__(self):
        from . import llm_advisor
        self._m = llm_advisor

    def warmup(self):
        return True

    def get_advice(self, detections, nutrition, recommendations, weight, name_map=None, category_map=None):
        return self._m.get_llm_advice(detections, nutrition, recommendations, weight,
                                      name_map=name_map, category_map=category_map)


def _make_detector100():
    from .detector100 import UECFood100Detector
    return UECFood100Detector(conf_threshold=0.15, iou_threshold=0.45)


def create():
    from .detector100 import IMG_SIZE
    return Platform(
        name='genio',
        llm=_GenioLLM(),
        features={'meal100': True, 'async_advice': True},
        warmup_at_import=False,
        meal100=Meal100Config(
            detector_factory=_make_detector100,
            nutrition_path=os.path.join(_MATRIX, 'nutrition_data_100_uecfood.json'),
            img_size=IMG_SIZE,
            min_confidence=0.3,
        ),
    )
