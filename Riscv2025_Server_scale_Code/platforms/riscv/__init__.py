"""
RISC-V（Andes Voyager，晶心科技，原 main 分支）平台。

llm_advisor.py 是整併前 main 的 LLMAdvisor.py，位元組完全相同（以 git 的 100% rename 保存，tests 也會比對雜湊），
這裡只是把它包成 LLMBackend 介面。LLM 與偵測行為維持整併前；與 Matrix 看齊的部分是 AI 建議改為非同步（結果頁先出，約 9 秒後補上），
可用環境變數 RISCV_ASYNC_ADVICE=0 退回同步。只在 `python3 main.py` 時暖機、100 類預設關閉，設 RISCV_MEAL100=1 且備妥模型檔才啟用（純量 CPU，速度慢）。
"""
import os

from hal.interfaces import Platform, Meal100Config

_HERE = os.path.dirname(os.path.abspath(__file__))


class _RiscvLLM:
    def __init__(self):
        from . import llm_advisor          # 延遲載入：只有選到 riscv 才會 import
        self._m = llm_advisor

    def warmup(self):
        return self._m._ensure_prompt_cache()

    def get_advice(self, detections, nutrition, recommendations, weight, name_map=None, category_map=None):
        # 原版簽名只有四個參數；name_map/category_map 是 Matrix 100 類才用的，這裡忽略
        return self._m.get_llm_advice(detections, nutrition, recommendations, weight)


def _meal100_config():
    """RISCV_MEAL100=1 時啟用 100 類（torch CPU，TorchScript 版 YOLO11s）。模型檔缺少時印出原因並維持關閉，不讓服務起不來。"""
    from .detector100 import IMG_SIZE, MODEL_PATH, LABELS_PATH
    for path in (MODEL_PATH, LABELS_PATH):
        if not os.path.exists(path):
            print('[riscv] 100 類未啟用：找不到 %s' % path)
            return None
    if os.path.getsize(MODEL_PATH) < 1_000_000:
        print('[riscv] 100 類未啟用：%s 只是 Git LFS 指標檔，請先執行 git lfs pull --include="models/uecfood_s512/best.torchscript"' % MODEL_PATH)
        return None

    def factory():
        from .detector100 import UECFood100Detector
        return UECFood100Detector(conf_threshold=0.15, iou_threshold=0.45)

    return Meal100Config(
        detector_factory=factory,
        nutrition_path=os.path.join(_HERE, '..', 'matrix', 'nutrition_data_100_uecfood.json'),
        img_size=IMG_SIZE,
        min_confidence=0.3,
    )


def create():
    meal100 = _meal100_config() if os.environ.get('RISCV_MEAL100', '0') == '1' else None
    return Platform(
        name='riscv',
        llm=_RiscvLLM(),
        features={'meal100': meal100 is not None, 'async_advice': os.environ.get('RISCV_ASYNC_ADVICE', '1') != '0'},
        warmup_at_import=False,
        meal100=meal100,
    )
