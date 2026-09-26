"""
RISC-V（Qilai EVB / HiFive，原 main 分支）平台。

llm_advisor.py 是整併前 main 的 LLMAdvisor.py，位元組完全相同（以 git 的 100% rename 保存，tests 也會比對雜湊），
這裡只是把它包成 LLMBackend 介面。行為維持整併前：同步產生建議、只在 `python3 main.py` 時暖機、不支援 100 類。
"""
from hal.interfaces import Platform


class _RiscvLLM:
    def __init__(self):
        from . import llm_advisor          # 延遲載入：只有選到 riscv 才會 import
        self._m = llm_advisor

    def warmup(self):
        return self._m._ensure_prompt_cache()

    def get_advice(self, detections, nutrition, recommendations, weight, name_map=None, category_map=None):
        # 原版簽名只有四個參數；name_map/category_map 是 Matrix 100 類才用的，這裡忽略
        return self._m.get_llm_advice(detections, nutrition, recommendations, weight)


def create():
    return Platform(
        name='riscv',
        llm=_RiscvLLM(),
        features={'meal100': False, 'async_advice': False},
        warmup_at_import=False,
        meal100=None,
    )
