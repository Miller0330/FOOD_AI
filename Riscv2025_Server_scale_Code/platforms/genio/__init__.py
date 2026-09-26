"""
MediaTek Genio 720（ARM64 Linux）平台：llama.cpp b10964（llama-completion）＋ prompt cache。

llm_advisor.py 是 Genio_720 分支的 LLMAdvisor.py（未修改）。對外介面與 RISC-V 版相同
（_ensure_prompt_cache / get_llm_advice），所以同樣維持同步建議與 __main__ 暖機。
額外相依（torch、opencc）見 requirements/genio.txt。
"""
from hal.interfaces import Platform


class _GenioLLM:
    def __init__(self):
        from . import llm_advisor
        self._m = llm_advisor

    def warmup(self):
        return self._m._ensure_prompt_cache()

    def get_advice(self, detections, nutrition, recommendations, weight, name_map=None, category_map=None):
        return self._m.get_llm_advice(detections, nutrition, recommendations, weight)


def create():
    return Platform(
        name='genio',
        llm=_GenioLLM(),
        features={'meal100': False, 'async_advice': False},
        warmup_at_import=False,
        meal100=None,
    )
