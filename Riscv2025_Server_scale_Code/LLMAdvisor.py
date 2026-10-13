"""
相容層：舊程式碼（含 debug 腳本）仍可 `from LLMAdvisor import get_llm_advice, _ensure_prompt_cache`。

實際的 LLM 實作已依平台搬到 platforms/<name>/llm_advisor.py（RISC-V 那份與整併前的 LLMAdvisor.py 位元組完全相同）。
這裡依 PLATFORM（環境變數 / 自動偵測 / 預設 riscv）載入對應那一份，並把它的所有名稱轉出來。
新程式請改用 hal.get_platform().llm。
"""
import importlib

from hal import selected_platform_name

_module = importlib.import_module(f'platforms.{selected_platform_name()}.llm_advisor')
globals().update({k: v for k, v in vars(_module).items() if not k.startswith('__')})
