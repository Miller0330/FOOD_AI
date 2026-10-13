"""
平台選擇：環境變數 PLATFORM > 檔案特徵自動偵測 > 預設 riscv。

預設 riscv 是刻意的：沒有設定任何東西時，行為必須與整併前的 main（RISC-V）完全一致。
只會 import 被選中的那個平台，其他平台的程式（含 Matrix 的 NPU 相關模組）完全不會被載入。
"""
import importlib
import os

ALLOWED = ('riscv', 'genio', 'matrix')
DEFAULT = 'riscv'

# 自動偵測用的特徵檔案（依序檢查，第一個符合的勝出）。
# 這些路徑只用來「判斷是哪塊板子」，不會被讀寫。
MARKERS = (
    ('matrix', ('/opt/matrix/bin/matrixrt_yolo', '/data/models/gemma4_e4b/MatriXRT-llm-cli')),
    ('genio', ('/root/llama-b10964/llama-b10964/llama-completion',)),
    ('riscv', ('/home/ademo/llama-cli-new',)),
)

_cache = {}


def selected_platform_name():
    """回傳平台名稱，同時記錄是怎麼決定的（'env'、'autodetect:<檔案>'、'default'）。"""
    if 'name' in _cache:
        return _cache['name']
    env = os.environ.get('PLATFORM', '').strip().lower()
    if env:
        if env not in ALLOWED:
            raise ValueError(f"PLATFORM={env!r} 不支援，可用值：{', '.join(ALLOWED)}")
        name, how = env, 'env'
    else:
        name, how = DEFAULT, 'default'
        for candidate, files in MARKERS:
            hit = next((f for f in files if os.path.exists(f)), None)
            if hit:
                name, how = candidate, f'autodetect:{hit}'
                break
    _cache['name'], _cache['how'] = name, how
    print(f"[HAL] 平台 = {name}（來源：{how}）")
    return name


def get_platform():
    """建立並回傳目前平台的 Platform 物件（每個程序只建立一次）。"""
    if 'platform' not in _cache:
        name = selected_platform_name()
        module = importlib.import_module(f'platforms.{name}')
        _cache['platform'] = module.create()
    return _cache['platform']
