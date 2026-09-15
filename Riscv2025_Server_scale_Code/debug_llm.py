import subprocess
import time
from LLMAdvisor import _build_prompt, _extract_advice

user_query = (
    "這餐食物是白飯1份、雞胸肉2份、花椰菜1份，總重量420公克，熱量520大卡，"
    "這餐熱量佔每日建議的26.0%，蛋白質攝取達到每日建議的42.0%，"
    "碳水化合物達到20.0%，膳食纖維達到15.0%。"
)
prompt = _build_prompt(user_query)

print("=" * 60)
print("Prompt 長度:", len(prompt))
print("=" * 60)

t0 = time.time()
try:
    result = subprocess.run([
        '/home/ademo/llama-cli',
        '-m', '/home/ademo/qwen2.5-0.5b-instruct-q4_0.gguf',
        '-p', prompt,
        '-n', '70',
        '-t', '4',
        '-c', '768',
        '-no-cnv',
        '--temp', '0.4',
        '--top-p', '0.9',
        '--top-k', '40',
        '--repeat-penalty', '1.15',
    ], capture_output=True, timeout=75)
    elapsed = time.time() - t0
    stdout = result.stdout
    timed_out = False
except subprocess.TimeoutExpired as e:
    elapsed = time.time() - t0
    stdout = e.stdout or b''
    timed_out = True

print(f"耗時: {elapsed:.1f} 秒  逾時: {timed_out}")
print(f"STDOUT 長度: {len(stdout)} bytes")
print("=" * 60)
print("--- 原始輸出最後 800 字 ---")
print(stdout.decode('utf-8', errors='ignore')[-800:])
print("=" * 60)
print("--- _extract_advice 解析結果 ---")
print(repr(_extract_advice(stdout)))