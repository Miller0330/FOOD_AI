import subprocess
import re
import os
import hashlib

# ============ 可調參數（依裝置速度調整）============
GEN_MAX_TOKENS = 150
CONTEXT_SIZE = 768
N_THREADS = 8
TIMEOUT_SECONDS = 75
USE_FEWSHOT = False

# ============ Prompt Cache 設定 ============
# 固定前綴（system prompt ± few-shot 範例）的 KV cache 存放路徑。
# 這段內容每次呼叫都完全相同，llama-cli 支援把算好的 KV 值存成檔案，
# 下次直接還原，跳過重新計算，可省下大量 prompt eval 時間。
CACHE_PATH = '/root/RIV-FoodAI-Backup/Riscv2025_Server_scale_Code/sysprompt.cache'
CACHE_META_PATH = '/root/RIV-FoodAI-Backup/Riscv2025_Server_scale_Code/sysprompt.cache.meta'

try:
    import opencc
    _CONVERTER = opencc.OpenCC('s2twp')
except Exception:
    _CONVERTER = None

_S2T_FALLBACK = {
    '纤': '纖', '维': '維', '摄': '攝', '饮': '飲', '营': '營',
    '养': '養', '热': '熱', '质': '質', '补': '補', '够': '夠',
    '则': '則', '时': '時', '间': '間', '类': '類', '积': '積',
    '谷': '穀', '钠': '鈉', '钙': '鈣', '铁': '鐵', '锌': '鋅',
    '钾': '鉀', '镁': '鎂', '调': '調', '择': '擇', '减': '減',
    '该': '該', '还': '還', '与': '與', '为': '為', '后': '後',
    '从': '從', '会': '會', '变': '變', '让': '讓', '觉': '覺',
    '处': '處', '汤': '湯', '长': '長', '应': '應', '议': '議',
    '稳': '穩', '显': '顯', '帮': '幫', '选': '選',
}

# ---- 這次修改的重點 1：加入語意定義與因果一致規則 ----
# 新增內容：
#   (a) 說明「偏低/不足/充足」是相對於全天建議量的單餐佔比，不是營養缺乏
#   (b) 要求建議與指出的營養缺口保持因果一致，並列出對應食物類型
#   (c) 明確禁止「把某食物同時當作已吃的東西，又當成補其他營養素的解方」
_SYSTEM_PROMPT = (
    "你是專業的營養師，只使用台灣繁體中文回答。只根據使用者提供的熱量、蛋白質、"
    "碳水化合物、膳食纖維這四項資訊給建議，禁止提及脂肪、維生素或其他未提供的營養素，"
    "禁止杜撰資訊。「偏低」或「不足」是指這一餐佔全天建議攝取量的比例較低，"
    "不代表營養缺乏，不需要在同一餐內完全補足。"
    "建議必須因果一致：指出某營養素偏低時，只能推薦能直接補充該營養素的食物類型，"
    "不可推薦與該缺口無關的食物。蛋白質偏低建議蛋、奶、豆、肉；碳水偏低建議適量主食；"
    "蔬菜不足建議蔬菜；水果不足建議水果。不可把使用者已經吃過的食物種類，"
    "同時當成另一項營養缺口的建議解方。至少給出3句具體建議，內容盡量完整，不要條列。"
)

_FEWSHOT_USER = "這餐食物是白飯1份、雞胸肉2份，總重量350公克，熱量450大卡，這餐熱量佔每日建議的20.0%，蛋白質攝取達到每日建議的35.0%，碳水化合物達到25.0%，膳食纖維達到10.0%。"
_FEWSHOT_ASSISTANT = "這餐蛋白質充足、熱量與碳水比例適中。但膳食纖維偏低，建議下一餐加一份蔬菜或換成糙米補足纖維。"

_ASSISTANT_MARKER = "assistant\n這餐"
_STOP_MARKERS = ('<|im_end|>', '<|im_start|>')

_FALLBACK_ADVICE = "這餐整體攝取尚屬均衡，建議持續維持均衡飲食，適度增加蔬菜與全穀類以補充膳食纖維，並注意蛋白質與水分的攝取。"

FOOD_NAME_MAP = {
    'Apple': '蘋果', 'Banana': '香蕉', 'Mango': '芒果', 'Orange': '柳橙',
    'Pineapple': '鳳梨', 'Watermelon': '西瓜', 'MiniBiscuits': '小餅乾',
    'Snack': '零食', 'noodles': '麵條',
}
FOOD_CATEGORY_MAP = {
    'Apple': '水果類', 'Banana': '水果類', 'Mango': '水果類', 'Orange': '水果類',
    'Pineapple': '水果類', 'Watermelon': '水果類', 'MiniBiscuits': '點心類',
    'Snack': '點心類', 'noodles': '主食類',
}


def _s2t(text: str) -> str:
    if _CONVERTER is not None:
        try:
            return _CONVERTER.convert(text)
        except Exception:
            pass
    for s, t in _S2T_FALLBACK.items():
        text = text.replace(s, t)
    return text


# ---- 這次修改的重點 2：把「固定前綴」獨立成一個函式 ----
# 純粹是把原本 _build_prompt 裡兩個分支開頭重複的那段文字抽出來，
# 文字內容跟原本完全一樣，只是拆成一個可以單獨呼叫的函式，
# 讓建立 cache 時可以只跑這一段，不影響 _build_prompt 的輸出結果。
def _fixed_prefix() -> str:
    if USE_FEWSHOT:
        return f"""<|im_start|>system
{_SYSTEM_PROMPT}
<|im_end|>
<|im_start|>user
{_FEWSHOT_USER}
<|im_end|>
<|im_start|>assistant
{_FEWSHOT_ASSISTANT}
<|im_end|>
<|im_start|>user
"""
    return f"""<|im_start|>system
{_SYSTEM_PROMPT}
<|im_end|>
<|im_start|>user
"""


def _build_prompt(user_query: str) -> str:
    return _fixed_prefix() + f"""{user_query}
<|im_end|>
<|im_start|>assistant
這餐"""


def _extract_advice(stdout_bytes: bytes) -> str:
    stdout = stdout_bytes.decode('utf-8', errors='ignore')

    idx = stdout.rfind(_ASSISTANT_MARKER)
    if idx != -1:
        advice = stdout[idx + len('assistant\n'):]
    else:
        advice = stdout.strip()

    advice = advice.replace('assistant\n', '')

    for marker in _STOP_MARKERS:
        advice = advice.split(marker)[0]
    advice = advice.strip()

    advice = _s2t(advice)

    advice = re.sub(
        r'[^一-鿿　-〿＀-￯\d\s，。！？、：；%.,:;()「」『』…—～]',
        '', advice
    )
    advice = advice.replace('end of text', '').strip()

    last_end = max(advice.rfind('。'), advice.rfind('！'), advice.rfind('？'))
    if last_end != -1 and last_end + 1 < len(advice):
        advice = advice[:last_end + 1]
    elif last_end == -1 and advice:
        advice = ''

    if advice and not advice.endswith(('。', '！', '？')):
        advice = advice.rstrip('，、；：') + '。'

    return advice


# ---- 這次修改的重點 3：Prompt Cache 建立與驗證 ----
# 邏輯：
#   1. 算出目前固定前綴的 hash
#   2. 如果 cache 檔案存在，且上次記錄的 hash 跟這次一致 -> 直接回傳可用
#   3. 否則（第一次執行 / system prompt 或 few-shot 內容改過了）重新建立 cache
#   4. 建立失敗就回傳 False，呼叫端會自動退回不用 cache 的原始行為，不會壞掉
def _ensure_prompt_cache() -> bool:
    prefix = _fixed_prefix()
    prefix_hash = hashlib.sha256(prefix.encode('utf-8')).hexdigest()

    if os.path.exists(CACHE_PATH) and os.path.exists(CACHE_META_PATH):
        try:
            with open(CACHE_META_PATH, 'r') as f:
                saved_hash = f.read().strip()
            if saved_hash == prefix_hash:
                return True
        except Exception:
            pass

    try:
        subprocess.run([
            '/root/llama-b10964/llama-b10964/llama-completion',
            '-m', '/root/models/qwen2.5-3b-instruct-q4_0.gguf',
            '-p', prefix,
            '-n', '1',
            '-t', str(N_THREADS),
            '-c', str(CONTEXT_SIZE),
            '--prompt-cache', CACHE_PATH,
            '-no-cnv',
        ], capture_output=True, timeout=240, env={**os.environ, 'LD_LIBRARY_PATH': '/root/llama-b10964/llama-b10964'}, stdin=subprocess.DEVNULL)
        with open(CACHE_META_PATH, 'w') as f:
            f.write(prefix_hash)
        return os.path.exists(CACHE_PATH)
    except Exception:
        return False


def get_llm_advice(detections, nutrition, recommendations, weight):
    food_list = '、'.join([f"{FOOD_NAME_MAP.get(k, k)}{v}份" for k, v in detections.items()])
    cal = round(nutrition.get('calories', 0))
    cal_pct = round(recommendations.get('calories', {}).get('percentage', 0), 1)
    protein_pct = round(recommendations.get('protein', {}).get('percentage', 0), 1)
    carbs_pct = round(recommendations.get('carbs', {}).get('percentage', 0), 1)
    fiber_pct = round(recommendations.get('fiber', {}).get('percentage', 0), 1)

    cal_desc = "偏低" if cal_pct < 25 else ("適中" if cal_pct < 40 else "偏高")
    protein_desc = "不足" if protein_pct < 20 else ("適中" if protein_pct < 40 else "充足")
    carbs_desc = "不足" if carbs_pct < 20 else ("適中" if carbs_pct < 40 else "偏高")
    fiber_desc = "不足" if fiber_pct < 15 else "充足"

    categories_eaten = sorted({FOOD_CATEGORY_MAP[k] for k in detections.keys() if k in FOOD_CATEGORY_MAP})
    category_note = f"已攝取的食物類別：{'、'.join(categories_eaten)}，請勿再推薦同類別食物作為建議。" if categories_eaten else ""

    user_query = f"這餐吃了{food_list}，熱量{cal_desc}，蛋白質{protein_desc}，碳水{carbs_desc}，纖維{fiber_desc}。{category_note}"
    prompt = _build_prompt(user_query)

    cache_ok = _ensure_prompt_cache()

    cmd = [
        '/root/llama-b10964/llama-b10964/llama-completion',
        '-m', '/root/models/qwen2.5-3b-instruct-q4_0.gguf',
        '-p', prompt,
        '-n', str(GEN_MAX_TOKENS),
        '-t', str(N_THREADS),
        '-c', str(CONTEXT_SIZE),
        '--temp', '0.4',
        '--top-p', '0.9',
        '--top-k', '40',
        '--repeat-penalty', '1.15',
    ]
    cmd += ['-no-cnv']
    if cache_ok:
        cmd += ['--prompt-cache', CACHE_PATH, '--prompt-cache-ro']

    advice = ''
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=TIMEOUT_SECONDS, env={**os.environ, 'LD_LIBRARY_PATH': '/root/llama-b10964/llama-b10964'}, stdin=subprocess.DEVNULL)
        advice = _extract_advice(result.stdout)
    except subprocess.TimeoutExpired as e:
        if e.stdout:
            advice = _extract_advice(e.stdout)
    except Exception:
        advice = ''

    if not advice or len(advice) < 15:
        advice = _FALLBACK_ADVICE

    return advice


if __name__ == '__main__':
    import time

    demo_detections = {'白飯': 1, '雞胸肉': 2, '花椰菜': 1}
    demo_nutrition = {'calories': 520}
    demo_recommendations = {
        'calories': {'percentage': 26.0},
        'protein': {'percentage': 42.0},
        'carbs': {'percentage': 20.0},
        'fiber': {'percentage': 15.0},
    }
    t0 = time.time()
    print(get_llm_advice(demo_detections, demo_nutrition, demo_recommendations, 420))
    print(f"耗時：{time.time() - t0:.1f} 秒")