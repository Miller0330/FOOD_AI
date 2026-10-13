import subprocess
import re
import os
import hashlib
import threading

_lock = threading.Lock()

# ============ 可調參數（依裝置速度調整）============
LLAMA_DIR = os.environ.get('LLAMA_DIR', '/root/llama-b10964/llama-b10964')
LLAMA_BIN = LLAMA_DIR + '/llama-completion'
MODEL_PATH = os.environ.get('LLM_MODEL', '/root/models/qwen2.5-3b-instruct-q4_0.gguf')
IS_GEMMA = 'gemma' in MODEL_PATH.lower()
# 網頁可選的模型（id -> GGUF 路徑）；沒指定 id 時用上面的 MODEL_PATH（環境變數 LLM_MODEL）
MODELS = {
    'qwen': '/root/models/qwen2.5-3b-instruct-q4_0.gguf',
    'gemma': '/root/models/gemma-4-E4B-it-Q4_0.gguf',
}


def resolve_model(model_id=None):
    """回傳 (GGUF 路徑, 是否為 Gemma)。id 不認得就退回預設模型。"""
    path = MODELS.get(model_id) if model_id else None
    if path is None:
        return MODEL_PATH, IS_GEMMA
    return path, 'gemma' in path.lower()
GEN_MAX_TOKENS = int(os.environ.get('LLM_MAX_TOKENS', '300'))
CONTEXT_SIZE = 1024
N_THREADS = int(os.environ.get('LLM_THREADS', '8'))
TIMEOUT_SECONDS = int(os.environ.get('LLM_TIMEOUT', '240'))
USE_FEWSHOT = False

# ============ Prompt Cache 設定 ============
# 固定前綴（system prompt ± few-shot 範例）的 KV cache 存放路徑。
# 這段內容每次呼叫都完全相同，llama-cli 支援把算好的 KV 值存成檔案，
# 下次直接還原，跳過重新計算，可省下大量 prompt eval 時間。
CACHE_PATH = '/tmp/sysprompt.cache'
CACHE_META_PATH = '/tmp/sysprompt.cache.meta'

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

_ASSISTANT_MARKER = "assistant" + chr(10)
_STOP_MARKERS = ('<|im_end|>', '<|im_start|>', '<turn|>', '<|turn>')

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


def _extract_advice(stdout_bytes: bytes, is_gemma=None) -> str:
    stdout = stdout_bytes.decode('utf-8', errors='ignore')
    if is_gemma is None:
        is_gemma = IS_GEMMA

    # Gemma 以 --no-display-prompt 執行，輸出本身就只有回答；Qwen 才需要用 assistant 標記切出回答
    idx = -1 if is_gemma else stdout.rfind(_ASSISTANT_MARKER)
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
        result = subprocess.run([
            '/root/llama-b10964/llama-b10964/llama-completion',
            '-m', '/root/models/qwen2.5-3b-instruct-q4_0.gguf',
            '-p', prefix,
            '-n', '1',
            '-t', str(N_THREADS),
            '-c', str(CONTEXT_SIZE),
            '--prompt-cache', CACHE_PATH,
            '-no-cnv',
        ], capture_output=True, timeout=240, env={**os.environ, 'LD_LIBRARY_PATH': '/root/llama-b10964/llama-b10964'}, stdin=subprocess.DEVNULL)

        if result.returncode == 0 and os.path.exists(CACHE_PATH):
            with open(CACHE_META_PATH, 'w') as f:
                f.write(prefix_hash)
            return True

        # 建置失敗或中途被中斷：清掉可能不完整的殘留檔案，避免下次被誤判為有效 cache
        if os.path.exists(CACHE_PATH):
            os.remove(CACHE_PATH)
        if os.path.exists(CACHE_META_PATH):
            os.remove(CACHE_META_PATH)
        return False
    except Exception:
        return False


def get_llm_advice(detections, nutrition, recommendations, weight, name_map=None, category_map=None, model=None):
    model_path, is_gemma = resolve_model(model)
    if not detections:
        return "這次沒有偵測到食物，請重新拍攝或確認照片中有清楚的食物內容。"

    name_map = name_map if name_map is not None else FOOD_NAME_MAP
    category_map = category_map if category_map is not None else FOOD_CATEGORY_MAP

    food_list = '、'.join([f"{name_map.get(k, k)}{v}份" for k, v in detections.items()])
    cal_pct = round(recommendations.get('calories', {}).get('percentage', 0), 1)
    protein_pct = round(recommendations.get('protein', {}).get('percentage', 0), 1)
    carbs_pct = round(recommendations.get('carbs', {}).get('percentage', 0), 1)
    fiber_pct = round(recommendations.get('fiber', {}).get('percentage', 0), 1)
    fat_pct = round(recommendations.get('fat', {}).get('percentage', 0), 1)

    cal_desc = "偏低" if cal_pct < 25 else ("適中" if cal_pct < 40 else "偏高")
    protein_desc = "不足" if protein_pct < 20 else ("適中" if protein_pct < 40 else "充足")
    carbs_desc = "不足" if carbs_pct < 20 else ("適中" if carbs_pct < 40 else "偏高")
    fiber_desc = "不足" if fiber_pct < 15 else "充足"
    fat_desc = "偏低" if fat_pct < 20 else ("適中" if fat_pct < 40 else "偏高")

    categories_eaten = sorted({category_map[k] for k in detections.keys() if k in category_map})
    category_note = f"已攝取的食物類別：{'、'.join(categories_eaten)}，請勿再推薦同類別食物作為建議。" if categories_eaten else ""

    # 限制長度/格式的指示直接寫進使用者問題本身（-p system prompt 這個引擎版本不生效，已實測確認）。
    # 明確要求評論「這餐吃的東西」本身，不然像蘋果、香蕉這種營養屬性很像的食物，
    # 分類結果(偏低/不足/適中)幾乎一樣，模型很容易套同一套通用建議範本、完全不提到實際吃了什麼。
    # 把實際數字跟「缺什麼、可以補什麼具體食物」都算好交給模型，讓建議有內容、不空泛；
    # 補充食物用程式先挑好候選（避開已吃過的類別），模型只負責用口語組織，
    # 不然小模型自己想常常只會講「搭配優質蛋白質」這種沒有具體食物名的話。
    weight_note = f"，總重約{round(weight)}公克" if weight and weight > 0 else ""
    numbers = (
        f"約{round(nutrition.get('calories', 0))}大卡、蛋白質{round(nutrition.get('protein', 0), 1)}公克、"
        f"碳水{round(nutrition.get('carbs', 0), 1)}公克、纖維{round(nutrition.get('fiber', 0), 1)}公克、"
        f"脂肪{round(nutrition.get('fat', 0), 1)}公克{weight_note}"
    )

    needs = []  # (缺的營養素, [(候選食物, 類別)])
    if protein_pct < 20:
        needs.append(('蛋白質', [('水煮蛋', '蛋類'), ('無糖豆漿或豆腐', '豆製品類'), ('雞胸肉或魚肉', '肉類'), ('無糖優格', '乳品類')]))
    if carbs_pct < 20:
        needs.append(('複合碳水', [('地瓜', '主食類'), ('燕麥', '主食類'), ('全麥吐司', '主食類')]))
    if fiber_pct < 15:
        needs.append(('膳食纖維', [('燙青菜', '蔬菜類'), ('菇類', '蔬菜類')]))
    if fat_pct < 15 and len(needs) < 3:
        needs.append(('好的脂肪', [('一小把堅果', '點心類')]))

    suggestion_lines = []
    for nutrient, candidates in needs[:3]:
        picks = [name for name, cat in candidates if cat not in categories_eaten][:3]
        if picks:
            suggestion_lines.append(f"缺{nutrient}，可補{'、'.join(picks)}")
    if cal_pct >= 40 or fat_pct >= 40:
        suggestion_lines.append("熱量或油脂偏高，提醒下一餐清淡一點")
    suggestion_note = ("可以參考的補充方向：" + "；".join(suggestion_lines) + "。") if suggestion_lines else "這餐整體均衡，給予肯定並提醒維持。"

    query = (
        f"使用者這餐吃了{food_list}，營養約：{numbers}。整體評估：熱量{cal_desc}，蛋白質{protein_desc}，"
        f"碳水{carbs_desc}，纖維{fiber_desc}，脂肪{fat_desc}。{category_note}{suggestion_note}"
        f"你是一位親切的營養師，正在跟朋友面對面聊天。請像口說一樣自然地回應："
        f"先具體說說這餐{food_list}的優點（可以提到它的特色營養或對身體的好處），"
        f"再誠實指出還缺什麼，最後用上面提到的補充食物給出1到2個容易做到的建議，並簡單說明為什麼有幫助。"
        f"語氣溫暖口語，用「你」稱呼對方，用台灣繁體中文，"
        f"寫成一段話，約3到4句，共100到150字，不要超過150字，回答中必須提到食物名稱（自然地說，不要說「1份」「這份」），不要用「欸」開頭，"
        f"禁止使用條列、標題、表格、星號、表情符號。"
    )

    if is_gemma:
        prompt = "<|turn>user" + chr(10) + query + "<turn|>" + chr(10) + "<|turn>model" + chr(10)
    else:
        prompt = ("<|im_start|>system" + chr(10) + "你是專業的營養師，只使用台灣繁體中文回答。<|im_end|>" + chr(10)
                  + "<|im_start|>user" + chr(10) + query + chr(10) + "<|im_end|>" + chr(10)
                  + "<|im_start|>assistant" + chr(10))

    cmd = [
        LLAMA_BIN, '-m', model_path, '-p', prompt,
        '-n', str(GEN_MAX_TOKENS), '-t', str(N_THREADS), '-c', str(CONTEXT_SIZE),
        '--temp', '0.4', '--top-p', '0.9', '--top-k', '40', '--repeat-penalty', '1.15',
        '-no-cnv',
    ]
    if is_gemma:
        cmd.append('--no-display-prompt')   # Gemma 沒有 Qwen 的 assistant 標記，不加的話回應會夾帶提示詞原文

    advice = ''
    try:
        with _lock:
            result = subprocess.run(cmd, capture_output=True, timeout=TIMEOUT_SECONDS,
                                    env={**os.environ, 'LD_LIBRARY_PATH': LLAMA_DIR}, stdin=subprocess.DEVNULL)
        advice = _extract_advice(result.stdout, is_gemma)
    except subprocess.TimeoutExpired as e:
        if e.stdout:
            advice = _extract_advice(e.stdout, is_gemma)
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