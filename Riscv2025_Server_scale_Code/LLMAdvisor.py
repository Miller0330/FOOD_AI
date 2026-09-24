import subprocess
import threading
import re

# ============ 可調參數 ============
CLI_PATH = '/data/models/gemma4_e4b/MatriXRT-llm-cli'
ENGINE_PATH = '/data/models/gemma4_e4b/packed_engine_e4b_fixed_strip_llmonly.json'
ENGINE_DIR = '/data/models/gemma4_e4b'
READY_MARKER = 'Ask your question'
ANSWER_MARKER = '[A]:'
RESPONSE_TIMEOUT = 90  # 單次生成最長等待秒數（回答變長後從60調高）
STARTUP_TIMEOUT = 120  # 引擎載入最長等待秒數

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

_proc = None
_lock = threading.Lock()


# 常駐啟動 MatriXRT-llm-cli，避免每次請求都要重新載入引擎（載入一次要 20-75 秒）
def _ensure_engine_running():
    global _proc
    if _proc is not None and _proc.poll() is None:
        return True

    _proc = subprocess.Popen(
        [CLI_PATH, ENGINE_PATH],
        cwd=ENGINE_DIR,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    buf = ''
    import time
    start = time.time()
    while time.time() - start < STARTUP_TIMEOUT:
        line = _proc.stdout.readline()
        if not line:
            break
        buf += line
        if READY_MARKER in buf:
            return True
    return False


def _read_until_ready(timeout):
    import time
    buf = ''
    start = time.time()
    while time.time() - start < timeout:
        line = _proc.stdout.readline()
        if not line:
            break
        buf += line
        if READY_MARKER in buf:
            break
    return buf


def _extract_answer(raw: str) -> str:
    idx = raw.rfind(ANSWER_MARKER)
    if idx == -1:
        return ''
    text = raw[idx + len(ANSWER_MARKER):]
    idx2 = text.find(READY_MARKER)
    if idx2 != -1:
        text = text[:idx2]
    text = text.strip()

    # 過濾掉 thermal log 這類雜訊行
    lines = [ln for ln in text.split('\n') if not re.match(r'^\[\s*\d+\.\d+\]', ln.strip())]
    text = '\n'.join(lines).strip()
    return text


def get_llm_advice(detections, nutrition, recommendations, weight, name_map=None, category_map=None):
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

    advice = ''
    try:
        with _lock:
            if not _ensure_engine_running():
                return _FALLBACK_ADVICE
            _proc.stdin.write(query + '\n')
            _proc.stdin.flush()
            raw = _read_until_ready(RESPONSE_TIMEOUT)
            advice = _extract_answer(raw)
    except Exception:
        advice = ''

    if not advice or len(advice) < 10:
        advice = _FALLBACK_ADVICE

    return advice


if __name__ == '__main__':
    import time

    demo_detections = {'Apple': 1}
    demo_nutrition = {'calories': 52, 'protein': 0.3, 'carbs': 13.8, 'fiber': 2.4, 'fat': 0.15}
    demo_recommendations = {
        'calories': {'percentage': 2.6},
        'protein': {'percentage': 0.5},
        'carbs': {'percentage': 4.6},
        'fiber': {'percentage': 9.6},
        'fat': {'percentage': 0.2},
    }
    t0 = time.time()
    print(get_llm_advice(demo_detections, demo_nutrition, demo_recommendations, 100))
    print(f"耗時：{time.time() - t0:.1f} 秒")
