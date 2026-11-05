"""
Windows 版 LLM 建議：透過 HTTP 呼叫 llama-server（OpenAI 相容的 /v1/chat/completions）。

提示詞的內容與格式沿用 Matrix 版（程式先算好數字與補充方向，模型只負責用口語說出來）；
換模型只要改 llama-server 載入的 GGUF，這裡不用動。可用環境變數調整：
  LLM_URL         llama-server 位址（預設 http://127.0.0.1:8080）
  LLM_MODEL       送給伺服器的 model 名稱（llama-server 只載入一個模型，預設值即可）
  LLM_MAX_TOKENS  單次生成上限（預設 400）
  LLM_TIMEOUT     單次請求最長等待秒數（預設 180）
  LLM_TEMP        取樣溫度（預設 0.4）
"""
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

LLM_URL = os.environ.get('LLM_URL', 'http://127.0.0.1:8080').rstrip('/')
LLM_MODEL = os.environ.get('LLM_MODEL', 'local')
GEN_MAX_TOKENS = int(os.environ.get('LLM_MAX_TOKENS', '400'))
TIMEOUT_SECONDS = int(os.environ.get('LLM_TIMEOUT', '180'))
TEMPERATURE = float(os.environ.get('LLM_TEMP', '0.4'))
WARMUP_WAIT_SECONDS = int(os.environ.get('LLM_WARMUP_WAIT', '120'))   # llama-server 載入模型期間（503）最多等多久

SYSTEM_PROMPT = "你是專業的營養師，只使用台灣繁體中文回答。"

_FALLBACK_ADVICE = "這餐整體攝取尚屬均衡，建議持續維持均衡飲食，適度增加蔬菜與全穀類以補充膳食纖維，並注意蛋白質與水分的攝取。"

try:
    import opencc
    _CONVERTER = opencc.OpenCC('s2twp')
except Exception:
    _CONVERTER = None

_lock = threading.Lock()
last_stats = {}   # 最近一次生成的速度指標（維運比較用）


def _post(path, payload, timeout):
    req = urllib.request.Request(LLM_URL + path, data=json.dumps(payload).encode('utf-8'),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def chat(messages, max_tokens=None, temperature=None):
    """送出一次對話，回傳 (文字, finish_reason)。連不上或逾時會丟出例外，由呼叫端決定退路。"""
    payload = {
        'model': LLM_MODEL,
        'messages': messages,
        'max_tokens': max_tokens or GEN_MAX_TOKENS,
        'temperature': TEMPERATURE if temperature is None else temperature,
        'top_p': 0.9, 'top_k': 40, 'repeat_penalty': 1.1,
        # Qwen3 這類有思考模式的模型：關掉思考，只要最後的口語建議；其他模型會忽略這個參數
        'chat_template_kwargs': {'enable_thinking': False},
    }
    t0 = time.time()
    data = _post('/v1/chat/completions', payload, TIMEOUT_SECONDS)
    choice = data['choices'][0]
    timings = data.get('timings', {})
    last_stats.update({
        'total_s': round(time.time() - t0, 2),
        'prompt_tokens': data.get('usage', {}).get('prompt_tokens'),
        'completion_tokens': data.get('usage', {}).get('completion_tokens'),
        'prompt_per_second': timings.get('prompt_per_second'),
        'predicted_per_second': timings.get('predicted_per_second'),
    })
    print(f"[LLM] {last_stats}")
    return choice['message'].get('content') or '', choice.get('finish_reason')


_model_name = None


def model_name():
    """llama-server 目前載入的模型名稱（維運比較用）；連不上就回傳 None，不要拿預設值冒充模型名稱。"""
    global _model_name
    if _model_name is None:
        try:
            with urllib.request.urlopen(LLM_URL + '/v1/models', timeout=5) as resp:
                model_id = json.loads(resp.read().decode('utf-8'))['data'][0]['id']
            _model_name = re.split(r'[\\/]', model_id)[-1]   # llama-server 回的是模型檔完整路徑，只留檔名
        except Exception:
            return None
    return _model_name


def _clean(text, truncated, keep_paragraphs=False):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S)
    if _CONVERTER is not None:
        try:
            text = _CONVERTER.convert(text)
        except Exception:
            pass
    # 去掉 markdown 符號與條列記號（小模型常不守「不要條列」的指示）
    text = re.sub(r'^\s*(?:[-*•]|\d+[.、)])\s+', '', text, flags=re.M)
    text = re.sub(r'[*#`_>|]', '', text)
    if keep_paragraphs:
        text = re.sub(r'[ \t]*\n\s*\n\s*', '\n\n', text)
        text = re.sub(r'(?<!\n)\n(?!\n)', '', text).strip()
    else:
        text = re.sub(r'\s*\n+\s*', '', text).strip()
    if truncated:   # 被 max_tokens 截斷：退到最後一個完整句子
        last_end = max(text.rfind('。'), text.rfind('！'), text.rfind('？'))
        text = text[:last_end + 1] if last_end != -1 else ''
    return text


def warmup():
    """確認 llama-server 可用並跑一次極短的生成。連不上只記錄警告，不丟例外（服務仍可啟動）。"""
    deadline = time.time() + WARMUP_WAIT_SECONDS
    while True:
        try:
            with urllib.request.urlopen(LLM_URL + '/health', timeout=5) as resp:
                if resp.status == 200:
                    break
        except urllib.error.HTTPError as e:
            if e.code != 503:   # 503 = 還在載入模型
                print(f"[LLM] llama-server 回應異常：HTTP {e.code}")
                return False
        except Exception as e:
            print(f"[LLM] 連不上 llama-server（{LLM_URL}）：{e}；建議將使用備用文字")
            return False
        if time.time() > deadline:
            print("[LLM] 等待 llama-server 載入模型逾時")
            return False
        time.sleep(2)
    try:
        chat([{'role': 'user', 'content': '你好'}], max_tokens=4)
    except Exception as e:
        print(f"[LLM] 暖機生成失敗：{e}")
        return False
    return True


def build_query(detections, nutrition, recommendations, weight, name_map, category_map):
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

    return (
        f"使用者這餐吃了{food_list}，營養約：{numbers}。整體評估：熱量{cal_desc}，蛋白質{protein_desc}，"
        f"碳水{carbs_desc}，纖維{fiber_desc}，脂肪{fat_desc}。{category_note}{suggestion_note}"
        f"你是一位親切的營養師，正在跟朋友面對面聊天。請像口說一樣自然地回應："
        f"先具體說說這餐{food_list}的優點（可以提到它的特色營養或對身體的好處），"
        f"再誠實指出還缺什麼，最後用上面提到的補充食物給出1到2個容易做到的建議，並簡單說明為什麼有幫助。"
        f"語氣溫暖口語，用「你」稱呼對方，用台灣繁體中文，"
        f"寫成一段話，約3到4句，共100到150字，不要超過150字，回答中必須提到食物名稱（自然地說，不要說「1份」「這份」），不要用「欸」開頭，"
        f"禁止使用條列、標題、表格、星號、表情符號。"
    )


def get_llm_advice(detections, nutrition, recommendations, weight, name_map=None, category_map=None):
    if not detections:
        return "這次沒有偵測到食物，請重新拍攝或確認照片中有清楚的食物內容。"

    query = build_query(detections, nutrition, recommendations, weight, name_map or {}, category_map or {})
    advice = ''
    try:
        with _lock:
            text, finish = chat([{'role': 'system', 'content': SYSTEM_PROMPT},
                                 {'role': 'user', 'content': query}])
        advice = _clean(text, truncated=(finish == 'length'))
    except Exception as e:
        print(f"[LLM] 取得建議失敗：{e}")

    if not advice or len(advice) < 15:
        advice = _FALLBACK_ADVICE
    return advice


def _level(pct):
    return "明顯不足" if pct < 50 else ("偏低" if pct < 85 else ("適中" if pct <= 115 else "偏高"))


_DAILY_LABELS = (('calories', '熱量', '大卡'), ('protein', '蛋白質', '公克'), ('carbs', '碳水', '公克'),
                 ('fiber', '纖維', '公克'), ('fat', '脂肪', '公克'))


def _fmt(key, v):
    return str(round(v)) if key == 'calories' else str(round(v, 1))


def _record_text(payload):
    """payload 由 main.py 組好：meals（時間、食物、熱量）、totals、targets、categories。數字全部是程式算的。"""
    meal_lines = [f"第{i}餐（{m['time']}）：{'、'.join(m['foods'])}，約{round(m['calories'])}大卡"
                  for i, m in enumerate(payload['meals'], 1)]
    t, tg = payload['totals'], payload['targets']
    total_lines = []
    for key, label, unit in _DAILY_LABELS:
        pct = round(t[key] / tg[key] * 100) if tg[key] else 0
        total_lines.append(f"{label}{_fmt(key, t[key])}{unit}（每日建議量{_fmt(key, tg[key])}{unit}的{pct}%，{_level(pct)}）")
    cats = '、'.join(payload['categories']) or '無資料'
    return (f"【各餐】\n" + "\n".join(meal_lines) + "\n"
            f"【全天合計】\n" + "\n".join(total_lines) + "\n"
            f"【吃過的食物類別】{cats}\n")


def build_daily_query(payload):
    return (
        f"以下是使用者{payload['day']}一整天的飲食紀錄，數字都已經由程式算好，請直接引用，不要自己重新計算，"
        f"也不要提到紀錄裡沒有出現的食物。\n" + _record_text(payload) +
        f"你是一位親切的營養師，正在跟朋友聊他今天的飲食。請用台灣繁體中文，寫成兩段話，共180到250字："
        f"第一段總結今天吃了什麼，以及跟每日建議量比起來的整體狀況（挑最突出的一兩項說就好）；"
        f"第二段說說你看出的飲食模式（例如哪一餐最豐盛、少了哪類食物），再給明天1到2個具體容易做到的建議。"
        f"語氣溫暖口語，用「你」稱呼對方，不要使用條列、標題、表格、星號、表情符號。"
    )


def fallback_daily_summary(payload):
    """LLM 不可用時，由程式直接組一段只含數字的總結（不編造任何內容）。"""
    t, tg = payload['totals'], payload['targets']
    parts = [f"今天共記錄{len(payload['meals'])}餐"]
    for key, label, unit in _DAILY_LABELS:
        pct = round(t[key] / tg[key] * 100) if tg[key] else 0
        parts.append(f"{label}{_fmt(key, t[key])}{unit}（{pct}%）")
    return "，".join(parts[:1]) + "，累計" + "、".join(parts[1:]) + "。（AI 營養師目前無法使用，以上為程式統計結果）"


def get_daily_summary(payload):
    """回傳 {'text', 'fallback', 'model', 'seconds'}。LLM 失敗時 fallback 為 True，文字是程式組的統計。"""
    t0 = time.time()
    text = ''
    try:
        with _lock:
            raw, finish = chat([{'role': 'system', 'content': SYSTEM_PROMPT},
                                {'role': 'user', 'content': build_daily_query(payload)}],
                               max_tokens=max(GEN_MAX_TOKENS, 700))
        text = _clean(raw, truncated=(finish == 'length'), keep_paragraphs=True)
    except Exception as e:
        print(f"[LLM] 取得每日總結失敗：{e}")
    fallback = len(text) < 30
    return {'text': fallback_daily_summary(payload) if fallback else text, 'fallback': fallback,
            'model': model_name(), 'seconds': round(time.time() - t0, 1)}


CHAT_MAX_TOKENS = int(os.environ.get('LLM_CHAT_MAX_TOKENS', '350'))

_CHAT_RULES = (
    "你是專業又親切的營養師，正在跟使用者持續聊天，只使用台灣繁體中文，語氣溫暖口語，用「你」稱呼對方。"
    "請像聊天一樣簡短自然地回答，每次不超過120字，不要使用條列、標題、表格、星號、表情符號。"
    "下面是使用者今天的飲食紀錄，數字都已經由程式算好，請直接引用，不要自己重新計算。"
    "只能根據這份紀錄與對話內容回答：不要提到紀錄裡沒有的食物，也不要編造使用者沒說過的喜好、病史或習慣；"
    "使用者問該怎麼吃時，先給出具體、容易做到的食物建議，需要時最後才反問一個問題；不知道的事就直接說不知道。只提供一般飲食建議，不做醫療診斷，有疾病或特殊狀況請使用者諮詢醫師或營養師。"
)


def build_chat_system(payload):
    record = _record_text(payload) if payload['meals'] else "使用者今天還沒有任何飲食紀錄。\n"
    return _CHAT_RULES + f"\n今天是{payload['day']}。\n" + record


def _clean_piece(piece):
    """串流時逐段套用的輕量清理：轉繁體、去掉 markdown 符號。"""
    if _CONVERTER is not None:
        try:
            piece = _CONVERTER.convert(piece)
        except Exception:
            pass
    return re.sub(r'[*#`_>|]', '', piece)


def chat_stream(payload, history, user_message):
    """
    跟營養師對話（串流）。每一輪都把「今天的紀錄＋對話歷史＋這次的問題」整段組給模型（模型本身不記得上一輪）；
    llama-server 會重用已處理過的前綴，所以歷史變長也不會讓每輪明顯變慢。
    逐一 yield ('text', 一段文字)，最後 yield ('done', {'model', 'seconds', 'first_token_s'})。
    """
    messages = [{'role': 'system', 'content': build_chat_system(payload)}]
    messages += [{'role': m['role'], 'content': m['content']} for m in history]
    messages.append({'role': 'user', 'content': user_message})
    body = {
        'model': LLM_MODEL, 'messages': messages, 'max_tokens': CHAT_MAX_TOKENS, 'temperature': 0.5,
        'top_p': 0.9, 'top_k': 40, 'repeat_penalty': 1.1, 'stream': True,
        'stream_options': {'include_usage': True},
        'chat_template_kwargs': {'enable_thinking': False},
    }
    req = urllib.request.Request(LLM_URL + '/v1/chat/completions', data=json.dumps(body).encode('utf-8'),
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    first = None
    usage = {}
    with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
        for raw in resp:
            line = raw.decode('utf-8').strip()
            if not line.startswith('data:') or line.endswith('[DONE]'):
                continue
            d = json.loads(line[5:])
            usage = d.get('usage') or usage
            for c in d.get('choices', []):
                piece = c.get('delta', {}).get('content')
                if piece:
                    if first is None:
                        first = round(time.time() - t0, 2)
                    yield ('text', _clean_piece(piece))
    total = round(time.time() - t0, 1)
    print(f"[LLM] chat first_token={first}s total={total}s tokens={usage.get('completion_tokens')}/{usage.get('prompt_tokens')}")
    yield ('done', {'model': model_name(), 'seconds': total, 'first_token_s': first})


if __name__ == '__main__':
    demo_detections = {'白飯': 1, '雞胸肉': 2, '花椰菜': 1}
    demo_nutrition = {'calories': 520, 'protein': 38.0, 'carbs': 60.0, 'fiber': 4.0, 'fat': 12.0}
    demo_recommendations = {
        'calories': {'percentage': 26.0},
        'protein': {'percentage': 42.0},
        'carbs': {'percentage': 20.0},
        'fiber': {'percentage': 15.0},
        'fat': {'percentage': 18.0},
    }
    print('warmup:', warmup())
    t0 = time.time()
    print(get_llm_advice(demo_detections, demo_nutrition, demo_recommendations, 420))
    print(f"耗時：{time.time() - t0:.1f} 秒")
