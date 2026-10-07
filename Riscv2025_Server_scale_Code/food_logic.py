"""
與模型無關的共用邏輯（營養計算、結果頁疊圖資料、份量滑桿資料、非同步 AI 建議暫存、上傳資料夾清理）。

刻意不 import 任何偵測器或 LLM：各板子（Realtek Matrix / Qilai EVB / Genio 720）自己的
Detector、LLMAdvisor、模型檔完全不受影響，只需要在 main.py 的 /detect 呼叫這裡的函式。
整合方式見 SHARED_UI_INTEGRATION.md。
"""
import collections
import os
import threading
import time
import uuid

NUTRIENT_KEYS = ['calories', 'protein', 'carbs', 'fiber', 'fat']

# 9 類（水果／零食）的中文顯示名。放在共用層，各平台的 LLMAdvisor 不需要提供。
FOOD_NAME_MAP = {
    'Apple': '蘋果', 'Banana': '香蕉', 'Mango': '芒果', 'Orange': '柳橙',
    'Pineapple': '鳳梨', 'Watermelon': '西瓜', 'MiniBiscuits': '小餅乾',
    'Snack': '零食', 'noodles': '麵條',
}


def split_by_confidence(detections, min_conf):
    """信心值太低的偵測不列入營養計算（仍可畫在圖上，標成不確定）。回傳 (confident, low_count)。"""
    confident = [d for d in detections if d['confidence'] >= min_conf]
    return confident, len(detections) - len(confident)


def _bbox_area(det):
    x_min, y_min, x_max, y_max = det['bbox']
    return max(0, x_max - x_min) * max(0, y_max - y_min)


def compute_nutrition(confident_detections, nutrition_data, detection_weight):
    """
    營養計算。nutrition_data[class] 是每 100g 的數值，並含 default_weight_g（典型單份重量）。
    - 有秤重（detection_weight > 0）：依每個偵測框面積占全部框面積的比例分配總重量
    - 沒有秤重：每個偵測用該食物的 default_weight_g
    回傳 dict：detection_counts、total_nutrition、food_breakdown（每種食物的重量與熱量，
    跟總量用同一套邏輯算，網頁上的表格才對得起來）。
    """
    detection_counts = {}
    total_nutrition = {k: 0 for k in NUTRIENT_KEYS}
    food_breakdown = {}

    for d in confident_detections:
        detection_counts[d['class_name']] = detection_counts.get(d['class_name'], 0) + 1

    def add(class_name, item_weight):
        ratio = item_weight / 100
        for k in NUTRIENT_KEYS:
            total_nutrition[k] += nutrition_data[class_name].get(k, 0) * ratio
        b = food_breakdown.setdefault(class_name, {'weight_g': 0.0, 'calories': 0.0})
        b['weight_g'] += item_weight
        b['calories'] += nutrition_data[class_name].get('calories', 0) * ratio

    total_area = sum(_bbox_area(d) for d in confident_detections)
    for d in confident_detections:
        name = d['class_name']
        if name not in nutrition_data:
            continue
        if detection_weight > 0:
            if total_area > 0:
                weight = detection_weight * _bbox_area(d) / total_area
            else:
                weight = detection_weight / len(confident_detections)
        else:
            weight = nutrition_data[name].get('default_weight_g', 100)
        add(name, weight)

    return {'detection_counts': detection_counts, 'total_nutrition': total_nutrition, 'food_breakdown': food_breakdown}


def build_boxes(detections, img_w, img_h, label_map, min_conf):
    """轉成網頁疊圖用的百分比座標（框線與文字由瀏覽器畫，不燒進圖片，也不對使用者顯示信心度）。"""
    def pct(v, total):
        return round(max(0.0, min(100.0, v / total * 100)), 2)
    boxes = []
    for d in detections:
        x1, y1, x2, y2 = d['bbox']
        boxes.append({
            'label': label_map.get(d['class_name'], d['class_name']),
            'confidence': round(d['confidence'], 2),
            'counted': d['confidence'] >= min_conf,
            'x': pct(x1, img_w), 'y': pct(y1, img_h),
            'w': pct(x2 - x1, img_w), 'h': pct(y2 - y1, img_h),
        })
    return boxes


def build_portion_items(food_breakdown, counts, nutrition_src, name_of):
    """份量滑桿用：每種食物的原始重量與每 100g 營養值，前端即時重算，不必重新辨識。"""
    return [{
        'name': name_of(cn),
        'count': counts.get(cn, 1),
        'weight_g': round(b['weight_g'], 1),
        'per100': {k: nutrition_src[cn].get(k, 0) for k in NUTRIENT_KEYS},
    } for cn, b in food_breakdown.items()]


class AdviceRegistry:
    """
    LLM 建議要好幾秒，結果頁先渲染，建議由前端向 /api/advice/<token> 非同步取得。
    這裡暫存「等待生成」的參數（單次使用、最多保留 max_pending 筆，重啟即清空）。
    """
    def __init__(self, max_pending=50):
        self._items = collections.OrderedDict()
        self._lock = threading.Lock()
        self._max = max_pending

    def register(self, params):
        token = uuid.uuid4().hex
        with self._lock:
            self._items[token] = params
            while len(self._items) > self._max:
                self._items.popitem(last=False)
        return token

    def pop(self, token):
        with self._lock:
            return self._items.pop(token, None)


def cleanup_old_uploads(folder, max_kept=200):
    """上傳資料夾最多保留 max_kept 個檔案，避免長期執行把儲存空間吃光。"""
    if max_kept <= 0:      # 0 = 不清理（RISC-V / Genio 維持原本行為）
        return
    try:
        files = [os.path.join(folder, f) for f in os.listdir(folder)]
    except FileNotFoundError:
        return
    if len(files) <= max_kept:
        return
    # 板子沒有 RTC，重開機後時間可能倒退，舊檔的修改時間會比現在還晚；這類檔案先刪，
    # 否則剛存的結果圖會被當成最舊而被刪掉（網頁圖片 404）。
    now = time.time()
    files.sort(key=lambda f: (0 if os.path.getmtime(f) > now else 1, os.path.getmtime(f)))
    for f in files[:len(files) - max_kept]:
        try:
            os.remove(f)
        except OSError:
            pass
