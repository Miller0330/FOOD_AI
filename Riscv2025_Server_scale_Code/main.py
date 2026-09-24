from LLMAdvisor import get_llm_advice, _ensure_engine_running, FOOD_NAME_MAP
from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import base64
from datetime import datetime
import time
import json
import uuid
import threading
import collections

# 導入.py檔案
from Detector import TFLiteDetector
from Detector100 import UECFood100Detector, IMG_SIZE as DETECTOR100_IMG_SIZE
from Calculate import calculate_diet_recommendations
from WeightSensor import initialize_weight_sensor, get_weight_data
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB最大上傳限制
app.config['MODEL_PATH'] = 'detectv2_float16.tflite'

# 如果上傳資料夾不存在，則創建
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 初始化檢測器
detector = TFLiteDetector(
    model_path=app.config['MODEL_PATH'],
    img_size=640,
    conf_threshold=0.25,
    iou_threshold=0.45
)

# 從JSON檔案載入營養數據
with open('nutrition_data.json', 'r') as file:
    nutrition_data = json.load(file)

NUTRIENT_KEYS = ['calories', 'protein', 'carbs', 'fiber', 'fat']
MIN_CONFIDENCE_FOR_NUTRITION = 0.5  # 偵測器本身的門檻只有0.25(為了不漏掉東西)，
                                     # 但信心值太低的偵測不該被算進營養總量，容易誤導使用者

# ===== 100類UEC-FOOD主食辨識（方案C：先當獨立第二入口，測試效果好不好再決定要不要合併）=====
with open('nutrition_data_100_uecfood.json', 'r', encoding='utf-8') as file:
    nutrition_data_100_raw = json.load(file)

detector100 = UECFood100Detector(conf_threshold=0.15, iou_threshold=0.45)

# matrixrt_yolo標註圖上/偵測結果的class_name是英文（來自labels.txt，即訓練時的UEC類別英文名），
# 這裡依UEC編號排序建立「英文名-> 完整營養資料」的對照表，跟訓練時的class index順序完全一致
_uec_keys_sorted = sorted(nutrition_data_100_raw.keys(), key=lambda k: int(k[3:6]))
nutrition_data_100 = {}
name_map_100 = {}
category_map_100 = {}
for key in _uec_keys_sorted:
    entry = nutrition_data_100_raw[key]
    english_key = entry['english_name'].replace(' ', '_').replace("'", '').replace('-', '_')
    chinese_name = key.split('_', 1)[1]  # "UEC003_炒什錦飯" -> "炒什錦飯"
    nutrition_data_100[english_key] = entry
    name_map_100[english_key] = chinese_name
    category_map_100[english_key] = entry.get('category', '')
MIN_CONFIDENCE_FOR_NUTRITION_100 = 0.3  # 100類模型的原始信心值普遍比9類模型低（小資料集+多類別），門檻放寬一些

# 初始化重量感測器
try:
    weight_sensor_connected = initialize_weight_sensor()
    if weight_sensor_connected:
        print("重量感測器初始化成功")
    else:
        print("重量感測器初始化失敗，但系統會繼續運行")
except Exception as e:
    print(f"重量感測器初始化錯誤: {e}")
    weight_sensor_connected = False

# 在模組載入時就把AI引擎叫醒（而不是放在 __main__ 區塊），
# 這樣不管是直接執行 main.py 還是被 gunicorn 這類 WSGI伺服器 import，引擎都會啟動
print("正在準備 AI 模型，請稍候...")
_ensure_engine_running()
print("AI 模型準備完成")

MAX_KEPT_UPLOADS = 200  # 上傳資料夾最多保留幾個檔案（原圖+標註圖各算一個），避免長期執行把 /data 空間吃光

def _cleanup_old_uploads():
    folder = app.config['UPLOAD_FOLDER']
    try:
        files = [os.path.join(folder, f) for f in os.listdir(folder)]
    except FileNotFoundError:
        return
    if len(files) <= MAX_KEPT_UPLOADS:
        return
    files.sort(key=lambda f: os.path.getmtime(f))
    for f in files[:len(files) - MAX_KEPT_UPLOADS]:
        try:
            os.remove(f)
        except OSError:
            pass

# LLM建議要好幾秒才生成完，不讓它擋住整個結果頁：結果頁先渲染，建議由頁面載入完成後
# 再向 /api/advice/<token> 非同步取得。這裡暫存「等待生成」的參數（重啟即清空，無所謂）。
MAX_PENDING_ADVICE = 50
_pending_advice = collections.OrderedDict()
_pending_lock = threading.Lock()

def _attach_advice(result_data, detection_counts, total_nutrition, recommendations, weight,
                   name_map=None, category_map=None):
    if not detection_counts:
        # 沒偵測到食物時建議是固定文字，不需要動用LLM，直接內嵌
        result_data['llm_advice'] = get_llm_advice(detection_counts, total_nutrition, recommendations, weight)
        return
    token = uuid.uuid4().hex
    with _pending_lock:
        _pending_advice[token] = (detection_counts, total_nutrition, recommendations, weight, name_map, category_map)
        while len(_pending_advice) > MAX_PENDING_ADVICE:
            _pending_advice.popitem(last=False)
    result_data['advice_token'] = token

@app.route('/api/advice/<token>')
def get_advice(token):
    with _pending_lock:
        params = _pending_advice.pop(token, None)
    if params is None:
        return jsonify({'success': False, 'error': '建議已過期，請重新辨識'}), 404
    detection_counts, total_nutrition, recommendations, weight, name_map, category_map = params
    t0 = time.time()
    print("開始呼叫LLM...")
    advice = get_llm_advice(detection_counts, total_nutrition, recommendations, weight,
                            name_map=name_map, category_map=category_map)
    return jsonify({'success': True, 'advice': advice, 'seconds': round(time.time() - t0, 1)})

def _build_boxes(detections, img_w, img_h, label_map, min_conf):
    """轉成網頁疊圖用的百分比座標。標註不再燒進圖片：中文字由瀏覽器畫，也不對使用者顯示信心度。"""
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


def _build_portion_items(food_breakdown, counts, nutrition_src, name_of):
    """給網頁份量滑桿用：每種食物的原始重量與每100g營養值，前端即時重算不必重新辨識。"""
    return [{
        'name': name_of(cn),
        'count': counts.get(cn, 1),
        'weight_g': round(b['weight_g'], 1),
        'per100': {k: nutrition_src[cn].get(k, 0) for k in NUTRIENT_KEYS},
    } for cn, b in food_breakdown.items()]


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    # 獲取表單中的重量數據
    detection_weight = float(request.form.get('weight', 0))
    print(f"辨識時重量: {detection_weight}g")
    
    if file:
        request_start = time.time()
        # 儲存上傳的檔案
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = secure_filename(f"{timestamp}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 使用OpenCV讀取圖像
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': '無法處理圖像'})
        
        # 執行檢測
        start_time = time.time()
        detections = detector.detect(img)
        inference_time = time.time() - start_time
        
        # 存一份乾淨的檢視用圖（不燒框線與文字，框線由網頁疊上去）。一律存成.jpg，
        # 不要照抄原始上傳檔名的副檔名——手機/Windows常見的.jfif，cv2.imwrite不認得會直接噴錯
        result_stem = os.path.splitext(filename)[0]
        result_filename = f"result_{result_stem}.jpg"
        result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        cv2.imwrite(result_filepath, img)
        
        # 信心值太低的偵測不列入營養計算（但仍會畫在標註圖上），避免誤導使用者
        low_confidence_count = sum(1 for d in detections if d['confidence'] < MIN_CONFIDENCE_FOR_NUTRITION)
        confident_detections = [d for d in detections if d['confidence'] >= MIN_CONFIDENCE_FOR_NUTRITION]
        boxes = _build_boxes(detections, img.shape[1], img.shape[0], FOOD_NAME_MAP, MIN_CONFIDENCE_FOR_NUTRITION)

        # 按類別計數檢測結果
        detection_counts = {}
        total_nutrition = {'calories': 0, 'protein': 0, 'carbs': 0, 'fiber': 0, 'fat': 0}

        for detection in confident_detections:
            class_name = detection['class_name']
            if class_name in detection_counts:
                detection_counts[class_name] += 1
            else:
                detection_counts[class_name] = 1

        # 每個偵測框的面積（用來按大小分配重量，而不是齊頭式平均）
        def _bbox_area(det):
            x_min, y_min, x_max, y_max = det['bbox']
            return max(0, x_max - x_min) * max(0, y_max - y_min)

        total_area = sum(_bbox_area(d) for d in confident_detections)
        detections = confident_detections

        # 每個食物類別的實際明細（重量、熱量），跟下面 total_nutrition 用同一套邏輯算，
        # 讓網頁上「食物明細」表格跟「營養素攝取比例」的數字保證對得起來
        # （舊版食物明細是模板自己另外用 每100g熱量*數量 重算一次，跟這裡的加權結果對不上）
        food_breakdown = {}

        def _add_breakdown(class_name, item_weight):
            b = food_breakdown.setdefault(class_name, {'weight_g': 0.0, 'calories': 0.0})
            b['weight_g'] += item_weight
            b['calories'] += nutrition_data[class_name].get('calories', 0) * (item_weight / 100)

        if detections and detection_weight > 0:
            # 有秤重數據：依每個偵測框佔全部框面積的比例分配總重量，
            # 而不是不分物件大小地平均分配（例如一根香蕉跟一顆小餅乾框選面積差很多）
            print(f"總重量: {detection_weight}g, 總項目數: {len(detections)}, 依偵測框面積比例分配")

            for detection in detections:
                class_name = detection['class_name']
                if class_name not in nutrition_data:
                    continue
                area = _bbox_area(detection)
                item_weight = (detection_weight * area / total_area) if total_area > 0 else (detection_weight / len(detections))
                weight_ratio = item_weight / 100  # nutrition_data 是每100g的數值

                for nutrient in NUTRIENT_KEYS:
                    total_nutrition[nutrient] += nutrition_data[class_name].get(nutrient, 0) * weight_ratio
                _add_breakdown(class_name, item_weight)
        else:
            # 沒有秤重數據：用每種食物的典型單位重量估算，而不是把每個偵測到的物件都當成100g
            for detection in detections:
                class_name = detection['class_name']
                if class_name not in nutrition_data:
                    continue
                default_weight = nutrition_data[class_name].get('default_weight_g', 100)
                weight_ratio = default_weight / 100

                for nutrient in NUTRIENT_KEYS:
                    total_nutrition[nutrient] += nutrition_data[class_name].get(nutrient, 0) * weight_ratio
                _add_breakdown(class_name, default_weight)

        # 計算飲食建議
        dietary_recommendations = calculate_diet_recommendations(total_nutrition)

        # 準備回應資料
        result_data = {
            'detections': detection_counts,
            'food_breakdown': food_breakdown,
            'portion_items': _build_portion_items(food_breakdown, detection_counts, nutrition_data,
                                                  lambda c: FOOD_NAME_MAP.get(c, c)),
            'boxes': boxes,
            'result_image': f"/static/uploads/{result_filename}",
            'inference_time': f"{inference_time*1000:.2f}ms",
            'total_items': len(detections),
            'low_confidence_count': low_confidence_count,
            'detection_weight': detection_weight,  # 辨識時的重量
            'nutrition': total_nutrition,
            'recommendations': dietary_recommendations
        }
        _attach_advice(result_data, detection_counts, total_nutrition, dietary_recommendations, detection_weight)
        result_data['total_time'] = f"{time.time() - request_start:.1f}s"

        _cleanup_old_uploads()

        return render_template('result.html',
                              result=result_data, 
                              nutrition_data=nutrition_data,
                              name_map=FOOD_NAME_MAP,
                              timestamp=timestamp)

@app.route('/detect100', methods=['POST'])
def detect100():
    """100類UEC-FOOD三餐主食辨識（方案C獨立入口，測試用，尚未跟9類水果/零食偵測器合併）"""
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    detection_weight = float(request.form.get('weight', 0))

    if file:
        request_start = time.time()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = secure_filename(f"100_{timestamp}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result_stem = os.path.splitext(filename)[0]
        result_filename = f"result_{result_stem}.jpg"
        result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)

        view_img = cv2.imread(filepath)
        if view_img is None:
            return jsonify({'error': '無法處理圖像'})
        cv2.imwrite(result_filepath, view_img)  # 乾淨的檢視圖，框線與中文標籤由網頁疊上去

        start_time = time.time()
        detections = detector100.detect(filepath)
        inference_time = time.time() - start_time

        low_confidence_count = sum(1 for d in detections if d['confidence'] < MIN_CONFIDENCE_FOR_NUTRITION_100)
        confident_detections = [d for d in detections if d['confidence'] >= MIN_CONFIDENCE_FOR_NUTRITION_100]
        boxes = _build_boxes(detections, DETECTOR100_IMG_SIZE, DETECTOR100_IMG_SIZE, name_map_100, MIN_CONFIDENCE_FOR_NUTRITION_100)

        detection_counts = {}
        total_nutrition = {'calories': 0, 'protein': 0, 'carbs': 0, 'fiber': 0, 'fat': 0}
        for detection in confident_detections:
            class_name = detection['class_name']
            detection_counts[class_name] = detection_counts.get(class_name, 0) + 1

        def _bbox_area(det):
            x_min, y_min, x_max, y_max = det['bbox']
            return max(0, x_max - x_min) * max(0, y_max - y_min)

        total_area = sum(_bbox_area(d) for d in confident_detections)

        food_breakdown = {}

        def _add_breakdown(class_name, item_weight):
            b = food_breakdown.setdefault(class_name, {'weight_g': 0.0, 'calories': 0.0})
            b['weight_g'] += item_weight
            b['calories'] += nutrition_data_100[class_name].get('calories', 0) * (item_weight / 100)

        if confident_detections and detection_weight > 0:
            for detection in confident_detections:
                class_name = detection['class_name']
                if class_name not in nutrition_data_100:
                    continue
                area = _bbox_area(detection)
                item_weight = (detection_weight * area / total_area) if total_area > 0 else (detection_weight / len(confident_detections))
                weight_ratio = item_weight / 100
                for nutrient in NUTRIENT_KEYS:
                    total_nutrition[nutrient] += nutrition_data_100[class_name].get(nutrient, 0) * weight_ratio
                _add_breakdown(class_name, item_weight)
        else:
            for detection in confident_detections:
                class_name = detection['class_name']
                if class_name not in nutrition_data_100:
                    continue
                default_weight = nutrition_data_100[class_name].get('default_weight_g', 100)
                weight_ratio = default_weight / 100
                for nutrient in NUTRIENT_KEYS:
                    total_nutrition[nutrient] += nutrition_data_100[class_name].get(nutrient, 0) * weight_ratio
                _add_breakdown(class_name, default_weight)

        dietary_recommendations = calculate_diet_recommendations(total_nutrition)

        portion_items = _build_portion_items(food_breakdown, detection_counts, nutrition_data_100,
                                             lambda c: name_map_100.get(c, c))

        # 表格顯示用中文名（食物明細/攝取比例），detection_counts本身仍用英文key對應nutrition_data_100
        detection_counts_display = {name_map_100.get(k, k): v for k, v in detection_counts.items()}
        food_breakdown_display = {name_map_100.get(k, k): v for k, v in food_breakdown.items()}

        result_data = {
            'detections': detection_counts_display,
            'food_breakdown': food_breakdown_display,
            'portion_items': portion_items,
            'boxes': boxes,
            'result_image': f"/static/uploads/{result_filename}",
            'inference_time': f"{inference_time*1000:.2f}ms",
            'total_items': len(confident_detections),
            'low_confidence_count': low_confidence_count,
            'detection_weight': detection_weight,
            'nutrition': total_nutrition,
            'recommendations': dietary_recommendations
        }

        _attach_advice(result_data, detection_counts, total_nutrition, dietary_recommendations, detection_weight,
                       name_map=name_map_100, category_map=category_map_100)
        result_data['total_time'] = f"{time.time() - request_start:.1f}s"

        _cleanup_old_uploads()

        return render_template('result.html',
                              result=result_data,
                              nutrition_data=nutrition_data_100,
                              name_map={},
                              timestamp=timestamp)


@app.route('/api/weight')
def get_weight():
    """獲取當前重量數據的API端點"""
    try:
        print("開始獲取重量數據...")
        weight_data = get_weight_data()
        print(f"獲取到的重量數據: {weight_data}")
        return jsonify({
            'success': True,
            'data': weight_data
        })
    except Exception as e:
        print(f"獲取重量數據時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/nutrition/calculate', methods=['POST'])
def calculate_nutrition_with_weight():
    """根據重量和食物類型計算營養的API端點"""
    try:
        data = request.get_json()
        food_type = data.get('food_type')
        weight = data.get('weight', 0)
        
        if not food_type or food_type not in nutrition_data:
            return jsonify({
                'success': False,
                'error': '無效的食物類型'
            }), 400
        
        # 根據重量計算營養成分（假設nutrition_data中的數據是100g的營養成分）
        base_nutrition = nutrition_data[food_type]
        weight_ratio = weight / 100  # 重量比例
        
        calculated_nutrition = {}
        for nutrient in NUTRIENT_KEYS:
            calculated_nutrition[nutrient] = base_nutrition.get(nutrient, 0) * weight_ratio
        
        # 計算飲食建議
        dietary_recommendations = calculate_diet_recommendations(calculated_nutrition)
        
        return jsonify({
            'success': True,
            'data': {
                'food_type': food_type,
                'weight': weight,
                'nutrition': calculated_nutrition,
                'recommendations': dietary_recommendations
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)