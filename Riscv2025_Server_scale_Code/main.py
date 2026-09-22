from LLMAdvisor import get_llm_advice, _ensure_engine_running
from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import base64
from datetime import datetime
import time
import json

# 導入.py檔案
from Detector import TFLiteDetector
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

NUTRIENT_KEYS = ['calories', 'protein', 'carbs', 'fiber']

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
        
        # 在圖像上繪製檢測結果
        result_img = detector.draw_detections(img, detections)
        
        # 儲存結果圖像
        result_filename = f"result_{filename}"
        result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        cv2.imwrite(result_filepath, result_img)
        
        # 按類別計數檢測結果
        detection_counts = {}
        total_nutrition = {'calories': 0, 'protein': 0, 'carbs': 0, 'fiber': 0}
        
        for detection in detections:
            class_name = detection['class_name']
            if class_name in detection_counts:
                detection_counts[class_name] += 1
            else:
                detection_counts[class_name] = 1
        
        # 每個偵測框的面積（用來按大小分配重量，而不是齊頭式平均）
        def _bbox_area(det):
            x_min, y_min, x_max, y_max = det['bbox']
            return max(0, x_max - x_min) * max(0, y_max - y_min)

        total_area = sum(_bbox_area(d) for d in detections)

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
        
        # 計算飲食建議
        dietary_recommendations = calculate_diet_recommendations(total_nutrition)
        
        # 準備回應資料
        result_data = {
            'detections': detection_counts,
            'result_image': f"/static/uploads/{result_filename}",
            'inference_time': f"{inference_time*1000:.2f}ms",
            'total_items': len(detections),
            'detection_weight': detection_weight,  # 辨識時的重量
            'nutrition': total_nutrition,
            'recommendations': dietary_recommendations
        }
        print("開始呼叫LLM...")
        llm_advice = get_llm_advice(detection_counts, total_nutrition, dietary_recommendations, detection_weight)
        result_data['llm_advice'] = llm_advice
        result_data['total_time'] = f"{time.time() - request_start:.1f}s"

        _cleanup_old_uploads()

        return render_template('result.html',
                              result=result_data, 
                              nutrition_data=nutrition_data,
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