from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
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
        
        # 根據實際重量計算營養成分
        if detection_counts and detection_weight > 0:
            # 計算每個檢測到的食物項目的平均重量
            total_items = sum(detection_counts.values())
            average_weight_per_item = detection_weight / total_items
            
            print(f"總重量: {detection_weight}g, 總項目數: {total_items}, 平均每項重量: {average_weight_per_item:.1f}g")
            
            for class_name, count in detection_counts.items():
                if class_name in nutrition_data:
                    # 根據實際重量計算營養成分（假設nutrition_data中的數據是100g的營養成分）
                    item_weight = average_weight_per_item * count
                    weight_ratio = item_weight / 100  # 重量比例
                    
                    for nutrient, amount in nutrition_data[class_name].items():
                        total_nutrition[nutrient] += amount * weight_ratio
        else:
            # 如果沒有重量數據，使用預設計算方式
            for detection in detections:
                class_name = detection['class_name']
                if class_name in nutrition_data:
                    for nutrient, amount in nutrition_data[class_name].items():
                        total_nutrition[nutrient] += amount
        
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
        for nutrient, value in base_nutrition.items():
            calculated_nutrition[nutrient] = value * weight_ratio
        
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