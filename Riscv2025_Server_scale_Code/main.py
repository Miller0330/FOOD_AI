"""
RIV-FoodAI 伺服器入口（多平台）。

  python3 main.py                  # RISC-V / Genio 的舊用法，維持不變（Flask 內建伺服器，port 5000）
  gunicorn ... main:app            # Matrix 的用法（模組載入時暖機）

平台由環境變數 PLATFORM=riscv|genio|matrix 決定；沒設時先用檔案特徵自動偵測，再不行預設 riscv（= 整併前 main 的行為）。
所有與硬體相關的東西（LLM、100 類 NPU 偵測器）都在 hal/ 與 platforms/<name>/ 之下，這個檔案只依賴 hal 的介面。
"""
import os
import json
import time
import threading
from datetime import datetime

import cv2
from flask import Flask, render_template, request, redirect, jsonify
from werkzeug.utils import secure_filename

from hal import get_platform
from Detector import TFLiteDetector
from Calculate import calculate_diet_recommendations
from WeightSensor import initialize_weight_sensor, get_weight_data
from food_logic import (NUTRIENT_KEYS, FOOD_NAME_MAP, split_by_confidence, compute_nutrition, build_boxes,
                        build_portion_items, AdviceRegistry, cleanup_old_uploads,
                        group_candidates, attach_candidates, build_portion_rows)

hw = get_platform()   # 目前平台（只載入被選中的那個平台的程式）

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB最大上傳限制
app.config['MODEL_PATH'] = 'detectv2_float16.tflite'

# 如果上傳資料夾不存在，則創建
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 初始化檢測器（9 類，所有平台共用）
detector = TFLiteDetector(
    model_path=app.config['MODEL_PATH'],
    img_size=640,
    conf_threshold=0.25,
    iou_threshold=0.45
)

# 從JSON檔案載入營養數據（每 100g）
with open('nutrition_data.json', 'r') as file:
    nutrition_data = json.load(file)

MIN_CONFIDENCE_FOR_NUTRITION = 0.5  # 偵測器本身的門檻只有0.25(為了不漏掉東西)，
                                     # 但信心值太低的偵測不該被算進營養總量，容易誤導使用者

# 上傳資料夾最多保留幾個檔案。Matrix 預設 200（長期執行避免吃光儲存空間）；
# Matrix / RISC-V 預設保留最近 200 個檔案；Genio 預設 0 = 不清理。可用環境變數 MAX_KEPT_UPLOADS 覆寫（0 = 不清理）。
MAX_KEPT_UPLOADS = int(os.environ.get('MAX_KEPT_UPLOADS', '200' if hw.name in ('matrix', 'riscv') else '0'))

# ===== 三餐主食（100 類）：只有提供 NPU 偵測器的平台（目前是 Matrix）才會啟用 =====
detector100 = None
nutrition_data_100 = {}
name_map_100 = {}
category_map_100 = {}
if hw.meal100:
    with open(hw.meal100.nutrition_path, 'r', encoding='utf-8') as file:
        _raw100 = json.load(file)
    detector100 = hw.meal100.detector_factory()
    # 偵測結果的 class_name 是英文（訓練時的 UEC 類別名）；依 UEC 編號排序建立
    # 「英文名 -> 營養資料 / 中文名 / 類別」，順序與訓練的 class index 完全一致
    for _key in sorted(_raw100.keys(), key=lambda k: int(k[3:6])):
        _entry = _raw100[_key]
        _english = _entry['english_name'].replace(' ', '_').replace("'", '').replace('-', '_')
        nutrition_data_100[_english] = _entry
        name_map_100[_english] = _key.split('_', 1)[1]          # "UEC003_炒什錦飯" -> "炒什錦飯"
        category_map_100[_english] = _entry.get('category', '')

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

# 由 WSGI 伺服器（gunicorn）載入的平台在模組載入時就暖機；
# 其餘平台維持原本行為，只在 `python3 main.py`（__main__）暖機。
if hw.warmup_at_import:
    print("正在準備 AI 模型，請稍候...")
    hw.llm.warmup()
    print("AI 模型準備完成")

# LLM 建議非同步取得（僅 features['async_advice'] 的平台）：結果頁先渲染，建議由 /api/advice/<token> 補上
advice_registry = AdviceRegistry(50)
pending_registry = AdviceRegistry(30)   # 完全沒有高信心結果時，暫存「請使用者確認」的低信心偵測
MAX_PENDING_CONFIRM = 3                 # 最多請使用者確認幾個
_llm_lock = threading.Lock()   # 不同平台的 LLM 不一定執行緒安全，非同步端點自己序列化


def _attach_advice(result_data, counts, total, recs, weight, name_map=None, category_map=None):
    if hw.features.get('async_advice') and counts:
        result_data['advice_token'] = advice_registry.register((counts, total, recs, weight, name_map, category_map))
        return
    # 同步（整併前的行為）：算完建議才回應；沒偵測到食物時也照舊呼叫
    print("開始呼叫LLM...")
    result_data['llm_advice'] = hw.llm.get_advice(counts, total, recs, weight,
                                                  name_map=name_map, category_map=category_map)


@app.route('/api/advice/<token>')
def get_advice(token):
    params = advice_registry.pop(token)
    if params is None:
        return jsonify({'success': False, 'error': '建議已過期，請重新辨識'}), 404
    counts, total, recs, weight, name_map, category_map = params
    t0 = time.time()
    print("開始呼叫LLM...")
    with _llm_lock:
        advice = hw.llm.get_advice(counts, total, recs, weight, name_map=name_map, category_map=category_map)
    return jsonify({'success': True, 'advice': advice, 'seconds': round(time.time() - t0, 1)})


def _process_upload(mode, file, detection_weight):
    """/detect 與 /detect100 共用的流程；兩者只差在偵測器、營養資料與顯示名稱。"""
    request_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = secure_filename(f"{mode['prefix']}{timestamp}_{file.filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    img = cv2.imread(filepath)
    if img is None:
        return jsonify({'error': '無法處理圖像'})

    # 存一份乾淨的檢視用圖（不燒框線與文字，框線與中文標籤由網頁疊上去）。一律存成.jpg，
    # 不要照抄原始上傳檔名的副檔名——手機/Windows常見的.jfif，cv2.imwrite不認得會直接噴錯
    result_filename = f"result_{os.path.splitext(filename)[0]}.jpg"
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], result_filename), img)

    start_time = time.time()
    detections = mode['detect'](img, filepath)
    inference_time = time.time() - start_time

    # 框住同一份食物但類別不同的偵測（例如義大利麵／肉醬義大利麵）歸成一組：預設用信心最高的，
    # 結果頁讓使用者自己換選；同類別的重疊框只算一次
    groups = group_candidates(detections, allowed=set(mode['nutrition']))
    ctx = {'mode_id': mode['id'], 'groups': groups, 'img_shape': img.shape, 'result_filename': result_filename,
           'inference_time': inference_time, 'detection_weight': detection_weight, 'timestamp': timestamp}
    return _render_result(mode, ctx, request_start)


def _render_result(mode, ctx, request_start, chosen=None):
    """
    由偵測分組算出結果頁。chosen=None 是第一次顯示；沒有任何高信心結果、但有低信心偵測時，
    不直接說「未檢測到」，改請使用者勾選哪些是對的（POST /confirm 後帶著 chosen 再算一次）。
    """
    groups, result_filename = ctx['groups'], ctx['result_filename']
    detection_weight, inference_time, timestamp = ctx['detection_weight'], ctx['inference_time'], ctx['timestamp']
    primaries = [g[0] for g in groups]

    # 信心值太低的偵測不列入營養計算（圖上以虛線框標成不確定），避免誤導使用者
    confident, low_confidence_count = split_by_confidence(primaries, mode['min_conf'])
    pending = []
    if chosen is not None:
        confident = confident + [primaries[i] for i in chosen]
        low_confidence_count = len(primaries) - len(confident)
    elif not confident:
        idx = [i for i, d in enumerate(primaries) if d['class_name'] in mode['nutrition']]
        idx = sorted(idx, key=lambda i: -primaries[i]['confidence'])[:MAX_PENDING_CONFIRM]
        pending = [{'idx': i, 'name': mode['name_of'](primaries[i]['class_name']),
                    'confidence': round(primaries[i]['confidence'] * 100)} for i in idx]
    box_w, box_h = mode['coord_size'](ctx['img_shape'])
    boxes = build_boxes(primaries, box_w, box_h, mode['label_map'], mode['min_conf'])
    for i in (chosen or []):
        boxes[i]['counted'] = True
    for n, p in enumerate(pending, 1):
        boxes[p['idx']]['pending_no'] = n
        p['no'] = n

    res = compute_nutrition(confident, mode['nutrition'], detection_weight)
    counts, total = res['detection_counts'], res['total_nutrition']
    recs = calculate_diet_recommendations(total)

    attach_candidates(res['items'], [g for g in groups if g[0] in confident], mode['nutrition'], mode['name_of'], detection_weight)
    portion_rows = build_portion_rows(res['items'], mode['nutrition'], mode['name_of'])
    gid_of = {id(it['det']): it['gid'] for it in res['items'] if 'gid' in it}
    for b, d in zip(boxes, primaries):
        if id(d) in gid_of:
            b['gid'] = gid_of[id(d)]
    plain_counts = {}
    for it in res['items']:
        if 'gid' not in it:
            plain_counts[it['class']] = plain_counts.get(it['class'], 0) + 1

    display = mode['display_map']   # 100 類：英文 key 換成中文顯示；9 類維持原 key（模板依英文 key 挑圖示）
    result_data = {
        'mode': mode['id'],
        'detections': {display.get(k, k): v for k, v in plain_counts.items()},
        'choices': [r for r in portion_rows if r.get('cands')],
        'portion_items': portion_rows,
        'boxes': boxes,
        'result_image': f"/static/uploads/{result_filename}",
        'inference_time': f"{inference_time*1000:.2f}ms",
        'total_items': len(confident),
        'low_confidence_count': low_confidence_count,
        'pending': pending,
        'pending_token': pending_registry.register(ctx) if pending else None,
        'detection_weight': detection_weight,  # 辨識時的重量
        'nutrition': total,
        'recommendations': recs,
    }
    _attach_advice(result_data, counts, total, recs, detection_weight,
                   name_map=mode['advice_name_map'], category_map=mode['advice_category_map'])
    result_data['total_time'] = f"{time.time() - request_start:.1f}s"

    if MAX_KEPT_UPLOADS > 0:
        cleanup_old_uploads(app.config['UPLOAD_FOLDER'], MAX_KEPT_UPLOADS)

    return render_template('result.html', result=result_data, nutrition_data=mode['nutrition'],
                           name_map=mode['template_name_map'], timestamp=timestamp)


_MODE_9 = {
    'id': '9',
    'prefix': '',
    'detect': lambda img, path: detector.detect(img),
    'min_conf': MIN_CONFIDENCE_FOR_NUTRITION,
    'coord_size': lambda img: (img[1], img[0]),   # 9 類偵測器回傳原圖像素座標
    'label_map': FOOD_NAME_MAP,
    'nutrition': nutrition_data,
    'display_map': {},
    'name_of': lambda c: FOOD_NAME_MAP.get(c, c),
    'advice_name_map': None,
    'advice_category_map': None,
    'template_name_map': FOOD_NAME_MAP,
}


@app.route('/')
def index():
    return render_template('index.html', features=hw.features)


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
        return _process_upload(_MODE_9, file, detection_weight)


if hw.meal100:
    _MODE_100 = {
        'id': '100',
        'prefix': '100_',
        'detect': lambda img, path: detector100.detect(path),
        'min_conf': hw.meal100.min_confidence,
        # matrixrt_yolo 的座標是模型輸入空間（整張圖拉伸成 img_size 正方形），除以 img_size 即為對整張原圖的比例
        'coord_size': lambda img: (hw.meal100.img_size, hw.meal100.img_size),
        'label_map': name_map_100,
        'nutrition': nutrition_data_100,
        'display_map': name_map_100,
        'name_of': lambda c: name_map_100.get(c, c),
        'advice_name_map': name_map_100,
        'advice_category_map': category_map_100,
        'template_name_map': {},
    }

    @app.route('/detect100', methods=['POST'])
    def detect100():
        """100類UEC-FOOD三餐主食辨識（獨立入口，測試用，尚未跟9類水果/零食偵測器合併）"""
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        detection_weight = float(request.form.get('weight', 0))
        if file:
            return _process_upload(_MODE_100, file, detection_weight)


_MODES = {'9': _MODE_9}
if hw.meal100:
    _MODES['100'] = _MODE_100


@app.route('/confirm', methods=['POST'])
def confirm_detections():
    """「未檢測到食物」時，使用者勾選哪些低信心偵測是對的，再依勾選結果顯示完整結果頁。"""
    ctx = pending_registry.get(request.form.get('token', ''))
    mode = _MODES.get(ctx['mode_id']) if ctx else None
    if ctx is None or mode is None:
        return render_template('index.html', features=hw.features), 410
    n = len(ctx['groups'])
    chosen = sorted({int(v) for v in request.form.getlist('pick') if v.isdigit() and int(v) < n})
    return _render_result(mode, ctx, time.time(), chosen)


@app.route('/api/advice/regen', methods=['POST'])
def regen_advice():
    """使用者在結果頁換選食物／調整份量後，依目前的選擇重新產生 AI 建議（回傳新的 token，再走 /api/advice/<token>）。"""
    data = request.get_json(silent=True) or {}
    mode = _MODES.get(data.get('mode'))
    if mode is None or not hw.features.get('async_advice'):
        return jsonify({'success': False, 'error': '無法重新產生建議'}), 400
    nutrition = mode['nutrition']
    counts, total = {}, {k: 0.0 for k in NUTRIENT_KEYS}
    try:
        for p in list(data.get('picks') or [])[:40]:
            cn = p.get('class')
            if cn not in nutrition:
                continue
            weight_g = max(0.0, min(float(p.get('weight_g', 0)), 5000.0))
            counts[cn] = counts.get(cn, 0) + max(1, min(int(p.get('count', 1)), 20))
            for k in NUTRIENT_KEYS:
                total[k] += nutrition[cn].get(k, 0) * weight_g / 100
        detection_weight = float(data.get('weight', 0))
    except (TypeError, ValueError):
        return jsonify({'success': False, 'error': '資料格式錯誤'}), 400
    if not counts:
        return jsonify({'success': False, 'error': '沒有可分析的食物'}), 400
    token = advice_registry.register((counts, total, calculate_diet_recommendations(total), detection_weight,
                                      mode['advice_name_map'], mode['advice_category_map']))
    return jsonify({'success': True, 'token': token})


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
    if not hw.warmup_at_import:
        print("正在準備 AI 模型，請稍候...")
        hw.llm.warmup()
        print("AI 模型準備完成")
    app.run(host='0.0.0.0', port=5000, debug=False)
