# 共用 UI／營養邏輯整合指南

給 Andes Voyager（`main`）與 Genio 720（`Genio_720`）分支使用。這包內容來自 `Realtek-Matrix` 分支，**只包含與模型無關的部分**：
各板子自己的偵測器（`Detector.py`、模型檔）與 LLM（`LLMAdvisor.py`、prompt、引擎）完全不需要、也不應該被覆蓋。

## 帶來什麼

| 功能 | 說明 |
|---|---|
| 新結果頁 | 米色主題；瀏覽器端畫中文框線（不燒進圖、不顯示信心度，不確定的偵測用灰色虛線＋「？」）；重量面板縮小對齊；營養素達成率進度條修正（舊版沒有寬度，從來沒顯示過） |
| 份量滑桿 | 每種食物 0.25–3 倍，重量、熱量、營養表即時重算，不必重新辨識 |
| 非同步 AI 建議 | 結果頁約 0.6 秒先出，AI 建議由 `/api/advice/<token>` 補上（LLM 幾秒到幾十秒都不再卡整頁） |
| 新營養資料庫 | 9 類全部改為 USDA SR Legacy 實查數值（舊版 Snack、MiniBiscuits 熱量被低估約 3 倍），新增脂肪與 `default_weight_g` |
| 新營養計算 | 有秤重：依偵測框面積比例分配總重量；無秤重：用各食物典型單份重量；信心值低的偵測不列入計算；食物明細與總表用同一套數字，保證對得起來 |
| 首頁 | Bootstrap 本機化（不依賴外網）、上傳前瀏覽器先縮圖到 1280px（手機原圖上傳是最大的隱藏等待）、送出後鎖按鈕防連點、顯示「使用者實際等待」與伺服器時間 |

## 不包含（各分支保留自己的）
`Detector.py`、模型檔、`LLMAdvisor.py`、`WeightSensor.py`、`main.py` 的偵測與 LLM 呼叫。
Realtek 專屬的 `Detector100.py`、`/detect100`、`nutrition_data_100_uecfood.json` 也不在內（首頁預設只顯示單一辨識表單，見下方 `features`）。

## 檔案（皆位於 `Riscv2025_Server_scale_Code/`）
```
food_logic.py                      新增：與模型無關的共用邏輯
templates/index.html               覆蓋
templates/result.html              覆蓋
static/vendor/bootstrap.min.css    新增
static/vendor/bootstrap.bundle.min.js  新增
nutrition_data.json                覆蓋（⚠ 見下）
Calculate.py                       僅加入 fat 每日參考值 65g
```
取用方式（在自己的分支上）：
```
git fetch origin
git checkout origin/shared-ui-nutrition -- Riscv2025_Server_scale_Code/food_logic.py Riscv2025_Server_scale_Code/templates Riscv2025_Server_scale_Code/static/vendor Riscv2025_Server_scale_Code/nutrition_data.json Riscv2025_Server_scale_Code/Calculate.py
```

## ⚠ 一定要一起改 main.py
新的 `nutrition_data.json` 每筆多了 `fat`、`default_weight_g`、`usda_fdc_id`、`usda_description`。
**舊版 `main.py` 用 `for nutrient, amount in nutrition_data[class_name].items()` 迴圈累加所有欄位，遇到字串欄位會直接出錯。**
所以 JSON、`food_logic.py`、模板要跟下面的 `main.py` 修改一起套用，不能只換 JSON。

## main.py 整合步驟

1. import 與初始化
```python
from food_logic import (split_by_confidence, compute_nutrition, build_boxes,
                        build_portion_items, AdviceRegistry, cleanup_old_uploads)
import threading

MIN_CONFIDENCE_FOR_NUTRITION = 0.5   # 偵測器門檻 0.25 太寬，信心值低的不列入營養計算
MAX_KEPT_UPLOADS = 200
advice_registry = AdviceRegistry(50)
llm_lock = threading.Lock()          # 若你的 LLMAdvisor 不是執行緒安全的，非同步端點要自己加鎖
```
`FOOD_NAME_MAP = {'Apple': '蘋果', 'Banana': '香蕉', 'Mango': '芒果', 'Orange': '柳橙', 'Pineapple': '鳳梨', 'Watermelon': '西瓜', 'MiniBiscuits': '小餅乾', 'Snack': '零食', 'noodles': '麵條'}`（頁面顯示中文名用）。

2. `/` 首頁：`return render_template('index.html')` 不需改（沒傳 `features` 就只顯示單一辨識表單）。

3. `/detect` 改成：
```python
img = cv2.imread(filepath)
detections = detector.detect(img)                       # ← 你自己的偵測器，不動

# 存一份乾淨的檢視圖（不再呼叫 draw_detections；一律存 .jpg，避免 .jfif 之類副檔名讓 imwrite 出錯）
result_filename = f"result_{os.path.splitext(filename)[0]}.jpg"
cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], result_filename), img)

confident, low_count = split_by_confidence(detections, MIN_CONFIDENCE_FOR_NUTRITION)
boxes = build_boxes(detections, img.shape[1], img.shape[0], FOOD_NAME_MAP, MIN_CONFIDENCE_FOR_NUTRITION)
res = compute_nutrition(confident, nutrition_data, detection_weight)
counts, total, breakdown = res['detection_counts'], res['total_nutrition'], res['food_breakdown']
recs = calculate_diet_recommendations(total)

result_data = {
    'detections': counts, 'food_breakdown': breakdown,
    'portion_items': build_portion_items(breakdown, counts, nutrition_data, lambda c: FOOD_NAME_MAP.get(c, c)),
    'boxes': boxes,
    'result_image': f"/static/uploads/{result_filename}",
    'inference_time': f"{inference_time*1000:.2f}ms",
    'total_items': len(confident), 'low_confidence_count': low_count,
    'detection_weight': detection_weight, 'nutrition': total, 'recommendations': recs,
}
if counts:   # 非同步：先渲染，建議稍後由 /api/advice 取得
    result_data['advice_token'] = advice_registry.register((counts, total, recs, detection_weight))
else:
    result_data['llm_advice'] = get_llm_advice(counts, total, recs, detection_weight)   # 固定文字，不動用 LLM
result_data['total_time'] = f"{time.time() - request_start:.1f}s"
cleanup_old_uploads(app.config['UPLOAD_FOLDER'], MAX_KEPT_UPLOADS)
return render_template('result.html', result=result_data, nutrition_data=nutrition_data,
                       name_map=FOOD_NAME_MAP, timestamp=timestamp)
```
（`request_start = time.time()` 放在儲存上傳檔案之前。）

4. 新增非同步建議端點（你的 `get_llm_advice(detections, nutrition, recommendations, weight)` 簽名不變）：
```python
@app.route('/api/advice/<token>')
def get_advice(token):
    params = advice_registry.pop(token)
    if params is None:
        return jsonify({'success': False, 'error': '建議已過期，請重新辨識'}), 404
    t0 = time.time()
    with llm_lock:
        advice = get_llm_advice(*params)
    return jsonify({'success': True, 'advice': advice, 'seconds': round(time.time() - t0, 1)})
```

5. 若偵測器回傳的 `bbox` 是**原圖像素座標**（Realtek 的 9 類偵測器就是，`draw_detections` 直接畫在原圖上），`build_boxes` 的 `img_w/img_h` 用 `img.shape[1]/[0]`。若座標是模型輸入空間（例如整張圖拉伸成 512×512），就傳模型輸入邊長。

## 可選：`features` 旗標
`index.html` 預設只顯示單一辨識表單。若某板子有第二組類別集（如 Realtek 的 100 類 `/detect100`），在首頁路由傳 `render_template('index.html', features={'meal100': True})` 才會顯示分頁。

## 模板相容性
`result.html` 對缺少的新變數（`name_map`、`portion_items`、`boxes`）有預設值；沒有 `advice_token` 時直接顯示 `result.llm_advice`（同步模式）。
但不接上 `portion_items` 時份量滑桿區塊會是空的，建議照上面接完整。

## 驗證清單
1. 首頁載入，Bootstrap 樣式正常（離線也要正常）。
2. 上傳一張水果照：結果頁先出現、框線與中文品名對得準、沒有信心度；AI 建議幾秒後出現在下方卡片。
3. 拖動份量滑桿：熱量、營養表、進度條即時變動，「還原份量」可用。
4. 熱量檢查：無秤重時 = 每 100g 熱量 × `default_weight_g`／100（例如蘋果 1 顆：USDA 57 大卡／100g × 150g ≈ 86 大卡；食物明細與營養素表兩邊數字必須一致）。
5. 在結果頁按重新整理應顯示「建議已過期」而不是報錯（token 一次性）。
