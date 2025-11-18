# RIV-FoodAI-Y2

本專案旨在建立一套可在 RISC-V 開發板或桌面伺服器上運行的食品影像偵測與營養估算系統。專案內容包含資料集建立、YOLO 模型訓練、推理伺服器、以及（可選）磅秤重量整合，讓使用者能上傳照片、取得食品辨識結果、並根據實際重量計算營養成分與飲食建議。

## 特色
- 使用輕量化 TFLite 模型（float16）做影像偵測
- 支援上傳圖片後在伺服器上推理並產生標註結果圖
- 若有外接磅秤（CT1 + HX711），可整合重量計算更精準的營養估算
- 提供 REST API 以供前端或行動裝置呼叫

## 硬體／教材設備
- HiFive Unmatched Rev B（或任意能執行 TFLite 的開發板/主機）
- USB 傳輸線（視設備而定）
- MicroSD 卡（若在開發板上使用）
- 網路攝影機（例如 Logitech C270）或上傳的相片檔案
- （可選）CT1 磅秤 + HX711 模組，用於讀取重量

## 專案結構（重點目錄與檔案）
根目錄包含幾個重要子資料夾，說明如下：

- `Riscv2025_Server_Code/` - 推理伺服器（無重量整合）
	- `main.py` - Flask 應用入口（影像上傳、偵測、營養計算）
	- `Detector.py` - TFLite 偵測器封裝
	- `Calculate.py` - 營養計算與飲食建議邏輯
	- `detectv2_float16.tflite` - 預訓練的 TFLite 模型
	- `nutrition_data.json` - 食物對應營養資料（以 100g 為基準）
	- `static/`, `templates/` - 前端靜態檔與模板

- `Riscv2025_Server_scale_Code/` - 推理伺服器（含秤重整合）
	- 與上面相似，另外包含 `WeightSensor.py`，以及在 `main.py` 中處理來自磅秤的重量讀取與 API。

- `Riscv2025_CT1_scale/` - Arduino (或相容) 程式碼（CT1 磅秤）

## 快速開始（在開發機或伺服器上）
以下指令假設您使用 Windows PowerShell 並在專案根目錄執行：

1. 建議建立與啟用虛擬環境（可選）：

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. 安裝 Python 依賴：

```powershell
pip install -r requirements.txt
```

3. 啟動推理伺服器（以含秤重的版本為例）：

```powershell
cd Riscv2025_Server_scale_Code; python main.py
```

伺服器預設監聽在 `http://0.0.0.0:5000`。在本機測試時可開啟瀏覽器並造訪 `http://127.0.0.1:5000`。

## 使用說明
- 首頁（`/`）提供上傳圖片的介面，可同時在表單填入 `weight`（克）數值，或使用實際連接的磅秤讀取。
- 圖像上傳後會執行偵測並在結果頁顯示：偵測項目、結果圖、推理時間、估算營養成分與飲食建議。

### API
- 取得即時磅秤重量：
	- GET `/api/weight` → 回傳 JSON
- 根據食物類型與重量計算營養：
	- POST `/api/nutrition/calculate` (JSON body: `{ "food_type": "apple", "weight": 150 }`)

回傳範例（成功）：

```json
{
	"success": true,
	"data": {
		"food_type": "apple",
		"weight": 150,
		"nutrition": { /* 計算後的營養數值 */ },
		"recommendations": { /* 飲食建議 */ }
	}
}
```

## nutrition_data.json
- 此檔案存放每種食物對應的營養素（以 100g 為基準）。伺服器會根據傳入的重量比例放大或縮小。

## 模型與偵測器
- `detectv2_float16.tflite` 為已量化的 TFLite 模型，`Detector.py` 封裝了解析推理、NMS 與繪圖的流程。

## 磅秤整合（可選）
- `WeightSensor.py` 包含初始化與讀取磅秤的程式（例如透過串列或 GPIO 與 HX711 通訊）。
- 若啟動時找不到實體裝置，後端會記錄錯誤並允許系統在沒有硬體的情況下繼續運作（可手動輸入重量）。

## 開發與貢獻
- 若要新增食物類別，請在 `nutrition_data.json` 新增對應條目，並在訓練資料集上加入該類別再重新訓練模型。
- 建議拉取 issue / PR 的流程與檢查項目：模型正確性、API 相容、相容性測試（含有無磅秤情境）。

## 快速啟動腳本
- `run_server_normal.sh`
```bash
chmod +x run_server_normal.sh
./run_server_normal.sh
```

```bash
chmod +x run_server_scale.sh
./run_server_scale.sh
```