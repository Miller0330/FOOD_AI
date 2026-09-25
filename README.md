# RIV-FoodAI — Realtek Matrix 版（分支 `Realtek-Matrix`）

在 Realtek Matrix 開發板（RTD2811，2 核 A73 + NPU）上執行的食物辨識與營養分析系統：
拍照 → 偵測食物 → （搭配電子秤重量）估算熱量／營養 → 由板端 LLM 產生飲食建議。

- **9 類水果／零食**：CPU 上跑 torch（`/detect`）
- **100 類三餐主食（UEC FOOD 100）**：NPU 上跑 YOLO11s / 512px（`/detect100`），單張約 11ms
- **AI 營養師**：Gemma4-E4B 跑在 NPU，結果頁先出、建議非同步補上

> 詳細的部署與已知限制見 [`MATRIX_STATUS.md`](MATRIX_STATUS.md)；把 UI 帶到其他板子見 [`Riscv2025_Server_scale_Code/SHARED_UI_INTEGRATION.md`](Riscv2025_Server_scale_Code/SHARED_UI_INTEGRATION.md)。

---

## 1. 目前進度

| 項目 | 狀態 |
|---|---|
| 100 類模型 GPU 訓練（YOLO11s、512px、80 epochs） | 完成，mAP50 **0.782** |
| ONNX → NPU `.nb` 轉換（校正圖 512px、200 張） | 完成，已上板驗證 |
| `/detect100` 使用新模型 | 完成（`IMG_SIZE = 512`） |
| 板端常駐服務（systemd + gunicorn 單 worker） | 完成，開機自動啟動 |
| 非同步 AI 建議、份量滑桿、瀏覽器端畫框 | 完成 |
| 9 類偵測改跑 NPU（去掉 torch） | 未做 |
| 9 類與 100 類偵測是否合併 | 未做，等實測後決定 |
| 每日累計與個人化營養目標 | 未做 |
| CT1 磅秤 USB 供電問題 | 未解，需用有供電的 hub 實機測 |

---

## 2. 架構與目錄

```
 手機/相機照片 ─┐
                ├─► Flask (gunicorn, :5000) ─► 偵測 ─► 營養計算 ─► 結果頁
 電子秤(CT1) ───┘        │                     ├─ /detect     9 類（torch, CPU）
   USB 串列              │                     └─ /detect100  100 類（matrixrt_yolo, NPU）
                         └─► /api/advice/<token> ─► LLM（Gemma4-E4B, NPU）非同步回傳建議
```

| 路徑 | 說明 |
|---|---|
| `Riscv2025_Server_scale_Code/` | **主要服務**（見下） |
| `Riscv2025_Server_Code/` | 早期基礎版（僅 TFLite 9 類，無秤、無 LLM），保留備查 |
| `Riscv2025_CT1_scale_Code/` | 電子秤 Arduino：`ct1_scale`（HX711 + TM1637）、`ct1_scale_calibration`（校正）、AICE 驅動 |
| `training/uecfood_gpu/` | 100 類 GPU 訓練腳本與 `TASK.md` |
| `training/eval/` | NPU 對浮點模型的比較評估腳本 |
| `MATRIX_STATUS.md` | 板端部署細節、路徑、限制 |

`Riscv2025_Server_scale_Code/` 內：
- `main.py`：路由 `/`、`/detect`、`/detect100`、`/api/weight`、`/api/nutrition/calculate`、`/api/advice/<token>`
- `Detector.py`：9 類偵測（Apple、Banana、Mango、MiniBiscuits、Orange、Pineapple、Snack、Watermelon、noodles）
- `Detector100.py`：以 subprocess 呼叫 `matrixrt_yolo`，每次請求啟動一次（含載入約 0.3 秒，不需常駐）
- `food_logic.py`：與模型無關的共用邏輯（營養計算、框資料、份量滑桿、非同步建議暫存、上傳清理）
- `LLMAdvisor.py`：常駐 LLM 引擎、prompt 與 fallback 建議
- `WeightSensor.py`：讀秤（預設 `/dev/ttyUSB1`、115200）
- `nutrition_data.json`（9 類，USDA 實查）、`nutrition_data_100_uecfood.json`（100 類，多為估計值）
- `llama-cli` / `llama-cli-new`：riscv64 版 llama.cpp（見第 6 節）

---

## 3. 板端部署與啟用

### 3.1 連線
板子以 USB-C 接電腦後是 NCM 網路：板子 `192.100.100.2`、電腦 `192.100.100.1`，SSH 用金鑰登入 root。網頁位址 `http://192.100.100.2:5000`。

### 3.2 服務
服務名稱 `riv-foodai.service`（systemd），gunicorn **只能單一 worker**（NPU 與 LLM 引擎各只有一份，加 worker 會搶資源）。

更新程式的流程：
```bash
# 電腦端：把檔案 scp 到板子的 /data/RIV-FoodAI-Backup/Riscv2025_Server_scale_Code/
# 板子端：
find /data/RIV-FoodAI-Backup -name __pycache__ -exec rm -rf {} +   # 板子沒有 RTC，時間每次開機重置，會誤用舊 pyc
systemctl restart riv-foodai.service
journalctl -u riv-foodai.service                                   # 看紀錄
```
重啟後要等 **約 2–3 分鐘暖機**（torch 載入 + LLM 引擎載入）再測試。

### 3.3 板子上需要的檔案（不在 repo 內）
| 項目 | 路徑 |
|---|---|
| 100 類模型（使用中） | `/data/models/uecfood_s512/network_binary.nb`（YOLO11s、512px） |
| 100 類舊版（備份，`labels.txt` 仍被共用） | `/data/models/uecfood_full100/`（YOLO11n、416px） |
| LLM | `/data/models/gemma4_e4b/`（`packed_engine_e4b_fixed_strip_llmonly.json`） |
| 已 patch 的 YOLO 執行檔 | `/opt/matrix/bin/matrixrt_yolo`（備份在 `/data/matrixrt_yolo_patched/`，開機由 `restore-matrixrt-yolo.service` 還原） |
| Python 套件 | `/data/pylibs`（服務用 `PYTHONPATH` 指向） |

> ⚠ `/root`、`/opt`、`/home`、`/usr/local`、`/var` 會在 OTA 或重置時被清掉，**模型與程式一律放 `/data`**。
> ⚠ repo 內的 `requirements.txt` 只列 Flask 等基本套件；板端實際還需要 gunicorn、torch、opencv 等（放在 `/data/pylibs`），systemd unit 檔也不在 repo 內。

`matrixrt_yolo` 已改為可指定 `--imgsz --nc --cores`：
- 100 類新模型：`--imgsz 512 --nc 100 --cores 2`
- 舊版 11n 模型：`--imgsz 416`

### 3.4 API
- `POST /detect`、`POST /detect100`：表單上傳 `file`，可帶 `weight`（克）
- `GET /api/weight`：目前秤重
- `POST /api/nutrition/calculate`：`{ "food_type": "apple", "weight": 150 }`
- `GET /api/advice/<token>`：取得非同步的 LLM 建議

---

## 4. 100 類模型：訓練、轉換與成果

### 4.1 訓練（電腦端，RTX 4060 Laptop 8GB）
```powershell
python -m venv train_env
train_env\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics pillow
python -c "import torch; print(torch.cuda.is_available())"   # 必須 True
cd training\uecfood_gpu
python setup_dataset.py      # 下載 UEC FOOD 100（約 945MB）→ yolo_full100/
python run_train_gpu.py      # 訓練 + 匯出 ONNX（opset 12）
```
中斷後續訓：`YOLO("runs/detect/runs/uecfood_gpu_s512/weights/last.pt").train(resume=True)`。

實測：14,361 張影像、100 類、驗證集 2,109 張；batch 32 佔 5.3GB VRAM；約 2.6 分鐘 / epoch，共 3.4 小時。ONNX 輸出 `(1, 104, 5376)`（100 類 + 4 座標，512px 對應 5376 個 anchor）。

### 4.2 結果

| | 第一版 | 現行版 |
|---|---|---|
| 模型 / 解析度 | YOLO11n / 416 | YOLO11s / 512 |
| mAP50 | 0.683 | **0.782** |
| mAP50-95 | 0.528 | **0.610** |
| Precision / Recall | — | 0.741 / 0.715 |
| 訓練資料 | 每類最多 200 張 | 全部影像 |
| 訓練耗時 | 10.6 小時（CPU） | 3.4 小時（GPU） |

- 數字為 best.pt；最後一個 epoch 為 mAP50 0.760 / mAP50-95 0.594。
- **兩版驗證集不同**（第一版有每類上限），只能大致比較。
- 最強（mAP50 ≥ 0.98）：bibimbap、tempura_udon、eels_on_rice、tensin_noodle。
- 最弱：fried_fish 0.27、chip_butty 0.40、ginger_pork_saute 0.47、rice 0.48（recall 僅 0.13）、steamed_meat_dumpling 0.49、green_salad 0.51（recall 0.25）、miso_soup 0.55（recall 0.36）。

### 4.3 轉成 NPU（VirtualBox `Andes_Qilai_Ubuntu`）
1. `importer.py`
2. `quantize_hybrid_yolo.py <dir> asymu8 yolov11n --iterations 200`（校正圖每類 2 張共 200 張、與輸入同解析度）
3. `export_nbg_setup.py <dir> asymu8 --use_hybrid`

匯出與校正圖**一律 512px**。板端實測（2,109 張驗證圖）NPU 與浮點差距很小：信心值 0.3 時召回 72.0% vs 73.2%。評估腳本在 `training/eval/`。

---

## 5. 營養計算規則
- 信心值門檻：偵測器 `conf` 0.15（100 類）／0.25（9 類）；**只有** ≥ `MIN_CONFIDENCE_FOR_NUTRITION_100`（0.3）／`MIN_CONFIDENCE_FOR_NUTRITION`（0.5）的偵測才列入營養計算，較低的會在圖上標成不確定（灰色虛線 + 「？」）。
- 有秤重：總重量依各偵測框**面積比例**分配；沒秤重：每個偵測用 `default_weight_g`（典型單份）。這是估算，不是逐項秤重。
- 結果頁的份量滑桿（0.25–3 倍）即時重算重量、熱量與營養表，不必重新辨識。
- 每筆營養資料以 100g 為基準；`nutrition_data.json` 每筆含 `fat`、`default_weight_g`、`usda_fdc_id`、`usda_description`，**程式不可假設所有欄位都是數值**（見 `SHARED_UI_INTEGRATION.md`）。

---

## 6. 注意事項

1. **單 worker**：gunicorn 不可增加 worker；NPU 與 LLM 引擎只有一份。
2. **暖機**：重啟服務後需 2–3 分鐘；LLM 產生建議約 20 秒，所以建議是非同步載入。
3. **清 `__pycache__`**：板子沒有 RTC，覆蓋檔案後務必清掉再重啟。
4. **類別順序**：`labels.txt` 必須與訓練 `data.yaml`（`training/eval/labels_100.txt`）一致；改類別名稱時，`nutrition_data_100_uecfood.json` 與 `name_map_100` 也要同步。
5. **100 類營養資料多為估計值**（USDA 對應不到的整道菜）；發現不合理請直接修 `nutrition_data_100_uecfood.json`。
6. **弱類別不可靠**：rice、green_salad、miso_soup recall 很低（湯與白飯常佔滿整張圖、標註範圍不一致），這幾類的熱量結果請視為參考。
7. **`training/eval/run_board_eval.sh` 仍寫舊的 416 / `uecfood_full100` 模型**；評估新模型時要改成 `--imgsz 512` 與 `/data/models/uecfood_s512/network_binary.nb`。
8. **LLM 限制**：引擎無對話記憶；`-p` 系統提示不生效，格式限制要寫在使用者問題裡；板子可用記憶體僅約 1.5GB，換更大模型前先確認 Realtek 工具鏈支援。對話版原型（port 5001）已擱置，且不可與主服務同時執行。
9. **llama-cli 只適用 riscv64**：`llama-cli`（build 6815）保留對照，`llama-cli-new`（build 6817）prompt eval 4→30 token/秒，`LLMAdvisor.py` 使用後者。移植到 ARM（如 Genio 720）需另取對應版本。
10. **電子秤**：預設 `/dev/ttyUSB1`；更換秤重元件需用 `ct1_scale_calibration.ino`（基準重量 `sample_weight = 1500` g）重新校正，再把比例因子填入 `ct1_scale.ino`。USB 供電問題尚未解。
11. **資安**：`run_server_normal.sh` 內有硬編碼使用者密碼、`run_server_scale.sh` 註解寫有 root 密碼，只適用內網開發板，正式部署前請移除；服務不要暴露到公網。
12. **上傳**：瀏覽器先把照片縮到最長邊 1280px；伺服器上限 16MB，`static/uploads/` 只保留最近 200 個檔案。
13. **資料授權**：UEC FOOD 100 為非商業研究用途，商用前需確認授權。
14. **不要進版控**：`train_env/`、`UECFOOD100/`、`yolo_full100/`、`dataset100.zip`（約 1GB）、`runs/`、`*.pt`、`*.onnx`、`.nb` 等大檔。

---

## 7. 相關分支
- `main`：Qilai EVB　`Genio_720`：MediaTek Genio 720　`shared-ui-nutrition`：只含與模型無關的共用 UI／營養邏輯

## 8. 待辦
- [ ] 9 類偵測改跑 NPU（省記憶體與啟動時間）
- [ ] 評估兩個偵測器是否合併為單一入口
- [ ] 針對弱類別補資料或調整標註後重訓
- [ ] 每日累計與個人化營養目標
- [ ] 解決 CT1 磅秤 USB 供電問題
