> **整併後的路徑說明**：本檔是 Matrix 平台專屬說明。專案已改為多平台 HAL 結構（見根目錄 README 與 `docs/HAL_ARCHITECTURE.md`）：
> `Detector100.py` → `Riscv2025_Server_scale_Code/platforms/matrix/detector100.py`、Matrix 的 `LLMAdvisor.py` → `platforms/matrix/llm_advisor.py`、
> `nutrition_data_100_uecfood.json` → `platforms/matrix/`、部署腳本與服務檔 → `deploy/matrix/`（服務檔已加 `Environment=PLATFORM=matrix`）。
> 下文若仍寫舊路徑，請照上述對應。

# RIV-FoodAI｜Realtek Matrix 版本

在 **Realtek Matrix（RTD2811）開發板**上完全離線執行的「食物拍照辨識＋營養估算＋AI 營養師建議」系統：
偵測用 NPU 跑 YOLO，AI 建議用 NPU 跑 Gemma4 E4B，網頁介面用 Flask 提供，全部都在板子上完成，不需要外網。

> 本分支（`Realtek-Matrix`）針對 Matrix 板子。Qilai EVB（`main`）與 MediaTek Genio 720（`Genio_720`）是各自的分支，
> 使用不同的偵測與 LLM 模型。與模型無關的新 UI／營養功能另外整理在 `shared-ui-nutrition` 分支，見文末〈與其他分支的關係〉。
> 目前進度摘要另見 [`MATRIX_STATUS.md`](MATRIX_STATUS.md)。

---

## 目錄
1. [功能總覽](#1-功能總覽)
2. [系統架構與 API](#2-系統架構與-api)
3. [硬體規格與效能數據](#3-硬體規格與效能數據)
4. [模型與準確度數據](#4-模型與準確度數據)
5. [專案結構](#5-專案結構)
6. [連線到板子](#6-連線到板子)
7. [從零啟用（部署）](#7-從零啟用部署)
8. [日常操作](#8-日常操作)
9. [設定與門檻一覽](#9-設定與門檻一覽)
10. [資料：營養資料庫與訓練資料](#10-資料營養資料庫與訓練資料)
11. [模型訓練與轉換流程](#11-模型訓練與轉換流程)
12. [注意事項（務必先讀）](#12-注意事項務必先讀)
13. [疑難排解](#13-疑難排解)
14. [已知限制與待辦](#14-已知限制與待辦)
15. [與其他分支的關係](#15-與其他分支的關係)
16. [授權與資料來源](#16-授權與資料來源)

---

## 1. 功能總覽

- **兩組食物類別，首頁分頁選擇**（彼此獨立，尚未合併）
  - 🍎 **水果／零食（9 類）**：蘋果、香蕉、芒果、柳橙、鳳梨、西瓜、小餅乾、零食、麵條。路由 `/detect`，偵測器為 TorchScript 模型（CPU）。
  - 🍚 **三餐主食（100 類，測試版）**：UEC FOOD 100（飯、麵、麵包、炸物、魚、蛋、湯、沙拉等日式家常菜為主）。路由 `/detect100`，偵測器為 YOLO11s／512px，跑在 NPU 上。
- **營養估算**：有磅秤重量時依偵測框面積比例分配重量；沒有時用每種食物的典型單份重量；信心值低的偵測不列入計算。輸出熱量、蛋白質、碳水、纖維、脂肪與每日參考值達成率。
- **份量滑桿**：結果頁每種食物可調 0.25–3 倍，重量、熱量與營養表即時重算，不必重新辨識。
- **AI 營養師建議**：Gemma4 E4B（NPU）依實際營養數值與缺少的營養素，用口語給 3–4 句建議與具體可補的食物。結果頁先顯示，建議約 20 秒後非同步補上。
- **結果頁標註**：框線與中文品名由瀏覽器疊在乾淨的原圖上，不顯示信心度（收在「偵測明細」）；不確定的偵測用灰色虛線框、名稱後加「？」，且不計入營養。
- **體感優化**：Bootstrap 本機化（板子無外網）、上傳前瀏覽器先縮圖到 1280px、送出後鎖按鈕、顯示「伺服器運算時間」與「使用者實際等待」兩種時間。
- **可選磅秤**：CT1 + HX711，經 `/dev/ttyUSB1`（未接上時系統照常運作，可手動忽略重量）。

## 2. 系統架構與 API

```
瀏覽器（縮圖 1280px）
   │  POST /detect 或 /detect100 (file, weight)
   ▼
Flask + gunicorn（單一 worker）
   ├─ 偵測：9 類  → Detector.py（torch，CPU）
   │        100 類 → Detector100.py → 呼叫 /opt/matrix/bin/matrixrt_yolo（NPU，每次請求啟動一次，約 0.3s）
   ├─ 營養計算：food_logic.py（與模型無關）＋ nutrition_data*.json
   ├─ 結果頁：先渲染（約 0.6s），AI 建議稍後由頁面向 /api/advice/<token> 取得
   └─ AI 建議：LLMAdvisor.py → 常駐子程序 MatriXRT-llm-cli（Gemma4 E4B，NPU）
```

| 路由 | 方法 | 說明 |
|---|---|---|
| `/` | GET | 首頁（上傳表單、兩個分頁） |
| `/detect` | POST | 9 類辨識。表單欄位 `file`（圖片）、`weight`（克，0 表示無秤重） |
| `/detect100` | POST | 100 類辨識，欄位同上 |
| `/api/advice/<token>` | GET | 取得非同步 AI 建議；token 一次性，重新整理結果頁會顯示「建議已過期」 |
| `/api/weight` | GET | 讀取磅秤 |
| `/api/nutrition/calculate` | POST | JSON `{"food_type": "Apple", "weight": 150}` 回傳營養 |

## 3. 硬體規格與效能數據

**板子**：RTD2811，2× Cortex-A73，RAM 約 3.9GB，NPU(20TOPS)，Ubuntu 24.04 aarch64。
儲存：`/dev/root` 3GB（要保持很空）、`/data` 約 21GB（放程式與模型）。

| 項目 | 數值 |
|---|---|
| YOLO11n 416px NPU 推論 | 約 6 ms |
| YOLO11s 512px NPU 推論（現行） | 約 11 ms |
| 100 類單次呼叫（含載入）| 約 0.32 s |
| 結果頁出現 | 約 0.6 s（100 類）／約 2.7 s（9 類，CPU torch） |
| AI 建議生成 | 約 19–24 s（約 120–150 字；舊版短提示約 9 s） |
| 服務冷啟動到可用 | 約 2–3 分鐘（torch 載入＋LLM 引擎載入） |
| Gemma4 E4B 引擎常駐記憶體 | 約 575 MB（權重邊跑邊串流，不整包載入 RAM） |
| Flask＋torch 常駐記憶體 | 約 380 MB |
| 系統整體可動用記憶體 | 約 1 GB 上下（含可回收快取） |

**記憶體是目前最緊的資源**，換更大的 LLM 前必須先確認 Realtek 工具鏈是否支援該模型，並先為 9 類偵測器去掉 torch 依賴。

## 4. 模型與準確度數據

### 100 類偵測模型（UEC FOOD 100）

| | 第一版 | 現行版（已上線） |
|---|---|---|
| 模型 | YOLO11n | YOLO11s |
| 解析度／回合 | 416px／25 epochs | 512px／80 epochs |
| 訓練設備 | CPU 約 10.6 小時 | RTX 4060 筆電 GPU |
| 每類影像上限 | 200 張 | 無上限（共 14,361 張） |
| mAP50／mAP50-95 | 0.683／0.528 | **0.782／0.610** |
| Precision／Recall | 0.622／0.661 | 0.741／0.715 |
| 板端 NPU 召回（信心 0.3） | 約 42%（浮點 66%） | **72.0%**（浮點 73.2%） |

現行版的板端實測（2,109 張驗證圖，浮點 vs NPU，信心值 0.3）：有偵測到的圖片 91.6% vs 91.0%、召回 72.0% vs 73.2%、精確度 66.0% vs 70.9%。量化損失已縮小到約 1 個百分點。
較弱的類別：炸魚、豆腐蔬菜濃湯、薯條三明治、薑燒豬肉、白飯、牛排、生菜沙拉；很穩的：鰻魚飯、天津炒麵、大阪燒、拌飯、天婦羅烏龍麵。
各類別完整表格（品名、訓練張數、各項指標）見專案簡報／報告；評估腳本在 [`training/eval/`](training/eval/)。

### 9 類偵測模型
TorchScript（`detectv2_torch.pt`），訓練用的原始影像已遺失（只剩權重），因此無法重訓，也無法與 100 類直接合併訓練。

### LLM
Gemma4 **E4B**（約 4B，q6k 量化）為現行；板上另有 **E2B**（約 2B）備用，實測 E2B 反而較慢（約 12s vs 8s，短提示），速度不隨參數量線性變化，取決於 Realtek 對該尺寸的打包優化。

## 5. 專案結構

```
README.md                         本檔
MATRIX_STATUS.md                  進度小總結
Riscv2025_Server_scale_Code/      ← 板子上實際執行的程式（含秤重整合）
  main.py                         Flask 路由（/detect、/detect100、/api/*）
  food_logic.py                   與模型無關的共用邏輯（營養計算、框線資料、份量、非同步建議暫存）
  Detector.py                     9 類偵測器（TorchScript，CPU）
  Detector100.py                  100 類偵測器（呼叫 matrixrt_yolo，NPU）
  LLMAdvisor.py                   AI 建議（常駐 MatriXRT-llm-cli 子程序）＋ prompt
  Calculate.py                    每日參考值與達成率
  WeightSensor.py                 磅秤序列埠讀取
  nutrition_data.json             9 類營養資料（USDA）
  nutrition_data_100_uecfood.json 100 類營養資料
  templates/ static/vendor/       前端（結果頁疊圖、份量滑桿）與本機化 Bootstrap
  SHARED_UI_INTEGRATION.md        給其他板子分支的整合指南
  detectv2_torch.pt               9 類模型權重
models/uecfood_s512/             100 類 NPU 模型 network_binary.nb（約 15MB，YOLO11s／512px）＋ labels.txt
deploy/                           一鍵部署腳本 deploy_to_board.sh 與 systemd 服務檔
training/
  uecfood_gpu/                    GPU 重訓腳本與任務說明（TASK.md）
  eval/                           NPU vs 浮點的驗證集評估腳本
Riscv2025_CT1_scale_Code/         磅秤（CT1 + HX711）韌體
Riscv2025_Server_Code/            原始不含秤重的舊版伺服器（沿用，非 Matrix 使用）
```

**訓練成果已進 git**：[`training/uecfood_gpu/results/`](training/uecfood_gpu/results/) 有 `best.pt`（19MB）、`best.onnx`（38MB）、`data.yaml`、每類指標、訓練曲線、混淆矩陣與完整 log，可直接拿去轉 NPU，不必重訓。

**100 類 NPU 模型也已進 git**（`models/uecfood_s512/`），拉下來就能部署。
**仍不在 git 裡的檔案**（屬於板子環境或 vendor，見 [7.1](#71-需要準備的檔案)）：Gemma4 LLM 引擎（數 GB）、`matrixrt_yolo` 修改版執行檔、Python 套件（`/data/pylibs`）、訓練資料集。
`Riscv2025_Server_scale_Code/` 內的 `llama-cli`、`llama-cli-new`、`debug_llm.py` 是 RISC-V 分支留下的東西，**Matrix 版不使用**（Matrix 用的是 `MatriXRT-llm-cli`）。

## 6. 連線到板子

板子沒有一般網路：**USB-C 接電腦**後會變成複合裝置（網路 NCM＋ADB＋MTP），板子 IP `192.100.100.2`，電腦端設 `192.100.100.1`。

1. **Windows 驅動**（NCM 是內建驅動，但要手動指定）：裝置管理員 → 找到 CDC NCM 裝置（狀態異常）→ 更新驅動程式 → 瀏覽 → 讓我從清單挑選 → 從磁碟安裝 → 瀏覽到 `C:\Windows\INF\usbncm.inf` → 選「UsbNcm Host Device」。
2. **直接接電腦，不要經過 USB hub**（實測經 hub 會讓裝置消失）。
3. 電腦端網卡設固定 IP（需系統管理員）：`192.100.100.1`／`255.255.255.0`，然後 `ping 192.100.100.2`。
4. **SSH 用金鑰**（密碼登入在這條連線上不通）：先用序列埠（COM14，115200，`root`/`root`）把公鑰寫進板子的 `/root/.ssh/authorized_keys`，之後 `ssh root@192.100.100.2`、`scp` 都免密碼。
   `/root` 在 OTA 或重置時會被清掉，金鑰要重新寫。
5. 瀏覽器開 `http://192.100.100.2:5000`（電腦或經同網段的裝置）。

## 7. 從零啟用（部署）

### 7.0 快速部署（推薦）
板子已具備 **LLM 引擎、`/data/pylibs` 套件、修改版 `matrixrt_yolo`**（見 7.1，這三樣屬於板子環境）且電腦能 SSH 到板子時：
```bash
git clone -b Realtek-Matrix https://github.com/Miller0330/RIV-FoodAI-Backup.git
cd RIV-FoodAI-Backup
./deploy/matrix/deploy_to_board.sh              # 預設板子 IP 192.100.100.2，可帶參數指定其他 IP
```
腳本會：檢查前置條件（缺什麼會印出警告）→ 複製 100 類模型與標籤到 `/data/models/uecfood_s512/` → 複製程式、營養資料、前端到 `/data/RIV-FoodAI-Backup/Riscv2025_Server_scale_Code/` → 安裝並啟用兩個 systemd 服務 → 清 `__pycache__` → 重啟服務。
完成後等 **2–3 分鐘暖機**再開 `http://192.100.100.2:5000`。腳本可重複執行，也就是日後更新程式的標準做法。
（Windows 請在 Git Bash 或 WSL 執行；電腦端要先完成第 6 節的網路與 SSH 金鑰設定。）

### 7.1 需要準備的檔案

| 項目 | 板子上路徑 | 來源 |
|---|---|---|
| 程式 | `/data/RIV-FoodAI-Backup/Riscv2025_Server_scale_Code/` | 本 repo 的 `Riscv2025_Server_scale_Code/`（由部署腳本 `scp` 過去，板上不是 git checkout） |
| Gemma4 E4B 引擎 | `/data/models/gemma4_e4b/`（`MatriXRT-llm-cli`、`packed_engine_e4b_fixed_strip_llmonly.json`、`.alm`、`ple.bin`、`tokens.bin`、`tokenizer.json`） | Realtek 預載於板子，共約 5.8GB；`MatriXRT-llm-cli` 需要 `chmod +x` |
| 100 類 NPU 模型 | `/data/models/uecfood_s512/network_binary.nb` | **本 repo `models/uecfood_s512/`**，部署腳本會複製；重新產生見第 11 節 |
| 100 類標籤檔 | `/data/models/uecfood_s512/labels.txt` | 本 repo `models/uecfood_s512/labels.txt`；每行一個英文類別名，順序即訓練的 class index（同 [`training/eval/labels_100.txt`](training/eval/labels_100.txt)） |
| 修改版 YOLO 執行檔 | `/opt/matrix/bin/matrixrt_yolo`，備份在 `/data/matrixrt_yolo_patched/matrixrt_yolo_bin` | Realtek 範例程式修改而來（支援 `--imgsz --nc --cores`），因屬 vendor 程式碼**未放進 repo**；全新板子需依 7.4 自行修改編譯 |
| Python 套件 | `/data/pylibs`（flask、gunicorn、torch 2.5.1…） | 見 7.2 |
| 系統套件 | apt：`python3-opencv`、`python3-numpy`、`python3-serial` | 在 `/dev/root`，重置後仍在 |

### 7.2 Python 套件（一定要裝在 `/data`）
`/root`、`/opt`、`/home`、`/usr/local`、`/var` 都是會被 OTA／重置清空的暫存層（實際存在 `/data/_overlay`），
`pip install --user` 裝的東西**重開機就消失**。所有 pip 套件必須裝到 `/data/pylibs`：

```bash
# 在電腦上下載 aarch64 / cp312 的 wheel，scp 到板子後：
pip3 install --break-system-packages --no-index --find-links=<wheel資料夾> --target=/data/pylibs flask gunicorn torch
```
服務以 `PYTHONPATH=/data/pylibs` 找到它們。`cv2`、`numpy` 走 apt 安裝，不用放 `/data`。

### 7.3 systemd 服務
`/etc` 是真正的 ext4（不會被清），服務檔放這裡才會開機自動啟動。以下內容就是 repo 的 [`deploy/matrix/`](../deploy/matrix/) 資料夾裡的兩個檔案，部署腳本會自動安裝。

`/etc/systemd/system/riv-foodai.service`
```ini
[Unit]
Description=RIV-FoodAI-Backup Flask server
After=network.target

[Service]
Type=simple
WorkingDirectory=/data/RIV-FoodAI-Backup/Riscv2025_Server_scale_Code
Environment=PYTHONDONTWRITEBYTECODE=1
Environment=PYTHONPATH=/data/pylibs
ExecStart=/usr/bin/python3 -m gunicorn --workers 1 --timeout 120 --bind 0.0.0.0:5000 --access-logfile - --access-logformat "%(t)s %(h)s %(r)s %(s)s resp_time=%(D)sus" main:app
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```
`/etc/systemd/system/restore-matrixrt-yolo.service`（開機時把修改版執行檔從 `/data` 還原回被清掉的 `/opt`）
```ini
[Unit]
Description=Restore patched matrixrt_yolo (multi-class decoder) after OTA/reboot wipes /opt
After=local-fs.target
Before=riv-foodai.service

[Service]
Type=oneshot
ExecStart=/bin/cp /data/matrixrt_yolo_patched/matrixrt_yolo_bin /opt/matrix/bin/matrixrt_yolo
ExecStartPost=/bin/chmod +x /opt/matrix/bin/matrixrt_yolo
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```
```bash
systemctl daemon-reload
systemctl enable restore-matrixrt-yolo.service riv-foodai.service
systemctl start restore-matrixrt-yolo.service riv-foodai.service
```

### 7.4 修改版 `matrixrt_yolo`
Realtek 原版寫死 COCO 80 類／640px。修改版新增 `--imgsz <n> --nc <n> --cores <n>`（本模型需 `--imgsz 512 --nc 100 --cores 2`）。
原始碼在板子 `/opt/matrix/examples/matrixrt_yolo/src/matrixrt_yolo.cpp`（除錯映像內建 gcc/cmake，`cd` 到該目錄執行 `./build.sh` 即可重編），
修改版原始碼與編譯好的執行檔備份在 `/data/matrixrt_yolo_patched/`。

### 7.5 首次啟動與驗證
1. 啟動服務後**等 2–3 分鐘**（torch 載入＋Gemma 引擎載入）再測；太早測會誤以為壞了。
2. `journalctl -u riv-foodai.service --no-pager -n 30` 應看到「AI 模型準備完成」，沒有 Traceback。
3. 瀏覽器開首頁，上傳一張水果照，確認：結果頁先出現、框線與中文品名正確、AI 建議約 20 秒後出現在下方卡片。
4. 在「三餐主食」分頁上傳飯或麵的照片，確認有偵測並算出營養。
5. 拖動份量滑桿，熱量與營養表會即時變動。

## 8. 日常操作

**更新程式**
```bash
scp main.py root@192.100.100.2:/data/RIV-FoodAI-Backup/Riscv2025_Server_scale_Code/
ssh root@192.100.100.2 "find /data/RIV-FoodAI-Backup -name __pycache__ -exec rm -rf {} + ; systemctl restart riv-foodai.service"
```
**一定要清 `__pycache__`**：板子沒有 RTC，時間每次開機重置，Python 用檔案時間判斷 pyc 是否過期，會誤用舊版程式（改了程式卻沒效果就是這個）。

**看紀錄**：`journalctl -u riv-foodai.service -f`（沒有另外的 log 檔，避免吃掉 `/data` 空間）。

**確認執行中的模型**：`ps aux | grep -E "MatriXRT|gunicorn"`。

**換 100 類模型**：把新的 `network_binary.nb` 與 `labels.txt` 放到新資料夾（例如 `/data/models/<新名>/`，repo 內建議放 `models/<新名>/`），修改 `Detector100.py` 的 `MODEL_PATH`、`LABELS_PATH` 與 `IMG_SIZE`，清 pycache 後重啟（或改 `deploy_to_board.sh` 的模型資料夾後重新部署）。舊模型保留當備份，回滾只需改回這幾行。

## 9. 設定與門檻一覽

| 設定 | 位置 | 預設 | 說明 |
|---|---|---|---|
| 9 類偵測門檻 | `main.py`（`TFLiteDetector(conf_threshold=)`） | 0.25 | 偵測器本身的門檻，故意放寬避免漏掉 |
| 9 類納入營養的門檻 | `main.py` `MIN_CONFIDENCE_FOR_NUTRITION` | 0.5 | 低於此值畫成虛線框、不算營養 |
| 100 類偵測門檻 | `main.py`（`UECFood100Detector(conf_threshold=)`） | 0.15 | |
| 100 類納入營養的門檻 | `main.py` `MIN_CONFIDENCE_FOR_NUTRITION_100` | 0.3 | 100 類信心值普遍較低，門檻較寬 |
| 100 類 NMS IoU | 同上 | 0.45 | |
| 上傳大小上限 | `main.py` `MAX_CONTENT_LENGTH` | 16MB | 瀏覽器已先縮圖，通常遠小於此 |
| 上傳資料夾保留數 | `main.py` `MAX_KEPT_UPLOADS` | 200 個檔案 | 超過自動刪最舊的 |
| LLM 單次等待／載入上限 | `LLMAdvisor.py` `RESPONSE_TIMEOUT`／`STARTUP_TIMEOUT` | 90 s／120 s | 回答變長後從 60 調高 |
| gunicorn 逾時 | systemd `--timeout` | 120 s | 需大於冷啟動載入時間 |
| 每日參考值 | `Calculate.py` | 2000 kcal、蛋白質 60g、碳水 300g、纖維 25g、脂肪 65g | 一般成人標準，非個人化 |
| NPU 核心數 | `Detector100.py` `NPU_CORES` | 2 | 本模型 `.nb` 需要 2 核 |

門檻取捨（板端 2,109 張驗證圖，現行模型）：信心值 0.2 時召回 75.7%／精確度 61.5%；0.3 時 72.0%／66.0%；0.5 時 59.6%／78.7%。降低門檻少漏檢但多誤判，提高則相反。

## 10. 資料：營養資料庫與訓練資料

### 營養資料庫
- **`nutrition_data.json`（9 類）**：USDA FoodData Central SR Legacy 實查，每 100g 的 `calories/protein/carbs/fiber/fat`，另含 `default_weight_g`（典型單份重量）與 `usda_fdc_id`、`usda_description`。舊版的零食與小餅乾熱量曾被低估約 3 倍，已修正。
- **`nutrition_data_100_uecfood.json`（100 類）**：14 筆為 USDA 實查、86 筆為整道菜估計值（每 100g）。已審查過一輪，修正了約 20 筆 USDA 自動比對到錯誤食物的項目（如味噌湯抓到味噌醬、拉麵抓到乾燥泡麵、牛肉麵抓到罐頭濃縮湯、鰻魚飯只算魚肉等）；熱量依 4/4/9 大卡換算與三大營養素保持一致。
  鍵名是 `UEC001_白飯` 這種格式，依編號排序後與訓練的 class index 一一對應；程式用 `english_name`（空白、`-`、`'` 換成 `_`）對到 `labels.txt`。
- **修改資料要注意**：
  - 欄位不能隨意增減字串型欄位以外的數值欄位名稱；`food_logic.py` 只讀 `calories/protein/carbs/fiber/fat/default_weight_g`。
  - 估計值仍可能有偏差，發現不合理請直接修 JSON（熱量最好等於 4×蛋白質＋4×碳水＋9×脂肪），修改後需重啟服務（啟動時載入）。
  - 若重新用 USDA API 查詢，`DEMO_KEY` 會被限流（HTTP 429），請申請正式 key 或放慢速度。

### 訓練資料
- **UEC FOOD 100**（東京電通大學，http://foodcam.mobi/dataset100.zip，約 945MB）：100 類、每張附框標註，**僅限非商業研究使用**。日本料理為主，與台灣便當菜色有落差。
- 目前版本：訓練 12,252 張、驗證 2,109 張，標註框 12,469／2,142 個；每類訓練 85–619 張。固定隨機種子 0，85%／15% 切分，重跑 `prepare_dataset_full.py` 會得到相同切分。
- 目前沒有台灣便當／自助餐專用的公開框標註資料集，補資料需要自行拍攝標註。

## 11. 模型訓練與轉換流程

### 訓練（電腦端）
需要 GPU 才實用（第一版用 CPU 花 10.6 小時）。詳見 [`training/uecfood_gpu/TASK.md`](training/uecfood_gpu/TASK.md)：
```bash
python setup_dataset.py     # 下載資料集並轉成 YOLO 格式
python run_train_gpu.py     # YOLO11s / 512px / 80 epochs，完成後匯出 ONNX
```
必須安裝 **CUDA 版** PyTorch（腳本會檢查）。實測（RTX 4060 Laptop 8GB）：batch 32 約占 5.3GB VRAM，每個 epoch 約 2.6 分鐘，80 個 epoch 共約 3.4 小時；VRAM 不足時把 `run_train_gpu.py` 的 `batch` 改成 16。
中斷後續訓：`YOLO("runs/detect/runs/uecfood_gpu_s512/weights/last.pt").train(resume=True)`，不要從頭重跑。
`best.pt` 為 mAP50 0.782／0.610；最後一個 epoch 為 0.760／0.594（以 best 為準）。匯出的 ONNX 為固定解析度（1×3×512×512），輸出形狀應為 `(1, 104, 5376)`（100 類＋4 框座標；512px 對應 5376 個 anchor），不符代表匯出尺寸錯了。

### 轉成 NPU `.nb`（需 x86 Linux＋Python 3.10＋Realtek Reality 工具鏈）
在 VirtualBox 的 Ubuntu 22.04 虛擬機（`Andes_Qilai_Ubuntu`）內，工具鏈模組名稱為 `realitylib`：
```bash
python3.10 importer.py <目錄>/                       # 目錄內放 <名稱>.onnx；會產生 inputmeta.yml 並顯示需要的校正檔名
python3.10 quantize_hybrid_yolo.py <目錄>/ asymu8 yolov11s --hybrid_qtype dfpi16 --iterations 200
python3.10 export_nbg_setup.py <目錄>/ asymu8 --use_hybrid
# 產出 <目錄>/network_binary.nb
```
- `yolo_version` 字串是 `yolov11n` / `yolov11s`（不是 `yolo11n`）。
- 校正圖：每類 2 張共 200 張，**與輸入同解析度**，存成 `inputs/images_<N>_1_3_<H>_<W>_<i>.npy`（`<N>` 是 importer 印出的編號，`<i>` 從 0 遞增；前處理為 RGB、/255、直接拉伸縮放，要與板端 `matrixrt_yolo` 一致）。**只給 1 張校正圖時量化後信心值會明顯下降**。
- 前處理說明：訓練與 Ultralytics 驗證用 letterbox，但板端 `matrixrt_yolo` 是把整張圖**直接拉伸**成正方形（框座標也是拉伸後的座標，網頁疊圖用「除以輸入邊長」換算），校正圖也用同樣的拉伸縮放，實測 NPU 與浮點仍相當接近。
- 量化約 10 秒／張校正圖（YOLO11s，512px）。
- 這個 `.nb` 需要 **2 個 NPU 核心**，板端要加 `--cores 2`。

### 板端驗證（換模型後必做）
```bash
# 電腦端
tar cf - -C yolo_full100/images/val . | ssh root@192.100.100.2 "mkdir -p /data/val_eval/img && tar xf - -C /data/val_eval/img"
# 板子上跑 training/eval/run_board_eval.sh（先把裡面的模型路徑與 --imgsz 改成新模型）
python float_eval.py && python compare_eval.py      # 比較浮點與 NPU 的召回、精確度
```
`training/eval/run_board_eval.sh` 目前仍寫舊模型（416px、`uecfood_full100`），評估新模型要改成 `--imgsz 512` 與 `/data/models/uecfood_s512/network_binary.nb`。
換模型後請確認信心值 0.3 時 NPU 召回與浮點差距不大（現行約 1 個百分點）。

## 12. 注意事項（務必先讀）

1. **gunicorn 必須 `--workers 1`。** 只有一顆 NPU、一個常駐 LLM 引擎，多 worker 會各自啟動互搶的引擎。
2. **服務重啟後要等 2–3 分鐘。** 這段時間網頁可能已可開啟但辨識或 AI 建議還沒好；首次請求特別慢是正常的。
3. **一定要清 `__pycache__`**（見第 8 節），板子時間會重置。
4. **`/root`、`/opt`、`/home`、`/usr/local`、`/var` 會被 OTA／重置清空**：程式、模型、pip 套件一律放 `/data`；SSH 金鑰要重新寫入；`matrixrt_yolo` 靠 `restore-matrixrt-yolo.service` 從 `/data` 還原。`/etc` 與 `/data` 不受影響。
5. **根目錄 `/` 只有 3GB 且已用約 77%**，不要往上面放大檔。
6. **LLM 用法的限制**：`-p`（系統提示）在這個引擎版本無效，格式與長度限制必須寫在使用者問題本身；要用 `_llmonly` 引擎設定檔（另一個是多模態版，純文字呼叫會當掉）；載入需 20–75 秒，所以必須常駐子程序，不能每次請求啟動一次。
7. **AI 建議是一次性的**，且只在頁面載入時產生：使用者調整份量滑桿後，頁面會提示「建議依調整前的份量產生」。
8. **LLM 佔用期間其他請求會排隊**（單 worker）。前端已改成建議非同步，且等到結果圖載入完才發出建議請求，避免圖片排在 LLM 後面。
9. **板子沒有對外路由**：網頁不要引用 CDN；所有前端資源都在 `static/vendor/`。
10. **對話版原型**（`/data/RIV-FoodAI-Chat`，port 5001，`riv-foodai-chat.service`）預設停用，多輪回答不穩定，**絕對不能與主服務同時啟動**（會搶同一個 LLM 引擎）。
11. **記憶體吃緊**：不要在板子上同時再啟動別的重程式（例如另一份 LLM CLI 做測試會讓兩份引擎互搶而變得極慢）。
12. **100 類為測試版**：日式家常菜為主，湯品、白飯、外觀相近的魚類料理先天較弱，請把結果當作「輔助」，並用份量滑桿修正。
13. **UEC FOOD 100 僅限非商業研究**；Gemma 與 Realtek 工具鏈各有自己的授權條款，商用前需確認。
14. **類別順序必須一致**：`labels.txt`、訓練的 `data.yaml`（`training/eval/labels_100.txt`）、`nutrition_data_100_uecfood.json` 的編號排序三者順序要相同；改類別名稱時，`nutrition_data_100_uecfood.json` 的 `english_name` 與程式中的 `name_map_100` 也要同步。
15. **弱類別的熱量僅供參考**：白飯（recall 僅約 0.13）、生菜沙拉（0.25）、味噌湯（0.36）等，湯與白飯常佔滿整張圖、標註範圍不一致，先天難辨。
16. **電子秤（CT1）**：預設 `/dev/ttyUSB1`、115200。更換秤重元件需用 `ct1_scale_calibration.ino`（基準重量 `sample_weight = 1500` g）重新校正，再把比例因子填入 `ct1_scale.ino`。USB 供電問題尚未解。
17. **資安**：`run_server_normal.sh` 內有硬編碼的使用者密碼設定、`run_server_scale.sh` 的註解裡寫有 root 帳號資訊，只適用內網開發板，**正式部署前請移除**，服務也不要暴露到公網。
18. **不要進版控**：`train_env/`、`UECFOOD100/`、`yolo_full100/`、`dataset100.zip`（約 1GB）、`runs/`、預訓練權重（`yolo11s.pt` 等，Ultralytics 會自動下載）。例外：`training/uecfood_gpu/results/` 的 `best.pt`、`best.onnx` 是刻意納入版控的。

## 13. 疑難排解

| 症狀 | 可能原因 | 處理 |
|---|---|---|
| 改了程式卻沒效果 | `__pycache__` 誤用舊 pyc（無 RTC） | 清 pycache 後重啟 |
| 網頁打不開／連不上 | NCM 網卡沒設 IP、經過 hub、服務未起 | `ping 192.100.100.2`；直接插電腦；`systemctl status riv-foodai` |
| 剛重啟就測試，辨識或建議失敗 | 還在暖機 | 等 2–3 分鐘，看 `journalctl` 是否有「AI 模型準備完成」 |
| 上傳後 Internal Server Error | 看 `journalctl` 的 Traceback；曾因 `.jfif` 副檔名讓 `cv2.imwrite` 出錯，已改存 `.jpg` | 依 Traceback 修正 |
| AI 建議顯示「取得失敗」 | LLM 引擎沒起來或逾時 | `ps aux \| grep MatriXRT`；重啟服務 |
| 建議顯示「已過期」 | token 一次性，重新整理過 | 重新辨識 |
| 100 類完全偵測不到 | `matrixrt_yolo` 被還原成原版或模型路徑錯 | `ls /opt/matrix/bin/`；手動跑 `matrixrt_yolo <nb> <圖> --imgsz 512 --nc 100 --cores 2 -c labels.txt` |
| 手動跑 `matrixrt_yolo` 報「需要 2 cores」 | 沒加 `--cores 2` | 加上參數 |
| 重開機後套件找不到 | pip 套件裝在 `/root/.local` | 重裝到 `/data/pylibs` |
| SSH 突然要密碼／拒絕 | OTA 清掉 `/root/.ssh` | 用序列埠重新寫入公鑰 |
| 磅秤顯示「串口連接失敗 /dev/ttyUSB1」 | 沒接磅秤（正常） | 系統照常運作；要用磅秤需接上並確認 USB 供電（目前 CT1 供電問題未解，建議用有供電的 hub 排查） |
| 板端 NPU 偵測明顯比浮點差 | 校正圖太少或解析度不符 | 用每類 2 張、同解析度的校正圖重新量化 |
| 系統變得極慢／OOM | 同時跑了兩份 LLM 或別的大程式 | 停掉多餘程序，確認只有 `riv-foodai` |

## 14. 已知限制與待辦

**先天限制**
- RAM 約 3.9GB，可動用約 1GB；LLM 只能用 Realtek 預打包的 E2B／E4B，自行轉換其他模型需 vendor 工具鏈支援（尚未確認）。
- LLM CLI 不提供 tokens/s，無法拆解 prompt、串流、解碼各占多少時間；約 20 秒的生成難以再壓縮。
- 單 worker：同時多人使用會排隊。
- 營養數值多為估計，重量沒有磅秤時只能用預設份量。
- 資料集偏日式。

**待辦（依價值）**
1. 9 類偵測器改跑 NPU，去掉 torch，省記憶體與啟動時間
2. 兩組類別集是否合併（9 類訓練影像已遺失，需重新收集）
3. 補台灣便當／自助餐影像資料
4. 與 Realtek 確認更大 LLM 或更快 runtime 的可行性
5. 弱類別補資料、評估更高精度量化
6. 「這餐有主食嗎？」提示、偵測 Top-3 候選（需改 `matrixrt_yolo` 輸出前三類）
7. 每日累計與個人化營養目標

## 15. 與其他分支的關係

| 分支 | 板子 | 備註 |
|---|---|---|
| `Realtek-Matrix`（本分支） | Realtek Matrix RTD2811 | NPU 偵測＋NPU LLM |
| `main` | Qilai EVB（RISC-V） | 各自的偵測與 LLM |
| `Genio_720` | MediaTek Genio 720 | llama.cpp |
| `shared-ui-nutrition` | 不限 | 只含與模型無關的新 UI／營養邏輯（`food_logic.py`、模板、USDA 營養資料、`Calculate.py`），供其他分支取用；**不含**任何偵測器、LLM、模型檔。整合方式見 `Riscv2025_Server_scale_Code/SHARED_UI_INTEGRATION.md`；注意新的 `nutrition_data.json` 必須與 `food_logic.py` 及 `main.py` 修改一起套用 |

## 16. 授權與資料來源
- UEC FOOD 100：東京電通大學，僅限非商業研究使用。
- USDA FoodData Central（SR Legacy）：公有領域。
- Gemma：依 Google Gemma 使用條款；Realtek Reality／Acuity-LLM 工具鏈與 `matrixrt` 依 Realtek 授權。
- Ultralytics YOLO11：AGPL-3.0。
