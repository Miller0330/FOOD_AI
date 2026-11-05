# Windows（NVIDIA 獨立顯卡）平台說明

分支：`Windows-GPU`。在 Windows 筆電上用獨立顯卡跑大型語言模型（LLM）：拍照辨識食物、程式計算營養、LLM 給出飲食建議，並可記錄每天吃的東西、產生每日總結、跟營養師持續對話。全部在本機執行，照片與飲食紀錄不上傳雲端。

與其他平台共用網頁、營養計算與 100 類流程，只有 LLM 與偵測的推論實作不同（見 [`HAL_ARCHITECTURE.md`](HAL_ARCHITECTURE.md)）。

> 設計原則：**營養數字一律由程式計算**，LLM 只負責把數字說成人話，並被要求不要提到紀錄裡沒有的食物。

## 功能

| 功能 | 說明 | 位置 |
|---|---|---|
| 100 類食物辨識 `/detect100` | YOLO11s 512px（`best.onnx`），onnxruntime 跑 CPU，把顯卡留給 LLM | `platforms/windows/detector100.py`（沿用 Genio 的實作） |
| AI 單餐建議 | 呼叫本機 llama-server（HTTP），非同步顯示，結果頁顯示使用的模型與耗時 | `platforms/windows/llm_advisor.py` |
| 每日飲食紀錄 | 每次辨識結果自動記錄（SQLite）；結果頁調整份量或換選食物會同步更新；`/log` 看全天合計與建議量達成率，可刪除單筆 | `diet_log.py`、`templates/log.html` |
| AI 每日總結 | 程式算好全天合計，LLM 統整一天的飲食並給明天的建議；紀錄沒變就用快取，並記下模型與耗時 | `/api/log/summary` |
| 跟營養師對話 | 串流回覆；每輪帶入當天紀錄與最近 12 輪對話；歷史存在 SQLite，以天為單位 | `/api/log/chat` |
| 沒有磅秤時 | 重量為 0 時用每種食物的預設份量；磅秤初始化失敗不影響啟動 | `food_logic.py`、`WeightSensor.py` |

每日紀錄與對話只在 `features['diet_log']` 為真的平台啟用（目前只有 Windows），其他平台的行為不變。

## 架構

```
瀏覽器 ── Flask（waitress）── 100 類偵測（onnxruntime，CPU）
                │                    │
                │              營養計算（程式查營養表）
                │                    │
                ├── SQLite（data/diet_log.db）：每餐紀錄、總結快取、對話歷史
                │
                └── HTTP ── llama-server（llama.cpp，CUDA）── GGUF 模型
                            單餐建議、每日總結、對話
```

8GB 顯示記憶體放不下大模型，所以用 MoE 模型，並以 llama.cpp 的 `--n-cpu-moe` 把前 N 層的專家層放在系統記憶體，共用層留在顯卡。換模型只要換 llama-server 載入的 GGUF 與 `--n-cpu-moe`，網站程式不用改。

## 檔案

| 檔案 | 用途 |
|---|---|
| `platforms/windows/__init__.py` | 平台設定（`meal100`、`async_advice`、`diet_log`，載入時暖機） |
| `platforms/windows/llm_advisor.py` | 單餐建議、每日總結、對話（串流）的提示詞與 HTTP 呼叫 |
| `platforms/windows/detector100.py` | 100 類偵測器（沿用 Genio，只換預設路徑） |
| `platforms/windows/serve.py` | 用 waitress 啟動網站（Windows 不能用 gunicorn） |
| `platforms/windows/run_llama_server.ps1` | 啟動 llama-server（路徑用環境變數或參數） |
| `diet_log.py` | SQLite 儲存層 |
| `requirements/windows.txt` | 相依套件 |

## 測試環境

| 項目 | 規格 |
|---|---|
| 筆電 | NVIDIA GeForce RTX 4060 Laptop，8GB 專屬顯示記憶體 |
| CPU | Intel Core i7-13620H（10 核 16 緒） |
| 系統記憶體 | 64GB DDR5-4800（2×32GB） |
| 作業系統 | Windows 11 Home |
| 顯卡驅動 | 610.74（`nvidia-smi` 顯示支援 CUDA 13.3）；CUDA Toolkit 12.9 |
| llama.cpp | b12064，Windows CUDA 12.4 版（`llama-b12064-bin-win-cuda-12.4-x64.zip` 加 `cudart-llama-bin-win-cuda-12.4-x64.zip`）。驅動最高只支援 CUDA 13.3，所以沒有用 13.4 版 |
| Python | 3.12.4（虛擬環境） |
| 主要套件 | Flask 3.1.0、waitress 3.0.2、onnxruntime 1.31.1、torch 2.15.0（CPU 版）、numpy 2.5.4、Pillow 13.0.0、opencv-python-headless 5.0.0.93、opencc-python-reimplemented 0.1.7、pyserial 3.5 |

另一台 RTX 5070 Ti Laptop（12GB）的硬體效能對照尚未進行。

## 模型與實測

模型來源皆為 Hugging Face 的 GGUF 量化檔；授權依各模型頁標示（以下四個皆標示 Apache 2.0）。模型選擇原則：主模型採非中國來源（Gemma、Mistral），Qwen 僅作對照。

| 模型 | 來源 | 檔案 | 量化 | `--n-cpu-moe` | 生成速度 | 顯示記憶體（GiB） | 系統記憶體（GiB） |
|---|---|---|---|---|---|---|---|
| Qwen3 8B（小模型對照） | Alibaba | `Qwen/Qwen3-8B-GGUF` 4.7 GiB | Q4_K_M | 不需要（全放顯卡） | 約 39 tok/s | 5.7 | — |
| **Gemma 4 26B-A4B**（目前使用） | Google | `unsloth/gemma-4-26B-A4B-it-GGUF` 15.8 GiB | UD-Q4_K_M | 22 | 約 25 tok/s | 7.1 | 約 14 |
| Qwen3.6 35B-A3B（對照） | Alibaba | `unsloth/Qwen3.6-35B-A3B-GGUF` 20.6 GiB | UD-Q4_K_M | 30 | 約 29 tok/s | 7.2 | 約 18 |
| Mistral Small 4 119B | Mistral AI | `unsloth/Mistral-Small-4-119B-2603-GGUF` 37.4 GiB | UD-Q2_K_XL | 34 | 約 9 至 12 tok/s | 4.9 | 約 37 |

### 網站功能的耗時（本機實測，單次）

| 模型 | 單餐建議 | 每日總結 | 對話（每輪，串流） |
|---|---|---|---|
| Qwen3 8B | 約 3 秒 | 6.5 秒 | 未測 |
| Gemma 4 26B-A4B | 約 7 秒 | 約 10 秒 | 首字約 1.5 秒，整輪 4 至 6 秒 |
| Qwen3.6 35B-A3B | 約 7 秒 | 約 10 秒 | 首字約 0.4 秒（第一輪 3.7 秒），整輪 3 至 5 秒 |
| Mistral Small 4 119B | 32 秒 | 77 秒 | 未測 |

對話每輪都會把當天紀錄、歷史與新問題整段送給模型；llama-server 會重用已處理過的前綴，所以歷史變長時延遲增加不明顯。

### `--n-cpu-moe` 的影響（`llama-bench`，生成速度 tok/s）

放在 RAM 的專家層數太少，顯示記憶體會爆掉、退回共用記憶體，速度反而大幅下降。

| 模型 | 結果 |
|---|---|
| Gemma 4 26B-A4B | 10 層：6.5；16 層：13.5；19 至 22 層：約 26 至 28；25 層：25.0；30 層：24.4。選 22（19 至 21 層雖然也快，但顯示記憶體用到 7.7 GiB（共 8GB），太緊） |
| Qwen3.6 35B-A3B | 16 層：6.3；20 層：7.9；24 層：9.7；28 至 32 層：約 29 至 30；40 層：26.0。選 30 |
| Mistral Small 4 119B | 16 層：4.3；22 層：5.6；26 層：8.0；30 層：11.8；34 層：12.6；36 層以上約 10；全部專家放 CPU：11.8。選 34 |

### 品質觀察（初步，樣本很少，不能當結論）

- 單餐建議與每日總結：四個模型的繁體中文都可用，沒有出現簡體字（輸出一律再經 OpenCC 轉成繁體保底）。
- Qwen3 8B 曾編造照片裡沒有的食材。
- Gemma 4 曾編造一次使用者的喜好（「你喜歡的豆製品」），已在對話的系統提示詞加上「不編造使用者沒說過的喜好、病史或習慣」；之後問「我喜歡吃什麼」時它會回答不知道。
- Qwen3.6 曾把「25%」說成「一半」，違反數字只能引用的原則。
- Mistral Small 4（Q2 量化）每日總結寫太長（約 480 字，要求是 180 至 250 字），並把味噌湯說成濃湯；速度太慢，不適合日常使用，定位為 100B 等級的展示。
- 辨識流程只用合成圖驗證過，**沒有用真實食物照片量測辨識準確率**（模型本身的驗證集成績見根目錄 README）。

## 安裝與啟動

以下 `<…>` 請換成你自己的路徑。

### 1. 取得程式與 Python 環境

```powershell
git clone -b Windows-GPU https://github.com/Miller0330/FOOD_AI.git
cd FOOD_AI
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements\windows.txt
# torch 請裝 CPU 版
.\venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. llama.cpp 與模型

1. 到 [llama.cpp 的 release](https://github.com/ggml-org/llama.cpp/releases) 下載 Windows CUDA 版（依顯卡驅動支援的 CUDA 版本選擇）以及對應的 `cudart` 壓縮檔，兩個解壓縮到同一個資料夾。
2. 從 Hugging Face 下載 GGUF 模型（見上表）。
3. 100 類偵測模型 `best.onnx` 不在 repo 內（repo 只有 `models/uecfood_s512/` 的 NPU 與 TorchScript 版本）。請自備，並用環境變數指定。

### 3. 啟動（開兩個 PowerShell 視窗）

視窗 1，啟動 LLM（先開，約 10 秒就緒）：

```powershell
$env:LLAMA_DIR='<llama.cpp 資料夾>'; $env:LLM_GGUF='<模型檔 .gguf>'
powershell -NoProfile -ExecutionPolicy Bypass -File .\Riscv2025_Server_scale_Code\platforms\windows\run_llama_server.ps1 -NCpuMoe 22
```

視窗 2，啟動網站（約 7 秒就緒）：

```powershell
cd Riscv2025_Server_scale_Code
$env:UEC100_MODEL='<best.onnx>'; $env:UEC100_LABELS='<labels.txt>'; $env:PYTHONUTF8='1'
..\venv\Scripts\python.exe platforms\windows\serve.py
```

開啟 `http://127.0.0.1:8000`；每日紀錄與對話在 `http://127.0.0.1:8000/log`。停止時在各視窗按 `Ctrl+C`。

`UEC100_LABELS` 沒設時，會用 repo 內 `models/uecfood_s512/labels.txt`。

## 環境變數

| 變數 | 預設 | 說明 |
|---|---|---|
| `UEC100_MODEL` | `models/uecfood_s512/best.onnx` | 100 類偵測模型（需自備） |
| `UEC100_LABELS` | `models/uecfood_s512/labels.txt` | 類別名稱 |
| `UEC100_THREADS` | 8 | onnxruntime 執行緒數 |
| `LLM_URL` | `http://127.0.0.1:8080` | llama-server 位址 |
| `LLM_MAX_TOKENS` | 400 | 單餐建議的生成上限（每日總結至少 700） |
| `LLM_CHAT_MAX_TOKENS` | 350 | 對話每輪的生成上限 |
| `LLM_TIMEOUT` | 180 | 單次請求逾時（秒） |
| `LLM_TEMP` | 0.4 | 單餐建議與總結的溫度（對話固定 0.5） |
| `LLM_WARMUP_WAIT` | 120 | 啟動時等 llama-server 載入模型的秒數上限 |
| `DIET_LOG_DB` | `Riscv2025_Server_scale_Code/data/diet_log.db` | 每日紀錄資料庫 |
| `HOST` / `PORT` / `THREADS` | `127.0.0.1` / `8000` / `4` | 網站位址、埠、waitress 執行緒數 |
| `LLAMA_DIR` / `LLM_GGUF` | — | `run_llama_server.ps1` 用：llama.cpp 資料夾與模型檔 |

`HOST` 預設只允許本機連線。改成 `0.0.0.0` 讓手機連線時，同一個網路的人都能連，且沒有任何登入驗證，只建議在信任的網路使用。

## 路由

| 路由 | 說明 |
|---|---|
| `GET /`、`POST /detect`、`POST /detect100`、`POST /confirm` | 首頁、9 類與 100 類辨識、低信心確認（與其他平台相同） |
| `GET /api/advice/<token>`、`POST /api/advice/regen` | 單餐建議（非同步）與依調整後份量重新產生 |
| `GET /log?date=YYYY-MM-DD`、`GET /today` | 每日紀錄頁 |
| `DELETE /api/log/meals/<id>` | 刪除一筆紀錄 |
| `POST /api/log/meals/<id>/update` | 結果頁調整份量後同步更新（營養數字由伺服器重算） |
| `POST /api/log/summary` | 產生或取出快取的每日總結（`force` 為真則重新產生） |
| `GET`／`POST`／`DELETE /api/log/chat` | 讀取歷史、送出問題（串流，每行一個 JSON）、清除當天對話 |

## 資料與隱私

- 紀錄、總結快取與對話歷史存在 `data/diet_log.db`（已列入 `.gitignore`）；上傳的照片存在 `static/uploads/`（同樣不會被 commit）。
- 目前只有單一使用者（資料表已預留 `user` 欄位）。
- 每日建議量是固定的一般成人數值（熱量 2000 大卡、蛋白質 60g、碳水 300g、纖維 25g、脂肪 65g），個人化目標還沒做。

## 疑難排解

| 現象 | 原因與處理 |
|---|---|
| 啟動 `run_llama_server.ps1` 出現「running scripts is disabled」 | Windows 預設不允許執行 `.ps1`。用上面的 `powershell -ExecutionPolicy Bypass -File …`（只對該視窗有效，不改系統設定） |
| 建議變成千篇一律的「這餐整體攝取尚屬均衡…」，且沒有顯示模型名稱 | 網站連不到 llama-server。確認視窗 1 有沒有報錯、`http://127.0.0.1:8080/health` 是否回應 |
| 啟動網站出現權限錯誤（WinError 10013） | 5000 埠常被 Windows 系統服務佔用，所以預設改用 8000；用 `PORT` 換埠 |
| 生成速度突然掉到個位數 tok/s | 顯示記憶體爆掉退回共用記憶體：調高 `--n-cpu-moe`，或關掉其他吃顯卡的程式 |
| 載入模型失敗 | 兩個模型不能同時載入；換模型前先關掉原本的 llama-server |
| 找不到 `best.onnx` | 它不在 repo 內，請設定 `UEC100_MODEL` |

## 測試

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests
```

`RiscvBaselineProtection` 有 3 項在未修改的 `Realtek-Matrix` 分支上也會失敗（受保護檔案與基準雜湊不一致），與 Windows 平台無關。

## 已知限制與後續

- 提高圖片辨識準確率：改善偵測與前處理，信心低或不在 100 類內的食物交給 VLM，並擴充台灣食品營養成分資料庫。
- 個人化每日目標（年齡、性別、體重、活動量）。
- 與另一台 RTX 5070 Ti（12GB）筆電做不同硬體的效能對照。
- 更完整的模型比較與維運指標（固定測試集，量首字延遲、tokens/秒、記憶體，並檢查數字是否正確、有沒有編造）。
- 對話強化：自動檢查回答裡的數字與紀錄是否一致、結果頁加入對話入口、沒有秤時可手動輸入重量。
