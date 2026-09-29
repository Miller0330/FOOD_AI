# HAL 架構說明

## 設計目標
單一專案庫同時支援 RISC-V、Genio 720、Realtek Matrix；**整併不能改變 RISC-V（原 `main`）的可執行性與行為**。

## 分層
```
應用層（共用）   main.py  food_logic.py  Calculate.py  templates/  static/  Detector.py  WeightSensor.py
                      │ 只透過介面呼叫
HAL 介面        hal/interfaces.py：LLMBackend（warmup / get_advice）、Platform、Meal100Config
                hal/selector.py  ：PLATFORM 環境變數 > 檔案特徵自動偵測 > 預設 riscv
                      │ 延遲載入（只 import 被選中的平台）
平台層          platforms/riscv/   llm_advisor.py = 整併前 main 的 LLMAdvisor.py（位元組相同）
                platforms/genio/   llm_advisor.py = Genio_720 分支的 LLMAdvisor.py
                platforms/matrix/  llm_advisor.py（Acuity-LLM）、detector100.py（matrixrt_yolo）、100 類營養資料
```
`Platform` 物件描述一個平台：`llm`、功能旗標（`meal100`、`async_advice`）、是否在模組載入時暖機（`warmup_at_import`）、
100 類偵測器設定（`meal100`，不支援的平台為 `None`）。

## 各平台預設行為（刻意保守）
| | riscv | genio | matrix |
|---|---|---|---|
| AI 建議 | 同步（算完才出結果頁，與整併前相同） | 同步 | 非同步（結果先出，建議稍後補上） |
| 暖機時機 | 只在 `python3 main.py` 的 `__main__` | 同上 | 模組載入時（gunicorn） |
| 100 類 `/detect100` | 不存在 | 不存在 | 有 |
| 上傳資料夾清理 | 不清理（預設 0） | 不清理 | 保留最近 200 個 |
非同步／清理都可以事後調整（`Platform.features`、環境變數 `MAX_KEPT_UPLOADS`），不影響其他平台。

## 保護 RISC-V 的機制
1. **依賴隔離**：根目錄 `requirements.txt`、`run_server_*.sh` 沒有任何改動；各平台另有 `requirements/*.txt`。
2. **推論／LLM 隔離**：每個平台的 LLM 只在該平台被選中時才 import；`tests` 驗證選 `riscv` 時 `platforms.matrix`、`platforms.genio` 不會被載入。
3. **位元組保護**：`tests/riscv_baseline_hashes.json` 記錄 `main@146e45b` 的雜湊，測試確認 `platforms/riscv/llm_advisor.py`（原 `LLMAdvisor.py`，用 `git mv` 保存為 100% rename）、`Detector.py`、`WeightSensor.py`、`debug_llm.py`、`llama-cli-new`、`requirements.txt`、`run_server_*.sh` 完全相同。
4. **路徑與周邊 I/O**：RISC-V 的 `/home/ademo/...`、串口 `/dev/ttyUSB1`、相對路徑（`static/uploads`、`nutrition_data.json`）都沒有被改動；Matrix 的 `/data/...` 路徑只存在 `platforms/matrix/` 與 `deploy/matrix/`。
5. **相容層**：頂層 `LLMAdvisor.py` 轉出被選平台的所有名稱，舊程式碼仍可 `from LLMAdvisor import get_llm_advice, _ensure_prompt_cache`。
6. **備份**：整併前的三個分支保存在 `backup/main-riscv-baseline`、`backup/genio-720-baseline`、`backup/realtek-matrix-baseline`。

## 整併時有意帶進 RISC-V／Genio 的改變（行為與整併前不同的地方）
- 新 UI（結果頁疊圖、份量滑桿、本機化 Bootstrap、首頁上傳前縮圖）。
- 新營養資料庫（USDA 實查；舊版零食與小餅乾熱量被低估約 3 倍）與新營養計算（有秤重依框面積分配、無秤重用典型單份重量、信心值 < 0.5 的偵測不列入計算、加入脂肪）。
- `/api/advice/<token>` 端點存在（RISC-V／Genio 預設不使用）。

## 新增一個平台
1. 建 `platforms/<name>/`：放該平台的 LLM 實作（`llm_advisor.py`）與 `__init__.py`，提供 `create()` 回傳 `Platform`。
2. 在 `hal/selector.py` 的 `ALLOWED` 與（可選）`MARKERS` 加上名稱與特徵檔。
3. 在 `requirements/` 加該平台的依賴檔；在 `tests/test_hal.py` 加該平台的路由測試。
不需要改 `main.py`。

## Git LFS
`models/**/*.nb`、`training/uecfood_gpu/results/*.pt|*.onnx` 走 Git LFS。RISC-V／Genio 用不到這些檔案，clone 時可設 `GIT_LFS_SKIP_SMUDGE=1`。
`main` 原本就有的二進位檔（`detectv2*.`、`llama-cli-new`）維持原狀，沒有改成 LFS。
