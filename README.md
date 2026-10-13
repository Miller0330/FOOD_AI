# RIV-FoodAI｜多平台食物辨識與營養估算

拍照 → 偵測食物 → （搭配磅秤重量）估算熱量與營養 → 由板端 LLM 產生飲食建議。
同一份程式支援三種硬體平台，用**硬體抽象層（HAL）**把「應用層」與「平台專屬的推論／LLM」分開：

| 平台 | 板子 | LLM | 偵測 | 辨識類別 |
|---|---|---|---|---|
| `riscv` | RISC-V（Andes Voyager（晶心科技），原 `main`） | llama.cpp riscv64（`llama-cli-new`）＋ Qwen2.5-0.5B ＋ prompt cache | torch（CPU）9 類 | 水果／零食 9 類 |
| `genio` | MediaTek Genio 720（ARM64 Linux） | llama.cpp b10964（`llama-completion`，CPU）：Qwen2.5-3B 或 Gemma 4 E4B | 9 類（CPU）＋ **100 類 YOLO11s（onnxruntime CPU）** | 9 類＋三餐主食 100 類 |
| `matrix` | Realtek Matrix RTD2811（Ubuntu 24.04） | Gemma4 E4B（NPU，`MatriXRT-llm-cli`） | 9 類（CPU）＋ **100 類 YOLO11s（NPU）** | 9 類＋三餐主食 100 類 |

各平台的 UI、營養計算、營養資料庫、份量滑桿等**完全共用**；只有 LLM 與 100 類 NPU 偵測是各平台自己的。

## 如何選擇平台
環境變數 `PLATFORM=riscv|genio|matrix`。沒設定時：依板子特徵檔案自動偵測，再不行**預設 `riscv`**（＝整併前 `main` 的行為）。
啟動時第一行會印出 `[HAL] 平台 = … （來源：env / autodetect:… / default）`。設了不支援的值會直接報錯，不會靜默退回。

## 快速開始

### RISC-V（維持原本用法，不需要設定任何東西）
```bash
git clone <本 repo> && cd <repo>            # 可加 GIT_LFS_SKIP_SMUDGE=1，RISC-V 用不到 LFS 大檔
pip3 install -r requirements.txt            # 根目錄的 requirements.txt 完全沒有改動
cd Riscv2025_Server_scale_Code
python3 main.py                             # 監聽 0.0.0.0:5000；啟動時暖機 LLM，跟以前一樣
```
`run_server_normal.sh`、`run_server_scale.sh`、`requirements.txt` 與整併前逐位元組相同。詳細原始說明保留在 [`docs/README-riscv.md`](docs/README-riscv.md)。

### Genio 720
```bash
pip3 install -r requirements/genio.txt      # 比 RISC-V 多 torch、opencc、onnxruntime
# 模型放 /root/models/uecfood_s512/{best.onnx,labels.txt}；詳見 docs/README-genio.md
cd Riscv2025_Server_scale_Code
PLATFORM=genio python3 main.py
```

### Realtek Matrix
見 [`docs/README-matrix.md`](docs/README-matrix.md)（連線、部署、systemd、模型、疑難排解）。一鍵部署：
```bash
git lfs pull                                # NPU 模型以 Git LFS 存放
./deploy/matrix/deploy_to_board.sh          # 服務檔已設定 PLATFORM=matrix
```

## 專案結構
```
Riscv2025_Server_scale_Code/       ← 執行目錄（RISC-V 腳本會 cd 進來，路徑不變）
  main.py                          統一入口；只依賴 hal 的介面
  food_logic.py                    營養計算、框線資料、份量、非同步建議暫存（與硬體無關）
  Calculate.py  Detector.py  WeightSensor.py   共用（Detector、WeightSensor 與整併前完全相同）
  nutrition_data.json              9 類營養資料（USDA）
  templates/  static/vendor/       前端（本機化 Bootstrap，不依賴外網）
  hal/                             介面（LLMBackend、Platform）與平台選擇
  platforms/riscv|genio|matrix/    各平台的 LLM（與其他平台程式互不 import）；matrix 另有 100 類偵測器與營養資料
  LLMAdvisor.py                    相容層（舊程式碼 `from LLMAdvisor import ...` 仍可用）
requirements.txt                   原檔不動（RISC-V 用）
requirements/                      common、riscv（等同根目錄）、genio、matrix
deploy/matrix/                     Matrix 專用部署腳本與 systemd 服務檔
models/uecfood_s512/               100 類 NPU 模型（Git LFS）
training/                          訓練（GPU）與 NPU 對浮點驗證腳本
tests/                             整併的測試（不需要硬體）
docs/                              各平台 README、HAL 架構、雙平台驗證清單
```

## 文件
- [`docs/HAL_ARCHITECTURE.md`](docs/HAL_ARCHITECTURE.md)：架構、如何新增平台、保護 RISC-V 的機制
- [`docs/VERIFICATION.md`](docs/VERIFICATION.md)：**RISC-V／Genio／Matrix 的啟動與驗證清單**
- [`docs/README-riscv.md`](docs/README-riscv.md)、[`docs/README-matrix.md`](docs/README-matrix.md)
- 測試：`python -m unittest discover -s tests -v`（需要 flask、opencv、numpy、pyserial）

## 授權與資料來源
UEC FOOD 100 僅限非商業研究；USDA FoodData Central 為公有領域；Gemma、Realtek 工具鏈、Ultralytics YOLO（AGPL-3.0）依各自條款。
