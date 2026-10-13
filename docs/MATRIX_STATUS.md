# Realtek Matrix 版本進度小總結（分支 `Realtek-Matrix`，更新於 2026-09-24）

## 目前狀態
- 板子（RTD2811，3.9GB RAM，2 核 A73，NPU）上以 systemd 服務 `riv-foodai.service` 常駐執行，開機自動啟動，gunicorn **單一 worker**（NPU 與 LLM 引擎只有一份，不可增加 worker）。
- 網頁首頁有兩個分頁：
  - 水果／零食辨識（原本 9 類，torch 跑在 CPU，路由 `/detect`）
  - 三餐主食辨識（測試版，UEC FOOD 100 共 100 類，NPU 上的 YOLO11s / 512px，路由 `/detect100`）
- 兩個偵測器互相獨立，尚未合併，等實測後再決定。
- 辨識結果頁會先顯示偵測與營養表，AI 營養師建議由頁面載入後非同步向 `/api/advice/<token>` 取得（LLM 約 20 秒，Gemma4-E4B，NPU）。
- 網頁資源（Bootstrap）已本機化，不依賴外網；上傳前瀏覽器會先把照片縮到最長邊 1280px。

## 連線與部署
- 板子 USB-C 接電腦後為 NCM 網路，IP `192.100.100.2`（電腦端 `192.100.100.1`），SSH 用金鑰登入 root。
- 程式在 `/data/RIV-FoodAI-Backup/Riscv2025_Server_scale_Code/`，用 `scp` 覆蓋檔案後：
  1. 清掉 `__pycache__`（板子沒有 RTC，時間每次開機重置，會誤用舊 pyc）
  2. `systemctl restart riv-foodai.service`
  3. 等約 2–3 分鐘暖機（torch 載入 + LLM 引擎載入）再測試
- 看紀錄：`journalctl -u riv-foodai.service`
- `/root`、`/opt`、`/home`、`/usr/local`、`/var` 會在 OTA 或重置時被清掉，**模型與程式一律放 `/data`**。Python 套件在 `/data/pylibs`（服務以 `PYTHONPATH` 指向）。

## 模型與資料位置（板子上）
| 項目 | 路徑 |
|---|---|
| LLM（Gemma4 E4B） | `/data/models/gemma4_e4b/`（使用 `packed_engine_e4b_fixed_strip_llmonly.json`） |
| 100 類偵測模型（目前使用） | `/data/models/uecfood_s512/network_binary.nb`（YOLO11s，512px，約 11ms） |
| 100 類偵測舊版（備份） | `/data/models/uecfood_full100/`（YOLO11n，416px，僅備份；新版的 `labels.txt` 在 `uecfood_s512/`） |
| 已 patch 的 YOLO 執行檔 | `/opt/matrix/bin/matrixrt_yolo`（備份於 `/data/matrixrt_yolo_patched/`，開機由 `restore-matrixrt-yolo.service` 還原） |

`matrixrt_yolo` 已改為可指定 `--imgsz --nc --cores`，100 類模型需 `--imgsz 512 --nc 100 --cores 2`（舊版 416）。

## 模型訓練與轉換流程（電腦端）
1. 訓練：Ultralytics YOLO11，資料 UEC FOOD 100（`training/uecfood_gpu/` 有 GPU 版腳本）。第一版：11n、416px、25 epochs，mAP50=0.683；現行版（RTX 4060）：11s、512px、80 epochs，mAP50=0.782。
2. 匯出 ONNX（opset 12，固定解析度）。
3. VirtualBox 內 `Andes_Qilai_Ubuntu`：`importer.py` → `quantize_hybrid_yolo.py <dir> asymu8 yolov11n --iterations 60` → `export_nbg_setup.py <dir> asymu8 --use_hybrid`。
4. 校正圖用每類 2 張共 200 張、與輸入同解析度（`--iterations 200`）。板端實測（2109 張驗證圖）NPU 與浮點差距很小：信心值 0.3 時召回 72.0% vs 73.2%。評估腳本在 `training/eval/`。

## 已知限制
- 板子上只有 1.5GB 左右可用記憶體，LLM 換更大模型前要先確認 Realtek 工具鏈是否支援。
- LLM 引擎無對話記憶；`-p` 系統提示不生效，限制格式要寫在使用者問題裡。
- 對話版原型（port 5001）擱置，多輪回答不穩定，與主服務不可同時執行。
- CT1 磅秤的 USB 供電問題未解（需實機用有供電的 hub 測試）。
- 100 類營養資料多為估計值（USDA 對應不到的整道菜），發現不合理請直接修 `nutrition_data_100_uecfood.json`。

## 尚未做
- 9 類偵測器改跑 NPU（去掉 torch，省記憶體與啟動時間）
- 兩個偵測器是否合併
- 每日累計與個人化營養目標
