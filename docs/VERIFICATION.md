# 雙平台（RISC-V／Genio／Matrix）驗證清單

分支：`integration/hal`。**在 RISC-V 與 Genio 實機通過以前，請不要合併進 `main`。**
自動測試已在 PC 上通過（見 A）；下面 B–D 需要在真正的板子上執行。

回滾：任何平台出問題，`git checkout backup/main-riscv-baseline`（RISC-V）、`backup/genio-720-baseline`（Genio）、`backup/realtek-matrix-baseline`（Matrix）即回到整併前狀態。

## A. 自動測試（PC 或板子皆可，不需要硬體）
```bash
python -m unittest discover -s tests -v          # 需要 flask、opencv-python、numpy、pyserial
```
應顯示 `OK`。涵蓋：RISC-V 相依檔案與 `main@146e45b` 位元組相同、平台選擇與自動偵測、選 `riscv` 時不載入 matrix／genio 程式、
三個平台的端到端路由（RISC-V／Genio 同步建議且無 `/detect100`、Matrix 非同步建議與 100 類、token 一次性、兩張表熱量一致）。

## B. RISC-V（Andes Voyager（晶心科技））—— 必須實機驗證
1. 取得程式（RISC-V 用不到 LFS 大檔）：
   ```bash
   GIT_LFS_SKIP_SMUDGE=1 git clone -b integration/hal <repo URL> && cd <repo>
   ```
2. 依賴：`pip3 install -r requirements.txt`（與以前完全相同；torch、opencc 沿用原本預裝）。
3. 啟動（**不設定 PLATFORM**）：
   ```bash
   cd Riscv2025_Server_scale_Code && python3 main.py
   ```
   預期：第一行 `[HAL] 平台 = riscv（來源：autodetect:/home/ademo/llama-cli-new）`（或 `default`）；
   接著串口訊息、`正在準備 AI 模型，請稍候...`、`AI 模型準備完成`，Flask 監聽 `0.0.0.0:5000`。
   沒有出現 `matrix`、`genio`、`matrixrt`、`/data/` 之類字樣。
4. 瀏覽器開 `http://<板子IP>:5000`：
   - 首頁**只有一個**辨識表單（沒有「三餐主食」分頁）。
   - 上傳水果照片：**結果頁先出現，AI 建議約 9 秒後補上**（與 Matrix 相同的非同步流程）；
     若要退回整併前的同步行為，啟動前設 `RISCV_ASYNC_ADVICE=0`。
   - 上傳資料夾自動保留最近 200 個檔案（與 Matrix 相同）；不要清理就設 `MAX_KEPT_UPLOADS=0`。
   - 結果頁：中文框線與品名、無信心度、份量滑桿可拖動、熱量兩張表一致、AI 建議文字內容風格與整併前相同。
   - `POST /detect100` 應回 404。
5. 磅秤（若接了 CT1）：`GET /api/weight` 行為與整併前相同（`/dev/ttyUSB1`）。
6. 與基準對照（選做）：切到 `backup/main-riscv-baseline` 跑同一張照片，LLM 建議風格應相同；
   熱量與營養數字**會不同**（有意的改變：USDA 新資料、信心值 < 0.5 不列入、加入脂肪）。
7. 確認沒被動到：`python3 -m unittest tests.test_hal.RiscvBaselineProtection -v`。
8. 走原本的安裝腳本流程（`run_server_scale.sh`）也應照常運作（腳本本身未修改）。

## C. Genio 720 —— 必須實機驗證
```bash
pip3 install -r requirements/genio.txt
cd Riscv2025_Server_scale_Code && PLATFORM=genio python3 main.py
```
預期：`[HAL] 平台 = genio（來源：env）`，暖機出現 prompt cache 建立訊息；行為同 B 的第 4 點（同步建議、無 `/detect100`）。
確認 `/root/llama-b10964/llama-b10964/llama-completion` 與 `sysprompt.cache` 路徑照原樣被使用。

## D. Realtek Matrix
```bash
git lfs pull                                       # 取回 network_binary.nb（否則只是指標檔，部署腳本會擋下）
./deploy/matrix/deploy_to_board.sh [板子IP]        # 服務檔含 PLATFORM=matrix
ssh root@<板子IP> "journalctl -u riv-foodai.service --no-pager | grep HAL"    # 應為 平台 = matrix（來源：env）
```
等 2–3 分鐘暖機，然後：
- 首頁有兩個分頁；9 類與 100 類都能辨識；結果頁先出、AI 建議約 20 秒後補上；`/detect100` 存在。
- 與整併前相比結果一致（同一張圖，偵測與熱量相同）；`/api/advice/<token>` 重新整理後顯示「建議已過期」。

## 已知需要留意
- RISC-V 的 `run_server_normal.sh` 內有硬編碼帳密（既有，未修改），正式部署前請處理。
- RISC-V 板子上 `run_server_*.sh` 是 clone 舊版 repo，本整併只在我們自己的 repo，不會自動影響那邊。
