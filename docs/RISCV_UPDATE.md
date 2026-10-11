# Andes Voyager（晶心科技，RISC-V）更新說明

`main` 已更新為多平台整併版。這份文件說明這次改了什麼，以及怎麼在板子上更新、驗證與回滾。
**這個版本尚未在 Andes Voyager 實機上驗證**（PC 上的自動測試與煙霧測試已通過），請照下面的檢查表測一次。

## 這次更新的內容

與 Matrix 一致的部分（共用應用層）：
- 營養計算：依偵測框面積分配重量、無秤重時用各食物典型單份重量、信心值低於 0.5 的偵測不列入營養、USDA 營養資料（含脂肪）、熱量與營養素表一致
- 結果頁：框線與中文品名由瀏覽器疊圖、份量滑桿即時重算、離線 Bootstrap、上傳前瀏覽器縮圖
- AI 建議改為**非同步**：結果頁先出現，建議約 9 秒後補上
- 上傳資料夾自動保留最近 200 個檔案
- 修正：結果圖有時會被清理程式誤刪（板子沒有 RTC 時，時間倒退造成）

維持不變：
- LLM：`llama-cli-new` ＋ Qwen2.5-0.5B ＋ prompt cache（`platforms/riscv/llm_advisor.py` 與整併前位元組相同，測試會比對）
- 偵測：torch（CPU）9 類水果零食；`/detect100`（100 類）在 RISC-V 上不提供
- 磅秤：`/dev/ttyUSB1`，沒接時系統照常運作
- 依賴：`requirements.txt` 沒有改；原本的 `run_server_*.sh` 沒有改

## 在板子上更新

1. **確認板子上的 repo 指向哪裡**
   ```bash
   cd <板子上的專案目錄>
   git remote -v
   ```
   原腳本 clone 的是 `BiovlsiSystemGroup/RIV-FoodAI-Y2`，那個 repo **不會**有這次更新。要更新請指向我們的 repo：
   ```bash
   git remote set-url origin https://github.com/Miller0330/RIV-FoodAI-Backup.git
   ```
2. **先記下目前版本（回滾用）**
   ```bash
   git rev-parse HEAD            # 抄下這串
   ```
3. **更新（跳過 LFS 大檔，RISC-V 用不到）**
   ```bash
   GIT_LFS_SKIP_SMUDGE=1 git pull origin main
   ```
4. **啟動（不要設定 PLATFORM）**
   ```bash
   cd Riscv2025_Server_scale_Code
   python3 main.py
   ```
   預期輸出：`[HAL] 平台 = riscv（來源：autodetect:/home/ademo/llama-cli-new）`（或 `default`）、`正在準備 AI 模型，請稍候...`、`AI 模型準備完成`，Flask 監聽 `0.0.0.0:5000`。
   不應出現 `matrix`、`genio`、`matrixrt`、`/data/`。
5. **清掉舊的 `__pycache__`**（板子沒有 RTC 時，舊快取可能被誤用）：
   ```bash
   find . -name __pycache__ -exec rm -rf {} +
   ```

## 驗證檢查表（請在實機上做）

- [ ] 首頁只有一個辨識表單（沒有「三餐主食」分頁）
- [ ] 上傳水果照片：結果頁先出現，約 9 秒後下方出現 AI 建議
- [ ] 結果頁：框線與中文品名正確、份量滑桿可拖動、熱量兩張表一致
- [ ] 連續上傳超過 100 次後，結果圖仍能正常顯示
- [ ] `POST /detect100` 回 404
- [ ] 有接磅秤時 `GET /api/weight` 回傳重量；沒接時網頁照常可用
- [ ] LLM 建議文字風格與更新前相同

## 環境變數

| 變數 | 預設 | 說明 |
|---|---|---|
| `RISCV_ASYNC_ADVICE` | `1` | 設為 `0` 退回整併前的同步行為（算完建議才顯示結果頁） |
| `MAX_KEPT_UPLOADS` | `200` | 上傳資料夾保留的檔案數；`0` 表示不清理 |
| `PLATFORM` | 自動偵測 | 板上有 `/home/ademo/llama-cli-new` 時自動選 `riscv`，一般不需設定 |

## 回滾

任何一步出問題，回到更新前：
```bash
git checkout backup/main-riscv-baseline      # 整併前的 main（146e45b）
# 或回到步驟 2 記下的版本：
git reset --hard <步驟2記下的 commit>
```
`backup/main-riscv-baseline` 是 GitHub 上保留的整併前狀態。

## 已知限制

- 尚未在 Andes Voyager 實機驗證；沒有這塊板子的延遲與精度實測數據
- 沒有 100 類主食辨識，也沒有 NPU，LLM 只有 0.5B
- 沒有 systemd 服務，需手動 `python3 main.py`（要開機自動啟動需自行建立服務）
- `run_server_normal.sh` 內有硬編碼帳密，正式部署前請移除
