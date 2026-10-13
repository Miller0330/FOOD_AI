# Genio 720 平台說明

MediaTek Genio 720 EVK（ARM64 Linux, Yocto, 8 核 CPU；APU 僅收純 INT8，本專案全走 CPU）。與 Matrix 版共用同一套 UI／營養邏輯／100 類流程，只有推論實作不同。

## 組成
| 項目 | 實作 | 位置 |
|---|---|---|
| 100 類偵測 `/detect100` | YOLO11s 512px `best.onnx`（與 Matrix 同一顆），onnxruntime CPU | `platforms/genio/detector100.py` |
| LLM 建議 | llama.cpp **b10964** `llama-completion`，Matrix 版提示詞，非同步顯示 | `platforms/genio/llm_advisor.py` |
| 9 類偵測 `/detect` | torch（CPU）TorchScript | `Detector.py` |
| 磅秤 | pyserial `/dev/ttyUSB1`（核心無 ftdi_sio 時退回 pyftdi） | `WeightSensor.py` |

`platforms/genio/__init__.py`：`meal100=True`、`async_advice=True`，營養資料共用 `platforms/matrix/nutrition_data_100_uecfood.json`。

## 板上檔案位置（預設）
- 應用：`/root/riv-app`（本 repo 的 `Riscv2025_Server_scale_Code/`）
- 偵測模型：`/root/models/uecfood_s512/best.onnx`、`labels.txt`（環境變數 `UEC100_MODEL`、`UEC100_LABELS`、`UEC100_THREADS` 可覆寫）
- LLM：`/root/models/qwen2.5-3b-instruct-q4_0.gguf`（預設）、`/root/models/gemma-4-E4B-it-Q4_0.gguf`；用 `LLM_MODEL` 切換，模型檔名含 `gemma` 時自動改用 Gemma 4 對話格式（`<|turn>` / `<turn|>`）
- llama.cpp：`/root/llama-b10964/llama-b10964/`

## 安裝與啟動
```bash
sh deploy/genio/setup_models.sh [gemma]      # 裝相依套件；加 gemma 會下載 4.59GB 模型
cd Riscv2025_Server_scale_Code
PLATFORM=genio python3 main.py               # 監聽 0.0.0.0:5000
```
開機自動啟動：把 `deploy/genio/riv-foodai.service` 複製到 `/etc/systemd/system/` 後 `systemctl enable --now riv-foodai`。
板子沒有 sshd、私有 repo 無法直接 clone 時，可用序列埠傳檔（base64 分塊＋SHA256 驗證，約 45 KB/s；38MB 的 onnx 約 14 分鐘）。

## 實測數據（Genio 720 EVK，CPU）
| 項目 | Genio | Matrix（NPU） |
|---|---|---|
| 100 類偵測（YOLO11s 512） | 384 ms/張 | 約 11 ms/張 |
| `/detect100` 整個請求 | 0.59 s | — |
| 建議生成：Qwen2.5-3B Q4_0 | 30.8 s | — |
| 建議生成：Gemma 4 E4B Q4_0（同 Matrix 模型） | 59.4 s（約 4.9 tok/s） | 約 20 s |

## 嚴格對照測試（2026-10-02，同一 prompt / 同一張圖，greedy）
| 項目 | Genio 720（CPU） | Matrix（NPU） |
|---|---|---|
| 偵測：常駐推論（30 次） | 350.0 ms（σ 2.5） | NPU 純推論 12.5 ms |
| 偵測：每次呼叫（含載入） | 約 590 ms（整個請求） | 舊版 414 ms（每次重載模型）→ 常駐版 **26.0 ms**（σ 0.3，重開機後驗證） |
| LLM Gemma 4 E4B 牆鐘（約 245 token 輸入） | 52.0 s（σ 0.07，Q4_0） | 32.5 s（σ 0.01，Q6_K，TTFT 1.67 s） |
| LLM 解碼速度 | 4.7 tok/s | 約 5.5 tok/s（估算） |

測試腳本見 `benchmark/`。Matrix 常駐偵測程序 `matrixrt_yolod`（原始碼與編譯腳本在 `platforms/matrix/yolod/`，安裝於 `/data/yolod/`，因 `/opt` 的修改重開機後不保留）。

## 加速嘗試紀錄（2026-09-30，Qwen2.5-3B Q4_0，預填 / 解碼 tok/s）
| 方法 | 結果 |
|---|---|
| CPU 8 執行緒（基準） | 40.8 / 8.6 |
| 調速器改 performance（已寫入 systemd） | 45.5 / 8.8（約 +10%） |
| 僅用 2 大核（CPU6、7） | 32.7 / 8.6（較差） |
| 批次 -ub 512 + flash attention | 46.7（+2%） |
| llama.cpp b11284 新版 CPU 後端 | 45.1 / 8.9（無差異） |
| Vulkan 讀 Mali-G57（b11284，-ngl 99） | 0.78 / 1.73（慢很多，無 int dot / matrix core） |
| 推測式解碼（0.5B 草稿模型，draft-simple） | 解碼 4.3（比 8.6 慢，接受率約 31%） |
| 偵測器 onnxruntime 執行緒 4 / 6 / 8 | 635 / 463 / 362 ms（8 最快） |

結論：CPU 已到上限，GPU（Vulkan）與推測式解碼皆無效。進一步加速需 NPU：
- 板上有 `neuronrt`、Neuron 8.2.16 執行庫、`/dev/apusys`，但只能執行離線編譯好的 `.dla`；編譯器 `ncc-tflite` 在主機端 SDK，需廠商權限。
- 映像 Rity 25.1 沒有官方 `llm_cmdline_tool`（LLM 的 NPU 路徑）。

## 已知限制
- 磅秤（CT1）尚未在 Genio 上實測（測試時未接上；`/dev/ttyUSB*` 不存在）。
- `/detect`（9 類）與 `/detect100` 仍是分開的入口，與 Matrix 一致。
- 舊版（9 類＋prompt cache 的 Qwen 3B）保存在分支 `backup/genio-720-baseline`。
