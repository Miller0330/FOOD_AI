
## llama-cli 版本說明

- `llama-cli`：Andes 原始提供版本（build 6815），保留供對比測試
- `llama-cli-new`：Andes 更新版本（build 6817），prompt eval 速度大幅提升（4→30 token/秒），目前 `LLMAdvisor.py` 使用此版本
- **這兩個執行檔皆為 riscv64 架構編譯，僅適用於 RISC-V 平台（如 Qilai EVB）。若移植到 ARM 平台（如 MediaTek Genio），需另外取得對應架構的 llama.cpp 編譯版本，這兩個檔案無法直接使用。**
- Prompt Cache 機制（見 `LLMAdvisor.py` 內 `_ensure_prompt_cache`）與 llama-cli 版本無關，是平台無關的優化技巧，可沿用於任何 llama.cpp 相容的執行環境。
