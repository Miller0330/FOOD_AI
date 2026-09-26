# Genio 720 平台說明

MediaTek Genio 720 EVK（ARM64 Linux）。與 RISC-V 版共用同一份應用層，只有 LLM 實作不同：

- LLM：llama.cpp **b10964**，可執行檔 `/root/llama-b10964/llama-b10964/llama-completion`，搭配 prompt cache（`sysprompt.cache` 放在專案資料夾）。實作在 `Riscv2025_Server_scale_Code/platforms/genio/llm_advisor.py`（即 `Genio_720` 分支的 `LLMAdvisor.py`，未修改）。
- 依賴：`requirements/genio.txt`（比 RISC-V 多 `torch`、`opencc-python-reimplemented`）。
- 啟動：`PLATFORM=genio python3 main.py`（或放好 `llama-completion` 後由自動偵測選到）。
- 預設行為與 RISC-V 相同：同步建議、只在 `__main__` 暖機、不清理上傳資料夾、沒有 `/detect100`。

整併前的狀態保存在分支 `backup/genio-720-baseline`。實機驗證步驟見 [`VERIFICATION.md`](VERIFICATION.md) 的 C 節。
