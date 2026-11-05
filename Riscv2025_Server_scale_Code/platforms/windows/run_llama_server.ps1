# 啟動 llama-server（NVIDIA 顯卡）。路徑不寫死，用參數或環境變數：
#   $env:LLAMA_DIR = "<llama.cpp CUDA 版解壓縮的資料夾>"
#   $env:LLM_GGUF  = "<GGUF 模型檔>"
#   .\run_llama_server.ps1                      # 小模型：全部放顯卡
#   .\run_llama_server.ps1 -NCpuMoe 30          # MoE 大模型：前 30 層的專家放系統 RAM
#   RTX 4060 Laptop（8GB）實測：Gemma 4 26B-A4B UD-Q4_K_M 用 -NCpuMoe 22 約 25 tokens/秒、顯示記憶體 7.3GB；
#   設太小（10~16）會讓顯示記憶體爆掉、退回共用記憶體，速度掉到 6~13 tokens/秒。
param(
    [string]$LlamaDir = $env:LLAMA_DIR,
    [string]$Model = $env:LLM_GGUF,
    [int]$Port = 8080,
    [int]$Ctx = 8192,
    [int]$NGpuLayers = 99,
    [int]$NCpuMoe = 0
)

if (-not $LlamaDir -or -not (Test-Path "$LlamaDir\llama-server.exe")) { throw "找不到 llama-server.exe，請設定 LLAMA_DIR 或 -LlamaDir" }
if (-not $Model -or -not (Test-Path $Model)) { throw "找不到模型檔，請設定 LLM_GGUF 或 -Model" }

$serverArgs = @('-m', $Model, '-ngl', $NGpuLayers, '-c', $Ctx, '-np', 1, '--jinja', '--host', '127.0.0.1', '--port', $Port)
if ($NCpuMoe -gt 0) { $serverArgs += @('--n-cpu-moe', $NCpuMoe) }
& "$LlamaDir\llama-server.exe" @serverArgs
