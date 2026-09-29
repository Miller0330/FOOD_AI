#!/bin/sh
# 在 Genio 板上執行：建立模型資料夾並（可選）下載 Gemma 4 E4B。
# best.onnx / labels.txt 請由 training/uecfood_gpu/results/best.onnx、models/uecfood_s512/labels.txt 複製到 /root/models/uecfood_s512/
set -e
mkdir -p /root/models/uecfood_s512
if [ "$1" = "gemma" ]; then
  wget -c -O /root/models/gemma-4-E4B-it-Q4_0.gguf \
    https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_0.gguf
fi
pip3 install --break-system-packages -r "$(dirname "$0")/../../requirements/genio.txt"
