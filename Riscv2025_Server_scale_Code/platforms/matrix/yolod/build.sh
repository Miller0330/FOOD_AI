#!/bin/sh
# 在 Matrix 板上直接編譯常駐偵測程序，輸出到 /data/yolod/matrixrt_yolod
set -e
mkdir -p /data/yolod
cd "$(dirname "$0")"
g++ -std=c++17 -O2 matrixrt_yolod.cpp -o /data/yolod/matrixrt_yolod \
  -I/opt/matrix/include -I/usr/include/opencv4 \
  -Wl,-rpath,/opt/matrix/lib /opt/matrix/lib/libmatrixrt.so \
  -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lpthread -ldl
echo "built /data/yolod/matrixrt_yolod"
