#!/bin/bash
# ===============================
#  Riscv2025_Server_scale_Code Script
#  User: root
# ===============================

# 切換 root（若已是 root 可略過）
# su -

# 顯示網路資訊
ifconfig

# 下載專案
git clone https://github.com/BiovlsiSystemGroup/RIV-FoodAI-Y2.git
cd RIV-FoodAI-Y2

# 安裝套件
pip3 install -r requirements.txt

# 進入 scale 版本 Server
cd Riscv2025_Server_scale_Code

# 執行主程式
python3 main.py
