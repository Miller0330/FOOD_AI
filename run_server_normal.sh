#!/bin/bash
# ===============================
#  Riscv2025_Server_Code Script
#  User: hunter
# ===============================

# 建立使用者 hunter
useradd -m hunter


# 顯示網路資訊
ifconfig

# 切換到 hunter 帳號環境
su - hunter <<'EOF'

# 下載專案
git clone https://github.com/BiovlsiSystemGroup/RIV-FoodAI-Y2.git
cd RIV-FoodAI-Y2

# 安裝套件
pip3 install -r requirements.txt

# 進入一般版本 Server
cd Riscv2025_Server_Code

# 執行主程式
python3 main.py

EOF
