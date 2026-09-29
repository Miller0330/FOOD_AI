#!/usr/bin/env bash
# 把本 repo 的程式、100 類 NPU 模型與 systemd 服務檔部署到 Realtek Matrix 板子並重啟服務。
# 用法（在 repo 根目錄、電腦已能 ssh 到板子）：  ./deploy/matrix/deploy_to_board.sh [板子IP]
# 板子 IP 預設 192.100.100.2（USB-NCM）。需先設定好 SSH 金鑰登入 root（見 docs/README-matrix.md 第 6 節）。
#
# 只會部署 Matrix 需要的檔案（hal/ 與 platforms/matrix/），RISC-V、Genio 的平台程式不會被複製過去。
# 服務檔設定了 PLATFORM=matrix。
#
# 這個腳本「不會」處理下列東西（屬於板子環境或 vendor 提供）：
#   - Gemma4 LLM 引擎（/data/models/gemma4_e4b/）
#   - Python 套件（/data/pylibs：flask、gunicorn、torch）
#   - 修改版 matrixrt_yolo 執行檔（/data/matrixrt_yolo_patched/matrixrt_yolo_bin）
#   - 模型檔案以 Git LFS 存放：clone 後請先執行 `git lfs pull`，否則 network_binary.nb 只是幾行文字的指標檔
set -euo pipefail

BOARD="${1:-192.100.100.2}"
REMOTE_APP=/data/RIV-FoodAI-Backup/Riscv2025_Server_scale_Code
REMOTE_MODEL=/data/models/uecfood_s512
cd "$(dirname "$0")/../.."
SSH="ssh root@$BOARD"

# 防呆：LFS 指標檔只有 ~130 位元組
if [ "$(wc -c < models/uecfood_s512/network_binary.nb)" -lt 100000 ]; then
  echo "!! models/uecfood_s512/network_binary.nb 看起來只是 Git LFS 指標檔，請先執行：git lfs pull" >&2
  exit 1
fi

echo "==> 檢查板子前置條件"
$SSH "test -x /data/models/gemma4_e4b/MatriXRT-llm-cli" \
  || echo "!! 警告：找不到可執行的 /data/models/gemma4_e4b/MatriXRT-llm-cli，AI 建議會無法使用"
$SSH "test -d /data/pylibs/flask" \
  || echo "!! 警告：/data/pylibs 沒有 flask 等套件，服務會啟動失敗（見 requirements/matrix.txt）"
HAS_PATCHED_YOLO=0
if $SSH "test -f /data/matrixrt_yolo_patched/matrixrt_yolo_bin"; then HAS_PATCHED_YOLO=1; else
  echo "!! 警告：找不到 /data/matrixrt_yolo_patched/matrixrt_yolo_bin（修改版執行檔），100 類辨識會無法使用"
fi

echo "==> 複製 100 類 NPU 模型與標籤"
$SSH "mkdir -p $REMOTE_MODEL $REMOTE_APP/templates $REMOTE_APP/static/vendor $REMOTE_APP/platforms"
scp models/uecfood_s512/network_binary.nb models/uecfood_s512/labels.txt "root@$BOARD:$REMOTE_MODEL/"

echo "==> 複製程式、資料與前端"
S=Riscv2025_Server_scale_Code
scp $S/main.py $S/food_logic.py $S/Detector.py $S/LLMAdvisor.py $S/Calculate.py $S/WeightSensor.py \
    $S/nutrition_data.json $S/detectv2_torch.pt "root@$BOARD:$REMOTE_APP/"
scp $S/templates/*.html "root@$BOARD:$REMOTE_APP/templates/"
scp $S/static/vendor/* "root@$BOARD:$REMOTE_APP/static/vendor/"
scp -r $S/hal "root@$BOARD:$REMOTE_APP/"
scp $S/platforms/__init__.py "root@$BOARD:$REMOTE_APP/platforms/"
$SSH "rm -rf $REMOTE_APP/platforms/matrix"
scp -r $S/platforms/matrix "root@$BOARD:$REMOTE_APP/platforms/"

echo "==> 安裝 systemd 服務並重啟"
scp deploy/matrix/riv-foodai.service deploy/matrix/restore-matrixrt-yolo.service "root@$BOARD:/etc/systemd/system/"
$SSH "find /data/RIV-FoodAI-Backup -name __pycache__ -exec rm -rf {} + 2>/dev/null; systemctl daemon-reload; systemctl enable riv-foodai.service"
if [ "$HAS_PATCHED_YOLO" = 1 ]; then
  $SSH "systemctl enable restore-matrixrt-yolo.service; systemctl start restore-matrixrt-yolo.service"
fi
$SSH "systemctl restart riv-foodai.service"

echo "==> 完成。等 2-3 分鐘暖機（torch 與 LLM 引擎載入）後開 http://$BOARD:5000"
echo "    看紀錄：ssh root@$BOARD 'journalctl -u riv-foodai.service -f'（開頭應有「[HAL] 平台 = matrix」）"
