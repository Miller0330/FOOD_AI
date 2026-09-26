#!/bin/bash
# 在板子上執行：對 /data/val_eval/img/*.jpg 逐張跑 NPU 模型，輸出到 /data/val_eval/out.txt
# 用法：先把驗證圖 tar 過去（tar cf - -C yolo_full100/images/val . | ssh root@192.100.100.2 "mkdir -p /data/val_eval/img && tar xf - -C /data/val_eval/img"），
#       nohup ./run_board_eval.sh &，等最後一行出現 DONE 後把 out.txt 抓回電腦存成 board_out.txt
# 電腦端：float_eval.py 產生 float_preds.json，再跑 compare_eval.py 比較（需在 uecfood_training 資料夾內，含 yolo_full100/）
OUT=/data/val_eval/out.txt
: > $OUT
for f in /data/val_eval/img/*.jpg; do
  echo "IMG $(basename $f)" >> $OUT
  /opt/matrix/bin/matrixrt_yolo /data/models/uecfood_full100/network_binary.nb $f --imgsz 416 --nc 100 --cores 2 --conf 0.05 -c /data/models/uecfood_full100/labels.txt 2>&1 | grep -E 'class=' >> $OUT
done
echo DONE >> $OUT
