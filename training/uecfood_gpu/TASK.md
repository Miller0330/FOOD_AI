# 任務：用 RTX 4060 筆電重新訓練 UEC FOOD 100 偵測模型（YOLO11s）

## 背景
專案 RIV-FoodAI（Realtek Matrix 開發板，NPU 跑 YOLO）。第一版模型是在沒有 GPU 的電腦上訓練的：
YOLO11n、416px、25 epochs、每類最多 200 張，耗時 10.6 小時，結果 mAP50=0.683 / mAP50-95=0.528（100 類）。
最後幾個 epoch 的 mAP50 只從約 0.674 慢慢升到 0.683，尚未完全收斂。這次用 GPU 換更大的設定再訓練。

## 這次的設定（見 run_train_gpu.py）
- 模型 YOLO11s、imgsz 512、80 epochs、batch 32（VRAM 不夠改 16）
- 不設每類影像上限（第一版上限 200），使用 UEC FOOD 100 全部影像
- 注意：驗證集因此和第一版不同，mAP 數字只能大致比較，不是嚴格同一份驗證集

## 步驟
1. 安裝環境（Python 3.10–3.12 皆可；請確認裝的是 CUDA 版 PyTorch）：
   ```
   python -m venv train_env
   train_env\Scripts\activate
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   pip install ultralytics pillow
   python -c "import torch; print(torch.cuda.is_available())"   # 必須是 True
   ```
2. 準備資料（自動下載約 945MB 並轉成 YOLO 格式，若已有 dataset100.zip 放在此資料夾就會跳過下載）：
   ```
   python setup_dataset.py
   ```
   完成後應該出現 `yolo_full100/`，訓練影像數量會比第一版（12,997 張）多。
3. 訓練與匯出 ONNX：
   ```
   python run_train_gpu.py
   ```
   先看第一個 epoch 的速度和 VRAM（預期每個 epoch 數分鐘內）。若 CUDA out of memory 就把 batch 改 16 重跑。
4. 訓練中如果被中斷，用 `YOLO("runs/detect/runs/uecfood_gpu_s512/weights/last.pt").train(resume=True)` 續訓，不要從頭來。

## 完成後請回報／交回
- `runs/detect/runs/uecfood_gpu_s512/weights/best.pt` 和 `best.onnx`（匯出的 ONNX 輸出形狀應為 (1, 104, 5376)，即 100 類 + 4，512px 對應 5376 個 anchor）
- 最終整體與每類的 mAP50 / mAP50-95 / Precision / Recall（`results.csv` 與訓練 log 最後的表格）
- 和第一版比較：整體 mAP50 提升多少、哪些類別進步/退步

## 之後（不在這台筆電做）
best.onnx 要拿回原本的電腦，跑 Realtek Reality 工具鏈（VirtualBox 內的 Andes_Qilai_Ubuntu）轉成 NPU 的 `.nb`：
校正圖與匯出都要用 512px（不是 416），板端 `matrixrt_yolo --imgsz 512 --nc 100 --cores 2`，並用 60 張以上的校正圖。
這部分由原本那邊的對話處理。
