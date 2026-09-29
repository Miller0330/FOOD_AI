# 訓練成果（YOLO11s / 512px / 80 epochs / UEC FOOD 100）

| 檔案 | 說明 |
|---|---|
| `best.pt` | 最佳 epoch 的 PyTorch 權重（mAP50 0.782 / mAP50-95 0.610） |
| `best.onnx` | 匯出的 ONNX（opset 12，輸入 1x3x512x512，輸出 `(1, 104, 5376)`）；轉 NPU `.nb` 用這個 |
| `data.yaml` | 100 類名稱與順序（`labels.txt` 必須與此一致） |
| `per_class_metrics.txt` | 每類 Precision / Recall / mAP50 / mAP50-95 |
| `results.csv`、`results.png` | 逐 epoch 訓練曲線 |
| `confusion_matrix_normalized.png` | 混淆矩陣 |
| `args.yaml` | 訓練參數 |
| `train_log.txt` | 完整訓練 log（`.log` 副檔名被 .gitignore 排除，故改名） |

## 轉 NPU 的注意
- 校正圖與匯出一律 512px，校正圖至少 60 張（實際用每類 2 張共 200 張）。
- 前處理：letterbox 512、RGB、`/255`。
- 板端：`matrixrt_yolo <nb> <img> --imgsz 512 --nc 100 --cores 2`。
- 驗證集與第一版（11n/416）不同，mAP 只能大致比較。

## 重現
見上層 `TASK.md` 與根目錄 `README.md` 第 4 節。
