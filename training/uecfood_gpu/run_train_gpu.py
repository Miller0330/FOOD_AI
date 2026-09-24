"""GPU訓練（RTX 4060筆電，8GB VRAM）。YOLO11s / 512px / 80 epochs。"""
from ultralytics import YOLO
import torch

if __name__ == "__main__":
    assert torch.cuda.is_available(), "偵測不到CUDA，請先安裝CUDA版PyTorch（見 TASK.md）"
    model = YOLO("yolo11s.pt")
    model.train(
        data="yolo_full100/data.yaml",
        epochs=80,
        imgsz=512,
        batch=32,        # 若CUDA out of memory 改成 16
        workers=4,
        device=0,
        project="runs",
        name="uecfood_gpu_s512",
        patience=0,
        exist_ok=False,
    )
    best = YOLO("runs/detect/runs/uecfood_gpu_s512/weights/best.pt")
    best.export(format="onnx", imgsz=512, opset=12)
    print("完成：runs/detect/runs/uecfood_gpu_s512/weights/best.pt 與 best.onnx")
