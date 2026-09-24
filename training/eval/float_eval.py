import json, glob, os
from ultralytics import YOLO
m = YOLO('runs/detect/runs/uecfood_full100/weights/best.onnx', task='detect')
res = {}
files = sorted(glob.glob('yolo_full100/images/val/*.jpg'))
for i in range(0, len(files), 1):
    for r in m.predict(files[i:i+1], imgsz=416, conf=0.05, iou=0.45, verbose=False):
        res[os.path.basename(r.path)] = [(int(c), float(s)) for c, s in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist())]
json.dump(res, open('float_preds.json', 'w'))
print(len(res), 'images')
