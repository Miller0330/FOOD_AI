import numpy as np
import torch
import cv2
import time
from typing import List, Tuple, Dict, Any

class TFLiteDetector:
    def __init__(self, model_path: str, img_size: int = 640, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        self.class_names = ['Apple', 'Banana', 'Mango', 'MiniBiscuits', 'Orange', 'Pineapple', 'Snack', 'Watermelon', 'noodles']
        self.num_classes = len(self.class_names)

        # 載入 TorchScript 模型(替換 tflite)
        # model_path 傳入 .tflite 路徑時自動換成 .pt
        if model_path.endswith('.tflite'):
            model_path = model_path.replace('detectv2_float16.tflite', 'detectv2_torch.pt')

        self.model = torch.jit.load(model_path, map_location='cpu')
        self.model.eval()

    def preprocess_image(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int], Tuple[int, int, int, int]]:
        original_shape = img.shape[:2]
        h, w = original_shape
        scale = min(self.img_size / h, self.img_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        pad_h, pad_w = self.img_size - new_h, self.img_size - new_w
        top_pad, left_pad = pad_h // 2, pad_w // 2
        bottom_pad, right_pad = pad_h - top_pad, pad_w - left_pad

        resized_img = cv2.resize(img, (new_w, new_h))
        padded_img = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        padded_img[top_pad:top_pad+new_h, left_pad:left_pad+new_w, :] = resized_img

        img_rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_input = np.expand_dims(img_norm, axis=0)

        return img_input, scale, original_shape, (top_pad, left_pad, bottom_pad, right_pad)

    def detect(self, img: np.ndarray) -> List[Dict[str, Any]]:
        img_input, scale, original_shape, padding = self.preprocess_image(img)
        top_pad, left_pad, _, _ = padding

        # 轉成 torch tensor 做推理
        input_tensor = torch.from_numpy(img_input)
        start_time = time.time()
        with torch.no_grad():
            output = self.model(input_tensor)
        inference_time = time.time() - start_time

        # 轉回 numpy
        output = output.numpy()

        if output.shape[1] == 13 and output.shape[2] == 8400:
            output = np.transpose(output, (0, 2, 1))
            boxes = output[0, :, 0:4]
            scores = output[0, :, 4:]

            class_scores = np.max(scores, axis=1)
            class_ids = np.argmax(scores, axis=1)

            keep_indices = class_scores > self.conf_threshold

            if np.any(keep_indices):
                boxes = boxes[keep_indices]
                class_scores = class_scores[keep_indices]
                class_ids = class_ids[keep_indices]
            else:
                return []

            detections = []
            h, w = original_shape
            scale_factor = min(self.img_size / h, self.img_size / w)

            for i in range(len(boxes)):
                x_center, y_center, width, height = boxes[i]
                x_min = int((x_center - width / 2) * self.img_size)
                y_min = int((y_center - height / 2) * self.img_size)
                x_max = int((x_center + width / 2) * self.img_size)
                y_max = int((y_center + height / 2) * self.img_size)

                x_min = x_min - left_pad
                y_min = y_min - top_pad
                x_max = x_max - left_pad
                y_max = y_max - top_pad

                new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                x_min = max(0, min(x_min, new_w - 1))
                y_min = max(0, min(y_min, new_h - 1))
                x_max = max(0, min(x_max, new_w - 1))
                y_max = max(0, min(y_max, new_h - 1))

                x_min = int(x_min / scale_factor)
                y_min = int(y_min / scale_factor)
                x_max = int(x_max / scale_factor)
                y_max = int(y_max / scale_factor)

                detections.append({
                    'class_id': int(class_ids[i]),
                    'class_name': self.class_names[int(class_ids[i])],
                    'confidence': float(class_scores[i]),
                    'bbox': [x_min, y_min, x_max, y_max]
                })

            return self._non_max_suppression(detections)
        else:
            return []

    def _non_max_suppression(self, detections):
        if len(detections) == 0:
            return []
        class_groups = {}
        for detection in detections:
            class_id = detection['class_id']
            if class_id not in class_groups:
                class_groups[class_id] = []
            class_groups[class_id].append(detection)

        nms_results = []
        for class_id, group in class_groups.items():
            group = sorted(group, key=lambda x: x['confidence'], reverse=True)
            while len(group) > 0:
                best_detection = group.pop(0)
                nms_results.append(best_detection)
                group = [d for d in group if self._calculate_iou(best_detection['bbox'], d['bbox']) < self.iou_threshold]

        return nms_results

    def _calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
        if union_area == 0:
            return 0
        return intersection_area / union_area

    def draw_detections(self, img, detections):
        img_copy = img.copy()
        color_map = {}
        for i, name in enumerate(self.class_names):
            color_map[i] = (
                int(120 + (i * 30) % 135),
                int(50 + (i * 40) % 205),
                int(50 + (i * 60) % 205)
            )
        for detection in detections:
            class_id = detection['class_id']
            class_name = detection['class_name']
            confidence = detection['confidence']
            x1, y1, x2, y2 = detection['bbox']
            color = color_map.get(class_id, (0, 255, 0))
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {confidence:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img_copy, (x1, y1 - label_height - baseline - 5), (x1 + label_width, y1), color, -1)
            cv2.putText(img_copy, label, (x1, y1 - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return img_copy
