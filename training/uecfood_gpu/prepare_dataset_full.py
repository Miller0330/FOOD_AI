"""
Convert the FULL UEC FOOD 100 dataset (all 100 classes, all available images)
into YOLO detection format for the real training run (not the earlier
20-class pipeline-validation subset).
"""
import os
import random
import shutil
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).parent
SRC = ROOT / "UECFOOD100"
OUT = ROOT / "yolo_full100"

VAL_FRACTION = 0.15
MAX_IMAGES_PER_CLASS = None  # GPU版不設上限，使用每一類的全部影像（CPU版當時為了訓練時間限制為200）

random.seed(0)


def sanitize(name):
    return name.strip().replace(' ', '_').replace("'", '').replace('-', '_').replace('/', '_')


def load_categories():
    id_to_name = {}
    with open(SRC / "category.txt", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) != 2:
                continue
            food_id, name = parts
            id_to_name[int(food_id)] = sanitize(name)
    return id_to_name


def main():
    id_to_name = load_categories()
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    class_names = [id_to_name[i] for i in sorted(id_to_name)]
    id_to_index = {food_id: idx for idx, food_id in enumerate(sorted(id_to_name))}

    total_written = 0
    total_skipped_classes = 0
    for food_id in sorted(id_to_name):
        class_dir = SRC / str(food_id)
        bb_file = class_dir / "bb_info.txt"
        if not bb_file.exists():
            print(f"SKIP {food_id} ({id_to_name[food_id]}): no bb_info.txt")
            total_skipped_classes += 1
            continue

        boxes_by_img = {}
        with open(bb_file, encoding="utf-8") as f:
            next(f)  # header
            for line in f:
                parts = line.split()
                if len(parts) != 5:
                    continue
                img_id, x1, y1, x2, y2 = parts
                boxes_by_img.setdefault(img_id, []).append((int(x1), int(y1), int(x2), int(y2)))

        img_ids = list(boxes_by_img.keys())
        random.shuffle(img_ids)
        if MAX_IMAGES_PER_CLASS:
            img_ids = img_ids[:MAX_IMAGES_PER_CLASS]
        n_val = max(1, int(len(img_ids) * VAL_FRACTION))
        val_ids = set(img_ids[:n_val])

        for img_id in img_ids:
            src_img = class_dir / f"{img_id}.jpg"
            if not src_img.exists():
                continue
            try:
                with Image.open(src_img) as im:
                    w, h = im.size
            except Exception as e:
                print(f"SKIP unreadable {src_img}: {e}")
                continue

            split = "val" if img_id in val_ids else "train"
            dst_name = f"{food_id}_{img_id}.jpg"
            shutil.copy(src_img, OUT / "images" / split / dst_name)

            lines = []
            for (x1, y1, x2, y2) in boxes_by_img[img_id]:
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                cx, cy, bw, bh = (max(0.0, min(1.0, v)) for v in (cx, cy, bw, bh))
                lines.append(f"{id_to_index[food_id]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            label_path = OUT / "labels" / split / f"{food_id}_{img_id}.txt"
            label_path.write_text("\n".join(lines), encoding="utf-8")
            total_written += 1

    yaml_content = (
        f"path: {OUT.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )
    (OUT / "data.yaml").write_text(yaml_content, encoding="utf-8")

    print(f"Wrote {total_written} images across {len(class_names)} classes to {OUT}")
    print(f"Skipped classes (no bb_info): {total_skipped_classes}")

if __name__ == "__main__":
    main()
