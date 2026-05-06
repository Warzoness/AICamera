#!/usr/bin/env python3
"""
Demo: So sánh đọc trực tiếp từ COCO JSON vs TXT format
"""

import json
import time
from pathlib import Path


def load_from_txt_format(labels_dir, images_dir):
    """Load dữ liệu từ TXT format (như YOLOv5 hiện tại)"""
    start_time = time.time()
    data = {}

    txt_files = list(Path(labels_dir).glob('*.txt'))
    for txt_file in txt_files:
        image_name = txt_file.stem + '.png'
        image_path = Path(images_dir) / image_name

        if not image_path.exists():
            continue

        annotations = []
        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:])
                        annotations.append([class_id, x, y, w, h])

        data[image_name] = annotations

    load_time = time.time() - start_time
    return data, load_time


def load_from_coco_json(json_path, images_dir):
    """Load dữ liệu trực tiếp từ COCO JSON"""
    start_time = time.time()

    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    images_dict = {img['id']: img for img in coco_data['images']}
    categories_dict = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}

    data = {}

    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in images_dict:
            continue

        img_info = images_dict[image_id]
        image_name = img_info['file_name']
        image_path = Path(images_dir) / image_name

        if not image_path.exists():
            continue

        # Convert bbox
        x, y, w, h = ann['bbox']
        img_width = img_info['width']
        img_height = img_info['height']

        x_center = (x + w/2) / img_width
        y_center = (y + h/2) / img_height
        w_norm = w / img_width
        h_norm = h / img_height

        class_id = categories_dict.get(ann.get('category_id', 1), 0)

        if image_name not in data:
            data[image_name] = []

        data[image_name].append([class_id, x_center, y_center, w_norm, h_norm])

    load_time = time.time() - start_time
    return data, load_time


def main():
    BASE_DIR = Path(__file__).parent
    JSON_PATH = BASE_DIR / 'data' / 'dataexample' / 'labels' / 'merged_coco.json'
    TXT_LABELS_DIR = BASE_DIR / 'datasets' / 'pig_coco' / 'labels'
    IMAGES_DIR = BASE_DIR / 'data' / 'dataexample' / 'images'

    print("=" * 80)
    print("🏁 BENCHMARK: So sánh tốc độ load dữ liệu")
    print("=" * 80)

    # Test multiple runs
    num_runs = 5
    txt_times = []
    json_times = []

    for run in range(num_runs):
        print(f"\n🔄 Run {run + 1}/{num_runs}")

        # Load từ TXT
        _, txt_time = load_from_txt_format(TXT_LABELS_DIR, IMAGES_DIR)
        txt_times.append(txt_time)

        # Load từ JSON
        _, json_time = load_from_coco_json(JSON_PATH, IMAGES_DIR)
        json_times.append(json_time)

        print(".3f")
        print(".3f")

    # Statistics
    avg_txt = sum(txt_times) / len(txt_times)
    avg_json = sum(json_times) / len(json_times)
    speedup = avg_json / avg_txt

    print("\n" + "=" * 80)
    print("📊 KẾT QUẢ")
    print("=" * 80)
    print(".3f")
    print(".3f")
    print(".2f")

    print("\n💡 KẾT LUẬN:")
    if avg_txt < avg_json:
        print("  → TXT format NHANH HƠN {:.1f}x".format(speedup))
        print("  → Khuyến nghị: Dùng TXT format (như đã làm)")
    else:
        print("  → JSON direct nhanh hơn, nhưng hiếm khi xảy ra")

    print("\n📋 Lý do TXT format tốt hơn:")
    print("  ✅ YOLOv5 gốc hỗ trợ native")
    print("  ✅ Nhanh hơn khi train (không parse JSON)")
    print("  ✅ Dễ debug và visualize")
    print("  ✅ Chuẩn hóa dữ liệu một lần")


if __name__ == '__main__':
    main()