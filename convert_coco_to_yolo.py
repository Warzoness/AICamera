#!/usr/bin/env python3
"""
Convert COCO JSON format to YOLOv5 TXT format.
Script chuyển đổi annotation từ COCO JSON sang định dạng YOLOv5.
"""

import json
import os
from pathlib import Path
from collections import defaultdict


def convert_coco_to_yolo(coco_json_path, images_dir, output_labels_dir, copy_images_to=None):
    """
    Chuyển đổi COCO JSON annotation sang YOLOv5 TXT format.
    
    Args:
        coco_json_path: Đường dẫn tới file merged_coco.json
        images_dir: Thư mục chứa ảnh gốc
        output_labels_dir: Thư mục output cho file .txt
        copy_images_to: (Optional) Thư mục để copy ảnh vào. Nếu None thì không copy.
    """
    
    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(output_labels_dir, exist_ok=True)
    if copy_images_to:
        os.makedirs(copy_images_to, exist_ok=True)
    
    # Đọc COCO JSON
    print(f"📖 Đang đọc file: {coco_json_path}")
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # Tạo mapping
    # Image ID -> (file_name, width, height)
    images_dict = {}
    for img in coco_data['images']:
        images_dict[img['id']] = {
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height']
        }
    
    # Category ID -> Class Index (0-based)
    categories_dict = {}
    cat_id_to_name = {}
    for idx, cat in enumerate(coco_data['categories']):
        categories_dict[cat['id']] = idx
        cat_id_to_name[cat['id']] = cat['name']
    
    print(f"\n📊 Thống kê:")
    print(f"  - Số ảnh: {len(images_dict)}")
    print(f"  - Số category: {len(categories_dict)}")
    print(f"  - Categories: {', '.join([f'{name}(id={cid})' for cid, name in cat_id_to_name.items()])}")
    
    # Tạo dict lưu annotations theo image_id
    # image_id -> [(class_idx, x_center_norm, y_center_norm, w_norm, h_norm), ...]
    annotations_by_image = defaultdict(list)
    
    print(f"\n🔄 Đang xử lý {len(coco_data['annotations'])} annotation...")
    
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        # Nếu không có category_id, sử dụng category đầu tiên (hoặc default class 0)
        category_id = ann.get('category_id', list(categories_dict.keys())[0] if categories_dict else 1)
        bbox = ann.get('bbox', None)
        
        # Nếu không có bbox, bỏ qua annotation này
        if bbox is None:
            continue
        
        if image_id not in images_dict:
            print(f"⚠️  Warning: image_id {image_id} không tìm thấy trong images list")
            continue
        
        # Lấy kích thước ảnh
        img_width = images_dict[image_id]['width']
        img_height = images_dict[image_id]['height']
        
        # Chuyển đổi bbox từ COCO format sang YOLO format
        x, y, w, h = bbox
        x_center = x + w / 2
        y_center = y + h / 2
        
        # Chuẩn hoá (normalize)
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        w_norm = w / img_width
        h_norm = h / img_height
        
        # Lấy class index
        class_idx = categories_dict.get(category_id, 0)
        
        # Thêm vào dict
        annotations_by_image[image_id].append({
            'class': class_idx,
            'x': x_center_norm,
            'y': y_center_norm,
            'w': w_norm,
            'h': h_norm
        })
    
    print(f"✓ Đã xử lý xong")
    
    # Ghi file .txt
    print(f"\n💾 Đang ghi file .txt...")
    
    txt_count = 0
    empty_count = 0
    
    for image_id, img_info in images_dict.items():
        file_name = img_info['file_name']
        
        # Tạo tên file .txt
        txt_file_name = Path(file_name).stem + '.txt'
        txt_file_path = os.path.join(output_labels_dir, txt_file_name)
        
        # Ghi annotations cho ảnh này
        annotations = annotations_by_image.get(image_id, [])
        
        with open(txt_file_path, 'w', encoding='utf-8') as f:
            for ann in annotations:
                line = f"{ann['class']} {ann['x']:.6f} {ann['y']:.6f} {ann['w']:.6f} {ann['h']:.6f}\n"
                f.write(line)
        
        if annotations:
            txt_count += 1
        else:
            empty_count += 1
        
        # Copy ảnh nếu cần
        if copy_images_to:
            src_path = os.path.join(images_dir, file_name)
            dst_path = os.path.join(copy_images_to, file_name)
            
            if os.path.exists(src_path):
                # Sử dụng shutil để copy
                import shutil
                shutil.copy2(src_path, dst_path)
            else:
                print(f"⚠️  Warning: ảnh không tìm thấy: {src_path}")
    
    print(f"✓ Đã ghi {txt_count} file .txt (có annotation)")
    print(f"  ({empty_count} ảnh không có annotation, tạo file rỗng)")
    
    return {
        'total_images': len(images_dict),
        'images_with_annotations': txt_count,
        'empty_images': empty_count,
        'categories': {idx: name for cat_id, name in cat_id_to_name.items() 
                      for idx, _ in [(categories_dict[cat_id], None)]},
    }


def validate_conversion(labels_dir, images_dir):
    """
    Kiểm tra tính hợp lệ của dữ liệu đã chuyển đổi.
    """
    print(f"\n✅ Đang validate dữ liệu...")
    
    txt_files = list(Path(labels_dir).glob('*.txt'))
    print(f"  - Số file .txt: {len(txt_files)}")
    
    errors = []
    stats = {
        'total_annotations': 0,
        'valid_lines': 0,
        'invalid_lines': 0,
        'out_of_bounds': 0,
    }
    
    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                stats['total_annotations'] += 1
                parts = line.split()
                
                if len(parts) != 5:
                    errors.append(f"{txt_file.name}:{line_num} - Số field không đúng (cần 5, có {len(parts)})")
                    stats['invalid_lines'] += 1
                    continue
                
                try:
                    class_idx = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                    
                    # Kiểm tra giá trị normalized
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        errors.append(f"{txt_file.name}:{line_num} - Giá trị ngoài [0,1]: {line}")
                        stats['out_of_bounds'] += 1
                    else:
                        stats['valid_lines'] += 1
                
                except ValueError:
                    errors.append(f"{txt_file.name}:{line_num} - Giá trị không phải số: {line}")
                    stats['invalid_lines'] += 1
    
    print(f"  - Tổng annotation: {stats['total_annotations']}")
    print(f"  - Valid lines: {stats['valid_lines']}")
    if stats['invalid_lines'] > 0:
        print(f"  - Invalid lines: {stats['invalid_lines']} ⚠️")
    if stats['out_of_bounds'] > 0:
        print(f"  - Out of bounds: {stats['out_of_bounds']} ⚠️")
    
    if errors:
        print(f"\n❌ Tìm thấy {len(errors)} lỗi:")
        for error in errors[:10]:  # Chỉ hiển thị 10 lỗi đầu
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... và {len(errors) - 10} lỗi khác")
    else:
        print("  ✓ Không tìm thấy lỗi")
    
    return stats


if __name__ == '__main__':
    import sys
    
    # Cấu hình
    BASE_DIR = Path(__file__).parent
    
    COCO_JSON = BASE_DIR / 'data' / 'dataexample' / 'labels' / 'merged_coco.json'
    IMAGES_DIR = BASE_DIR / 'data' / 'dataexample' / 'images'
    OUTPUT_DIR = BASE_DIR / 'datasets' / 'pig_coco'
    OUTPUT_LABELS = OUTPUT_DIR / 'labels'
    OUTPUT_IMAGES = OUTPUT_DIR / 'images'
    
    print("=" * 60)
    print("🔄 Chuyển đổi COCO JSON -> YOLOv5 TXT Format")
    print("=" * 60)
    print(f"\nCấu hình:")
    print(f"  COCO JSON: {COCO_JSON}")
    print(f"  Images:    {IMAGES_DIR}")
    print(f"  Output:    {OUTPUT_DIR}")
    
    # Kiểm tra file input
    if not COCO_JSON.exists():
        print(f"\n❌ Lỗi: File không tìm thấy: {COCO_JSON}")
        sys.exit(1)
    
    if not IMAGES_DIR.exists():
        print(f"\n❌ Lỗi: Thư mục không tìm thấy: {IMAGES_DIR}")
        sys.exit(1)
    
    # Thực hiện chuyển đổi
    result = convert_coco_to_yolo(
        str(COCO_JSON),
        str(IMAGES_DIR),
        str(OUTPUT_LABELS),
        copy_images_to=str(OUTPUT_IMAGES)
    )
    
    # Validate
    validate_conversion(str(OUTPUT_LABELS), str(IMAGES_DIR))
    
    print("\n" + "=" * 60)
    print("✅ Chuyển đổi hoàn tất!")
    print("=" * 60)
    print(f"\nOutput locations:")
    print(f"  - Ảnh: {OUTPUT_IMAGES}")
    print(f"  - Labels: {OUTPUT_LABELS}")
    print(f"\nBước tiếp theo: Tạo file data/pig_coco.yaml")
