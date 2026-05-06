import argparse
import random
import shutil
from pathlib import Path


def split_dataset(src_dir: Path, train_ratio: float, copy: bool):
    images_dir = src_dir / "images"
    labels_dir = src_dir / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError("Thư mục images hoặc labels không tồn tại trong " + str(src_dir))

    image_files = sorted(
        [
            p
            for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
    )
    if not image_files:
        raise ValueError("Không tìm thấy file ảnh nào trong " + str(images_dir))

    random.shuffle(image_files)
    split_index = int(len(image_files) * train_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    for subset in ["train", "val"]:
        (src_dir / subset / "images").mkdir(parents=True, exist_ok=True)
        (src_dir / subset / "labels").mkdir(parents=True, exist_ok=True)

    def copy_or_move(files, subset):
        for img_path in files:
            label_path = labels_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                raise FileNotFoundError(f"Label không tồn tại cho ảnh: {img_path.name}")

            dest_img = src_dir / subset / "images" / img_path.name
            dest_label = src_dir / subset / "labels" / label_path.name
            if copy:
                shutil.copy2(img_path, dest_img)
                shutil.copy2(label_path, dest_label)
            else:
                shutil.move(img_path, dest_img)
                shutil.move(label_path, dest_label)

    copy_or_move(train_files, "train")
    copy_or_move(val_files, "val")
    print(f"Đã tạo dataset split: {len(train_files)} ảnh train, {len(val_files)} ảnh val")
    print("Thư mục mới:")
    print(f"  {src_dir}/train/images")
    print(f"  {src_dir}/train/labels")
    print(f"  {src_dir}/val/images")
    print(f"  {src_dir}/val/labels")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataexample into train/val for YOLOv5")
    parser.add_argument("--src", type=Path, default=Path("data/dataexample"), help="Thư mục dataset nguồn")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Tỷ lệ ảnh train")
    parser.add_argument("--copy", action="store_true", help="Copy file thay vì di chuyển")
    args = parser.parse_args()

    split_dataset(args.src, args.train_ratio, args.copy)
