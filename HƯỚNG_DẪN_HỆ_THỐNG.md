# 🎯 HƯỚNG DẪN HỆ THỐNG YOLOV5 - AI PHÁT HIỆN VẬT THỂ

> **Tổng hợp từ tài liệu hệ thống | Ngày: 2026**

---

## 📖 MỤC LỤC

1. [Cấu trúc hệ thống](#cấu-trúc-hệ-thống)
2. [Chuẩn bị dữ liệu](#chuẩn-bị-dữ-liệu)
3. [Cách feed dữ liệu](#cách-feed-dữ-liệu)
4. [Cách hệ thống vận hành](#cách-hệ-thống-vận-hành)
5. [Kết quả trả về](#kết-quả-trả-về)
6. [Ví dụ thực tế](#ví-dụ-thực-tế)

---

## 🏗️ CẤU TRÚC HỆ THỐNG

### 1. Cấu trúc thư mục dự án

```
yolov5/
├── train.py                          # Script train model
├── detect.py                         # Script phát hiện vật thể
├── yolov5_web_demo.py                # Web interface Flask
├── convert_coco_to_yolo.py           # Tool chuyển đổi COCO → YOLO format
├── requirements.txt                  # Thư viện Python cần thiết
├── models/                           # Định nghĩa các kiến trúc model
│   ├── yolov5s.yaml                 # Model nhỏ (small)
│   ├── yolov5m.yaml                 # Model trung bình (medium)
│   └── ...
├── data/                             # Cấu hình dataset
│   ├── pig_coco.yaml                # Config cho dataset lợn (COCO format)
│   ├── dataexample/
│   │   ├── images/                  # Ảnh gốc (21 ảnh lợn)
│   │   └── labels/
│   │       └── merged_coco.json     # Annotation COCO JSON
│   └── hyps/                         # Cấu hình hyperparameter
├── datasets/
│   └── pig_coco/                     # Dataset sau khi chuyển đổi
│       ├── images/
│       │   ├── train/               # Ảnh train
│       │   └── val/                 # Ảnh validation
│       └── labels/
│           ├── train/               # Labels train (format YOLO .txt)
│           └── val/                 # Labels validation (format YOLO .txt)
├── runs/
│   └── train/
│       └── exp3/
│           └── weights/
│               ├── best.pt          # Model tốt nhất (dùng cho predict)
│               └── last.pt          # Model checkpoint cuối cùng
├── outputs/                          # Ảnh/video kết quả từ web demo
├── uploads/                          # File upload tạm thời
├── utils/                            # Hàm tiện ích
│   ├── dataloaders.py               # Load dữ liệu
│   ├── general.py                   # Hàm chung
│   └── ...
└── templates/
    └── index.html                   # Giao diện web Flask
```

### 2. Kiến trúc YOLOv5

**YOLOv5 là model phát hiện vật thể real-time với cấu trúc:**
- **Backbone**: Trích xuất đặc trưng từ ảnh
- **Neck**: Kết hợp đặc trưng ở các tầng khác nhau
- **Head**: Dự đoán bounding box và class

**Các kích thước model:**
| Model | Than số | Tốc độ | Độ chính xác |
|-------|---------|--------|--------------|
| yolov5n | 1.9M | Rất nhanh | Trung bình |
| yolov5s | 7.2M | Nhanh | Tốt |
| yolov5m | 21.2M | Trung bình | Rất tốt |
| yolov5l | 46.5M | Chậm | Xuất sắc |
| yolov5x | 86.7M | Rất chậm | Tốt nhất |

---

## 📦 CHUẨN BỊ DỮ LIỆU

### 1. Cấu trúc dữ liệu bắt buộc

Hệ thống yêu cầu dữ liệu được tổ chức như sau:

```
data_folder/
├── images/                           # Ảnh gốc
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
└── labels/                           # Annotation COCO JSON
    └── annotations.json              # Định dạng COCO
```

### 2. Định dạng COCO JSON

COCO JSON là format chuẩn công nghiệp. Cấu trúc cơ bản:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "001.jpg",
      "width": 640,
      "height": 480
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "pig"
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 50, 200, 150],
      "area": 30000,
      "iscrowd": 0
    }
  ]
}
```

**Chi tiết bbox COCO:**
- `[x, y, width, height]` - tọa độ pixel (tuyệt đối)
- `x, y` - góc trái trên của bounding box
- `width, height` - kích thước box

### 3. Chuẩn bị ảnh

**Yêu cầu:**
- Định dạng: JPG, PNG, BMP, GIF, WEBP
- Kích thước: ít nhất 640x640 (hoặc lớn hơn)
- Số lượng: tối thiểu 20-50 ảnh cho mỗi class (càng nhiều càng tốt)
- Đa dạng: các điều kiện ánh sáng, góc quay, vị trí khác nhau

**Lời khuyên:**
- Ảnh nên chứa chủ thể rõ ràng
- Cân bằng các class (số ảnh trong mỗi class không chênh lệch quá lớn)
- Tránh ảnh bị mờ hoặc quá đen/sáng

### 4. Tách train/val (nếu cần)

```
data_folder/
├── images/
│   ├── train/           # 80% ảnh
│   │   ├── 001.jpg
│   │   └── ...
│   └── val/             # 20% ảnh
│       ├── 101.jpg
│       └── ...
└── labels/
    └── annotations.json # Một file COCO duy nhất hoặc tách thành train/val
```

---

## 🔌 CÁCH FEED DỮ LIỆU

### 1. Bước 1: Đặt dữ liệu vào thư mục

Tạo thư mục dữ liệu theo cấu trúc bắt buộc:

```bash
# Ví dụ tạo thư mục
mkdir -p my_data/images
mkdir -p my_data/labels

# Copy ảnh vào my_data/images/
cp /path/to/ảnh/*.jpg my_data/images/

# Copy annotation vào my_data/labels/
cp /path/to/annotations.json my_data/labels/
```

### 2. Bước 2: Chuyển đổi COCO JSON → YOLOv5 TXT format

Hệ thống nội bộ sử dụng format TXT thay vì JSON để tốc độ cao hơn (6x nhanh hơn).

**Sử dụng script chuyển đổi:**

```bash
python convert_coco_to_yolo.py \
  --coco_json my_data/labels/annotations.json \
  --images_dir my_data/images \
  --output_labels_dir my_data/labels_yolo
```

**Kết quả tạo ra:**

```
my_data/labels_yolo/
├── 001.txt
├── 002.txt
├── ...
```

**Format file `.txt`:**

```
# Mỗi dòng là một object
0 0.512 0.422 0.234 0.318
1 0.732 0.201 0.160 0.210

# Format: class_id x_center_norm y_center_norm width_norm height_norm
# - class_id: chỉ số class (0-based)
# - x_center_norm, y_center_norm, width_norm, height_norm: giá trị chuẩn hóa [0, 1]
```

### 3. Bước 3: Tạo file cấu hình YAML

Tạo file `data/my_dataset.yaml`:

```yaml
# Đường dẫn dữ liệu
path: ../datasets/my_dataset          # root path
train: images/train                    # train ảnh (relative to path)
val: images/val                        # val ảnh (relative to path)

# Số class và tên class
nc: 2                                  # số class
names: ['pig', 'cow']                  # tên class
```

**Ví dụ file đầy đủ:**

```yaml
path: ../datasets/pig_dataset
train: images/train
val: images/val

nc: 1
names: ['pig']
```

---

## ⚙️ CÁCH HỆ THỐNG VẬN HÀNH

### 1. Training (Huấn luyện model)

**Lệnh cơ bản:**

```bash
python train.py \
  --data data/my_dataset.yaml \
  --weights yolov5s.pt \
  --img 640 \
  --batch 16 \
  --epochs 100 \
  --device 0
```

**Giải thích than số:**

| Than số | Ý nghĩa | Gợi ý |
|---------|---------|-------|
| `--data` | Đường dẫn file YAML cấu hình | data/my_dataset.yaml |
| `--weights` | Pre-trained model (transfer learning) | yolov5s.pt (khuyến nghị) hoặc '' (train từ 0) |
| `--img` | Kích thước input ảnh | 640 (tiêu chuẩn), 416, 512 |
| `--batch` | Batch size (tùy vào GPU memory) | 16, 32, 64 |
| `--epochs` | Số vòng training | 50-200 |
| `--device` | GPU device | 0 (GPU 1) hoặc cpu |
| `--patience` | Early stopping (dừng nếu không cải thiện) | 20 |

**Quá trình training:**

1. **Load dataset** → Đọc ảnh + labels từ thư mục
2. **Data augmentation** → Xoay, co dãn, thay đổi độ sáng ảnh
3. **Forward pass** → Đưa ảnh qua model
4. **Tính loss** → So sánh dự đoán với ground truth
5. **Backward pass** → Cập nhật weights
6. **Validation** → Kiểm tra trên tập val mỗi epoch
7. **Lưu model** → Lưu best.pt (mAP cao nhất)

**Output training:**

```
runs/train/exp{N}/
├── weights/
│   ├── best.pt                       # Model tốt nhất (dùng để predict)
│   ├── last.pt                       # Model cuối cùng
│   └── epoch0.pt
├── results.csv                       # Metrics qua mỗi epoch
├── confusion_matrix.png              # Ma trận confusion
├── results.png                       # Biểu đồ training
└── ...
```

### 2. Inference (Phát hiện vật thể)

#### A. Sử dụng detect.py (CLI)

```bash
# Phát hiện trên ảnh
python detect.py \
  --weights runs/train/exp3/weights/best.pt \
  --source my_image.jpg \
  --conf 0.25 \
  --iou 0.45 \
  --img 640

# Phát hiện trên video
python detect.py \
  --weights runs/train/exp3/weights/best.pt \
  --source my_video.mp4 \
  --conf 0.25

# Phát hiện trên webcam
python detect.py \
  --weights runs/train/exp3/weights/best.pt \
  --source 0

# Phát hiện trên thư mục ảnh
python detect.py \
  --weights runs/train/exp3/weights/best.pt \
  --source path/to/images/
```

**Than số detect:**

| Than số | Ý nghĩa |
|---------|---------|
| `--weights` | Đường dẫn model `.pt` |
| `--source` | Nguồn input (ảnh, video, webcam, thư mục) |
| `--conf` | Confidence threshold (0.0-1.0) - chỉ lấy detection có tin cậy cao hơn |
| `--iou` | NMS IOU threshold - loại bỏ box trùng lặp |
| `--img` | Kích thước inference |
| `--save-txt` | Lưu kết quả dưới dạng `.txt` |
| `--save-crop` | Crop và lưu object phát hiện |

#### B. Sử dụng PyTorch Hub (Python code)

```python
import torch
from PIL import Image

# Load model
model = torch.hub.load(".", "custom", path="runs/train/exp3/weights/best.pt", source="local")
model.conf = 0.25  # Confidence threshold

# Inference trên ảnh
img = Image.open("my_image.jpg")
results = model(img)

# Xem kết quả
results.print()
results.show()
results.save()
```

#### C. Sử dụng Web Demo Flask

```bash
# Khởi động web server
python yolov5_web_demo.py

# Mở browser
# http://localhost:5000
```

**Tính năng:**
- Upload ảnh, video
- Phát hiện tự động
- Hiển thị kết quả trực quan
- Download ảnh/video có annotation

---

## 📊 KẾT QUẢ TRẢ VỀ

### 1. Định dạng kết quả cơ bản

**Khi chạy detect.py:**

```
runs/detect/exp/
├── labels/                           # Nếu dùng --save-txt
│   ├── my_image.txt
│   └── ...
├── crops/                            # Nếu dùng --save-crop
│   ├── pig/
│   │   ├── 001.jpg
│   │   └── ...
│   └── cow/
└── my_image.jpg                      # Ảnh có box và label
```

### 2. Format kết quả TXT (nếu --save-txt)

```txt
# my_image.txt
pig 0.512 0.422 0.234 0.318 0.95
pig 0.732 0.201 0.160 0.210 0.87
cow 0.234 0.567 0.123 0.234 0.78

# Format: class x_center y_center width height confidence
# Giá trị: tọa độ chuẩn hóa [0, 1], confidence [0, 1]
```

### 3. Pandas DataFrame (Python API)

```python

import torch

model = torch.hub.load(".", "custom", path="runs/train/exp3/weights/best.pt", source="local")

# Inference
results = model("my_image.jpg")

# Lấy kết quả dưới dạng dataframe
df = results.pandas().xyxy[0]

# Cột trong dataframe:
# - xmin, ymin, xmax, ymax: tọa độ pixel (tuyệt đối)
# - confidence: độ tin cậy
# - class: chỉ số class
# - name: tên class

print(df)
#    xmin   ymin   xmax   ymax  confidence  class  name
# 0   100    50    300   200        0.95      0   pig
# 1   400   150    550   350        0.87      0   pig
# 2   200   400    380   600        0.78      1   cow
```

### 4. Web Demo JSON response

**Khi upload ảnh qua web:**

```json
{
  "success": true,
  "output_image": "/outputs/output_1234567890_image.jpg",
  "detections": [
    {
      "class": "pig",
      "confidence": 0.95,
      "bbox": [100, 50, 300, 200]
    },
    {
      "class": "pig",
      "confidence": 0.87,
      "bbox": [400, 150, 550, 350]
    }
  ],
  "total_objects": 2
}
```

### 5. Metrics sau training

**File results.csv:**

```
epoch,train/box_loss,train/obj_loss,train/cls_loss,metrics/precision,metrics/recall,metrics/mAP_0.5,metrics/mAP_0.5:0.95,val/box_loss,val/obj_loss,val/cls_loss
0,0.5231,0.4123,0.1234,0.92,0.89,0.87,0.65,0.5421,0.4234,0.1345
1,0.4892,0.3892,0.1123,0.93,0.90,0.88,0.66,0.5123,0.4012,0.1234
```

**Ý nghĩa metrics:**
- **mAP@0.5**: Độ chính xác trung bình ở IOU=0.5
- **mAP@0.5:0.95**: Độ chính xác trung bình ở IOU=0.5:0.95
- **Precision**: Tỷ lệ detection đúng / tổng detection
- **Recall**: Tỷ lệ object bị phát hiện / tổng object

---

## 💡 VÍ DỤ THỰC TẾ

### Ví dụ 1: Train model phát hiện lợn từ COCO JSON

**Bước 1: Chuẩn bị dữ liệu**

```bash
# Dữ liệu đã sẵn trong:
# data/dataexample/images/    (21 ảnh lợn)
# data/dataexample/labels/merged_coco.json
```

**Bước 2: Chuyển đổi sang YOLOv5 format**

```bash
python convert_coco_to_yolo.py \
  --coco_json data/dataexample/labels/merged_coco.json \
  --images_dir data/dataexample/images \
  --output_labels_dir datasets/pig_coco/labels
```

**Bước 3: Tạo file YAML**

`data/pig_coco.yaml`:
```yaml
path: ../datasets/pig_coco
train: images
val: images
nc: 1
names: ['pig']
```

**Bước 4: Train model**

```bash
python train.py \
  --data data/pig_coco.yaml \
  --weights yolov5s.pt \
  --img 640 \
  --batch 16 \
  --epochs 50 \
  --device 0
```

**Bước 5: Test model**

```bash
python detect.py \
  --weights runs/train/exp3/weights/best.pt \
  --source data/dataexample/images \
  --conf 0.25
```

### Ví dụ 2: Sử dụng Web Demo

```bash
# Khởi động web
python yolov5_web_demo.py

# Browser:
# http://localhost:5000
# → Upload ảnh
# → Xem kết quả
```

### Ví dụ 3: Inference trên video

```bash
# Chạy detection trên video
python detect.py \
  --weights runs/train/exp3/weights/best.pt \
  --source uploads/video.mp4 \
  --conf 0.25 \
  --save-txt

# Kết quả: runs/detect/exp/video.mp4 (video có box)
```

### Ví dụ 4: Sử dụng Python API

```python

import torch

# Load model
model = torch.hub.load(".", "custom", path="runs/train/exp3/weights/best.pt", source="local")

# Config
model.conf = 0.5
model.iou = 0.45

# Inference
results = model("my_image.jpg")

# Lấy predictions
predictions = results.pandas().xyxy[0]

# Xử lý kết quả
for idx, row in predictions.iterrows():
    xmin, ymin, xmax, ymax = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
    confidence = row["confidence"]
    class_name = row["name"]

    print(f"Phát hiện: {class_name} - Tin cậy: {confidence:.2f}")
    print(f"  Tọa độ: ({xmin}, {ymin}) -> ({xmax}, {ymax})")
```

---

## ⚙️ CẤU HÌNH NÂNG CAO

### 1. Hyperparameters quan trọng

| Than số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `--batch` | 16 | Kích thước batch (phụ thuộc GPU memory) |
| `--epochs` | 100 | Số vòng training |
| `--lr0` | 0.01 | Learning rate ban đầu |
| `--patience` | 20 | Early stopping (dừng sau n epoch không cải thiện) |
| `--conf` | 0.25 | Confidence threshold |
| `--iou` | 0.45 | IOU threshold cho NMS |
| `--augment` | True | Áp dụng data augmentation |
| `--mosaic` | 1.0 | Mosaic augmentation ratio |

### 2. GPU Memory

**Yêu cầu memory (GB) theo model:**
- yolov5n: 1.5 GB
- yolov5s: 3.5 GB
- yolov5m: 8 GB
- yolov5l: 16 GB
- yolov5x: 24 GB

**Điều chỉnh batch size nếu bị OOM:**

```bash
# Nếu GPU memory không đủ
python train.py ... --batch 8 # Giảm batch size
```

### 3. Model architecture

```
YOLOv5s:
├── Backbone (CSPDarknet)
│   ├── Conv 3x3, stride 2, 32 channels
│   ├── CSPBottleneck (1 residual block)
│   ├── Conv 3x3, stride 2, 64 channels
│   ├── CSPBottleneck (3 residual blocks)
│   └── ...
├── Neck (PANet)
│   ├── Upsample + concat (multi-scale fusion)
│   └── ...
└── Head (Detect)
    ├── 3 detection layers (8x, 16x, 32x stride)
    ├── Anchor boxes
    └── Output: (batch, 3 anchors × 85 values, H/stride, W/stride)
```

---

## 🔧 TROUBLESHOOTING

### Problem 1: CUDA out of memory

**Giải pháp:**
```bash
# Giảm batch size
python train.py ... --batch 8

# Hoặc giảm kích thước input
python train.py ... --img 416

# Hoặc dùng CPU (chậm)
python train.py ... --device cpu
```

### Problem 2: Model không hội tụ (loss không giảm)

**Giải pháp:**
- Tăng learning rate: `--lr0 0.02`
- Tăng epochs: `--epochs 200`
- Kiểm tra chất lượng dữ liệu
- Tăng augmentation: `--mosaic 1.0`

### Problem 3: Detection kết quả kém (false positives)

**Giải pháp:**
- Tăng confidence threshold: `--conf 0.5`
- Giảm IOU threshold: `--iou 0.35`
- Train lâu hơn
- Thêm dữ liệu training

### Problem 4: ImportError PyTorch

**Giải pháp:**
```bash
pip install -r requirements.txt
# Hoặc cài đặc thứ viện CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 📚 TÀI NGUYÊN THAN KHẢO

- **YOLOv5 Official**: https://github.com/ultralytics/yolov5
- **Ultralytics Docs**: https://docs.ultralytics.com/yolov5/
- **COCO Dataset Format**: https://cocodataset.org/

---

## 📝 GHI CHÚ QUAN TRỌNG

1. **Format dữ liệu:**
   - Input: Thư mục với `images/` và `labels/` (COCO JSON)
   - Internal: Chuyển đổi sang YOLO `.txt` format (nhanh 6x)
   - Output: Bounding box + confidence scores

2. **Training best practices:**
   - Sử dụng pre-trained model (yolov5s.pt) cho transfer learning
   - Tối thiểu 20-50 ảnh per class
   - Tách train/val ratio 80/20
   - Monitor validation metrics

3. **Inference best practices:**
   - Điều chỉnh confidence threshold theo nhu cầu
   - Sử dụng confidence threshold cao (0.5+) để giảm false positives
   - Kiểm tra IoU threshold để loại bỏ box trùng lặp

4. **Performance:**
   - YOLOv5s: Nhanh + chính xác (khuyên dùng)
   - YOLOv5m: Nhanh hơn v5s, chính xác hơn
   - YOLOv5l/x: Chậm, nhưng độ chính xác cao nhất

---

**Phiên bản:** 1.0  
**Cập nhật:** 2026  
**Tác giả:** Ultralytics + Hệ thống AI Camera
