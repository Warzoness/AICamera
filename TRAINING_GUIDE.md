# 🎓 YOLOv5 Training Feature - Hướng Dẫn Sử Dụng

## 📋 Tổng Quan

Hệ thống web này cung cấp một giao diện hoàn chỉnh để training YOLOv5 model với khả năng:
- ✅ Upload ảnh training
- ✅ Upload file COCO annotations
- ✅ Theo dõi tiến trình training real-time
- ✅ Logging chi tiết về:
  - 🔴 Ảnh có độ nhận diện thấp (Low Detection)
  - 🟡 Các false positives trên ảnh
- ✅ Lưu trữ kết quả training trong thư mục riêng

---

## 🚀 Cách Sử Dụng

### Bước 1: Chuẩn Bị Dữ Liệu

#### Chuẩn Bị Ảnh Training
- Chuẩn bị các ảnh training (JPG, PNG, BMP, WEBP)
- Ảnh nên có độ phân giải đủ cao (tối thiểu 416x416 pixels)
- Ảnh nên bao gồm các đối tượng cần phát hiện

#### Chuẩn Bị File COCO Annotations
File COCO JSON phải có cấu trúc sau:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "width": 640,
      "height": 480
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": 1234,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "pig",
      "supercategory": "animal"
    }
  ]
}
```

### Bước 2: Truy Cập Giao Diện Training

1. Mở web interface: `http://localhost:5000`
2. Nhấp vào tab **"🎓 Training Model"**

### Bước 3: Upload Dữ Liệu

#### 📸 Section 1: Upload Ảnh Training
1. Kéo thả ảnh hoặc click để chọn nhiều ảnh
2. Hệ thống sẽ hiển thị danh sách ảnh đã upload
3. Kiểm tra số lượng ảnh được upload thành công

#### 📋 Section 2: Upload File COCO
1. Kéo thả file COCO JSON hoặc click để chọn
2. Hệ thống sẽ validate cấu trúc JSON
3. Nếu thành công, sẽ hiển thị:
   - Số lượng ảnh
   - Số lượng annotations
   - Danh sách các classes

### Bước 4: Cấu Hình Than Số Training

| Than Số | Mặc Định | Phạm Vi | Mô Tả |
|---------|---------|--------|-------|
| **Epochs** | 10 | 1-100 | Số vòng training |
| **Batch Size** | 16 | 1-128 | Số ảnh xử lý mỗi lần |
| **Image Size** | 640 | 416/512/640/800 | Kích thước ảnh input |
| **Confidence Threshold** | 0.5 | 0-1 | Ngưỡng confidence để log |

**Khuyến Cáo:**
- **Epochs**: 20-50 cho tập dữ liệu nhỏ, 50-100+ cho tập lớn
- **Batch Size**: 16-32 cho GPU có 4GB+, 8 cho GPU có 2GB
- **Image Size**: 640 là cân bằng tốt giữa tốc độ và chính xác
- **Confidence Threshold**: 0.3-0.5 để catch low confidence predictions

### Bước 5: Bắt Đầu Training

1. Nhấp nút **"🚀 Bắt Đầu Training"**
2. Hệ thống sẽ:
   - Validate dữ liệu
   - Khởi tạo training logger
   - Bắt đầu training trong background thread

### Bước 6: Theo Dõi Tiến Trình

Trong quá trình training, bạn sẽ thấy:
- **Progress Bar**: % hoàn thành
- **Status Message**: Thông tin epoch và ảnh hiện tại
- **Real-time Updates**: Cập nhật mỗi 1 giây

### Bước 7: Xem Kết Quả

Sau khi training hoàn tất, sẽ hiển thị:

#### 📊 Thống Kê Chung
- **Ảnh đã xử lý**: Tổng số ảnh training
- **Độ chính xác TB**: Trung bình confidence
- **Ảnh detection thấp**: Số ảnh có detection < threshold
- **False Positives**: Số ảnh có false positive detections

#### 🔴 Ảnh Detection Thấp
Những ảnh mà model phát hiện object nhưng với confidence thấp:
- Ảnh được annotate với hình chữ nhật **màu đỏ**
- Kèm theo confidence score
- Click để xem full size

**Ý Nghĩa:**
- Model khó nhận dạng những object này
- Cần thêm dữ liệu hoặc improve model

#### 🟡 False Positives
Những vùng mà model phát hiện object nhưng không có trong ground truth:
- Ảnh được annotate với hình chữ nhật **màu vàng**
- Thường do model overfitting hoặc dữ liệu thiếu

**Ý Nghĩa:**
- Model phát hiện sai những vùng
- Cần fine-tune model hoặc review training data

---

## 📁 Cấu Trúc Folder Training Logs

Mỗi session training tạo ra một folder với cấu trúc:

```
training_logs/
└── {YYYYMMDD_HHMMSS}/           # Timestamp của training session
    ├── training_summary.json         # Tóm tắt training
    ├── training_log.txt              # Log chi tiết
    ├── low_detection_report.json     # Report low detection
    ├── false_positives_report.json   # Report false positives
    ├── low_detection_images/
    │   ├── low_conf_image1.jpg       # Ảnh có detection thấp
    │   ├── low_conf_image2.jpg
    │   └── ...
    ├── false_positives/
    │   ├── false_pos_image1.jpg      # Ảnh có false positives
    │   ├── false_pos_image2.jpg
    │   └── ...
    └── detailed_logs/
        └── training_log.txt          # Log chi tiết
```

---

## 📊 Hiểu File JSON Output

### training_summary.json
```json
{
  "training_name": "20240515_143022",
  "start_time": "2024-05-15T14:30:22.123456",
  "end_time": "2024-05-15T16:45:30.987654",
  "total_images": 500,
  "low_detection_count": 45,
  "false_positives_count": 23,
  "average_confidence": 0.87,
  "status": "completed"
}
```

### low_detection_report.json
```json
[
  {
    "original_image": "training_data/images/pig1.jpg",
    "saved_image": "training_logs/20240515_143022/low_detection_images/low_conf_pig1.jpg",
    "timestamp": "2024-05-15T14:35:10.123456",
    "low_confidence_detections": [
      {
        "class": "pig",
        "confidence": 0.42,
        "bbox": [100, 150, 300, 400]
      }
    ],
    "threshold": 0.5
  }
]
```

---

## 🔧 Troubleshooting

### ❌ "Không có ảnh training"
- Kiểm tra xem ảnh có được upload thành công không
- Kiểm tra định dạng ảnh (JPG, PNG, BMP, WEBP)
- Refresh trang và thử lại

### ❌ "File COCO không hợp lệ"
- Kiểm tra cấu trúc JSON (phải có "images" và "annotations")
- Validate JSON trên [jsonlint.com](https://jsonlint.com)
- Kiểm tra image names khớp giữa ảnh thực và file COCO

### ❌ Training bị hang
- Kiểm tra console của Flask (port 5000)
- Có thể là model inference bị hang
- Thử giảm batch size hoặc image size
- Restart Flask server

### ⚠️ Ảnh không hiển thị trong kết quả
- Kiểm tra browser console (F12) có lỗi không
- Kiểm tra file tồn tại trong thư mục training_logs
- Kiểm tra quyền file và folder

---

## 💡 Best Practices

### 1. Chuẩn Bị Dữ Liệu
- ✅ Sử dụng ảnh diverse (khác góc, ánh sáng, background)
- ✅ Cân bằng số lượng ảnh theo class
- ✅ Validate annotations trước khi training

### 2. Cấu Hình Training
- ✅ Bắt đầu với epochs nhỏ (10-20) để test
- ✅ Sử dụng batch size phù hợp với GPU memory
- ✅ Confidence threshold 0.3-0.5 để catch issues

### 3. Đánh Giá Kết Quả
- ✅ Review ảnh detection thấp để improve
- ✅ Kiểm tra false positives để fix overfitting
- ✅ Giữ logs để tái hiện kết quả

### 4. Cải Thiện Model
- ✅ Nếu nhiều low detection: thêm dữ liệu tương tự
- ✅ Nếu nhiều false positives: reduce epochs hoặc adjust data
- ✅ Sử dụng logs để làm data augmentation

---

## 📞 Support

Khi gặp vấn đề:
1. Kiểm tra Flask console logs
2. Xem training_logs/{session_id}/detailed_logs/training_log.txt
3. Kiểm tra file trong training_logs folder structure

---

## 📈 Workflow Khuyến Cáo

```
1. Chuẩn Bị Dữ Liệu (10%)
   ↓
2. Upload Ảnh & COCO (5%)
   ↓
3. Cấu Hình Than Số (5%)
   ↓
4. Training (60%)
   ↓
5. Đánh Giá Kết Quả (10%)
   ├─→ Review Low Detection Images
   ├─→ Review False Positives
   └─→ Adjust Data & Retrain
   ↓
6. Production Ready (10%)
```

---

Chúc bạn training model thành công! 🚀

