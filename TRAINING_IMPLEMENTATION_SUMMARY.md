# 📋 Tóm Tắt Hệ Thống Training YOLOv5

## 🎯 Mục Đích
Tạo hệ thống logging chi tiết và giao diện training để:
- Theo dõi các ảnh có độ nhận diện thấp
- Phát hiện và ghi lại các vùng bị nhiễu false positives
- Lưu trữ kết quả training với metadata đầy đủ

---

## 📁 Những File/Folder Được Tạo

### 1. **training_logs/** - Thư mục chứa logs
```
training_logs/
├── {YYYYMMDD_HHMMSS}/
│   ├── training_summary.json
│   ├── low_detection_report.json
│   ├── false_positives_report.json
│   ├── training_log.txt
│   ├── low_detection_images/
│   ├── false_positives/
│   └── detailed_logs/
```

### 2. **train_logger.py** - Module logging
Module này cung cấp class `TrainingLogger` với các tính năng:

#### Methods:
- `__init__(training_name)` - Khởi tạo logger
- `log(message)` - Ghi log message
- `record_low_detection()` - Ghi lại ảnh/vùng detection thấp
- `record_false_positives()` - Ghi lại false positive detections
- `record_statistics()` - Ghi statistics chung
- `finalize()` - Hoàn tất training

#### Features:
- 🔴 Vẽ detection thấp (confidence-low) bằng hình chữ nhật **đỏ**
- 🟡 Vẽ false positives bằng hình chữ nhật **vàng**
- 📊 Tính IoU giữa predictions và ground truth
- 💾 Lưu JSON reports chi tiết
- 📝 Logging cấp detail

### 3. **yolov5_web_demo.py** - Backend Updates
#### Các Endpoints Mới:
- `POST /upload_train_images` - Upload ảnh training
- `POST /upload_coco_file` - Upload file COCO annotations
- `POST /start_training` - Bắt đầu training
- `GET /training_status` - Lấy trạng thái training
- `GET /get_training_logs/<training_id>` - Lấy logs
- `GET /training_logs/<training_id>/<image_type>/<filename>` - Serve ảnh logs

#### Global State:
```python
training_state = {
    'is_training': False,
    'current_training': None,
    'progress': 0,
    'message': ''
}
```

#### Config:
```python
app.config['TRAIN_IMAGES_FOLDER'] = 'training_data/images'
app.config['TRAIN_LABELS_FOLDER'] = 'training_data/labels'
```

### 4. **index.html** - Frontend Updates
#### Tab Mới: "🎓 Training Model"
Giao diện 2-column:

**Column 1: Upload Ảnh Training**
- Upload area với drag-drop
- Danh sách ảnh uploaded
- Counter số ảnh

**Column 2: Upload File COCO**
- Upload area cho JSON file
- Validate file structure
- Hiển thị metadata (images count, annotations count, categories)

**Tham Số Training:**
- Epochs (input number)
- Batch Size (input number)
- Image Size (dropdown: 416/512/640/800)
- Confidence Threshold (slider 0-1)

**Training Progress:**
- Progress bar với % completion
- Status message real-time

**Kết Quả Training:**
- Stats cards (4 metrics)
- Log display area
- Ảnh detection thấp (grid)
- Ảnh false positives (grid)
- Download logs & Open folder buttons

### 5. **TRAINING_GUIDE.md** - Hướng Dẫn Sử Dụng
Tài liệu hoàn chỉnh với:
- 📋 Tổng quan
- 🚀 Hướng dẫn sử dụng từng bước
- 📁 Cấu trúc folder output
- 📊 Format JSON files
- 🔧 Troubleshooting
- 💡 Best practices

---

## 🔧 Cách Hoạt Động

### Training Flow:
```
1. User uploads images → saved to training_data/images/
2. User uploads COCO JSON → saved to training_data/labels/
3. User clicks "Bắt đầu Training" → triggers /start_training
4. Backend creates TrainingLogger instance
5. Training runs in background thread:
   - For each image:
     - Run model inference
     - Log detections if confidence < threshold (low_detection)
     - Compare with ground truth to detect false positives
     - Annotate and save images
   - Update progress state every image
6. Frontend polls /training_status every 1 second
7. When complete, display results with images

```

### Logging Detail:
```
Low Detection (Confidence < threshold):
- Original image path
- All detections with low confidence
- Confidence scores
- Bounding boxes
- Annotated image (red boxes)

False Positives:
- Predictions không khớp với ground truth (IoU < 0.3)
- IoU scores
- Confidence scores  
- Annotated image (yellow boxes)

Statistics:
- Total images processed
- Average confidence
- Low detection count
- False positive count
```

---

## 💾 Data Structure

### JSON Report Format

**training_summary.json:**
```json
{
  "training_name": "YYYYMMDD_HHMMSS",
  "start_time": "ISO datetime",
  "end_time": "ISO datetime",
  "total_images": number,
  "low_detection_count": number,
  "false_positives_count": number,
  "average_confidence": float,
  "status": "completed|failed"
}
```

**low_detection_report.json:**
```json
[
  {
    "original_image": "path",
    "saved_image": "path",
    "timestamp": "ISO datetime",
    "low_confidence_detections": [
      {
        "class": "class_name",
        "confidence": float,
        "bbox": [x1, y1, x2, y2]
      }
    ],
    "threshold": float
  }
]
```

---

## 🎨 Frontend Features

### Real-time Updates:
- Progress bar mỗi 1 second
- Status message updates
- Training process termination detection

### Image Display:
- Grid layout (auto-fill columns)
- Click to open full image
- Count badges

### State Management:
- `trainingImageFiles[]` - Track uploaded images
- `cocoFileSelected` - Track COCO file status
- `currentTrainingId` - Track current session

---

## 🚀 Usage Steps

1. **Navigate** to Training tab (🎓)
2. **Upload** training images (Section 1)
3. **Upload** COCO annotations file (Section 2)
4. **Configure** training parameters
5. **Click** "Bắt Đầu Training"
6. **Monitor** progress bar
7. **Review** results, images, and logs
8. **Download** logs or open folder

---

## ⚙️ Configuration

### Thay Đổi Nếu Cần:

#### Low Detection Threshold (default: 0.5)
- Đổi trong slider hoặc hard-code trong train_logger.py
- Thấp hơn = catch nhiều detection thấp hơn

#### False Positive IoU Threshold (default: 0.3)
- Hard-coded trong `record_false_positives()` method
- Thấp hơn = detection cần khớp chặt hơn với ground truth

#### Log Directory (default: training_logs/)
- Cấu hình trong yolov5_web_demo.py line 26

#### Training Images Folder (default: training_data/images/)
- Cấu hình trong yolov5_web_demo.py line 17

---

## 🔗 Dependencies

- **Python**: Flask, torch, cv2, numpy
- **Frontend**: HTML5, CSS3, JavaScript (vanilla, no frameworks)
- **YOLOv5**: torch.hub model loading

---

## 📞 Quick Reference

### Port: 5000
```
http://localhost:5000
```

### Endpoints:
```
POST   /upload_train_images          Upload training images
POST   /upload_coco_file             Upload COCO JSON
POST   /start_training               Start training
GET    /training_status              Get training progress
GET    /get_training_logs/:id        Get logs data
GET    /training_logs/:id/:type/:fn  Serve log images
```

### Folders:
```
training_logs/          → Training logs by session
training_data/images/   → Training images
training_data/labels/   → COCO annotations
```

---

## ✅ Testing Checklist

- [ ] Upload multiple images successfully
- [ ] Upload valid COCO JSON
- [ ] Start training begins progress
- [ ] Progress bar updates real-time
- [ ] Low detection images saved correctly
- [ ] False positives detected correctly
- [ ] Results display with images
- [ ] Open training logs folder works
- [ ] Click image zooms/opens
- [ ] Logs persist after refresh

---

Hệ thống training đã sẵn sàng! 🎉

