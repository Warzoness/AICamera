"""
Module để logging và tracking chi tiết trong quá trình training YOLOv5
Theo dõi ảnh có độ nhận diện thấp và vùng bị nhiễu object.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2


class TrainingLogger:
    def __init__(self, training_name=None):
        """Khởi tạo training logger.

        Args:
            training_name: Tên của session training (tự động tạo nếu None)
        """
        self.training_name = training_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(f"training_logs/{self.training_name}")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Thư mục con
        self.low_detection_dir = self.log_dir / "low_detection_images"
        self.false_positives_dir = self.log_dir / "false_positives"
        self.detailed_logs_dir = self.log_dir / "detailed_logs"

        self.low_detection_dir.mkdir(parents=True, exist_ok=True)
        self.false_positives_dir.mkdir(parents=True, exist_ok=True)
        self.detailed_logs_dir.mkdir(parents=True, exist_ok=True)

        # Khởi tạo file chính
        self.summary_file = self.log_dir / "training_summary.json"
        self.detailed_log_file = self.detailed_logs_dir / "training_log.txt"
        self.low_detection_file = self.log_dir / "low_detection_report.json"
        self.false_positives_file = self.log_dir / "false_positives_report.json"

        # Dữ liệu summary
        self.summary = {
            "training_name": self.training_name,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "total_images": 0,
            "low_detection_count": 0,
            "false_positives_count": 0,
            "average_confidence": 0.0,
            "status": "in_progress",
        }

        # Dữ liệu chi tiết
        self.low_detection_images = []
        self.false_positives_images = []

        # Setup logging
        self._setup_logging()
        self.log("🟢 Training Logger initialized")

    def _setup_logging(self):
        """Setup Python logging."""
        logger = logging.getLogger(f"TrainingLogger_{self.training_name}")
        logger.setLevel(logging.DEBUG)

        handler = logging.FileHandler(self.detailed_log_file, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        self.logger = logger

    def _safe_print(self, message):
        try:
            print(message)
        except UnicodeEncodeError:
            print(message.encode(sys.stdout.encoding, errors="replace").decode(sys.stdout.encoding, errors="replace"))

    def log(self, message):
        """Ghi log tin nhắn."""
        self._safe_print(message)
        self.logger.info(message)

    def record_low_detection(self, image_path, detections, confidence_threshold=0.5):
        """Ghi lại ảnh có độ nhận diện thấp.

        Args:
            image_path: Đường dẫn ảnh
            detections: Danh sách các detection (list of dicts với keys: class, confidence, bbox)
            confidence_threshold: Ngưỡng confidence
        """
        low_conf_detections = [d for d in detections if d.get("confidence", 0) < confidence_threshold]

        if low_conf_detections:
            # Đọc ảnh gốc
            image = cv2.imread(image_path)
            if image is None:
                self.log(f"⚠️ Không thể đọc ảnh: {image_path}")
                return

            # Vẽ các detection có confidence thấp
            annotated_image = image.copy()
            for det in low_conf_detections:
                bbox = det.get("bbox", [])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    # Vẽ hình chữ nhật màu đỏ cho detection có confidence thấp
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # Ghi confidence
                    conf = det.get("confidence", 0)
                    cv2.putText(
                        annotated_image,
                        f"{det.get('class', 'Unknown')}: {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )

            # Lưu ảnh annotated
            filename = os.path.basename(image_path)
            output_path = self.low_detection_dir / f"low_conf_{filename}"
            cv2.imwrite(str(output_path), annotated_image)

            # Ghi thông tin vào JSON
            record = {
                "original_image": image_path,
                "saved_image": str(output_path),
                "timestamp": datetime.now().isoformat(),
                "low_confidence_detections": low_conf_detections,
                "threshold": confidence_threshold,
            }
            self.low_detection_images.append(record)
            self.summary["low_detection_count"] += 1

            self.log(f"📸 Ghi nhận ảnh có detection thấp: {filename} ({len(low_conf_detections)} detections)")

    def record_false_positives(self, image_path, all_detections, ground_truth_boxes, iou_threshold=0.3):
        """Ghi lại false positives (detection không khớp với ground truth).

        Args:
            image_path: Đường dẫn ảnh
            all_detections: Tất cả các detection
            ground_truth_boxes: Các bounding box ground truth [[x1, y1, x2, y2], ...]
            iou_threshold: Ngưỡng IoU để xác định match
        """
        false_positives = []

        for det in all_detections:
            bbox = det.get("bbox", [])
            if len(bbox) != 4:
                continue

            # Kiểm tra IoU với ground truth
            max_iou = 0
            for gt_box in ground_truth_boxes:
                iou = self._calculate_iou(bbox, gt_box)
                max_iou = max(max_iou, iou)

            # Nếu IoU < threshold, đó là false positive
            if max_iou < iou_threshold:
                det["iou_with_gt"] = max_iou
                false_positives.append(det)

        if false_positives:
            # Đọc ảnh gốc
            image = cv2.imread(image_path)
            if image is None:
                return

            # Vẽ false positives
            annotated_image = image.copy()
            for det in false_positives:
                bbox = det.get("bbox", [])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    # Vẽ hình chữ nhật màu vàng cho false positive
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(
                        annotated_image,
                        f"FP: {det.get('class', 'Unknown')}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2,
                    )

            # Lưu ảnh
            filename = os.path.basename(image_path)
            output_path = self.false_positives_dir / f"false_pos_{filename}"
            cv2.imwrite(str(output_path), annotated_image)

            # Ghi thông tin
            record = {
                "original_image": image_path,
                "saved_image": str(output_path),
                "timestamp": datetime.now().isoformat(),
                "false_positives": false_positives,
                "fp_count": len(false_positives),
            }
            self.false_positives_images.append(record)
            self.summary["false_positives_count"] += 1

            self.log(f"⚠️ Phát hiện {len(false_positives)} false positives trong {filename}")

    @staticmethod
    def _calculate_iou(box1, box2):
        """Tính IoU giữa 2 bounding box [x1, y1, x2, y2]."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)

        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area if union_area > 0 else 0

        return iou

    def record_statistics(self, total_images, average_confidence):
        """Ghi lại thống kê chung."""
        self.summary["total_images"] = total_images
        self.summary["average_confidence"] = average_confidence
        self.log(f"📊 Thống kê: {total_images} ảnh, confidence TB: {average_confidence:.2%}")

    def finalize(self, success=True, error_message=None):
        """Hoàn tất training logger."""
        self.summary["end_time"] = datetime.now().isoformat()
        self.summary["status"] = "completed" if success else f"failed: {error_message}"

        # Lưu tất cả files JSON
        with open(self.summary_file, "w", encoding="utf-8") as f:
            json.dump(self.summary, f, indent=2, ensure_ascii=False)

        with open(self.low_detection_file, "w", encoding="utf-8") as f:
            json.dump(self.low_detection_images, f, indent=2, ensure_ascii=False)

        with open(self.false_positives_file, "w", encoding="utf-8") as f:
            json.dump(self.false_positives_images, f, indent=2, ensure_ascii=False)

        status_msg = "✅ Training hoàn tất thành công" if success else f"❌ Training thất bại: {error_message}"
        self.log(f"\n{status_msg}")
        self.log(f"📁 Log directory: {self.log_dir}")
        self.log(f"📊 Summary: {self.summary}")

    def get_summary_dict(self):
        """Trả về dictionary summary (dùng để trả về frontend)."""
        return {
            **self.summary,
            "log_dir": str(self.log_dir),
            "low_detection_count": len(self.low_detection_images),
            "false_positives_count": len(self.false_positives_images),
        }
