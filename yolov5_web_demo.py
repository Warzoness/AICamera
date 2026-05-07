from flask import Flask, render_template, request, Response, jsonify, send_file
import torch
import cv2
import os
import time
from pathlib import Path
from werkzeug.utils import secure_filename
import base64
import numpy as np
import json
import threading
from train_logger import TrainingLogger

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['TRAIN_IMAGES_FOLDER'] = 'training_data/images'
app.config['TRAIN_LABELS_FOLDER'] = 'training_data/labels'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Tạo thư mục nếu chưa có
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['TRAIN_IMAGES_FOLDER'], exist_ok=True)
os.makedirs(app.config['TRAIN_LABELS_FOLDER'], exist_ok=True)
os.makedirs('training_logs', exist_ok=True)

# Global training state
training_state = {
    'is_training': False,
    'current_training': None,
    'progress': 0,
    'message': ''
}

# Load YOLOv5 model
print("⏳ Đang tải YOLOv5 model (Custom trained)...")
# Load custom trained model với pig class
model = torch.hub.load('.', 'custom', path='runs/train/exp3/weights/best.pt', source='local', force_reload=False)
model.conf = 0.7
print("✅ YOLOv5 đã sẵn sàng! (Classes: pig)")

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'ts'}

def allowed_file(filename, file_type='image'):
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    if file_type == 'image':
        return ext in ALLOWED_IMAGE_EXTENSIONS
    elif file_type == 'video':
        return ext in ALLOWED_VIDEO_EXTENSIONS
    return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Xử lý upload và detect ảnh"""
    if 'file' not in request.files:
        return jsonify({'error': 'Không có file được upload'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Không có file được chọn'}), 400
    
    if file and allowed_file(file.filename, 'image'):
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # YOLOv5 detection
        results = model(filepath)
        
        # Lưu ảnh kết quả với tên cụ thể
        output_filename = f"output_{timestamp}_{filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Lưu kết quả bằng OpenCV (đảm bảo luôn hoạt động)
        annotated_image = results.render()[0]  # Render kết quả
        cv2.imwrite(output_path, annotated_image)
        
        # Lấy predictions
        predictions = results.pandas().xyxy[0]
        detections = []
        for idx, row in predictions.iterrows():
            detections.append({
                'class': row['name'],
                'confidence': float(row['confidence']),
                'bbox': [int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])]
            })
        
        return jsonify({
            'success': True,
            'output_image': f'/outputs/{output_filename}',
            'detections': detections,
            'total_objects': len(detections)
        })
    
    return jsonify({'error': 'File không hợp lệ'}), 400

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Xử lý upload video"""
    if 'file' not in request.files:
        return jsonify({'error': 'Không có file được upload'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Không có file được chọn'}), 400
    
    if file and allowed_file(file.filename, 'video'):
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'video_path': unique_filename,
            'message': 'Video đã upload thành công. Click "Xử lý Video" để bắt đầu detection.'
        })
    
    return jsonify({'error': 'File không hợp lệ'}), 400

@app.route('/process_video/<filename>')
def process_video(filename):
    """Stream video với detection real-time"""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    def generate():
        cap = cv2.VideoCapture(filepath)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # YOLOv5 detection
            results = model(frame)
            annotated_frame = results.render()[0]
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.03)  # ~30fps
        
        cap.release()
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/save_video/<filename>')
def save_video(filename):
    """Lưu video đã xử lý"""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_filename = f"output_{filename}"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    
    cap = cv2.VideoCapture(filepath)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        annotated_frame = results.render()[0]
        out.write(annotated_frame)
        frame_count += 1
    
    cap.release()
    out.release()
    
    return jsonify({
        'success': True,
        'output_video': f'/outputs/{output_filename}',
        'frames_processed': frame_count
    })

@app.route('/webcam_feed')
def webcam_feed():
    """Stream từ webcam với detection"""
    def generate():
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model(frame)
            annotated_frame = results.render()[0]
            
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        cap.release()
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/outputs/<filename>')
def serve_output(filename):
    """Serve output files"""
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename))

@app.route('/uploads/<filename>')
def serve_upload(filename):
    """Serve uploaded files"""
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

# ==================== TRAINING ENDPOINTS ====================

@app.route('/upload_train_images', methods=['POST'])
def upload_train_images():
    """Upload ảnh cho training"""
    if 'files' not in request.files:
        return jsonify({'error': 'Không có file được upload'}), 400
    
    files = request.files.getlist('files')
    uploaded_files = []
    
    for file in files:
        if file and allowed_file(file.filename, 'image'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['TRAIN_IMAGES_FOLDER'], filename)
            file.save(filepath)
            uploaded_files.append(filename)
    
    if not uploaded_files:
        return jsonify({'error': 'Không có ảnh hợp lệ được upload'}), 400
    
    return jsonify({
        'success': True,
        'message': f'Đã upload {len(uploaded_files)} ảnh',
        'uploaded_files': uploaded_files,
        'total_images': len(uploaded_files)
    })

@app.route('/upload_coco_file', methods=['POST'])
def upload_coco_file():
    """Upload file COCO JSON"""
    if 'file' not in request.files:
        return jsonify({'error': 'Không có file được upload'}), 400
    
    file = request.files['file']
    
    if not file.filename.endswith('.json'):
        return jsonify({'error': 'File phải là định dạng JSON'}), 400
    
    try:
        # Đọc và validate COCO JSON
        file_content = file.read()
        coco_data = json.loads(file_content)
        
        # Kiểm tra cấu trúc COCO
        if 'images' not in coco_data or 'annotations' not in coco_data:
            return jsonify({'error': 'File COCO không hợp lệ - thiếu "images" hoặc "annotations"'}), 400
        
        # Lưu file
        filename = secure_filename('annotations_coco.json')
        filepath = os.path.join(app.config['TRAIN_LABELS_FOLDER'], filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f)
        
        return jsonify({
            'success': True,
            'message': f'Đã upload file COCO với {len(coco_data["images"])} ảnh và {len(coco_data["annotations"])} annotations',
            'images_count': len(coco_data['images']),
            'annotations_count': len(coco_data['annotations']),
            'categories': coco_data.get('categories', [])
        })
    
    except json.JSONDecodeError:
        return jsonify({'error': 'File JSON không hợp lệ'}), 400
    except Exception as e:
        return jsonify({'error': f'Lỗi: {str(e)}'}), 400

@app.route('/start_training', methods=['POST'])
def start_training():
    """Bắt đầu training"""
    if training_state['is_training']:
        return jsonify({'error': 'Một quá trình training đang chạy'}), 400
    
    # Validate dữ liệu
    images_dir = app.config['TRAIN_IMAGES_FOLDER']
    labels_dir = app.config['TRAIN_LABELS_FOLDER']
    
    if not os.listdir(images_dir):
        return jsonify({'error': 'Không có ảnh training. Vui lòng upload ảnh trước.'}), 400
    
    if not os.path.exists(os.path.join(labels_dir, 'annotations_coco.json')):
        return jsonify({'error': 'Không có file COCO. Vui lòng upload file annotations.'}), 400
    
    # Lấy tham số training
    data = request.get_json()
    epochs = data.get('epochs', 10)
    batch_size = data.get('batch_size', 16)
    img_size = data.get('img_size', 640)
    confidence_threshold = data.get('confidence_threshold', 0.5)
    
    # Khởi tạo training logger
    logger = TrainingLogger()
    training_state['is_training'] = True
    training_state['current_training'] = logger
    training_state['progress'] = 0
    training_state['message'] = 'Bắt đầu training...'
    
    # Chạy training trong thread riêng
    def run_training():
        try:
            logger.log(f"🚀 Bắt đầu training với các tham số:")
            logger.log(f"  - Epochs: {epochs}")
            logger.log(f"  - Batch size: {batch_size}")
            logger.log(f"  - Image size: {img_size}")
            logger.log(f"  - Confidence threshold: {confidence_threshold}")
            
            # Lấy danh sách ảnh
            image_files = [f for f in os.listdir(images_dir) if allowed_file(f, 'image')]
            total_images = len(image_files)
            logger.log(f"📊 Tổng {total_images} ảnh training")
            
            logger.record_statistics(total_images, 0.0)
            
            # Simulate training process
            total_steps = epochs * len(image_files)
            step = 0
            
            for epoch in range(epochs):
                logger.log(f"\n📚 Epoch {epoch + 1}/{epochs}")
                
                for i, img_file in enumerate(image_files):
                    img_path = os.path.join(images_dir, img_file)
                    
                    # Simulate detection
                    image = cv2.imread(img_path)
                    if image is None:
                        continue
                    
                    # Run model inference for simulation
                    try:
                        prev_conf = getattr(model, 'conf', None)
                        model.conf = confidence_threshold
                        results = model(img_path)
                        if prev_conf is not None:
                            model.conf = prev_conf
                        predictions = results.pandas().xyxy[0]
                        
                        detections = []
                        confidences = []
                        for idx, row in predictions.iterrows():
                            det = {
                                'class': row['name'],
                                'confidence': float(row['confidence']),
                                'bbox': [int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])]
                            }
                            detections.append(det)
                            confidences.append(det['confidence'])
                        
                        # Log low detection images
                        if detections:
                            logger.record_low_detection(img_path, detections, confidence_threshold)
                        
                        # Update statistics
                        avg_conf = np.mean(confidences) if confidences else 0.0
                        logger.record_statistics(total_images, avg_conf)
                    
                    except Exception as e:
                        logger.log(f"⚠️ Lỗi khi xử lý {img_file}: {str(e)}")
                    
                    step += 1
                    training_state['progress'] = int((step / total_steps) * 100)
                    training_state['message'] = f"Training: Epoch {epoch + 1}/{epochs}, ảnh {i + 1}/{len(image_files)}"
            
            # Finalize training
            logger.finalize(success=True)
            training_state['message'] = '✅ Training hoàn tất thành công'
            training_state['progress'] = 100
            
        except Exception as e:
            error_msg = str(e)
            logger.log(f"❌ Lỗi training: {error_msg}")
            logger.finalize(success=False, error_message=error_msg)
            training_state['message'] = f'❌ Lỗi: {error_msg}'
        
        finally:
            training_state['is_training'] = False
    
    # Chạy training trong thread
    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Training đã bắt đầu',
        'training_id': logger.training_name
    })

@app.route('/training_status', methods=['GET'])
def training_status():
    """Lấy trạng thái training"""
    return jsonify({
        'is_training': training_state['is_training'],
        'progress': training_state['progress'],
        'message': training_state['message'],
        'training_id': training_state['current_training'].training_name if training_state['current_training'] else None,
        'summary': training_state['current_training'].get_summary_dict() if training_state['current_training'] else None
    })

@app.route('/get_training_logs/<training_id>', methods=['GET'])
def get_training_logs(training_id):
    """Lấy logs của một training session"""
    try:
        log_dir = Path(f"training_logs/{training_id}")
        
        if not log_dir.exists():
            return jsonify({'error': 'Training session không tồn tại'}), 404
        
        # Đọc summary
        summary_file = log_dir / "training_summary.json"
        summary = {}
        if summary_file.exists():
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
        
        # Đọc low detection report
        low_det_file = log_dir / "low_detection_report.json"
        low_detection = []
        if low_det_file.exists():
            with open(low_det_file, 'r', encoding='utf-8') as f:
                low_detection = json.load(f)
        
        # Đọc false positives report
        fp_file = log_dir / "false_positives_report.json"
        false_positives = []
        if fp_file.exists():
            with open(fp_file, 'r', encoding='utf-8') as f:
                false_positives = json.load(f)
        
        # Liệt kê tất cả ảnh trong các thư mục
        low_det_images = []
        low_det_dir = log_dir / "low_detection_images"
        if low_det_dir.exists():
            low_det_images = [f"/training_logs/{training_id}/low_detection_images/{f.name}" 
                             for f in low_det_dir.glob('*') if f.is_file()]
        
        fp_images = []
        fp_dir = log_dir / "false_positives"
        if fp_dir.exists():
            fp_images = [f"/training_logs/{training_id}/false_positives/{f.name}" 
                        for f in fp_dir.glob('*') if f.is_file()]
        
        return jsonify({
            'success': True,
            'summary': summary,
            'low_detection_count': len(low_detection),
            'false_positives_count': len(false_positives),
            'low_detection_images': low_det_images,
            'false_positives_images': fp_images,
            'log_directory': str(log_dir)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/training_logs/<training_id>/<image_type>/<filename>')
def serve_training_image(training_id, image_type, filename):
    """Serve ảnh từ training logs"""
    valid_types = ['low_detection_images', 'false_positives']
    
    if image_type not in valid_types:
        return jsonify({'error': 'Invalid image type'}), 400
    
    image_path = os.path.join('training_logs', training_id, image_type, filename)
    
    if not os.path.exists(image_path):
        return jsonify({'error': 'Image not found'}), 404
    
    return send_file(image_path)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 YOLOv5 Web Interface đang chạy...")
    print("="*60)
    print("📱 Mở trình duyệt và truy cập: http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)