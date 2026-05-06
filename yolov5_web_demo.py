from flask import Flask, render_template, request, Response, jsonify, send_file
import torch
import cv2
import os
import time
from pathlib import Path
from werkzeug.utils import secure_filename
import base64
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Tạo thư mục nếu chưa có
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

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

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 YOLOv5 Web Interface đang chạy...")
    print("="*60)
    print("📱 Mở trình duyệt và truy cập: http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)