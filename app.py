from flask import Flask, request, send_file, render_template
import cv2
import numpy as np
from ultralytics import YOLO
import io
import os  # Thêm thư viện os để lấy giá trị cổng từ biến môi trường
import gdown  # Thư viện gdown để tải file từ Google Drive

app = Flask(__name__)

# Đường dẫn tới file yolov8n.pt
model_path = "yolov8n.pt"

# Kiểm tra nếu file đã tồn tại chưa, nếu chưa thì tải về
if not os.path.exists(model_path):
    print("Downloading YOLO model from Google Drive...")
    url = "https://drive.google.com/uc?id=1qBw9RFCPp8Cavzu-On9CVtTCPgkadpzJ"  # ID file từ Google Drive
    gdown.download(url, model_path, quiet=False)

# Load the YOLO model
model = YOLO(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    # Read the frame from the request
    file = request.data
    np_arr = np.frombuffer(file, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Perform object detection
    results = model(frame)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            class_index = int(box.cls[0])
            class_name = model.names[class_index]
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Add label
            cv2.putText(frame, class_name, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Encode the processed frame to JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    return send_file(io.BytesIO(buffer), mimetype='image/jpeg')

if __name__ == '__main__':
    # Sử dụng cổng mà Render cung cấp
    port = int(os.environ.get("PORT", 5000))  # Mặc định là 5000 nếu không có cổng
    app.run(host='0.0.0.0', port=port)
