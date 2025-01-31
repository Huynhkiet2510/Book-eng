from flask import Flask, request, send_file, render_template
import cv2
import numpy as np
from ultralytics import YOLO
import io
import os  # Thêm thư viện os để lấy giá trị cổng từ biến môi trường
import logging

# Thiết lập logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Load the YOLO model
model = YOLO('yolov8n.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        # Read the frame from the request
        file = request.data
        np_arr = np.frombuffer(file, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Kiểm tra độ phân giải của ảnh (tùy chỉnh độ phân giải)
        frame = cv2.resize(frame, (640, 480))  # Giảm độ phân giải nếu cần

        # Perform object detection
        results = model(frame)

        # Draw boxes and labels on the frame
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

    except Exception as e:
        logging.error(f"Error processing frame: {e}")
        return "Error processing frame", 500

if __name__ == '__main__':
    # Sử dụng cổng mà Render cung cấp
    port = int(os.environ.get("PORT", 5000))  # Mặc định là 5000 nếu không có cổng
    app.run(host='0.0.0.0', port=port, threaded=True)
