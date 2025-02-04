from flask import Flask, request, send_file, render_template
import cv2
import numpy as np
from ultralytics import YOLO
import io
import os  
import logging


logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)


model = YOLO('yolov8n.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        file = request.data
        np_arr = np.frombuffer(file, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        frame = cv2.resize(frame, (320, 240))  

       
        results = model(frame)

       
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                class_index = int(box.cls[0])
                class_name = model.names[class_index]
              
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                cv2.putText(frame, class_name, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

      
        _, buffer = cv2.imencode('.jpg', frame)
        return send_file(io.BytesIO(buffer), mimetype='image/jpeg')

    except Exception as e:
        logging.error(f"Error processing frame: {e}")
        return "Error processing frame", 500

if __name__ == '__main__':
   
    port = int(os.environ.get("PORT", 5000))  
    app.run(host='0.0.0.0', port=port, threaded=True)
