from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import time
from waitress import serve  

app = Flask(__name__)
model = YOLO('yolov8n.pt')

@app.route('/')
def index():
    return render_template('index.html')


def gen_frames():
    cap = cv2.VideoCapture(0)  
    
    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture image")
            break


        results = model(frame)

        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]  
                class_index = int(box.cls[0])  
                class_name = model.names[class_index]  

               
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

               
                font_scale = 1.5
                thickness = 1
                label = f'{class_name}'
                cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

        
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()  


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    
    serve(app, host='0.0.0.0', port=5000)  
