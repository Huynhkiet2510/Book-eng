import cv2
from ultralytics import YOLO


model = YOLO("yolov8n.pt")  


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    
    results = model(frame)
    
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            label = model.names[int(box.cls[0])]  
            confidence = box.conf[0].item()       
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = label
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    
    cv2.imshow("YOLOv8 Detection", frame)
    
   
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

cap.release()
cv2.destroyAllWindows()
