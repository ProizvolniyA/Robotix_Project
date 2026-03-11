import cv2
from ultralytics import YOLO
import time

model = YOLO('yolo26n.pt') 

cap = cv2.VideoCapture('golf.mp4')

WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
CENTER_X = WIDTH // 2
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)


print("Запуск теста YOLO26. Нажмите 'q' для выхода.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    start_time = time.time() # Для расчета FPS

    # Detection
    # conf=0.5 — уверенность выше 50%
    # classes=[0] 
    results = model.predict(frame, conf=0.1, classes=[0], verbose=False)
    
    annotated_frame = frame.copy()
    
    # прицел
    cv2.line(annotated_frame, (CENTER_X, 0), (CENTER_X, HEIGHT), (255, 255, 0), 1)

    for result in results:
        
        best_box = None
        min_offset = float('inf')

        for box in result.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            obj_center_x = (x1 + x2) // 2
            
            offset = obj_center_x - CENTER_X
            
            if abs(offset) < min_offset:
                min_offset = abs(offset)
                best_box = (x1, y1, x2, y2, offset)

        if best_box:
            x1, y1, x2, y2, offset = best_box
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # линия отклонения
            cv2.line(annotated_frame, (CENTER_X, (y1+y2)//2), (CENTER_X + offset, (y1+y2)//2), (0, 0, 255), 2)
            # значение отклонения (то, что пойдет в PX4)
            cv2.putText(annotated_frame, f"Offset: {offset}px", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    fps = 1.0 / (time.time() - start_time)
    cv2.putText(annotated_frame, f"FPS: {round(fps, 1)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    
    cv2.imshow("YOLO26 Person Detection Test", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()