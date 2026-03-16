import cv2
import math
import time
from ultralytics import YOLO
import collections
# Исправление совместимости для Python 3.11+
if not hasattr(collections, 'MutableMapping'):
    import collections.abc
    collections.MutableMapping = collections.abc.MutableMapping
from dronekit import connect, VehicleMode

# ==========================================
# 1. ПОДКЛЮЧЕНИЕ К ДРОНУ
# ==========================================
print("Подключение к PX4...")

#vehicle = connect('127.0.0.1:14550', wait_ready=True)
# Замените '/dev/serial0' на ваш порт
vehicle = connect('/dev/serial0', baud=57600, wait_ready=True)
print(f"Связь установлена! Текущий режим: {vehicle.mode.name}")

# ==========================================
# 2. НАСТРОЙКИ КАМЕРЫ И YOLO
# ==========================================

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('alone_wolf.mp4')
FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_HEIGHT= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
CENTER_X = FRAME_WIDTH // 2
CENTER_Y = FRAME_HEIGHT // 2  # для расчета расстояния до центра кадра


cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

model = YOLO('yolo26n.pt') 

# P-controller
Kp = 0.8               # Коэффициент резкости поворота
DEADBAND = 20          # Мертвая зона 
TRACK_CHANNEL = '6'    # Канал пульта для переключения режимов (тумблер)

# ==========================================
# 3. ОСНОВНОЙ ЦИКЛ
# ==========================================
print("Запуск системы зрения. Нажмите 'q' для выхода.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    start_time = time.time()
    # Если пульт выключен, вернет None, поэтому ставим значение по умолчанию 1000 (выкл)
    ch6_value = vehicle.channels.get(TRACK_CHANNEL) or 1000
    auto_track_mode = 1
    #auto_track_mode = (ch6_value > 1500)
    
    results = model.predict(frame, conf=0.1, classes=[0], verbose=False)
    
    target_found = False
    min_distance_to_center = float('inf') 
    best_target_x = CENTER_X
    best_target_y = CENTER_Y
    best_box = None

    for result in results:
        for box in result.boxes:
            target_found = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            obj_center_x = (x1 + x2) // 2
            obj_center_y = (y1 + y2) // 2
            
            
            distance = math.hypot(obj_center_x - CENTER_X, obj_center_y - CENTER_Y)
            
            
            if distance < min_distance_to_center:
                min_distance_to_center = distance
                best_target_x = obj_center_x
                best_target_y = obj_center_y
                best_box = (x1, y1, x2, y2)

    # ==========================================
    # ЛОГИКА УПРАВЛЕНИЯ
    # ==========================================
    if auto_track_mode:
        # Режим 1: АВТОНОМНОЕ СЛЕЖЕНИЕ
        cv2.putText(frame, "MODE: AUTO TRACKING", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if target_found:
            error_x = best_target_x - CENTER_X
            
            if abs(error_x) < DEADBAND:
                yaw_pwm = 1500 
            else:
                yaw_pwm = 1500 + int(error_x * Kp)
            
            yaw_pwm = max(1300, min(1700, yaw_pwm))
            
            vehicle.channels.overrides['4'] = yaw_pwm
            
            cv2.rectangle(frame, (best_box[0], best_box[1]), (best_box[2], best_box[3]), (0, 255, 0), 2)
            cv2.circle(frame, (best_target_x, best_target_y), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"YAW PWM: {yaw_pwm}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            fps = 1.0 / (time.time() - start_time)
    
            cv2.putText(frame, f"FPS: {round(fps, 1)}", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            
        else:
            vehicle.channels.overrides['4'] = 1500
            cv2.putText(frame, "NO TARGET", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
    else:
        # Режим 2: РУЧНОЕ УПРАВЛЕНИЕ (Пульт)
        cv2.putText(frame, "MODE: MANUAL", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        vehicle.channels.overrides = {}

        if target_found:
            cv2.rectangle(frame, (best_box[0], best_box[1]), (best_box[2], best_box[3]), (255, 0, 0), 2)
            cv2.circle(frame, (best_target_x, best_target_y), 5, (0, 255, 0), -1)

    # прицел
    cv2.line(frame, (CENTER_X, 0), (CENTER_X, FRAME_HEIGHT), (255, 255, 0), 1)
    
    cv2.imshow("Drone Vision", frame)

    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# ==========================================
# 4. БЕЗОПАСНОЕ ЗАВЕРШЕНИЕ
# ==========================================
print("Отключение...")
vehicle.channels.overrides = {} 
vehicle.close()                 
cap.release()
cv2.destroyAllWindows()
#vehicle.close()
