import cv2
import time
import math
import random as r
from ultralytics import YOLO

class YOLOclass:

    def __init__(self, capture_index):

        # Путь к вашему файлу yolov8n.pt
        model_path = 'C:\\Users\\1314126\\models\\y11n3_40.onnx'

        # Создание экземпляра модели YOLO
        self.model = YOLO(model_path, task='detect')
        self.capture_index = capture_index

    def __call__(self):

        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        t = float(0)
        n = 10
        fps_history = []
        x = []
        y = []
        sins = []
        cosins = []
        flags = []

        for j in range(n):
            x.append(r.randint(40, 640))
            y.append(r.randint(40, 480))
            sins.append(math.sin(r.randint(0, 3)))
            cosins.append(math.cos(r.randint(0, 3)))
            flags.append(True)

        while cap.isOpened():

            start_time = time.perf_counter()
            ret, frame = cap.read()
            results = self.model.predict(frame, imgsz=480)
            for j in range(n):
                if flags[j]:
                    cv2.circle(frame, (int(x[j]+t*sins[j]), int(y[j]+t*cosins[j])), 20, (0,0,255), 5)

            for i in range(len(results[0].boxes.xyxy)):
                box = list(map(int, results[0].boxes.xyxy[i]))
                cv2.rectangle(frame, (box[0],box[1]), (box[2],box[3]), (0, 255, 255), 5)
                for j in range(n):
                    if box[0]<x[j]<box[2] and box[1]<y[j]<box[3]:
                        flags[j] = False

            end_time = time.perf_counter()
            fps = 1/ (end_time - start_time)
            fps_history.append(fps)
            avg_fps = round(sum(fps_history[-50:]) / len(fps_history[-50:]), 1)
            t+=0.1

            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 2)
            cv2.imshow("High5AR v0.3", frame)

            if cv2.waitKey(1) == ord("q"):
                exit()

        cap.release()
        cv2.destroyAllWindows()

print('start')
transformer_detector = YOLOclass(0)
transformer_detector()
print('done')

    