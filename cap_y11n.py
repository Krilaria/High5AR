import cv2
import time
import random as r
from ultralytics import YOLO

class YOLOclass:

    def __init__(self, capture_index):

        # Путь к вашему файлу yolov8n.pt
        model_path = 'C:\\Users\\1314126\\models\\y11n2_30.pt'
        #"C:\Users\1314126\models\y11n_15.pt"

        # Создание экземпляра модели YOLO
        self.model = YOLO(model_path, task='detect')
        self.capture_index = capture_index

    def __call__(self):

        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        fps_history = []

        while cap.isOpened():

            start_time = time.perf_counter()
            ret, frame = cap.read()
            results = self.model.predict(frame, imgsz=320)
            try:
                for i in range(len(results[0].boxes.xyxy)):
                    box = results[0].boxes.xyxy[i]
                    cv2.rectangle(frame, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (0,250,255), 5)
            except:
                box = [1, 1, 1, 1]

            end_time = time.perf_counter()
            fps = 1/ (end_time - start_time)
            fps_history.append(fps)
            avg_fps = round((sum(fps_history)/len(fps_history)), 1)

            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 2)
            cv2.imshow("High5AR v0.2", frame)

            if cv2.waitKey(1) == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

print('start')
transformer_detector = YOLOclass(0)
transformer_detector()
print('done')

    