import cv2
import time
import math
import random as r
import numpy as np
from ultralytics import YOLO


class YOLOclass:
    def __init__(self, capture_index):
        # Путь к модели
        self.model_path = 'C:\\Users\\1314126\\models\\y11n3_40.onnx'
        self.model = YOLO(self.model_path, task='detect')
        self.capture_index = capture_index
        self.n = 10  # Количество шариков

    def initialize_balls(self):
        """Инициализация начальных параметров шариков."""
        x = np.random.randint(40, 600, size=self.n)
        y = np.random.randint(40, 480, size=self.n)
        angles = np.random.randint(0, 360, size=self.n) * math.pi / 180
        sins = np.sin(angles)
        cosins = np.cos(angles)
        flags = np.ones(self.n, dtype=bool)
        return x, y, sins, cosins, flags

    def draw_balls(self, frame, x, y, t, flags):
        """Отрисовка шариков на экране."""
        for j in range(self.n):
            if flags[j]:
                cv2.circle(frame, (int(x[j] + t * sins[j]), int(y[j] + t * cosins[j])), 20, (0, 0, 255), 5)

    def check_collisions(self, boxes, x, y, t, flags):
        """Проверка столкновений шариков с детектированными ладонями."""
        for box in boxes:
            box_x1, box_y1, box_x2, box_y2 = map(int, box)
            for j in range(self.n):
                ball_x = int(x[j] + t * sins[j])
                ball_y = int(y[j] + t * cosins[j])
                if box_x1 < ball_x < box_x2 and box_y1 < ball_y < box_y2:
                    flags[j] = False

    def run(self):
        """Основной цикл программы."""
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened(), "Не удалось открыть камеру"

        # Инициализация шариков
        x, y, sins, cosins, flags = self.initialize_balls()

        fps_history = []
        t = 0.0

        while cap.isOpened():
            start_time = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break

            # Детекция ладоней
            results = self.model.predict(frame, imgsz=480)
            boxes = results[0].boxes.xyxy.cpu().numpy()  # Получаем bounding boxes

            # Отрисовка шариков
            self.draw_balls(frame, x, y, t, flags)

            # Проверка столкновений
            self.check_collisions(boxes, x, y, t, flags)

            # Отрисовка bounding boxes
            for box in boxes:
                box = list(map(int, box))
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 5)

            # Расчет FPS
            end_time = time.perf_counter()
            fps = 1 / (end_time - start_time)
            fps_history.append(fps)
            avg_fps = round(sum(fps_history[-30:]) / len(fps_history[-30:]), 1)  # Среднее за последние 30 кадров

            # Отображение FPS
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("High5AR v0.3", frame)

            # Обновление времени и выход по клавише 'q'
            t += 0.1
            if cv2.waitKey(1) == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print('start')
    transformer_detector = YOLOclass(0)
    transformer_detector.run()
    print('done')