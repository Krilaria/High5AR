import cv2
import time
import random as r

class DETRclass:

    def __init__(self, capture_index):

        #self.model = model
        self.capture_index = capture_index

    def __call__(self):

        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        fps_history = []
        x = r.randint(50, 500)
        y = r.randint(50, 500)

        while cap.isOpened():

            start_time = time.perf_counter()
            ret, frame = cap.read()
            end_time = time.perf_counter()
            fps = 1/ (end_time - start_time)
            fps_history.append(fps)
            avg_fps = round((sum(fps_history)/len(fps_history)), 1)
            #print(avg_fps)

            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 2)
            cv2.imshow("High5AR v0.1", frame)

            if cv2.waitKey(1) == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

print('start')
transformer_detector = DETRclass(0)
transformer_detector()
print('done')
        
        




    