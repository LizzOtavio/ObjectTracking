import cv2
import numpy as np
import functions as f
from filterpy.kalman import KalmanFilter

video = cv2.VideoCapture('Objects.mp4')

frame_cnt = int(video. get(cv2.CAP_PROP_FRAME_COUNT))
frame_number = 0

detections = np.zeros((1, frame_cnt))
predictions = np.zeros((1, frame_cnt))
kalman = KalmanFilter(dim_x=1, dim_z=1) 
colors  = []
while (video.isOpened()):

    ret, frame = video.read()


    if ret:
        obj = f.image_processing(frame)

        if (frame_number == 0):
            kalman, colors  = f.KF_Initializer(len(obj))
            predictions = np.resize(predictions, (len(obj)*2, frame_cnt))
            detections = np.resize(detections, (len(obj)*2, frame_cnt))
          

        detection, prediction, frame = f.get_points(kalman, obj, frame)

        detections, predictions, frame = f.print_prediction_lines(detection, detections, prediction, predictions, frame, colors, frame_number)

        detection =  f.label_assignment(detection, prediction, kalman)

        kalman.update(z=detection)
     
        frame_number += 1

        cv2.imshow('', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    else:
        break


video.release()
cv2.destroyAllWindows()
