import cv2
import numpy as np
import scipy.linalg
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


''' Kalman Filter Parameters in filterpy library

x : ndarray (dim_x, 1), default = [0,0,0…0]
filter state estimate
P : ndarray (dim_x, dim_x), default eye(dim_x)
covariance matrix
Q : ndarray (dim_x, dim_x), default eye(dim_x)
Process uncertainty/noise
R : ndarray (dim_z, dim_z), default eye(dim_x)
measurement uncertainty/noise
H : ndarray (dim_z, dim_x)
measurement function
F : ndarray (dim_x, dim_x)
state transistion matrix
B : ndarray (dim_x, dim_u), default 0
control transition matrix
                        
https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html
'''


def image_processing(frame):
    '''
    All image processing techniques are applied in this function. 
    '''
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(gray, 128,
                                    255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(
        im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    obj = [cv2.boundingRect(cnt)
           for cnt in contours if cv2.contourArea(cnt) > 5]

    return obj


def KF_Initializer(n):
    '''
    This code will be executed only for the first frame of the video. 

    The matrices for filterpy.kalman are created with dimensions according to the amount of objects detected on the first frame.   
    '''
    # each object contains (x, y) points and their derivative (dx, dy)
    dim_x = n * 4
    dim_z = n * 2  # only the points are measured (x, y)
    kalman = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
    kalman.x = np.zeros((dim_x, 1))
    kalman.P = 1 * kalman.P  # already contains np.eye(dim_x)
    kalman.R = 2  # already contains np.eye(dim_z)

    kalman.F = []
    kalman.H = []
    colors = []
    for i in range(0, n):
        colors.append(np.random.rand(3,)*255)
        kalman.F = scipy.linalg.block_diag(np.array([[1., 1., 0, 0],
                                                     [0, 1., 0, 0],
                                                     [0, 0, 1., 1.],
                                                     [0, 0, 0, 1.]]), kalman.F)

        kalman.H = scipy.linalg.block_diag(np.array([[1., 0, 0, 0],
                                                     [0, 0, 1., 0]]), kalman.H)
    kalman.F = kalman.F[:-1]
    kalman.H = kalman.H[:-1]

    return kalman, colors


def get_points(kalman, obj, frame):
    ''' 
    kalman.x - > predictions for the object centroid points on previous frame ( Prediction[k-1] ). 

    Predictions are initiated with zero in the first loop.
    '''
    n = len(obj)
    pt_detected = np.zeros((n, 2))
    for i in range(0, n):
        obj_x, obj_y, obj_w, obj_h = obj[i]
        pt_detected[i][0] = int(obj_x + obj_w / 2)
        pt_detected[i][1] = int(obj_y + obj_h / 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        p_text = (obj_x - obj_w - 5, obj_y - obj_h - 5)
        start_pt = (obj_x - obj_w, obj_y - obj_h)
        end_pt = (obj_x + obj_w*2, obj_y + obj_h*2)
        frame = cv2.rectangle(frame, (start_pt), (end_pt), (0, 255, 0), 2)
        frame = cv2.putText(frame, 'Object Detected', p_text, font,
                            0.6, (0, 255, 0), 2)
    pt_predicted = np.zeros((n, 2))
    c = 0
    for i in range(0, n):
        try:
            pt_predicted[i][0] = kalman.x[c][0]
            c = c + 2
            pt_predicted[i][1] = kalman.x[c][0]
            c = c + 2
        except:
            pass

    return pt_detected, pt_predicted, frame


def label_assignment(detection, prediction, kalman):
    '''
    Labels are re-assigned in order to maintain the tracking on the same object. By default, label order changes according to object position on screen.

    Spicy.Linear_Sum_Assignment was used. This method calculates the lowest cost assignment given a Cost Matrix (parameters are distance between points predicted and measured) as input.

    It is assumed that an object will move smoothly, without any abrupt changes.
    '''
    reassigned_detection = detection.copy()
    M = len(detection)
    N = len(prediction)
    cost = np.zeros(shape=(N, M))
    for i in range(N):
        for j in range(M):
            try:
                diff_x = prediction[i][0] - detection[j][0]
                diff_y = prediction[i][1] - detection[j][1]
                distance = np.sqrt(diff_x*diff_x + diff_y*diff_y)
                cost[i][j] = distance
            except:
                pass

    cost = 0.5 * cost
    assignment = []
    for _ in range(N):
        assignment.append(-1)
    row_ind, col_ind = linear_sum_assignment(cost)
    for i in range(len(row_ind)):
        assignment[row_ind[i]] = col_ind[i]

    kalman.predict()
    for i in range(len(assignment)):
        if(assignment[i] != -1):
            reassigned_detection[i] = detection[assignment[i]]
        else:
            reassigned_detection[i] == np.array([[0, 0]])
    try:
        reassigned_detection = reassigned_detection.reshape(len(kalman.z), 1)
    except:
        reassigned_detection = np.zeros((len(kalman.z), 1))

    return reassigned_detection


def print_prediction_lines(detection, detections, prediction, predictions, frame, colors, frame_number):
    '''
    Reassigned points are applied to its correct order in the original detection order for next prediction.
    '''
    for i in range(len(detection)):

        try:
            detections[2*i][frame_number] = int(detection[i][0])
            detections[2*i+1][frame_number] = int(detection[i][1])
            predictions[2*i][frame_number] = int(prediction[i][0])
            predictions[2*i+1][frame_number] = int(prediction[i][1])

            x_past = int(predictions[2*i][frame_number - 5])
            y_past = int(predictions[2*i+1][frame_number - 5])
            x2_now = int(predictions[2*i][frame_number])
            y2_now = int(predictions[2*i+1][frame_number])

            frame = cv2.line(frame, (x_past, y_past),
                             (x2_now, y2_now), colors[i], 4)
        except:
            pass
    return detections, predictions, frame
