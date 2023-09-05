from ultralytics import YOLO
import cv2
import numpy as np
from os.path import dirname as up

PATH_ROOT = up(up(up(__file__)))

PATH_DETECTION = PATH_ROOT + "/models/best_detection.pt"
PATH_FINGER= PATH_ROOT  + "/models/best_finger.pt"


model_detection = YOLO(PATH_DETECTION)
model_finger = YOLO(PATH_FINGER)

def detect_qr_code(image, model = model_detection, threshold = 0.4):
    results = model(image)[0]
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, _ = result
        if score > threshold:
            new_image = image[int(y1):int(y2), int(x1):int(x2)]
            return new_image
        else:
            print(f"Little score: {score}")
    return None

def extract_qr_code(image, model = model_finger, threshold = 0.5):
    results = model(image)[0]

    dots = []

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, _ = result
        if score > threshold:
            dot = (int((x1+x2)/2),int((y1+y2)/2))
            dots.append(dot)

    def get_rotacion_angle(coordenates, epsilon = 230*2 ):
        p1, p2, p3 = coordenates
        vector1 = np.array(p2) - np.array(p3)
        vector2 = np.array(p1) - np.array(p3)
        vector3 = np.array(p1) - np.array(p2)

        if abs(np.dot(vector1,vector2)) <= epsilon:
            if np.cross(np.append(vector1, 0), np.append(vector2, 0))[2]  < 0:
                angulo_radianes = np.arctan2(*vector1)
            else:
                angulo_radianes = np.arctan2(*vector2)
        elif abs(np.dot(vector3,vector2)) <= epsilon:
            if np.cross(np.append(vector3, 0), np.append(vector2, 0))[2]  < 0:
                angulo_radianes = np.arctan2(*vector3)
            else:
                angulo_radianes = np.arctan2(*vector2)
        elif abs(np.dot(vector1,vector3)) <= epsilon:
            if np.cross(np.append(vector1, 0), np.append(vector3, 0))[2]  < 0:
                angulo_radianes = np.arctan2(*vector1)
            else:
                angulo_radianes = np.arctan2(*vector3)
        else:
            angulo_radianes = np.arctan2(*vector3)
        angulo_grados = 180 - np.degrees(angulo_radianes)
        return angulo_grados

    rotation_angle = get_rotacion_angle(dots)

    heigth, width = image.shape[:2]
    center = (heigth // 2, width // 2)
    matrix_rot = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    image_rotated = cv2.warpAffine(image, matrix_rot, (heigth, width), flags=cv2.INTER_LINEAR,  borderValue=(255,255,255))

    return image_rotated
def gen_qr_code(image):
    return image
