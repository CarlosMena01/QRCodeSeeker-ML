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
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            new_image = image[int(y1):int(y2), int(x1):int(x2)]
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
            return new_image
        else:
            print(f"Little score: {score}")
    return None
def extract_qr_code(image):
    return image
def gen_qr_code(image):
    return image