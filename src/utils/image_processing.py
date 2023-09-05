from ultralytics import YOLO
import cv2
import numpy as np
from os.path import dirname as up

PATH_ROOT = up(up(up(__file__)))

PATH_DETECTION = PATH_ROOT + "/models/best_detection.pt"
PATH_PATTERN = PATH_ROOT + "/models/best_pattern.pt"


model_detection = YOLO(PATH_DETECTION)
model_pattern = YOLO(PATH_PATTERN)


def detect_qr_code(image, model=model_detection, threshold=0.3):
    results = model(image)[0]
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, _ = result
        if score > threshold:
            new_image = image[int(y1):int(y2), int(x1):int(x2)]
            return new_image
        else:
            print(f"Little score: {score}")
    return None


def extract_qr_code(image, model=model_pattern, threshold=0.5):
    results = model(image)[0]

    dots = []

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, _ = result
        if score > threshold:
            dot = (int((x1+x2)/2), int((y1+y2)/2))
            dots.append(dot)

    def get_rotacion_angle(coordenates):
        p1, p2, p3 = coordenates
        vector1 = np.array(p2) - np.array(p3)
        vector2 = np.array(p1) - np.array(p3)
        vector3 = np.array(p1) - np.array(p2)

        def min_dot_product(v1, v2, v3):
            # Calculate dot products between all pairs of vectors
            dot_product_v1_v2 = np.abs(np.dot(v1, v2))
            dot_product_v1_v3 = np.abs(np.dot(v1, v3))
            dot_product_v2_v3 = np.abs(np.dot(v2, v3))

            # Create a dictionary to map dot products to their respective vector pairs
            dot_products = {
                dot_product_v1_v2: (v1, v2),
                dot_product_v1_v3: (v1, v3),
                dot_product_v2_v3: (v2, v3)
            }

            # Find the pair with the minimum absolute dot product
            min_dot_product = min(dot_products.keys())
            min_vectors = dot_products[min_dot_product]

            return min_vectors

        minvectors = min_dot_product(vector1, vector2, vector3)

        minvectors = np.asarray(minvectors)
        if np.any(np.all(vector1 == minvectors, axis=1)):
            if vector3 in minvectors:
                vector1 = -vector1
        elif np.any(np.all(vector2 == minvectors, axis=1)):
            if vector3 in minvectors:
                vector2 = -vector2
                vector3 = -vector3

        if np.cross(np.append(minvectors[0], 0), np.append(minvectors[1], 0))[2] > 0:
            angulo_radianes = np.arctan2(*minvectors[0])
        else:
            angulo_radianes = np.arctan2(*minvectors[1])

        angulo_grados = -90 - np.degrees(angulo_radianes)
        print(angulo_grados)
        return angulo_grados

    rotation_angle = get_rotacion_angle(dots)

    heigth, width = image.shape[:2]
    center = (heigth // 2, width // 2)
    matrix_rot = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    image_rotated = cv2.warpAffine(
        image, matrix_rot, (heigth, width), flags=cv2.INTER_LINEAR,  borderValue=(255, 255, 255))

    return image_rotated


def gen_qr_code(image):
    return image
