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

        triangle = (vector1, vector2, vector3)

        def min_dot_product(list_vectors):
            # Take a list of vectors and return the index of the mins dot product pair
            v1, v2, v3 = list_vectors
            # Calculate dot products between all pairs of vectors
            dot_product_v1_v2 = np.abs(np.dot(v1, v2))
            dot_product_v1_v3 = np.abs(np.dot(v1, v3))
            dot_product_v2_v3 = np.abs(np.dot(v2, v3))

            # Create a dictionary to map dot products to their respective vector pairs
            dot_products = {
                dot_product_v1_v2: (0, 1),
                dot_product_v1_v3: (0, 2),
                dot_product_v2_v3: (1, 2)
            }

            # Find the pair with the minimum absolute dot product
            min_dot_product = min(dot_products.keys())
            min_vectors = dot_products[min_dot_product]

            return min_vectors

        minvectors_indexs = min_dot_product(triangle)

        if 0 in minvectors_indexs:
            if 2 in minvectors_indexs:
                vector1 = -vector1
        elif 1 in minvectors_indexs:
            if 2 in minvectors_indexs:
                vector2 = -vector2
                vector3 = -vector3
        # Update the triangle
        triangle = (vector1, vector2, vector3)
        if np.cross(np.append(triangle[minvectors_indexs[0]], 0), np.append(triangle[minvectors_indexs[1]], 0))[2] < 0:
            angulo_radianes = np.arctan2(
                triangle[minvectors_indexs[0]][1], triangle[minvectors_indexs[0]][0])
        else:
            angulo_radianes = np.arctan2(
                triangle[minvectors_indexs[1]][1], triangle[minvectors_indexs[1]][0])

        angulo_grados = - 90 + np.degrees(angulo_radianes)
        return angulo_grados

    rotation_angle = get_rotacion_angle(dots)

    heigth, width = image.shape[:2]
    center = (heigth // 2, width // 2)
    matrix_rot = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    image_rotated = cv2.warpAffine(
        image, matrix_rot, (heigth, width), flags=cv2.INTER_LINEAR,  borderValue=(255, 255, 255))

    return image_rotated


def post_processing(image, m=5):
    """Apply the posprocessing to a image

    Args:
        image (RGB image): The source image on opencv format
        m (int): The border size in pixels. Defaults to 5.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarice the image
    img_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]
    img_bin = cv2.morphologyEx(
        img_bin, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    # Add white borders
    output = cv2.copyMakeBorder(
        img_bin, m, m, m, m, cv2.BORDER_CONSTANT, value=255)

    output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

    return output
