import cv2
import numpy as np
import os

def preprocess_image(img_path, target_size=(224,224)):
    if not os.path.exists(img_path):
        return None
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    return img
