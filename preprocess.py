import cv2
import sys

def preprocess_image(img_path):
    """
    Loads an image, converts to grayscale, resizes to 224x224, and normalizes pixel values.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Unable to load image {img_path}")
        return None
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  # normalize to 0-1
    return img

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <path_to_image>")
    else:
        image = preprocess_image(sys.argv[1])
        if image is not None:
            print(f"Image preprocessed successfully! Shape: {image.shape}")
