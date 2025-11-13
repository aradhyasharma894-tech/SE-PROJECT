import random

def predict_image(processed_img):
    labels = ["Normal", "Pneumonia", "COVID-19"]
    label = random.choice(labels)
    conf = round(0.6 + random.random()*0.4, 2)
    return label, conf
