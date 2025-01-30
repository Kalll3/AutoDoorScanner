import cv2
import os
import random

DATASET_PATH = "dataset/"

def show_random_images(dataset_path):
    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        images = [img for img in os.listdir(person_path) if img.startswith("processed_")]

        if len(images) > 0:
            img_path = os.path.join(person_path, random.choice(images))
            img = cv2.imread(img_path)
            cv2.imshow(f"Sample Face from {person}", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

show_random_images(DATASET_PATH)
