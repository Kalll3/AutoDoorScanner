import face_recognition
import numpy as np
import os
import pickle

def encode_faces(dataset_path):
    known_encodings = []
    known_names = []

    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = face_recognition.load_image_file(img_path)
            encoding = face_recognition.face_encodings(img)

            if len(encoding) > 0:
                known_encodings.append(encoding[0])
                known_names.append(person)

    data = {"encodings": known_encodings, "names": known_names}
    with open("face_encodings.pkl", "wb") as f:
        pickle.dump(data, f)

encode_faces("dataset/")
