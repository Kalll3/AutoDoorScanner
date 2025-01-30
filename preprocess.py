import cv2
import os

# Path to dataset
DATASET_PATH = "dataset/"

# Function to preprocess images
def preprocess_images(dataset_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    for person in os.listdir(dataset_path):  # Loop through each person's folder
        person_path = os.path.join(dataset_path, person)

        for img_name in os.listdir(person_path):  # Loop through each image
            img_path = os.path.join(person_path, img_name)

            # Load image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Skipping {img_name} (Not a valid image)")
                continue

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                print(f"No face detected in {img_name}, skipping...")
                continue

            # Crop and resize the first detected face
            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]  # Crop the face
                face = cv2.resize(face, (160, 160))  # Resize to 160x160 pixels
                save_path = os.path.join(person_path, f"processed_{img_name}")
                cv2.imwrite(save_path, face)
                print(f"Processed and saved: {save_path}")
                break  # Process only the first face detected in the image

# Run preprocessing
preprocess_images(DATASET_PATH)
