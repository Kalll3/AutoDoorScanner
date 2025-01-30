from flask import Flask, render_template, Response
import cv2
import face_recognition
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load known encodings
with open("face_encodings.pkl", "rb") as f:
    data = pickle.load(f)

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

# Function to capture video frames and perform face recognition
def gen_frames():
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Process frame for face recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(data["encodings"], face_encoding)
            name = "Unknown"
            if True in matches:
                match_idx = matches.index(True)
                name = data["names"][match_idx]

            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Convert the frame to JPEG and return as a response for the webpage
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for video streaming
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
