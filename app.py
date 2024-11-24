from flask import Flask, render_template, Response, request, jsonify
import cv2
import face_recognition
import pickle
import firebase_admin
from firebase_admin import credentials, db
import numpy as np

# Initialize Flask
app = Flask(__name__)

# Firebase Initialization
cred = credentials.Certificate("serviceaccount.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://faceattendencerealtime-f66ae-default-rtdb.firebaseio.com/"
})

# Load face encodings
with open('encodedFile.p', 'rb') as file:
    encodeListunknownwithIds = pickle.load(file)
encodeListknown, studentIds = encodeListunknownwithIds

# Global Variables
cap = None

def gen_frames():
    """Generate frames for live video feed."""
    global cap
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Stream the video feed."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_attendance', methods=['POST'])
def start_attendance():
    """Start attendance marking."""
    global cap
    if not cap or not cap.isOpened():
        return jsonify({"error": "Camera is not started!"}), 400

    success, frame = cap.read()
    if not success:
        return jsonify({"error": "Failed to capture frame!"}), 500

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    attendance_marked = []
    student_details = []
    for encodeFace, faceLoc in zip(encodings, face_locations):
        matches = face_recognition.compare_faces(encodeListknown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListknown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            student_id = studentIds[matchIndex]
            student_info = db.reference(f'Students/{student_id}').get()
            attendance_marked.append(student_info['name'])
            student_details.append(student_info)

    return jsonify({
        "attendance": attendance_marked,
        "student_details": student_details
    })

if __name__ == "__main__":
    app.run(debug=True)
