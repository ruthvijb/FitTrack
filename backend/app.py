from flask import Flask, Response, request
from flask_cors import CORS
import cv2
import mediapipe as mp
import threading
#import pyttsx3

from detect import score_table
from types_of_exercise import TypeOfExercise

app = Flask(__name__)
CORS(app)

#engine = pyttsx3.init()
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

#def speak_async(message):
    #threading.Thread(target=lambda: engine.say(message) or engine.runAndWait()).start()

def generate_frames(exercise_type, video_source=None):
    cap = cv2.VideoCapture("Exercise Videos/" + video_source) if video_source else cv2.VideoCapture(0)
    cap.set(3, 800)
    cap.set(5, 480)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        counter = 0
        status = True

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (800, 480))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = pose.process(rgb_frame)
            rgb_frame.flags.writeable = True
            frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                exercise = TypeOfExercise(landmarks)

                exercise_methods = {
                    "squat": exercise.squat,
                    "push-up": exercise.push_up,
                    "pull-up": exercise.pull_up,
                    "sit-up": exercise.sit_up,
                    "walk": exercise.walk,
                }

                if exercise_type not in exercise_methods:
                    raise ValueError(f"Exercise type '{exercise_type}' not supported.")

                counter, status, feedback_dict = exercise_methods[exercise_type](counter, status)

                y_offset = 50
                for msg in feedback_dict:
                    color = (0, 255, 0) if "should" in msg or "grounded" in msg else (0, 0, 255)
                    cv2.putText(frame, msg, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    #speak_async(msg)
                    y_offset += 30

            except Exception as e:
                print(f"[Error] {e}")

            frame = score_table(exercise_type, frame, counter, status)

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            )

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    exercise_type = request.args.get("exercise_type", "squat")
    video_source = request.args.get("video_source", None)
    return Response(generate_frames(exercise_type, video_source),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
