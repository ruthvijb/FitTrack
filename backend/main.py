import cv2
#import pyttsx3
#import threading
import argparse
from detect import *
import mediapipe as mp
from angle_finder import BodyPartAngle
from types_of_exercise import TypeOfExercise

#engine = pyttsx3.init()

#def speak_async(message):
    #engine.say(message)
    #engine.runAndWait()

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--exercise_type", type=str, required=True, help="Type of activity to do")
ap.add_argument("-vs", "--video_source", type=str, required=False, help="Video file name inside Exercise Videos/")
args = vars(ap.parse_args())

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture("Exercise Videos/" + args["video_source"]) if args["video_source"] else cv2.VideoCapture(0)
cap.set(3, 800)
cap.set(5, 480)

# Create a named window with the option to resize
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

# Resize the window to your desired dimensions (e.g., 1200x800)
cv2.resizeWindow('Video', 1200, 800)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    counter = 0
    status = True

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (800, 480))  # Resize the frame if necessary
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = pose.process(rgb_frame)
        rgb_frame.flags.writeable = True
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            if landmarks is None:
                print("No landmarks detected.")
                continue
            
            exercise = TypeOfExercise(landmarks)

            exercise_methods = {
                "squat": exercise.squat,
                "push-up": exercise.push_up,
                "pull-up": exercise.pull_up,
                "sit-up": exercise.sit_up,
                "walk": exercise.walk,
            }

            if args["exercise_type"] not in exercise_methods:
                raise ValueError(f"Exercise type '{args['exercise_type']}' not supported.")

            counter, status, feedback_dict = exercise_methods[args["exercise_type"]](counter, status)

            # Removed the display of status, counter, and exercise type
            # cv2.putText(frame, f"Reps: {counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # cv2.putText(frame, f"Status: {status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            y_offset = 50
            for msg in feedback_dict:
                color = (0, 255, 0) if "should" in msg or "grounded" in msg else (0, 0, 255)
                cv2.putText(frame, msg, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                #threading.Thread(target=speak_async, args=(msg,)).start()  #asynchronous-tts
                y_offset += 30

        except Exception as e:
            print(f"[Error] {e}")

        # Draw landmarks and pose connections
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
        )

        # Display the frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('a'):
            break

    cap.release()
    cv2.destroyAllWindows()
