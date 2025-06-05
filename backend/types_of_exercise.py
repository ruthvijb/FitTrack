import numpy as np
from angle_finder import BodyPartAngle
from detect import detection_body_part, calculate_angle
from utils import *


class TypeOfExercise(BodyPartAngle):
    def __init__(self, landmarks):
        super().__init__(landmarks)

    def push_up(self, counter, status):
        left_arm_angle = self.angle_of_the_left_arm()
        right_arm_angle = self.angle_of_the_left_arm()
        avg_arm_angle = (left_arm_angle + right_arm_angle) // 2

        if status:
            if avg_arm_angle < 70:
                counter += 1
                status = False
        else:
            if avg_arm_angle > 160:
                status = True

        return counter, status

    def pull_up(self, counter, status):
        nose = detection_body_part(self.landmarks, "NOSE")
        left_elbow = detection_body_part(self.landmarks, "LEFT_ELBOW")
        right_elbow = detection_body_part(self.landmarks, "RIGHT_ELBOW")
        avg_shoulder_y = (left_elbow[1] + right_elbow[1]) / 2

        if status:
            if nose[1] > avg_shoulder_y:
                counter += 1
                status = False
        else:
            if nose[1] < avg_shoulder_y:
                status = True

        return counter, status

    def squat(self, counter, status):
        feedback = []  # Initialize feedback list

        # 1. Torso Lean - Increased range from (60-75) to (50-85)
        torso_angle = calculate_angle(
            detection_body_part(self.landmarks, "LEFT_SHOULDER"),
            detection_body_part(self.landmarks, "LEFT_HIP"),
            detection_body_part(self.landmarks, "LEFT_KNEE")
        )
        if torso_angle > 180:
            feedback.append("Lean forward slightly")
        elif torso_angle < 50:
            feedback.append("Lean backward slightly")

        # 2. Squat Depth - Increased range from (80-100) to (70-110)
        depth_angle = self.angle_of_the_left_leg()
        if depth_angle > 90:
            feedback.append("Squat deeper")
        elif depth_angle < 60:
            feedback.append("Don't squat too low")

        '''# 3. Knee Alignment - Increased range from (170-180) to (160-190)
        valgus_angle = calculate_angle(
            detection_body_part(self.landmarks, "LEFT_HIP"),
            detection_body_part(self.landmarks, "LEFT_KNEE"),
            detection_body_part(self.landmarks, "LEFT_ANKLE")
        )
        if valgus_angle < 160:
            feedback.append("Knees are caving in")
        elif valgus_angle > 190:
            feedback.append("Knees are bowing out")'''

        # Print angles for debugging
        #print(f"Torso: {torso_angle:.1f}°, Depth: {depth_angle:.1f}° ,Valgus: {valgus_angle:.1f}°")
        print(f"Torso: {torso_angle:.1f}°, Depth: {depth_angle:.1f}°")

        # Only count a rep if form is reasonable
        if status == "up" and 70 <= depth_angle <= 110:
            counter += 1
            status = "down"
        elif depth_angle > 160:
            status = "up"

        return counter, status, feedback

    def walk(self, counter, status):
        right_knee = detection_body_part(self.landmarks, "RIGHT_KNEE")
        left_knee = detection_body_part(self.landmarks, "LEFT_KNEE")

        if status:
            if left_knee[0] > right_knee[0]:
                counter += 1
                status = False
        else:
            if left_knee[0] < right_knee[0]:
                counter += 1
                status = True

        return counter, status

    def sit_up(self, counter, status):
        angle = self.angle_of_the_abdomen()
        if status:
            if angle < 55:
                counter += 1
                status = False
        else:
            if angle > 105:
                status = True

        return counter, status

    def calculate_exercise(self, exercise_type, counter, status):
        if exercise_type == "push-up":
            counter, status = self.push_up(counter, status)
            return counter, status, []
        elif exercise_type == "pull-up":
            counter, status = self.pull_up(counter, status)
            return counter, status, []
        elif exercise_type == "squat":
            return self.squat(counter, status)
        elif exercise_type == "walk":
            counter, status = self.walk(counter, status)
            return counter, status, []
        elif exercise_type == "sit-up":
            counter, status = self.sit_up(counter, status)
            return counter, status, []

        return counter, status, []
