# backend/analysis_engine.py
import numpy as np
from PyQt5.QtCore import QObject, pyqtSlot, pyqtSignal
import mediapipe as mp

mp_pose = mp.solutions.pose

class AnalysisEngine(QObject):
    analysis_complete = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.frame_count = 0
    
    @pyqtSlot(object, np.ndarray)
    def process_frame(self, landmarks, frame):
        self.frame_count += 1
        
        try:
            landmark_list = landmarks.landmark
            
            # Access landmarks by index VALUE
            right_shoulder = landmark_list[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            right_elbow = landmark_list[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            right_wrist = landmark_list[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            
            right_elbow_angle = self.calculate_angle_3d(right_shoulder, right_elbow, right_wrist)
            
            if self.frame_count % 5 == 0:
                print(f"Frame {self.frame_count}: Right Elbow Angle = {right_elbow_angle:.2f} degrees")
                
        except Exception as e:
            try:
                landmark_list = landmarks.landmark
                left_shoulder = landmark_list[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                left_elbow = landmark_list[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                left_wrist = landmark_list[mp_pose.PoseLandmark.LEFT_WRIST.value]
                
                left_elbow_angle = self.calculate_angle_3d(left_shoulder, left_elbow, left_wrist)
                
                if self.frame_count % 5 == 0:
                    print(f"Frame {self.frame_count}: Left Elbow Angle = {left_elbow_angle:.2f} degrees")
                    
            except Exception as e:
                if self.frame_count % 5 == 0:
                    print(f"Frame {self.frame_count}: Could not calculate angle. {e}")
    
    def calculate_angle_3d(self, a, b, c):
        """Calculates the angle between three 3D points (angle at point b)."""
        a = np.array([a.x, a.y, a.z])
        b = np.array([b.x, b.y, b.z])  # Fixed: was a.z
        c = np.array([c.x, c.y, c.z])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)