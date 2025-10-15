# vision/video_processor.py
from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
import mediapipe as mp
import cv2

class VideoProcessor(QThread):
    frame_processed = pyqtSignal(np.ndarray)
    pose_data_extracted = pyqtSignal(object, np.ndarray)
    processing_finished = pyqtSignal()

    def __init__(self, video_path):
        super().__init__()
        self._run_flag = True
        self.video_path = video_path

    def run(self):
        # --- Initialization moved outside the loop for efficiency ---
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        mp_drawing = mp.solutions.drawing_utils
        
        cap = cv2.VideoCapture(self.video_path)
        
        while cap.isOpened() and self._run_flag:
            ret, frame = cap.read()
            if not ret:
                break
            
            # MediaPipe Processing Logic 
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            # Draw the pose annotation on the og BGR frame
            annotated_frame = frame.copy()
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS)
                
                # Emit raw landmark data for analysis 
                self.pose_data_extracted.emit(results.pose_landmarks, frame)

            # Emit wtih landmarks drawn on it
            self.frame_processed.emit(annotated_frame)

        cap.release()
        pose.close()
        self.processing_finished.emit()

    def stop(self):
        """Sets run flag to False and waits for the thread to finish."""
        self._run_flag = False
        self.wait()