# backend/analysis_engine.py
import numpy as np
from PyQt5.QtCore import QObject, pyqtSlot, pyqtSignal

class AnalysisEngine(QObject):
    analysis_complete = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.frame_count = 0

    @pyqtSlot(object, np.ndarray)
    def process_frame(self, landmarks, frame):
        """
        This slot receives the pose landmarks for each frame from the VideoProcessor.
        """
        self.frame_count += 1
        if self.frame_count % 30 == 0: # Print every 30 frames
            print(f"AnalysisEngine: Received data for frame {self.frame_count}")
            #for now

    def calculate_angle_3d(self, a, b, c):
        """Calculates the angle between three 3D points (angle at point b)."""
        a = np.array([a.x, a.y, a.z])
        b = np.array([b.x, b.y, b.z])
        c = np.array([c.x, c.y, c.z])

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        return np.degrees(angle)