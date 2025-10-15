import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout,QWidget
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtCore import Qt , pyqtSlot
import numpy as np

from vision.video_processor import VideoProcessor
from backend.analysis_engine import AnalysisEngine

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpikeSight")
        self.setGeometry(700, 300, 800, 600)
        self.setWindowIcon(QIcon("MOLTEN.png"))
        self.video_thread = None

        self.initUI()
    
    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)
        
        self.video_label = QLabel("Please open a video file to begin.",self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            Label {
                border: 2px dashed #aaa;
                background-color: #f0f0f0;
                font-size: 18px;
                color: #555;
            }
        """)
        self.layout.addWidget(self.video_label, 1) 

        self.open_button = QPushButton("Open Video", self)
        self.open_button.setStyleSheet("QPushButton { font-size: 16px; padding: 10px; }")
        self.layout.addWidget(self.open_button)

        self.feedback_label = QLabel("Feedback will appear here.", self)
        self.feedback_label.setAlignment(Qt.AlignCenter)
        self.feedback_label.setStyleSheet("QLabel { font-size: 16px; font-weight: bold; padding: 10px; }")
        self.layout.addWidget(self.feedback_label)

        self.open_button.clicked.connect(self.open_video_file)

    def open_video_file(self):
        """Opens a file dialog to select a video file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video File",
            "", 
            "Video Files (*.mp4 *.mov *.avi);;All Files (*)"
        )

        if filepath:
            if self.video_thread and self.video_thread.isRunning():
                self.video_thread.stop()

            self.feedback_label.setText("Analyzing video...")

            self.video_thread = VideoProcessor(filepath)
            self.analysis_engine = AnalysisEngine()

            self.video_thread.frame_processed.connect(self.update_image)
            self.video_thread.pose_data_extracted.connect(self.analysis_engine.process_frame)
            self.video_thread.processing_finished.connect(self.on_processing_finished)

            self.video_thread.start()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        #Updates video_label with a new opencv image
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)

    
    def convert_cv_qt(self, cv_img):
        #"Convert from an OpenCV image to QPixmap."
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = QPixmap.fromImage(convert_to_Qt_format)
        # Scale the pixmap to fit the label while maintaining aspect ratio
        return p.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)

    @pyqtSlot()
    def on_processing_finished(self):
        self.feedback_label.setText("Analysis Complete!")
        self.video_label.setText("Processing Finished.")

    def closeEvent(self, event):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
