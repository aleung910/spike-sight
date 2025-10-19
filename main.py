import sys
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget, QTextEdit, QHBoxLayout)
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSlot, QThread, pyqtSignal
import numpy as np

from vision.video_processor import VideoProcessor
from backend.analysis_engine import AnalysisEngine

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpikeSight")
        self.setGeometry(700, 300, 1000, 700)
        self.setWindowIcon(QIcon("MOLTEN.png"))
        self.video_thread = None
        self.analysis_engine = None
        self.initUI()
    
    def initUI(self):
        central_widget = QWidget()
        central_widget.setStyleSheet("background-color: #2b2b2b;")
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)
        
        # Video display area
        self.video_label = QLabel("Please open a video file to begin.", self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #555;
                background-color: #1e1e1e;
                font-size: 16px;
                color: #aaa;
            }
        """)
        self.video_label.setMinimumHeight(450)
        self.layout.addWidget(self.video_label, 3)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.open_button = QPushButton("Open Video", self)
        self.open_button.setStyleSheet("""
            QPushButton { 
                font-size: 16px; 
                padding: 10px;
                background-color: #404040;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #505050;
            }
        """)
        
        button_layout.addWidget(self.open_button)
        self.layout.addLayout(button_layout)
        
        self.status_label = QLabel("", self)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("QLabel { font-size: 12px; padding: 3px; color: #888; }")
        self.layout.addWidget(self.status_label)
        
        # Feedback area
        self.feedback_text = QTextEdit(self)
        self.feedback_text.setReadOnly(True)
        self.feedback_text.setStyleSheet("""
            QTextEdit {
                font-size: 13px;
                padding: 10px;
                background-color: #1e1e1e;
                border: 1px solid #444;
                border-radius: 5px;
                color: #ddd;
            }
        """)
        self.feedback_text.setMinimumHeight(250)
        self.feedback_text.setHtml("")
        self.layout.addWidget(self.feedback_text, 1)
        
        # Connect buttons
        self.open_button.clicked.connect(self.open_video_file)
    
    def open_video_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video File",
            "",
            "Video Files (*.mp4 *.mov *.avi);;All Files (*)"
        )
        
        if filepath:
            if self.video_thread and self.video_thread.isRunning():
                self.video_thread.stop()
            
            self.feedback_text.setHtml("")
            self.status_label.setText("Processing...")
            
            self.video_thread = VideoProcessor(filepath)
            self.analysis_engine = AnalysisEngine()
            
            self.video_thread.frame_processed.connect(self.update_image)
            self.video_thread.pose_data_extracted.connect(self.analysis_engine.process_frame)
            self.video_thread.processing_finished.connect(self.on_processing_finished)
            self.analysis_engine.analysis_complete.connect(self.display_feedback)
            
            self.video_thread.start()
    
    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates video_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an OpenCV image to QPixmap."""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = QPixmap.fromImage(convert_to_Qt_format)
        return p.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)
    
    @pyqtSlot()
    def on_processing_finished(self):
        """Called when video processing is complete"""
        self.status_label.setText("Generating feedback...")
        self.analysis_engine.finalize_analysis()
    
    @pyqtSlot(dict)
    def display_feedback(self, feedback_dict):
        """Display the analysis feedback in the GUI"""
        self.status_label.setText("Analyzing with AI...")
        
        html = self.build_feedback_html(feedback_dict)
        self.feedback_text.setHtml(html)
        
        # Get AI analysis in background
        self.get_ai_analysis(feedback_dict)
    
    def build_feedback_html(self, feedback_dict, ai_feedback=None):
        """Build HTML for feedback display"""
        html = "<div style='font-family: Arial, sans-serif; color: #ddd;'>"
        html += f"<h3 style='color: #4a9eff; margin: 5px 0;'>{feedback_dict.get('title', 'Analysis Complete')}</h3>"
        
        # Show detected phases
        if feedback_dict.get('phases_detected'):
            html += "<p style='margin: 8px 0; color: #bbb;'><b>Detected:</b> "
            html += ", ".join(feedback_dict['phases_detected'])
            html += "</p>"
        
        if feedback_dict.get('recommendations'):
            html += "<div style='margin-top: 10px;'>"
            for rec in feedback_dict['recommendations']:
                html += f"<div style='background-color: #3a3a3a; padding: 8px; margin: 5px 0; border-left: 3px solid #ff9800; border-radius: 3px;'>"
                html += f"<b style='color: #ffa726;'>{rec['title']}</b><br>"
                html += f"<span style='font-size: 12px; color: #ccc;'>{rec['advice']}</span>"
                html += "</div>"
            html += "</div>"
        else:
            html += "<p style='color: #66bb6a; font-size: 14px; margin: 10px 0;'>✓ Good form detected</p>"
        
        if ai_feedback:
            html += "<hr style='border: 1px solid #444; margin: 15px 0;'>"
            html += "<h3 style='color: #66bb6a; margin: 10px 0;'>🤖 AI Coach Analysis</h3>"
            html += f"<div style='background-color: #2a2a2a; padding: 12px; border-radius: 5px; line-height: 1.6;'>"
            formatted_ai = ai_feedback.replace('\n', '<br>')
            html += f"<span style='color: #ddd; font-size: 13px;'>{formatted_ai}</span>"
            html += "</div>"
        
        html += "</div>"
        return html
    
    def get_ai_analysis(self, feedback_dict):
        """Get AI analysis from OpenAI in a separate thread"""
        
        class AIAnalysisThread(QThread):
            analysis_complete = pyqtSignal(str)
            
            def __init__(self, feedback_data):
                super().__init__()
                self.feedback_data = feedback_data
            
            def run(self):
                from backend.api_helper import analyze_with_openai
                try:
                    enhanced = analyze_with_openai(self.feedback_data)
                    if enhanced.get('ai_analysis'):
                        self.analysis_complete.emit(enhanced['ai_analysis'])
                    else:
                        self.analysis_complete.emit("")
                except Exception as e:
                    print(f"AI analysis failed: {e}")
                    self.analysis_complete.emit("")
        
        self.ai_thread = AIAnalysisThread(feedback_dict)
        self.ai_thread.analysis_complete.connect(lambda ai_text: self.update_with_ai(feedback_dict, ai_text))
        self.ai_thread.start()
    
    def update_with_ai(self, feedback_dict, ai_feedback):
        """Update display with AI feedback"""
        self.status_label.setText("Complete")
        if ai_feedback:
            html = self.build_feedback_html(feedback_dict, ai_feedback)
            self.feedback_text.setHtml(html)
    
    def closeEvent(self, event):
        """Clean up when window is closed"""
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