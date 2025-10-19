# backend/analysis_engine.py
import numpy as np
from PyQt5.QtCore import QObject, pyqtSlot, pyqtSignal
import mediapipe as mp
from enum import Enum

mp_pose = mp.solutions.pose

class ServePhase(Enum):
    STANCE = 0
    ARM_COCKING = 1
    ACCELERATION = 2
    BALL_CONTACT = 3
    FOLLOW_THROUGH = 4

class AnalysisEngine(QObject):
    analysis_complete = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.frame_count = 0
        self.current_phase = ServePhase.STANCE
        
        # Store metrics at key phases
        self.phase_metrics = {
            ServePhase.STANCE: {},
            ServePhase.ARM_COCKING: {},
            ServePhase.ACCELERATION: {},
            ServePhase.BALL_CONTACT: {},
            ServePhase.FOLLOW_THROUGH: {}
        }
        
        # Tracking variables
        self.prev_wrist_y = None
        self.max_wrist_velocity = 0
        self.contact_frame = None
        self.min_elbow_angle = 180
        self.min_elbow_frame = None
        self.max_arm_height = 1.0  # Start high (y decreases going up)
        self.arm_raising = False
        
        # Store all frame data for better analysis
        self.all_frame_data = []
    
    @pyqtSlot(object, np.ndarray)
    def process_frame(self, landmarks, frame):
        self.frame_count += 1
        
        try:
            landmark_list = landmarks.landmark
            
            right_shoulder = landmark_list[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            right_elbow = landmark_list[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            right_wrist = landmark_list[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            right_hip = landmark_list[mp_pose.PoseLandmark.RIGHT_HIP.value]
            
            elbow_angle = self.calculate_angle_3d(right_shoulder, right_elbow, right_wrist)
            shoulder_abduction = self.calculate_shoulder_abduction_improved(right_shoulder, right_elbow, right_hip)
            wrist_height = right_wrist.y
            
            wrist_velocity = 0
            if self.prev_wrist_y is not None:
                wrist_velocity = abs(self.prev_wrist_y - right_wrist.y) * 30
            self.prev_wrist_y = right_wrist.y
            
            frame_data = {
                'frame': self.frame_count,
                'elbow_angle': elbow_angle,
                'shoulder_abduction': shoulder_abduction,
                'wrist_height': wrist_height,
                'wrist_velocity': wrist_velocity
            }
            self.all_frame_data.append(frame_data)
            
            self.update_phase(elbow_angle, shoulder_abduction, wrist_height, wrist_velocity)
            
            if self.frame_count % 5 == 0:
                print(f"Frame {self.frame_count}: Phase={self.current_phase.name}, Elbow={elbow_angle:.1f}°, Shoulder={shoulder_abduction:.1f}°, Height={wrist_height:.3f}, Vel={wrist_velocity:.3f}")
            
        except Exception as e:
            if self.frame_count % 30 == 0:
                print(f"Frame {self.frame_count}: Processing error - {e}")
    
    def update_phase(self, elbow_angle, shoulder_abduction, wrist_height, wrist_velocity):
        """Improved state machine logic for serve phase detection"""
        
        if self.current_phase == ServePhase.STANCE:
            # Detect arm raising (wrist moving up)
            if wrist_height < 0.6 and shoulder_abduction > 60:
                self.current_phase = ServePhase.ARM_COCKING
                self.arm_raising = True
                print(f"\n>>> Transitioned to ARM_COCKING at frame {self.frame_count}")
                print(f"    Wrist height: {wrist_height:.3f}, Shoulder: {shoulder_abduction:.1f}°")
        
        elif self.current_phase == ServePhase.ARM_COCKING:
            # Track the min elbow angle (maximum cocking)
            if elbow_angle < self.min_elbow_angle:
                self.min_elbow_angle = elbow_angle
                self.min_elbow_frame = self.frame_count
            
            # Track max arm height
            if wrist_height < self.max_arm_height:
                self.max_arm_height = wrist_height
            
            # Trophy pose is when arm is highest and elbow is around 90-120°
            # Transition when elbow starts extending rapidly (angle increasing)
            if self.min_elbow_angle < 130 and elbow_angle > self.min_elbow_angle + 10:
                self.phase_metrics[ServePhase.ARM_COCKING] = {
                    'frame': self.min_elbow_frame,
                    'elbow_flexion': self.min_elbow_angle,
                    'wrist_height': self.max_arm_height
                }
                self.current_phase = ServePhase.ACCELERATION
                print(f">>> Transitioned to ACCELERATION at frame {self.frame_count}")
                print(f"    Trophy pose was at frame {self.min_elbow_frame} with elbow {self.min_elbow_angle:.1f}°")
        
        elif self.current_phase == ServePhase.ACCELERATION:
            # Track maximum wrist velocity (indicates contact point)
            if wrist_velocity > self.max_wrist_velocity:
                self.max_wrist_velocity = wrist_velocity
                self.contact_frame = self.frame_count
            
            # Only transition after significant velocity build-up
            # Contact happens when velocity starts decreasing after peak
            if (wrist_velocity < self.max_wrist_velocity * 0.6 and 
                self.max_wrist_velocity > 0.3 and
                self.frame_count > self.contact_frame + 3):  
                
                self.phase_metrics[ServePhase.BALL_CONTACT] = {
                    'frame': self.contact_frame,
                    'shoulder_abduction': shoulder_abduction,
                    'elbow_extension': elbow_angle,
                    'max_velocity': self.max_wrist_velocity
                }
                self.current_phase = ServePhase.BALL_CONTACT
                print(f">>> Transitioned to BALL_CONTACT at frame {self.contact_frame}")
                print(f"    Max velocity: {self.max_wrist_velocity:.3f}, Shoulder: {shoulder_abduction:.1f}°")
        
        elif self.current_phase == ServePhase.BALL_CONTACT:
            self.current_phase = ServePhase.FOLLOW_THROUGH
            print(f">>> Transitioned to FOLLOW_THROUGH at frame {self.frame_count}")
    
    def calculate_angle_3d(self, a, b, c):
        """Calculates the angle at point b between points a-b-c"""
        a = np.array([a.x, a.y, a.z])
        b = np.array([b.x, b.y, b.z])
        c = np.array([c.x, c.y, c.z])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def calculate_shoulder_abduction_improved(self, shoulder, elbow, hip):
        """Improved shoulder abduction - angle from vertical"""
        # Create vertical reference (straight up)
        vertical = np.array([0, -1, 0])  # Negative y is up
        
        arm_vector = np.array([
            elbow.x - shoulder.x,
            elbow.y - shoulder.y,
            elbow.z - shoulder.z
        ])
        
        cosine_angle = np.dot(vertical, arm_vector) / np.linalg.norm(arm_vector)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    @pyqtSlot()
    def finalize_analysis(self):
        """Generate comprehensive feedback"""
        feedback = self.generate_feedback()
        self.analysis_complete.emit(feedback)
    
    def export_frame_data(self):    
        return {
              'total_frames': self.frame_count,
            'phases': {
                'trophy_pose': self.phase_metrics[ServePhase.ARM_COCKING],
                'ball_contact': self.phase_metrics[ServePhase.BALL_CONTACT],
            },
            'all_frames': self.all_frame_data,
            'summary_stats': {
                'min_elbow_angle': self.min_elbow_angle,
                'min_elbow_frame': self.min_elbow_frame,
                'max_wrist_velocity': self.max_wrist_velocity,
                'contact_frame': self.contact_frame,
            }
        }

    def generate_feedback(self):
        """Generate detailed feedback based on detected phases"""
        feedback = {
            'title': 'Serve Analysis Complete',
            'phases_detected': [],
            'recommendations': []
        }
        
        # Trophy Pose 
        if self.phase_metrics[ServePhase.ARM_COCKING]:
            trophy = self.phase_metrics[ServePhase.ARM_COCKING]
            elbow_angle = trophy['elbow_flexion']
            
            feedback['phases_detected'].append(
                f"Trophy Pose (Frame {trophy['frame']}): Elbow at {elbow_angle:.1f}°"
            )
            
            if elbow_angle < 80:
                feedback['recommendations'].append({
                    'title': 'Elbow Too Bent in Trophy Pose',
                    'advice': f"Your elbow was at {elbow_angle:.1f}° (target: 90-110°). This reduces your power arc. Form a 'bow and arrow' shape with your arm more extended."
                })
            elif elbow_angle > 130:
                feedback['recommendations'].append({
                    'title': 'Elbow Too Straight in Trophy Pose',
                    'advice': f"Your elbow was at {elbow_angle:.1f}° (target: 90-110°). Bend your elbow more to create a powerful cocking position."
                })
            else:
                feedback['phases_detected'].append("✓ Good trophy pose elbow angle")
        
        if self.phase_metrics[ServePhase.BALL_CONTACT]:
            contact = self.phase_metrics[ServePhase.BALL_CONTACT]
            shoulder_angle = contact['shoulder_abduction']
            elbow_ext = contact['elbow_extension']
            
            feedback['phases_detected'].append(
                f"Ball Contact (Frame {contact['frame']}): Shoulder {shoulder_angle:.1f}°, Elbow {elbow_ext:.1f}°"
            )
            
            if shoulder_angle < 100:
                feedback['recommendations'].append({
                    'title': 'Low Contact Point',
                    'advice': f"Shoulder angle was {shoulder_angle:.1f}° (target: 120-150°). Reach higher! Contact the ball at full extension for better power and angle."
                })
            
            if elbow_ext < 160:
                feedback['recommendations'].append({
                    'title': 'Incomplete Arm Extension',
                    'advice': f"Your elbow was at {elbow_ext:.1f}° at contact (target: 170-180°). Fully extend your arm for maximum reach and power."
                })
        
        if not feedback['recommendations']:
            feedback['recommendations'].append({
                'title': 'Excellent Form!',
                'advice': 'Your serve mechanics look solid. Keep practicing to maintain consistency.'
            })
        
        return feedback