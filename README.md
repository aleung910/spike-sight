# 🏐 SpikeSight

**Your AI-Powered Volleyball Serve Coach**

SpikeSight is a desktop GUI tool built with Python that acts as a digital volleyball coach. By uploading a video of a serve, athletes receive real-time skeletal tracking, detailed biomechanical analysis, and AI-powered feedback to improve their technique.

The application deconstructs the complex motion of a volleyball serve into understandable metrics and actionable advice, helping players identify areas for improvement in their form, power, and consistency.

![gif](https://github.com/user-attachments/assets/248f0159-b2af-4cc0-abe0-c683205d7e1e)

---
## 📁 Project Structure    
```
spikesight/
├── main.py                      # Application entry point and GUI controller
├── requirements.txt             # Project dependencies
├── .env                         # Environment variables (API keys)
├── .gitignore                   # Git ignore rules
├── backend/
│   ├── __init__.py
│   ├── analysis_engine.py       # Biomechanical analysis and phase detection
│   └── api_helper.py            # Handles communication with OpenAI API
├── vision/
    ├── __init__.py
    └── video_processor.py       # QThread worker for video & pose detection
```
---
## Features
- **Real-Time Pose Detection**: Uses MediaPipe to track 33 body landmarks frame-by-frame
- **Biomechanical Analysis**: Measures elbow angles, shoulder abduction, wrist velocity, and contact height
- **Phase Detection**: Automatically identifies serve phases (Stance → Trophy Pose → Acceleration → Contact → Follow-Through)
- **AI Coaching**: OpenAI GPT-4o analyzes your technique and provides personalized feedback
- **Detailed Metrics**: Frame-by-frame data export for advanced analysis

## 🚀 Installation

### Prerequisites

- **Python 3.8+** (3.11 recommended)
- **pip** (Python package manager)
- **OpenAI API Key** ([Get one here](https://platform.openai.com/api-keys))

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/spikesight.git
cd spikesight
```

### Step 2: Create a Virtual Environment

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the project root:

```bash
touch .env
```

Add your OpenAI API key to this file:

```env
OPENAI_API_KEY=sk-your-api-key-here
```
---

## 🎮 Usage

### Run the Application

From the project root directory:

```bash
python main.py
```

### Using SpikeSight

1. **Open Video**: Click "Open Video" and select a video file (`.mp4`, `.mov`, `.avi`)
2. **Watch Analysis**: The video will play with skeletal tracking overlaid
3. **Pause/Resume**: Use the "Pause" button to examine specific frames
4. **Review Feedback**: After processing, view:
   - Detected phases (trophy pose, ball contact)
   - Automated biomechanical feedback
   - AI-powered coaching recommendations
```
---
## Technical Details

### How It Works

1. **Video Processing** (`video_processor.py`):
   - Runs in a separate QThread to prevent GUI freezing
   - Processes video frame-by-frame using OpenCV
   - Applies MediaPipe Pose detection to extract 33 landmarks per frame

2. **Biomechanical Analysis** (`analysis_engine.py`):
   - Calculates joint angles using 3D vector mathematics
   - Tracks key metrics: elbow flexion, shoulder abduction, wrist velocity
   - Implements state machine for phase detection
   - Exports comprehensive frame-by-frame data

3. **AI Integration** (`api_helper.py`):
   - Sends biomechanical data to OpenAI GPT-4o
   - Provides context about measurements and ideal ranges
   - Receives personalized coaching feedback

4. **GUI** (`main.py`):
   - Built with PyQt5 for responsive desktop interface
   - Real-time video display with pose overlay
   - Threaded AI analysis to keep UI responsive

### Key Measurements

| Metric | Description | Ideal Range |
|--------|-------------|-------------|
| **Elbow Angle** | Angle at elbow joint during trophy pose | 90-110° |
| **Shoulder Abduction** | Arm angle from vertical at contact | 120-150° |
| **Elbow Extension** | Arm straightness at contact | 170-180° |
| **Wrist Velocity** | Speed of wrist movement | Higher = more power |

---

## 📦 Dependencies

```txt
PyQt5==5.15.10
opencv-contrib-python==4.8.1.78
mediapipe==0.10.8
numpy==1.24.3
openai==1.12.0
python-dotenv==1.0.0
```
---
## Acknowledgments

- **MediaPipe** by Google for pose detection technology
- **OpenAI** for GPT-4o API
- Volleyball coaching research and biomechanics literature
---
## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
