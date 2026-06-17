# OpenCV & MediaPipe Gesture & Emotion Detector

An interactive computer vision application built in Python 3.10 that detects hand gestures and facial emotions live using a webcam. Instead of using heavy machine learning models, this project utilizes precise geometric tracking and facial landmark ratios for fast performance.

## 🚀 Features

### 1. Hand Gesture Tracking (`thumb_bubble.py`)
Tracks 21 hand landmarks to detect 6 specific gestures and displays a dynamic floating status bubble:
- **Thumbs Up** (Displays "GOOD!")
- **Thumbs Down** (Displays "BAD!")
- **High-Five** (Displays "HIGH FIVE!")
- **Punch** (Displays "PUNCH!")
- **Victory/Peace Sign** (Displays "VICTORY!")
- **OK Sign** (Displays "OK!")

### 2. Face Emotion Recognition (`face_emotion.py`)
Tracks 468 facial mesh landmarks to measure real-time feature proportions. It auto-calibrates to your distance from the camera to detect:
- **Happy :D** (Triggered by horizontal mouth stretch ratios)
- **Surprised!** (Triggered by vertical jaw drop and eye widen ratios)
- **Sad :(** (Triggered by furrowed eyebrow and narrow mouth ratios)
- **Neutral** (Default state)

## 🛠️ Tech Stack & Setup

- **Language:** Python 3.10
- **Libraries:** OpenCV, MediaPipe, NumPy 1.x

### Installation
To run these scripts locally, install the exact stable library versions used:
```bash
pip install opencv-python==4.8.1.78 mediapipe==0.10.8 "numpy<2"

📝 How To Run
Run either script in IDLE or your preferred terminal: python face_emotion.py

Focus on the camera pop-up window.

Press 'q' on your keyboard to safely exit the application.


---

### How to add it right now on the website:
1. Go to your repository page: `https://github.com/TechPro-AI007/Gesture-and-Emotion-Detector`
2. Scroll down slightly and look for a green button that says **"Add a README"** (or click the **"Add file"** dropdown and select **"Create new file"**, then name it exactly `README.md`).
3. Paste the markdown block above into the main editor window.
4. Click **Commit changes...** in the top right corner.
5. Click the green **Commit changes** button to save it.

Once you do that, your repository will look clean, official, and ready to show off t
