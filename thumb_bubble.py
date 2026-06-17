import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            lm = face_landmarks.landmark
            
            def get_dist(p1, p2):
                x1, y1 = int(lm[p1].x * w), int(lm[p1].y * h)
                x2, y2 = int(lm[p2].x * w), int(lm[p2].y * h)
                return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # --- CALCULATE DISTANCES ---
            lip_width = get_dist(61, 291)     # Left to right lip corners
            lip_height = get_dist(13, 14)     # Inner top lip to inner bottom lip
            eye_height = get_dist(159, 145)   # Top to bottom of right eye
            
            # Distance from inner eyebrow to the bridge of the nose
            eyebrow_to_nose = get_dist(70, 6) 

            # --- THE SECRET SAUCE: RATIOS ---
            # Using lip_width as a baseline prevents distance/zoom issues
            mouth_open_ratio = lip_height / max(1.0, lip_width)
            smile_ratio = lip_width / max(1.0, eye_height)

            # Default state
            emotion = "NEUTRAL"
            box_color = (255, 255, 0) # Cyan

            # --- CALIBRATED EMOTION LOGIC ---
            
            # 1. SURPRISED: Mouth drops open significantly (height is large compared to width)
            if mouth_open_ratio > 0.45:
                emotion = "SURPRISED!"
                box_color = (0, 255, 255) # Yellow
                
            # 2. HAPPY: Mouth stretches wide horizontally
            elif smile_ratio > 5.5:
                emotion = "HAPPY :D"
                box_color = (0, 255, 0) # Green
                
            # 3. SAD: Corners of mouth drop, or eyebrows furrow downward/closer to nose
            elif eyebrow_to_nose < 22.0 and mouth_open_ratio < 0.15:
                emotion = "SAD :("
                box_color = (255, 0, 0) # Blue

            # --- DRAW BOUNDING BOX ---
            all_x = [int(landmark.x * w) for landmark in face_landmarks.landmark]
            all_y = [int(landmark.y * h) for landmark in face_landmarks.landmark]
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)

            cv2.rectangle(frame, (x_min - 10, y_min - 20), (x_max + 10, y_max + 10), box_color, 2)
            cv2.putText(frame, emotion, (x_min, y_min - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

    cv2.imshow('Face Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
