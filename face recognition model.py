import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True, # Gives us detailed eye/lip tracking
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip horizontally for selfie-view
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            lm = face_landmarks.landmark
            
            # Helper function to calculate pixel distance between two landmarks
            def get_dist(p1, p2):
                x1, y1 = int(lm[p1].x * w), int(lm[p1].y * h)
                x2, y2 = int(lm[p2].x * w), int(lm[p2].y * h)
                return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # --- KEY LANDMARKS FOR EMOTIONS ---
            # 61 and 291: Left and Right Corners of the Lips
            # 0 and 17: Top and Bottom Lip centers
            # 159 and 145: Top and Bottom of Right Eye
            # 107 and 9: Right Eyebrow inner edge and Forehead center
            
            lip_width = get_dist(61, 291)
            lip_height = get_dist(0, 17)
            eye_height = get_dist(159, 145)
            eyebrow_drop = get_dist(107, 9)
            
            # Default state
            emotion = "NEUTRAL"
            box_color = (255, 255, 0) # Cyan/Yellow

            # --- EMOTION LOGIC BASED ON RATIOS ---
            # 1. HAPPY: Lip width stretches wide compared to total lip height
            if lip_width > (lip_height * 3.2):
                emotion = "HAPPY :D"
                box_color = (0, 255, 0) # Green
                
            # 2. SURPRISED: Mouth opens wide vertically, eyes open wide
            elif lip_height > (lip_width * 0.5) and eye_height > 12:
                emotion = "SURPRISED!"
                box_color = (0, 255, 255) # Yellow
                
            # 3. SAD: Eyebrows pull down/together, mouth is narrow
            elif eyebrow_drop > 35 and lip_width < (lip_height * 2.5):
                emotion = "SAD :("
                box_color = (255, 0, 0) # Blue

            # --- DRAW THE BOUNDING BOX AND TEXT ---
            # Find rough boundaries of the face for the box
            all_x = [int(landmark.x * w) for landmark in face_landmarks.landmark]
            all_y = [int(landmark.y * h) for landmark in face_landmarks.landmark]
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)

            # Draw rectangle around face
            cv2.rectangle(frame, (x_min - 10, y_min - 20), (x_max + 10, y_max + 10), box_color, 2)
            
            # Label with predicted emotion
            cv2.putText(frame, emotion, (x_min, y_min - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

    # Display the output
    cv2.imshow('Face Emotion Detector', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
