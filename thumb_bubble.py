import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Open Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw lines on the hand
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            lm = hand_landmarks.landmark
            
            # 1. Determine if fingers are completely straight (Tip is ABOVE the joint)
            index_straight  = lm[8].y < lm[6].y
            middle_straight = lm[12].y < lm[10].y
            ring_straight   = lm[16].y < lm[14].y
            pinky_straight  = lm[20].y < lm[18].y
            
            # Determine if fingers are folded (Tip is BELOW the joint)
            index_folded  = lm[8].y > lm[6].y
            middle_folded = lm[12].y > lm[10].y
            ring_folded   = lm[16].y > lm[14].y
            pinky_folded  = lm[20].y > lm[18].y

            # 2. Track Thumb positions
            thumb_upright   = lm[4].y < lm[2].y and (lm[4].y < lm[3].y)
            thumb_downward  = lm[4].y > lm[2].y and (lm[4].y > lm[3].y)

            # Variables to store what bubble text and color to show
            gesture_text = ""
            bubble_color = (0, 0, 0) # Default Black
            
            # --- GESTURE LOGIC ---
            
            # A. THUMBS UP
            if thumb_upright and index_folded and middle_folded and ring_folded and pinky_folded:
                gesture_text = "GOOD!"
                bubble_color = (0, 200, 0) # Green
                
            # B. THUMBS DOWN
            elif thumb_downward and index_folded and middle_folded and ring_folded and pinky_folded:
                gesture_text = "BAD!"
                bubble_color = (0, 0, 255) # Red

            # C. HIGH-FIVE (All fingers out straight)
            elif index_straight and middle_straight and ring_straight and pinky_straight:
                gesture_text = "HIGH FIVE!"
                bubble_color = (255, 128, 0) # Orange

            # D. PUNCH (All fingers completely curled into the palm)
            elif index_folded and middle_folded and ring_folded and pinky_folded and not thumb_upright:
                gesture_text = "PUNCH!"
                bubble_color = (0, 0, 100) # Dark Maroon

            # E. VICTORY / PEACE (Index and Middle straight, others folded)
            elif index_straight and middle_straight and ring_folded and pinky_folded:
                gesture_text = "VICTORY!"
                bubble_color = (200, 0, 200) # Purple

            # F. OK SIGN (Index touches Thumb, others up. Simple check: Middle, Ring, Pinky are straight)
            elif index_folded and middle_straight and ring_straight and pinky_straight:
                gesture_text = "OK!"
                bubble_color = (0, 255, 255) # Yellow

            # --- DRAW THE BUBBLE ---
            if gesture_text != "":
                # Get the wrist coordinate (landmark 0) to place a big floating bubble over the hand
                wrist_x = int(lm[0].x * w)
                wrist_y = int(lm[0].y * h)
                
                # Floating bubble positioned above the wrist/hand area
                bubble_center = (wrist_x, wrist_y - 180)
                
                # Draw the colored bubble background
                cv2.circle(frame, bubble_center, 65, bubble_color, -1)
                
                # Add text centered inside the bubble
                cv2.putText(frame, gesture_text, (bubble_center[0] - 50, bubble_center[1] + 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display the output
    cv2.imshow('Multi-Gesture Bubble Detector', frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
