#Hand Tracker Num Ref

import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the hand tracking model
hands = mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

prev_frame_time = 0
new_frame_time = 0

while True:
    # Read frame from webcam
    success, img = cap.read()
    if not success:
        print("Error: Failed to capture image")
        break
    
    # Flip the image horizontally for a more intuitive mirror view
    img = cv2.flip(img, 1)
    
    # Convert the image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # To improve performance, mark the image as not writeable
    img_rgb.flags.writeable = False
    
    # Process the image and find hands
    results = hands.process(img_rgb)
    
    # Mark the image as writeable again
    img_rgb.flags.writeable = True
    
    # Convert back to BGR for display
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks and connections
            mp_drawing.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            # Get landmarks positions and label each point with its index
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                # Draw circle and index number for each landmark
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                cv2.putText(img, str(id), (cx, cy - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Calculate and display FPS
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2)
    
    # Display the number of hands detected
    num_hands = 0 if results.multi_hand_landmarks is None else len(results.multi_hand_landmarks)
    cv2.putText(img, f'Hands: {num_hands}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2)
    
    # Display the resulting image
    cv2.imshow('Hand Tracking with Landmark Numbers', img)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()