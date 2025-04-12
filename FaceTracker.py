import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

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
    
    # Convert the image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process the image and find faces
    results = face_detection.process(img_rgb)
    
    # Convert back to BGR for display
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    # Draw face detections
    if results.detections:
        for detection in results.detections:
            # Draw detection box
            mp_drawing.draw_detection(img, detection)
            
            # Get bounding box coordinates
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Draw additional information
            confidence = round(detection.score[0] * 100)
            cv2.putText(img, f'Confidence: {confidence}%', (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Calculate and display FPS
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2)
    
    # Display the resulting image
    cv2.imshow('Face Detection', img)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()