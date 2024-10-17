import cv2  # OpenCV library for image and video processing
import mediapipe as mp  # Mediapipe library for hand landmark detection and other ML solutions
import numpy as np  # NumPy for numerical computations, especially working with arrays
import joblib  # Used for loading the pre-trained machine learning model
import serial  # Library for handling serial communication (e.g., sending data to Arduino)
import warnings  # Used to filter warnin


# Ignore warnings related to potential deprecation issues
warnings.filterwarnings("ignore")

# Load the pre-trained machine learning model for gesture recognition
loaded_model = joblib.load('final.joblib')

# Initialize the Mediapipe Hands solution for hand landmark detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Drawing utilities to visualize landmarks and connections between them
mp_drawing = mp.solutions.drawing_utils
drawing_styles = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Gesture mapping to associate predictions with gestures
gesture_mapping = {0: 'idle', 1: 'forward', 2: 'backward', 3: 'left', 4: 'right'}

# Open the default camera for capturing video frames
cap = cv2.VideoCapture(0)

# Initialize the serial communication with a specified port and baud rate
serial_port = 'COM9'  
baud_rate = 115200  
ser = serial.Serial(serial_port, baud_rate, dsrdtr=True, timeout=5)

# Track the current gesture to avoid repeating the same gesture multiple times
current_gesture = None

# Store a history of gestures for smoothing the predictions
smoothing_level = 30  
gesture_history = []

# Function to read and print incoming serial data (likely from hardware)
def read_serial():
    while True:
        response = ser.readline().decode().strip()
        if response:
            print(response)

# Create a separate thread to read the serial input data continuously
serial_thread = threading.Thread(target=read_serial)
serial_thread.daemon = True
serial_thread.start()

# Main loop to continuously read frames from the webcam
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB format for Mediapipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand landmarks
    results = hands.process(rgb_frame)

    # If landmarks are detected, process them
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmark_data = []
            # Extract x, y, and z coordinates from each hand landmark
            for landmark in hand_landmarks.landmark:
                landmark_data.extend([landmark.x, landmark.y, landmark.z])

            # Ensure the landmark data is valid and has 63 values (21 points x 3 coordinates)
            if len(landmark_data) == 63 and not any(np.isnan(landmark_data)):
                
                # Prepare input data for gesture recognition
                input_data = np.array(landmark_data).reshape(1, -1)

                # Use the loaded model to predict the gesture
                prediction = loaded_model.predict(input_data)

                # Map the predicted label to the corresponding gesture
                gesture_name = gesture_mapping.get(prediction[0], 'Unknown')

                # Append the current gesture to the gesture history for smoothing
                gesture_history.append(gesture_name)

                # Keep only the last 'smoothing_level' gestures in the history
                if len(gesture_history) > smoothing_level:
                    gesture_history.pop(0)

                # Determine the most frequent gesture in the history (smoothing)
                smoothed_gesture = max(set(gesture_history), key=gesture_history.count)

                # If the gesture changes, send the corresponding command to the serial port
                if smoothed_gesture != current_gesture:
                    if smoothed_gesture == 'forward':
                        ser.write(str(1).encode())  # Send '1' for forward movement
                    elif smoothed_gesture == 'backward':
                        ser.write(str(2).encode())  # Send '2' for backward movement
                    elif smoothed_gesture == 'left':
                        ser.write(str(3).encode())  # Send '3' for left turn
                    elif smoothed_gesture == 'right':
                        ser.write(str(4).encode())  # Send '4' for right turn
                    elif smoothed_gesture == 'idle':
                        ser.write(str(5).encode())  # Send '5' for idle (no movement)
                    current_gesture = smoothed_gesture

                # Display the detected gesture on the frame
                cv2.putText(frame, f'Gesture: {smoothed_gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Draw the hand landmarks and connections on the frame
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=drawing_styles,
                    connection_drawing_spec=drawing_styles
                )

    # Flip the frame horizontally to mirror it (useful for display)
    mirrored_frame = cv2.flip(frame, 1)

    # Display the frame in a window
    cv2.imshow('Hand Landmarks Detection', mirrored_frame)

    # Exit the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV windows when done
cap.release()
cv2.destroyAllWindows()
