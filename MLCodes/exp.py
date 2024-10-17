import cv2 # for camera
import mediapipe as mp #tracking lines in finger
import pandas as pd #data Manipulation 

def detect_hand_landmarks():
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

    # Initialize MediaPipe for drawing landmarks
    mp_drawing = mp.solutions.drawing_utils
    drawing_styles = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    # Set the desired window size
    window_width, window_height = 800, 600

    # Open webcam
    cap = cv2.VideoCapture(0)

    # Create a named window with a specific size
    cv2.namedWindow('Hand Landmarks Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Hand Landmarks Detection', window_width, window_height)

    # Create an empty list to store landmarks data
    landmarks_data_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect landmarks
        results = hands.process(rgb_frame)

        # Draw landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_data = []
                for landmark in hand_landmarks.landmark:
                    landmark_data.extend([landmark.x, landmark.y, landmark.z])

                # Append landmarks data to the list
                landmarks_data_list.append(landmark_data)

                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=drawing_styles,
                    connection_drawing_spec=drawing_styles
                )

        # Mirror the width (horizontally flip) the frame
        mirrored_frame = cv2.flip(frame, 1)

        # Display the mirrored frame in the window
        cv2.imshow('Hand Landmarks Detection', mirrored_frame)

        # Check for key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Convert the list to a DataFrame
    columns = [f'Landmark_{i}_X' for i in range(21)] + [f'Landmark_{i}_Y' for i in range(21)] + [f'Landmark_{i}_Z' for i in range(21)]
    landmarks_data = pd.DataFrame(landmarks_data_list, columns=columns)

    # Save the landmarks data to a CSV file
    landmarks_data.to_csv('right.csv', index=False)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_hand_landmarks()
