import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np

# Configuration constants
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # RGB color for text
EXIT_BUTTON_COLOR = (0, 0, 255)  # Red color for exit button
EXIT_BUTTON_HOVER_COLOR = (0, 255, 0)  # Green color when hovering


def is_point_in_rectangle(point, rect):
    """Check if a point is inside a rectangle."""
    x, y = point
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh


def draw_landmarks_on_image(rgb_image, detection_result):
    """Draws hand landmarks and connections on the image."""
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Define exit button position (top right corner)
    height, width, _ = annotated_image.shape
    exit_button = (width - 100, 10, 90, 50)  # x, y, width, height

    # Draw exit button
    button_color = EXIT_BUTTON_COLOR
    is_exit_triggered = False

    # Loop through detected hands to visualize each
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Swap the handedness when the image is flipped
        current_handedness = handedness[0].category_name
        flipped_handedness = 'Right' if current_handedness == 'Left' else 'Left'

        # Convert landmarks to a protobuf message
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks
        ])

        # Draw landmarks and connections
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style())

        # Calculate text position (top left of hand)
        x_coords = [lm.x for lm in hand_landmarks]
        y_coords = [lm.y for lm in hand_landmarks]
        text_x = int(min(x_coords) * width)
        text_y = int(min(y_coords) * height) - MARGIN

        # Draw handedness (left/right) text with corrected handedness
        cv2.putText(annotated_image, f"{flipped_handedness}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        # Check for exit button interaction using index finger tip
        if flipped_handedness == 'Right':
            index_tip = hand_landmarks[8]  # MediaPipe hand landmark index for index finger tip
            index_tip_pixel = (
                int(index_tip.x * width),
                int(index_tip.y * height)
            )

            # Check if index finger tip is in exit button area
            if is_point_in_rectangle(index_tip_pixel, exit_button):
                button_color = EXIT_BUTTON_HOVER_COLOR
                is_exit_triggered = True

    # Draw exit button
    cv2.rectangle(annotated_image, 
                  (exit_button[0], exit_button[1]), 
                  (exit_button[0] + exit_button[2], exit_button[1] + exit_button[3]), 
                  button_color, -1)
    
    # Add exit text
    cv2.putText(annotated_image, "EXIT", 
                (width - 90, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (255, 255, 255), 2)

    return annotated_image, is_exit_triggered


def main():
    # Initialize the hand landmark detector
    base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2  # Detect up to two hands
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # Start video capture
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Convert frame to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect hand landmarks
        detection_result = detector.detect(mp_image)

        # Draw landmarks on the image and check for exit
        annotated_image, exit_triggered = draw_landmarks_on_image(rgb_frame, detection_result)

        # Convert back to BGR for OpenCV display
        annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Hand Landmarks', annotated_image_bgr)

        # Exit conditions
        if exit_triggered or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()