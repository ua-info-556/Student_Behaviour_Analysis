# src/gaze_classifier.py
import numpy as np

def classify_gaze(landmarks, frame_width, frame_height):
    if landmarks is None or len(landmarks) == 0:
        return "Looking Center"

    # Compute approximate face center
    x_coords = landmarks[:, 0]
    y_coords = landmarks[:, 1]
    face_center_x = np.mean(x_coords)
    face_center_y = np.mean(y_coords)

    # Horizontal gaze classification
    if face_center_x < frame_width * 0.40:
        horizontal_gaze = "Looking Left"
    elif face_center_x > frame_width * 0.60:
        horizontal_gaze = "Looking Right"
    else:
        horizontal_gaze = "Looking Center"

    # Vertical gaze classification
    if face_center_y < frame_height * 0.40:
        vertical_gaze = "Looking Up"
    elif face_center_y > frame_height * 0.60:
        vertical_gaze = "Looking Down"
    else:
        vertical_gaze = "Looking Center"

    # Combine horizontal and vertical gaze
    if horizontal_gaze == "Looking Center" and vertical_gaze == "Looking Center":
        return "Looking Center"
    elif horizontal_gaze == "Looking Left" and vertical_gaze == "Looking Up":
        return "Looking Left Up"
    elif horizontal_gaze == "Looking Left" and vertical_gaze == "Looking Down":
        return "Looking Left Down"
    elif horizontal_gaze == "Looking Right" and vertical_gaze == "Looking Up":
        return "Looking Right Up"
    elif horizontal_gaze == "Looking Right" and vertical_gaze == "Looking Down":
        return "Looking Right Down"
    elif horizontal_gaze == "Looking Left":
        return "Looking Left"
    elif horizontal_gaze == "Looking Right":
        return "Looking Right"
    elif vertical_gaze == "Looking Up":
        return "Looking Up"
    elif vertical_gaze == "Looking Down":
        return "Looking Down"

    return "Looking Center"
