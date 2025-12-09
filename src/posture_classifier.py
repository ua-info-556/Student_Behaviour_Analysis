# src/posture_classifier.py

def classify_posture(pose_box):
    if pose_box is None:
        return "Unknown"

    x1, y1, x2, y2 = pose_box
    width = x2 - x1
    height = y2 - y1

    if height == 0 or width == 0:
        return "Unknown"

    aspect_ratio = height / width

    if aspect_ratio > 2.0:
        return "Upright"
    elif 1.2 < aspect_ratio <= 2.0:
        return "Sitting"
    else:
        return "Slouching"
