# src/emotion_classifier.py
import numpy as np

def classify_emotion(face_keypoints):
    kp = np.array(face_keypoints, dtype=np.float32)
    if kp.ndim != 2 or kp.shape[0] != 5 or kp.shape[1] != 2:
        return "Neutral"

    left_mouth, right_mouth = kp[3], kp[4]
    mouth_width = np.linalg.norm(right_mouth - left_mouth)

    if mouth_width > 20:
        return "Happy"
    return "Neutral"
