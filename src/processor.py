import os
import cv2
import pandas as pd
from collections import defaultdict
from tracker import StudentTracker
from ultralytics import YOLO

from src.gaze_classifier import classify_gaze
from src.posture_classifier import classify_posture
from src.emotion_classifier import classify_emotion

class BehaviorProcessor:
    def __init__(self, video_path, frame_skip=2, width=640, height=480):
        self.video_path = video_path
        self.frame_skip = frame_skip
        self.width = width
        self.height = height
        self.tracker = StudentTracker(max_dist=50, max_missed=5)
        self.update_progress = None

        self.per_frame_features = defaultdict(lambda: defaultdict(list))
        self.student_start_times = {}
        self.student_end_times = {}

        # YOLO models
        self.face_model = YOLO("models/yolov8n-face.pt")
        self.pose_model = YOLO("models/yolov8n-pose.pt")

        os.makedirs("output", exist_ok=True)

    # ===== Gaze estimation =====
    def estimate_gaze(self, keypoints):
        try:
            return classify_gaze(keypoints, self.width, self.height)
        except Exception:
            return "Looking Center"

    # ===== Posture estimation =====
    def estimate_posture(self, pose_box):
        try:
            return classify_posture(pose_box)
        except Exception:
            return "Upright"

    # ===== Emotion estimation =====
    def estimate_emotion(self, keypoints):
        try:
            return classify_emotion(keypoints)
        except Exception:
            return "Neutral"

    # ===== Process Video =====
    def process(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {self.video_path}")
            return

        frame_idx = 0
        fps = cap.get(cv2.CAP_PROP_FPS)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            if self.update_progress:
                self.update_progress(frame_idx, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

            if frame_idx % self.frame_skip != 0:
                continue

            frame = cv2.resize(frame, (self.width, self.height))
            timestamp = int(frame_idx / fps)


            # ===== YOLO detections =====
            face_results = self.face_model(frame)[0]
            pose_results = self.pose_model(frame)[0]

            centers = []
            gaze_preds = []
            emotion_preds = []
            face_boxes = []

            # Process faces
            if len(face_results.boxes) > 0:
                for box in face_results.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box)
                    face_boxes.append([x1, y1, x2, y2])
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    centers.append((cx, cy))

                    # Crop face for keypoints
                    face_crop = frame[y1:y2, x1:x2]

                    # YOLO face keypoints
                    keypoints = face_results.keypoints.xy.cpu().numpy() if hasattr(face_results, "keypoints") else None
                    if keypoints is not None and len(keypoints) > 0:
                        kp = keypoints[0]  # first face
                        gaze_preds.append(self.estimate_gaze(kp))
                        emotion_preds.append(self.estimate_emotion(kp))
                    else:
                        gaze_preds.append("Looking Center")
                        emotion_preds.append("Neutral")
            else:
                face_boxes.append([0, 0, 0, 0])
                centers.append((0, 0))
                gaze_preds.append("Looking Center")
                emotion_preds.append("Neutral")

            # Assign student IDs
            student_ids = self.tracker.assign_ids(face_boxes)

            # Posture prediction: first detected pose box
            posture_pred = self.estimate_posture(
                pose_results.boxes.xyxy[0].cpu().numpy() if len(pose_results.boxes) > 0 else None
            )

            # Aggregate features per student
            for sid, cx, cy, gaze, emotion in zip(student_ids,
                                                  [c[0] for c in centers],
                                                  [c[1] for c in centers],
                                                  gaze_preds,
                                                  emotion_preds):
                if sid not in self.student_start_times:
                    self.student_start_times[sid] = timestamp
                self.student_end_times[sid] = timestamp

                self.per_frame_features[timestamp][f"Gaze_{sid}"].append(gaze)
                self.per_frame_features[timestamp][f"Posture_{sid}"].append(posture_pred)
                self.per_frame_features[timestamp][f"Emotion_{sid}"].append(emotion)

        cap.release()
        print(f"✅ Video processed. Total frames: {frame_idx}")

    # ===== Save Aggregated CSV =====
    def save_results(self):
        data = []
        for t in sorted(self.per_frame_features.keys()):
            row = {"Time": t}
            for key, vals in self.per_frame_features[t].items():
                row[key] = max(set(vals), key=vals.count)
            data.append(row)

        df = pd.DataFrame(data)
        df.to_csv("output/behavior_results_aggregated.csv", index=False)
        print("✅ CSV saved as 'output/behavior_results_aggregated.csv'")
