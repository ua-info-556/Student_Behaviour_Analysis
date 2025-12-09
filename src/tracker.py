# src/tracker.py
import numpy as np
from scipy.spatial.distance import cdist

class StudentTracker:
    def __init__(self, max_dist=50, max_missed=5):
        self.next_id = 1
        self.tracks = {} 
        self.max_dist = max_dist
        self.max_missed = max_missed

    def _centroid(self, box):
        x1, y1, x2, y2 = box
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    # assign IDs to detections based on centroid distance
    def assign_ids(self, detections):
        if len(detections) == 0:
            to_remove = []
            for sid, (centroid, missed) in self.tracks.items():
                if missed >= self.max_missed:
                    to_remove.append(sid)
                else:
                    self.tracks[sid] = (centroid, missed + 1)
            for sid in to_remove:
                del self.tracks[sid]
            return []

        new_centroids = np.array([self._centroid(box) for box in detections])

        # If only one detection then reshape the array to avoid dimension issues
        if new_centroids.ndim == 1:
            new_centroids = new_centroids.reshape(1, -1)

        if len(self.tracks) == 0:
            # If no existing tracks then assign new IDs
            ids = []
            for i in range(len(detections)):
                ids.append(self.next_id)
                self.tracks[self.next_id] = (new_centroids[i], 0)
                self.next_id += 1
            return ids

        old_ids = list(self.tracks.keys())
        old_centroids = np.array([self.tracks[i][0] for i in old_ids])

        # If there is only one old centroid then reshape for distance calculation
        if old_centroids.ndim == 1:
            old_centroids = old_centroids.reshape(1, -1)

        # Calculate the distance matrix between old and new centroids
        dist_matrix = cdist(old_centroids, new_centroids)

        assigned = {}
        used_new = set()

        # Greedy assignment based on the minimum distance
        for i, oid in enumerate(old_ids):
            if dist_matrix.shape[1] == 0:
                break
            distances = dist_matrix[i]
            min_dist_idx = np.argmin(distances)
            if distances[min_dist_idx] < self.max_dist and min_dist_idx not in used_new:
                assigned[oid] = min_dist_idx
                used_new.add(min_dist_idx)

        # If a new detection was not assigned an existing ID then create a new ID
        ids = []
        for idx in range(len(detections)):
            found = False
            for oid, j in assigned.items():
                if j == idx:
                    ids.append(oid) 
                    self.tracks[oid] = (new_centroids[idx], 0)
                    found = True
                    break
            if not found:
                # If the detection does not match any existing tracks then assign a new ID
                ids.append(self.next_id)
                self.tracks[self.next_id] = (new_centroids[idx], 0)
                self.next_id += 1

        return ids
