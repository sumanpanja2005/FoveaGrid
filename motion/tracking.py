import numpy as np
from typing import List, Dict, Any

class TrackedObject:
    def __init__(self, track_id: int, position: np.ndarray, size: np.ndarray, velocity: np.ndarray):
        self.track_id = track_id
        self.position = position # [x, y, z]
        self.size = size # [dx, dy, dz]
        self.velocity = velocity # [vx, vy, vz]
        self.age = 1

class DynamicObjectTracker:
    """
    DBSCAN dynamic cluster tracker for estimating velocity vectors and active targets.
    """

    def __init__(self, eps: float = 0.8, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.tracked_objects: List[TrackedObject] = []
        self.next_track_id = 1

    def update(self, dynamic_points: np.ndarray, dt: float = 0.1) -> List[TrackedObject]:
        if len(dynamic_points) < self.min_samples:
            return self.tracked_objects

        from sklearn.cluster import DBSCAN
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(dynamic_points)
        labels = db.labels_

        current_clusters = []
        unique_labels = set(labels) - {-1}

        for lbl in unique_labels:
            pts = dynamic_points[labels == lbl]
            pos = np.mean(pts, axis=0)
            p_min = np.min(pts, axis=0)
            p_max = np.max(pts, axis=0)
            size = np.maximum(p_max - p_min, 0.3)

            current_clusters.append({
                "pos": pos,
                "size": size,
                "pts": pts
            })

        # Match clusters with existing tracked objects
        updated_tracks = []
        for cluster in current_clusters:
            pos = cluster["pos"]
            size = cluster["size"]

            best_match = None
            min_dist = 3.0

            for track in self.tracked_objects:
                dist = np.linalg.norm(track.position + track.velocity * dt - pos)
                if dist < min_dist:
                    min_dist = dist
                    best_match = track

            if best_match is not None:
                # Update existing track
                vel = (pos - best_match.position) / dt
                best_match.position = pos
                best_match.size = size
                best_match.velocity = 0.7 * best_match.velocity + 0.3 * vel
                best_match.age += 1
                updated_tracks.append(best_match)
            else:
                # Create new track
                new_track = TrackedObject(
                    track_id=self.next_track_id,
                    position=pos,
                    size=size,
                    velocity=np.zeros(3, dtype=np.float32)
                )
                self.next_track_id += 1
                updated_tracks.append(new_track)

        self.tracked_objects = updated_tracks
        return self.tracked_objects
