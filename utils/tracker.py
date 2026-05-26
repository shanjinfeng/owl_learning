"""
包装的跟踪辅助工具：从外部示例借鉴并移植到本仓库

包含：
- ClassSmoother: 对跟踪对象进行多数投票的类别平滑
- CropMaskStabilizer: 在检测缺失帧期间保持作物掩膜位置稳定

这些工具与任意跟踪器（如 ByteTrack）配合使用。
"""

from collections import Counter, deque

import cv2
import numpy as np


def _iou_xyxy(box_a, box_b):
    """Compute IOU for two xyxy boxes."""
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


class ClassSmoother:
    """Majority-vote class assignment per tracked object.

    Maintains a sliding window of class observations per `track_id` and
    returns the majority-vote class.

    Args:
        window: Number of frames of class history to keep per track.
    """

    def __init__(self, window=5):
        self.window = window
        self._history = {}   # {track_id: deque of class_id}
        self._last_seen = {} # {track_id: frame_count} for stale pruning

    def update(self, track_ids, class_ids, confidences=None, frame_count=0):
        """Record class observations and return smoothed class per track_id.

        Args:
            track_ids: list of int track IDs from tracker
            class_ids: list of int class IDs per detection
            confidences: list of float confidence per detection (optional)
            frame_count: current frame number for stale pruning

        Returns:
            dict: {track_id: smoothed_class_id}
        """
        seen = set()
        for tid, cls in zip(track_ids, class_ids):
            tid = int(tid)
            cls = int(cls)
            seen.add(tid)

            if tid not in self._history:
                self._history[tid] = deque(maxlen=self.window)
            self._history[tid].append(cls)
            self._last_seen[tid] = frame_count

        # Prune tracks not seen for 2x window frames
        if frame_count > 0:
            stale_threshold = frame_count - (self.window * 2)
            stale = [k for k, v in self._last_seen.items()
                     if v < stale_threshold and k not in seen]
            for k in stale:
                del self._history[k]
                del self._last_seen[k]

        return {tid: self._majority(tid) for tid in seen}

    def get_class(self, track_id):
        """Return majority-vote class_id for a specific track."""
        return self._majority(int(track_id))

    def _majority(self, track_id):
        """Compute majority-vote class from history."""
        hist = self._history.get(track_id)
        if not hist:
            return -1
        counts = Counter(hist)
        return counts.most_common(1)[0][0]

    def reset(self):
        """Clear all history."""
        self._history.clear()
        self._last_seen.clear()


class CropMaskStabilizer:
    """Persist crop detection positions for stable hybrid-mode masking.

    When detections drop for a few frames, this keeps last-known positions
    so downstream processing (e.g. weed detection) doesn't see holes.

    Supports both bbox and contour (segmentation) inputs.

    Args:
        max_age: Frames to persist a crop position after detection drops.
    """

    def __init__(self, max_age=3):
        self.max_age = max_age
        self._tracks = {}  # {track_id: {'box': [...], 'contour': ndarray|None, 'age': int}}
        self._next_track_id = 1

    def update(self, track_ids, boxes, contours=None):
        """Update with current frame's tracked crop detections.

        Args:
            track_ids: list of int track IDs for crop detections
            boxes: list of [x1, y1, x2, y2] bounding boxes (xyxy)
            contours: optional list of numpy contour arrays (for segmentation models)
        """
        seen = set()
        for i, (tid, box) in enumerate(zip(track_ids, boxes)):
            tid = int(tid)
            seen.add(tid)
            contour = contours[i] if contours and i < len(contours) else None
            self._tracks[tid] = {
                'track_id': tid,
                'box': list(box),
                'contour': contour,
                'age': 0,
            }

        # Age unseen tracks, prune expired
        to_remove = []
        for tid in list(self._tracks.keys()):
            if tid not in seen:
                self._tracks[tid]['age'] += 1
                if self._tracks[tid]['age'] > self.max_age:
                    to_remove.append(tid)
        for tid in to_remove:
            del self._tracks[tid]

    def update_from_boxes(self, boxes, contours=None, iou_threshold=0.3):
        """Assign stable track IDs to detections by IOU and update tracker state.

        Args:
            boxes: list of [x1, y1, x2, y2]
            contours: optional list of contours
            iou_threshold: minimum IOU to reuse an existing track_id

        Returns:
            list[int]: assigned track IDs in the same order as boxes.
        """
        if not boxes:
            self.update([], [], contours=None)
            return []

        existing = [(tid, info['box']) for tid, info in self._tracks.items()]
        assigned_ids = [-1] * len(boxes)
        used_track_ids = set()
        used_det_idx = set()

        # Greedy IOU matching: repeatedly pick best pair above threshold
        while True:
            best = (0.0, -1, -1)
            for d_idx, det in enumerate(boxes):
                if d_idx in used_det_idx:
                    continue
                for tid, t_box in existing:
                    if tid in used_track_ids:
                        continue
                    score = _iou_xyxy(det, t_box)
                    if score > best[0]:
                        best = (score, d_idx, tid)
            best_iou, d_idx, tid = best
            if best_iou < iou_threshold or d_idx < 0 or tid < 0:
                break
            assigned_ids[d_idx] = tid
            used_det_idx.add(d_idx)
            used_track_ids.add(tid)

        # Unmatched detections get new track IDs
        for d_idx in range(len(boxes)):
            if assigned_ids[d_idx] == -1:
                assigned_ids[d_idx] = self._next_track_id
                self._next_track_id += 1

        self.update(assigned_ids, boxes, contours=contours)
        return assigned_ids

    def get_all_crop_regions(self):
        """Return ALL crop regions (detected + persisted).

        Returns:
            list of dicts with 'box', 'contour', 'age' keys.
        """
        return list(self._tracks.values())

    def build_stabilized_mask(self, shape):
        """Build crop mask including persisted positions.

        Args:
            shape: (height, width) or (height, width, channels) tuple

        Returns:
            numpy uint8 mask (255 = crop, 0 = background)
        """
        h, w = shape[0], shape[1]
        mask = np.zeros((h, w), dtype=np.uint8)

        for info in self._tracks.values():
            if info['contour'] is not None:
                cv2.drawContours(mask, [info['contour']], -1, 255, -1)
            else:
                box = info['box']
                if len(box) >= 4:
                    x1, y1 = int(box[0]), int(box[1])
                    x2, y2 = int(box[2]), int(box[3])

                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    if x2 > x1 and y2 > y1:
                        mask[y1:y2, x1:x2] = 255

        return mask

    @property
    def active_count(self):
        """Number of crop regions currently being tracked (including persisted)."""
        return len(self._tracks)

    @property
    def persisted_count(self):
        """Number of crop regions being persisted (not detected this frame)."""
        return sum(1 for info in self._tracks.values() if info['age'] > 0)

    def reset(self):
        """Clear all tracked crop positions."""
        self._tracks.clear()
        self._next_track_id = 1
