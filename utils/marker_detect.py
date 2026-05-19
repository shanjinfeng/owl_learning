import cv2
import numpy as np
from typing import List, Tuple, Optional
import time


class ArucoMarkerDetector:
    """
    ArUco Marker 检测器封装类
    
    输出格式: 每个 box = [x1, y1, x2, y2, marker_id] (左上角和右下角坐标及ID)
    """
    
    def __init__(
        self,
        dictionary_type: int = cv2.aruco.DICT_6X6_250,
        detector_params: Optional[cv2.aruco.DetectorParameters] = None
    ):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_type)
        
        if detector_params is None:
            self.parameters = cv2.aruco.DetectorParameters()
            self.parameters.adaptiveThreshWinSizeMin = 3
            self.parameters.adaptiveThreshWinSizeMax = 23
            self.parameters.adaptiveThreshWinSizeStep = 10
            self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        else:
            self.parameters = detector_params
        
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        
        self.last_corners = None
        self.last_ids = None
    
    def predict(self, frame: np.ndarray) -> List[List[float]]:
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        self.last_corners = corners
        self.last_ids = ids
        
        boxes = []
        
        if ids is None or len(ids) == 0:
            return boxes
        
        for i, corner in enumerate(corners):
            points = corner[0]
            
            x_min = float(np.min(points[:, 0]))
            y_min = float(np.min(points[:, 1]))
            x_max = float(np.max(points[:, 0]))
            y_max = float(np.max(points[:, 1]))
            
            marker_id = int(ids[i][0])
            
            # 修复：将 marker_id 加入到返回列表中
            box = [x_min, y_min, x_max, y_max, marker_id]
            boxes.append(box)
        
        return boxes
    
    def predict_with_timing(self, frame: np.ndarray) -> Tuple[List[List[float]], float]:
        start_time = time.perf_counter()
        boxes = self.predict(frame)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return boxes, elapsed_ms
    
    def predict_without_id(self, frame: np.ndarray) -> List[List[float]]:
        boxes_with_id = self.predict(frame)
        return [[b[0], b[1], b[2], b[3]] for b in boxes_with_id]
    
    def draw_boxes(
        self,
        frame: np.ndarray,
        boxes: List[List[float]],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        show_id: bool = True
    ) -> np.ndarray:
        img = frame.copy()
        
        for box in boxes:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            if show_id and len(box) >= 5:
                marker_id = int(box[4])
                cv2.putText(
                    img, f"ID: {marker_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2
                )
        
        return img
    
    def get_marker_centers(self) -> Optional[List[Tuple[float, float, int]]]:
        if self.last_corners is None or self.last_ids is None:
            return None
        
        centers = []
        for i, corner in enumerate(self.last_corners):
            points = corner[0]
            cx = float(np.mean(points[:, 0]))
            cy = float(np.mean(points[:, 1]))
            marker_id = int(self.last_ids[i][0])
            centers.append((cx, cy, marker_id))
        
        return centers