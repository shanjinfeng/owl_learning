import cv2
import numpy as np

class ArucoMarkerDetector:
    """
    ArUco Marker 检测器，基于 OpenCV 4.7+
    """
    def __init__(self, dictionary_type=cv2.aruco.DICT_6X6_250):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_type)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)

    def predict(self, frame):
        """
        检测图像中的 ArUco Marker
        返回: [[x1, y1, x2, y2, marker_id], ...]
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        boxes = []
        if ids is not None and len(ids) > 0:
            for i, corner in enumerate(corners):
                points = corner[0]
                x_min, y_min = float(np.min(points[:, 0])), float(np.min(points[:, 1]))
                x_max, y_max = float(np.max(points[:, 0])), float(np.max(points[:, 1]))
                marker_id = int(ids[i][0])
                boxes.append([x_min, y_min, x_max, y_max, marker_id])
                
        return boxes


class GreenOnBrown:
    """
    联合 Excess-G (ExG) 与 HSV 色彩空间的绿色方块/目标检测器
    同时支持检测 ArUco 二维码
    """
    def __init__(self):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # 初始化 ArUco 检测器
        self.aruco_detector = ArucoMarkerDetector()

    def inference(self, image,
                  exg_min=30, exg_max=250,
                  hue_min=35, hue_max=85,
                  saturation_min=40, saturation_max=255,
                  brightness_min=40, brightness_max=255,
                  min_detection_area=50,
                  show_display=False):
        
        # ==========================================
        # 1. 绿色方块检测 (ExG + HSV 联合过滤)
        # ==========================================
        
        # A. 计算 Excess-G (ExG) 特征: ExG = 2G - R - B
        img_float = image.astype(np.float32)
        b, g, r = cv2.split(img_float)
        exg = 2 * g - r - b
        
        # 对 ExG 进行阈值过滤
        exg_mask = np.zeros(exg.shape, dtype=np.uint8)
        exg_mask[(exg >= exg_min) & (exg <= exg_max)] = 255

        # B. 计算 HSV 特征进行色彩过滤
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([hue_min, saturation_min, brightness_min])
        upper_hsv = np.array([hue_max, saturation_max, brightness_max])
        hsv_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # C. 联合过滤 (求交集，提高对环境光及黄化叶片的鲁棒性)
        combined_mask = cv2.bitwise_and(exg_mask, hsv_mask)

        # 形态学闭运算，去除噪点并填补内部空洞
        closed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)

        # 查找轮廓
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        green_boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > min_detection_area:
                x, y, w, h = cv2.boundingRect(c)
                green_boxes.append([x, y, x + w, y + h])  # 统一格式：[x1, y1, x2, y2]

        # ==========================================
        # 2. ArUco 二维码检测
        # ==========================================
        aruco_boxes = self.aruco_detector.predict(image)

        # ==========================================
        # 3. 结果可视化 (可选)
        # ==========================================
        if show_display:
            image_out = image.copy()
            
            # 绘制绿色方块 (红色框)
            for box in green_boxes:
                cv2.rectangle(image_out, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                cv2.putText(image_out, "Green Square", (int(box[0]), int(box[1]) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
            # 绘制 ArUco 二维码 (蓝色框)
            for box in aruco_boxes:
                x1, y1, x2, y2, marker_id = int(box[0]), int(box[1]), int(box[2]), int(box[3]), int(box[4])
                cv2.rectangle(image_out, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image_out, f"ID: {marker_id}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            return green_boxes, aruco_boxes, image_out

        return green_boxes, aruco_boxes, image