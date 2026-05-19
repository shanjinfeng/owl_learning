import cv2
import numpy as np
from typing import List, Tuple, Optional
import time


class ArucoMarkerDetector:
    """
    ArUco Marker 检测器封装类
    
    输出格式: 每个 box = [x1, y1, x2, y2, marker_id] (左上角和右下角坐标)
    """
    
    def __init__(
        self,
        dictionary_type: int = cv2.aruco.DICT_6X6_250,
        detector_params: Optional[cv2.aruco.DetectorParameters] = None
    ):
        """
        初始化检测器
        
        Args:
            dictionary_type: ArUco 字典类型
            detector_params: 自定义检测参数，None 则使用默认
        """
        # 获取预定义字典
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_type)
        
        # 检测参数
        if detector_params is None:
            self.parameters = cv2.aruco.DetectorParameters()
            # 可调参数示例
            self.parameters.adaptiveThreshWinSizeMin = 3
            self.parameters.adaptiveThreshWinSizeMax = 23
            self.parameters.adaptiveThreshWinSizeStep = 10
            self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        else:
            self.parameters = detector_params
        
        # 创建检测器 (OpenCV 4.7+)
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        
        # 存储上次检测的详细信息
        self.last_corners = None
        self.last_ids = None
    
    def predict(self, frame: np.ndarray) -> List[List[float]]:
        """
        检测图像中的 ArUco Marker
        
        Args:
            frame: OpenCV 图像 (cv::Mat), BGR 或灰度格式
        
        Returns:
            boxes: 检测框列表，每个 box = [x1, y1, x2, y2, marker_id]
        """
        # 确保是灰度图
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # 检测 markers
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        # 保存原始结果供其他方法使用
        self.last_corners = corners
        self.last_ids = ids
        
        boxes = []
        
        if ids is None or len(ids) == 0:
            return boxes
        
        # 将 corners 转换为 [x1, y1, x2, y2, id] 格式
        for i, corner in enumerate(corners):
            # corner 形状: (1, 4, 2) -> 4个角点坐标
            points = corner[0]  # 形状: (4, 2)
            
            # 计算外接矩形
            x_min = float(np.min(points[:, 0]))
            y_min = float(np.min(points[:, 1]))
            x_max = float(np.max(points[:, 0]))
            y_max = float(np.max(points[:, 1]))
            
            marker_id = int(ids[i][0])
            
            # 格式: [x1, y1, x2, y2, marker_id]
            box = [x_min, y_min, x_max, y_max]
            boxes.append(box)
        
        return boxes
    
    def predict_with_timing(self, frame: np.ndarray) -> Tuple[List[List[float]], float]:
        """
        检测图像中的 ArUco Marker，并返回耗时
        
        Args:
            frame: OpenCV 图像 (cv::Mat), BGR 或灰度格式
        
        Returns:
            (boxes, elapsed_ms): 检测框列表 和 耗时（毫秒）
        """
        # 记录开始时间（使用 time.perf_counter() 获取高精度计时）
        start_time = time.perf_counter()
        
        # 执行检测
        boxes = self.predict(frame)
        
        # 计算耗时（转换为毫秒）
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return boxes, elapsed_ms
    
    def predict_without_id(self, frame: np.ndarray) -> List[List[float]]:
        """
        只返回检测框，不包含 marker ID
        
        Returns:
            boxes: [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
        """
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
        """
        在图像上绘制检测框
        
        Args:
            frame: 原始图像
            boxes: predict 返回的 boxes
            color: 框颜色 (B, G, R)
            thickness: 线宽
            show_id: 是否显示 ID
        
        Returns:
            绘制后的图像
        """
        img = frame.copy()
        
        for box in boxes:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            
            # 绘制矩形框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            # 显示 ID
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
        """
        获取上次检测的 marker 中心点坐标
        
        Returns:
            [(cx, cy, id), ...] 或 None
        """
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


# ==================== 使用示例 ====================

# 创建虚拟测试图像
def create_test_image():
    img = np.ones((640, 640, 3), dtype=np.uint8) * 255
    
    # 生成一个 marker 并绘制到图像上
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    marker = cv2.aruco.generateImageMarker(aruco_dict, 1, 100)
    
    # 放置到图像中心
    x, y = 270, 190
    img[y:y+100, x:x+100] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

    m, n = 400, 400
    img[m:m+100, n:n+100] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    
    return img


if __name__ == "__main__":
    # 初始化检测器
    detector = ArucoMarkerDetector(dictionary_type=cv2.aruco.DICT_6X6_250)
    
    # ========== 方法1: 单次检测计时 ==========
    print("=== 单次检测耗时测试 ===")
    frame = create_test_image()
    
    boxes, elapsed_ms = detector.predict_with_timing(frame)
    print(f"检测耗时: {elapsed_ms:.3f} ms ({1000/elapsed_ms:.1f} FPS)" if elapsed_ms > 0 else "检测耗时: 0 ms")
    print(f"检测到 {len(boxes)} 个 marker")
    
    # ========== 方法2: 多次检测取平均（更准确） ==========
    print("\n=== 多次检测平均耗时（推荐） ===")
    
    num_iterations = 100  # 测试100次
    timings = []
    
    for i in range(num_iterations):
        boxes, elapsed_ms = detector.predict_with_timing(frame)
        timings.append(elapsed_ms)
    
    # 计算统计信息
    avg_time = np.mean(timings)
    std_time = np.std(timings)
    min_time = np.min(timings)
    max_time = np.max(timings)
    median_time = np.median(timings)
    
    print(f"迭代次数: {num_iterations}")
    print(f"平均耗时: {avg_time:.3f} ms")
    print(f"中位数: {median_time:.3f} ms")
    print(f"标准差: {std_time:.3f} ms")
    print(f"最小值: {min_time:.3f} ms")
    print(f"最大值: {max_time:.3f} ms")
    print(f"理论最大FPS: {1000/avg_time:.1f}")
    
    # ========== 方法3: 实时视频流计时 ==========
    print("\n=== 实时视频流检测（按 q 退出） ===")
    
    # 使用摄像头或测试图像循环
    use_camera = False  # 设为 True 使用摄像头，False 使用测试图像
    
    if use_camera:
        cap = cv2.VideoCapture(0)
    
    frame_count = 0
    total_time = 0
    
    while True:
        # 获取图像
        if use_camera:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame = create_test_image()
        
        # 检测并计时
        boxes, elapsed_ms = detector.predict_with_timing(frame)
        
        # 累计统计
        frame_count += 1
        total_time += elapsed_ms
        
        # 可视化
        vis_frame = detector.draw_boxes(frame, boxes)
        
        # 在图像上显示耗时
        fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0
        text = f"Time: {elapsed_ms:.1f}ms | FPS: {fps:.1f}"
        cv2.putText(vis_frame, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('ArUco Detection with Timing', vis_frame)
        
        # 每30帧打印一次平均耗时
        if frame_count % 30 == 0:
            avg_fps = frame_count / (total_time / 1000.0) if total_time > 0 else 0
            print(f"已处理 {frame_count} 帧, 平均耗时: {total_time/frame_count:.2f} ms, 平均FPS: {avg_fps:.1f}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    if use_camera:
        cap.release()
    cv2.destroyAllWindows()
    
    # 最终统计
    if frame_count > 0:
        print(f"\n最终统计: 共 {frame_count} 帧, 总耗时 {total_time:.1f} ms")
        print(f"平均每帧: {total_time/frame_count:.3f} ms")