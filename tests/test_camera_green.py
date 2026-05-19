import sys
import os
import cv2
import numpy as np
import time

# 添加项目根目录到 sys.path 以便导入 utils 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.video_manager import VideoStream
from utils.greenonbrown import GreenOnBrown

class GroundCoordinateMapperIPM:
    """
    负责将像素坐标转换为地面物理坐标的映射器。
    基于标定的单应性矩阵 H 及其逆矩阵 H_inv，
    并自动建立“原点=画面底部中点，X=向右，Y=向前”的相对物理坐标系。
    """
    def __init__(self, matrix_path, img_width=2048, img_height=1536):
        self.H_inv = np.eye(3, dtype=np.float32)
        
        # 1. 加载并求逆单应性矩阵
        if os.path.exists(matrix_path):
            try:
                data = np.load(matrix_path)
                H = data['H'] if isinstance(data, np.lib.npyio.NpzFile) and 'H' in data else data
                self.H_inv = np.linalg.inv(H)
                print(f"[IPM] 成功加载并求逆单应性矩阵: {matrix_path}")
            except Exception as e:
                print(f"[IPM] 无法读取矩阵文件 {matrix_path}: {e}")
        else:
            print(f"[IPM] 警告: 未找到矩阵文件 {matrix_path}。")

        # 2. 建立相对物理坐标系 (消除标定板摆放角度干扰)
        # 原点：图像底部中点
        self.p_bot = np.array(self._get_raw_coords(img_width / 2.0, img_height))
        # 前方参考点：图像顶部中点
        p_top = np.array(self._get_raw_coords(img_width / 2.0, 0))
        # 右方参考点：图像右下角
        p_right_edge = np.array(self._get_raw_coords(img_width, img_height))
        
        # 计算“向前”的单位向量
        vec_fwd_raw = p_top - self.p_bot
        self.vec_fwd = vec_fwd_raw / np.linalg.norm(vec_fwd_raw)
        
        # 计算“向右”的单位向量（强制与“向前”垂直）
        vec_right_raw = p_right_edge - self.p_bot
        vec_right_ortho = vec_right_raw - np.dot(vec_right_raw, self.vec_fwd) * self.vec_fwd
        self.vec_right = vec_right_ortho / np.linalg.norm(vec_right_ortho)

        print(f"[IPM] 物理原点(画面底部中点)基准坐标: X={self.p_bot[0]:.1f}mm, Y={self.p_bot[1]:.1f}mm")

    def _get_raw_coords(self, u: float, v: float) -> tuple[float, float]:
        """使用 H_inv 进行齐次坐标变换，输出原始物理坐标 (单位: 毫米)"""
        w = self.H_inv[2, 0] * u + self.H_inv[2, 1] * v + self.H_inv[2, 2]
        if abs(w) < 1e-9:
            return 0.0, 0.0
        x = (self.H_inv[0, 0] * u + self.H_inv[0, 1] * v + self.H_inv[0, 2]) / w
        y = (self.H_inv[1, 0] * u + self.H_inv[1, 1] * v + self.H_inv[1, 2]) / w
        return x, y

    def pixel_to_ground(self, u: float, v: float) -> tuple[float, float]:
        """
        像素 (u, v) → 地面相对坐标 (X, Y)，单位：米 (m)
        X: 右正
        Y: 前正
        """
        p_marker = np.array(self._get_raw_coords(u, v))
        
        # 计算相对于底部中点的向量 (单位：毫米)
        vec_target = p_marker - self.p_bot
        
        # 投影到前、右向量上
        forward_dist_mm = np.dot(vec_target, self.vec_fwd)
        right_dist_mm = np.dot(vec_target, self.vec_right)
        
        # 转换为米 (m)
        return float(right_dist_mm / 1000.0), float(forward_dist_mm / 1000.0)


def main():
    print("=== 大恒 GigE 相机 绿色物体(方块) IPM 逆透视坐标识别测试 ===")
    
    # 1. 配置参数
    RESOLUTION = (2048, 1536) # 大恒相机常用分辨率
    matrix_file_path = "/home/sjf/owl/calibration/calibration/H.npy"
    
    mapper = GroundCoordinateMapperIPM(
        matrix_path=matrix_file_path, 
        img_width=RESOLUTION[0], 
        img_height=RESOLUTION[1]
    )
    
    # 2. 初始化绿色检测器
    print("正在初始化 ExG + HSV 绿色目标检测器...")
    detector = GreenOnBrown()
    
    # 3. 初始化并启动相机
    try:
        print("正在连接大恒 GigE 相机...")
        camera = VideoStream(src=0, resolution=RESOLUTION)
        camera.start()
        print("相机启动成功！")
    except Exception as e:
        print(f"相机启动失败，请检查连接: {e}")
        return

    # 等待相机稳定
    time.sleep(1.0)
    
    print("按 'q' 键退出测试...\n")

    try:
        while True:
            frame = camera.read()
            if frame is None:
                time.sleep(0.01)
                continue

            # 使用我们修改后的 GreenOnBrown 进行检测
            green_boxes, aruco_boxes, _ = detector.inference(
                frame,
                exg_min=30, exg_max=250,
                hue_min=35, hue_max=85,
                saturation_min=40, saturation_max=255,
                brightness_min=40, brightness_max=255,
                min_detection_area=200,  # 过滤掉太小的绿点
                show_display=False
            )
            
            # 创建显示用图像
            display_frame = frame.copy()
            
            # 处理每个检测到的绿色方块
            for i, box in enumerate(green_boxes):
                x1, y1, x2, y2 = box
                
                # 计算中心点像素坐标
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                
                # 像素坐标 -> 相对地面坐标 (单位：米)
                gx_m, gy_m = mapper.pixel_to_ground(cx, cy)
                
                # 在图像上绘制识别结果
                cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3) # 红色框
                cv2.circle(display_frame, (int(cx), int(cy)), 6, (255, 0, 0), -1) # 蓝色定位点
                
                # 打印并显示坐标信息 (转换为 cm 显示)
                coord_text = f"Green[{i}] R:{gx_m*100:.1f}cm F:{gy_m*100:.1f}cm"
                cv2.putText(display_frame, coord_text, (int(x1), int(y1)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                print(f"检测到 绿色物体 [{i}]: 像素中心({cx:.1f}, {cy:.1f}) -> 地面(右={gx_m*100:.1f}cm, 前={gy_m*100:.1f}cm)")

            # 显示图像 (等比例缩小到 1024x768 适应屏幕)
            display_frame_resized = cv2.resize(display_frame, (1024, 768))
            cv2.imshow("Green Cube IPM Ground Detection", display_frame_resized)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n测试被用户中断")
    finally:
        print("正在关闭相机...")
        camera.stop()
        cv2.destroyAllWindows()
        print("测试结束。")

if __name__ == "__main__":
    main()