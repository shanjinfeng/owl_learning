import sys
import os
import cv2
import numpy as np
import time
import configparser

# 获取当前脚本的绝对路径的上一级目录（即项目根目录）并加入到环境变量
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils.video_manager import VideoStream
from utils.greenonbrown import GreenOnBrown
from utils.output_manager import RelayController
from utils.vis_manager import RelayVis
from utils.tracker import CropMaskStabilizer


class GroundCoordinateMapperIPM:
    """逆透视映射器：提取物理坐标"""
    def __init__(self, matrix_path, img_width=2048, img_height=1536):
        self.H_inv = np.eye(3, dtype=np.float32)
        if os.path.exists(matrix_path):
            data = np.load(matrix_path)
            H = data['H'] if isinstance(data, np.lib.npyio.NpzFile) and 'H' in data else data
            self.H_inv = np.linalg.inv(H)
            print(f"[IPM] 成功加载单应性矩阵: {matrix_path}")
        else:
            print(f"[IPM] 警告: 未找到矩阵文件 {matrix_path}，将使用单位矩阵。")

        # 建立相对坐标系：原点=画面底部中点，X=向右，Y=向前
        self.p_bot = np.array(self._get_raw_coords(img_width / 2.0, img_height))
        p_top = np.array(self._get_raw_coords(img_width / 2.0, 0))
        p_right = np.array(self._get_raw_coords(img_width, img_height))
        
        # 计算“向前”的单位向量
        vec_fwd_raw = p_top - self.p_bot
        self.vec_fwd = vec_fwd_raw / np.linalg.norm(vec_fwd_raw)
        
        # 计算“向右”的单位向量（强制与“向前”垂直）
        vec_right_raw = p_right - self.p_bot
        vec_right_ortho = vec_right_raw - np.dot(vec_right_raw, self.vec_fwd) * self.vec_fwd
        self.vec_right = vec_right_ortho / np.linalg.norm(vec_right_ortho)

    # 私有方法：像素坐标转换为原始物理坐标
    def _get_raw_coords(self, u: float, v: float) -> tuple[float, float]:
        w = self.H_inv[2, 0] * u + self.H_inv[2, 1] * v + self.H_inv[2, 2]
        if abs(w) < 1e-9: return 0.0, 0.0
        x = (self.H_inv[0, 0] * u + self.H_inv[0, 1] * v + self.H_inv[0, 2]) / w
        y = (self.H_inv[1, 0] * u + self.H_inv[1, 1] * v + self.H_inv[1, 2]) / w
        return x, y

    def pixel_to_ground(self, u: float, v: float) -> tuple[float, float]:
        """返回相对坐标 (X_right_m, Y_forward_m)"""
        p_marker = np.array(self._get_raw_coords(u, v))
        vec_target = p_marker - self.p_bot
        right_dist_m = np.dot(vec_target, self.vec_right) / 1000.0
        forward_dist_m = np.dot(vec_target, self.vec_fwd) / 1000.0
        return float(right_dist_m), float(forward_dist_m)


class GreenWeeder:
    def __init__(self, config_path='config/green_weeding.ini'):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)

        # 参数读取
        self.res = (self.config.getint('Camera', 'resolution_width'),
                    self.config.getint('Camera', 'resolution_height'))
        self.speed_mps = self.config.getfloat('System', 'speed')
        self.spray_duration = self.config.getfloat('System', 'spray_duration')
        self.nozzle_offset = self.config.getfloat('System', 'nozzle_offset_m')
        self.track_timeout = self.config.getfloat('System', 'track_timeout_s')
        self.relay_response_s = self.config.getfloat('System', 'relay_response_s', fallback=0.05)
        print(f"[System] 继电器物理响应补偿: {self.relay_response_s} 秒")

        # 继电器映射
        relay_dict = {}
        for key, value in self.config['Relays'].items():
            relay_dict[int(key)] = int(value)

        # 车道/喷头分配：将逆透视后的 X 视野均分为 lane_count 条车道，
        # 每条车道对应一个喷头（relay）。
        self.relay_ids = sorted(relay_dict.keys())
        self.lane_count = self.config.getint('Nozzles', 'lane_count', fallback=len(self.relay_ids))
        if self.lane_count <= 0:
            self.lane_count = len(self.relay_ids)

        self.fov_ground_width_m = self.config.getfloat('Nozzles', 'fov_ground_width_m', fallback=1.2)
        half_w = self.fov_ground_width_m / 2.0
        self.lane_x_bounds = np.linspace(-half_w, half_w, self.lane_count + 1).tolist()
        self.nozzle_x_positions = [
            0.5 * (self.lane_x_bounds[i] + self.lane_x_bounds[i + 1])
            for i in range(self.lane_count)
        ]

        if len(self.relay_ids) != self.lane_count:
            print(f"[System] 警告: relay 数量({len(self.relay_ids)}) 与 lane_count({self.lane_count}) 不一致，按较小值匹配。")
            use_n = min(len(self.relay_ids), self.lane_count)
            self.relay_ids = self.relay_ids[:use_n]
            self.lane_count = use_n
            self.lane_x_bounds = self.lane_x_bounds[:self.lane_count + 1]
            self.nozzle_x_positions = self.nozzle_x_positions[:self.lane_count]

        print(f"[System] 逆透视车道边界 X (m): {self.lane_x_bounds}")
        print(f"[System] 车道中心/喷头 X (m): {self.nozzle_x_positions}")
            
        # 初始化终端可视化管理器
        self.vis = RelayVis(relays=len(relay_dict))
            
        # 初始化带 on_state_change 回调的继电器控制器
        self.relay_controller = RelayController(relay_dict, on_state_change=self.vis.update)
        
        # 初始化 IPM 映射器
        matrix_path = self.config.get('System', 'ipm_matrix_path')
        self.mapper = GroundCoordinateMapperIPM(matrix_path, self.res[0], self.res[1])
        
        # 初始化视觉组件 (GreenOnBrown)
        self.detector = GreenOnBrown()
        # 读取检测去抖参数
        try:
            min_frames = self.config.getint('Detection', 'min_consecutive_frames')
            self.detector.min_consecutive_frames = max(1, min_frames)
        except Exception:
            pass

        self.camera = VideoStream(src=0, resolution=self.res)
        
        # 防重复喷洒的记忆字典：{track_id: spray_completion_time}
        self.sprayed_markers = {}

        # 初始化跟踪稳定器（IOU 自动分配 track_id）
        self.track_iou_threshold = self.config.getfloat('Detection', 'track_iou_threshold', fallback=0.3)
        self.track_max_age = self.config.getint('Detection', 'track_max_age', fallback=5)
        self.track_stabilizer = CropMaskStabilizer(max_age=self.track_max_age)

    def get_nearest_nozzle(self, target_x_m: float) -> int:
        """根据逆透视后的 X 坐标判断车道，并返回对应喷头 relay_id。"""
        if self.lane_count <= 1:
            return self.relay_ids[0]

        # 落在视野外时，夹紧到最左/最右车道
        if target_x_m <= self.lane_x_bounds[0]:
            lane_idx = 0
        elif target_x_m >= self.lane_x_bounds[-1]:
            lane_idx = self.lane_count - 1
        else:
            lane_idx = 0
            for i in range(self.lane_count):
                if self.lane_x_bounds[i] <= target_x_m < self.lane_x_bounds[i + 1]:
                    lane_idx = i
                    break

        return self.relay_ids[lane_idx]

    def clean_old_tracks(self):
        """清理过期的历史记录"""
        now = time.time()
        expired = [m_id for m_id, completion_t in self.sprayed_markers.items() if now > completion_t + self.track_timeout]
        for m_id in expired:
            del self.sprayed_markers[m_id]

    def run(self):
        print("[System] 正在启动相机...")
        self.camera.start()
        time.sleep(1.0)
        print("[System] 开始绿色目标喷洒作业 (按 'q' 退出)...\n")
        
        # 启动可视化终端界面
        self.vis.setup()

        try:
            while True:
                frame = self.camera.read()
                if frame is None:
                    time.sleep(0.01)
                    continue

                frame_capture_time = getattr(self.camera.stream, 'frame_timestamp', None)
                if frame_capture_time is None:
                    frame_capture_time = time.time()

                # 使用 GreenOnBrown 的 inference
                green_boxes, aruco_boxes, _ = self.detector.inference(frame, show_display=False)
                now = time.time()
                proc_time = now - frame_capture_time

                self.clean_old_tracks()

                vis_frame = frame.copy()

                # 先处理 ArUco 检测到的二维码（如果有），与原逻辑保持兼容
                for box in aruco_boxes:
                    x1, y1, x2, y2, marker_id = box
                    cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)
                    cv2.putText(vis_frame, f"ID:{marker_id}", (int(x1), int(y1)-30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                # 使用跟踪稳定器获得稳定的 track id
                dets_for_tracker = [[float(b[0]), float(b[1]), float(b[2]), float(b[3])] for b in green_boxes]
                self.track_stabilizer.update_from_boxes(
                    boxes=dets_for_tracker,
                    iou_threshold=self.track_iou_threshold
                )
                tracks = self.track_stabilizer.get_all_crop_regions()

                # 在 vis_frame 上绘制跟踪的边界框与 ID
                for tr in tracks:
                    tid = tr['track_id']
                    bx = tr['box']
                    x1, y1, x2, y2 = bx
                    cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                    cv2.putText(vis_frame, f"TID:{tid}", (int(x1), int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0

                    # 像素映射物理距离 (m)
                    gx_m, gy_m = self.mapper.pixel_to_ground(cx, cy)

                    # 使用 track id 去重
                    if tid in self.sprayed_markers:
                        continue

                    total_dist = gy_m + self.nozzle_offset
                    if total_dist <= 0:
                        continue

                    relay_id = self.get_nearest_nozzle(gx_m)
                    ideal_delay_s = total_dist / self.speed_mps
                    adjusted_delay_s = ideal_delay_s - proc_time - self.relay_response_s
                    if adjusted_delay_s < 0:
                        adjusted_delay_s = 0.0

                    print(f"\r[Action] Green target | 坐标: 右={gx_m*100:.1f}cm, 前={gy_m*100:.1f}cm | 匹配喷头: Relay {relay_id} | 理想延迟: {ideal_delay_s:.2f}s | 实际延迟: {adjusted_delay_s:.2f}s")

                    self.relay_controller.schedule_spray(
                        relay_id=relay_id,
                        delay_s=adjusted_delay_s,
                        duration_s=self.spray_duration
                    )

                    spray_completion_time = now + adjusted_delay_s + self.spray_duration
                    self.sprayed_markers[tid] = spray_completion_time

                # 显示小窗口用于调试（可选）
                cv2.imshow('green_weeder', vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.camera.stop()
            self.relay_controller.cleanup()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    gw = GreenWeeder()
    gw.run()
