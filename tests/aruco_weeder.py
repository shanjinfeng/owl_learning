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
from utils.marker_detect import ArucoMarkerDetector
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

        # 用于将地面物理坐标反投影回图像像素
        self.H = np.linalg.inv(self.H_inv)

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

    def relative_to_pixel(self, x_right_m: float, y_forward_m: float) -> tuple[float, float] | None:
        """相对地面坐标 (m) -> 图像像素坐标。"""
        # 相对坐标先回到原始物理坐标系（mm）
        p_raw = self.p_bot + (x_right_m * 1000.0) * self.vec_right + (y_forward_m * 1000.0) * self.vec_fwd
        x_raw = float(p_raw[0])
        y_raw = float(p_raw[1])

        # 再用单应矩阵投影到像素平面
        w = self.H[2, 0] * x_raw + self.H[2, 1] * y_raw + self.H[2, 2]
        if abs(w) < 1e-9:
            return None
        u = (self.H[0, 0] * x_raw + self.H[0, 1] * y_raw + self.H[0, 2]) / w
        v = (self.H[1, 0] * x_raw + self.H[1, 1] * y_raw + self.H[1, 2]) / w
        if not np.isfinite(u) or not np.isfinite(v):
            return None
        return float(u), float(v)

class ArucoWeeder:
    def __init__(self, config_path='config/aruco_weeding.ini'):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)

        # 参数读取
        self.res = (self.config.getint('Camera', 'resolution_width'),
                    self.config.getint('Camera', 'resolution_height'))
        self.speed_mps = self.config.getfloat('System', 'speed')
        self.spray_duration = self.config.getfloat('System', 'spray_duration')
        self.nozzle_offset = self.config.getfloat('System', 'nozzle_offset_m')
        self.track_timeout = self.config.getfloat('System', 'track_timeout_s')
        
        # 新增：读取继电器物理响应时间补偿参数，如果没有配置则默认补偿 0.05 秒（50毫秒）
        self.relay_response_s = self.config.getfloat('System', 'relay_response_s', fallback=0.05)
        print(f"[System] 继电器物理响应补偿: {self.relay_response_s} 秒")
        
        # 继电器映射
        relay_dict = {}
        for key, value in self.config['Relays'].items():
            relay_dict[int(key)] = int(value)

        # 车道/喷头分配：将逆透视后的 X 视野平均分成 lane_count 份，
        # 每个车道中心对应一个喷头，二维码按所在车道分配喷头。
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
            
        # 初始化带 on_state_change 回调的继电器控制器 (回调绑定可视化的更新)
        self.relay_controller = RelayController(relay_dict, on_state_change=self.vis.update)
        
        # 初始化 IPM 映射器
        matrix_path = self.config.get('System', 'ipm_matrix_path')
        self.mapper = GroundCoordinateMapperIPM(matrix_path, self.res[0], self.res[1])
        top_center = self.mapper.pixel_to_ground(self.res[0] / 2.0, 0.0)
        self.lane_vis_y_end_m = max(0.1, top_center[1])
        
        # 初始化视觉组件
        self.detector = ArucoMarkerDetector()
        self.camera = VideoStream(src=0, resolution=self.res)

        # 跟踪稳定器：按 marker_id 稳定 bbox，短时丢检仍可保持
        self.track_stabilizer = CropMaskStabilizer(max_age=2)
        
        # 防重复喷洒的记忆字典：{marker_id: last_seen_time}
        self.sprayed_markers = {}

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
        # sprayed_markers 存储的是喷洒预计完成时间, 过期判断为 当前时间 > 完成时间 + track_timeout
        expired = [m_id for m_id, completion_t in self.sprayed_markers.items() if now > completion_t + self.track_timeout]
        for m_id in expired:
            del self.sprayed_markers[m_id]

    def draw_lane_overlay(self, frame):
        """在画面中绘制逆透视 X 等分后的车道线。"""
        h, w = frame.shape[:2]

        for i, x_bound_m in enumerate(self.lane_x_bounds):
            p0 = self.mapper.relative_to_pixel(x_bound_m, 0.0)
            p1 = self.mapper.relative_to_pixel(x_bound_m, self.lane_vis_y_end_m)
            if p0 is None or p1 is None:
                continue

            u0, v0 = p0
            u1, v1 = p1
            if not (-2 * w <= u0 <= 3 * w and -2 * h <= v0 <= 3 * h and -2 * w <= u1 <= 3 * w and -2 * h <= v1 <= 3 * h):
                continue

            thickness = 3 if i in (0, len(self.lane_x_bounds) - 1) else 2
            cv2.line(frame, (int(u0), int(v0)), (int(u1), int(v1)), (255, 255, 0), thickness)

        text_y_forward = min(self.lane_vis_y_end_m * 0.15, 0.3)
        for i, x_center_m in enumerate(self.nozzle_x_positions):
            p_txt = self.mapper.relative_to_pixel(x_center_m, text_y_forward)
            if p_txt is None:
                continue
            u, v = p_txt
            if -w <= u <= 2 * w and -h <= v <= 2 * h:
                cv2.putText(frame, f"L{i} R{self.relay_ids[i]}", (int(u) - 30, int(v) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    def run(self):
        print("[System] 正在启动相机...")
        self.camera.start()
        time.sleep(1.0)
        print("[System] 开始除草作业 (按 'q' 退出)...\n")
        
        # 启动可视化终端界面
        self.vis.setup()

        try:
            while True:
                frame = self.camera.read()
                if frame is None:
                    time.sleep(0.01)
                    continue

                # 优先使用摄像头端记录的帧时间戳（如果可用），否则回退到当前时间
                frame_capture_time = getattr(self.camera.stream, 'frame_timestamp', None)
                if frame_capture_time is None:
                    frame_capture_time = time.time()

                boxes = self.detector.predict(frame)
                now = time.time()

                # 计算这帧图像从采集到当前处理时刻的耗时（秒）
                proc_time = now - frame_capture_time
                
                self.clean_old_tracks()

                vis_frame = frame.copy()
                self.draw_lane_overlay(vis_frame)

                marker_ids = []
                marker_boxes = []
                for box in boxes:
                    x1, y1, x2, y2, marker_id = box
                    marker_ids.append(int(marker_id))
                    marker_boxes.append([float(x1), float(y1), float(x2), float(y2)])

                # 用 marker_id 更新稳定器，并处理稳定后的目标
                self.track_stabilizer.update(track_ids=marker_ids, boxes=marker_boxes)
                tracked = self.track_stabilizer.get_all_crop_regions()

                for info in tracked:
                    marker_id = int(info['track_id'])
                    x1, y1, x2, y2 = info['box']
                    
                    # 无论是否喷洒过，都在图像上始终画出 Aruco 框和 ID
                    cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                    cv2.putText(vis_frame, f"ID:{marker_id}", (int(x1), int(y1)-30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # 已经喷洒过且在冷却期内，跳过 (防止重复触发喷嘴)
                    if marker_id in self.sprayed_markers:
                        continue
                        
                    # 计算中心点像素坐标 (与 test_camera_aruco.py 保持一致)
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    
                    # 像素映射物理距离 (m)
                    gx_m, gy_m = self.mapper.pixel_to_ground(cx, cy)
                    
                    # 如果物体已经越过了喷嘴后方，则无法喷洒
                    total_dist = gy_m + self.nozzle_offset
                    if total_dist <= 0:
                        continue

                    # 1. 匹配喷头
                    relay_id = self.get_nearest_nozzle(gx_m)
                    
                    # 2. 计算理想延迟时间 (如果不考虑系统延迟的情况)
                    ideal_delay_s = total_dist / self.speed_mps
                    
                    # 3. 补偿系统耗时和物理延迟
                    # 实际需要等待的时间 = 理想延迟 - 图像处理花费的时间 - 继电器/电磁阀物理响应所需时间
                    adjusted_delay_s = ideal_delay_s - proc_time - self.relay_response_s
                    
                    # 如果补偿后的延迟小于 0，说明由于车速快或处理慢，目标已经处于甚至越过了喷头正下方，需立即喷洒
                    if adjusted_delay_s < 0:
                        adjusted_delay_s = 0.0
                    
                    # 打印终端信息
                    print(f"\r[Action] 识别 ID:{int(marker_id)} | 坐标: 右={gx_m*100:.1f}cm, 前={gy_m*100:.1f}cm | 匹配喷头: Relay {relay_id} | 理想延迟: {ideal_delay_s:.2f}s | 实际延迟: {adjusted_delay_s:.2f}s")
                    # 4. 调度 GPIO 动作 (使用补偿后的精确延迟时间)
                    self.relay_controller.schedule_spray(
                        relay_id=relay_id, 
                        delay_s=adjusted_delay_s, 
                        duration_s=self.spray_duration
                    )
                    
                    # 5. 记录防重喷：记录喷洒预计完成时间，避免在喷洒过程中重复触发
                    spray_completion_time = now + adjusted_delay_s + self.spray_duration
                    self.sprayed_markers[marker_id] = spray_completion_time
                    
                    # 新增喷洒任务时，在框旁显示喷头和延迟信息
                    cv2.putText(vis_frame, f"N:{relay_id} D:{adjusted_delay_s:.2f}s", (int(x1), int(y1)-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # 显示图像 (缩小显示)
                cv2.imshow("Aruco Weeder with Terminal Vis", cv2.resize(vis_frame, (1024, 768)))
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            pass  # 静默捕捉Ctrl+C退出
        finally:
            print("\n[System] 正在清理资源...")
            self.vis.close()
            self.relay_controller.cleanup()
            self.camera.stop()
            cv2.destroyAllWindows()
            print("[System] 退出完毕。")

if __name__ == '__main__':
    weeder = ArucoWeeder()
    weeder.run()
