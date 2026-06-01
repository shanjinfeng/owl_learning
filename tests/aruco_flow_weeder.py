import sys
import os
import cv2
import numpy as np
import time
import configparser
import threading
import logging

# 获取当前脚本的绝对路径的上一级目录（即项目根目录）并加入到环境变量
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils.video_manager import VideoStream
from utils.marker_detect import ArucoMarkerDetector
from utils.optical_flow import OpticalFlowSpeedometer, AsyncOpticalFlowWorker
from utils.output_manager import RelayController
from utils.vis_manager import RelayVis


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

        self.p_bot = np.array(self._get_raw_coords(img_width / 2.0, img_height))
        p_top = np.array(self._get_raw_coords(img_width / 2.0, 0))
        p_right = np.array(self._get_raw_coords(img_width, img_height))

        vec_fwd_raw = p_top - self.p_bot
        self.vec_fwd = vec_fwd_raw / np.linalg.norm(vec_fwd_raw)

        vec_right_raw = p_right - self.p_bot
        vec_right_ortho = vec_right_raw - np.dot(vec_right_raw, self.vec_fwd) * self.vec_fwd
        self.vec_right = vec_right_ortho / np.linalg.norm(vec_right_ortho)

        # 用于将相对地面坐标反投影回图像像素
        self.H = np.linalg.inv(self.H_inv)

    # 私有方法：像素坐标转换为原始物理坐标
    def _get_raw_coords(self, u: float, v: float) -> tuple[float, float]:
        w = self.H_inv[2, 0] * u + self.H_inv[2, 1] * v + self.H_inv[2, 2]
        if abs(w) < 1e-9:
            return 0.0, 0.0
        x = (self.H_inv[0, 0] * u + self.H_inv[0, 1] * v + self.H_inv[0, 2]) / w
        y = (self.H_inv[1, 0] * u + self.H_inv[1, 1] * v + self.H_inv[1, 2]) / w
        return x, y

    # 公共方法：像素坐标转换为相对地面坐标
    def pixel_to_ground(self, u: float, v: float) -> tuple[float, float]:
        """返回相对坐标 (X_right_m, Y_forward_m)"""
        p_marker = np.array(self._get_raw_coords(u, v))
        vec_target = p_marker - self.p_bot
        right_dist_m = np.dot(vec_target, self.vec_right) / 1000.0
        forward_dist_m = np.dot(vec_target, self.vec_fwd) / 1000.0
        return float(right_dist_m), float(forward_dist_m)

    def relative_to_pixel(self, x_right_m: float, y_forward_m: float) -> tuple[float, float] | None:
        """相对地面坐标 (m) -> 图像像素坐标。"""
        p_raw = self.p_bot + (x_right_m * 1000.0) * self.vec_right + (y_forward_m * 1000.0) * self.vec_fwd
        x_raw = float(p_raw[0])
        y_raw = float(p_raw[1])

        w = self.H[2, 0] * x_raw + self.H[2, 1] * y_raw + self.H[2, 2]
        if abs(w) < 1e-9:
            return None
        u = (self.H[0, 0] * x_raw + self.H[0, 1] * y_raw + self.H[0, 2]) / w
        v = (self.H[1, 0] * x_raw + self.H[1, 1] * y_raw + self.H[1, 2]) / w
        if not np.isfinite(u) or not np.isfinite(v):
            return None
        return float(u), float(v)

# 以下代码实现了 ArucoFlowWeeder 类，负责检测 Aruco 标记、估计其运动并控制喷洒继电器。
def _clamp_box(box, width, height):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2

def _ransac_forward_speed(displacements_m, threshold_m=0.03, max_iter=30):
    """沿用 optical_flow.py 的思想，对世界位移做稳健估计。"""
    n = len(displacements_m)
    if n == 0:
        return 0.0, 0
    if n == 1:
        return float(displacements_m[0]), 1

    best_inliers = 0
    best_value = 0.0

    for _ in range(max_iter):
        idx = np.random.randint(0, n)
        sample = float(displacements_m[idx])
        distances = np.abs(displacements_m - sample)
        inlier_mask = distances < threshold_m
        inlier_count = int(np.sum(inlier_mask))
        if inlier_count > best_inliers:
            best_inliers = inlier_count
            best_value = float(np.mean(displacements_m[inlier_mask]))

    return best_value, best_inliers


class ArucoFlowTrack:
    def __init__(self, marker_id, light_name, light_index, bbox, gx, gy, target_distance, created_at):
        self.marker_id = int(marker_id)
        self.light_name = light_name
        self.light_index = light_index
        self.bbox = list(bbox)
        self.gx = float(gx)
        self.gy = float(gy)
        self.target_distance = float(target_distance)
        self.created_at = created_at
        self.last_seen = created_at
        self.state = 'pending'
        self.distance_covered = 0.0
        self.last_v = 0.0
        self.first_v = True
        self.safety_timer = None
        self.off_timer = None


class ArucoFlowWeeder:
    # 系统配置参数从 `config/aruco_flow_weeding.ini` 加载，包含相机参数、喷洒参数、车道划分等设置。
    def __init__(self, config_path='config/aruco_flow_weeding.ini'):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.logger = logging.getLogger(__name__)

        self.res = (self.config.getint('Camera', 'resolution_width'),
                    self.config.getint('Camera', 'resolution_height'))
        self.camera_to_light_y = self.config.getfloat('System', 'camera_to_light_y')
        self.spray_duration = self.config.getfloat('System', 'spray_duration')
        self.nozzle_offset = self.config.getfloat('System', 'nozzle_offset_m')
        self.track_timeout = self.config.getfloat('System', 'track_timeout_s')
        self.relay_response_s = self.config.getfloat('System', 'relay_response_s', fallback=0.05)
        self.safety_timeout_factor = self.config.getfloat('System', 'safety_timeout_factor', fallback=2.0)
        self.processing_delay_s = self.config.getfloat('System', 'processing_delay_s', fallback=0.02)
        self.scheduler_delay_s = self.config.getfloat('System', 'scheduler_delay_s', fallback=0.01)
        self.safety_margin_s = self.config.getfloat('System', 'safety_margin_s', fallback=0.02)

        self.flow_scale = self.config.getfloat('Flow', 'downsample_scale', fallback=0.25)
        self.flow_min_inliers = self.config.getint('Flow', 'min_inliers', fallback=5)

        print(f"[System] 继电器响应补偿: {self.relay_response_s} 秒")
        print(f"[Flow] downsample_scale={self.flow_scale}, min_inliers={self.flow_min_inliers}")

        relay_dict = {}
        for key, value in self.config['Relays'].items():
            relay_dict[int(key)] = int(value)

        # 车道/喷头分配：将逆透视后的 X 视野均分成 lane_count 份，
        # 二维码按所在车道分配对应喷头（relay）。
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

        self.vis = RelayVis(relays=len(relay_dict))
        self.relay_controller = RelayController(relay_dict, on_state_change=self.vis.update)

        self.flow_speedometer = OpticalFlowSpeedometer(
            h_matrix_path=self.config.get('System', 'ipm_matrix_path'),
            downsample_scale=self.config.getfloat('Flow', 'downsample_scale', fallback=0.25),
            image_width=self.res[0],
            image_height=self.res[1],
            min_inliers=self.config.getint('Flow', 'min_inliers', fallback=5),
            ransac_threshold_m=self.config.getfloat('Flow', 'ransac_threshold_m', fallback=0.03),
            max_corners=self.config.getint('Flow', 'max_corners', fallback=120),
            quality_level=self.config.getfloat('Flow', 'quality_level', fallback=0.03),
            min_distance=self.config.getint('Flow', 'min_distance', fallback=5),
        )
        self.flow_worker = AsyncOpticalFlowWorker(
            self.flow_speedometer,
            on_velocity=self._on_velocity_update,
            max_queue_size=self.config.getint('Flow', 'queue_size', fallback=8),
        )

        matrix_path = self.config.get('System', 'ipm_matrix_path')
        self.mapper = GroundCoordinateMapperIPM(matrix_path, self.res[0], self.res[1])
        top_center = self.mapper.pixel_to_ground(self.res[0] / 2.0, 0.0)
        self.lane_vis_y_end_m = max(0.1, top_center[1])

        self.detector = ArucoMarkerDetector()
        self.camera = VideoStream(src=0, resolution=self.res)

        self.tracks = {}
        self.track_timeout_s = self.track_timeout
        self._cleanup_timer = None
        self._tracks_lock = threading.Lock()
        self._flow_lock = threading.Lock()
        self._current_v = 0.0
        self._last_velocity_time = None
        self._flow_status = 'INIT'
        self._flow_inliers = 0

        # 运行时可视化统计
        self.display_fps = 0.0
        self.last_frame_time = None
        self.display_speed_mps = 0.0
        self.display_flow_status = 'INIT'

    def get_nearest_nozzle(self, target_x_m: float) -> int:
        """根据逆透视后的 X 坐标判断车道，并返回对应喷头 relay_id。"""
        if self.lane_count <= 1:
            return self.relay_ids[0]

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

    def _new_track(self, marker_id, bbox, gx, gy, timestamp):
        # 计算二维码与喷头的距离，作为触发喷洒的依据。理论上应该是二维码进入喷洒范围时开始计时，距离越近越快触发。
        target_distance = gy + self.nozzle_offset
        if target_distance <= 0:
            return None

        light_index = self.get_nearest_nozzle(gx)
        light_name = f'relay_{light_index}'
        track = ArucoFlowTrack(marker_id, light_name, light_index, bbox, gx, gy, target_distance, timestamp)
        with self._flow_lock:
            v_now = self._current_v
            v_ts = self._last_velocity_time
        if v_now > 0.001 and v_ts is not None and (timestamp - v_ts) < 0.5:
            track.last_v = float(v_now)
            track.first_v = False
        return track

    def _compensation_seconds(self):
        return self.relay_response_s + self.processing_delay_s + self.scheduler_delay_s + self.safety_margin_s

    def _on_velocity_update(self, flow_result: dict, frame_ts: float):
        if not flow_result.get('ready', False):
            return

        speed = float(flow_result.get('speed_fwd', 0.0))
        inliers = int(flow_result.get('inliers_cnt', 0))
        status_txt = str(flow_result.get('status_txt', 'INIT'))

        if not np.isfinite(speed) or inliers < self.flow_min_inliers:
            with self._flow_lock:
                prev_speed = self._current_v
            speed = max(0.0, prev_speed * 0.8)
            status_txt = f"{status_txt}|DEGRADED"

        event_ts = time.time() if frame_ts is None else float(frame_ts)
        with self._flow_lock:
            prev_t = self._last_velocity_time
            self._current_v = speed
            self._last_velocity_time = event_ts
            self._flow_status = status_txt
            self._flow_inliers = inliers

        self.display_speed_mps = speed if self.display_speed_mps <= 0 else (0.8 * self.display_speed_mps + 0.2 * speed)
        self.display_flow_status = status_txt

        if prev_t is None:
            return

        dt = event_ts - prev_t
        if dt <= 0.0 or dt > 0.2:
            return

        with self._tracks_lock:
            track_items = list(self.tracks.items())

        for marker_id, track in track_items:
            if track.state != 'pending':
                continue

            if track.first_v:
                track.last_v = speed
                track.first_v = False
                continue

            displacement = (track.last_v + speed) * 0.5 * dt
            track.distance_covered += max(0.0, displacement)
            track.last_v = speed

            compensation_dist = max(0.0, speed * self._compensation_seconds())
            remaining = track.target_distance - track.distance_covered - compensation_dist

            if remaining <= 0.0:
                self._safe_cancel_timer(track.safety_timer)
                track.safety_timer = None
                self._start_spray(track, event_ts)
            else:
                self._refresh_safety_timer(track, remaining, speed)

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

    def draw_runtime_overlay(self, frame):
        """在左上角实时显示光流速度与 FPS。"""
        speed_text = f"Flow v: {self.display_speed_mps:.3f} m/s"
        fps_text = f"FPS: {self.display_fps:.1f}"
        status_text = f"Flow: {self.display_flow_status} | Inliers: {self._flow_inliers}"
        cv2.putText(frame, speed_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, fps_text, (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, status_text, (20, 101), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def _refresh_safety_timer(self, track: ArucoFlowTrack, remaining: float, speed_mps: float):
        # 根据当前速度和剩余距离动态调整安全触发时间，避免因光流估计不稳导致过早或过晚喷洒。
        if speed_mps <= 0.01:
            return
        timeout_s = (max(0.0, remaining) / speed_mps) * self.safety_timeout_factor
        timeout_s = max(0.05, min(timeout_s, 10.0))

        self._safe_cancel_timer(track.safety_timer)

        def safety_cb(t=track):
            if t.state == 'pending':
                self.logger.warning(f'[SAFETY] marker={t.marker_id}')
                self._start_spray(t)

        track.safety_timer = threading.Timer(timeout_s, safety_cb)
        track.safety_timer.daemon = True
        track.safety_timer.start()

    def _safe_cancel_timer(self, timer):
        try:
            if timer is not None:
                timer.cancel()
        except Exception:
            pass

    def _start_spray(self, track: ArucoFlowTrack, event_ts: float | None = None):
        # 触发喷洒后进入 spraying 状态，等待喷洒完成的 off_timer 定时器回调将状态改为 done。
        if track.state != 'pending':
            return
        self._safe_cancel_timer(track.safety_timer)
        track.safety_timer = None
        track.state = 'spraying'
        track.last_seen = time.time() if event_ts is None else event_ts

        relay_id = track.light_index
        self.relay_controller.schedule_spray(
            relay_id=relay_id,
            delay_s=0.0,
            duration_s=self.spray_duration,
        )

        self._safe_cancel_timer(track.off_timer)

        def finish_cb(t=track):
            self._finish_spray(t)

        track.off_timer = threading.Timer(self.spray_duration, finish_cb)
        track.off_timer.daemon = True
        track.off_timer.start()

    def _finish_spray(self, track: ArucoFlowTrack, event_ts: float | None = None):
        # 喷洒完成后进入 done 状态，等待 track_timeout_s 后被清理掉。done 状态的 track 不再响应任何更新。
        if track.state == 'done':
            return
        track.state = 'done'
        track.last_seen = time.time() if event_ts is None else event_ts
        self.logger.info(
            f'[DONE] marker={track.marker_id} {track.light_name} '
            f'd={track.distance_covered:.3f}/{track.target_distance:.3f}m'
        )

    def _cleanup_old_tracks(self, now_ts: float | None = None):
        # 定期清理过旧的 track，避免内存泄漏。pending 状态的 track 超过 timeout 可能是光流估计失效导致的误触发，直接丢弃；spraying 状态的 track 超过 timeout 可能是喷洒完成但未正确进入 done 状态，强制进入 done；done 状态的 track 超过 timeout 正常清理掉。
        now = time.time() if now_ts is None else now_ts
        with self._tracks_lock:
            track_items = list(self.tracks.items())
        stale = []
        for marker_id, track in track_items:
            age = now - track.last_seen
            if track.state == 'pending' and age > self.track_timeout_s:
                self._safe_cancel_timer(track.safety_timer)
                stale.append(marker_id)
                print(f'[TIMEOUT] marker={marker_id} ({track.distance_covered:.3f}/{track.target_distance:.3f}m)')
            elif track.state == 'spraying' and age > self.track_timeout_s:
                stale.append(marker_id)
            elif track.state == 'done' and age > self.track_timeout_s:
                stale.append(marker_id)
        if stale:
            with self._tracks_lock:
                for marker_id in stale:
                    self.tracks.pop(marker_id, None)

    def run(self):
        print('[System] 正在启动相机...')
        self.camera.start()
        self.flow_worker.start()
        time.sleep(1.0)
        print("[System] 开始二维码光流距离喷洒作业 (按 'q' 退出)...\n")
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

                self.flow_worker.submit_frame(frame, frame_capture_time)
                
                boxes = self.detector.predict(frame)
                vis_frame = frame.copy()
                self.draw_lane_overlay(vis_frame)
                loop_ts = time.time()
                frame_time = frame_capture_time

                # 计算并平滑 FPS
                if self.last_frame_time is not None:
                    dt_frame = loop_ts - self.last_frame_time
                    if dt_frame > 1e-6:
                        fps_now = 1.0 / dt_frame
                        self.display_fps = fps_now if self.display_fps <= 0 else (0.85 * self.display_fps + 0.15 * fps_now)
                self.last_frame_time = loop_ts

                self._cleanup_old_tracks(frame_time)

                seen_marker_ids = set()
                for box in boxes:
                    x1, y1, x2, y2, marker_id = box
                    marker_id = int(marker_id)
                    seen_marker_ids.add(marker_id)

                    cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                    cv2.putText(vis_frame, f'ID:{marker_id}', (int(x1), int(y1) - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    gx_m, gy_m = self.mapper.pixel_to_ground(cx, cy)

                    with self._tracks_lock:
                        track = self.tracks.get(marker_id)
                    if track is None:
                        with self._tracks_lock:
                            track = self._new_track(marker_id, [x1, y1, x2, y2], gx_m, gy_m, frame_time)
                            if track is not None:
                                self.tracks[marker_id] = track
                        if track is None:
                            continue

                    if track.state != 'pending':
                        cv2.putText(
                            vis_frame,
                            f"STATE:{track.state}",
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2,
                        )
                        continue

                    track.bbox = [x1, y1, x2, y2]
                    track.gx = gx_m
                    track.gy = gy_m
                    track.last_seen = frame_time

                    cv2.putText(
                        vis_frame,
                        f"D:{track.distance_covered:.2f}/{track.target_distance:.2f}m",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                self.draw_runtime_overlay(vis_frame)

                cv2.imshow('aruco_flow_weeder', cv2.resize(vis_frame, (1024, 768)))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            pass
        finally:
            print('\n[System] 正在清理资源...')
            self.flow_worker.stop()
            self.vis.close()
            self.relay_controller.cleanup()
            self.camera.stop()
            cv2.destroyAllWindows()
            print('[System] 退出完毕。')


if __name__ == '__main__':
    weeder = ArucoFlowWeeder()
    weeder.run()
