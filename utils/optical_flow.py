#!/usr/bin/env python3
import cv2
import numpy as np
import time
import os
import collections
import random
import sys

try:
    import gxipy as gx
except ImportError:
    gx = None

# ================= 核心配置 =================
H_MATRIX_PATH = "/home/jetson/Downloads/owl/calibration/calibration/H.npy"
DOWNSAMPLE_SCALE = 0.25  # 降采样比例，提升光流计算速度
IMAGE_WIDTH = 2048
IMAGE_HEIGHT = 1536

# 光流与坏帧处理参数
MIN_INLIERS = 5
MAX_BAD_FRAMES = 30
HISTORY_LEN = 15
RANSAC_THRESHOLD_M = 0.03  # RANSAC 内点容忍误差(米)

def configure_camera(cam):
    """配置相机参数以保证高帧率"""
    # 开启自动白平衡
    for node_name in ["BalanceWhiteAuto", "BalanceWhiteAutoMode", "AWBMode"]:
        if hasattr(cam, node_name):
            try: cam.getattr(node_name).set(2)
            except: pass
            
    # 开启自动曝光
    if hasattr(cam, "ExposureAuto"):
        try: cam.ExposureAuto.set(2)
        except: pass
        
    # 【核心优化 1】强制限制自动曝光的最大时间为 20ms (保证物理极限最低 50 FPS)
    # 如果环境比较暗，画面可能会变暗，但帧率能保证
    if hasattr(cam, "AutoExposureTimeMax"):
        try: cam.AutoExposureTimeMax.set(20000.0)
        except: pass
        
    # 开启自动增益并放宽增益上限，以弥补曝光时间缩短带来的画面变暗
    if hasattr(cam, "GainAuto"):
        try: cam.GainAuto.set(2)
        except: pass
    if hasattr(cam, "AutoGainMax"):
        try: cam.AutoGainMax.set(16.0)
        except: pass

def pixel_to_world(u, v, H_inv):
    """映射单个像素到物理世界 (单位: 毫米)"""
    w = H_inv[2, 0] * u + H_inv[2, 1] * v + H_inv[2, 2]
    if abs(w) < 1e-9: return (0.0, 0.0)
    x = (H_inv[0, 0] * u + H_inv[0, 1] * v + H_inv[0, 2]) / w
    y = (H_inv[1, 0] * u + H_inv[1, 1] * v + H_inv[1, 2]) / w
    return (x, y)

def ransac_physical_translation(displacements_m, max_iter=30, threshold_m=0.03):
    """物理世界中的 RANSAC 去噪"""
    n = len(displacements_m)
    if n == 0: return 0.0, 0.0, 0
    if n == 1: return float(displacements_m[0, 0]), float(displacements_m[0, 1]), 1

    best_inliers = 0
    best_dx, best_dy = 0.0, 0.0

    for _ in range(max_iter):
        idx = random.randint(0, n - 1)
        tx, ty = displacements_m[idx, 0], displacements_m[idx, 1]
        distances = np.hypot(displacements_m[:, 0] - tx, displacements_m[:, 1] - ty)
        inlier_mask = distances < threshold_m
        inlier_count = int(np.sum(inlier_mask))

        if inlier_count > best_inliers:
            best_inliers = inlier_count
            inlier_disps = displacements_m[inlier_mask]
            best_dx = float(np.mean(inlier_disps[:, 0]))
            best_dy = float(np.mean(inlier_disps[:, 1]))

    return best_dx, best_dy, best_inliers


class OpticalFlowSpeedometer:
    def __init__(
        self,
        h_matrix_path=H_MATRIX_PATH,
        downsample_scale=DOWNSAMPLE_SCALE,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        min_inliers=MIN_INLIERS,
        max_bad_frames=MAX_BAD_FRAMES,
        history_len=HISTORY_LEN,
        ransac_threshold_m=RANSAC_THRESHOLD_M,
        max_corners=200,
        quality_level=0.05,
        min_distance=5,
    ):
        self.downsample_scale = downsample_scale
        self.image_width = image_width
        self.image_height = image_height
        self.min_inliers = min_inliers
        self.max_bad_frames = max_bad_frames
        self.ransac_threshold_m = ransac_threshold_m
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance

        if not os.path.exists(h_matrix_path):
            raise FileNotFoundError(f"找不到 {h_matrix_path}")

        H = np.load(h_matrix_path)
        self.H_inv = np.linalg.inv(H)

        p_bot = pixel_to_world(image_width / 2.0, image_height, self.H_inv)
        p_top = pixel_to_world(image_width / 2.0, 0, self.H_inv)
        p_right_edge = pixel_to_world(image_width, image_height, self.H_inv)

        vec_fwd_raw = np.array(p_top) - np.array(p_bot)
        self.vec_fwd = vec_fwd_raw / np.linalg.norm(vec_fwd_raw)

        vec_right_raw = np.array(p_right_edge) - np.array(p_bot)
        vec_right_ortho = vec_right_raw - np.dot(vec_right_raw, self.vec_fwd) * self.vec_fwd
        self.vec_right = vec_right_ortho / np.linalg.norm(vec_right_ortho)

        self.prev_gray = None
        self.prev_pts = None
        self.prev_time = None
        self.vx_history = collections.deque(maxlen=history_len)
        self.vy_history = collections.deque(maxlen=history_len)
        self.prev_vx = 0.0
        self.prev_vy = 0.0
        self.bad_frames = 0

    def update(self, frame):
        current_time = time.time()
        frame_small = cv2.resize(
            frame,
            None,
            fx=self.downsample_scale,
            fy=self.downsample_scale,
            interpolation=cv2.INTER_LINEAR,
        )
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        vis_img = frame_small.copy()

        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) == 0:
            self.prev_gray = gray
            self.prev_time = current_time
            self.prev_pts = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                blockSize=7,
            )
            return {
                'ready': False,
                'speed_fwd': 0.0,
                'speed_right': 0.0,
                'inliers_cnt': 0,
                'is_bad_frame': True,
                'status_txt': 'INIT',
                'dt': 0.0,
                'vis_img': vis_img,
                'good_next': None,
            }

        dt = current_time - (self.prev_time or current_time)
        if dt <= 0:
            self.prev_gray = gray
            self.prev_time = current_time
            return {
                'ready': False,
                'speed_fwd': 0.0,
                'speed_right': 0.0,
                'inliers_cnt': 0,
                'is_bad_frame': True,
                'status_txt': 'INVALID_DT',
                'dt': 0.0,
                'vis_img': vis_img,
                'good_next': None,
            }

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.001)
        )

        vx, vy = 0.0, 0.0
        inliers_cnt = 0
        is_bad_frame = True
        good_next = None

        if next_pts is not None and status is not None:
            good_prev = self.prev_pts[status.flatten() == 1].reshape(-1, 2)
            good_next = next_pts[status.flatten() == 1].reshape(-1, 2)

            if len(good_prev) > 0:
                for (p0, p1) in zip(good_prev, good_next):
                    a, b = int(p0[0]), int(p0[1])
                    c, d = int(p1[0]), int(p1[1])
                    cv2.line(vis_img, (a, b), (c, d), (0, 255, 0), 2)
                    cv2.circle(vis_img, (c, d), 3, (0, 0, 255), -1)

                pts_prev_orig = good_prev / self.downsample_scale
                pts_next_orig = good_next / self.downsample_scale

                world_prev_mm = cv2.perspectiveTransform(
                    pts_prev_orig.reshape(-1, 1, 2), self.H_inv
                ).reshape(-1, 2)
                world_next_mm = cv2.perspectiveTransform(
                    pts_next_orig.reshape(-1, 1, 2), self.H_inv
                ).reshape(-1, 2)

                disp_world_mm = world_next_mm - world_prev_mm
                disp_fwd_m = -np.dot(disp_world_mm, self.vec_fwd) / 1000.0
                disp_right_m = -np.dot(disp_world_mm, self.vec_right) / 1000.0
                phys_displacements = np.column_stack((disp_fwd_m, disp_right_m))

                dx_m, dy_m, inliers_cnt = ransac_physical_translation(
                    phys_displacements,
                    threshold_m=self.ransac_threshold_m,
                )

                if inliers_cnt >= self.min_inliers:
                    vx = dx_m / dt
                    vy = dy_m / dt
                    is_bad_frame = False
                    self.vx_history.append(vx)
                    self.vy_history.append(vy)
                    self.prev_vx, self.prev_vy = vx, vy
                    self.bad_frames = 0

        if is_bad_frame:
            self.bad_frames += 1
            if len(self.vx_history) >= 3:
                vx = float(np.mean(self.vx_history))
                vy = float(np.mean(self.vy_history))
            else:
                vx, vy = self.prev_vx, self.prev_vy

            if self.bad_frames > self.max_bad_frames:
                vx *= 0.98
                vy *= 0.98

        status_txt = 'NORMAL' if not is_bad_frame else f'PREDICT (Bad:{self.bad_frames})'

        self.prev_gray = gray
        self.prev_time = current_time

        if not is_bad_frame and good_next is not None and len(good_next) > 30:
            self.prev_pts = good_next.reshape(-1, 1, 2)
        else:
            self.prev_pts = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                blockSize=7,
            )

        return {
            'ready': True,
            'speed_fwd': float(vx),
            'speed_right': float(vy),
            'inliers_cnt': int(inliers_cnt),
            'is_bad_frame': bool(is_bad_frame),
            'status_txt': status_txt,
            'dt': float(dt),
            'vis_img': vis_img,
            'good_next': good_next,
        }

def main():
    if gx is None:
        print("[ERROR] 无法导入 gxipy，请先安装大恒 Galaxy SDK Python 绑定。")
        return

    manager = gx.DeviceManager()
    dev_num, _ = manager.update_device_list()
    if dev_num == 0:
        print("[ERROR] 未发现大恒 GigE 相机！")
        return

    cam = manager.open_device_by_index(1)
    configure_camera(cam)
    cam.stream_on()
    print("[INFO] 相机已启动，开始光流测速...")

    speedometer = OpticalFlowSpeedometer()
    cv2.namedWindow("Optical Flow Speedometer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Optical Flow Speedometer", 1024, 768)

    try:
        while True:
            raw = cam.data_stream[0].get_image()
            if raw is None:
                continue

            arr = raw.get_numpy_array()

            if len(arr.shape) == 2:
                frame = cv2.cvtColor(arr, cv2.COLOR_BayerBG2BGR)
            else:
                frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

            result = speedometer.update(frame)
            vis_img = result['vis_img']

            if result['ready']:
                fps = 1.0 / result['dt'] if result['dt'] > 0 else 0.0
                print(
                    f"[SPEED] 向前: {result['speed_fwd']:.3f} m/s | 向右: {result['speed_right']:.3f} m/s | "
                    f"状态: {result['status_txt']} | 内点: {result['inliers_cnt']}"
                )
                cv2.putText(vis_img, f"Speed Fwd: {result['speed_fwd']:.3f} m/s", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(vis_img, f"Speed Right: {result['speed_right']:.3f} m/s", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(vis_img, f"Status: {result['status_txt']}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if not result['is_bad_frame'] else (0, 165, 255), 2)
                cv2.putText(vis_img, f"Inliers: {result['inliers_cnt']} | FPS: {fps:.1f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Optical Flow Speedometer", vis_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("[INFO] 用户中断")
    finally:
        print("[INFO] 正在关闭相机...")
        cam.stream_off()
        cam.close_device()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()