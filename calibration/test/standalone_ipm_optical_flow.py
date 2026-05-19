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
    print("[ERROR] 无法导入 gxipy，请先安装大恒 Galaxy SDK Python 绑定。")
    sys.exit(1)

# ================= 核心配置 =================
H_MATRIX_PATH = "/home/sjf/owl/calibration/calibration/H.npy"
DOWNSAMPLE_SCALE = 0.25  # 降采样比例，提升光流计算速度
IMAGE_WIDTH = 2048
IMAGE_HEIGHT = 1536

# 光流与坏帧处理参数
MIN_INLIERS = 5
MAX_BAD_FRAMES = 30
HISTORY_LEN = 15
RANSAC_THRESHOLD_M = 0.03  # RANSAC 内点容忍误差(米)

def configure_camera(cam):
    """简单配置相机，开启自动白平衡和自动曝光"""
    for node_name in ["BalanceWhiteAuto", "BalanceWhiteAutoMode", "AWBMode"]:
        if hasattr(cam, node_name):
            try: cam.getattr(node_name).set(2)
            except: pass
    if hasattr(cam, "ExposureAuto"):
        try: cam.ExposureAuto.set(2)
        except: pass
    if hasattr(cam, "GainAuto"):
        try: cam.GainAuto.set(2)
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

def main():
    # ========== 1. 加载单应性矩阵与坐标系 ==========
    if not os.path.exists(H_MATRIX_PATH):
        print(f"[ERROR] 找不到 {H_MATRIX_PATH}，请确认标定文件存在！")
        return
    
    H = np.load(H_MATRIX_PATH)
    H_inv = np.linalg.inv(H)
    
    # 建立车辆正前方和正右方向量
    p_bot = pixel_to_world(IMAGE_WIDTH / 2.0, IMAGE_HEIGHT, H_inv)
    p_top = pixel_to_world(IMAGE_WIDTH / 2.0, 0, H_inv)
    p_right_edge = pixel_to_world(IMAGE_WIDTH, IMAGE_HEIGHT, H_inv)

    vec_fwd_raw = np.array(p_top) - np.array(p_bot)
    vec_fwd = vec_fwd_raw / np.linalg.norm(vec_fwd_raw)
    
    vec_right_raw = np.array(p_right_edge) - np.array(p_bot)
    vec_right_ortho = vec_right_raw - np.dot(vec_right_raw, vec_fwd) * vec_fwd
    vec_right = vec_right_ortho / np.linalg.norm(vec_right_ortho)

    # ========== 2. 初始化大恒相机 ==========
    manager = gx.DeviceManager()
    dev_num, _ = manager.update_device_list()
    if dev_num == 0:
        print("[ERROR] 未发现大恒 GigE 相机！")
        return
    
    cam = manager.open_device_by_index(1)
    configure_camera(cam)
    cam.stream_on()
    print("[INFO] 相机已启动，开始光流测速...")

    # ========== 3. 状态与历史变量 ==========
    prev_gray = None
    prev_pts = None
    prev_time = time.time()
    
    vx_history = collections.deque(maxlen=HISTORY_LEN)
    vy_history = collections.deque(maxlen=HISTORY_LEN)
    prev_vx, prev_vy = 0.0, 0.0
    bad_frames = 0
    
    cv2.namedWindow("Optical Flow Speedometer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Optical Flow Speedometer", 1024, 768)

    # ========== 4. 主循环 ==========
    try:
        while True:
            # 抓取图像
            raw = cam.data_stream[0].get_image()
            if raw is None: continue
            
            arr = raw.convert("RGB").get_numpy_array()
            frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            current_time = time.time()
            dt = current_time - prev_time
            
            # 降采样提速
            frame_small = cv2.resize(frame, None, fx=DOWNSAMPLE_SCALE, fy=DOWNSAMPLE_SCALE, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            vis_img = frame_small.copy()

            if prev_gray is None or prev_pts is None or len(prev_pts) == 0:
                prev_gray = gray
                prev_time = current_time
                prev_pts = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.05, minDistance=5, blockSize=7)
                continue

            if dt > 0:
                # KLT 光流追踪
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, prev_pts, None,
                    winSize=(21, 21), maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.001)
                )

                good_prev = prev_pts[status.flatten() == 1].reshape(-1, 2)
                good_next = next_pts[status.flatten() == 1].reshape(-1, 2)

                vx, vy = 0.0, 0.0
                inliers_cnt = 0
                is_bad_frame = True

                if len(good_prev) > 0:
                    # 画出光流轨迹 (在缩放图上)
                    for (p0, p1) in zip(good_prev, good_next):
                        a, b = int(p0[0]), int(p0[1])
                        c, d = int(p1[0]), int(p1[1])
                        cv2.line(vis_img, (a, b), (c, d), (0, 255, 0), 2)
                        cv2.circle(vis_img, (c, d), 3, (0, 0, 255), -1)

                    # 1. 将特征点还原回原始尺寸 (因为 H 矩阵是用原图尺寸算的)
                    pts_prev_orig = good_prev / DOWNSAMPLE_SCALE
                    pts_next_orig = good_next / DOWNSAMPLE_SCALE
                    
                    # 2. 批量将像素映射到物理地面 (毫米)
                    world_prev_mm = cv2.perspectiveTransform(pts_prev_orig.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
                    world_next_mm = cv2.perspectiveTransform(pts_next_orig.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
                    
                    # 3. 计算位移 (毫米)
                    disp_world_mm = world_next_mm - world_prev_mm
                    
                    # 4. 投影到向前的方向并转化为米 (注意：地面往后退，说明车往前走，所以加负号)
                    disp_fwd_m = -np.dot(disp_world_mm, vec_fwd) / 1000.0
                    disp_right_m = -np.dot(disp_world_mm, vec_right) / 1000.0
                    
                    phys_displacements = np.column_stack((disp_fwd_m, disp_right_m))

                    # 5. RANSAC 过滤错误匹配
                    dx_m, dy_m, inliers_cnt = ransac_physical_translation(phys_displacements, threshold_m=RANSAC_THRESHOLD_M)

                    if inliers_cnt >= MIN_INLIERS:
                        vx = dx_m / dt
                        vy = dy_m / dt
                        is_bad_frame = False
                        
                        vx_history.append(vx)
                        vy_history.append(vy)
                        prev_vx, prev_vy = vx, vy
                        bad_frames = 0

                # 坏帧处理
                if is_bad_frame:
                    bad_frames += 1
                    if len(vx_history) >= 3:
                        vx = float(np.mean(vx_history))
                        vy = float(np.mean(vy_history))
                    else:
                        vx, vy = prev_vx, prev_vy
                        
                    if bad_frames > MAX_BAD_FRAMES:
                        vx *= 0.98
                        vy *= 0.98

                # ================= HUD 绘制与终端输出 =================
                status_txt = "NORMAL" if not is_bad_frame else f"PREDICT (Bad:{bad_frames})"
                color = (0, 255, 0) if not is_bad_frame else (0, 165, 255)
                
                # 1. 终端输出预测速度
                print(f"[SPEED] 向前: {vx:.3f} m/s | 向右: {vy:.3f} m/s | 状态: {status_txt} | 内点: {inliers_cnt}")

                # 2. 绘制图像速度信息 (调小字体 scale 和 line_thickness)
                fps = 1.0 / dt
                cv2.putText(vis_img, f"Speed Fwd: {vx:.3f} m/s", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(vis_img, f"Speed Right: {vy:.3f} m/s", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(vis_img, f"Status: {status_txt}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(vis_img, f"Inliers: {inliers_cnt} | FPS: {fps:.1f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                cv2.imshow("Optical Flow Speedometer", vis_img)

            # 更新下一帧状态
            prev_gray = gray
            prev_time = current_time
            
            # 判断成功追踪的内点数量
            # 只有当追踪丢失或点太少时（比如低于30个），才触发耗时的全局特征点重新检测
            if not is_bad_frame and len(good_next) > 30:
                prev_pts = good_next.reshape(-1, 1, 2)
            else:
                prev_pts = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.05, minDistance=5, blockSize=7)

            # 按 Q 退出
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