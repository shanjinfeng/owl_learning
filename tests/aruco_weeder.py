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
        
        # 喷头分布坐标
        self.nozzle_x_positions = [float(x.strip()) for x in self.config.get('Nozzles', 'x_positions').split(',')]
        print(f"[System] 喷头 X 坐标 (m): {self.nozzle_x_positions}")
        
        # 继电器映射
        relay_dict = {}
        for key, value in self.config['Relays'].items():
            relay_dict[int(key)] = int(value)
            
        # 初始化终端可视化管理器
        self.vis = RelayVis(relays=len(relay_dict))
            
        # 初始化带 on_state_change 回调的继电器控制器 (回调绑定可视化的更新)
        self.relay_controller = RelayController(relay_dict, on_state_change=self.vis.update)
        
        # 初始化 IPM 映射器
        matrix_path = self.config.get('System', 'ipm_matrix_path')
        self.mapper = GroundCoordinateMapperIPM(matrix_path, self.res[0], self.res[1])
        
        # 初始化视觉组件
        self.detector = ArucoMarkerDetector()
        self.camera = VideoStream(src=0, resolution=self.res)
        
        # 防重复喷洒的记忆字典：{marker_id: last_seen_time}
        self.sprayed_markers = {}

    def get_nearest_nozzle(self, target_x_m: float) -> int:
        """根据 X 坐标(向右为正)分派最近的喷头"""
        best_idx = 0
        min_dist = float('inf')
        for i, nx in enumerate(self.nozzle_x_positions):
            dist = abs(target_x_m - nx)
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        return best_idx

    def clean_old_tracks(self):
        """清理过期的历史记录"""
        now = time.time()
        expired = [m_id for m_id, t in self.sprayed_markers.items() if now - t > self.track_timeout]
        for m_id in expired:
            del self.sprayed_markers[m_id]

    def run(self):
        print("[System] 正在启动相机...")
        self.camera.start()
        time.sleep(1.0)
        print("[System] 开始除草作业 (按 'q' 退出)...\n")
        
        # 启动可视化终端界面
        self.vis.setup()

        try:
            while True:
                # 获取图像的同时，记录这一帧画面产生的时间
                frame_capture_time = time.time()
                
                frame = self.camera.read()
                if frame is None:
                    time.sleep(0.01)
                    continue

                boxes = self.detector.predict(frame)
                now = time.time()
                
                # 计算这帧图像经过模型推理等步骤耗费的时间（秒）
                proc_time = now - frame_capture_time
                
                self.clean_old_tracks()

                vis_frame = frame.copy()

                for box in boxes:
                    x1, y1, x2, y2, marker_id = box
                    
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
                    
                    # 5. 记录防重喷 (这里记录为现在的绝对时间，避免短时间内被再次触发)
                    self.sprayed_markers[marker_id] = now
                    
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
