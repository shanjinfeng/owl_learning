#!/usr/bin/env python3
import sys
import os
import cv2
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils.optical_flow import gx, configure_camera, OpticalFlowSpeedometer


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
    print("[INFO] 相机已启动，开始光流测试，按 q 退出。")

    speedometer = OpticalFlowSpeedometer()
    cv2.namedWindow("Optical Flow Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Optical Flow Test", 1024, 768)

    try:
        while True:
            raw = cam.data_stream[0].get_image()
            if raw is None:
                time.sleep(0.01)
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

            cv2.imshow("Optical Flow Test", vis_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        print("[INFO] 正在关闭相机...")
        cam.stream_off()
        cam.close_device()
        cv2.destroyAllWindows()
        print("[INFO] 退出完毕。")


if __name__ == "__main__":
    main()
