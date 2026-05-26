#!/usr/bin/env python3
# coding: UTF-8
"""
IMU 输出速度测试脚本。

默认使用模拟数据验证：
1. 体坐标系加速度经过 roll/pitch/yaw 旋转后进入 ENU 坐标系。
2. 静止段通过 ZUPT 将速度拉回 0。
3. 输出格式满足 vx, vy, vz (m/s) + 合速度。

如需接真机串口，可加 --port /dev/ttyUSB0。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Iterable

import numpy as np


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.imu_usb import G, IMUVelocityEstimator, body_to_enu_matrix  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="IMU velocity output test")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="串口设备，mode=serial 时生效")
    parser.add_argument("--baud", type=int, default=9600, help="串口波特率")
    parser.add_argument("--duration", type=float, default=0.0, help="测试时长（秒），默认0表示持续运行直到手动终止")
    parser.add_argument("--compact", action="store_true", help="使用紧凑输出")
    parser.add_argument("--auto", action="store_true", help="自动扫描 /dev/ttyUSB* /dev/ttyACM* 并连接第一个设备")
    parser.add_argument("--calibrate", action="store_true", help="在开始前运行静态加速度零偏校准（需要串口）")
    parser.add_argument("--save", default=None, help="若指定则保存 CSV（仅在 serial 模式下有效）")
    return parser.parse_args()


def format_velocity_line(estimator: IMUVelocityEstimator) -> str:
    vx, vy, vz = estimator.velocity_enu
    speed = float(np.linalg.norm(estimator.velocity_enu))
    return f"vx={vx:.4f} m/s, vy={vy:.4f} m/s, vz={vz:.4f} m/s | speed={speed:.4f} m/s"


def run_serial(port: str, baud: int, duration_s: float, compact: bool, args) -> None:
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("pyserial 未安装，无法运行串口测试") from exc

    from utils.imu_usb import DueData, estimator, calibrate_accel_bias

    # 自动检测端口（优先用户指定）
    if port is None and getattr(args, 'auto', False):
        port_candidates = []
    else:
        port_candidates = [port]

    # 打开串口
    print("=== IMU 串口速度测试 ===")
    print(f"port={port}, baud={baud}")
    ser = serial.Serial(port, baud, timeout=1)

    # 可选校准
    if args.calibrate:
        try:
            calibrate_accel_bias(ser)
        except Exception as e:
            print(f"[WARN] 校准失败: {e}")

    start_time = time.time()
    frame_count = 0
    try:
        while True:
            # 若指定了正数 duration，则在超过时长后结束
            if duration_s > 0 and (time.time() - start_time) >= duration_s:
                break

            rx = ser.read(1)
            if not rx:
                continue
            DueData(int(rx[0]))
            frame_count += 1
            if frame_count % 30 == 0 and estimator.history:
                if compact:
                    print(format_velocity_line(estimator))
                else:
                    print(format_velocity_line(estimator))
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断，正在退出...")
    finally:
        ser.close()
        print("\n最终结果:")
        print(format_velocity_line(estimator))

    # 保存 CSV（如果指定）
    if args.save and estimator.history:
        from utils.imu_usb import save_to_csv as imu_save
        imu_save(args.save)


def main():
    args = parse_args()
    # Serial-only test
    run_serial(args.port, args.baud, args.duration, args.compact, args)


if __name__ == "__main__":
    main()
