#!/usr/bin/env python3
# coding: UTF-8
"""IMU 梯形积分测试脚本。

功能：
1. 调用 utils/imu_usb.py 的速度估计结果（vx, vy, vz）。
2. 对各轴速度做梯形积分，输出位移 dx, dy, dz。
3. 支持串口实时运行与可选 CSV 保存。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils import imu_usb


@dataclass
class TrapezoidIntegrator:
    last_time: float | None = None
    last_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    displacement: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def reset(self) -> None:
        self.last_time = None
        self.last_velocity[:] = 0.0
        self.displacement[:] = 0.0

    def update(self, velocity: np.ndarray, timestamp: float) -> tuple[np.ndarray, float]:
        velocity = np.asarray(velocity, dtype=np.float64)
        if self.last_time is None:
            self.last_time = float(timestamp)
            self.last_velocity = velocity.copy()
            return self.displacement.copy(), 0.0

        dt = float(timestamp) - float(self.last_time)
        if dt <= 0.0:
            self.last_time = float(timestamp)
            self.last_velocity = velocity.copy()
            return self.displacement.copy(), 0.0

        # 梯形积分：s += (v_prev + v_now) / 2 * dt
        self.displacement += 0.5 * (self.last_velocity + velocity) * dt
        self.last_velocity = velocity.copy()
        self.last_time = float(timestamp)
        return self.displacement.copy(), dt


def parse_args():
    parser = argparse.ArgumentParser(description='IMU 各轴速度与位移梯形积分测试')
    parser.add_argument('--port', default='/dev/ttyUSB0', help='串口设备')
    parser.add_argument('--baud', type=int, default=9600, help='波特率')
    parser.add_argument('--calibrate', action='store_true', help='启动时自动校准零偏')
    parser.add_argument('--samples', type=int, default=120, help='校准样本数')
    parser.add_argument('--compact', action='store_true', help='紧凑输出')
    parser.add_argument('--save', default=None, help='保存 CSV 文件名')
    parser.add_argument('--duration', type=float, default=0.0, help='运行时长(秒)，0 表示一直运行')
    return parser.parse_args()


def open_serial(port: str, baud: int):
    return imu_usb._open_serial(port, baud)


def print_status(integrator: TrapezoidIntegrator, latest_result: dict):
    vx, vy, vz = latest_result['velocity_enu_ms']
    dx, dy, dz = integrator.displacement
    speed = float(np.linalg.norm([vx, vy, vz]))
    print(
        f"\n{'=' * 72}\n"
        f"vx, vy, vz (m/s): {vx:.4f}, {vy:.4f}, {vz:.4f}\n"
        f"dx, dy, dz (m):   {dx:.4f}, {dy:.4f}, {dz:.4f}\n"
        f"speed (m/s):      {speed:.4f}\n"
        f"roll/pitch/yaw:   {latest_result['roll']:.2f}, {latest_result['pitch']:.2f}, {latest_result['yaw']:.2f}\n"
        f"stationary:       {latest_result['stationary']}\n"
        f"samples:          {len(imu_usb.estimator.history)}\n"
        f"{'=' * 72}"
    )


def print_compact(integrator: TrapezoidIntegrator, latest_result: dict):
    vx, vy, vz = latest_result['velocity_enu_ms']
    dx, dy, dz = integrator.displacement
    speed = float(np.linalg.norm([vx, vy, vz]))
    print(
        f"\rT:{time.time() - _start_time:6.1f}s | "
        f"v:[{vx:7.3f},{vy:7.3f},{vz:7.3f}] m/s | "
        f"s:[{dx:7.3f},{dy:7.3f},{dz:7.3f}] m | "
        f"spd:{speed:7.3f} | n:{len(imu_usb.estimator.history):5d}",
        end='',
        flush=True,
    )


def save_to_csv(filename: str, rows: list[dict]):
    import csv

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'time',
            'vx', 'vy', 'vz',
            'dx', 'dy', 'dz',
            'speed',
            'roll', 'pitch', 'yaw',
            'stationary',
        ])
        for row in rows:
            vx, vy, vz = row['velocity_enu_ms']
            dx, dy, dz = row['displacement_m']
            writer.writerow([
                row['timestamp'] - _start_time,
                vx, vy, vz,
                dx, dy, dz,
                row['speed_ms'],
                row['roll'], row['pitch'], row['yaw'],
                row['stationary'],
            ])
    print(f"\n[OK] 数据已保存: {filename} ({len(rows)} 条记录)")


def main():
    global _start_time
    args = parse_args()

    try:
        ser = open_serial(args.port, args.baud)
        print(f"[OK] 串口已打开: {args.port} @ {args.baud}")
    except Exception as exc:
        print(f"[ERROR] 串口打开失败: {exc}")
        return

    imu_usb.estimator.reset()
    integrator = TrapezoidIntegrator()
    saved_rows: list[dict] = []

    try:
        if args.calibrate:
            imu_usb.calibrate_accel_bias(ser, samples=args.samples)
        else:
            print("[INFO] 跳过校准，默认零偏为 [0, 0, 0]")

        print("\n" + "=" * 72)
        print("IMU 梯形积分测试开始")
        print("按 Ctrl+C 停止")
        print("=" * 72 + "\n")

        _start_time = time.time()
        last_report_count = 0

        while True:
            rx = ser.read(1)
            if not rx:
                continue

            imu_usb.DueData(int(rx[0]))
            if not imu_usb.estimator.history:
                continue

            latest = imu_usb.estimator.history[-1]
            velocity = np.asarray(latest['velocity_enu_ms'], dtype=np.float64)
            displacement, dt = integrator.update(velocity, latest['timestamp'])
            latest['displacement_m'] = displacement.copy()
            saved_rows.append(latest)

            if len(saved_rows) > 1 and len(imu_usb.estimator.history) != last_report_count:
                last_report_count = len(imu_usb.estimator.history)
                if args.compact:
                    print_compact(integrator, latest)
                else:
                    print_status(integrator, latest)

            if args.duration > 0 and (time.time() - _start_time) >= args.duration:
                break

    except KeyboardInterrupt:
        print("\n\n[INFO] 用户中断")
    finally:
        if args.save and saved_rows:
            save_to_csv(args.save, saved_rows)

        print("\n" + "=" * 72)
        print("最终状态")
        print("=" * 72)
        if imu_usb.estimator.history:
            print_status(integrator, imu_usb.estimator.history[-1])
            speeds = [float(np.linalg.norm(h['velocity_enu_ms'])) for h in imu_usb.estimator.history]
            print(f"[统计] 速度范围: {min(speeds):.4f} ~ {max(speeds):.4f} m/s")
            print(f"[统计] 平均速度: {np.mean(speeds):.4f} m/s")
            print(f"[统计] 最大速度: {max(speeds):.4f} m/s")
            print(f"[统计] 总数据点: {len(imu_usb.estimator.history)}")
        else:
            print("[统计] 没有可用数据")

        try:
            ser.close()
        except Exception:
            pass


if __name__ == '__main__':
    _start_time = time.time()
    main()
