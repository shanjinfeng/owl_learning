#!/usr/bin/env python3
# coding: UTF-8
"""
IMU 速度输出脚本
兼容维特智能 10 轴 IMU V1.5.1 协议（逐字节读取）。

功能：
1. 利用 roll / pitch / yaw 做重力补偿。
2. 将体坐标系加速度旋转到东北天（ENU）坐标系。
3. 通过零速更新和轻微速度高通抑制积分漂移。
4. 输出 vx, vy, vz (m/s) 及合速度。
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

try:
    import serial
except ImportError:  # pragma: no cover - 串口依赖缺失时仍可运行模拟测试
    serial = None


G = 9.80665
FRAME_LENGTH = 11
FRAME_HEADER = 0x55
ACC_SCALE_G = 16.0
GYRO_SCALE_DPS = 2000.0
ANGLE_SCALE_DEG = 180.0


RxBuff = [0] * FRAME_LENGTH
ACCData = [0.0] * 8
GYROData = [0.0] * 8
AngleData = [0.0] * 8

start = 0
data_length = 0
CheckSum = 0

acc = [0.0] * 3
gyro = [0.0] * 3
Angle = [0.0] * 3


def _decode_signed_scaled(low_byte: float, high_byte: float, scale: float) -> float:
    value = (int(high_byte) << 8) | int(low_byte)
    if value >= 0x8000:
        value -= 0x10000
    return value / 32768.0 * scale


def get_acc(datahex):
    """解析加速度，单位 g。"""
    return (
        _decode_signed_scaled(datahex[0], datahex[1], ACC_SCALE_G),
        _decode_signed_scaled(datahex[2], datahex[3], ACC_SCALE_G),
        _decode_signed_scaled(datahex[4], datahex[5], ACC_SCALE_G),
    )


def get_gyro(datahex):
    """解析角速度，单位 °/s。"""
    return (
        _decode_signed_scaled(datahex[0], datahex[1], GYRO_SCALE_DPS),
        _decode_signed_scaled(datahex[2], datahex[3], GYRO_SCALE_DPS),
        _decode_signed_scaled(datahex[4], datahex[5], GYRO_SCALE_DPS),
    )


def get_angle(datahex):
    """解析姿态角，单位 °。"""
    return (
        _decode_signed_scaled(datahex[0], datahex[1], ANGLE_SCALE_DEG),
        _decode_signed_scaled(datahex[2], datahex[3], ANGLE_SCALE_DEG),
        _decode_signed_scaled(datahex[4], datahex[5], ANGLE_SCALE_DEG),
    )


def body_to_enu_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """根据 roll / pitch / yaw 生成体坐标系 -> ENU 坐标系旋转矩阵。"""
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


@dataclass
class IMUVelocityEstimator:
    """将 IMU 原始加速度和姿态角转换为 ENU 速度估计。"""

    accel_bias_body: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    velocity_enu: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    position_enu: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    last_timestamp: float | None = None
    history: deque = field(default_factory=lambda: deque(maxlen=2000))
    velocity_hpf_alpha: float = 0.999
    zupt_acc_threshold: float = 0.35
    zupt_gyro_threshold: float = 4.0
    max_dt: float = 0.2

    def reset(self) -> None:
        self.velocity_enu[:] = 0.0
        self.position_enu[:] = 0.0
        self.last_timestamp = None
        self.history.clear()

    def set_accel_bias(self, bias: np.ndarray | list[float]) -> None:
        self.accel_bias_body = np.asarray(bias, dtype=np.float64)

    def _linear_acc_enu(self, acc_g, angle_deg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        acc_body_ms2 = np.asarray(acc_g, dtype=np.float64) * G - self.accel_bias_body
        rotation = body_to_enu_matrix(*angle_deg)
        gravity_enu = np.array([0.0, 0.0, -G], dtype=np.float64)
        acc_enu = rotation @ acc_body_ms2
        linear_acc_enu = acc_enu + gravity_enu
        return acc_body_ms2, acc_enu, linear_acc_enu

    def process_sample(self, acc_g, gyro_dps, angle_deg, timestamp: float | None = None):
        """处理一组同步样本，返回本次解算结果。"""
        now = time.time() if timestamp is None else float(timestamp)
        if self.last_timestamp is None:
            dt = 0.0
        else:
            dt = now - self.last_timestamp
        self.last_timestamp = now

        acc_body_ms2, acc_enu, linear_acc_enu = self._linear_acc_enu(acc_g, angle_deg)
        gyro_vec = np.asarray(gyro_dps, dtype=np.float64)

        stationary = bool(
            np.linalg.norm(linear_acc_enu) < self.zupt_acc_threshold
            and np.linalg.norm(gyro_vec) < self.zupt_gyro_threshold
        )

        if dt > 0.0 and dt <= self.max_dt:
            self.velocity_enu += linear_acc_enu * dt
            if stationary:
                self.velocity_enu[:] = 0.0
            else:
                self.velocity_enu *= self.velocity_hpf_alpha
            self.position_enu += self.velocity_enu * dt

        speed = float(np.linalg.norm(self.velocity_enu))
        result = {
            "timestamp": now,
            "dt": dt,
            "acc_body_ms2": acc_body_ms2,
            "acc_enu_ms2": acc_enu,
            "linear_acc_enu_ms2": linear_acc_enu,
            "velocity_enu_ms": self.velocity_enu.copy(),
            "position_enu_m": self.position_enu.copy(),
            "speed_ms": speed,
            "roll": float(angle_deg[0]),
            "pitch": float(angle_deg[1]),
            "yaw": float(angle_deg[2]),
            "acc_g": tuple(float(v) for v in acc_g),
            "gyro_dps": tuple(float(v) for v in gyro_dps),
            "stationary": stationary,
        }
        self.history.append(result)
        return result

    def calibrate_bias_from_static_samples(self, samples: list[dict]) -> np.ndarray:
        """用静止样本估计体坐标系加速度零偏。"""
        biases = []
        for sample in samples:
            acc_body_ms2 = np.asarray(sample["acc_g"], dtype=np.float64) * G
            rotation = body_to_enu_matrix(*sample["angle_deg"])
            expected_body = rotation.T @ np.array([0.0, 0.0, G], dtype=np.float64)
            biases.append(acc_body_ms2 - expected_body)

        if biases:
            self.accel_bias_body = np.mean(np.asarray(biases), axis=0)
        return self.accel_bias_body


estimator = IMUVelocityEstimator()


def GetDataDeal(list_buf):
    """处理完整数据帧。"""
    global acc, gyro, Angle, CheckSum

    if list_buf[FRAME_LENGTH - 1] != CheckSum:
        return

    if list_buf[1] == 0x51:
        for i in range(6):
            ACCData[i] = list_buf[2 + i]
        acc = get_acc(ACCData)

    elif list_buf[1] == 0x52:
        for i in range(6):
            GYROData[i] = list_buf[2 + i]
        gyro = get_gyro(GYROData)

    elif list_buf[1] == 0x53:
        for i in range(6):
            AngleData[i] = list_buf[2 + i]
        Angle = get_angle(AngleData)
        update_velocity()


def DueData(inputdata):
    """逐字节接收处理。"""
    global start, CheckSum, data_length, RxBuff

    if inputdata == FRAME_HEADER and start == 0:
        start = 1
        data_length = FRAME_LENGTH
        CheckSum = 0
        for i in range(FRAME_LENGTH):
            RxBuff[i] = 0

    if start == 1:
        CheckSum += inputdata
        RxBuff[FRAME_LENGTH - data_length] = inputdata
        data_length -= 1
        if data_length == 0:
            CheckSum = (CheckSum - inputdata) & 0xFF
            start = 0
            GetDataDeal(RxBuff)


def update_velocity(timestamp: float | None = None):
    """基于当前全局 acc / gyro / Angle 更新速度状态。"""
    global acc, gyro, Angle
    return estimator.process_sample(acc, gyro, Angle, timestamp=timestamp)


def calibrate_accel_bias(ser, samples=300):
    """静态校准加速度零偏。"""
    estimator.reset()
    collected = []
    start_cal = time.time()

    print("[CALIBRATION] 请保持 IMU 完全静止...")
    print(f"[CALIBRATION] 采集 {samples} 个角度帧样本")

    while len(collected) < samples and (time.time() - start_cal) < 20:
        try:
            rx = ser.read(1)
            if not rx:
                continue
            DueData(int(rx[0]))

            if estimator.history:
                latest = estimator.history[-1]
                collected.append(
                    {
                        "acc_g": latest["acc_g"],
                        "angle_deg": (latest["roll"], latest["pitch"], latest["yaw"]),
                    }
                )

                if len(collected) % 50 == 0:
                    print(f"  进度: {len(collected)}/{samples}")
        except Exception as exc:
            print(f"[WARNING] 校准读取异常: {exc}")

    if len(collected) < 10:
        print("[WARNING] 校准样本不足，使用默认零偏 [0, 0, 0]")
        estimator.set_accel_bias([0.0, 0.0, 0.0])
        return False

    bias = estimator.calibrate_bias_from_static_samples(collected)
    print("[OK] 校准完成")
    print(f"  零偏: [{bias[0]:.4f}, {bias[1]:.4f}, {bias[2]:.4f}] m/s²")
    estimator.reset()
    return True


def print_velocity_status():
    """打印当前速度状态。"""
    speed = float(np.linalg.norm(estimator.velocity_enu))
    elapsed = time.time() - _start_time
    vx, vy, vz = estimator.velocity_enu
    px, py, pz = estimator.position_enu

    print(f"\n{'=' * 72}")
    print(f"运行时间: {elapsed:.2f}s | 数据点数: {len(estimator.history)}")
    print(f"vx, vy, vz (m/s): {vx:.4f}, {vy:.4f}, {vz:.4f}")
    print(f"合速度 speed (m/s): {speed:.4f}")
    print(f"位置 (m): x={px:.4f}, y={py:.4f}, z={pz:.4f}")
    print(f"姿态 (deg): roll={Angle[0]:.2f}, pitch={Angle[1]:.2f}, yaw={Angle[2]:.2f}")
    print(f"原始加速度 (g): ax={acc[0]:.4f}, ay={acc[1]:.4f}, az={acc[2]:.4f}")
    print(f"{'=' * 72}")


def print_compact():
    """紧凑输出模式。"""
    speed = float(np.linalg.norm(estimator.velocity_enu))
    elapsed = time.time() - _start_time
    vx, vy, vz = estimator.velocity_enu
    print(
        f"\rT:{elapsed:6.1f}s | vx:{vx:7.3f} vy:{vy:7.3f} vz:{vz:7.3f} | speed:{speed:7.3f} | "
        f"r:{Angle[0]:6.1f} p:{Angle[1]:6.1f} y:{Angle[2]:6.1f} | n:{len(estimator.history):5d}",
        end="",
        flush=True,
    )


def save_to_csv(filename='imu_velocity.csv'):
    """保存历史数据到 CSV。"""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'time', 'vx', 'vy', 'vz', 'speed',
            'ax_enu', 'ay_enu', 'az_enu',
            'lin_ax', 'lin_ay', 'lin_az',
            'roll', 'pitch', 'yaw',
            'pos_x', 'pos_y', 'pos_z'
        ])
        for h in estimator.history:
            vx, vy, vz = h['velocity_enu_ms']
            ax_enu, ay_enu, az_enu = h['acc_enu_ms2']
            lin_ax, lin_ay, lin_az = h['linear_acc_enu_ms2']
            px, py, pz = h['position_enu_m']
            writer.writerow([
                h['timestamp'] - _start_time,
                vx, vy, vz, h['speed_ms'],
                ax_enu, ay_enu, az_enu,
                lin_ax, lin_ay, lin_az,
                h['roll'], h['pitch'], h['yaw'],
                px, py, pz,
            ])
    print(f"\n[OK] 数据已保存: {filename} ({len(estimator.history)} 条记录)")


def _open_serial(port: str, baud: int):
    if serial is None:
        raise RuntimeError('pyserial 未安装，无法打开串口')
    return serial.Serial(port, baud, timeout=10)


def main():
    parser = argparse.ArgumentParser(description='IMU 速度输出工具')
    parser.add_argument('--port', default='/dev/ttyUSB0', help='串口设备')
    parser.add_argument('--baud', type=int, default=9600, help='波特率')
    parser.add_argument('--calibrate', action='store_true', help='启动校准')
    parser.add_argument('--compact', action='store_true', help='紧凑输出')
    parser.add_argument('--save', default=None, help='保存 CSV 文件名')
    parser.add_argument('--duration', type=float, default=0, help='运行时长(秒)')
    args = parser.parse_args()

    try:
        ser = _open_serial(args.port, args.baud)
        print(f"[OK] 串口已打开: {args.port} @ {args.baud}")
    except Exception as exc:
        print(f"[ERROR] 串口打开失败: {exc}")
        return

    try:
        if args.calibrate:
            calibrate_accel_bias(ser)
        else:
            print("[INFO] 跳过校准，默认零偏为 [0, 0, 0]")

        print("\n" + "=" * 72)
        print("IMU 速度输出开始")
        print("按 Ctrl+C 停止")
        print("=" * 72 + "\n")

        start_time = time.time()
        frame_count = 0

        while True:
            rx = ser.read(1)
            if not rx:
                continue

            DueData(int(rx[0]))
            frame_count += 1

            if frame_count % 30 == 0 and estimator.history:
                if args.compact:
                    print_compact()
                else:
                    print_velocity_status()

                if args.duration > 0 and (time.time() - start_time) > args.duration:
                    break

    except KeyboardInterrupt:
        print("\n\n[INFO] 用户中断")
    finally:
        if args.save and estimator.history:
            save_to_csv(args.save)

        print("\n" + "=" * 72)
        print("最终状态")
        print("=" * 72)
        print_velocity_status()

        if estimator.history:
            speeds = [h['speed_ms'] for h in estimator.history]
            print(f"[统计] 速度范围: {min(speeds):.4f} ~ {max(speeds):.4f} m/s")
            print(f"[统计] 平均速度: {np.mean(speeds):.4f} m/s")
            print(f"[统计] 最大速度: {max(speeds):.4f} m/s")
            print(f"[统计] 总数据点: {len(estimator.history)}")

        ser.close()
        print("[OK] 串口已关闭")


_start_time = time.time()


if __name__ == '__main__':
    main()