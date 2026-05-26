#!/usr/bin/env python3
# coding: UTF-8
"""Aruco IMU weeder.

逻辑与 aruco_flow_weeder.py 保持一致，仅将速度来源从光流替换为 IMU 的 y 轴速度。
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

import aruco_flow_weeder as flow_weeder
from utils.imu_usb import IMUSpeedometer


class ArucoIMUWeeder(flow_weeder.ArucoFlowWeeder):
    def __init__(self, config_path='config/aruco_imu_weeding.ini'):
        super().__init__(config_path)

        imu_port = self.config.get('IMU', 'port', fallback='/dev/ttyUSB0')
        imu_baud = self.config.getint('IMU', 'baud', fallback=9600)
        auto_calibrate = self.config.getboolean('IMU', 'auto_calibrate', fallback=True)
        calibration_samples = self.config.getint('IMU', 'calibration_samples', fallback=120)
        calibration_timeout_s = self.config.getfloat('IMU', 'calibration_timeout_s', fallback=15.0)
        stationary_accel_threshold_g = self.config.getfloat('IMU', 'stationary_accel_threshold_g', fallback=0.15)
        stationary_gyro_threshold_dps = self.config.getfloat('IMU', 'stationary_gyro_threshold_dps', fallback=3.0)
        forward_axis = self.config.get('IMU', 'forward_axis', fallback='y')
        forward_sign = self.config.getfloat('IMU', 'forward_sign', fallback=1.0)
        velocity_damp_tau_s = self.config.getfloat('IMU', 'velocity_damp_tau_s', fallback=8.0)
        read_timeout_s = self.config.getfloat('IMU', 'read_timeout_s', fallback=0.03)

        self.flow_speedometer = IMUSpeedometer(
            port=imu_port,
            baud=imu_baud,
            auto_calibrate=auto_calibrate,
            calibration_samples=calibration_samples,
            calibration_timeout_s=calibration_timeout_s,
            stationary_accel_threshold_g=stationary_accel_threshold_g,
            stationary_gyro_threshold_dps=stationary_gyro_threshold_dps,
            forward_axis=forward_axis,
            forward_sign=forward_sign,
            velocity_damp_tau_s=velocity_damp_tau_s,
            read_timeout_s=read_timeout_s,
        )


if __name__ == '__main__':
    weeder = ArucoIMUWeeder()
    weeder.run()