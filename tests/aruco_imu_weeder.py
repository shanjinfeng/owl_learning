#!/usr/bin/env python3
# coding: UTF-8
"""Aruco IMU weeder.

逻辑与 aruco_flow_weeder.py 保持一致，仅将速度来源从光流替换为 IMU 的 y 轴速度。
"""

import os
import sys
import math
import threading
from collections import deque
from typing import Callable, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

import aruco_flow_weeder as flow_weeder
from utils import imu_usb


class QueuedIMUVelocityWorker:
    def __init__(
        self,
        port: str,
        baud: int,
        auto_calibrate: bool,
        calibration_samples: int,
        calibration_timeout_s: float,
        forward_axis: str,
        forward_sign: float,
        read_timeout_s: float,
        on_velocity: Optional[Callable[[dict, float], None]] = None,
        max_queue_size: int = 64,
    ):
        self.port = port
        self.baud = baud
        self.auto_calibrate = auto_calibrate
        self.calibration_samples = calibration_samples
        self.calibration_timeout_s = calibration_timeout_s
        self.forward_axis = str(forward_axis).strip().lower()
        self.forward_sign = float(forward_sign)
        self.read_timeout_s = float(read_timeout_s)
        self.on_velocity = on_velocity
        self.max_queue_size = max(1, int(max_queue_size))
        self._lock = threading.Lock()
        self._queue = deque()
        self._queue_event = threading.Event()
        self._stop_event = threading.Event()
        self._reader_thread = None
        self._dispatch_thread = None
        self._serial = None
        self._dropped_samples = 0

    def start(self):
        if self._reader_thread is not None and self._reader_thread.is_alive():
            return

        imu_usb.estimator.reset()
        imu_usb.estimator.zupt_acc_threshold = imu_usb.G * 0.15
        imu_usb.estimator.zupt_gyro_threshold = 3.0
        imu_usb.estimator.velocity_hpf_alpha = math.exp(-self.read_timeout_s / 8.0) if self.read_timeout_s > 0 else 0.999
        self._serial = imu_usb._open_serial(self.port, self.baud)
        try:
            self._serial.timeout = self.read_timeout_s
        except Exception:
            pass

        if self.auto_calibrate:
            imu_usb.calibrate_accel_bias(self._serial, samples=self.calibration_samples)

        self._stop_event.clear()
        self._reader_thread = threading.Thread(target=self._read_loop, name='QueuedIMUVelocityReader', daemon=True)
        self._dispatch_thread = threading.Thread(target=self._dispatch_loop, name='QueuedIMUVelocityDispatch', daemon=True)
        self._reader_thread.start()
        self._dispatch_thread.start()

    def stop(self):
        self._stop_event.set()
        self._queue_event.set()

        for thread in (self._reader_thread, self._dispatch_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)

        try:
            if self._serial is not None:
                self._serial.close()
        except Exception:
            pass

    def submit_frame(self, frame, timestamp=None):
        return

    def _read_loop(self):
        try:
            while not self._stop_event.is_set():
                rx = self._serial.read(1)
                if not rx:
                    continue

                before_len = len(imu_usb.estimator.history)
                imu_usb.DueData(int(rx[0]))
                after_len = len(imu_usb.estimator.history)
                if after_len <= before_len:
                    continue

                sample = imu_usb.estimator.history[-1]
                result = self._sample_to_result(sample)
                with self._lock:
                    if len(self._queue) >= self.max_queue_size:
                        self._queue.popleft()
                        self._dropped_samples += 1
                    self._queue.append((result, float(sample['timestamp'])))
                self._queue_event.set()
        except Exception:
            pass

    def _dispatch_loop(self):
        while not self._stop_event.is_set():
            self._queue_event.wait(timeout=0.1)
            if self._stop_event.is_set():
                break

            item = None
            with self._lock:
                if self._queue:
                    item = self._queue.popleft()
                if not self._queue:
                    self._queue_event.clear()

            if item is None:
                continue

            result, timestamp = item
            if self.on_velocity is not None:
                try:
                    self.on_velocity(result, timestamp)
                except Exception:
                    pass

    def _sample_to_result(self, sample: dict) -> dict:
        velocity = sample.get('velocity_enu_ms', (0.0, 0.0, 0.0))
        axis_index = {'x': 0, 'y': 1, 'z': 2}.get(self.forward_axis, 1)
        speed_fwd = float(velocity[axis_index]) * self.forward_sign
        status_txt = 'IMU_STATIC' if sample.get('stationary', False) else 'IMU'
        return {
            'ready': True,
            'speed_fwd': speed_fwd,
            'speed_right': 0.0,
            'inliers_cnt': 1,
            'is_bad_frame': False,
            'status_txt': status_txt,
            'dt': float(sample.get('dt', 0.0)),
            'vis_img': None,
            'good_next': None,
        }


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

        imu_usb.estimator.zupt_acc_threshold = imu_usb.G * stationary_accel_threshold_g
        imu_usb.estimator.zupt_gyro_threshold = stationary_gyro_threshold_dps
        imu_usb.estimator.velocity_hpf_alpha = (
            math.exp(-read_timeout_s / velocity_damp_tau_s)
            if velocity_damp_tau_s > 0 and read_timeout_s > 0
            else 0.999
        )

        self.flow_worker = QueuedIMUVelocityWorker(
            port=imu_port,
            baud=imu_baud,
            auto_calibrate=auto_calibrate,
            calibration_samples=calibration_samples,
            calibration_timeout_s=calibration_timeout_s,
            forward_axis=forward_axis,
            forward_sign=forward_sign,
            read_timeout_s=read_timeout_s,
            on_velocity=self._on_velocity_update,
            max_queue_size=self.config.getint('IMU', 'queue_size', fallback=64),
        )


if __name__ == '__main__':
    weeder = ArucoIMUWeeder()
    weeder.run()