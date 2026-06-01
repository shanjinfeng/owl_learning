# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OWL (Optical Weeding Laboratory) is an agricultural robotics project running on **Jetson Orin Nano** with JetPack 6.1. It performs targeted weed removal using computer vision (ArUco markers, green vegetation detection, YOLO models) with real-time relay-controlled spray nozzles. The system uses Inverse Perspective Mapping (IPM) to convert camera pixel coordinates to physical ground coordinates.

## Hardware Setup

```bash
# GPIO output mode (run once per boot)
sudo busybox devmem 0x02430070 w 0x08
sudo busybox devmem 0x02434040 w 0x04
sudo busybox devmem 0x02430068 w 0x08
sudo busybox devmem 0x02434080 w 0x05

# Max power mode
sudo nvpmodel -m 2
sudo jetson_clocks

# Required for Jetson.GPIO on JetPack 6.1
export JETSON_MODEL_NAME=JETSON_ORIN_NANO
```

## Running the System

The `tests/` directory contains the main entry points (they are standalone scripts, not unit tests):

```bash
# ArUco marker weeding (constant speed)
python tests/aruco_weeder.py

# ArUco marker weeding with optical flow speed estimation
python tests/aruco_flow_weeder.py

# ArUco marker weeding with IMU velocity estimation
python tests/aruco_imu_weeder.py

# Green-on-brown weed detection spraying
python tests/green_weeder.py
```

Each script loads its config from `config/<name>.ini`. Config paths are relative to the project root, so always run from the repo root.

## Architecture

### Data flow
```
Camera (Daheng GigE) → VideoStream → frame
                                      ↓
                    ┌─→ ArucoMarkerDetector → marker boxes
                    ├─→ OpticalFlowSpeedometer / IMU → vehicle velocity
                    ├─→ GreenOnBrown / GreenOnGreen → weed detections
                    └─→ GroundCoordinateMapperIPM → physical coordinates
                                      ↓
                           RelayController → GPIO → spray nozzles
```

### Velocity estimation strategies (selectable by config/entry script)
1. **Constant speed** (`aruco_weeder.py`) — fixed speed from config, simplest
2. **Optical flow** (`aruco_flow_weeder.py`) — KLT sparse optical flow + RANSAC on ground texture
3. **IMU** (`aruco_imu_weeder.py`) — WT-IMU serial protocol, attitude-compensated integration

### Key modules (`utils/`)

- **`video_manager.py`** — Daheng GigE camera wrapper. `DahengGigEStream` handles SDK init, threaded frame reading, and BGR conversion. `VideoStream` is a thin facade.
- **`output_manager.py`** — GPIO relay control with priority-queue scheduling (`SprayScheduler`) and reference counting. Auto-detects Jetson vs test hardware.
- **`marker_detect.py`** — OpenCV ArUco detector with consecutive-frame dedup (`min_consecutive_frames`).
- **`optical_flow.py`** — `OpticalFlowSpeedometer` does KLT tracking + RANSAC in world coordinates. `AsyncOpticalFlowWorker` runs it in a background thread with a bounded frame queue.
- **`imu_usb.py`** — WIT IMU protocol parser (0x55-framed packets). `IMUVelocityEstimator` does attitude rotation, ZUPT, and velocity integration. **Heavy use of module-level globals** (`estimator`, `acc`, `gyro`, `Angle`, `RxBuff`, etc.) — be aware when reusing.
- **`tracker.py`** — `ClassSmoother` (majority-vote class per track), `CropMaskStabilizer` (IOU-based track ID assignment with age-based pruning).
- **`algorithms.py`** — Vegetation indices (ExG, ExGR, CIVE, GNDVI, etc.) and blur detection.
- **`greenonbrown.py`** — `GreenOnBrown` combines ExG + HSV thresholding with morphological filtering. Duplicates its own `ArucoMarkerDetector` (simpler than the one in `marker_detect.py`).
- **`greenongreen.py`** — `GreenOnGreen` uses YOLO (NCNN or PyTorch) for crop detection with a hybrid mode that masks crops and runs ExHSV on remaining areas. Requires `ultralytics`.
- **`vis_manager.py`** — Terminal-based relay status visualization using ANSI escape codes (prefers `blessed` library).

### Known code duplication

- `GroundCoordinateMapperIPM` is copy-pasted into at least 4 files (`aruco_weeder.py`, `aruco_flow_weeder.py`, `green_weeder.py`, `test_camera_aruco.py`). Changes to coordinate logic must be replicated manually.
- `greenonbrown.py` duplicates `ArucoMarkerDetector` instead of importing from `marker_detect.py`.
- Lane/nozzle allocation logic is duplicated across all weeder classes.

### Calibration pipeline

```
calibration/calibration/camera_calibration.py → K.npy, dist.npy, H.npy
calibration/test/test_ipm_homography.py → ArUco physical positioning
calibration/test/standalone_ipm_optical_flow.py → optical flow speed test
```

## Dependencies

Core: `opencv-python`, `numpy`
Hardware: `Jetson.GPIO` (Jetson only), `gxipy` (Daheng Galaxy SDK), `pyserial` (IMU)
Optional: `ultralytics` (YOLO models), `torch` (GPU inference), `blessed` (terminal UI), `pywt` (wavelet blur)

No `requirements.txt` or `setup.py` exists — dependencies must be installed manually.
