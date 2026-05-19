# OWL - 相机标定与视觉感知工具集

基于棋盘格的单目相机标定工具，支持内参标定、畸变校正、鸟瞰图（BEV）生成、ArUco 二维码地面定位以及光流测速。

## 目录结构

```
owl/
└── calibration/
    ├── calibration/
    │   └── camera_calibration.py          # 相机标定主程序
    ├── calibration_png/                   # 标定用棋盘格图像（20张）
    ├── test_png/                          # 测试图像（5张）
    └── test/
        ├── marker_detect.py               # ArUco 二维码检测器
        ├── test_ipm_homography.py         # 逆透视变换 + 二维码物理定位
        └── standalone_ipm_optical_flow.py # 光流测速（大恒相机）
```

## 依赖

- Python 3.x
- OpenCV（`cv2`）
- NumPy
- gxipy（大恒 Galaxy SDK Python 绑定，仅光流测速需要）,下载了其官方的python库，可以问ai怎么将相关库导入到现有的环境中

安装基础依赖：

```bash
pip install opencv-python numpy
```

## 模块说明

### 1. 相机标定 — `camera_calibration.py`

基于棋盘格的单目相机标定，输出内参矩阵、畸变系数和单应性矩阵，并生成鸟瞰图。

**配置参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `IMAGE_DIR` | 标定图像目录 | 需修改为实际路径 |
| `chessboard_size` | 棋盘格内角点数 | `(11, 8)` |
| `square_size` | 方格边长（mm） | `20.1` |

**使用方法：**

```bash
cd calibration/calibration
python camera_calibration.py
```

**输出文件：**

| 文件 | 说明 |
|------|------|
| `K.npy` | 相机内参矩阵（3×3） |
| `dist.npy` | 畸变系数（径向 + 切向） |
| `H.npy` | 单应性矩阵（3×3），用于图像到地面的透视变换 |
| `bev_result.jpg` | 鸟瞰图结果 |

标定图像不少于 5 张，建议 15-20 张覆盖图像各区域、多角度拍摄。

---

### 2. ArUco 二维码检测器 — `marker_detect.py`

封装了 OpenCV ArUco 检测功能，提供统一接口。

**核心类：`ArucoMarkerDetector`**

| 方法 | 说明 |
|------|------|
| `predict(frame)` | 检测 ArUco 二维码，返回 `[x1, y1, x2, y2, marker_id]` 格式的检测框列表 |
| `predict_with_timing(frame)` | 同上，附加返回耗时（ms） |
| `predict_without_id(frame)` | 仅返回检测框坐标，不含 ID |
| `draw_boxes(frame, boxes)` | 在图像上绘制检测框和 ID 标签 |
| `get_marker_centers()` | 获取上次检测的二维码中心坐标及 ID |

**参数配置：**

- 默认字典类型：`DICT_6X6_250`
- 默认开启亚像素角点细化（`CORNER_REFINE_SUBPIX`）
- 自适应阈值窗口：3 ~ 23

**使用示例：**

```python
from marker_detect import ArucoMarkerDetector

detector = ArucoMarkerDetector()
boxes, elapsed_ms = detector.predict_with_timing(frame)
vis = detector.draw_boxes(frame, boxes)
```

直接运行脚本可进行性能基准测试（100 次迭代取平均 + 实时帧率显示）：

```bash
cd calibration/test
python marker_detect.py
```

---

### 3. 逆透视定位 — `test_ipm_homography.py`

利用标定阶段生成的单应性矩阵 `H.npy`，将图像中检测到的 ArUco 二维码像素坐标转换为真实世界物理坐标。

**核心功能：**

- 加载逆单应性矩阵，建立车辆坐标系（以图像底部中点为原点）
- 检测测试图像中的 ArUco 二维码
- 输出每个二维码的相对物理位置：**前方距离（cm）** 和 **右侧偏移（cm）**

**依赖文件：**
- `H.npy`（需先运行 `camera_calibration.py` 生成）
- `marker_detect.py`（同目录下的二维码检测器）

**使用方法：**

```bash
cd calibration/test
python test_ipm_homography.py
```

**坐标系说明：**

- 原点：图像底部中点（相机正下方地面）
- 向前：图像垂直向上的方向在物理地面的投影，正值为正前方
- 向右：与向前方向正交的右侧，正值为右侧

---

### 4. 光流测速 — `standalone_ipm_optical_flow.py`

基于大恒（Galaxy）相机实时视频流，使用 KLT 稀疏光流结合 RANSAC 去噪，将地面纹理运动映射为车辆速度。

**核心流程：**

1. **相机初始化** — 通过 `gxipy` 打开大恒 GigE 相机，配置自动曝光和白平衡
2. **特征提取** — 使用 `goodFeaturesToTrack` 提取 Shi-Tomasi 角点
3. **光流追踪** — KLT 金字塔光流（`calcOpticalFlowPyrLK`）逐帧追踪特征点
4. **坐标映射** — 通过逆单应性矩阵将特征点的像素位移映射到物理地面位移
5. **RANSAC 滤波** — 剔除错误匹配，输出平稳的速度估计
6. **坏帧处理** — 对质量不足的帧使用历史均值预测，超过阈值逐渐衰减

**主要参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `DOWNSAMPLE_SCALE` | 降采样比例，提升光流速度 | `0.25` |
| `MIN_INLIERS` | RANSAC 最少内点数 | `5` |
| `MAX_BAD_FRAMES` | 连续坏帧容忍上限 | `30` |
| `HISTORY_LEN` | 速度历史窗口长度 | `15` |
| `RANSAC_THRESHOLD_M` | RANSAC 内点容忍误差（米） | `0.03` |

**输出：**

- 终端实时打印前向速度（m/s）和横向速度（m/s）
- 可视化窗口显示光流轨迹和速度 HUD，按 `Q` 退出

**依赖文件：**
- `H.npy`（需先运行 `camera_calibration.py` 生成）
- `gxipy`（大恒相机 SDK）

**使用方法：**

```bash
cd calibration/test
python standalone_ipm_optical_flow.py
```

---

## 工作流程概要

```
标定图像 → camera_calibration.py → K.npy / dist.npy / H.npy
                                         │
                    ┌────────────────────┘
                    ▼
         test_ipm_homography.py     standalone_ipm_optical_flow.py
         (二维码物理定位)             (光流实时测速)
              │                           │
              ▼                           ▼
         依赖 marker_detect.py        依赖 gxipy + 大恒相机
```

## 注意事项

- **棋盘格标定**：至少拍摄 5 张不同角度的棋盘格图像，建议覆盖画面中心、四角及远近不同距离
- **二维码定位**：需确保二维码完整出现在画面中，定位精度取决于标定质量
- **光流测速**：依赖于地面纹理特征；光滑/无纹理地面可能导致特征点不足，触发坏帧预测
- 光流测速需要安装大恒 Galaxy SDK 的 Python 绑定（`gxipy`），其他模块无需此依赖
