#!/usr/bin/env python
"""使用 Ultralytics YOLO 进行绿对绿杂草检测。

支持 NCNN 或 PyTorch 格式的检测和分割模型。
NCNN 是 Jetson Orin Nano 的推荐格式（ARM CPU 上最快）。

混合模式：YOLO 在低分辨率下识别作物，然后 GreenOnBrown
在全分辨率下对非作物区域运行 ExHSV 以查找杂草。

输入：BGR 图像
输出：contours（分割模型的掩膜轮廓或 None），boxes（[x, y, w, h] 列表），weed_centres（[[cx, cy], ...]），image_out（带有检测叠加的图像副本，如果 show_display=True，否则为原始图像）
两种输出模式：
- 纯 GoG 模式：直接使用 YOLO 进行检测，输出所有检测结果。
- 混合模式：首先使用 YOLO 检测作物区域，生成作物掩膜，然后在非作物区域运行 GreenOnBrown 的 ExHSV 算法进行杂草检测，输出过滤后的杂草检测结果。
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)


class GreenOnGreen:
    def __init__(self, model_path='models', confidence=0.5, detect_classes=None,
                 hybrid_mode=False, inference_resolution=320, crop_buffer_px=20,
                 tracking_enabled=False, crop_stabilizer=None,
                 detection_persist_frames=0):
        """
        参数：
            model_path: NCNN 模型目录、.pt 文件或包含模型的父目录路径。
            confidence: 检测置信度阈值 (0.0-1.0)。
            detect_classes: 要检测的类名列表 (None = 全部)。
            hybrid_mode: 若为 True，使用 YOLO 进行作物掩膜 + GreenOnBrown 进行杂草检测。
            inference_resolution: 混合模式下 YOLO 输入分辨率 (越低越快)。
            crop_buffer_px: 检测到的作物周围膨胀缓冲像素数 (混合模式)。
        """
        if YOLO is None:
            raise ImportError(
                "ultralytics is required for Green-on-Green detection but is not installed. "
                "Install with: pip install -r requirements-gog.txt"
            )

        self.model_path = Path(model_path)
        self.confidence = confidence
        self.hybrid_mode = hybrid_mode
        self.inference_resolution = inference_resolution
        self.crop_buffer_px = crop_buffer_px
        self._model_filename = ''
        self._is_ncnn = False
        self.model = self._load_model()
        self.device = self._select_device()
        self.task = self.model.task  # 'detect' or 'segment'
        self.detection_mask = None  # Combined binary mask, set after inference (seg only)

        # 在模型加载后将类名映射到 ID
        # 在模型加载后将类名映射到 ID
        self._detect_class_ids = self._resolve_classes(detect_classes)

        # 跟踪状态 (通过 model.track() 使用 ByteTrack)
        # 启用跟踪时，存储上一帧的跟踪 ID、原始边界框、类别 ID 和置信度，以便在后续帧中进行稳定性处理和检测持久化。
        self.tracking_enabled = tracking_enabled
        self._crop_stabilizer = crop_stabilizer
        self.detection_persist_frames = detection_persist_frames
        # 原始检测属性 (启用跟踪时每帧填充)
        # 原始检测属性 (启用跟踪时每帧填充)
        self.last_track_ids = []
        self.last_raw_boxes = []
        self.last_class_ids = []
        self.last_confidences = []

        # 混合模式：创建内部 GreenOnBrown 实例、膨胀核和线程池
        # 混合模式：创建内部 GreenOnBrown 实例、膨胀核和线程池
        self._gob = None
        self._dilate_kernel = None
        self._executor = None
        if self.hybrid_mode:
            from utils.greenonbrown import GreenOnBrown
            self._gob = GreenOnBrown(algorithm='exhsv')
            self._dilate_kernel = self._build_dilate_kernel(crop_buffer_px)
            self._executor = ThreadPoolExecutor(max_workers=1)
            logger.info(f'Hybrid mode enabled: YOLO crop mask + ExHSV weed detection, '
                        f'buffer={crop_buffer_px}px, imgsz={inference_resolution}')

        logger.info(f'GreenOnGreen initialized: task={self.task}, '
                     f'classes={list(self.model.names.values())}, '
                     f'filtering={detect_classes or "all"}, '
                     f'device={self.device}')

    def _select_device(self):
        """Prefer GPU for PyTorch models; keep NCNN on CPU."""
        if self._is_ncnn:
            return 'cpu'
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    return 'cuda'
            except Exception:
                pass
        return 'cpu'

    @staticmethod
    def _infer_task(name, model_path=None):
        """从文件名或 metadata.yaml 推断 YOLO 任务类型。"""
        if '-seg' in name.lower() or '_seg' in name.lower():
            return 'segment'
        # 回退到 NCNN 模型目录内的 metadata.yaml
        if model_path is not None:
            meta = Path(model_path) / 'metadata.yaml'
            if meta.exists():
                try:
                    with open(meta) as f:
                        for line in f:
                            if line.startswith('task:'):
                                task = line.split(':', 1)[1].strip()
                                if task:
                                    logger.info(f'Task "{task}" read from metadata.yaml')
                                    return task
                except Exception:
                    pass
        return None

    def _load_model(self):
        """加载 YOLO 模型 -- 支持 NCNN 目录和 .pt 文件。"""
        if self.model_path.is_dir():
            # 检查这是否是 NCNN 模型目录 (包含 .param + .bin)
            if list(self.model_path.glob('*.param')):
                logger.info(f'Using NCNN model: {self.model_path.name}')
                self._model_filename = self.model_path.name
                self._is_ncnn = True
                task = self._infer_task(self.model_path.name, self.model_path)
                return YOLO(str(self.model_path), task=task)

            # 先搜索 NCNN 子目录，然后搜索 .pt 文件
            ncnn_dirs = [d for d in self.model_path.iterdir()
                         if d.is_dir() and list(d.glob('*.param'))]
            if ncnn_dirs:
                selected = ncnn_dirs[0]
                logger.info(f'Using NCNN model: {selected.name}')
                self._model_filename = selected.name
                self._is_ncnn = True
                task = self._infer_task(selected.name, selected)
                return YOLO(str(selected), task=task)

            pt_files = list(self.model_path.glob('*.pt'))
            if pt_files:
                logger.info(f'Using PyTorch model: {pt_files[0].name}')
                self._model_filename = pt_files[0].name
                self._is_ncnn = False
                task = self._infer_task(pt_files[0].name)
                return YOLO(str(pt_files[0]), task=task)

            raise FileNotFoundError(f'No YOLO models found in {self.model_path}')

        elif self.model_path.exists():
            logger.info(f'Using model: {self.model_path.name}')
            self._model_filename = self.model_path.name
            self._is_ncnn = False
            task = self._infer_task(self.model_path.name)
            return YOLO(str(self.model_path), task=task)

        raise FileNotFoundError(f'Model path does not exist: {self.model_path}')

    def _resolve_classes(self, class_names):
        """将类名映射到模型类 ID。若为全部类则返回 None。"""
        if not class_names:
            return None

        name_to_id = {v.lower(): k for k, v in self.model.names.items()}
        ids = []
        for name in class_names:
            name_lower = name.strip().lower()
            if name_lower in name_to_id:
                ids.append(name_to_id[name_lower])
            else:
                logger.warning(f"Class '{name}' not found in model. "
                               f"Available: {list(self.model.names.values())}")

        return ids if ids else None

    def _build_dilate_kernel(self, px):
        """为作物缓冲构建椭圆膨胀核。"""
        if px <= 0:
            return None
        size = 2 * px + 1
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    def set_crop_buffer(self, px):
        """更新作物缓冲并在值改变时重建核。"""
        if px == self.crop_buffer_px:
            return
        self.crop_buffer_px = px
        self._dilate_kernel = self._build_dilate_kernel(px)
        logger.info(f'Crop buffer updated to {px}px')

    @staticmethod
    def _build_crop_mask(results, h, w):
        """从单帧 YOLO 结果构建作物掩膜 (无稳定化)。"""
        mask = np.zeros((h, w), dtype=np.uint8)
        for result in results:
            if result.masks is not None:
                contours_full = [c.astype(np.int32).reshape(-1, 1, 2)
                                 for c in result.masks.xy]
                cv2.drawContours(mask, contours_full, -1, 255, -1)
            elif len(result.boxes):
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        return mask

    # 跟踪稳定性预置 —— 将用户可见名称映射到 ByteTrack 参数
    TRACK_STABILITY_PRESETS = {
        'low': {
            'track_high_thresh': 0.3,
            'track_low_thresh': 0.15,
            'new_track_thresh': 0.3,
            'track_buffer': 30,
            'match_thresh': 0.8,
        },
        'medium': {
            'track_high_thresh': 0.2,
            'track_low_thresh': 0.05,
            'new_track_thresh': 0.2,
            'track_buffer': 60,
            'match_thresh': 0.7,
        },
        'high': {
            'track_high_thresh': 0.15,
            'track_low_thresh': 0.05,
            'new_track_thresh': 0.15,
            'track_buffer': 90,
            'match_thresh': 0.6,
        },
    }

    def update_tracker_params(self, stability_level):
        """从稳定性预置在运行时更新 ByteTrack 参数。

        参数：
            stability_level: 'low'、'medium' 或 'high'
        """
        preset = self.TRACK_STABILITY_PRESETS.get(stability_level)
        if not preset:
            logger.warning(f"Unknown track stability level: {stability_level}")
            return

        if not hasattr(self.model, 'predictor') or self.model.predictor is None:
            logger.info(f"Track stability set to '{stability_level}' (will apply on next track call)")
            return

        for tracker in getattr(self.model.predictor, 'trackers', []):
            for key, value in preset.items():
                if hasattr(tracker, key):
                    setattr(tracker, key, value)
                # track_buffer 设置 BYTETracker 中的 max_time_lost
                if key == 'track_buffer' and hasattr(tracker, 'max_time_lost'):
                    tracker.max_time_lost = value
            tracker.reset()

        logger.info(f"Track stability updated to '{stability_level}': {preset}")

    def update_tracker_params_direct(self, params):
        """从单个值的字典在运行时更新 ByteTrack 参数。

        参数：
            params: 包含 'track_high_thresh'、'match_thresh' 等键的字典。
        """
        if not hasattr(self.model, 'predictor') or self.model.predictor is None:
            logger.info(f"Tracker params queued (will apply on next track call): {params}")
            return

        for tracker in getattr(self.model.predictor, 'trackers', []):
            for key, value in params.items():
                if hasattr(tracker, key):
                    setattr(tracker, key, value)
                if key == 'track_buffer' and hasattr(tracker, 'max_time_lost'):
                    tracker.max_time_lost = int(value)
            tracker.reset()

        logger.info(f"Tracker params updated directly: {params}")

    def get_lost_tracks(self, max_age=None):
        """从 ByteTrack 读取丢失轨迹的卡尔曼预测位置。

        ByteTrack 内部维护 lost_stracks，每帧更新卡尔曼预测位置。
        这使其可用于检测持久化 —— 边界框在 YOLO 闪烁期间持续存在。

        参数：
            max_age: 自上次匹配以来的最大帧数 (None = 无限制)。

        返回：
            字典列表: [{'track_id', 'xyxy', 'cls', 'score', 'age'}]
        """
        if not (hasattr(self.model, 'predictor') and self.model.predictor):
            return []
        trackers = getattr(self.model.predictor, 'trackers', [])
        if not trackers:
            return []

        tracker = trackers[0]  # single-stream
        lost = []
        try:
            for strack in tracker.lost_stracks:
                age = tracker.frame_id - strack.end_frame
                if max_age is not None and age > max_age:
                    continue
                xyxy = strack.xyxy
                lost.append({
                    'track_id': strack.track_id,
                    'xyxy': xyxy,
                    'cls': int(strack.cls),
                    'score': float(strack.score),
                    'age': age,
                })
        except Exception:
            # 防御性 —— Ultralytics 内部 API 可能变化
            pass
        return lost

    def reset_tracker(self):
        """重置 ByteTrack 状态和跟踪层。检测关闭时调用。"""
        self.last_track_ids = []
        self.last_raw_boxes = []
        self.last_class_ids = []
        self.last_confidences = []
        if self._crop_stabilizer:
            self._crop_stabilizer.reset()
        # 重置 ByteTrack 内部状态
        if hasattr(self.model, 'predictor') and self.model.predictor is not None:
            for tracker in getattr(self.model.predictor, 'trackers', []):
                tracker.reset()

    def update_detect_classes(self, class_names):
        """无需重新加载模型即可热更新 detect_classes 过滤器。"""
        self._detect_class_ids = self._resolve_classes(class_names)
        logger.info(f'detect_classes updated: {class_names} -> IDs {self._detect_class_ids}')

    @property
    def class_names(self):
        """从已加载模型返回 {id: name} 字典。"""
        return self.model.names

    def inference(self, image, confidence=0.5, show_display=False,
                  filter_id=None, label='WEED', build_mask=False,
                  # 混合模式参数 (非混合模式下忽略)：
                  exg_min=30, exg_max=250, hue_min=30, hue_max=90,
                  saturation_min=30, saturation_max=255,
                  brightness_min=5, brightness_max=200,
                  min_detection_area=1, invert_hue=False):
        """
        运行 YOLO 推理。返回与 GreenOnBrown 相同的元组。

        在混合模式下，运行 YOLO 查找作物，将其掩膜，然后在
        剩余区域运行 GreenOnBrown ExHSV 以查找杂草。

        参数：
            image: BGR numpy 数组。
            confidence: 检测置信度阈值。
            show_display: 若为 True，返回标注图像副本。
            filter_id: 未使用，保留用于 API 兼容性。
            label: 显示时的回退标签。
            build_mask: 若为 True 且模型为分割模型，构建 self.detection_mask
                        用于基于区域的触发。为节省 CPU，False 时跳过。
            exg_min..invert_hue: GreenOnBrown 参数，仅在混合模式下使用。

        返回：
            (contours, boxes, weed_centres, image_out)
            - contours: 掩膜多边形 (分割) 或 None (检测)
            - boxes: [x, y, w, h] 列表
            - weed_centres: [cx, cy] 列表
            - image_out: 若 show_display 则为标注图像，否则为原始图像
        """
        if self.hybrid_mode:
            return self._hybrid_inference(
                image, confidence, show_display,
                exg_min=exg_min, exg_max=exg_max,
                hue_min=hue_min, hue_max=hue_max,
                saturation_min=saturation_min, saturation_max=saturation_max,
                brightness_min=brightness_min, brightness_max=brightness_max,
                min_detection_area=min_detection_area, invert_hue=invert_hue
            )

        # --- 纯 GoG 模式 ---
        self.detection_mask = None  # 每帧重置

        if self.tracking_enabled:
            # 跟踪所有类 —— owl.py 中的 ClassSmoother 进行类过滤
            results = self.model.track(
                source=image,
                conf=confidence,
                persist=True,
                tracker='config/bytetrack_owl.yaml',
                verbose=False,
                device=self.device
            )
        else:
            results = self.model.predict(
                source=image,
                conf=confidence,
                classes=self._detect_class_ids,
                verbose=False,
                device=self.device
            )

        boxes = []
        weed_centres = []
        contours = None
        track_ids = []
        raw_class_ids = []
        raw_confidences = []

        for result in results:
            has_ids = (self.tracking_enabled
                       and result.boxes.id is not None
                       and len(result.boxes.id) > 0)

            # 处理边界框 (适用于检测和分割)
            for i, box in enumerate(result.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                boxes.append([x1, y1, w, h])
                weed_centres.append([x1 + w // 2, y1 + h // 2])

                if has_ids:
                    track_ids.append(int(result.boxes.id[i]))
                    raw_class_ids.append(int(box.cls[0]))
                    raw_confidences.append(float(box.conf[0]))

            # 提取分割模型的掩膜轮廓
            if result.masks is not None:
                contours = [c.astype(np.int32).reshape(-1, 1, 2)
                            for c in result.masks.xy]

                # 仅在区域触发需要时构建组合二值掩膜
                if build_mask:
                    self.detection_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    cv2.drawContours(self.detection_mask, contours, -1, 255, -1)

        # 为 owl.py 的 ClassSmoother 存储原始跟踪数据
        self.last_track_ids = track_ids
        self.last_raw_boxes = list(boxes)
        self.last_class_ids = raw_class_ids
        self.last_confidences = raw_confidences

        if show_display:
            image_out = image.copy()

            # 若可用则绘制分割掩膜
            if contours is not None:
                overlay = image_out.copy()
                for contour in contours:
                    cv2.drawContours(overlay, [contour], -1, (0, 255, 0), -1)
                cv2.addWeighted(overlay, 0.3, image_out, 0.7, 0, image_out)

            # 绘制边界框 + 标签 (跟踪中为绿色，未跟踪为红色)
            for i, box_data in enumerate(boxes):
                x, y, w, h = box_data
                conf_val = float(result.boxes[i].conf[0]) if i < len(result.boxes) else confidence
                cls_id = int(result.boxes[i].cls[0]) if i < len(result.boxes) else 0
                cls_name = self.model.names.get(cls_id, label)

                has_track = (self.tracking_enabled and i < len(track_ids)
                             and track_ids[i] is not None)
                if has_track:
                    box_label = f'ID{track_ids[i]} {int(conf_val * 100)}% {cls_name}'
                    box_color = (0, 200, 0)   # green — tracked
                else:
                    box_label = f'{int(conf_val * 100)}% {cls_name}'
                    box_color = (0, 0, 255)   # red — untracked

                cv2.rectangle(image_out, (x, y), (x + w, y + h), box_color, 2)
                cv2.putText(image_out, box_label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            return contours, boxes, weed_centres, image_out

        return contours, boxes, weed_centres, image

    def _hybrid_inference(self, image, confidence, show_display,
                          exg_min=30, exg_max=250, hue_min=30, hue_max=90,
                          saturation_min=30, saturation_max=255,
                          brightness_min=5, brightness_max=200,
                          min_detection_area=1, invert_hue=False):
        """
        混合流水线：YOLO 作物掩膜 + ExHSV 杂草检测 (并行)。

        ExHSV 在后台线程中对完整 (未掩膜) 图像运行，同时
        YOLO 在主线程中运行。YOLO (NCNN) 和 ExHSV (OpenCV/NumPy)
        都释放 GIL，因此可在不同核心上实现真正的并行。

        步骤 1: 将完整图像的 ExHSV 提交到线程池
        步骤 2: 在主线程上运行 YOLO 预测 (瓶颈)
        步骤 3: 从 masks.xy 或 boxes.xyxy 构建全分辨率 crop_mask
        步骤 4: 按缓冲值膨胀 crop_mask
        步骤 5: 等待 ExHSV 结果
        步骤 6: 过滤掉中心落在作物掩膜内的任何检测
        步骤 7: 可视化 (若 show_display)
        """
        h_full, w_full = image.shape[:2]

        # 步骤 1: 将完整图像的 ExHSV 提交到后台线程
        exhsv_future = self._executor.submit(
            self._gob.inference,
            image,
            exg_min=exg_min, exg_max=exg_max,
            hue_min=hue_min, hue_max=hue_max,
            saturation_min=saturation_min, saturation_max=saturation_max,
            brightness_min=brightness_min, brightness_max=brightness_max,
            min_detection_area=min_detection_area,
            show_display=False,
            algorithm='exhsv',
            invert_hue=invert_hue
        )

        # 步骤 2: 在主线程上运行 YOLO 推理 (imgsz 处理缩放)
        if self.tracking_enabled:
            results = self.model.track(
                source=image,
                conf=confidence,
                classes=self._detect_class_ids,
                persist=True,
                tracker='config/bytetrack_owl.yaml',
                imgsz=self.inference_resolution,
                verbose=False,
                device=self.device
            )
        else:
            results = self.model.predict(
                source=image,
                conf=confidence,
                classes=self._detect_class_ids,
                imgsz=self.inference_resolution,
                verbose=False,
                device=self.device
            )

        # 步骤 3: 构建全分辨率作物掩膜
        if self.tracking_enabled and self._crop_stabilizer:
            # 将跟踪的作物检测送入稳定器以实现时间持久化
            crop_track_ids = []
            crop_boxes_xyxy = []
            crop_contours = []

            for result in results:
                has_ids = (result.boxes.id is not None
                           and len(result.boxes.id) > 0)

                if result.masks is not None:
                    crop_contours.extend(
                        c.astype(np.int32).reshape(-1, 1, 2)
                        for c in result.masks.xy)

                for i, box in enumerate(result.boxes):
                    if has_ids:
                        crop_track_ids.append(int(result.boxes.id[i]))
                        crop_boxes_xyxy.append(list(map(int, box.xyxy[0])))

            if crop_track_ids:
                self._crop_stabilizer.update(
                    crop_track_ids,
                    crop_boxes_xyxy,
                    contours=crop_contours if crop_contours else None
                )
                crop_mask = self._crop_stabilizer.build_stabilized_mask(
                    (h_full, w_full))
            else:
                # 无可用跟踪 ID —— 回退到逐帧掩膜
                crop_mask = self._build_crop_mask(results, h_full, w_full)

            # 将卡尔曼预测的丢失作物轨迹绘制到掩膜中
            # ByteTrack 预测丢失作物的移动位置 —— 填充掩膜空洞
            persist = self.detection_persist_frames
            lost_crops = self.get_lost_tracks(
                max_age=persist if persist > 0
                else (self._crop_stabilizer.max_age if self._crop_stabilizer else 3))
            for lc in lost_crops:
                x1, y1, x2, y2 = [max(0, int(v)) for v in lc['xyxy']]
                x2, y2 = min(w_full, x2), min(h_full, y2)
                if x2 > x1 and y2 > y1:
                    crop_mask[y1:y2, x1:x2] = 255
        else:
            crop_mask = self._build_crop_mask(results, h_full, w_full)

        # 步骤 4: 按缓冲值膨胀作物掩膜
        crop_mask_undilated = crop_mask.copy() if show_display else None
        if self._dilate_kernel is not None and np.any(crop_mask):
            crop_mask = cv2.dilate(crop_mask, self._dilate_kernel)

        # 步骤 5: 等待 ExHSV 结果
        cnts, boxes, weed_centres, _ = exhsv_future.result()

        # 步骤 6: 安全过滤器 —— 丢弃中心落在作物掩膜内的检测
        from utils.greenonbrown import MAX_DETECTIONS
        filtered_boxes = []
        filtered_centres = []
        for i, centre in enumerate(weed_centres):
            cx, cy = centre
            if 0 <= cy < h_full and 0 <= cx < w_full:
                if crop_mask[cy, cx] == 0:  # Not in crop zone
                    filtered_boxes.append(boxes[i])
                    filtered_centres.append(centre)

        # 安全过滤器后限制数量以控制下游处理量
        filtered_boxes = filtered_boxes[:MAX_DETECTIONS]
        filtered_centres = filtered_centres[:MAX_DETECTIONS]

        # 步骤 7: 可视化
        if show_display:
            image_out = image.copy()

            # 作物掩膜上的蓝色叠加
            crop_overlay = image_out.copy()
            crop_overlay[crop_mask_undilated > 0] = (200, 150, 50)
            cv2.addWeighted(crop_overlay, 0.5, image_out, 0.5, 0, image_out)

            # 缓冲区上的浅蓝色 (膨胀后 - 原始)
            if self._dilate_kernel is not None:
                buffer_zone = cv2.subtract(crop_mask, crop_mask_undilated)
                if np.any(buffer_zone):
                    buffer_overlay = image_out.copy()
                    buffer_overlay[buffer_zone > 0] = (200, 180, 100)
                    cv2.addWeighted(buffer_overlay, 0.35, image_out, 0.65, 0, image_out)

            # 杂草检测上的红色框
            for box_data in filtered_boxes:
                x, y, w, h = box_data
                cv2.rectangle(image_out, (x, y), (x + w, y + h), (0, 0, 255), 2)

            return cnts, filtered_boxes, filtered_centres, image_out

        return cnts, filtered_boxes, filtered_centres, image
