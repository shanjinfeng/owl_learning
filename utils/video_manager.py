import cv2
import time
import logging
from threading import Thread, Event

# 移除对其他文件 (log_manager, error_manager) 的依赖
# 自定义异常类直接在当前文件内实现
class CameraNotFoundError(Exception):
    def __init__(self, error_type, original_error, camera_type):
        self.error_type = error_type
        self.original_error = original_error
        self.camera_type = camera_type
        super().__init__(f"[{camera_type.upper()}] Camera Error ({error_type}): {original_error}")

# 仅保留大恒 Galaxy / Mercury GigE 相机 SDK
try:
    import gxipy as gx
except Exception:
    gx = None


class DahengGigEStream:
    """大恒 Galaxy / Mercury GigE 相机的线程化帧读取器。"""

    def __init__(self, src=0, resolution=(2048, 1536), **kwargs):
        # 使用 Python 标准 logging 模块
        self.logger = logging.getLogger(__name__)
        self.name = 'DahengGigEStream'
        self.logger.info(f'相机类型: {self.name}')

        self.source_order = str(kwargs.pop('source_order', 'bgr')).strip().lower()
        if self.source_order not in {'rgb', 'bgr'}:
            self.logger.warning(f"Unsupported source_order='{self.source_order}', fallback to 'bgr'.")
            self.source_order = 'bgr'

        if gx is None:
            raise CameraNotFoundError(
                error_type='missing_sdk',
                original_error='gxipy is not installed. 请安装大恒 Galaxy SDK。',
                camera_type='gige'
            )

        self.device_manager = gx.DeviceManager()
        self.device_num, self.device_info_list = self.device_manager.update_device_list()

        if self.device_num <= 0:
            raise CameraNotFoundError(
                error_type='not_found',
                original_error='未在网络中检测到大恒 GigE 相机。',
                camera_type='gige'
            )

        self.device_index = max(1, src + 1 if src == 0 else src)

        if self.device_index > self.device_num:
            raise CameraNotFoundError(
                error_type='index_out_of_range',
                original_error=f'请求设备索引 {self.device_index}, 但只找到 {self.device_num} 个相机。',
                camera_type='gige'
            )

        self.logger.info(f'检测到 {self.device_num} 个大恒相机。正在打开设备索引 {self.device_index}。')

        try:
            self.camera = self.device_manager.open_device_by_index(self.device_index)
        except Exception as e:
            raise CameraNotFoundError(error_type='open_failed', original_error=str(e), camera_type='gige') from e

        # 应用可选参数
        for key, value in kwargs.items():
            try:
                setattr(self.camera, key, value)
            except Exception:
                self.logger.debug(f'忽略不支持的大恒属性: {key}={value}')

        try:
            self.camera.stream_on()
        except Exception as e:
            raise CameraNotFoundError(error_type='stream_on_failed', original_error=str(e), camera_type='gige') from e

        self.frame_width = resolution[0]
        self.frame_height = resolution[1]
        self.frame = None

        # 预热并获取第一帧
        warmup_deadline = time.time() + 2.0
        while self.frame is None and time.time() < warmup_deadline:
            raw_image = self._read_raw_frame()
            if raw_image is None:
                time.sleep(0.01)
                continue
            self.frame = self._to_bgr(raw_image)

        if self.frame is None:
            raise CameraNotFoundError(
                error_type='no_frame',
                original_error='数据流已启动，但在预热期间未收到图像。',
                camera_type='gige'
            )

        self.stop_event = Event()
        self.thread = Thread(target=self.update, name=self.name, args=())
        self.thread.daemon = True

    def start(self):
        self.thread.start()
        return self

    def _read_raw_frame(self):
        if hasattr(self.camera, 'data_stream') and self.camera.data_stream:
            return self.camera.data_stream[0].get_image()
        return None

    def _to_bgr(self, raw_image):
        try:
            frame = raw_image.convert('RGB').get_numpy_array()
            if frame is None:
                return None
            if self.source_order == 'rgb':
                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame
        except Exception:
            frame = raw_image.get_numpy_array()
            if frame is None:
                return None
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                if self.source_order == 'rgb':
                    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return frame
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    def update(self):
        try:
            while not self.stop_event.is_set():
                raw_image = self._read_raw_frame()
                if raw_image is None:
                    time.sleep(0.01)
                    continue

                frame = self._to_bgr(raw_image)
                if frame is None:
                    continue

                if (frame.shape[1], frame.shape[0]) != (self.frame_width, self.frame_height):
                    frame = cv2.resize(frame, (self.frame_width, self.frame_height))

                self.frame = frame
                time.sleep(0.001)
        except Exception as e:
            self.logger.error(f'更新循环异常: {e}', exc_info=True)
        finally:
            self._cleanup()

    def read(self):
        return self.frame

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self._cleanup()

    def _cleanup(self):
        try:
            if hasattr(self.camera, 'stream_off'):
                self.camera.stream_off()
        except Exception:
            pass
        try:
            if hasattr(self.camera, 'close_device'):
                self.camera.close_device()
        except Exception:
            pass


class VideoStream:
    """统一的大恒 GigE 视频流包装器"""
    def __init__(self, src=0, resolution=(2048, 1536), **kwargs):
        self.logger = logging.getLogger(__name__)
        self.stream = DahengGigEStream(src=src, resolution=resolution, **kwargs)
        self.frame_width = self.stream.frame_width
        self.frame_height = self.stream.frame_height

    def start(self):
        return self.stream.start()

    def update(self):
        self.stream.update()

    def read(self):
        return self.stream.read()

    def stop(self):
        self.stream.stop()