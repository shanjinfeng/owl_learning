import threading
import time
import logging
import platform

logger = logging.getLogger(__name__)

def _detect_hardware_platform() -> str:
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().lower()
        if 'jetson' in model or 'nvidia' in model:
            return 'jetson'
    except (FileNotFoundError, OSError):
        pass
    return 'other'

HARDWARE_PLATFORM = _detect_hardware_platform()
TESTING = HARDWARE_PLATFORM != 'jetson'

if not TESTING:
    try:
        import Jetson.GPIO as JGPIO
        JGPIO.setwarnings(False)
        JGPIO.setmode(JGPIO.BOARD)
    except Exception as e:
        logger.warning(f"Jetson.GPIO 初始化失败: {e}。回退到测试模式。")
        TESTING = True

def _parse_board_pin(pin) -> int:
    if isinstance(pin, int):
        return pin
    pin_str = str(pin).strip().upper()
    if pin_str.startswith('BOARD'):
        pin_str = pin_str[5:]
    return int(pin_str)

class OutputDevice:
    """底层硬件 GPIO 输出设备"""
    def __init__(self, pin):
        self.pin = _parse_board_pin(pin)
        self.testing = TESTING
        
        if not self.testing:
            JGPIO.setup(self.pin, JGPIO.OUT)
            JGPIO.output(self.pin, JGPIO.LOW)
        else:
            logger.info(f"[TEST] Setup GPIO OUT on pin {self.pin}")

    def on(self):
        if not self.testing:
            JGPIO.output(self.pin, JGPIO.HIGH)

    def off(self):
        if not self.testing:
            JGPIO.output(self.pin, JGPIO.LOW)

    def cleanup(self):
        if not self.testing:
            try:
                JGPIO.cleanup(self.pin)
            except Exception:
                pass


class RelayController:
    """
    核心喷洒控制器：
    引入引用计数 (Reference Counting)，完美支持多目标重叠在同一喷嘴时的状态管理。
    支持 on_state_change 回调以联动 UI 可视化。
    """
    def __init__(self, relay_dict, on_state_change=None):
        """
        :param relay_dict: dict 映射, 例如 {0: 'BOARD11', 1: 'BOARD13', ...}
        :param on_state_change: 回调函数 callback(relay_id, status: bool)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.relays = {}
        self.ref_counts = {}
        self.lock = threading.Lock()
        self.on_state_change = on_state_change

        for relay_id, pin in relay_dict.items():
            self.relays[relay_id] = OutputDevice(pin)
            self.ref_counts[relay_id] = 0

        self.logger.info(f"RelayController initialized with {len(relay_dict)} relays.")

    def _turn_on(self, relay_id):
        """内部方法：引用计数开阀"""
        with self.lock:
            count = self.ref_counts.get(relay_id, 0)
            if count == 0:
                self.relays[relay_id].on()
                if self.on_state_change:
                    self.on_state_change(relay_id, True)
            self.ref_counts[relay_id] = count + 1

    def _turn_off(self, relay_id):
        """内部方法：引用计数关阀"""
        with self.lock:
            count = self.ref_counts.get(relay_id, 0)
            if count <= 1:
                self.ref_counts[relay_id] = 0
                self.relays[relay_id].off()
                if self.on_state_change:
                    self.on_state_change(relay_id, False)
            else:
                self.ref_counts[relay_id] = count - 1

    def schedule_spray(self, relay_id, delay_s, duration_s):
        """非阻塞定时触发"""
        if relay_id not in self.relays:
            self.logger.error(f"Invalid relay_id: {relay_id}")
            return

        def off_cb():
            self._turn_off(relay_id)

        def on_cb():
            self._turn_on(relay_id)
            t_off = threading.Timer(duration_s, off_cb)
            t_off.daemon = True
            t_off.start()

        if delay_s <= 0:
            on_cb()
        else:
            t_on = threading.Timer(delay_s, on_cb)
            t_on.daemon = True
            t_on.start()

    def all_off(self):
        """强制关闭所有继电器并清空引用计数"""
        with self.lock:
            for relay_id, relay in self.relays.items():
                self.ref_counts[relay_id] = 0
                relay.off()
                if self.on_state_change:
                    self.on_state_change(relay_id, False)

    def cleanup(self):
        self.all_off()
        for relay in self.relays.values():
            relay.cleanup()
        self.logger.info("RelayController cleaned up.")