import threading
import time
import logging
import platform
import heapq
import uuid

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

        # 调度器（集中式优先队列）
        self.scheduler = SprayScheduler(self)
        self.scheduler.start()

        # 事件日志（用于记录实际触发时间与计划时间）
        self.event_log = []

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
        now = time.time()
        on_time = now + max(0.0, delay_s)
        off_time = on_time + max(0.0, duration_s)

        # 记录计划并提交给调度器
        event_id = self.scheduler.schedule(relay_id=relay_id, on_time=on_time, off_time=off_time)
        self.logger.debug(f"Scheduled spray relay={relay_id} on={on_time:.3f} off={off_time:.3f} id={event_id}")
        return event_id

    def all_off(self):
        """强制关闭所有继电器并清空引用计数"""
        with self.lock:
            for relay_id, relay in self.relays.items():
                self.ref_counts[relay_id] = 0
                relay.off()
                if self.on_state_change:
                    self.on_state_change(relay_id, False)
            # 取消并清理调度器中的所有事件
            try:
                self.scheduler.cancel_all()
            except Exception:
                pass

    def cleanup(self):
        self.all_off()
        for relay in self.relays.values():
            relay.cleanup()
        try:
            self.scheduler.stop()
        except Exception:
            pass
        self.logger.info("RelayController cleaned up.")


class SprayScheduler:
    """集中式优先队列调度器，管理喷洒 on/off 事件并记录实际触发时间。"""
    def __init__(self, controller: 'RelayController'):
        self.controller = controller
        self.pq = []  # heap of (time, seq, action, relay_id, event_id)
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.seq = 0
        self.running = False
        self.thread = None
        self.cancelled = set()

    def start(self):
        with self.lock:
            if self.running:
                return
            self.running = True
            self.thread = threading.Thread(target=self._run, name='SprayScheduler')
            self.thread.daemon = True
            self.thread.start()

    def stop(self):
        with self.lock:
            self.running = False
            self.cv.notify_all()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def schedule(self, relay_id: int, on_time: float, off_time: float):
        event_id = uuid.uuid4().hex
        with self.lock:
            seq_on = self.seq; self.seq += 1
            heapq.heappush(self.pq, (on_time, seq_on, 'on', relay_id, event_id))
            seq_off = self.seq; self.seq += 1
            heapq.heappush(self.pq, (off_time, seq_off, 'off', relay_id, event_id))
            self.cv.notify()
        return event_id

    def cancel_all(self):
        with self.lock:
            self.cancelled.update({e[4] for e in self.pq})
            self.pq.clear()
            self.cv.notify_all()

    def _run(self):
        while True:
            with self.lock:
                if not self.running and not self.pq:
                    break
                if not self.pq:
                    self.cv.wait(timeout=1.0)
                    continue
                next_time, _, action, relay_id, event_id = self.pq[0]
                now = time.time()
                wait = next_time - now
                if wait > 0:
                    self.cv.wait(timeout=wait)
                    continue
                # pop and execute
                heapq.heappop(self.pq)
            # outside lock: execute action
            if event_id in self.cancelled:
                continue
            planned = next_time
            actual = time.time()
            try:
                if action == 'on':
                    self.controller._turn_on(relay_id)
                    self.controller.logger.info(f"[SprayScheduler] ON relay={relay_id} planned={planned:.3f} actual={actual:.3f} delay={actual-planned:.3f}s")
                    self.controller.event_log.append({'relay': relay_id, 'action': 'on', 'planned': planned, 'actual': actual})
                else:
                    self.controller._turn_off(relay_id)
                    self.controller.logger.info(f"[SprayScheduler] OFF relay={relay_id} planned={planned:.3f} actual={actual:.3f} delay={actual-planned:.3f}s")
                    self.controller.event_log.append({'relay': relay_id, 'action': 'off', 'planned': planned, 'actual': actual})
            except Exception as e:
                self.controller.logger.exception(f"Error executing {action} for relay {relay_id}: {e}")