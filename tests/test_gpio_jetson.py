import argparse
import os
import sys
import time


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.output_manager import HARDWARE_PLATFORM, TESTING, RelayController
from utils.vis_manager import RelayVis


RELAYS = {
    0: 29,
    1: 31,
    2: 32,
    3: 33,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Jetson GPIO relay test for pins 29/31/32/33")
    parser.add_argument("--cycles", type=int, default=30, help="完整循环次数，默认 30")
    parser.add_argument("--on-time", type=float, default=1.0, help="每个通道点亮时长(秒)，默认 1.0")
    parser.add_argument("--off-time", type=float, default=0.5, help="通道之间关闭等待时间(秒)，默认 0.5")
    parser.add_argument("--final-wait", type=float, default=1.0, help="结束后等待所有继电器回落的时间(秒)")
    parser.add_argument("--step", choices=["sequential", "all"], default="sequential", help="按通道轮流测试或四路同时测试")
    return parser.parse_args()


def run_sequential(controller: RelayController, vis: RelayVis, on_time: float, off_time: float, cycles: int):
    relay_ids = list(RELAYS.keys())
    for cycle in range(cycles):
        print(f"\n[Cycle {cycle + 1}/{cycles}] sequential test start")
        for relay_id in relay_ids:
            pin = RELAYS[relay_id]
            print(f"[ON ] relay={relay_id} pin={pin}")
            controller.schedule_spray(relay_id=relay_id, delay_s=0.0, duration_s=on_time)
            time.sleep(on_time + off_time)


def run_all(controller: RelayController, vis: RelayVis, on_time: float, off_time: float, cycles: int):
    relay_ids = list(RELAYS.keys())
    for cycle in range(cycles):
        print(f"\n[Cycle {cycle + 1}/{cycles}] all-relay test start")
        for relay_id in relay_ids:
            pin = RELAYS[relay_id]
            print(f"[ON ] relay={relay_id} pin={pin}")
            controller.schedule_spray(relay_id=relay_id, delay_s=0.0, duration_s=on_time)
        time.sleep(on_time + off_time)


def main():
    args = parse_args()

    print("=== Jetson GPIO 实际控制测试 ===")
    print(f"[System] platform={HARDWARE_PLATFORM}, testing_mode={TESTING}")
    print(f"[System] relays={RELAYS}")

    if TESTING:
        print("[WARN] 当前不是 Jetson 环境，`utils/output_manager.py` 会进入测试模式，不会真正驱动硬件 GPIO。")

    vis = RelayVis(relays=len(RELAYS))
    controller = RelayController(RELAYS, on_state_change=vis.update)
    vis.setup()

    try:
        if args.step == "sequential":
            run_sequential(controller, vis, args.on_time, args.off_time, args.cycles)
        else:
            run_all(controller, vis, args.on_time, args.off_time, args.cycles)

        print("\n[INFO] 触发结束，等待所有继电器回落...")
        time.sleep(args.final_wait)
        print(f"[INFO] event_log entries = {len(controller.event_log)}")
        for event in controller.event_log[-8:]:
            print(f"[LOG] {event}")

    except KeyboardInterrupt:
        print("\n[INFO] 用户中断，正在关闭所有继电器...")
    finally:
        controller.cleanup()
        vis.close()
        print("[INFO] GPIO 测试结束，资源已清理。")


if __name__ == "__main__":
    main()