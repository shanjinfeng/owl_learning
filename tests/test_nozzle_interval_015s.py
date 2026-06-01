# 测试喷嘴间隔为 0.15s 的情况，验证继电器的响应时间和稳定性。
import configparser
import os
import sys
import time
from collections import defaultdict


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.output_manager import HARDWARE_PLATFORM, TESTING, RelayController
from utils.vis_manager import RelayVis


CONFIG_PATH = "config/aruco_weeding.ini"
CYCLES = 40
ON_TIME_S = 0.05
OFF_TIME_S = 1
FINAL_WAIT_S = 1.0
DRY_RUN = False


def load_relays(config_path: str) -> dict:
    config = configparser.ConfigParser()
    if not config.read(config_path):
        raise FileNotFoundError(f"未能读取配置文件: {config_path}")
    if "Relays" not in config:
        raise KeyError(f"配置文件缺少 [Relays] 节: {config_path}")

    relay_dict = {}
    for key, value in config["Relays"].items():
        relay_dict[int(key)] = int(value)

    if not relay_dict:
        raise ValueError(f"[Relays] 为空: {config_path}")

    return dict(sorted(relay_dict.items(), key=lambda kv: kv[0]))


def summarize_event_log(event_log):
    """统计每个喷嘴的实际开阀时长与触发偏差。"""
    pending_on = {}
    durations = defaultdict(list)
    on_delay = defaultdict(list)
    off_delay = defaultdict(list)

    for event in event_log:
        relay = event["relay"]
        action = event["action"]
        planned = float(event["planned"])
        actual = float(event["actual"])

        if action == "on":
            pending_on[relay] = actual
            on_delay[relay].append(actual - planned)
        elif action == "off":
            off_delay[relay].append(actual - planned)
            if relay in pending_on:
                durations[relay].append(actual - pending_on.pop(relay))

    return durations, on_delay, off_delay


def print_summary(relay_ids, durations, on_delay, off_delay):
    # 打印每个继电器的统计结果
    print("\n=== 触发统计（实际） ===")
    for relay_id in relay_ids:
        d_list = durations.get(relay_id, [])
        on_list = on_delay.get(relay_id, [])
        off_list = off_delay.get(relay_id, [])

        if not d_list:
            print(f"Relay {relay_id}: 无有效 ON/OFF 成对事件")
            continue

        d_avg = sum(d_list) / len(d_list)
        d_min = min(d_list)
        d_max = max(d_list)
        on_avg = sum(on_list) / len(on_list) if on_list else 0.0
        off_avg = sum(off_list) / len(off_list) if off_list else 0.0

        print(
            f"Relay {relay_id}: "
            f"samples={len(d_list)}, "
            f"open_avg={d_avg*1000:.1f}ms, open_min={d_min*1000:.1f}ms, open_max={d_max*1000:.1f}ms, "
            f"on_delay_avg={on_avg*1000:.1f}ms, off_delay_avg={off_avg*1000:.1f}ms"
        )


def run_sequential(controller, relay_ids, pin_map, cycles, on_time, off_time, dry_run):
    # 每次循环依次触发每个喷嘴，保持 on_time 和 off_time 的间隔
    for cycle in range(cycles):
        print(f"\n[Cycle {cycle + 1}/{cycles}] sequential")
        for relay_id in relay_ids:
            pin = pin_map[relay_id]
            print(f"[ON ] relay={relay_id} pin={pin} on={on_time:.3f}s off={off_time:.3f}s")
            if not dry_run:
                controller.schedule_spray(relay_id=relay_id, delay_s=0.0, duration_s=on_time)
            time.sleep(on_time + off_time)


def main():
    relay_dict = load_relays(CONFIG_PATH)
    relay_ids = list(relay_dict.keys())

    print("=== Nozzle Interval Test ===")
    print(f"[System] platform={HARDWARE_PLATFORM}, testing_mode={TESTING}, dry_run={DRY_RUN}")
    print(f"[System] config={CONFIG_PATH}")
    print(f"[System] relays={relay_dict}")
    print(f"[System] cycles={CYCLES}, on={ON_TIME_S:.3f}s, off={OFF_TIME_S:.3f}s")

    if TESTING and not DRY_RUN:
        print("[WARN] 当前为测试模式，不会真正驱动 Jetson GPIO。")

    if ON_TIME_S <= 0 or OFF_TIME_S < 0:
        raise ValueError("on-time 必须 > 0，off-time 必须 >= 0")

    vis = RelayVis(relays=len(relay_ids))
    controller = RelayController(relay_dict, on_state_change=vis.update)
    vis.setup()

    try:
        run_sequential(controller, relay_ids, relay_dict, CYCLES, ON_TIME_S, OFF_TIME_S, DRY_RUN)

        print("\n[INFO] 触发结束，等待所有继电器回落...")
        time.sleep(FINAL_WAIT_S)

        if not DRY_RUN:
            durations, on_delay, off_delay = summarize_event_log(controller.event_log)
            print_summary(relay_ids, durations, on_delay, off_delay)
            print(f"[INFO] 总事件数: {len(controller.event_log)}")

    except KeyboardInterrupt:
        print("\n[INFO] 用户中断，正在清理...")
    finally:
        controller.cleanup()
        vis.close()
        print("[INFO] 测试结束，资源已清理。")


if __name__ == "__main__":
    main()
