import sys
import os
import time
import random

# 添加项目根目录到 sys.path 以便导入 utils 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.vis_manager import RelayVis

# 尝试导入 Jetson.GPIO，如果在非 Jetson 设备上运行，则使用 Mock 类代替以免报错
try:
    import Jetson.GPIO as GPIO
    HAS_GPIO = True
except ImportError:
    print("[WARNING] 未检测到 Jetson.GPIO 库。正在使用虚拟 GPIO 进行终端可视化测试。")
    HAS_GPIO = False
    class MockGPIO:
        BOARD = "BOARD"
        OUT = "OUT"
        HIGH = 1
        LOW = 0
        @staticmethod
        def setmode(mode): pass
        @staticmethod
        def setwarnings(flag): pass
        @staticmethod
        def setup(pin, mode): pass
        @staticmethod
        def output(pin, state): pass
        @staticmethod
        def cleanup(): pass
    GPIO = MockGPIO

def main():
    print("=== Jetson GPIO 继电器状态终端可视化测试 ===\n")
    
    # 选用的 Jetson GPIO 引脚 (BOARD 编号)
    PINS = [29, 31, 32, 33]
    NUM_RELAYS = len(PINS)
    
    # 1. 初始化 GPIO
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)
    
    for pin in PINS:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)
        
    print(f"[INFO] 已初始化 GPIO 引脚 (BOARD): {PINS}")
    print("[INFO] 每隔 1 秒将随机切换引脚状态。按 Ctrl+C 退出。\n")
    
    # 2. 初始化终端可视化工具
    vis = RelayVis(relays=NUM_RELAYS)
    vis.setup()
    
    try:
        while True:
            # 周期为 1 秒
            time.sleep(1.0)
            
            # 遍历每个喷嘴/引脚，随机生成开/关状态
            for i in range(NUM_RELAYS):
                pin = PINS[i]
                # 随机生成 True (开) 或 False (关)
                is_active = random.choice([True, False])
                
                # 3. 设置实际硬件 GPIO 高低电平
                if is_active:
                    GPIO.output(pin, GPIO.HIGH)
                else:
                    GPIO.output(pin, GPIO.LOW)
                
                # 4. 同步更新终端可视化（终端中的色块会对应变绿或变灰）
                vis.update(relay=i, status=is_active)
                
    except KeyboardInterrupt:
        pass  # 捕捉 Ctrl+C
    finally:
        # 清理终端换行
        vis.close()
        # 清理 GPIO 状态
        GPIO.cleanup()
        print("\n[INFO] 测试结束，GPIO 引脚已重置。")

if __name__ == "__main__":
    main()