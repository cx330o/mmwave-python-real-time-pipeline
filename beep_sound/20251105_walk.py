# exp2_walk_two_cycles.py
import winsound
import time
import datetime

E2_MOVE_FREQ = 900   # 走动 / 掉头提示
FINISH_FREQ  = 440   # 实验结束长音

def beep(freq, duration_ms, label=""):
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] BEEP {freq} Hz {duration_ms} ms  {label}")
    winsound.Beep(int(freq), int(duration_ms))

def experiment_2_walk_two_cycles(leg_time_s=3.0):
    """
    试验 2：在 0m (雷达附近) 与 1.5m (椅子位置) 之间来回走 2 个来回，一共 4 段。
    每段默认 leg_time_s 秒（默认 3s）。
    始终面向雷达（推荐：从雷达向椅子“倒退走”，再向雷达“前走”回来）。
    """
    print("=== 实验 2：0–1.5m 之间来回走（2 个来回），始终面向雷达 ===")
    print("请先站在雷达附近 (0 m) 面对雷达，准备开始走。")

    # 段 1：雷达 -> 椅子
    beep(E2_MOVE_FREQ, 400, "E2: 段1 雷达 -> 椅子 (0-{:.1f}s)".format(leg_time_s))
    time.sleep(leg_time_s)

    # 段 2：椅子 -> 雷达
    beep(E2_MOVE_FREQ, 400, "E2: 段2 椅子 -> 雷达")
    time.sleep(leg_time_s)

    # 段 3：雷达 -> 椅子
    beep(E2_MOVE_FREQ, 400, "E2: 段3 雷达 -> 椅子")
    time.sleep(leg_time_s)

    # 段 4：椅子 -> 雷达
    beep(E2_MOVE_FREQ, 400, "E2: 段4 椅子 -> 雷达")
    time.sleep(leg_time_s)

    # 结束
    beep(FINISH_FREQ, 2000, "E2: 实验 2 结束 (~{:.1f}s)".format(4*leg_time_s))
    print("=== 实验 2 完成，请保持原地不动再停采集 ===")


if __name__ == "__main__":
    experiment_2_walk_two_cycles()
