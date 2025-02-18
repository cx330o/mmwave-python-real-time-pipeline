# exp1_sit_stand_two_cycles.py
import winsound
import time
import datetime

# 试验 1 用到的频率
E1_PREP_FREQ   = 600   # 静止提示：站好 / 坐好，保持不动
E1_ACTION_FREQ = 800   # 动作提示：开始坐下 / 开始站起
FINISH_FREQ    = 440   # 实验结束长音

def beep(freq, duration_ms, label=""):
    """简单的 beep 封装：打一声，并打印时间戳 & 说明。"""
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] BEEP {freq} Hz {duration_ms} ms  {label}")
    winsound.Beep(int(freq), int(duration_ms))

def experiment_1_sit_stand_two_cycles():
    """
    试验 1：椅子在雷达前 1.5 m，始终面向雷达。
    做 2 个来回：站 -> 坐 -> 坐着不动 -> 站起。
    时间安排见函数内部注释。
    """
    print("=== 实验 1：站立 ↔ 坐下（2 个来回），始终面向雷达 ===")
    print("请先站在椅子前 1.5 m 左右、面对雷达，准备开始。")

    # ------- 第 1 个来回 -------
    # t = 0 s：站立静止开始，持续 0–4 s
    beep(E1_PREP_FREQ, 500, "E1: 站立静止 (0-4s)")
    time.sleep(4.0)

    # t = 4 s：开始从站立 → 坐下（4–7 s 完成）
    beep(E1_ACTION_FREQ, 400, "E1: 动作 站->坐 (4-7s)")
    time.sleep(3.0)

    # t = 7 s：坐姿静止（7–11 s）
    beep(E1_PREP_FREQ, 500, "E1: 坐姿静止 (7-11s)")
    time.sleep(4.0)

    # t = 11 s：从坐下 → 站起（11–14 s）
    beep(E1_ACTION_FREQ, 400, "E1: 动作 坐->站 (11-14s)")
    time.sleep(3.0)

    # ------- 第 2 个来回 -------
    # t = 14 s：站立静止（14–18 s）
    beep(E1_PREP_FREQ, 500, "E1: 站立静止 (14-18s)")
    time.sleep(4.0)

    # t = 18 s：站 -> 坐（18–21 s）
    beep(E1_ACTION_FREQ, 400, "E1: 动作 站->坐 (18-21s)")
    time.sleep(3.0)

    # t = 21 s：坐姿静止（21–25 s）
    beep(E1_PREP_FREQ, 500, "E1: 坐姿静止 (21-25s)")
    time.sleep(4.0)

    # t = 25 s：坐 -> 站（25–28 s）
    beep(E1_ACTION_FREQ, 400, "E1: 动作 坐->站 (25-28s)")
    time.sleep(3.0)

    # t = 28 s：实验结束长音
    beep(FINISH_FREQ, 2000, "E1: 实验 1 结束 (28s)")
    print("=== 实验 1 完成，请保持站立不动再停采集 ===")


if __name__ == "__main__":
    experiment_1_sit_stand_two_cycles()
