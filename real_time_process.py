# C:\github\cx330_radar\real_time_process.py

import threading as th
import numpy as np
import DSP
from dsp.utils import Window
from ctypes import *
import os
import queue

# ------- DLL 路径（按当前文件所在目录拼绝对路径）-------
_here = os.path.dirname(os.path.abspath(__file__))
dll_path = os.path.join(_here, 'libs', 'UDPCAPTUREADCRAWDATA.dll')

if not os.path.isfile(dll_path):
    raise FileNotFoundError(
        f"Cannot find DLL at: {dll_path}\n"
        "请确认仓库里的 libs/UDPCAPTUREADCRAWDATA.dll 存在；"
        "若在其他位置，请移动到 libs/ 或修改此文件中的 dll_path。"
    )

dll = cdll.LoadLibrary(dll_path)

# ------- 采集双缓冲区 -------
# HALF_BUFFER_SHORTS 要和 main.py 里 frame_length 的计算一致：
# frame_length = NUM_ADC_SAMPLES * NUM_CHIRPS * NUM_TX * NUM_RX * 2
HALF_BUFFER_SHORTS = 98304  # 每个“半缓冲”的 short 数

# a: 当前写入的半区标志 0/1
a = np.zeros(1, dtype=np.int32)
# b: 实际数据，包含两个半缓冲
b = np.zeros(HALF_BUFFER_SHORTS * 2, dtype=np.int16)

a_ctypes_ptr = cast(a.ctypes.data, POINTER(c_int))
b_ctypes_ptr = cast(b.ctypes.data, POINTER(c_short))


class UdpListener(th.Thread):
    """
    负责与 DLL 交互，把 LVDS RAW 写入到我们提供的双缓冲 a/b。
    """
    def __init__(self, name, bin_data_queue: "queue.Queue", data_frame_length: int):
        super().__init__(name=name, daemon=True)
        self.bin_queue = bin_data_queue  # 目前没用，但保留接口
        self.frame_length = data_frame_length

    def run(self):
        # DLL 内部会不停往 b 里写数据，同时在 a_ctypes_ptr[0] 上切换 0/1
        dll.captureudp(a_ctypes_ptr, b_ctypes_ptr, self.frame_length)


class DataProcessor(th.Thread):
    """
    读取 DLL 维护的双缓冲，按“另一半缓冲”的当前快照，做：
      - I/Q 重构 → [chirp*tx, samples, rx]
      - 拆 TX → 拼虚拟阵列
      - DSP：RDA_Time（RTI/RDI/DTI） + Range_Angle（RAI/REI）
      - 推入上层队列

    同时（可选）把“该半缓冲的原始 short 流”写入 raw_recorder。
    """
    def __init__(self,
                 name,
                 config,                    # [adc_sample, chirp_num, tx_num, rx_num]
                 bin_queue,
                 rti_queue, dti_queue,
                 rdi_queue, rai_queue, rei_queue,
                 raw_recorder=None):
        super().__init__(name=name, daemon=True)

        self.adc_sample = int(config[0])
        self.chirp_num  = int(config[1])
        self.tx_num     = int(config[2])
        self.rx_num     = int(config[3])

        self.bin_queue = bin_queue
        self.rti_queue = rti_queue
        self.dti_queue = dti_queue
        self.rdi_queue = rdi_queue
        self.rai_queue = rai_queue
        self.rei_queue = rei_queue

        # 录 raw.bin 用的句柄（由 main.py 里的 RecordingSession 传进来）
        self.raw_recorder = raw_recorder
        self._stop = False

        # 一半缓冲的 short 数，要和 main.py 里 frame_length 保持一致
        self._frame_shorts = (
            self.adc_sample * self.chirp_num *
            self.tx_num * self.rx_num * 2
        )

    def stop(self):
        self._stop = True

    def _write_raw_if_needed(self, short_array_view: np.ndarray):
        """
        把该半缓冲 short[] 写入 RawRecorder。
        short_array_view: np.int16 一维
        """
        if self.raw_recorder is None:
            return
        try:
            self.raw_recorder.append_halfbuffer(
                short_array_view.astype(np.int16, copy=False)
            )
        except Exception as e:
            print("raw_recorder write error:", e)

    def run(self):
        global a_ctypes_ptr, b_ctypes_ptr
        lastflar = 0

        while not self._stop:
            # DLL 在 a_ctypes_ptr[0] 里维护“当前写入的半区”标志（0 或 1）
            if lastflar != a_ctypes_ptr[0]:
                lastflar = a_ctypes_ptr[0]

                # 当前写的是 a==0 时，稳定的就是后半区；反之亦然
                start = self._frame_shorts * (1 - a_ctypes_ptr[0])
                end   = self._frame_shorts * (2 - a_ctypes_ptr[0])

                # 从 b 对应区间构造一个 np.int16 视图（不拷贝）
                raw_view = np.frombuffer(
                    (c_short * (end - start)).from_address(
                        addressof(b_ctypes_ptr.contents) +
                        start * sizeof(c_short)
                    ),
                    dtype=np.int16
                )

                # 录制原始 short（如果 main.py 传进了 raw_recorder）
                self._write_raw_if_needed(raw_view)

                # ============ I/Q 重构 ============
                # 每 4 个 short: I0, Q0, I1, Q1
                data = np.reshape(raw_view, [-1, 4])
                data = data[:, 0:2:] + 1j * data[:, 2::]  # complex64

                # 形状调整为 [chirps*tx, rx, samples]
                data = np.reshape(
                    data,
                    [self.chirp_num * self.tx_num, -1, self.adc_sample]
                )

                # 转为 [chirps*tx, samples, rx]
                data = data.transpose([0, 2, 1])

                # 拆分 TX，拼虚拟阵列 (3 Tx × 4 Rx = 12 虚拟阵元)
                # 这里用 self.adc_sample * 3 是因为一帧有 3 组 TDM 发射
                ch1_data = data[0:self.adc_sample * 3:3, :, :]
                ch2_data = data[1:self.adc_sample * 3:3, :, :]
                ch3_data = data[2:self.adc_sample * 3:3, :, :]
                # 拼成 [chirps, samples, 12]
                data = np.concatenate([ch1_data, ch2_data, ch3_data], axis=2)

                # ============ DSP 处理 ============
                # RDA_Time: RTI / RDI / DTI
                rti, rdi, dti = DSP.RDA_Time(
                    data,
                    window_type_1d=Window.HANNING,
                    axis=1
                )
                # Range_Angle: RAI / REI
                rai, rei = DSP.Range_Angle(
                    data,
                    padding_size=[128, 64, 64]
                )

                # 推入上层队列（供 GUI 绘图）
                self.rti_queue.put(rti)
                self.dti_queue.put(dti)
                self.rdi_queue.put(rdi)
                self.rai_queue.put(rai)
                self.rei_queue.put(rei)

        # RawRecorder 的 close() 由 RecordingSession 在 main.py 中统一管理
