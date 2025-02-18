# recorders.py
# -*- coding: utf-8 -*-
import os, json, threading, datetime as dt
from typing import Optional, Dict, Any

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None  # 允许无相机环境导入


def _now_str():
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


class RawRecorder:
    """顺序写入原始 ADC 半缓冲，并维护 meta.json。线程安全。"""
    def __init__(self, session_dir: str):
        os.makedirs(session_dir, exist_ok=True)
        self.session_dir = session_dir
        self.bin_path = os.path.join(session_dir, "raw.bin")
        self.meta_path = os.path.join(session_dir, "meta.json")
        self._fh = open(self.bin_path, "wb")
        self._lock = threading.Lock()
        self.meta: Dict[str, Any] = {
            "format": "TI_mmWave_ADC_raw",
            "version": "1.0",
            "session_dir": self.session_dir,
            "start_time": dt.datetime.now().isoformat(timespec="seconds"),
            # 运行时填充/更新
            "frame_length_shorts": None,        # 每半缓冲长度(以int16计)
            "halfbuffer_blocks": 0,             # 写入的半缓冲块计数
            "adc_samples": None,
            "chirps": None,
            "tx": None,
            "rx": None,
            "iq_layout": "I0,Q0,I1,Q1 interleaved (int16)",
            "notes": "",
            "config_file": None,
        }

    def set_config(self, *, adc_samples: int, chirps: int, tx: int, rx: int,
                   frame_length_shorts: int, config_file: Optional[str] = None):
        self.meta["adc_samples"] = int(adc_samples)
        self.meta["chirps"] = int(chirps)
        self.meta["tx"] = int(tx)
        self.meta["rx"] = int(rx)
        self.meta["frame_length_shorts"] = int(frame_length_shorts)
        if config_file:
            self.meta["config_file"] = str(config_file)

    def append_halfbuffer(self, shorts: np.ndarray):
        """shorts: np.int16 1D，恰好为一半缓冲长度"""
        if shorts.dtype != np.int16:
            shorts = shorts.astype(np.int16, copy=False)
        with self._lock:
            self._fh.write(shorts.tobytes(order="C"))
            self.meta["halfbuffer_blocks"] += 1

    def close(self):
        with self._lock:
            if self._fh:
                self._fh.flush()
                self._fh.close()
                self._fh = None
        self.meta["end_time"] = dt.datetime.now().isoformat(timespec="seconds")
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2, ensure_ascii=False)


class GuiRecorder:
    """把传进来的 GUI 帧（BGR ndarray）写成 mp4。"""
    def __init__(self, session_dir: str, fps: int = 20, size: Optional[tuple[int, int]] = None):
        if cv2 is None:
            raise RuntimeError("OpenCV(cv2) 不可用，无法录制 GUI。请安装 opencv-python")
        os.makedirs(session_dir, exist_ok=True)
        self.session_dir = session_dir
        self.fps = fps
        self.size = size  # (w, h)，第一次写帧时如果没给就用帧本身尺寸
        self.video_path = os.path.join(session_dir, "gui.mp4")

        self.writer = None
        self._lock = threading.Lock()
        self._closed = False

    def write_frame(self, frame_bgr: np.ndarray):
        """
        frame_bgr: H×W×3, BGR uint8
        """
        if self._closed:
            return
        h, w = frame_bgr.shape[:2]

        if self.writer is None:
            if self.size is None:
                self.size = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.video_path, fourcc, self.fps, self.size)
            if not self.writer.isOpened():
                raise RuntimeError("无法创建 GUI 视频文件：{}".format(self.video_path))

        if self.size != (w, h):
            frame_bgr = cv2.resize(frame_bgr, self.size)

        with self._lock:
            self.writer.write(frame_bgr)

    def close(self):
        if self._closed:
            return
        with self._lock:
            try:
                if self.writer is not None:
                    self.writer.release()
            except Exception:
                pass
        self._closed = True


class RecordingSession:
    """管理一个完整会话（目录 + 原始ADC + GUI 录屏）。"""
    def __init__(self, base_dir: str = "recordings", tag: Optional[str] = None):
        stamp = _now_str()
        folder = stamp if tag is None else f"{stamp}_{tag}"
        self.session_dir = os.path.join(base_dir, folder)
        os.makedirs(self.session_dir, exist_ok=True)
        self.raw: Optional[RawRecorder] = None
        self.gui: Optional[GuiRecorder] = None

    def start_raw(self, *, adc_samples: int, chirps: int, tx: int, rx: int,
                  frame_length_shorts: int, config_file: Optional[str] = None):
        self.raw = RawRecorder(self.session_dir)
        self.raw.set_config(adc_samples=adc_samples, chirps=chirps, tx=tx, rx=rx,
                            frame_length_shorts=frame_length_shorts, config_file=config_file)

    def start_gui(self, fps: int = 20):
        self.gui = GuiRecorder(self.session_dir, fps=fps)

    def close(self):
        if self.gui:
            try:
                self.gui.close()
            except Exception:
                pass
        if self.raw:
            try:
                self.raw.close()
            except Exception:
                pass
