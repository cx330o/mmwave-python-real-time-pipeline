import json
import numpy as np
import os
from numpy.fft import fft

# ====== 配置：录制文件所在的目录 ======
# 因为我们把这个脚本放在“那一次实验的目录”里，
# 所以直接写当前目录 "."
SESSION_DIR = "."

# 注意：有的系统会显示成 "meta"，但实际文件名是 meta.json
meta_path = os.path.join(SESSION_DIR, "meta.json")
raw_path  = os.path.join(SESSION_DIR, "raw.bin")

with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)

adc_samples = int(meta["adc_samples"])
chirps      = int(meta["chirps"])
tx          = int(meta["tx"])
rx          = int(meta["rx"])
frame_len   = int(meta["frame_length_shorts"])   # 每帧 short 数量

print("adc_samples:", adc_samples)
print("chirps:", chirps)
print("tx:", tx)
print("rx:", rx)
print("frame_len_shorts:", frame_len)

# 读取全部 short 数据
raw = np.fromfile(raw_path, dtype=np.int16)

# 能整除 frame_len 的就是完整帧
n_frames = raw.size // frame_len
if n_frames == 0:
    raise RuntimeError("raw.bin 里没有完整的一帧数据？")

raw = raw[: n_frames * frame_len]
raw = raw.reshape(n_frames, frame_len)
print("total frames:", n_frames)


def decode_one_frame(frame_shorts: np.ndarray) -> np.ndarray:
    """
    把一帧 short[] 解码成
    (chirps, adc_samples, virt_ant) 的复数数组，
    解码规则和 real_time_process.DataProcessor 里完全一致。
    """
    # [N,4] -> (I0,Q0,I1,Q1)
    d = frame_shorts.reshape(-1, 4)
    iq = d[:, 0:2] + 1j * d[:, 2:4]          # (N,2) complex

    # (chirps*tx, rx, adc_samples)
    iq = iq.reshape(chirps * tx, rx, adc_samples)
    # -> (chirps*tx, adc_samples, rx)
    iq = iq.transpose(0, 2, 1)

    # 按 TDM 把 TX 拆开，得到虚拟阵列 (参考 real_time_process.py)
    # 0,3,6,... -> TX1; 1,4,7,... -> TX2; 2,5,8,... -> TX3
    ch1 = iq[0: adc_samples * 3: 3, :, :]   # (chirps, adc, rx)
    ch2 = iq[1: adc_samples * 3: 3, :, :]
    ch3 = iq[2: adc_samples * 3: 3, :, :]
    virt = np.concatenate([ch1, ch2, ch3], axis=2)   # (chirps, adc, rx*tx=12)

    return virt


# ---------- 1) 时域立方体: (frames, chirps, adc, virt_ant) ----------
adc_cube = np.stack([decode_one_frame(fr) for fr in raw], axis=0)
print("adc_cube shape (frames, chirps, adc, virt_ant):", adc_cube.shape)

frames, c, s, ants = adc_cube.shape

# ---------- 2) 对 ADC 方向做 Range FFT ----------
window = np.hanning(s).astype(np.float32)
adc_cube_win = adc_cube * window[None, None, :, None]

# FFT 沿 adc_samples 轴（axis=2）
range_cube = fft(adc_cube_win, axis=2)   # (frames, chirps, range_bin, virt_ant)

# 和 Notebook 一样的排列: [frames, chirps, antenna, range_bin]
radar_cube = np.transpose(range_cube, (0, 1, 3, 2))   # (frames, chirps, virt_ant, range_bin)

save_path = os.path.join(SESSION_DIR, "my_recording.npy")
np.save(save_path, radar_cube.astype(np.complex64))

print("saved:", save_path)
