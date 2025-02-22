# version: 1.0

import numpy as np
from collections import deque
import dsp
from dsp.doppler_processing import doppler_processing
import dsp.range_processing as range_processing
import dsp.angle_estimation as Angle_dsp
import dsp.utils as utils
import dsp.compensation as Compensation
from dsp.utils import Window
import globalvar as gl

# ---- Robust params ----
_EPS = 1e-12        # for safe log
_MIN_ENERGY = 1e-9  # frame guard: skip ultra-weak/empty frames

rti_queue = deque(maxlen=12)
rdi_queue = deque(maxlen=12)
rai_queue = deque(maxlen=12)
rei_queue = deque(maxlen=12)

gesturetimecnt = 0
gesturetimecnt2 = 0

NUM_TX = 3
NUM_RX = 4
VIRT_ANT = 4
VIRT_ANT1 = 1
# Data specific parameters
NUM_CHIRPS = 64
NUM_ADC_SAMPLES = 64
RANGE_RESOLUTION = .0488
DOPPLER_RESOLUTION = 0.0806
NUM_FRAMES = 300

# DSP processing parameters
SKIP_SIZE = 4  # 忽略边缘角度的目标
ANGLE_RES = 2  # 角度分辨率
ANGLE_RANGE = 90  # 角度范围 +/-90°
ANGLE_FFT_BINS = 64
ANGLE_BINS = (ANGLE_RANGE * 2) // ANGLE_RES + 1
BINS_PROCESSED = 64

numRangeBins = NUM_ADC_SAMPLES
numDopplerBins = NUM_CHIRPS

# 计算分辨率
range_resolution, bandwidth = dsp.range_resolution(NUM_ADC_SAMPLES)
doppler_resolution = dsp.doppler_resolution(bandwidth)

# Start DSP processing
range_azimuth = np.zeros((int(ANGLE_BINS), BINS_PROCESSED))
range_elevation = np.zeros((int(ANGLE_BINS), BINS_PROCESSED))
azimuth_elevation = np.zeros((ANGLE_FFT_BINS, ANGLE_FFT_BINS, NUM_ADC_SAMPLES))

# theta跨度 theta分辨率 Vrx天线信号的数量
num_vec, steering_vec = Angle_dsp.gen_steering_vec(ANGLE_RANGE, ANGLE_RES, VIRT_ANT)


def _safe_log2_abs(x):
    return np.log2(np.abs(x) + _EPS)


def doppler_fft(x, window_type_2d=None):
    """
    对单帧做 2D Doppler FFT（微多普勒用）
    x: [num_chirps, num_range_bins]
    """
    fft2d_in = np.transpose(x, axes=(1, 0))  # [range_bin, chirps]
    if window_type_2d:
        fft2d_in = utils.windowing(fft2d_in, window_type_2d, axis=1)

    fft2d_out = np.fft.fft(fft2d_in, axis=1)       # [range_bin, doppler_bin]
    fft2d_log_abs = _safe_log2_abs(fft2d_out)
    det_matrix_vis = np.fft.fftshift(fft2d_log_abs, axes=1)
    return det_matrix_vis


framecnt = 0


def RDA_Time(adc_data, window_type_1d=None, clutter_removal_enabled=True,
             CFAR_enable=False, axis=-1):
    """
    Robust RTI/RDI/DTI, tolerant to empty frames:
    - energy guard: if frame energy < _MIN_ENERGY -> push zeros & return
    - safe log2
    """
    global gesturetimecnt, framecnt

    # 转换成(num_chirps_per_frame, num_rx_antennas, num_adc_samples)
    adc_data = np.transpose(adc_data, [0, 2, 1])

    # 能量守门：空帧/超弱帧直接返回零矩阵但维度一致
    if np.max(np.abs(adc_data)) < _MIN_ENERGY:
        _rti_zero = np.zeros((NUM_CHIRPS * 1, NUM_ADC_SAMPLES, 1))
        _rdi_zero = np.zeros((1, NUM_CHIRPS, NUM_ADC_SAMPLES, 12))
        _md_zero = np.zeros((16, 64))

        rti_queue.append(np.zeros((NUM_CHIRPS, NUM_ADC_SAMPLES)))
        rdi_queue.append(
            np.zeros((NUM_ADC_SAMPLES, NUM_CHIRPS, 12))
        )
        return _rti_zero, np.array(rdi_queue), _md_zero

    # 距离 FFT
    radar_cube = range_processing(2 * adc_data[:, :, 0:64:2], window_type_1d, 2)

    if clutter_removal_enabled:
        radar_cube = Compensation.clutter_removal(radar_cube, axis=0)

    # 距离-多普勒
    range_doppler_fft, aoa_input = doppler_processing(
        radar_cube,
        num_tx_antennas=3,
        interleaved=False,
        clutter_removal_enabled=False,  # 前面已做
        window_type_2d=Window.HANNING,
        accumulate=False
    )

    # range_doppler_fft: [range, virt_ant, doppler]
    rdi_abs = np.transpose(
        np.fft.fftshift(np.abs(range_doppler_fft), axes=2),
        [0, 2, 1]
    )  # [range, doppler, virt_ant]
    rdi_abs = np.flip(rdi_abs, axis=0)   # 与原始显示保持一致
    rdi_queue.append(rdi_abs)
    rdi_framearray = np.array(rdi_queue)  # [frames, range, doppler, virt_ant]

    # 距离-时间 (RTI) 使用 rx=0
    det_matrix = radar_cube[:, 0, :]  # [chirps, range_bins]

    # 手势触发逻辑（与原逻辑一致）
    Iscapture = gl.get_value('IsRecognizeorCapture')
    if np.sum(det_matrix[:, 36:62] > 3e3) > 14:
        if Iscapture:
            gesturetimecnt = gesturetimecnt + 1

    if (gesturetimecnt >= 2) and Iscapture:
        framecnt = framecnt + 1
        if framecnt >= 8:
            if gl.get_value('timer_2s'):
                gl.set_value('usr_gesture', True)
                gl.set_value('timer_2s', False)
            framecnt = 0
            gesturetimecnt = 0

    rti_queue.append(det_matrix)
    rti_framearray = np.array(rti_queue)          # [frames, chirps, range]
    rti_array = np.reshape(rti_framearray, (1, -1, 64))  # [1, frames*chirps, range]
    rti_array_out = np.transpose(rti_array, [1, 2, 0])   # [time, range, 1]

    # 微多普勒 (DTI)
    micro_doppler_data = np.zeros(
        (rti_framearray.shape[0],
         rti_framearray.shape[1],
         rti_framearray.shape[2]),
        dtype=np.float64
    )
    for i, frame in enumerate(rti_framearray):
        det_matrix_vis = doppler_fft(frame, window_type_2d=Window.HANNING)
        micro_doppler_data[i, :, :] = det_matrix_vis

    # RTI 阈值
    rti_array_out = np.flip(np.abs(rti_array_out), axis=1)
    rti_array_out[rti_array_out < 3e3] = 0

    # DTI: 聚合 range 维度
    micro_doppler_data_out = micro_doppler_data.sum(axis=1)
    micro_doppler_data_out[micro_doppler_data_out < 20] = 0

    return rti_array_out, rdi_framearray, micro_doppler_data_out


def Range_Angle(data, padding_size=None, clutter_removal_enabled=True,
                window_type_1d=Window.HANNING, Music_enable=False):
    """
    生成 RAI / REI（范围-方位 / 范围-俯仰）。

    相比原始版本：
    - 删除了 rdi_ab1[:,40:90] = 0 / rdi_ab2[:,40:90] = 0 这种硬裁剪，
      不再人为抹掉一整段 range，避免“永远是空白”的区域。
    - 做了一点温和的限幅（防止单个点特别亮把整个色标拉爆）。
    """

    # 转换成 (num_chirps, num_rx, num_adc_samples)
    adc_data = np.transpose(data, [0, 2, 1])

    # 空/弱帧兜底
    if np.max(np.abs(adc_data)) < _MIN_ENERGY:
        rai_queue.append(np.zeros((int(ANGLE_BINS), BINS_PROCESSED)))
        rei_queue.append(np.zeros((int(ANGLE_BINS), BINS_PROCESSED)))
        return np.array(rai_queue), np.array(rei_queue)

    # 距离 FFT
    radar_cube = range_processing(2 * adc_data[:, :, 0:64:3], window_type_1d, 2)

    if clutter_removal_enabled:
        radar_cube = Compensation.clutter_removal(radar_cube, axis=0)

    # 粗略算一个帧 SNR，用来做整体缩放（只影响颜色，不影响检测）
    frame_SNR = np.log(np.sum(np.abs(radar_cube[:, :])) + _EPS) - 14.7
    if np.abs(frame_SNR) < 1.8:
        frame_SNR = 0.0

    # --- Capon 波束形成 ---
    # 方位角（Azimuth）
    for i in range(BINS_PROCESSED):
        if Music_enable:
            range_azimuth[:, i] = dsp.aoa_music_1D(
                steering_vec, radar_cube[:, [10, 8, 6, 4], i].T, num_sources=1
            )
        else:
            range_azimuth[:, i], _ = dsp.aoa_capon(
                radar_cube[:, [7, 4, 3, 0], i].T, steering_vec, magnitude=True
            )

    # 俯仰角（Elevation）
    for i in range(BINS_PROCESSED):
        if Music_enable:
            range_elevation[:, i] = dsp.aoa_music_1D(
                steering_vec, radar_cube[:, [1, 0, 9, 8], i].T, num_sources=1
            )
        else:
            range_elevation[:, i], _ = dsp.aoa_capon(
                radar_cube[:, [7, 6, 11, 10], i].T * [[1], [-1], [1], [-1]],
                steering_vec,
                magnitude=True
            )

    # 取幅度并翻转 range 方向（和原代码保持一致）
    rai_mag = np.flip(np.abs(range_azimuth), axis=1)
    rei_mag = np.flip(np.abs(range_elevation), axis=1)

    # 防止个别点过亮：做一个简单的上限（max/2）
    if rai_mag.size:
        vmax1 = rai_mag.max()
        if vmax1 > 0:
            rai_mag = np.minimum(rai_mag, vmax1 / 2.0)
    if rei_mag.size:
        vmax2 = rei_mag.max()
        if vmax2 > 0:
            rei_mag = np.minimum(rei_mag, vmax2 / 2.0)

    # 用 frame_SNR 做一个整体缩放，避免动态范围过小
    if rai_mag.max() > 0:
        rai_mag = rai_mag / rai_mag.max() * frame_SNR
    if rei_mag.max() > 0:
        rei_mag = rei_mag / rei_mag.max() * frame_SNR

    # 入队列，做时间滑窗
    rai_queue.append(rai_mag)
    rei_queue.append(rei_mag)

    rai_framearray = np.array(rai_queue)
    rei_framearray = np.array(rei_queue)
    return rai_framearray, rei_framearray
