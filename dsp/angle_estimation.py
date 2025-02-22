# Copyright 2019 The OpenRadar Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import warnings

# ---- Robustness params ----
_EPS = 1e-9         # small epsilon for diagonal loading / divisions
_DIAG_ALPHA = 1e-3  # diagonal loading factor (可按需要调)

# ----- helper constants used by original pipeline -----
RANGEIDX = 0
DOPPLERIDX = 1

def DOPPLER_IDX_TO_SIGNED(idx, num_bins):
    """Convert doppler index [0..N-1] to signed index [-N/2..N/2-1] with fftshift convention."""
    half = num_bins // 2
    idx = np.asarray(idx).astype(np.int32)
    signed = idx.copy()
    signed[signed >= half] -= num_bins
    return signed

# ------------------------------- Core AOA utilities -------------------------------

def cov_matrix(x):
    """
    Rxx = 1/N * X * X^H
    x shape: (Vrx, N)  (rows = virtual antennas, columns = snapshots)
    """
    if x.ndim > 2:
        raise ValueError("x has more than 2 dimensions.")

    if x.shape[0] > x.shape[1]:
        warnings.warn("cov_matrix input should have Vrx as rows. Needs to be transposed", RuntimeWarning)
        x = x.T

    _, num_adc_samples = x.shape
    Rxx = x @ np.conjugate(x.T)
    Rxx = np.divide(Rxx, max(num_adc_samples, 1))
    return Rxx

def cov_matrix_x1_x2(x1, x2):
    """Cross-covariance variant used by aoa_capon_new (保留接口以兼容旧代码)."""
    if x1.ndim > 2 or x2.ndim > 2:
        raise ValueError("x has more than 2 dimensions.")

    if x1.shape[0] > x1.shape[1] or x2.shape[0] > x2.shape[1]:
        warnings.warn("cov_matrix input should have Vrx as rows. Needs to be transposed", RuntimeWarning)
        x1 = x1.T
        x2 = x2.T

    _, num_adc_samples = x1.shape
    Rxx = x1 @ np.conjugate(x2.T)
    Rxx1 = x2 @ np.conjugate(x1.T)
    Rxx = np.divide(Rxx * Rxx1 / 4, max(num_adc_samples, 1))
    return Rxx

def forward_backward_avg(Rxx):
    """前后向平均，改善协方差矩阵的条件数。"""
    assert np.size(Rxx, 0) == np.size(Rxx, 1)
    M = np.size(Rxx, 0)
    Rxx = np.matrix(Rxx)
    J = np.fliplr(np.eye(M))
    J = np.matrix(J)
    R_fb = 0.5 * (Rxx + J * np.conjugate(Rxx) * J)
    return np.array(R_fb)

# ------------------------------- Steering vector generators -------------------------------

def gen_steering_vec(angle_range_deg, angle_res_deg, num_virt_ant):
    """
    生成 ULA(间距=λ/2) 的 1D 导向矢量表。
    返回:
      num_vec: 角度采样点个数
      steering_vec: shape=(num_vec, num_virt_ant)
    """
    # 角度取值：[-range, +range]，步长 = res
    thetas = np.arange(-angle_range_deg, angle_range_deg + 1e-6, angle_res_deg, dtype=np.float64)
    num_vec = len(thetas)
    theta_rad = np.deg2rad(thetas)

    # 虚拟天线索引 n = 0..M-1；ULA，d = λ/2 -> 相位增量为 π*sin(theta)
    n = np.arange(num_virt_ant, dtype=np.float64)[None, :]  # shape (1, M)
    phase = np.pi * np.sin(theta_rad)[:, None] * n          # shape (K, M)
    steering = np.exp(1j * phase)                           # (K, M)

    return num_vec, steering

def gen_steering_vec_el(angle_range_deg, angle_res_deg, num_virt_ant):
    """兼容旧接口：俯仰向 1D 导向矢量（此处与方位相同做法，保持一致性）"""
    return gen_steering_vec(angle_range_deg, angle_res_deg, num_virt_ant)

def gen_steering_vec_2D(angle_range_deg, angle_res_deg, num_virt_ant_az, num_virt_ant_el):
    """
    2D 导向矢量（方位×俯仰），这里给出简化实现，按 separable 假设构造 Kronecker product。
    返回:
      steering2d: shape=(K*K, num_virt_ant_az*num_virt_ant_el)
    """
    _, az = gen_steering_vec(angle_range_deg, angle_res_deg, num_virt_ant_az)   # (K, Maz)
    _, el = gen_steering_vec(angle_range_deg, angle_res_deg, num_virt_ant_el)   # (K, Mel)

    K = az.shape[0]
    steering2d = np.zeros((K*K, num_virt_ant_az * num_virt_ant_el), dtype=np.complex128)
    idx = 0
    for i in range(K):
        for j in range(K):
            # Kronecker (逐元素外积) -> (Maz*Mel,)
            v = np.kron(az[i, :], el[j, :])
            steering2d[idx, :] = v
            idx += 1
    return steering2d

# ------------------------------- AOA algorithms (robust) -------------------------------

def aoa_bartlett(steering_vec, sig_in, axis):
    y = np.matmul(np.conjugate(steering_vec), sig_in.swapaxes(axis, np.arange(len(sig_in.shape))[-2]))
    doa_spectrum = np.abs(y) ** 2
    return doa_spectrum.swapaxes(axis, np.arange(len(sig_in.shape))[-2])

def _invert_loaded(Rxx):
    """
    对角加载 + 失败回退到 pinv。
    """
    M = Rxx.shape[0]
    trace = float(np.trace(Rxx).real) if M > 0 else 0.0
    loaded = Rxx + (_DIAG_ALPHA * trace / (M + 1e-12) + _EPS) * np.eye(M, dtype=Rxx.dtype)
    try:
        Rin = np.linalg.inv(loaded)
    except np.linalg.LinAlgError:
        Rin = np.linalg.pinv(loaded, rcond=1e-6)
    return Rin

def aoa_capon(x, steering_vector, magnitude=False):
    """
    Robust Capon: diagonal loading + pinv fallback to avoid singular matrix.
    x shape: (Vrx, N)  (rows = virtual antennas)
    steering_vector shape: (K, Vrx)
    """
    if steering_vector.shape[1] != x.shape[0]:
        raise ValueError("'steering_vector' with shape (%d,%d) cannot matrix multiply 'input_data' with shape (%d,%d)" \
        % (steering_vector.shape[0], steering_vector.shape[1], x.shape[0], x.shape[1]))

    Rxx = cov_matrix(x)
    Rxx = forward_backward_avg(Rxx)

    Rxx_inv = _invert_loaded(Rxx)

    first = Rxx_inv @ steering_vector.T
    den = np.reciprocal(np.einsum('ij,ij->i', steering_vector.conj(), first.T))
    weights = np.matmul(first, den)

    if magnitude:
        return np.abs(den), weights
    else:
        return den, weights

def aoa_capon_2D(x, steering_vector, ang_est_range, ang_est_resolution, magnitude=False):
    num_vec = int(round(2 * ang_est_range / ang_est_resolution + 1))
    Rxx = cov_matrix(x)
    Rxx = forward_backward_avg(Rxx)

    Rxx_inv = _invert_loaded(Rxx)

    first = Rxx_inv @ steering_vector.T
    den = np.reciprocal(np.einsum('ij,ij->i', steering_vector.conj(), first.T))
    den = np.reshape(den, (num_vec, num_vec), order='F')
    return np.abs(den)

def aoa_capon_new(x, x1, steering_vector, magnitude=False):
    if steering_vector.shape[1] != x.shape[0]:
        raise ValueError("'steering_vector' with shape (%d,%d) cannot matrix multiply 'input_data' with shape (%d,%d)" \
        % (steering_vector.shape[0], steering_vector.shape[1], x.shape[0], x.shape[1]))

    Rxx = cov_matrix_x1_x2(x, x1)
    Rxx = forward_backward_avg(Rxx)

    Rxx_inv = _invert_loaded(Rxx)

    first = Rxx_inv @ steering_vector.T
    den = np.reciprocal(np.einsum('ij,ij->i', steering_vector.conj(), first.T))
    weights = np.matmul(first, den)

    if magnitude:
        return np.abs(den), weights
    else:
        return den, weights

# ------------------------------- 高层封装（与旧管线兼容） -------------------------------

def azimuth_processing(radar_cube, det_obj_2d, config, window_type_2d=None):
    """保持原接口，供旧管线调用。若你未使用，可忽略。"""
    num_det_obj = det_obj_2d.shape[0]
    fft2d_azimuth_in = radar_cube[..., det_obj_2d[:, RANGEIDX].astype(np.uint32)]

    # Rearrange same as 2D FFT stage in TI pipeline
    fft2d_azimuth_in = np.concatenate((fft2d_azimuth_in[0::2, ...], fft2d_azimuth_in[1::2, ...]), axis=1)
    fft2d_azimuth_in = np.transpose(fft2d_azimuth_in, axes=(2, 1, 0))  # (numDet, numVirtAnt, numDopp)

    fft2d_azimuth_out = np.fft.fft(fft2d_azimuth_in)
    fft2d_azimuth_out = np.fft.fftshift(fft2d_azimuth_out, axes=2)

    azimuth_in = np.zeros((num_det_obj, config.numAngleBins), dtype=np.complex_)
    azimuth_in[:, :config.numVirtualAntAzim] = np.array(
        [fft2d_azimuth_out[i, :, dopplerIdx] for i, dopplerIdx in enumerate(det_obj_2d[:, DOPPLERIDX].astype(np.uint32))]
    ).squeeze()

    # 3rd FFT (angle FFT)
    azimuth_out = np.fft.fft(azimuth_in)
    azimuth_mag_sqr = np.abs(azimuth_out) ** 2

    det_obj2d_azimuth = np.zeros_like(det_obj_2d)
    det_obj2d_azimuth[:, :] = det_obj_2d
    det_obj2d_azimuth[:, DOPPLERIDX] = DOPPLER_IDX_TO_SIGNED(det_obj2d_azimuth[:, DOPPLERIDX], config.numDopplerBins)
    return det_obj2d_azimuth
