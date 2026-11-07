"""Phát hiện sự kiện dáng đi theo phương pháp Zeni et al. (2008).

Phương pháp Zeni dùng **tốc độ tương đối** của các mốc bàn chân so với
điểm bắt nguồn (sacrum/pelvis): heel-strike xảy ra khi gót chân đạt cực đại
hướng tiến (trước sacrum nhất), toe-off xảy ra khi ngón cái đạt cực tiểu
(sau sacrum nhất). Phương pháp này không đòi hỏi ngưỡng lực chân hay nền sàn
phẳng — chỉ cần tọa độ 3D hoặc 2D (trục tiến).

Tham chiếu:
    Zeni JA Jr, Richards JG, Higginson JS. "Two simple methods for determining
    gait events during treadmill and overground walking using kinematic data."
    Gait Posture 2008;27(4):710-714. doi:10.1016/j.gaitpost.2007.07.007

Quy ước chỉ số keypoint (COCO-WholeBody 23 hoặc đầy đủ 133):
    - L_HIP = 11, R_HIP = 12
    - L_ANKLE = 15, R_ANKLE = 16
    - L_HEEL = 19,  R_HEEL = 22
    - L_BIG_TOE = 17, R_BIG_TOE = 20

API chính: ``detect_events(pose_3d, fps, ...)``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

# --- Chỉ số keypoint mặc định ---
_L_HIP, _R_HIP = 11, 12
_L_HEEL, _R_HEEL = 19, 22
_L_BIG_TOE, _R_BIG_TOE = 17, 20


@dataclass
class GaitEvents:
    """Container kết quả phát hiện sự kiện dáng đi.

    Mỗi mảng chứa các chỉ số frame (int) của sự kiện theo thứ tự thời gian.
    """

    heel_strike_L: IntArray = field(default_factory=lambda: np.array([], dtype=np.int64))
    heel_strike_R: IntArray = field(default_factory=lambda: np.array([], dtype=np.int64))
    toe_off_L: IntArray = field(default_factory=lambda: np.array([], dtype=np.int64))
    toe_off_R: IntArray = field(default_factory=lambda: np.array([], dtype=np.int64))

    def __repr__(self) -> str:
        return (
            f"GaitEvents("
            f"HS_L={len(self.heel_strike_L)}, HS_R={len(self.heel_strike_R)}, "
            f"TO_L={len(self.toe_off_L)}, TO_R={len(self.toe_off_R)})"
        )

    def as_dict(self) -> Dict[str, IntArray]:
        return {
            "heel_strike_L": self.heel_strike_L,
            "heel_strike_R": self.heel_strike_R,
            "toe_off_L": self.toe_off_L,
            "toe_off_R": self.toe_off_R,
        }


def _sacrum_position(
    pose: FloatArray,
    l_hip_idx: int,
    r_hip_idx: int,
) -> FloatArray:
    """Tính vị trí sacrum = trung điểm 2 khớp háng. (T, >=2)."""
    return 0.5 * (pose[:, l_hip_idx, :] + pose[:, r_hip_idx, :])


def _find_local_maxima(signal: FloatArray, min_distance: int = 1) -> IntArray:
    """Tìm các cực đại cục bộ trong tín hiệu 1D.

    Args:
        signal: Mảng 1D (T,).
        min_distance: Khoảng cách tối thiểu giữa hai cực đại (frames).

    Returns:
        Mảng chỉ số của các cực đại, đã sắp xếp tăng dần.
    """
    n = len(signal)
    peaks = []
    for i in range(1, n - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            peaks.append(i)

    if not peaks or min_distance <= 1:
        return np.array(peaks, dtype=np.int64)

    # Lọc theo min_distance: giữ cực đại lớn hơn trong vùng
    peaks_arr = np.array(peaks, dtype=np.int64)
    keep = np.ones(len(peaks_arr), dtype=bool)
    for i in range(len(peaks_arr)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(peaks_arr)):
            if peaks_arr[j] - peaks_arr[i] < min_distance:
                if signal[peaks_arr[j]] > signal[peaks_arr[i]]:
                    keep[i] = False
                    break
                else:
                    keep[j] = False
            else:
                break
    return peaks_arr[keep]


def _find_local_minima(signal: FloatArray, min_distance: int = 1) -> IntArray:
    """Tìm các cực tiểu cục bộ (= cực đại của -signal)."""
    return _find_local_maxima(-signal, min_distance=min_distance)


def detect_events(
    pose_3d: FloatArray,
    fps: float,
    fwd_axis: int = 0,
    min_stride_s: float = 0.3,
    l_hip_idx: int = _L_HIP,
    r_hip_idx: int = _R_HIP,
    l_heel_idx: int = _L_HEEL,
    r_heel_idx: int = _R_HEEL,
    l_toe_idx: int = _L_BIG_TOE,
    r_toe_idx: int = _R_BIG_TOE,
    smooth_sigma_s: float = 0.0,
) -> GaitEvents:
    """Phát hiện heel-strike và toe-off theo phương pháp Zeni et al. (2008).

    Args:
        pose_3d: Mảng (T, J, 3) hoặc (T, J, 2) — tọa độ 3D hoặc 2D.
        fps: Tốc độ khung hình (Hz).
        fwd_axis: Trục tiến trong tọa độ 3D (0=X, 1=Y, 2=Z). Mặc định 0 (X).
        min_stride_s: Khoảng cách tối thiểu giữa hai sự kiện cùng loại (giây).
        l_hip_idx: Chỉ số khớp háng trái.
        r_hip_idx: Chỉ số khớp háng phải.
        l_heel_idx: Chỉ số gót chân trái.
        r_heel_idx: Chỉ số gót chân phải.
        l_toe_idx: Chỉ số ngón cái trái.
        r_toe_idx: Chỉ số ngón cái phải.
        smooth_sigma_s: Độ lệch chuẩn Gaussian làm mượt tín hiệu (giây).
                        0 = không làm mượt.

    Returns:
        ``GaitEvents`` chứa các mảng chỉ số frame.

    Thuật toán (Zeni et al., 2008):
        - Sacrum = trung điểm 2 háng.
        - Tín hiệu heel (L/R) = heel_x - sacrum_x  (tọa độ trục tiến).
        - Tín hiệu toe  (L/R) = toe_x  - sacrum_x.
        - Heel-strike: cực đại của tín hiệu heel (gót ở xa trước sacrum nhất).
        - Toe-off:     cực tiểu của tín hiệu toe (ngón ở xa sau sacrum nhất).
    """
    pose = np.asarray(pose_3d, dtype=np.float64)
    if pose.ndim != 3 or pose.shape[2] < 2:
        raise ValueError(f"pose_3d phải có shape (T, J, >=2), nhận {pose.shape}.")
    if fps <= 0:
        raise ValueError(f"fps phải > 0, nhận {fps}.")

    ax = fwd_axis
    min_distance = max(1, int(round(min_stride_s * fps)))

    sacrum = _sacrum_position(pose, l_hip_idx, r_hip_idx)  # (T, >=2)
    sac_x = sacrum[:, ax]

    # Điền NaN bằng nội suy tuyến tính đơn giản
    def _fill_nan(sig: FloatArray) -> FloatArray:
        s = sig.copy()
        nans = np.isnan(s)
        if nans.any() and (~nans).any():
            idx = np.arange(len(s))
            s[nans] = np.interp(idx[nans], idx[~nans], s[~nans])
        return s

    sac_x = _fill_nan(sac_x)

    def _signal(joint_idx: int) -> FloatArray:
        return _fill_nan(pose[:, joint_idx, ax]) - sac_x

    sig_heel_L = _signal(l_heel_idx)
    sig_heel_R = _signal(r_heel_idx)
    sig_toe_L = _signal(l_toe_idx)
    sig_toe_R = _signal(r_toe_idx)

    # Làm mượt tùy chọn (Gaussian, chỉ dùng khi scipy có sẵn)
    if smooth_sigma_s > 0:
        try:
            from scipy.ndimage import gaussian_filter1d  # type: ignore

            sigma_frames = smooth_sigma_s * fps
            sig_heel_L = gaussian_filter1d(sig_heel_L, sigma_frames)
            sig_heel_R = gaussian_filter1d(sig_heel_R, sigma_frames)
            sig_toe_L = gaussian_filter1d(sig_toe_L, sigma_frames)
            sig_toe_R = gaussian_filter1d(sig_toe_R, sigma_frames)
        except ImportError:
            pass  # Bỏ qua làm mượt nếu scipy chưa cài

    return GaitEvents(
        heel_strike_L=_find_local_maxima(sig_heel_L, min_distance),
        heel_strike_R=_find_local_maxima(sig_heel_R, min_distance),
        toe_off_L=_find_local_minima(sig_toe_L, min_distance),
        toe_off_R=_find_local_minima(sig_toe_R, min_distance),
    )


def stride_time(events: GaitEvents, fps: float) -> Dict[str, FloatArray]:
    """Tính thời gian bước (stride time) từ GaitEvents.

    Stride time = khoảng cách giữa hai heel-strike liên tiếp cùng bên (giây).
    """
    out: Dict[str, FloatArray] = {}
    for side, hs in (("L", events.heel_strike_L), ("R", events.heel_strike_R)):
        if len(hs) >= 2:
            out[f"stride_time_{side}"] = np.diff(hs.astype(np.float64)) / fps
        else:
            out[f"stride_time_{side}"] = np.array([], dtype=np.float64)
    return out


def step_length_ratio(events: GaitEvents) -> Optional[float]:
    """Tỉ lệ số lần heel-strike Trái / Phải.

    Với LLD, bên chân ngắn thường có chu kỳ bước khác bên dài.
    Returns None nếu không đủ dữ liệu.
    """
    nL, nR = len(events.heel_strike_L), len(events.heel_strike_R)
    if nL == 0 or nR == 0:
        return None
    return float(nL) / float(nR)

