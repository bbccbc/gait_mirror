"""Tests tổng hợp cho gaitmirror.gait.zeni_events.

Kiểm tra:
- _find_local_maxima / _find_local_minima: tính đúng đắn thuật toán
- detect_events(): heel-strike và toe-off với tín hiệu tổng hợp
- Bộ phận xử lý biên: NaN, chuỗi ngắn, không có sự kiện
- stride_time() và step_length_ratio()
"""

import numpy as np
import pytest

from gaitmirror.gait.zeni_events import (
    GaitEvents,
    _find_local_maxima,
    _find_local_minima,
    detect_events,
    stride_time,
    step_length_ratio,
)


# ====================================================================== #
# Helpers
# ====================================================================== #

def _make_synthetic_gait(
    n_frames: int = 300,
    fps: float = 60.0,
    stride_frames: int = 90,
    phase_L: float = 0.0,
    phase_R: float = 0.5,
    n_joints: int = 23,
    fwd_axis: int = 0,
) -> np.ndarray:
    """Tạo pose 3D tổng hợp với chuyển động bước đi hình sin đơn giản.

    Sacrum đứng yên tại gốc tọa độ.
    Heel-strike xảy ra tại cực đại của sin(2π * t / T + phase).
    Toe-off xảy ra tại cực tiểu.

    Returns:
        pose: (n_frames, n_joints, 3)
    """
    pose = np.zeros((n_frames, n_joints, 3), dtype=np.float64)
    t = np.arange(n_frames)
    omega = 2 * np.pi / stride_frames

    # L_HIP=11, R_HIP=12: sacrum không di chuyển
    pose[:, 11, fwd_axis] = 0.0
    pose[:, 12, fwd_axis] = 0.0

    # L_HEEL=19, R_HEEL=22
    pose[:, 19, fwd_axis] = 0.4 * np.sin(omega * t + phase_L * 2 * np.pi)
    pose[:, 22, fwd_axis] = 0.4 * np.sin(omega * t + phase_R * 2 * np.pi)

    # L_BIG_TOE=17, R_BIG_TOE=20 (toe-off ở cực tiểu → đặt lùi sau sacrum)
    pose[:, 17, fwd_axis] = 0.4 * np.sin(omega * t + phase_L * 2 * np.pi + np.pi)
    pose[:, 20, fwd_axis] = 0.4 * np.sin(omega * t + phase_R * 2 * np.pi + np.pi)

    return pose


# ====================================================================== #
# 1. _find_local_maxima / _find_local_minima
# ====================================================================== #

class TestFindLocalMaxima:
    """Kiểm tra _find_local_maxima."""

    def test_single_peak(self):
        signal = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
        peaks = _find_local_maxima(signal)
        assert list(peaks) == [1]

    def test_two_peaks(self):
        signal = np.array([0.0, 1.0, 0.0, 0.5, 0.0])
        peaks = _find_local_maxima(signal)
        assert 1 in peaks
        assert 3 in peaks

    def test_no_peaks_flat(self):
        signal = np.ones(10)
        peaks = _find_local_maxima(signal)
        assert len(peaks) == 0

    def test_no_peaks_monotone(self):
        signal = np.arange(10, dtype=float)
        peaks = _find_local_maxima(signal)
        assert len(peaks) == 0

    def test_min_distance_filters_close_peaks(self):
        """Hai cực đại cách nhau ít hơn min_distance bị lọc."""
        signal = np.array([0, 1.0, 0, 0.9, 0, 0, 0, 1.2, 0])
        # Với min_distance=5: chỉ giữ cực đại tại index 7 (lớn nhất trong window)
        peaks = _find_local_maxima(signal, min_distance=5)
        assert 7 in peaks
        assert 1 not in peaks or 3 not in peaks  # ít nhất 1 bị lọc

    def test_returns_sorted_array(self):
        t = np.linspace(0, 4 * np.pi, 400)
        signal = np.sin(t)
        peaks = _find_local_maxima(signal)
        assert list(peaks) == sorted(peaks)

    def test_dtype_int64(self):
        signal = np.array([0.0, 1.0, 0.0])
        peaks = _find_local_maxima(signal)
        assert peaks.dtype == np.int64


class TestFindLocalMinima:
    """Kiểm tra _find_local_minima = cực đại của -signal."""

    def test_single_minimum(self):
        signal = np.array([0.0, -1.0, 0.0])
        mins = _find_local_minima(signal)
        assert list(mins) == [1]

    def test_minima_of_sin_at_pi(self):
        t = np.linspace(0, 2 * np.pi, 100)
        signal = np.sin(t)
        mins = _find_local_minima(signal)
        # Cực tiểu tại ~t = 3π/2 ≈ index 75
        assert len(mins) > 0
        assert all(signal[i] < 0 for i in mins)


# ====================================================================== #
# 2. detect_events() — tín hiệu tổng hợp
# ====================================================================== #

class TestDetectEvents:
    """Kiểm tra detect_events() với tín hiệu tổng hợp."""

    def test_returns_gait_events_instance(self):
        pose = _make_synthetic_gait()
        events = detect_events(pose, fps=60.0)
        assert isinstance(events, GaitEvents)

    def test_heel_strike_detected_both_sides(self):
        """Phải phát hiện ít nhất 1 HS mỗi bên với tín hiệu sin."""
        pose = _make_synthetic_gait(n_frames=300, stride_frames=90)
        events = detect_events(pose, fps=60.0, min_stride_s=0.5)
        assert len(events.heel_strike_L) >= 2
        assert len(events.heel_strike_R) >= 2

    def test_toe_off_detected_both_sides(self):
        """Phải phát hiện ít nhất 1 TO mỗi bên."""
        pose = _make_synthetic_gait(n_frames=300, stride_frames=90)
        events = detect_events(pose, fps=60.0, min_stride_s=0.5)
        assert len(events.toe_off_L) >= 2
        assert len(events.toe_off_R) >= 2

    def test_events_are_sorted_ascending(self):
        """Các sự kiện phải theo thứ tự tăng dần."""
        pose = _make_synthetic_gait(n_frames=300, stride_frames=90)
        events = detect_events(pose, fps=60.0)
        for arr in [events.heel_strike_L, events.heel_strike_R,
                    events.toe_off_L, events.toe_off_R]:
            assert list(arr) == sorted(arr)

    def test_events_in_valid_frame_range(self):
        """Tất cả frame index phải trong [1, T-2] (không phải biên)."""
        n_frames = 300
        pose = _make_synthetic_gait(n_frames=n_frames, stride_frames=90)
        events = detect_events(pose, fps=60.0)
        for arr in [events.heel_strike_L, events.heel_strike_R,
                    events.toe_off_L, events.toe_off_R]:
            assert all(0 < idx < n_frames - 1 for idx in arr)

    def test_stride_period_matches_expected(self):
        """Stride time ước tính phải xấp xỉ stride_frames / fps."""
        fps = 60.0
        stride_frames = 90
        pose = _make_synthetic_gait(n_frames=600, fps=fps,
                                    stride_frames=stride_frames)
        events = detect_events(pose, fps=fps, min_stride_s=0.5)
        st = stride_time(events, fps)
        if len(st["stride_time_L"]) > 0:
            expected_s = stride_frames / fps
            np.testing.assert_allclose(
                st["stride_time_L"], expected_s, rtol=0.15
            )

    def test_phase_offset_L_R(self):
        """HS_L và HS_R phải lệch nhau ~half-stride (phase_R=0.5)."""
        fps = 60.0
        stride_frames = 90
        pose = _make_synthetic_gait(n_frames=600, fps=fps,
                                    stride_frames=stride_frames,
                                    phase_L=0.0, phase_R=0.5)
        events = detect_events(pose, fps=fps, min_stride_s=0.5)
        if len(events.heel_strike_L) >= 1 and len(events.heel_strike_R) >= 1:
            diff = abs(int(events.heel_strike_L[0]) - int(events.heel_strike_R[0]))
            half_stride = stride_frames / 2
            # Cho phép sai số ±15% half-stride
            assert abs(diff - half_stride) <= 0.2 * half_stride

    def test_invalid_shape_raises(self):
        """pose_3d shape không đúng (T, J) thiếu chiều thứ 3 → ValueError."""
        pose_2d = np.ones((100, 23))  # thiếu chiều tọa độ
        with pytest.raises(ValueError):
            detect_events(pose_2d, fps=60.0)

    def test_negative_fps_raises(self):
        pose = _make_synthetic_gait()
        with pytest.raises(ValueError, match="fps"):
            detect_events(pose, fps=-1.0)

    def test_all_nan_pose_no_crash(self):
        """NaN trong toàn bộ pose không gây crash — trả về GaitEvents rỗng hoặc hợp lệ."""
        pose = np.full((50, 23, 3), np.nan)
        events = detect_events(pose, fps=60.0)
        assert isinstance(events, GaitEvents)

    def test_custom_joint_indices(self):
        """Hàm hoạt động đúng với chỉ số keypoint tùy chỉnh."""
        pose = _make_synthetic_gait(n_frames=200, n_joints=25, stride_frames=60)
        # Di chuyển heel vào chỉ số 23, 24
        pose[:, 23, 0] = 0.4 * np.sin(2 * np.pi * np.arange(200) / 60)
        pose[:, 24, 0] = 0.4 * np.sin(2 * np.pi * np.arange(200) / 60 + np.pi)
        events = detect_events(
            pose, fps=60.0,
            l_heel_idx=23, r_heel_idx=23,
            l_toe_idx=24, r_toe_idx=24,
            min_stride_s=0.5,
        )
        assert isinstance(events, GaitEvents)

    def test_fwd_axis_z(self):
        """Hỗ trợ fwd_axis=2 (Z làm trục tiến)."""
        pose = np.zeros((200, 23, 3))
        t = np.arange(200)
        pose[:, 19, 2] = 0.4 * np.sin(2 * np.pi * t / 60)
        pose[:, 22, 2] = 0.4 * np.sin(2 * np.pi * t / 60 + np.pi)
        pose[:, 17, 2] = -0.4 * np.sin(2 * np.pi * t / 60)
        pose[:, 20, 2] = -0.4 * np.sin(2 * np.pi * t / 60 + np.pi)
        events = detect_events(pose, fps=60.0, fwd_axis=2, min_stride_s=0.5)
        assert isinstance(events, GaitEvents)


# ====================================================================== #
# 3. stride_time() và step_length_ratio()
# ====================================================================== #

class TestStrideTime:
    """Kiểm tra stride_time()."""

    def test_stride_time_from_known_events(self):
        """3 HS cách nhau 90 frames @ 60fps → stride_time = 1.5s."""
        events = GaitEvents(
            heel_strike_L=np.array([0, 90, 180], dtype=np.int64),
            heel_strike_R=np.array([45, 135], dtype=np.int64),
        )
        st = stride_time(events, fps=60.0)
        np.testing.assert_allclose(st["stride_time_L"], [1.5, 1.5], atol=1e-6)
        np.testing.assert_allclose(st["stride_time_R"], [1.5], atol=1e-6)

    def test_empty_events_gives_empty_array(self):
        events = GaitEvents()
        st = stride_time(events, fps=60.0)
        assert len(st["stride_time_L"]) == 0
        assert len(st["stride_time_R"]) == 0

    def test_single_event_gives_empty_stride(self):
        events = GaitEvents(
            heel_strike_L=np.array([50], dtype=np.int64),
        )
        st = stride_time(events, fps=60.0)
        assert len(st["stride_time_L"]) == 0

    def test_output_keys_present(self):
        events = GaitEvents()
        st = stride_time(events, fps=60.0)
        assert "stride_time_L" in st
        assert "stride_time_R" in st


class TestStepLengthRatio:
    """Kiểm tra step_length_ratio()."""

    def test_equal_steps_gives_ratio_one(self):
        events = GaitEvents(
            heel_strike_L=np.array([0, 90, 180], dtype=np.int64),
            heel_strike_R=np.array([45, 135, 225], dtype=np.int64),
        )
        ratio = step_length_ratio(events)
        assert ratio == pytest.approx(1.0)

    def test_more_left_steps_gives_ratio_above_one(self):
        events = GaitEvents(
            heel_strike_L=np.array([0, 60, 120, 180], dtype=np.int64),
            heel_strike_R=np.array([30, 120], dtype=np.int64),
        )
        ratio = step_length_ratio(events)
        assert ratio == pytest.approx(2.0)

    def test_no_events_returns_none(self):
        events = GaitEvents()
        assert step_length_ratio(events) is None

    def test_one_side_empty_returns_none(self):
        events = GaitEvents(
            heel_strike_L=np.array([0, 90], dtype=np.int64),
        )
        assert step_length_ratio(events) is None


# ====================================================================== #
# 4. GaitEvents — dataclass
# ====================================================================== #

class TestGaitEvents:
    """Kiểm tra GaitEvents dataclass."""

    def test_default_empty_arrays(self):
        ev = GaitEvents()
        for arr in [ev.heel_strike_L, ev.heel_strike_R,
                    ev.toe_off_L, ev.toe_off_R]:
            assert len(arr) == 0

    def test_as_dict_has_all_keys(self):
        ev = GaitEvents()
        d = ev.as_dict()
        assert set(d.keys()) == {
            "heel_strike_L", "heel_strike_R", "toe_off_L", "toe_off_R"
        }

    def test_repr_shows_counts(self):
        ev = GaitEvents(
            heel_strike_L=np.array([1, 2, 3], dtype=np.int64),
            heel_strike_R=np.array([10, 20], dtype=np.int64),
        )
        r = repr(ev)
        assert "HS_L=3" in r
        assert "HS_R=2" in r

