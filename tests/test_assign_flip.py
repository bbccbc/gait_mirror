"""
Unit tests cho module assign_flip.py
Bao phủ các ca biên: NaN, người thứ ba, thiếu người, hòa thân, và tính lũy hoàn của flip.

KIẾN TRÚC QUAN TRỌNG (Ajisafe et al., 3DV 2024):
  - clean_and_associate() KHÔNG được gọi flip_wholebody() — tầng 2D lưu raw.
  - AssociatedFrame.mirror_kpts_raw phải == raw detection (KHÔNG hoán đổi).
  - Tầng 3D (mirror_optimization.correct_virt) tự quyết định flip per-frame.
"""

import numpy as np
import pytest

from gaitmirror.pose.assign_flip import (
    AssociatedFrame,
    clean_and_associate,
    drop_bystanders_frame,
    flip_wholebody,
)
from gaitmirror.pose.keypoint_mapping import (
    L_SHOULDER, R_SHOULDER, L_HIP, R_HIP,
    L_KNEE, R_KNEE, L_HEEL, R_HEEL
)


def _create_mock_kpts(torso_length: float, y_offset: float = 100.0) -> np.ndarray:
    """Tạo pose giả lập với chiều dài thân xác định."""
    kpts = np.zeros((133, 3), dtype=np.float32)
    kpts[:, 2] = 0.9  # Conf mặc định cao
    # Cổ nằm ở giữa vai
    kpts[L_SHOULDER, :2] = [200.0, y_offset]
    kpts[R_SHOULDER, :2] = [240.0, y_offset]
    # Hông chậu nằm ở y_offset + torso_length
    kpts[L_HIP, :2] = [210.0, y_offset + torso_length]
    kpts[R_HIP, :2] = [230.0, y_offset + torso_length]
    # Gối và gót chân
    kpts[L_KNEE, :2] = [210.0, y_offset + torso_length + 50.0]
    kpts[R_KNEE, :2] = [230.0, y_offset + torso_length + 50.0]
    kpts[L_HEEL, :2] = [210.0, y_offset + torso_length + 100.0]
    kpts[R_HEEL, :2] = [230.0, y_offset + torso_length + 100.0]
    return kpts


# -------------------------------------------------------------------------
# Test 1: Pose có thân chứa NaN -> Trả cờ QC THIEU_DIEM_THAN, không gán sai
# -------------------------------------------------------------------------
def test_nan_pose_raises_flag():
    kpts_valid = _create_mock_kpts(torso_length=150.0)
    kpts_nan = _create_mock_kpts(torso_length=100.0)
    kpts_nan[L_SHOULDER, :] = np.nan  # Làm hỏng điểm vai

    box_valid = np.array([150, 100, 300, 450])
    box_nan = np.array([350, 100, 500, 400])

    res, flags = clean_and_associate(
        kpts_a=kpts_nan,
        kpts_b=kpts_valid,
        box_a=box_nan,
        box_b=box_valid,
        iou_thresh=0.05,
    )
    assert res is None
    assert "THIEU_DIEM_THAN" in flags


# -------------------------------------------------------------------------
# Test 2: Ba người xuất hiện -> Bắt cờ CO_NGUOI_THU_BA và loại đúng NaN
# -------------------------------------------------------------------------
def test_three_persons_raises_flag_and_filters():
    person_adult = _create_mock_kpts(torso_length=220.0)  # Người lớn / Người thứ ba
    person_child_real = _create_mock_kpts(torso_length=140.0)  # Bé thật
    person_child_mirror = _create_mock_kpts(torso_length=110.0)  # Bé bóng gương

    kpts_list = [person_child_real, person_adult, person_child_mirror]
    bboxes = np.array([
        [100, 100, 200, 400],
        [400, 50, 550, 550],
        [600, 120, 700, 380]
    ])

    selected_kpts, selected_bboxes, flags = drop_bystanders_frame(
        kpts_list, bboxes, top_k=2
    )
    assert "CO_NGUOI_THU_BA" in flags
    assert len(selected_kpts) == 2
    # Người lớn (thân dài nhất) và bé thật sẽ bị nhấc ra nếu chỉ căn cứ vào độ dài
    # Cờ CO_NGUOI_THU_BA báo cho pipeline biết khung này không an toàn


# -------------------------------------------------------------------------
# Test 3: Một người duy nhất -> Trả cờ THIEU_NGUOI, không văng lỗi
# -------------------------------------------------------------------------
def test_single_person_raises_flag():
    kpts_single = _create_mock_kpts(torso_length=150.0)
    bboxes = np.array([[100, 100, 250, 450]])

    selected_kpts, selected_bboxes, flags = drop_bystanders_frame(
        [kpts_single], bboxes, top_k=2
    )
    assert "THIEU_NGUOI" in flags
    assert len(selected_kpts) == 1


# -------------------------------------------------------------------------
# Test 4: Hai thân bằng nhau / gần bằng nhau -> Bắt cờ mơ hồ
# -------------------------------------------------------------------------
def test_equal_and_close_torso_ratios():
    kpts_a = _create_mock_kpts(torso_length=150.0)
    kpts_b = _create_mock_kpts(torso_length=150.0)  # Bằng nhau tuyệt đối
    box_a = np.array([100, 100, 250, 450])
    box_b = np.array([300, 100, 450, 450])

    res, flags = clean_and_associate(
        kpts_a, kpts_b, box_a, box_b, iou_thresh=0.05, ratio_thresh=0.95
    )
    assert res is not None
    assert "THAN_BANG_NHAU" in flags
    assert "THAN_GAN_BANG_NHAU" in flags
    assert res.torso_ratio == 1.0


# -------------------------------------------------------------------------
# Test 5: Lũy hoàn (Involutory) của flip_wholebody: áp dụng 2 lần ra gốc
# -------------------------------------------------------------------------
def test_flip_wholebody_involution():
    kpts_orig = np.random.rand(133, 3).astype(np.float32)
    kpts_flipped_once = flip_wholebody(kpts_orig)
    kpts_flipped_twice = flip_wholebody(kpts_flipped_once)

    # Đảm bảo khớp trái và phải bị hoán đổi ở lần 1
    assert not np.allclose(kpts_orig[L_KNEE], kpts_flipped_once[L_KNEE])
    assert np.allclose(kpts_orig[L_KNEE], kpts_flipped_once[R_KNEE])

    # Đảm bảo lật 2 lần khôi phục 100% dữ liệu ban đầu
    assert np.allclose(kpts_orig, kpts_flipped_twice)


# -------------------------------------------------------------------------
# Test 6: KIẾN TRÚC — mirror_kpts_raw phải bảo toàn nguyên vẹn tọa độ RAW
# Xác nhận clean_and_associate() KHÔNG gọi flip_wholebody() nội bộ.
# -------------------------------------------------------------------------
def test_mirror_kpts_raw_is_unmodified():
    """
    Đây là test kiến trúc cốt lõi theo Ajisafe et al. (3DV 2024):
      - Tầng 2D KHÔNG được flip L/R lên raw detection.
      - mirror_kpts_raw phải == bản sao nguyên trạng của kpts input (bóng gương).
      - Việc so sánh hướng và quyết định flip thuộc trách nhiệm tầng 3D (correct_virt).
    """
    # Tạo pose thật (thân dài hơn) và pose bóng gương (thân ngắn hơn)
    kpts_real_input = _create_mock_kpts(torso_length=160.0)
    kpts_mirror_input = _create_mock_kpts(torso_length=100.0)

    # Phân biệt hai pose bằng giá trị ngẫu nhiên riêng biệt
    # (để không nhầm lẫn khi so sánh)
    rng = np.random.default_rng(42)
    kpts_real_input[:, 0] += rng.uniform(0, 5, 133).astype(np.float32)
    kpts_mirror_input[:, 0] += rng.uniform(100, 105, 133).astype(np.float32)

    box_real = np.array([100.0, 100.0, 300.0, 500.0])
    box_mirror = np.array([400.0, 100.0, 600.0, 460.0])

    res, flags = clean_and_associate(
        kpts_a=kpts_real_input,
        kpts_b=kpts_mirror_input,
        box_a=box_real,
        box_b=box_mirror,
        iou_thresh=0.05,
        ratio_thresh=0.95,
    )

    assert res is not None, "Phân định phải thành công với 2 pose hợp lệ"
    assert res.mirror_kpts_raw is not None, "mirror_kpts_raw không được None"

    # KHẲNG ĐỊNH TRUNG TÂM: mirror_kpts_raw phải == kpts_mirror_input (bóng gương)
    # và KHÔNG phải kpts_real_input (người thật)
    # Vì kpts_mirror_input có torso_length nhỏ hơn nên được phân định là gương.
    np.testing.assert_array_equal(
        res.mirror_kpts_raw,
        kpts_mirror_input,
        err_msg=(
            "mirror_kpts_raw phải bảo toàn nguyên trạng raw detection của bóng gương. "
            "clean_and_associate() không được gọi flip_wholebody() nội bộ!"
        ),
    )

    # Đảm bảo real_kpts cũng đúng (không nhầm thật/gương)
    np.testing.assert_array_equal(
        res.real_kpts,
        kpts_real_input,
        err_msg="real_kpts phải là pose có thân dài hơn (người thật).",
    )


# -------------------------------------------------------------------------
# Test 7: mirror_kpts_raw khác mirror_kpts đã flip (dùng flip_wholebody thủ công)
# Đảm bảo tầng 3D nhận được raw và CÓ THỂ tự quyết định flip.
# -------------------------------------------------------------------------
def test_raw_vs_flipped_are_different():
    """
    Kiểm tra rằng mirror_kpts_raw (raw detection) khác với kết quả
    sau khi áp dụng flip_wholebody() — xác nhận tầng 3D có dữ liệu
    đúng để thực hiện correct_virt().
    """
    kpts_real = _create_mock_kpts(torso_length=150.0)
    kpts_mirror_raw = _create_mock_kpts(torso_length=90.0)
    # Làm cho kpts_mirror_raw có L_KNEE và R_KNEE khác giá trị rõ ràng
    kpts_mirror_raw[L_KNEE, 0] = 50.0
    kpts_mirror_raw[R_KNEE, 0] = 70.0

    box_real = np.array([100.0, 100.0, 300.0, 500.0])
    box_mirror = np.array([400.0, 100.0, 600.0, 460.0])

    res, _ = clean_and_associate(
        kpts_a=kpts_real,
        kpts_b=kpts_mirror_raw,
        box_a=box_real,
        box_b=box_mirror,
        iou_thresh=0.05,
    )

    assert res is not None
    raw = res.mirror_kpts_raw

    # Thực hiện flip thủ công để có phiên bản đã hoán đổi
    flipped = flip_wholebody(raw)

    # raw và flipped phải KHÁC nhau (vì L_KNEE != R_KNEE)
    assert not np.allclose(raw[L_KNEE], raw[R_KNEE]), \
        "Setup test sai: L_KNEE và R_KNEE phải khác nhau để test có ý nghĩa"
    assert not np.allclose(raw, flipped), \
        "raw và flipped phải khác nhau (flip_wholebody chưa được áp dụng vào raw)"

    # Nhưng nếu flip 2 lần sẽ ra lại raw
    assert np.allclose(raw, flip_wholebody(flipped)), \
        "Áp dụng flip 2 lần phải trả về dữ liệu gốc"
