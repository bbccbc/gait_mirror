"""
Unit tests cho mirror_optimization.py:
  - compute_orientation_flag(): xác định hướng nhìn per-frame.
  - correct_virt(): so sánh hướng detection vs projection, kích hoạt flip.
  - DecoupledLegLengths và KinematicSkeleton: bone lengths độc lập cho LLD.

Theo kiến trúc gốc Ajisafe et al., "Mirror-Aware Neural Humans", 3DV 2024.
"""

import numpy as np
import pytest

from gaitmirror.reconstruction.mirror_optimization import (
    compute_orientation_flag,
    correct_virt,
    DecoupledLegLengths,
)
from gaitmirror.reconstruction.skeleton_fk import KinematicSkeleton
from gaitmirror.pose.keypoint_mapping import (
    NOSE,
    L_EAR, R_EAR,
    L_SHOULDER, R_SHOULDER,
    L_HIP, R_HIP,
    L_KNEE, R_KNEE,
    L_ANKLE, R_ANKLE,
    L_HEEL, R_HEEL,
    L_BIG_TOE, R_BIG_TOE,
    WHOLEBODY_FLIP_PAIRS,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_kpts_frame(T: int, K: int = 23) -> np.ndarray:
    """Tạo mảng keypoints (T, K, 3) toàn NaN để điền từng test."""
    return np.zeros((T, K, 3), dtype=np.float64)


def _set_shoulders(frame: np.ndarray, ls_x: float, rs_x: float, conf: float = 0.9):
    """Đặt tọa độ x vai trái/phải với confidence cao."""
    frame[L_SHOULDER, 0] = ls_x
    frame[L_SHOULDER, 2] = conf
    frame[R_SHOULDER, 0] = rs_x
    frame[R_SHOULDER, 2] = conf


def _set_ear_nose(frame: np.ndarray, nose_x: float, r_ear_x: float, conf: float = 0.9):
    """Đặt tọa độ mũi và tai phải (dùng cho fallback orientation)."""
    frame[NOSE, 0] = nose_x
    frame[NOSE, 2] = conf
    frame[R_EAR, 0] = r_ear_x
    frame[R_EAR, 2] = conf


# ===========================================================================
# Nhóm 1: Tests compute_orientation_flag
# ===========================================================================

class TestComputeOrientationFlag:

    def test_right_shoulder_further_right_returns_true(self):
        """x_right_shoulder > x_left_shoulder -> flag = True."""
        kpts = _make_kpts_frame(T=1)
        _set_shoulders(kpts[0], ls_x=200.0, rs_x=250.0)  # RS > LS
        flags = compute_orientation_flag(kpts, conf_thresh=0.3)
        assert flags.shape == (1,)
        assert flags[0] == True

    def test_left_shoulder_further_right_returns_false(self):
        """x_right_shoulder <= x_left_shoulder -> flag = False."""
        kpts = _make_kpts_frame(T=1)
        _set_shoulders(kpts[0], ls_x=270.0, rs_x=200.0)  # LS > RS
        flags = compute_orientation_flag(kpts, conf_thresh=0.3)
        assert flags[0] == False

    def test_fallback_ear_nose_when_shoulder_low_conf(self):
        """Vai confidence thấp -> dùng fallback tai phải vs mũi."""
        kpts = _make_kpts_frame(T=1)
        # Vai conf < thresh
        _set_shoulders(kpts[0], ls_x=200.0, rs_x=250.0, conf=0.1)
        # Tai phải ở xa hơn mũi -> True
        _set_ear_nose(kpts[0], nose_x=300.0, r_ear_x=350.0, conf=0.9)
        flags = compute_orientation_flag(kpts, conf_thresh=0.3)
        assert flags[0] == True

    def test_multi_frame_vectorized(self):
        """Kiểm tra nhiều frame cùng lúc."""
        T = 5
        kpts = _make_kpts_frame(T=T)
        # Frame 0, 2, 4: RS > LS -> True
        # Frame 1, 3: LS > RS -> False
        for t in range(T):
            if t % 2 == 0:
                _set_shoulders(kpts[t], ls_x=200.0, rs_x=260.0)
            else:
                _set_shoulders(kpts[t], ls_x=260.0, rs_x=200.0)
        flags = compute_orientation_flag(kpts, conf_thresh=0.3)
        assert flags.shape == (T,)
        for t in range(T):
            assert flags[t] == (t % 2 == 0), f"Frame {t}: expected {t%2==0}, got {flags[t]}"

    def test_no_valid_keypoints_returns_false(self):
        """Không có keypoint hợp lệ -> False (không raise exception)."""
        kpts = np.zeros((3, 23, 3), dtype=np.float64)
        kpts[:, :, 2] = 0.0  # Tất cả conf = 0
        flags = compute_orientation_flag(kpts, conf_thresh=0.3)
        assert flags.shape == (3,)
        assert not np.any(flags)  # Tất cả False

    def test_2d_input_treated_as_single_frame(self):
        """Input (K, 3) được xử lý như 1 frame."""
        kpts_2d = np.zeros((23, 3), dtype=np.float64)
        kpts_2d[L_SHOULDER, :] = [200.0, 300.0, 0.9]
        kpts_2d[R_SHOULDER, :] = [250.0, 300.0, 0.9]
        flags = compute_orientation_flag(kpts_2d, conf_thresh=0.3)
        assert flags.shape == (1,)
        assert flags[0] == True


# ===========================================================================
# Nhóm 2: Tests correct_virt
# ===========================================================================

class TestCorrectVirt:

    def _build_det_proj_same_direction(self, T: int = 3):
        """
        Tạo detection và projection cùng hướng (RS > LS cả hai).
        where_virt == where_virt_proj -> flip_mask = False.
        """
        K = 23
        det = np.zeros((T, K, 3), dtype=np.float64)
        proj = np.zeros((T, K, 3), dtype=np.float64)
        for t in range(T):
            # Detection: RS > LS (flag = True)
            _set_shoulders(det[t], ls_x=200.0, rs_x=260.0)
            # Projection: RS > LS (flag = True)
            _set_shoulders(proj[t], ls_x=180.0, rs_x=240.0)
        return det, proj

    def _build_det_proj_opposite_direction(self, T: int = 3):
        """
        Tạo detection và projection ngược hướng.
        where_virt != where_virt_proj -> flip_mask = True.
        """
        K = 23
        det = np.zeros((T, K, 3), dtype=np.float64)
        proj = np.zeros((T, K, 3), dtype=np.float64)
        for t in range(T):
            # Detection: RS > LS (flag = True)
            _set_shoulders(det[t], ls_x=200.0, rs_x=260.0)
            # Projection: LS > RS (flag = False) — ngược hướng
            _set_shoulders(proj[t], ls_x=260.0, rs_x=200.0)
        return det, proj

    # -----------------------------------------------------------------
    # Test ca 1: Hướng trùng khớp -> flip_mask == False, không hoán đổi
    # -----------------------------------------------------------------
    def test_case1_same_direction_no_flip(self):
        """
        Ca 1 (Ajisafe): where_virt == where_virt_proj -> flip_mask == False.
        Tọa độ đầu ra PHẢI giống hệt detection đầu vào (không hoán đổi gì).
        """
        T = 3
        det, proj = self._build_det_proj_same_direction(T=T)

        # Điền dữ liệu phân biệt cho L_KNEE và R_KNEE để kiểm tra hoán đổi
        for t in range(T):
            det[t, L_KNEE, 0] = 100.0 + t  # Giá trị khác nhau per-frame
            det[t, R_KNEE, 0] = 200.0 + t
            det[t, L_KNEE, 2] = 0.8
            det[t, R_KNEE, 2] = 0.7

        kpts_aligned, weights_aligned, flipped_mask = correct_virt(
            virt_detection_kpts=det,
            virt_projected_kpts=proj,
            flip_pairs=WHOLEBODY_FLIP_PAIRS,
            conf_thresh=0.3,
        )

        # flip_mask phải toàn False
        assert flipped_mask.shape == (T,)
        assert not np.any(flipped_mask), (
            f"Kỳ vọng flip_mask == False tất cả, nhận: {flipped_mask}"
        )

        # kpts_aligned phải giống hệt det (không hoán đổi)
        np.testing.assert_array_equal(
            kpts_aligned, det,
            err_msg="Khi hướng trùng, kpts_aligned phải == detection đầu vào (không flip)",
        )

        # weights_aligned phải == cột conf của det
        np.testing.assert_array_equal(
            weights_aligned, det[:, :, 2],
            err_msg="weights_aligned phải == conf channel của detection khi không flip",
        )

    # -----------------------------------------------------------------
    # Test ca 2: Hướng ngược -> flip_mask == True, hoán đổi đúng keypoints
    # -----------------------------------------------------------------
    def test_case2_opposite_direction_flip_applied(self):
        """
        Ca 2 (Ajisafe): where_virt != where_virt_proj -> flip_mask == True.
        Các cặp khớp đối xứng (vai, hông, gối, cổ chân, bàn chân) phải được hoán đổi.
        """
        T = 2
        K = 23
        det = np.zeros((T, K, 3), dtype=np.float64)
        proj = np.zeros((T, K, 3), dtype=np.float64)

        for t in range(T):
            # Detection: RS > LS (flag = True)
            _set_shoulders(det[t], ls_x=200.0, rs_x=260.0)
            # Projection: LS > RS (flag = False) — ngược hướng -> sẽ flip
            _set_shoulders(proj[t], ls_x=260.0, rs_x=200.0)

        # Điền giá trị phân biệt cho các khớp đối xứng
        for t in range(T):
            # Gối
            det[t, L_KNEE, 0] = 50.0;  det[t, L_KNEE, 2] = 0.85
            det[t, R_KNEE, 0] = 150.0; det[t, R_KNEE, 2] = 0.75
            # Cổ chân
            det[t, L_ANKLE, 0] = 55.0;  det[t, L_ANKLE, 2] = 0.80
            det[t, R_ANKLE, 0] = 155.0; det[t, R_ANKLE, 2] = 0.70
            # Gót chân
            det[t, L_HEEL, 0] = 56.0;  det[t, L_HEEL, 2] = 0.78
            det[t, R_HEEL, 0] = 156.0; det[t, R_HEEL, 2] = 0.68

        kpts_aligned, weights_aligned, flipped_mask = correct_virt(
            virt_detection_kpts=det,
            virt_projected_kpts=proj,
            flip_pairs=WHOLEBODY_FLIP_PAIRS,
            conf_thresh=0.3,
        )

        # flip_mask phải toàn True
        assert flipped_mask.shape == (T,)
        assert np.all(flipped_mask), (
            f"Kỳ vọng flip_mask == True tất cả, nhận: {flipped_mask}"
        )

        # Kiểm tra hoán đổi gối (L_KNEE <-> R_KNEE)
        for t in range(T):
            # Sau flip: kpts_aligned[t, L_KNEE] = det[t, R_KNEE] và ngược lại
            assert kpts_aligned[t, L_KNEE, 0] == pytest.approx(150.0), \
                f"Frame {t}: L_KNEE.x sau flip phải = R_KNEE.x gốc (150.0)"
            assert kpts_aligned[t, R_KNEE, 0] == pytest.approx(50.0), \
                f"Frame {t}: R_KNEE.x sau flip phải = L_KNEE.x gốc (50.0)"

            # Kiểm tra hoán đổi weights gối
            assert weights_aligned[t, L_KNEE] == pytest.approx(0.75), \
                f"Frame {t}: weight L_KNEE sau flip phải = 0.75 (R_KNEE gốc)"
            assert weights_aligned[t, R_KNEE] == pytest.approx(0.85), \
                f"Frame {t}: weight R_KNEE sau flip phải = 0.85 (L_KNEE gốc)"

            # Kiểm tra hoán đổi cổ chân
            assert kpts_aligned[t, L_ANKLE, 0] == pytest.approx(155.0)
            assert kpts_aligned[t, R_ANKLE, 0] == pytest.approx(55.0)

            # Kiểm tra hoán đổi gót chân
            assert kpts_aligned[t, L_HEEL, 0] == pytest.approx(156.0)
            assert kpts_aligned[t, R_HEEL, 0] == pytest.approx(56.0)

    # -----------------------------------------------------------------
    # Test: Input gốc không bị biến đổi (pure copy semantics)
    # -----------------------------------------------------------------
    def test_input_not_mutated(self):
        """
        correct_virt() phải không làm thay đổi mảng virt_detection_kpts gốc.
        (Bảo toàn raw detection như kiến trúc yêu cầu.)
        """
        T = 2
        det, proj = self._build_det_proj_opposite_direction(T=T)
        det_original = det.copy()

        correct_virt(
            virt_detection_kpts=det,
            virt_projected_kpts=proj,
            flip_pairs=WHOLEBODY_FLIP_PAIRS,
        )

        # det phải vẫn == bản gốc
        np.testing.assert_array_equal(
            det, det_original,
            err_msg="correct_virt() không được làm biến đổi mảng detection đầu vào!",
        )

    # -----------------------------------------------------------------
    # Test: Projection với 2 channel (không có conf)
    # -----------------------------------------------------------------
    def test_projection_2channel_no_error(self):
        """Projection chỉ có (x, y) không có conf cũng hoạt động."""
        T = 2
        K = 23
        det = np.zeros((T, K, 3), dtype=np.float64)
        proj_2ch = np.zeros((T, K, 2), dtype=np.float64)

        for t in range(T):
            _set_shoulders(det[t], ls_x=200.0, rs_x=260.0)
            proj_2ch[t, L_SHOULDER, 0] = 200.0
            proj_2ch[t, R_SHOULDER, 0] = 260.0

        # Không nên raise exception
        kpts_aligned, weights_aligned, flipped_mask = correct_virt(
            virt_detection_kpts=det,
            virt_projected_kpts=proj_2ch,
            flip_pairs=WHOLEBODY_FLIP_PAIRS,
        )
        assert kpts_aligned.shape == (T, K, 3)
        assert flipped_mask.shape == (T,)

    # -----------------------------------------------------------------
    # Test: Mixed frames (một số flip, một số không)
    # -----------------------------------------------------------------
    def test_mixed_flip_per_frame(self):
        """
        Một số frame cùng hướng (no flip), một số ngược (flip).
        flipped_mask phải đúng per-frame.
        """
        T = 4
        K = 23
        det = np.zeros((T, K, 3), dtype=np.float64)
        proj = np.zeros((T, K, 3), dtype=np.float64)

        # Frame 0, 2: cùng hướng (RS > LS cả det và proj) -> no flip
        # Frame 1, 3: ngược hướng -> flip
        for t in range(T):
            _set_shoulders(det[t], ls_x=200.0, rs_x=260.0)  # RS > LS (det luôn True)
            if t % 2 == 0:
                _set_shoulders(proj[t], ls_x=180.0, rs_x=240.0)  # RS > LS -> same -> no flip
            else:
                _set_shoulders(proj[t], ls_x=260.0, rs_x=200.0)  # LS > RS -> opposite -> flip

        kpts_aligned, _, flipped_mask = correct_virt(
            virt_detection_kpts=det,
            virt_projected_kpts=proj,
            flip_pairs=WHOLEBODY_FLIP_PAIRS,
        )

        expected_flip = np.array([False, True, False, True])
        np.testing.assert_array_equal(
            flipped_mask, expected_flip,
            err_msg=f"flipped_mask per-frame sai: got {flipped_mask}, expected {expected_flip}",
        )


# ===========================================================================
# Nhóm 3: Tests bone lengths độc lập (LLD)
# ===========================================================================

class TestDecoupledBoneLengths:

    def test_independent_initialization(self):
        """4 biến bone lengths có thể khởi tạo với giá trị hoàn toàn khác nhau."""
        ll = DecoupledLegLengths(
            femur_l=420.0,
            femur_r=400.0,
            tibia_l=360.0,
            tibia_r=345.0,
        )
        assert ll.bone_length_femur_l == pytest.approx(420.0)
        assert ll.bone_length_femur_r == pytest.approx(400.0)
        assert ll.bone_length_tibia_l == pytest.approx(360.0)
        assert ll.bone_length_tibia_r == pytest.approx(345.0)

    def test_lld_calculation_correct(self):
        """Tính toán chênh lệch chiều dài chi đúng."""
        ll = DecoupledLegLengths(
            femur_l=420.0, femur_r=400.0,
            tibia_l=360.0, tibia_r=345.0,
        )
        assert ll.lld_femur_mm == pytest.approx(20.0)   # 420 - 400
        assert ll.lld_tibia_mm == pytest.approx(15.0)   # 360 - 345
        assert ll.lld_total_mm == pytest.approx(35.0)   # 20 + 15

    def test_as_array_roundtrip(self):
        """Chuyển sang mảng và cập nhật lại từ mảng đều nhất quán."""
        ll = DecoupledLegLengths(410.0, 390.0, 355.0, 340.0)
        arr = ll.as_array()
        assert arr.shape == (4,)
        assert arr.tolist() == pytest.approx([410.0, 390.0, 355.0, 340.0])

        # Cập nhật từ mảng
        ll2 = DecoupledLegLengths()
        ll2.update_from_array(arr)
        assert ll2.bone_length_femur_l == pytest.approx(410.0)
        assert ll2.bone_length_femur_r == pytest.approx(390.0)
        assert ll2.bone_length_tibia_l == pytest.approx(355.0)
        assert ll2.bone_length_tibia_r == pytest.approx(340.0)

    def test_symmetry_check(self):
        """is_symmetric() đúng khi chênh lệch < tol."""
        ll_sym = DecoupledLegLengths(400.0, 400.5, 350.0, 350.0)
        ll_asym = DecoupledLegLengths(420.0, 400.0, 360.0, 345.0)
        assert ll_sym.is_symmetric(tol_mm=1.0) == True
        assert ll_asym.is_symmetric(tol_mm=1.0) == False

    def test_negative_lld_right_longer(self):
        """LLD âm khi chân phải dài hơn."""
        ll = DecoupledLegLengths(femur_l=390.0, femur_r=410.0,
                                  tibia_l=340.0, tibia_r=360.0)
        assert ll.lld_femur_mm == pytest.approx(-20.0)
        assert ll.lld_total_mm == pytest.approx(-40.0)


class TestKinematicSkeletonLLD:
    """
    Test ca 3 (Ajisafe): Kiểm tra tính độc lập bone lengths chi dưới
    trong skeleton forward kinematics.
    """

    def test_femur_independence_left_right(self):
        """
        Thay đổi femur_l KHÔNG được ảnh hưởng đến femur_r và ngược lại.
        Đây là ràng buộc cốt lõi để đo LLD chính xác.
        """
        ll = DecoupledLegLengths(femur_l=420.0, femur_r=400.0,
                                  tibia_l=350.0, tibia_r=350.0)
        skel = KinematicSkeleton(leg_lengths=ll)

        # Joint 4 = l_knee, Joint 5 = r_knee
        # Joint 1 = l_hip,  Joint 2 = r_hip
        pos = skel.joint_positions
        l_femur_actual = abs(pos[4, 1] - pos[1, 1])  # l_hip y - l_knee y
        r_femur_actual = abs(pos[5, 1] - pos[2, 1])  # r_hip y - r_knee y

        assert l_femur_actual == pytest.approx(420.0, abs=0.1), \
            f"Chiều dài xương đùi trái phải = 420mm, nhận: {l_femur_actual:.1f}"
        assert r_femur_actual == pytest.approx(400.0, abs=0.1), \
            f"Chiều dài xương đùi phải phải = 400mm, nhận: {r_femur_actual:.1f}"

        # Hai bên phải KHÁC NHAU
        assert l_femur_actual != pytest.approx(r_femur_actual, abs=0.1), \
            "femur_l và femur_r phải độc lập (không ép bằng nhau)"

    def test_tibia_independence_left_right(self):
        """Tương tự cho cẳng chân."""
        ll = DecoupledLegLengths(femur_l=400.0, femur_r=400.0,
                                  tibia_l=360.0, tibia_r=340.0)
        skel = KinematicSkeleton(leg_lengths=ll)

        pos = skel.joint_positions
        # Joint 7 = l_ankle, Joint 4 = l_knee
        # Joint 8 = r_ankle, Joint 5 = r_knee
        l_tibia_actual = abs(pos[7, 1] - pos[4, 1])
        r_tibia_actual = abs(pos[8, 1] - pos[5, 1])

        assert l_tibia_actual == pytest.approx(360.0, abs=0.1)
        assert r_tibia_actual == pytest.approx(340.0, abs=0.1)
        assert l_tibia_actual != pytest.approx(r_tibia_actual, abs=0.1)

    def test_optimizer_update_preserves_independence(self):
        """
        Sau khi optimizer cập nhật params (4 biến), tính độc lập được bảo toàn.
        """
        skel = KinematicSkeleton()
        # Giả lập optimizer đề xuất params mới với 4 giá trị khác nhau
        new_params = np.array([415.0, 395.0, 358.0, 342.0])
        skel.set_bone_lengths_lower_limb(new_params)

        # Kiểm tra cập nhật đúng
        assert skel.leg_lengths.bone_length_femur_l == pytest.approx(415.0)
        assert skel.leg_lengths.bone_length_femur_r == pytest.approx(395.0)
        assert skel.leg_lengths.bone_length_tibia_l == pytest.approx(358.0)
        assert skel.leg_lengths.bone_length_tibia_r == pytest.approx(342.0)

        # Tính LLD
        lld_f, lld_t, lld_tot = skel.compute_lld_mm()
        assert lld_f == pytest.approx(20.0)   # 415 - 395
        assert lld_t == pytest.approx(16.0)   # 358 - 342
        assert lld_tot == pytest.approx(36.0) # 20 + 16

    def test_get_bone_lengths_lower_limb_array(self):
        """get_bone_lengths_lower_limb() trả về đúng 4 tham số dạng array."""
        ll = DecoupledLegLengths(410.0, 390.0, 355.0, 340.0)
        skel = KinematicSkeleton(leg_lengths=ll)
        arr = skel.get_bone_lengths_lower_limb()
        assert arr.shape == (4,)
        assert arr.tolist() == pytest.approx([410.0, 390.0, 355.0, 340.0])

    def test_symmetric_skeleton_has_zero_lld(self):
        """Bộ xương đối xứng hoàn toàn -> LLD = 0."""
        ll = DecoupledLegLengths(400.0, 400.0, 350.0, 350.0)
        skel = KinematicSkeleton(leg_lengths=ll)
        lld_f, lld_t, lld_tot = skel.compute_lld_mm()
        assert lld_tot == pytest.approx(0.0, abs=1e-6)

    def test_tpose_positions_consistent_with_bone_lengths(self):
        """
        Vị trí T-pose phải nhất quán với 4 bone lengths được đặt.
        Chiều dài thực tế tính từ hiệu tọa độ y phải == bone_length tương ứng.
        """
        femur_l, femur_r = 430.0, 405.0
        tibia_l, tibia_r = 365.0, 348.0
        ll = DecoupledLegLengths(femur_l, femur_r, tibia_l, tibia_r)
        skel = KinematicSkeleton(leg_lengths=ll)
        pos = skel.joint_positions

        # Đùi trái: l_hip (1) -> l_knee (4)
        assert abs(pos[4, 1] - pos[1, 1]) == pytest.approx(femur_l, abs=0.01)
        # Đùi phải: r_hip (2) -> r_knee (5)
        assert abs(pos[5, 1] - pos[2, 1]) == pytest.approx(femur_r, abs=0.01)
        # Cẳng trái: l_knee (4) -> l_ankle (7)
        assert abs(pos[7, 1] - pos[4, 1]) == pytest.approx(tibia_l, abs=0.01)
        # Cẳng phải: r_knee (5) -> r_ankle (8)
        assert abs(pos[8, 1] - pos[5, 1]) == pytest.approx(tibia_r, abs=0.01)

