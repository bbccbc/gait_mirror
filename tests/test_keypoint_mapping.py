"""Tests cho gaitmirror.pose.keypoint_mapping.

Kiểm tra:
- Các hằng số chỉ số COCO-WholeBody 133 đúng giá trị
- WHOLEBODY_FLIP_PAIRS: đủ cặp, không trùng, lũy hoán (involution)
- compute_neck_and_root(): đầu ra đúng hình học
"""

import numpy as np
import pytest

from gaitmirror.pose.keypoint_mapping import (
    NOSE,
    L_EYE, R_EYE,
    L_EAR, R_EAR,
    L_SHOULDER, R_SHOULDER,
    L_ELBOW, R_ELBOW,
    L_WRIST, R_WRIST,
    L_HIP, R_HIP,
    L_KNEE, R_KNEE,
    L_ANKLE, R_ANKLE,
    L_BIG_TOE, L_SMALL_TOE, L_HEEL,
    R_BIG_TOE, R_SMALL_TOE, R_HEEL,
    WHOLEBODY_FLIP_PAIRS,
    compute_neck_and_root,
)


# ====================================================================== #
# 1. Hằng số chỉ số COCO-WholeBody
# ====================================================================== #

class TestKeypointConstants:
    """Kiểm tra các chỉ số COCO-WholeBody 133 đúng theo đặc tả."""

    def test_nose_index(self):
        assert NOSE == 0

    def test_eye_indices(self):
        assert L_EYE == 1
        assert R_EYE == 2

    def test_ear_indices(self):
        """Tai trái = 3, tai phải = 4 (đặc tả task A)."""
        assert L_EAR == 3
        assert R_EAR == 4

    def test_shoulder_indices(self):
        """Vai trái = 5, vai phải = 6."""
        assert L_SHOULDER == 5
        assert R_SHOULDER == 6

    def test_elbow_wrist_indices(self):
        assert L_ELBOW == 7
        assert R_ELBOW == 8
        assert L_WRIST == 9
        assert R_WRIST == 10

    def test_hip_indices(self):
        """Hông trái = 11, hông phải = 12."""
        assert L_HIP == 11
        assert R_HIP == 12

    def test_knee_indices(self):
        """Gối trái = 13, gối phải = 14."""
        assert L_KNEE == 13
        assert R_KNEE == 14

    def test_ankle_indices(self):
        """Cổ chân trái = 15, cổ chân phải = 16."""
        assert L_ANKLE == 15
        assert R_ANKLE == 16

    def test_foot_indices(self):
        """Ngón cái, ngón út, gót — trái và phải."""
        # Trái
        assert L_BIG_TOE == 17
        assert L_SMALL_TOE == 18
        assert L_HEEL == 19
        # Phải
        assert R_BIG_TOE == 20
        assert R_SMALL_TOE == 21
        assert R_HEEL == 22

    def test_body_keypoints_are_contiguous_17(self):
        """17 khớp body COCO là chỉ số 0–16 liên tục (không có lỗ)."""
        body_indices = [
            NOSE, L_EYE, R_EYE, L_EAR, R_EAR,
            L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW,
            L_WRIST, R_WRIST, L_HIP, R_HIP,
            L_KNEE, R_KNEE, L_ANKLE, R_ANKLE,
        ]
        assert sorted(body_indices) == list(range(17))

    def test_left_right_symmetry_convention(self):
        """Luôn Left < Right với các cặp body — quy ước COCO."""
        pairs = [
            (L_EYE, R_EYE), (L_EAR, R_EAR),
            (L_SHOULDER, R_SHOULDER), (L_ELBOW, R_ELBOW),
            (L_WRIST, R_WRIST), (L_HIP, R_HIP),
            (L_KNEE, R_KNEE), (L_ANKLE, R_ANKLE),
            (L_BIG_TOE, R_BIG_TOE), (L_SMALL_TOE, R_SMALL_TOE),
            (L_HEEL, R_HEEL),
        ]
        for left, right in pairs:
            assert left < right, f"Kỳ vọng L({left}) < R({right})"


# ====================================================================== #
# 2. WHOLEBODY_FLIP_PAIRS — cấu trúc và tính đúng đắn
# ====================================================================== #

class TestFlipPairs:
    """Kiểm tra WHOLEBODY_FLIP_PAIRS."""

    def test_is_list_of_tuples(self):
        assert isinstance(WHOLEBODY_FLIP_PAIRS, list)
        for pair in WHOLEBODY_FLIP_PAIRS:
            assert isinstance(pair, tuple) and len(pair) == 2

    def test_no_self_pairing(self):
        """Không có cặp (i, i)."""
        for l, r in WHOLEBODY_FLIP_PAIRS:
            assert l != r, f"Cặp tự ánh xạ: ({l}, {r})"

    def test_no_duplicate_pairs(self):
        """Không có cặp trùng lặp (kể cả đảo ngược)."""
        seen = set()
        for l, r in WHOLEBODY_FLIP_PAIRS:
            key = (min(l, r), max(l, r))
            assert key not in seen, f"Cặp trùng: {key}"
            seen.add(key)

    def test_indices_in_range_133(self):
        """Tất cả chỉ số phải trong [0, 132]."""
        for l, r in WHOLEBODY_FLIP_PAIRS:
            assert 0 <= l <= 132, f"Chỉ số trái {l} ngoài phạm vi"
            assert 0 <= r <= 132, f"Chỉ số phải {r} ngoài phạm vi"

    def test_involution_property(self):
        """Áp dụng flip hai lần phải trả về trạng thái ban đầu (lũy hoán).

        Nếu (l, r) ∈ pairs thì KHÔNG được có (r, l) hay (r, x≠l) trong pairs.
        """
        # Xây dựng bảng forward mapping
        mapping = {}
        for l, r in WHOLEBODY_FLIP_PAIRS:
            assert l not in mapping, f"Chỉ số {l} xuất hiện nhiều lần phía trái"
            assert r not in mapping, f"Chỉ số {r} xuất hiện nhiều lần"
            mapping[l] = r
            mapping[r] = l
        # Kiểm tra lũy hoán: mapping[mapping[k]] == k
        for k, v in mapping.items():
            assert mapping[v] == k

    def test_body_pairs_included(self):
        """Các cặp body cơ bản phải có trong flip_pairs."""
        body_pairs = {(L_EAR, R_EAR), (L_SHOULDER, R_SHOULDER),
                      (L_HIP, R_HIP), (L_KNEE, R_KNEE), (L_ANKLE, R_ANKLE),
                      (L_HEEL, R_HEEL)}
        pair_set = set(WHOLEBODY_FLIP_PAIRS)
        for pair in body_pairs:
            assert pair in pair_set, f"Thiếu cặp {pair}"

    def test_hand_pairs_count(self):
        """21 cặp tay (91–111 trái, 112–132 phải) phải hiện diện."""
        hand_pairs = {(91 + i, 112 + i) for i in range(21)}
        pair_set = set(WHOLEBODY_FLIP_PAIRS)
        for pair in hand_pairs:
            assert pair in pair_set, f"Thiếu cặp tay {pair}"

    def test_minimum_61_pairs(self):
        """Phải có ít nhất 61 cặp (11 body+foot + 21 tay + 29 mặt)."""
        assert len(WHOLEBODY_FLIP_PAIRS) >= 61


# ====================================================================== #
# 3. compute_neck_and_root()
# ====================================================================== #

class TestComputeNeckAndRoot:
    """Kiểm tra hàm compute_neck_and_root."""

    def _make_kpts(self, l_shoulder, r_shoulder, l_hip, r_hip, ndim=3):
        """Tạo mảng keypoints tối thiểu."""
        kpts = np.zeros((133, ndim), dtype=np.float64)
        kpts[L_SHOULDER, :2] = l_shoulder
        kpts[R_SHOULDER, :2] = r_shoulder
        kpts[L_HIP, :2] = l_hip
        kpts[R_HIP, :2] = r_hip
        return kpts

    def test_neck_is_midpoint_of_shoulders(self):
        kpts = self._make_kpts([2.0, 0.0], [4.0, 0.0], [3.0, -1.0], [5.0, -1.0])
        neck, _ = compute_neck_and_root(kpts)
        np.testing.assert_allclose(neck, [3.0, 0.0], atol=1e-9)

    def test_root_is_midpoint_of_hips(self):
        kpts = self._make_kpts([2.0, 0.0], [4.0, 0.0], [1.0, -2.0], [3.0, -2.0])
        _, root = compute_neck_and_root(kpts)
        np.testing.assert_allclose(root, [2.0, -2.0], atol=1e-9)

    def test_symmetric_pose_gives_centered_neck(self):
        """Người đứng thẳng đối xứng: neck/root trên trục giữa."""
        kpts = self._make_kpts([-0.3, 1.5], [0.3, 1.5], [-0.15, 0.9], [0.15, 0.9])
        neck, root = compute_neck_and_root(kpts)
        assert abs(neck[0]) < 1e-9  # x = 0 (giữa)
        assert abs(root[0]) < 1e-9

    def test_works_with_2d_keypoints(self):
        """Hàm hoạt động với kpts shape (133, 2)."""
        kpts = self._make_kpts([0.0, 0.0], [2.0, 0.0], [0.0, -1.0], [2.0, -1.0], ndim=2)
        neck, root = compute_neck_and_root(kpts)
        assert neck.shape == (2,)
        np.testing.assert_allclose(neck, [1.0, 0.0], atol=1e-9)

    def test_output_is_2d_regardless_of_input_dim(self):
        """Đầu ra luôn là (2,) — chỉ trả về x, y."""
        kpts3d = self._make_kpts([0.0, 0.0], [2.0, 0.0], [0.0, -1.0], [2.0, -1.0], ndim=3)
        neck, root = compute_neck_and_root(kpts3d)
        assert neck.shape == (2,)
        assert root.shape == (2,)

