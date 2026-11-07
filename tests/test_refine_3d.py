"""Tests cho scripts/04_refine_3d.py.

Yêu cầu scipy — bỏ qua toàn bộ nếu scipy chưa được cài.

Kiểm tra:
- RefineConfig: 3 trường nhân trắc học mới có giá trị mặc định đúng
- weights_from_reproj(): khớp háng được nhân hip_weight_factor
- refine_pose_sequence():
  * Hội tụ với pose tổng hợp (T, 23, 3)
  * Tỉ lệ cẳng/đùi Trái và Phải độc lập (không ép đối xứng LLD)
  * bone lengths Trái và Phải có thể khác nhau sau tối ưu
- bone_length_series(): shape và NaN xử lý đúng
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

# ------------------------------------------------------------------ #
# Bỏ qua toàn bộ module nếu scipy chưa cài
# ------------------------------------------------------------------ #
scipy = pytest.importorskip("scipy", reason="scipy chưa được cài")

# Thêm thư mục scripts vào sys.path để import 04_refine_3d
_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

# Import module script dạng exec (file tên bắt đầu bằng số)
_script_path = _SCRIPTS_DIR / "04_refine_3d.py"
_mod = types.ModuleType("refine_3d")
exec(compile(_script_path.read_text(encoding="utf-8"), str(_script_path), "exec"),
     _mod.__dict__)

RefineConfig = _mod.RefineConfig
RefineResult = _mod.RefineResult
weights_from_reproj = _mod.weights_from_reproj
refine_pose_sequence = _mod.refine_pose_sequence
bone_length_series = _mod.bone_length_series
BONES = _mod.BONES
HIP_IDS = _mod.HIP_IDS


# ====================================================================== #
# Fixtures
# ====================================================================== #

def _make_synthetic_pose(
    T: int = 50,
    n_j: int = 23,
    seed: int = 42,
    thigh_L_m: float = 0.42,
    thigh_R_m: float = 0.38,   # LLD: đùi phải ngắn hơn 4 cm
    shank_L_m: float = 0.36,
    shank_R_m: float = 0.32,
    noise_m: float = 0.005,
) -> np.ndarray:
    """Tạo pose (T, 23, 3) với xương đùi/cẳng Trái-Phải độc lập.

    Chỉ số COCO (23 joints):
      11=L_HIP, 12=R_HIP, 13=L_KNEE, 14=R_KNEE, 15=L_ANKLE, 16=R_ANKLE
    """
    rng = np.random.default_rng(seed)
    pose = np.zeros((T, n_j, 3))

    # Sacrum ở gốc
    pose[:, 11] = [0.0, 0.0, 0.0]
    pose[:, 12] = [0.15, 0.0, 0.0]  # pelvis_width = 15 cm

    # Khớp gối (y = -thigh)
    pose[:, 13] = [0.0, -thigh_L_m, 0.0]
    pose[:, 14] = [0.15, -thigh_R_m, 0.0]

    # Cổ chân (y = -thigh - shank)
    pose[:, 15] = [0.0, -(thigh_L_m + shank_L_m), 0.0]
    pose[:, 16] = [0.15, -(thigh_R_m + shank_R_m), 0.0]

    # Vai (y = +0.5 m)
    pose[:, 5] = [-0.2, 0.5, 0.0]
    pose[:, 6] = [0.35, 0.5, 0.0]

    # Lặp T frame với nhiễu nhỏ
    pose = np.tile(pose[0:1], (T, 1, 1))
    pose += rng.normal(0, noise_m, pose.shape)

    # Thêm ngón chân và gót (cần cho BONES)
    pose[:, 17] = pose[:, 15] + [0.08, 0, 0]  # L_BIG_TOE
    pose[:, 18] = pose[:, 15] + [0.07, 0.01, 0]  # L_SMALL_TOE
    pose[:, 19] = pose[:, 15] + [-0.03, 0, 0]  # L_HEEL
    pose[:, 20] = pose[:, 16] + [0.08, 0, 0]  # R_BIG_TOE
    pose[:, 21] = pose[:, 16] + [0.07, 0.01, 0]  # R_SMALL_TOE
    pose[:, 22] = pose[:, 16] + [-0.03, 0, 0]  # R_HEEL

    # Các khớp còn lại (khuỷu, cổ tay, tai, mũi) — đặt theo vị trí đại khái
    pose[:, 7] = pose[:, 5] + [-0.1, -0.3, 0]
    pose[:, 8] = pose[:, 6] + [0.1, -0.3, 0]
    pose[:, 9] = pose[:, 7] + [-0.05, -0.3, 0]
    pose[:, 10] = pose[:, 8] + [0.05, -0.3, 0]
    pose[:, 0] = [0.075, 0.65, 0]   # nose
    pose[:, 1] = [0.025, 0.63, 0]   # L_EYE
    pose[:, 2] = [0.125, 0.63, 0]   # R_EYE
    pose[:, 3] = [0.0, 0.62, 0]     # L_EAR
    pose[:, 4] = [0.15, 0.62, 0]    # R_EAR

    return pose.astype(np.float64)


# ====================================================================== #
# 1. RefineConfig — 3 trường nhân trắc học mới
# ====================================================================== #

class TestRefineConfig:
    """Kiểm tra RefineConfig có đủ 3 trường nhân trắc học."""

    def test_hip_weight_factor_default(self):
        cfg = RefineConfig()
        assert cfg.hip_weight_factor == pytest.approx(0.15)

    def test_target_shank_thigh_ratio_default(self):
        cfg = RefineConfig()
        assert cfg.target_shank_thigh_ratio == pytest.approx(0.85)

    def test_sigma_ratio_m_default(self):
        cfg = RefineConfig()
        assert cfg.sigma_ratio_m == pytest.approx(0.005)

    def test_max_nfev_increased_to_80(self):
        """max_nfev phải là 80 (tăng từ 60)."""
        cfg = RefineConfig()
        assert cfg.max_nfev == 80

    def test_custom_hip_weight_factor(self):
        cfg = RefineConfig(hip_weight_factor=0.5)
        assert cfg.hip_weight_factor == pytest.approx(0.5)

    def test_all_original_fields_present(self):
        """Các trường gốc vẫn hiện diện."""
        cfg = RefineConfig()
        assert hasattr(cfg, "sigma_data_m")
        assert hasattr(cfg, "sigma_bone_m")
        assert hasattr(cfg, "sigma_smooth_m")
        assert hasattr(cfg, "gap_fill_weight")
        assert hasattr(cfg, "reproj_ref_px")


# ====================================================================== #
# 2. weights_from_reproj() — khớp háng bị hạ trọng số
# ====================================================================== #

class TestWeightsFromReproj:
    """Kiểm tra weights_from_reproj() nhân hip_weight_factor đúng."""

    def _make_reproj(self, T: int = 30, n_j: int = 23) -> np.ndarray:
        """Lỗi reproj (T, J) — tất cả bằng 0 (trọng số tối đa)."""
        return np.zeros((T, n_j), dtype=np.float64)

    def test_hip_weight_less_than_non_hip(self):
        """Trọng số khớp háng phải < trọng số khớp bình thường."""
        T, n_j = 30, 23
        pose = _make_synthetic_pose(T=T)
        reproj = self._make_reproj(T, n_j)
        cfg = RefineConfig(hip_weight_factor=0.15)
        w = weights_from_reproj(reproj, pose, None, cfg)

        w_hip = w[:, list(HIP_IDS)].mean()
        # Lấy khớp không phải háng để so sánh (vai = 5)
        w_shoulder = w[:, 5].mean()
        assert w_hip < w_shoulder

    def test_hip_weight_scaled_by_factor(self):
        """w_hip / w_shoulder ≈ hip_weight_factor (khi lỗi reproj = 0)."""
        T, n_j = 30, 23
        pose = _make_synthetic_pose(T=T)
        reproj = self._make_reproj(T, n_j)
        factor = 0.15
        cfg = RefineConfig(hip_weight_factor=factor)
        w = weights_from_reproj(reproj, pose, None, cfg)

        # reproj = 0 → w_base = 1.0 / (1 + 0) = 1.0
        # w_hip phải = 1.0 * factor
        for hip_id in HIP_IDS:
            assert w[:, hip_id].mean() == pytest.approx(factor, rel=0.01)

    def test_non_hip_weight_unaffected(self):
        """Các khớp khác không bị ảnh hưởng bởi hip_weight_factor."""
        T, n_j = 20, 23
        pose = _make_synthetic_pose(T=T)
        reproj = self._make_reproj(T, n_j)
        cfg = RefineConfig(hip_weight_factor=0.5)
        w = weights_from_reproj(reproj, pose, None, cfg)

        # Vai (5, 6) không bị scale
        for shoulder_id in [5, 6]:
            assert w[:, shoulder_id].mean() == pytest.approx(1.0, abs=0.01)

    def test_output_shape(self):
        T, n_j = 20, 23
        pose = _make_synthetic_pose(T=T)
        reproj = self._make_reproj(T, n_j)
        cfg = RefineConfig()
        w = weights_from_reproj(reproj, pose, None, cfg)
        assert w.shape == (T, n_j)

    def test_weights_nonnegative(self):
        T, n_j = 20, 23
        pose = _make_synthetic_pose(T=T)
        reproj = self._make_reproj(T, n_j)
        cfg = RefineConfig()
        w = weights_from_reproj(reproj, pose, None, cfg)
        assert np.all(w >= 0)

    def test_gap_filled_weight_applied(self):
        """Các frame gap_filled nhận gap_fill_weight."""
        T, n_j = 20, 23
        pose = _make_synthetic_pose(T=T)
        reproj = self._make_reproj(T, n_j)
        gap_filled = np.zeros((T, n_j), dtype=bool)
        gap_filled[5, 5] = True  # frame 5, joint 5 bị gap
        cfg = RefineConfig(gap_fill_weight=0.25)
        w = weights_from_reproj(reproj, pose, gap_filled, cfg)
        assert w[5, 5] == pytest.approx(0.25, abs=0.01)


# ====================================================================== #
# 3. refine_pose_sequence()
# ====================================================================== #

class TestRefinePoseSequence:
    """Kiểm tra refine_pose_sequence() với pose tổng hợp."""

    def test_returns_refine_result(self):
        pose = _make_synthetic_pose(T=50)
        w = np.ones((50, 23))
        cfg = RefineConfig(max_nfev=5)
        result = refine_pose_sequence(pose, w, cfg=cfg)
        assert isinstance(result, RefineResult)

    def test_output_pose_shape(self):
        T = 40
        pose = _make_synthetic_pose(T=T)
        w = np.ones((T, 23))
        cfg = RefineConfig(max_nfev=5)
        result = refine_pose_sequence(pose, w, cfg=cfg)
        assert result.pose.shape == (T, 23, 3)

    def test_lengths_array_length_matches_bones(self):
        pose = _make_synthetic_pose(T=40)
        w = np.ones((40, 23))
        cfg = RefineConfig(max_nfev=5)
        result = refine_pose_sequence(pose, w, cfg=cfg)
        assert len(result.lengths) == len(BONES)

    def test_valid_mask_has_no_nan_points(self):
        """Các điểm hợp lệ trong result.pose không được là NaN."""
        pose = _make_synthetic_pose(T=40)
        w = np.ones((40, 23))
        cfg = RefineConfig(max_nfev=5)
        result = refine_pose_sequence(pose, w, cfg=cfg)
        assert np.all(np.isfinite(result.pose[result.valid]))

    def test_lld_bone_lengths_independent(self):
        """thigh_L và thigh_R, shank_L và shank_R phải có thể KHÁC nhau.

        Pose tổng hợp có LLD (thigh_L=0.42 > thigh_R=0.38).
        Sau tối ưu, lengths[thigh_L] ≠ lengths[thigh_R] (không bị ép bằng nhau).
        """
        pose = _make_synthetic_pose(
            T=60, thigh_L_m=0.42, thigh_R_m=0.38,
            shank_L_m=0.36, shank_R_m=0.32, noise_m=0.001
        )
        w = np.ones((60, 23))
        cfg = RefineConfig(
            max_nfev=50,
            sigma_bone_m=0.001,  # siết xương chặt → lengths bám gần median
            sigma_ratio_m=0.1,   # nới ràng buộc tỉ lệ → tự do LLD
        )
        result = refine_pose_sequence(pose, w, cfg=cfg)
        bone_names = [b[0] for b in BONES]
        idx_tL = bone_names.index("thigh_L")
        idx_tR = bone_names.index("thigh_R")
        L_thigh_L = result.lengths[idx_tL]
        L_thigh_R = result.lengths[idx_tR]

        # Cả hai phải hữu hạn
        assert np.isfinite(L_thigh_L), "thigh_L length phải hữu hạn"
        assert np.isfinite(L_thigh_R), "thigh_R length phải hữu hạn"

        # Không được ép bằng nhau (LLD được giữ nguyên)
        assert abs(L_thigh_L - L_thigh_R) > 0.01, (
            f"thigh_L ({L_thigh_L:.4f}) và thigh_R ({L_thigh_R:.4f}) "
            f"quá gần nhau — có thể đang bị ép đối xứng"
        )

    def test_shank_thigh_ratio_approximately_satisfied(self):
        """Ràng buộc tỉ lệ cẳng/đùi phải được đáp ứng gần đúng.

        Với sigma_ratio_m nhỏ, ratio sau tối ưu gần target_shank_thigh_ratio.
        """
        pose = _make_synthetic_pose(T=80, noise_m=0.002)
        w = np.ones((80, 23))
        cfg = RefineConfig(
            max_nfev=100,
            target_shank_thigh_ratio=0.85,
            sigma_ratio_m=0.001,  # siết chặt tỉ lệ
            sigma_bone_m=0.01,    # nới xương → cho phép di chuyển
        )
        result = refine_pose_sequence(pose, w, cfg=cfg)
        bone_names = [b[0] for b in BONES]
        for side in ("L", "R"):
            idx_sh = bone_names.index(f"shank_{side}")
            idx_th = bone_names.index(f"thigh_{side}")
            L_sh = result.lengths[idx_sh]
            L_th = result.lengths[idx_th]
            if np.isfinite(L_sh) and np.isfinite(L_th) and L_th > 0:
                ratio = L_sh / L_th
                assert abs(ratio - 0.85) < 0.15, (
                    f"Tỉ lệ cẳng/đùi {side}: {ratio:.3f} xa target 0.85"
                )

    def test_cost_decreases_after_optimization(self):
        """Cost cuối < cost ban đầu (optimizer thực sự tối ưu)."""
        pose = _make_synthetic_pose(T=50, noise_m=0.01)
        w = np.ones((50, 23))
        cfg = RefineConfig(max_nfev=30)
        result = refine_pose_sequence(pose, w, cfg=cfg)
        assert result.cost_final <= result.cost_init

    def test_nan_input_preserved_in_output(self):
        """Điểm NaN đầu vào vẫn là NaN ở đầu ra (không bịa dữ liệu)."""
        pose = _make_synthetic_pose(T=40)
        pose[5, 7, :] = np.nan  # frame 5, khuỷu trái
        w = np.ones((40, 23))
        w[5, 7] = 0.0  # weight = 0 cho điểm NaN
        cfg = RefineConfig(max_nfev=5)
        result = refine_pose_sequence(pose, w, cfg=cfg)
        assert not np.isfinite(result.pose[5, 7]).all()


# ====================================================================== #
# 4. bone_length_series()
# ====================================================================== #

class TestBoneLengthSeries:
    """Kiểm tra bone_length_series()."""

    def test_output_shape(self):
        T = 30
        pose = _make_synthetic_pose(T=T)
        bl = bone_length_series(pose)
        assert bl.shape == (T, len(BONES))

    def test_known_bone_length(self):
        """thigh_L = khoảng cách L_HIP(11) - L_KNEE(13) = thigh_L_m + noise."""
        T = 50
        thigh_L = 0.42
        pose = _make_synthetic_pose(T=T, thigh_L_m=thigh_L, noise_m=0.001)
        bl = bone_length_series(pose)
        bone_names = [b[0] for b in BONES]
        idx = bone_names.index("thigh_L")
        mean_len = np.nanmean(bl[:, idx])
        assert abs(mean_len - thigh_L) < 0.01

    def test_nan_propagation(self):
        """Nếu một đầu mút xương là NaN → length tại đó cũng NaN."""
        pose = _make_synthetic_pose(T=20)
        pose[3, 11, :] = np.nan  # frame 3, L_HIP
        bl = bone_length_series(pose)
        bone_names = [b[0] for b in BONES]
        idx_thigh_L = bone_names.index("thigh_L")
        assert not np.isfinite(bl[3, idx_thigh_L])

    def test_all_positive(self):
        """Độ dài xương phải dương hoặc NaN."""
        pose = _make_synthetic_pose(T=20)
        bl = bone_length_series(pose)
        finite = bl[np.isfinite(bl)]
        assert np.all(finite > 0)

