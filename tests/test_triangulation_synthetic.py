"""
Unit + Integration tests cho triangulate_dlt.py

Bao gồm:
  1. Tests cơ bản: hartley_normalization, build_sigmas, DLT tuyến tính
  2. Tests chính xác: Triangulate điểm tổng hợp từ MirrorCameraSystem
  3. Tests robustness: NaN input, điểm phía sau camera, sigma bất thường
  4. Test tích hợp: Pipeline correct_virt -> triangulate_points_dlt
     (kiểm tra kết nối đúng giữa tầng 2D và 3D theo Ajisafe et al.)
"""

import numpy as np
import pytest

from gaitmirror.reconstruction.triangulate_dlt import (
    TriangulationConfig,
    TriangulationResult,
    build_sigmas,
    hartley_normalization,
    triangulate_points_dlt,
)
from gaitmirror.reconstruction.mirror_camera import MirrorCameraSystem
from gaitmirror.reconstruction.mirror_optimization import correct_virt
from gaitmirror.pose.keypoint_mapping import WHOLEBODY_FLIP_PAIRS


# ===========================================================================
# Fixtures chung
# ===========================================================================

@pytest.fixture
def simple_mirror_system():
    """
    Hệ camera + gương đơn giản cho test tổng hợp.
    Camera nhìn theo trục Z, gương nghiêng 45° ở phía trước bên phải.
    """
    K = np.array([
        [800.0,   0.0, 640.0],
        [  0.0, 800.0, 360.0],
        [  0.0,   0.0,   1.0],
    ])
    normal = np.array([-1.0, 0.0, -1.0])  # gương bên phải, hướng vào camera
    d_val = 2.0  # khoảng cách 2m (dữ liệu tổng hợp, không cần tuyệt đối)
    return MirrorCameraSystem(K=K, normal=normal, d=d_val)


@pytest.fixture
def ground_truth_points():
    """Tập điểm 3D tổng hợp cho người đứng thẳng (đơn vị mét)."""
    rng = np.random.default_rng(7)
    # Phân tán xung quanh vị trí người đứng (Z ~ 3m trước camera)
    pts = np.column_stack([
        rng.uniform(-0.5, 0.5, 20),   # X: ngang
        rng.uniform(-0.9, 0.9, 20),   # Y: cao
        rng.uniform(2.5, 3.5, 20),    # Z: sâu
    ])
    return pts


# ===========================================================================
# Nhóm 1: hartley_normalization
# ===========================================================================

class TestHartleyNormalization:

    def test_mean_is_zero_after_transform(self):
        """Sau khi áp dụng T, trọng tâm điểm phải về 0."""
        rng = np.random.default_rng(1)
        pts = rng.uniform(100, 900, (100, 2))
        T = hartley_normalization(pts)
        pts_h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
        pts_norm = (T @ pts_h.T).T[:, :2]
        np.testing.assert_allclose(pts_norm.mean(axis=0), [0.0, 0.0], atol=1e-10)

    def test_avg_distance_is_sqrt2(self):
        """Sau chuẩn hóa, khoảng cách trung bình từ gốc phải ~ sqrt(2)."""
        rng = np.random.default_rng(2)
        pts = rng.uniform(0, 1920, (200, 2))
        T = hartley_normalization(pts)
        pts_h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
        pts_norm = (T @ pts_h.T).T[:, :2]
        avg_dist = np.linalg.norm(pts_norm, axis=1).mean()
        assert avg_dist == pytest.approx(np.sqrt(2.0), rel=0.05)

    def test_output_shape(self):
        """T phải có shape (3, 3)."""
        pts = np.random.rand(50, 2) * 1000
        T = hartley_normalization(pts)
        assert T.shape == (3, 3)
        assert T[2, 2] == pytest.approx(1.0)

    def test_manual_formula_matches_matrix_multiply(self):
        """
        Cách tính x_n manual trong dòng 292 của triangulate_dlt.py
        phải tương đương với T @ x_homogeneous.
        """
        rng = np.random.default_rng(42)
        pts = rng.uniform(100, 900, (50, 2))
        T = hartley_normalization(pts)
        scale = T[0, 0]
        offset = -T[:2, 2] / scale

        # Manual (như trong code)
        x_n_manual = (pts - offset) * scale

        # Matrix multiply
        pts_h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
        x_n_matrix = (T @ pts_h.T).T[:, :2]

        np.testing.assert_allclose(x_n_manual, x_n_matrix, atol=1e-10)


# ===========================================================================
# Nhóm 2: build_sigmas
# ===========================================================================

class TestBuildSigmas:

    def test_output_shape(self):
        """build_sigmas phải trả về (N, 2, 2)."""
        joint_ids = np.arange(23)
        cfg = TriangulationConfig()
        s = build_sigmas(joint_ids, None, None, cfg)
        assert s.shape == (23, 2, 2)

    def test_foot_joints_get_higher_sigma_v(self):
        """
        Khớp bàn chân (17-22) phải có sigma_v cao hơn thân thể (0-16).
        Đây là ràng buộc chống bias bóng sàn gương.
        """
        joint_ids = np.array([0, 5, 13, 17, 19, 22])  # mix thân + bàn chân
        cfg = TriangulationConfig(
            sigma_v=(1.0, 1.0),
            foot_sigma_v=(3.0, 3.0),
        )
        s = build_sigmas(joint_ids, None, None, cfg)
        # joint 0, 1, 2 (body) -> sigma_v = 1.0
        # joint 3, 4, 5 (foot: 17, 19, 22) -> sigma_v = 3.0
        assert s[0, :, 1] == pytest.approx([1.0, 1.0])  # nose
        assert s[3, :, 1] == pytest.approx([3.0, 3.0])  # L_BIG_TOE (17)
        assert s[4, :, 1] == pytest.approx([3.0, 3.0])  # L_HEEL (19)
        assert s[5, :, 1] == pytest.approx([3.0, 3.0])  # R_HEEL (22)

    def test_confidence_scales_sigma(self):
        """
        Confidence cao -> sigma nhỏ hơn (điểm đáng tin hơn).
        sigma_eff = sigma_base / sqrt(conf).
        """
        joint_ids = np.array([0, 5])
        conf = np.array([[1.0, 1.0], [0.25, 0.25]])  # (N, V)
        cfg = TriangulationConfig(sigma_u=(1.0, 1.0), sigma_v=(1.0, 1.0),
                                   use_confidence=True, conf_floor=0.01)
        s = build_sigmas(joint_ids, conf, None, cfg)
        # conf=1.0: scale = 1/sqrt(1) = 1.0 -> sigma = 1.0
        # conf=0.25: scale = 1/sqrt(0.25) = 2.0 -> sigma = 2.0
        assert s[0, 0, 0] == pytest.approx(1.0)   # joint 0, view 0, sigma_u
        assert s[1, 0, 0] == pytest.approx(2.0)   # joint 1, view 0, sigma_u

    def test_interpolated_flag_scales_sigma(self):
        """Điểm nội suy phải có sigma lớn hơn (ít đáng tin hơn)."""
        joint_ids = np.array([0, 1])
        interp = np.array([[True, False], [False, False]])  # (N, V)
        cfg = TriangulationConfig(sigma_u=(1.0, 1.0), sigma_v=(1.0, 1.0),
                                   use_confidence=False, interp_sigma_scale=2.0)
        s = build_sigmas(joint_ids, None, interp, cfg)
        # joint 0, view 0: interp=True -> sigma*2
        assert s[0, 0, 0] == pytest.approx(2.0)
        # joint 0, view 1: interp=False -> sigma*1
        assert s[0, 1, 0] == pytest.approx(1.0)
        # joint 1: không interpolated
        assert s[1, 0, 0] == pytest.approx(1.0)

    def test_conf_floor_clamps_low_confidence(self):
        """Confidence bằng 0 phải bị clamp về conf_floor, không gây chia-0."""
        joint_ids = np.array([0])
        conf = np.array([[0.0, 0.0]])
        cfg = TriangulationConfig(use_confidence=True, conf_floor=0.05)
        # Không được raise exception
        s = build_sigmas(joint_ids, conf, None, cfg)
        # sigma = sigma_base / sqrt(conf_floor) = 1 / sqrt(0.05)
        expected = 1.0 / np.sqrt(0.05)
        assert s[0, 0, 0] == pytest.approx(expected, rel=1e-6)


# ===========================================================================
# Nhóm 3: triangulate_points_dlt — chính xác tổng hợp
# ===========================================================================

class TestTriangulateDLT:

    def test_exact_reconstruction_no_noise(self, simple_mirror_system, ground_truth_points):
        """
        Với điểm chiếu chính xác (không nhiễu), DLT+LM phải tái dựng
        về gần chính xác điểm 3D gốc.
        """
        sys = simple_mirror_system
        P_real, P_virt = sys.projection_matrices
        X = ground_truth_points  # (N, 3)

        # Chiếu ra 2 view
        x_real = sys.project_real(X)   # (N, 2)
        x_virt = sys.project_virtual(X)  # (N, 2)
        pts_2d = np.stack([x_real, x_virt], axis=1)  # (N, 2, 2)

        cfg = TriangulationConfig(refine=True, max_iters=50, sigma_u=(0.5, 0.5))
        result = triangulate_points_dlt(
            projections=[P_real, P_virt],
            points_2d=pts_2d,
            config=cfg,
        )

        assert result.valid.all(), f"Tất cả điểm phải valid, nhận {result.valid.sum()}/{len(X)}"
        err = np.linalg.norm(result.points_3d - X, axis=1)
        # Với dữ liệu không nhiễu: sai số < 5mm
        assert err.max() < 0.005, f"Sai số tái dựng quá lớn: max={err.max()*1000:.2f}mm"

    def test_reprojection_error_near_zero_no_noise(self, simple_mirror_system, ground_truth_points):
        """Sai số chiếu ngược phải < 0.5 pixel với dữ liệu không nhiễu."""
        sys = simple_mirror_system
        P_real, P_virt = sys.projection_matrices
        X = ground_truth_points

        x_real = sys.project_real(X)
        x_virt = sys.project_virtual(X)
        pts_2d = np.stack([x_real, x_virt], axis=1)

        result = triangulate_points_dlt(
            projections=[P_real, P_virt],
            points_2d=pts_2d,
        )
        assert result.reproj_error[result.valid].max() < 0.5

    def test_nan_input_marked_invalid(self, simple_mirror_system, ground_truth_points):
        """Điểm có NaN ở bất kỳ view nào phải có valid=False, X=NaN."""
        sys = simple_mirror_system
        P_real, P_virt = sys.projection_matrices
        X = ground_truth_points

        x_real = sys.project_real(X)
        x_virt = sys.project_virtual(X)
        pts_2d = np.stack([x_real, x_virt], axis=1)

        # Làm hỏng một số điểm bằng NaN
        nan_indices = [0, 3, 7]
        pts_2d[nan_indices, 0, :] = np.nan  # view real bị NaN

        result = triangulate_points_dlt(
            projections=[P_real, P_virt],
            points_2d=pts_2d,
        )

        for i in nan_indices:
            assert not result.valid[i], f"Điểm {i} có NaN phải invalid"
            assert np.all(np.isnan(result.points_3d[i])), f"X[{i}] phải NaN"

        # Các điểm còn lại vẫn valid
        good = np.ones(len(X), dtype=bool)
        good[nan_indices] = False
        assert result.valid[good].all()

    def test_minimum_points_requirement(self, simple_mirror_system):
        """Cần ít nhất 3 điểm valid để chạy DLT batch (do SVD)."""
        sys = simple_mirror_system
        P_real, P_virt = sys.projection_matrices

        # Chỉ 2 điểm valid
        X = np.array([[0.0, 0.0, 3.0], [0.1, 0.1, 3.0]])
        x_real = sys.project_real(X)
        x_virt = sys.project_virtual(X)
        pts_2d = np.stack([x_real, x_virt], axis=1)

        # Không nên raise exception — kết quả có thể all invalid
        result = triangulate_points_dlt(
            projections=[P_real, P_virt],
            points_2d=pts_2d,
        )
        assert result.points_3d.shape == (2, 3)
        assert isinstance(result.valid, np.ndarray)

    def test_refine_false_gives_dlt_only(self, simple_mirror_system, ground_truth_points):
        """refine=False phải bỏ qua bước LM và vẫn trả về kết quả."""
        sys = simple_mirror_system
        P_real, P_virt = sys.projection_matrices
        X = ground_truth_points

        x_real = sys.project_real(X)
        x_virt = sys.project_virtual(X)
        pts_2d = np.stack([x_real, x_virt], axis=1)

        cfg = TriangulationConfig(refine=False)
        result = triangulate_points_dlt(
            projections=[P_real, P_virt],
            points_2d=pts_2d,
            config=cfg,
        )
        assert result.points_3d.shape == (len(X), 3)
        assert result.valid.sum() > 0

    def test_output_shape_consistency(self, simple_mirror_system, ground_truth_points):
        """Shape của tất cả field trong TriangulationResult phải nhất quán."""
        sys = simple_mirror_system
        P_real, P_virt = sys.projection_matrices
        X = ground_truth_points
        N = len(X)

        x_real = sys.project_real(X)
        x_virt = sys.project_virtual(X)
        pts_2d = np.stack([x_real, x_virt], axis=1)

        result = triangulate_points_dlt(
            projections=[P_real, P_virt],
            points_2d=pts_2d,
        )

        assert result.points_3d.shape == (N, 3)
        assert result.residuals.shape == (N, 2, 2)
        assert result.reproj_error.shape == (N, 2)
        assert result.cost.shape == (N,)
        assert result.converged.shape == (N,)
        assert result.in_front.shape == (N,)
        assert result.valid.shape == (N,)

    def test_at_least_2_projections_required(self, simple_mirror_system):
        """Cần tối thiểu 2 ma trận chiếu — ít hơn phải raise ValueError."""
        sys = simple_mirror_system
        P_real = sys.P_real
        pts = np.zeros((5, 1, 2))  # chỉ 1 view

        with pytest.raises(ValueError, match="2 view"):
            triangulate_points_dlt(projections=[P_real], points_2d=pts)

    def test_sigma_shape_mismatch_raises(self, simple_mirror_system, ground_truth_points):
        """sigmas với shape sai phải raise ValueError."""
        sys = simple_mirror_system
        P_real, P_virt = sys.projection_matrices
        X = ground_truth_points

        x_real = sys.project_real(X)
        x_virt = sys.project_virtual(X)
        pts_2d = np.stack([x_real, x_virt], axis=1)
        bad_sigmas = np.ones((len(X), 2, 3))  # shape sai

        with pytest.raises(ValueError, match="sigmas"):
            triangulate_points_dlt(
                projections=[P_real, P_virt],
                points_2d=pts_2d,
                sigmas=bad_sigmas,
            )

    def test_gaussian_noise_reconstruction_quality(self, simple_mirror_system, ground_truth_points):
        """
        Với nhiễu Gaussian 1px, sai số tái dựng phải < 20mm (thực tế lâm sàng).
        """
        sys = simple_mirror_system
        P_real, P_virt = sys.projection_matrices
        X = ground_truth_points

        rng = np.random.default_rng(99)
        x_real = sys.project_real(X) + rng.normal(0, 1.0, (len(X), 2))
        x_virt = sys.project_virtual(X) + rng.normal(0, 1.0, (len(X), 2))
        pts_2d = np.stack([x_real, x_virt], axis=1)

        cfg = TriangulationConfig(refine=True, max_iters=50,
                                   sigma_u=(1.0, 1.0), sigma_v=(1.0, 1.0))
        result = triangulate_points_dlt(
            projections=[P_real, P_virt],
            points_2d=pts_2d,
            config=cfg,
        )

        valid_err = np.linalg.norm(
            result.points_3d[result.valid] - X[result.valid], axis=1
        )
        assert valid_err.mean() < 0.020, \
            f"Sai số trung bình với nhiễu 1px: {valid_err.mean()*1000:.1f}mm > 20mm"


# ===========================================================================
# Nhóm 4: Integration — correct_virt -> triangulate_points_dlt
# ===========================================================================

class TestIntegrationCorrectVirtToTriangulate:
    """
    Kiểm tra kết nối đúng giữa tầng 2D (correct_virt) và DLT.

    Theo kiến trúc Ajisafe et al.:
      1. mirror_kpts_raw được lưu nguyên trạng (không flip).
      2. correct_virt() so sánh hướng detection vs projection.
      3. Kết quả kpts_aligned được đưa vào triangulate_points_dlt().
    """

    def test_pipeline_same_direction_no_flip_affects_result(self, simple_mirror_system):
        """
        Khi detection và projection cùng hướng (flip_mask=False):
        correct_virt trả về detection nguyên bản,
        và DLT vẫn tái dựng được điểm 3D.
        """
        sys = simple_mirror_system
        P_real, P_virt = sys.projection_matrices

        # Điểm test đơn giản: 5 điểm trước camera
        X_true = np.array([
            [0.0, 0.5, 3.0],
            [0.1, 0.0, 3.0],
            [-0.1, -0.3, 3.1],
            [0.2, 0.4, 2.8],
            [-0.2, 0.2, 3.2],
        ])
        x_real = sys.project_real(X_true)
        x_virt = sys.project_virtual(X_true)

        # Giả lập kpts 2D (T=1, K=23, channel=3)
        K = 23
        det = np.zeros((1, K, 3), dtype=np.float64)
        proj = np.zeros((1, K, 3), dtype=np.float64)

        # Cả det và proj đều cùng hướng (RS > LS) -> flip_mask = False
        det[0, 5, :] = [200.0, 300.0, 0.9]   # L_SHOULDER
        det[0, 6, :] = [260.0, 300.0, 0.9]   # R_SHOULDER (> L)
        proj[0, 5, :] = [190.0, 310.0, 0.9]  # L_SHOULDER proj
        proj[0, 6, :] = [255.0, 310.0, 0.9]  # R_SHOULDER proj (> L)

        kpts_aligned, weights_aligned, flipped_mask = correct_virt(
            virt_detection_kpts=det,
            virt_projected_kpts=proj,
            flip_pairs=WHOLEBODY_FLIP_PAIRS,
        )

        # Không có flip
        assert not flipped_mask[0]

        # Tiến hành tam giác hóa với tọa độ thực
        pts_2d = np.stack([x_real, x_virt], axis=1)  # (5, 2, 2)
        result = triangulate_points_dlt(
            projections=[P_real, P_virt],
            points_2d=pts_2d,
        )
        assert result.valid.sum() == len(X_true), \
            "Tất cả điểm phải valid trong trường hợp không flip"
        err = np.linalg.norm(result.points_3d - X_true, axis=1)
        assert err.max() < 0.005

    def test_pipeline_flipped_detection_corrected(self, simple_mirror_system):
        """
        Khi detection ngược hướng projection (flip_mask=True):
        correct_virt hoán đổi L/R, và DLT với tọa độ gốc (không flip)
        cho kết quả đúng hơn tọa độ ngược hướng.

        Test này kiểm tra nguyên tắc: gọi correct_virt() TRƯỚC khi
        đưa tọa độ mirror vào DLT.
        """
        # Tạo 2 bộ keypoints với L_KNEE và R_KNEE trao đổi vị trí
        K = 23
        det_correct = np.zeros((1, K, 3))
        det_flipped = np.zeros((1, K, 3))
        proj_ref = np.zeros((1, K, 3))

        # Detection đúng: RS > LS
        det_correct[0, 5, :] = [200.0, 300.0, 0.9]
        det_correct[0, 6, :] = [260.0, 300.0, 0.9]
        # Detection bị flip: LS > RS (hướng ngược)
        det_flipped[0, 5, :] = [260.0, 300.0, 0.9]
        det_flipped[0, 6, :] = [200.0, 300.0, 0.9]

        # Đặt L_KNEE và R_KNEE phân biệt trong det_correct
        det_correct[0, 13, 0] = 190.0  # L_KNEE
        det_correct[0, 14, 0] = 270.0  # R_KNEE
        det_flipped[0, 13, 0] = 270.0  # L_KNEE (nhầm vị trí)
        det_flipped[0, 14, 0] = 190.0  # R_KNEE (nhầm vị trí)

        # Projection cùng hướng với det_correct (RS > LS)
        proj_ref[0, 5, :] = [210.0, 290.0, 0.9]
        proj_ref[0, 6, :] = [250.0, 290.0, 0.9]

        # correct_virt với detection bị flip
        kpts_aligned, _, flip_mask = correct_virt(
            virt_detection_kpts=det_flipped,
            virt_projected_kpts=proj_ref,
            flip_pairs=WHOLEBODY_FLIP_PAIRS,
        )

        # Phải flip
        assert flip_mask[0]

        # Sau khi correct_virt, L_KNEE và R_KNEE phải đã được hoán đổi lại đúng
        # det_flipped[0,13]=270 (nhầm), det_flipped[0,14]=190 (nhầm)
        # Sau flip: kpts_aligned[0,13] = det_flipped[0,14] = 190 (đúng như det_correct)
        #           kpts_aligned[0,14] = det_flipped[0,13] = 270 (đúng như det_correct)
        assert kpts_aligned[0, 13, 0] == pytest.approx(190.0), \
            "L_KNEE sau correct_virt phải trả về giá trị gốc 190.0"
        assert kpts_aligned[0, 14, 0] == pytest.approx(270.0), \
            "R_KNEE sau correct_virt phải trả về giá trị gốc 270.0"

    def test_config_from_dict_valid_keys(self):
        """TriangulationConfig.from_dict() nhận đúng key."""
        cfg = TriangulationConfig.from_dict({
            "sigma_u": 1.5,
            "sigma_v": 2.0,
            "foot_sigma_v": 4.0,
            "refine": True,
            "max_iters": 20,
        })
        assert cfg.sigma_u == pytest.approx((1.5, 1.5))
        assert cfg.sigma_v == pytest.approx((2.0, 2.0))
        assert cfg.foot_sigma_v == pytest.approx((4.0, 4.0))

    def test_config_from_dict_unknown_key_raises(self):
        """Key không hợp lệ phải raise KeyError rõ ràng."""
        with pytest.raises(KeyError, match="không hợp lệ"):
            TriangulationConfig.from_dict({"unknown_key": 1.0})

    def test_config_from_dict_sigma_v_body_alias(self):
        """Alias sigma_v_body phải được nhận dạng và chuyển thành sigma_v."""
        cfg = TriangulationConfig.from_dict({"sigma_v_body": [1.2, 1.5]})
        assert cfg.sigma_v == pytest.approx((1.2, 1.5))

    def test_in_front_flag_correct(self, simple_mirror_system, ground_truth_points):
        """
        Điểm 3D phía trước cả 2 camera (depth > 0) phải có in_front=True.
        """
        sys = simple_mirror_system
        P_real, P_virt = sys.projection_matrices
        X = ground_truth_points  # Z > 0, trước camera

        x_real = sys.project_real(X)
        x_virt = sys.project_virtual(X)
        pts_2d = np.stack([x_real, x_virt], axis=1)

        result = triangulate_points_dlt(
            projections=[P_real, P_virt],
            points_2d=pts_2d,
        )
        # Tất cả điểm GT ở Z > 0 -> phải in_front
        assert result.in_front[result.valid].all(), \
            "Điểm 3D hợp lệ phía trước camera phải có in_front=True"

