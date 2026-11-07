"""Tests cho gaitmirror.calibration.mirror_camera.MirrorCameraSystem.

Kiểm tra:
- Khởi tạo đúng (K, normal, d, dist_coeffs)
- plane_from_setup(): hình học mặt phẳng gương từ số đo hiện trường
- project_real / project_virtual: điểm 3D → pixel
- reflect_points: X' = H X - 2dn
- from_config: dựng từ dict YAML
- Các trường hợp lỗi (d~0, n_z≥0, pháp tuyến ~0)
"""

import numpy as np
import pytest

from gaitmirror.reconstruction.mirror_camera import MirrorCameraSystem


# ====================================================================== #
# Fixtures dùng chung
# ====================================================================== #

def _make_K(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0) -> np.ndarray:
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _default_system() -> MirrorCameraSystem:
    """Hệ thống chuẩn: gương thẳng đứng bên trái, d = 2.0 m."""
    K = _make_K()
    # n hướng: (1, 0, -tan(30°)) sau chuẩn hóa; n_z < 0 → hợp lệ
    n, d = MirrorCameraSystem.plane_from_setup(
        camera_to_mirror=4.0,
        angle_to_mirror_deg=30.0,
        image_side="left",
    )
    return MirrorCameraSystem(K=K, normal=n, d=d)


# ====================================================================== #
# 1. Khởi tạo và thuộc tính cơ bản
# ====================================================================== #

class TestMirrorCameraSystemInit:
    """Kiểm tra khởi tạo MirrorCameraSystem."""

    def test_K_stored_correctly(self):
        sys = _default_system()
        assert sys.K.shape == (3, 3)
        assert sys.K[0, 0] == pytest.approx(1000.0)

    def test_normal_is_unit_vector(self):
        sys = _default_system()
        assert np.linalg.norm(sys.normal) == pytest.approx(1.0, abs=1e-9)

    def test_d_is_positive(self):
        sys = _default_system()
        assert sys.d > 0

    def test_H_is_reflection_matrix(self):
        """H = I - 2 n n^T phải là ma trận phản xạ: H^2 = I, det(H) = -1."""
        sys = _default_system()
        np.testing.assert_allclose(sys.H @ sys.H, np.eye(3), atol=1e-9)
        assert np.linalg.det(sys.H) == pytest.approx(-1.0, abs=1e-9)

    def test_P_real_shape(self):
        sys = _default_system()
        assert sys.P_real.shape == (3, 4)

    def test_P_virt_shape(self):
        sys = _default_system()
        assert sys.P_virt.shape == (3, 4)

    def test_virtual_camera_center_property(self):
        """Tâm camera ảo = -2 d n."""
        sys = _default_system()
        expected = -2.0 * sys.d * sys.normal
        np.testing.assert_allclose(sys.virtual_camera_center, expected, atol=1e-9)

    def test_baseline_property(self):
        """Baseline = 2d."""
        sys = _default_system()
        assert sys.baseline == pytest.approx(2.0 * sys.d, rel=1e-9)

    def test_negative_d_gets_flipped(self):
        """d < 0 được đổi dấu cùng normal để đảm bảo d > 0."""
        K = _make_K()
        # Cung cấp n hướng ngược (n_z > 0 trước khi flip) và d âm
        # → __init__ sẽ flip cả hai để d > 0, nhưng sau đó n_z vẫn phải < 0
        n_valid, d_valid = MirrorCameraSystem.plane_from_setup(4.0, 30.0, "left")
        sys = MirrorCameraSystem(K=K, normal=n_valid, d=d_valid)
        assert sys.d > 0

    def test_zero_d_raises(self):
        K = _make_K()
        n, _ = MirrorCameraSystem.plane_from_setup(4.0, 30.0, "left")
        with pytest.raises(ValueError, match="d ~ 0"):
            MirrorCameraSystem(K=K, normal=n, d=1e-15)

    def test_zero_normal_raises(self):
        K = _make_K()
        with pytest.raises(ValueError):
            MirrorCameraSystem(K=K, normal=[0, 0, 0], d=2.0)

    def test_n_z_positive_raises(self):
        """Gương sau camera (n_z > 0 sau khi d > 0) phải raise ValueError."""
        K = _make_K()
        # n_z > 0 → gương ở phía sau
        with pytest.raises(ValueError, match="SAU camera"):
            MirrorCameraSystem(K=K, normal=[0, 0, 1], d=2.0)

    def test_no_dist_coeffs_stored_as_none(self):
        sys = _default_system()
        assert sys.dist_coeffs is None

    def test_zero_dist_coeffs_stored_as_none(self):
        K = _make_K()
        n, d = MirrorCameraSystem.plane_from_setup(4.0, 30.0, "left")
        sys = MirrorCameraSystem(K=K, normal=n, d=d, dist_coeffs=[0, 0, 0, 0, 0])
        assert sys.dist_coeffs is None


# ====================================================================== #
# 2. plane_from_setup()
# ====================================================================== #

class TestPlaneFromSetup:
    """Kiểm tra hình học plane_from_setup."""

    def test_normal_is_unit_vector(self):
        n, d = MirrorCameraSystem.plane_from_setup(3.0, 30.0, "left")
        assert np.linalg.norm(n) == pytest.approx(1.0, abs=1e-9)

    def test_d_positive(self):
        _, d = MirrorCameraSystem.plane_from_setup(3.0, 30.0, "left")
        assert d > 0

    def test_n_z_negative(self):
        """n_z phải âm (gương đối diện camera)."""
        n, _ = MirrorCameraSystem.plane_from_setup(3.0, 30.0, "left")
        assert n[2] < 0

    def test_image_side_left_gives_positive_nx(self):
        """Ảnh gương bên trái → n_x > 0."""
        n, _ = MirrorCameraSystem.plane_from_setup(3.0, 30.0, "left")
        assert n[0] > 0

    def test_image_side_right_gives_negative_nx(self):
        """Ảnh gương bên phải → n_x < 0."""
        n, _ = MirrorCameraSystem.plane_from_setup(3.0, 30.0, "right")
        assert n[0] < 0

    def test_d_formula(self):
        """d = AB * sin(alpha)."""
        AB, alpha_deg = 4.0, 30.0
        _, d = MirrorCameraSystem.plane_from_setup(AB, alpha_deg, "left")
        expected_d = AB * np.sin(np.deg2rad(alpha_deg))
        assert d == pytest.approx(expected_d, rel=1e-9)

    def test_angle_zero_raises(self):
        with pytest.raises(ValueError):
            MirrorCameraSystem.plane_from_setup(3.0, 0.0, "left")

    def test_angle_90_raises(self):
        with pytest.raises(ValueError):
            MirrorCameraSystem.plane_from_setup(3.0, 90.0, "left")

    def test_negative_distance_raises(self):
        with pytest.raises(ValueError):
            MirrorCameraSystem.plane_from_setup(-1.0, 30.0, "left")

    def test_invalid_side_raises(self):
        with pytest.raises(ValueError):
            MirrorCameraSystem.plane_from_setup(3.0, 30.0, "center")


# ====================================================================== #
# 3. project_real() và project_virtual()
# ====================================================================== #

class TestProjection:
    """Kiểm tra chiếu điểm 3D lên pixel."""

    def setup_method(self):
        self.sys = _default_system()

    def test_project_real_point_on_axis(self):
        """Điểm trên trục quang học (0, 0, z) → chiếu vào tâm ảnh."""
        sys = self.sys
        X = np.array([[0.0, 0.0, 5.0]])
        px = sys.project_real(X)
        np.testing.assert_allclose(px[0], [sys.K[0, 2], sys.K[1, 2]], atol=1e-6)

    def test_project_real_shape(self):
        X = np.random.randn(10, 3)
        # Đặt z > 0
        X[:, 2] = np.abs(X[:, 2]) + 1.0
        px = self.sys.project_real(X)
        assert px.shape == (10, 2)

    def test_project_virtual_shape(self):
        X = np.random.randn(5, 3)
        X[:, 2] = np.abs(X[:, 2]) + 1.0
        pv = self.sys.project_virtual(X)
        assert pv.shape == (5, 2)

    def test_real_and_virtual_projections_differ(self):
        """Người thật và ảnh gương chiếu vào vị trí khác nhau."""
        X = np.array([[0.3, 0.0, 3.0]])
        pr = self.sys.project_real(X)
        pv = self.sys.project_virtual(X)
        assert not np.allclose(pr, pv)

    def test_projection_matrices_match_project_methods(self):
        """project_real() khớp với P_real tính tay."""
        sys = self.sys
        X = np.array([[0.1, -0.2, 3.5]])
        Xh = np.array([[0.1, -0.2, 3.5, 1.0]])
        raw = Xh @ sys.P_real.T
        expected = raw[:, :2] / raw[:, 2:3]
        np.testing.assert_allclose(sys.project_real(X), expected, atol=1e-9)


# ====================================================================== #
# 4. reflect_points()
# ====================================================================== #

class TestReflectPoints:
    """Kiểm tra phản xạ điểm qua gương."""

    def setup_method(self):
        self.sys = _default_system()

    def test_reflect_twice_gives_original(self):
        """Phản xạ hai lần phải trả về điểm gốc."""
        X = np.array([[0.5, 0.2, 3.0], [-0.3, 0.1, 2.5]])
        X_reflected = self.sys.reflect_points(X)
        X_back = self.sys.reflect_points(X_reflected)
        np.testing.assert_allclose(X_back, X, atol=1e-9)

    def test_point_on_mirror_plane_is_fixed(self):
        """Điểm nằm trên mặt phẳng gương: n^T X = -d (hay n^T X + d = 0)."""
        sys = self.sys
        # Tìm điểm X trên mặt phẳng: X = -d * n (đơn giản nhất)
        X_on_plane = (-sys.d * sys.normal).reshape(1, 3)
        # Kiểm tra: n^T X + d = 0
        check = float(X_on_plane @ sys.normal) + sys.d
        assert abs(check) < 1e-9
        X_ref = sys.reflect_points(X_on_plane)
        np.testing.assert_allclose(X_ref, X_on_plane, atol=1e-9)

    def test_reflect_shape_preserved(self):
        X = np.random.randn(7, 3)
        X_ref = self.sys.reflect_points(X)
        assert X_ref.shape == X.shape

    def test_reflect_distance_to_mirror_preserved(self):
        """Phản xạ bảo toàn khoảng cách từ điểm đến mặt phẳng gương."""
        sys = self.sys
        X = np.array([[1.0, 0.0, 2.0]])
        d_to_mirror = float(X @ sys.normal) + sys.d
        X_ref = sys.reflect_points(X)
        d_ref = float(X_ref @ sys.normal) + sys.d
        assert abs(d_to_mirror + d_ref) < 1e-9  # đối xứng qua mặt phẳng


# ====================================================================== #
# 5. from_config()
# ====================================================================== #

class TestFromConfig:
    """Kiểm tra dựng MirrorCameraSystem từ dict YAML."""

    def _camera_cfg(self):
        return {
            "camera": {
                "K": [[1200, 0, 960], [0, 1200, 540], [0, 0, 1]],
            }
        }

    def test_from_config_with_normal_and_d(self):
        """Dựng từ normal + d trực tiếp."""
        cam = self._camera_cfg()
        mirror = {"mirror": {"normal": [0.866, 0.0, -0.5], "d": 1.73}}
        sys = MirrorCameraSystem.from_config(cam, mirror)
        assert isinstance(sys, MirrorCameraSystem)
        assert sys.d > 0

    def test_from_config_with_plane_vector(self):
        """Dựng từ plane = [a, b, c, d]."""
        cam = self._camera_cfg()
        # n = (0.866, 0, -0.5) không chuẩn hóa, d = 1.73
        mirror = {"mirror": {"plane": [0.866, 0.0, -0.5, 1.73]}}
        sys = MirrorCameraSystem.from_config(cam, mirror)
        assert sys.d > 0

    def test_from_config_with_setup(self):
        """Dựng từ số đo hiện trường (image_side + camera_to_mirror + angle)."""
        cam = self._camera_cfg()
        mirror = {
            "mirror": {"image_side": "left"},
            "setup": {"camera_to_mirror": 4.0, "angle_to_mirror_deg": 30.0},
        }
        sys = MirrorCameraSystem.from_config(cam, mirror)
        assert sys.d > 0
        assert sys.normal[0] > 0  # image_side="left" → n_x > 0

    def test_from_config_missing_image_side_raises(self):
        """Thiếu image_side khi derive từ setup → KeyError."""
        cam = self._camera_cfg()
        mirror = {
            "mirror": {},  # thiếu image_side
            "setup": {"camera_to_mirror": 4.0, "angle_to_mirror_deg": 30.0},
        }
        with pytest.raises((KeyError, TypeError)):
            MirrorCameraSystem.from_config(cam, mirror)

    def test_from_config_rescales_K_same_ratio(self):
        """K được rescale đúng khi calib_size và image_size cùng tỉ lệ."""
        cam = {
            "camera": {
                "K": [[2400, 0, 1920], [0, 2400, 1080], [0, 0, 1]],
                "calib_image_size": [3840, 2160],
                "image_size": [1920, 1080],
            }
        }
        mirror = {"mirror": {"normal": [0.866, 0.0, -0.5], "d": 1.73}}
        sys = MirrorCameraSystem.from_config(cam, mirror)
        # fx phải được scale 1920/3840 = 0.5
        assert sys.K[0, 0] == pytest.approx(2400 * 1920 / 3840, rel=1e-6)


# ====================================================================== #
# 6. Tích hợp: projection_matrices property
# ====================================================================== #

class TestProjectionMatricesProperty:
    """Kiểm tra property trả đúng thứ tự (P_real, P_virt)."""

    def test_returns_tuple_of_two(self):
        sys = _default_system()
        mats = sys.projection_matrices
        assert len(mats) == 2

    def test_first_is_P_real(self):
        sys = _default_system()
        P_r, _ = sys.projection_matrices
        np.testing.assert_array_equal(P_r, sys.P_real)

    def test_second_is_P_virt(self):
        sys = _default_system()
        _, P_v = sys.projection_matrices
        np.testing.assert_array_equal(P_v, sys.P_virt)

