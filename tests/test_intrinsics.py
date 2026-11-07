import cv2
import numpy as np
import pytest

from synth import make_set
from gaitmirror.calibration.intrinsics import (
    IntrinsicsResult, assert_same_resolution, calibrate_from_images, load_intrinsics,
    save_intrinsics, summary, undistort_points)

K_TRUE = np.array([[1000.0, 0, 640.0], [0, 1000.0, 360.0], [0, 0, 1.0]])


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    d = tmp_path_factory.mktemp("cal")
    paths, _ = make_set(str(d), n=25, K=K_TRUE)
    return paths


@pytest.fixture(scope="module")
def result(images):
    return calibrate_from_images(images)


def test_recovers_known_intrinsics(result):
    assert abs(result.K[0, 0] - 1000) < 10 and abs(result.K[1, 1] - 1000) < 10
    assert abs(result.K[0, 2] - 640) < 5 and abs(result.K[1, 2] - 360) < 5
    assert result.image_size == (1280, 720)
    assert len(result.used_paths) >= 10


def test_per_image_rms_matches_opencv_rms(result):
    """RMS gop tu RMS tung anh phai trung voi rms cua cv2.calibrateCamera (cung thang do)."""
    n = 8 * 5
    pooled = np.sqrt(np.mean(np.square(result.per_image_rms)))   # moi anh co cung n diem
    assert pooled == pytest.approx(result.rms, rel=1e-3)


def test_too_few_images_raises_clear_message(images):
    with pytest.raises(ValueError, match="can it nhat 10"):
        calibrate_from_images(images[:4])


def test_mixed_resolution_is_rejected_not_mixed(images, tmp_path):
    """Mot anh ban co HOP LE nhung khac do phan giai (960x540) phai bi loai, khong duoc tron vao."""
    K_small = np.array([[750.0, 0, 480.0], [0, 750.0, 270.0], [0, 0, 1.0]])
    odd_paths, _ = make_set(str(tmp_path), n=1, seed=0, K=K_small, size=(960, 540))
    odd = odd_paths[0]
    found, _ = cv2.findChessboardCorners(cv2.imread(odd, 0), (8, 5))
    assert found, "tien de: anh khac do phan giai phai nhan dien duoc ban co, neu khong test se vo nghia"
    res = calibrate_from_images(images + odd_paths)
    assert odd in res.rejected_paths
    assert res.image_size == (1280, 720)
    assert abs(res.K[0, 0] - 1000) < 10          # K khong bi keo ve phia anh khac do phan giai


def test_outlier_threshold_can_empty_the_set(images):
    with pytest.raises(ValueError):
        calibrate_from_images(images, max_image_rms_px=1e-9)


def test_fix_k3(images):
    res = calibrate_from_images(images, fix_k3=True)
    assert res.dist.ravel()[4] == 0.0


def test_undistort_points_inverts_distortion():
    dist = np.array([-0.25, 0.08, 0.001, -0.0005, 0.0])
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(-0.4, 0.4, 50), rng.uniform(-0.3, 0.3, 50), rng.uniform(1.5, 3.0, 50)])
    zero = np.zeros(3)
    distorted, _ = cv2.projectPoints(X, zero, zero, K_TRUE, dist)
    ideal, _ = cv2.projectPoints(X, zero, zero, K_TRUE, np.zeros(5))
    back = undistort_points(distorted.reshape(-1, 2), K_TRUE, dist)
    assert np.max(np.abs(back - ideal.reshape(-1, 2))) < 0.05      # pixel


def test_undistort_points_keeps_nan_and_shape():
    pts = np.array([[[640.0, 360.0], [np.nan, np.nan]], [[700.0, 400.0], [10.0, 20.0]]])   # (2,2,2)
    out = undistort_points(pts, K_TRUE, np.zeros(5))
    assert out.shape == pts.shape and np.isnan(out[0, 1]).all() and np.allclose(out[0, 0], [640, 360], atol=1e-6)


def test_save_load_roundtrip(result, tmp_path):
    p = str(tmp_path / "intr.json")
    save_intrinsics(p, result, meta={"camera": "test", "stabilization": "off"})
    K, dist, size = load_intrinsics(p)
    assert np.allclose(K, result.K) and np.allclose(dist, result.dist.ravel()) and size == (1280, 720)


def test_resolution_guard_and_summary(result):
    assert_same_resolution((1280, 720), (1280, 720))
    with pytest.raises(ValueError):
        assert_same_resolution((1280, 720), (1920, 1080))
    s = summary(result)
    assert 0.98 < s["fx_over_fy"] < 1.02 and abs(s["cx_offset_frac"]) < 0.02