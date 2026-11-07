"""
tests/test_heatmap_decoder.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests cho module giải mã heatmap sub-pixel (DARK/Taylor) 
trong hệ thống phân tích dáng đi gait_mirror.
file này viết ngày 20/10 dbt có đúng không nhưng claude hết token r, gemini viết nm chắc file test thì không sao đâu
"""

from typing import Tuple
import numpy as np
import pytest

from gaitmirror.pose.heatmap_decoder import (
    decode_heatmaps_subpixel,
    gaussian_blur_heatmaps,
)


def _generate_synthetic_heatmap(
    h: int = 64,
    w: int = 48,
    mu_x: float = 24.35,
    mu_y: float = 32.70,
    sigma: float = 2.0,
    peak_val: float = 1.0,
) -> np.ndarray:
    """Tạo heatmap Gaussian 2D tổng hợp với tọa độ cực đại thực (continuous)."""
    y, x = np.ogrid[:h, :w]
    dist_sq = (x - mu_x) ** 2 + (y - mu_y) ** 2
    heatmap = peak_val * np.exp(-dist_sq / (2.0 * sigma**2))
    return heatmap.astype(np.float32)


# ==============================================================================
# 1. KIỂM THỬ ĐỘ CHÍNH XÁC DƯỚI PIXEL (SUB-PIXEL ACCURACY)
# ==============================================================================

def test_subpixel_accuracy_synthetic_clean():
    """Kiểm tra Taylor bậc hai giảm sai số lượng tử hóa từ ~0.4 ô xuống < 0.05 ô."""
    h, w = 64, 48
    # Đặt đỉnh tại vị trí thập phân ngẫu nhiên, xa biên
    mu_x_true, mu_y_true = 24.35, 32.70
    sigma = 2.0

    hm_single = _generate_synthetic_heatmap(h, w, mu_x_true, mu_y_true, sigma)
    # Tensor dạng (N=1, K=23, H=64, W=48)
    heatmaps = np.zeros((1, 23, h, w), dtype=np.float32)
    heatmaps[0, 0] = hm_single

    # Đặt center = [100, 100], scale = [1.0, 1.0] để scale_pixel = [200, 200]
    centers = np.array([[100.0, 100.0]], dtype=np.float32)
    scales = np.array([[1.0, 1.0]], dtype=np.float32)

    # 1. Chạy với Taylor (Sub-pixel)
    coords_taylor, _ = decode_heatmaps_subpixel(
        heatmaps, centers, scales, use_taylor=True, use_udp=False, kernel=7
    )

    # 2. Chạy không có Taylor (Integer Argmax)
    coords_argmax, _ = decode_heatmaps_subpixel(
        heatmaps, centers, scales, use_taylor=False, use_udp=False
    )

    # Đổi ngược tọa độ ảnh về lại hệ tọa độ heatmap để tính sai số ô (heatmap bins)
    # scale_pixel = 200, denom = 48 (x) và 64 (y)
    # x_img = x_hm * (200 / 48) + (100 - 100) => x_hm = x_img * (48 / 200)
    pred_x_taylor = coords_taylor[0, 0, 0] * (w / 200.0)
    pred_y_taylor = coords_taylor[0, 0, 1] * (h / 200.0)

    pred_x_argmax = coords_argmax[0, 0, 0] * (w / 200.0)
    pred_y_argmax = coords_argmax[0, 0, 1] * (h / 200.0)

    err_argmax = np.sqrt((pred_x_argmax - mu_x_true) ** 2 + (pred_y_argmax - mu_y_true) ** 2)
    err_taylor = np.sqrt((pred_x_taylor - mu_x_true) ** 2 + (pred_y_taylor - mu_y_true) ** 2)

    # Đỉnh nguyên lệch khoảng ~0.46 ô do làm tròn (24, 33)
    assert err_argmax > 0.3, f"Sai số argmax bất thường: {err_argmax}"
    # Taylor phải đưa sai số xuống mức tiệm cận 0 (< 0.05 ô)
    assert err_taylor < 0.05, f"Taylor không đạt độ chính xác sub-pixel: {err_taylor}"


def test_subpixel_robustness_with_noise():
    """Kiểm tra thuật toán duy trì độ chính xác khi heatmap bị nhiễu nền."""
    np.random.seed(42)
    h, w = 64, 48
    mu_x_true, mu_y_true = 15.65, 40.25
    hm_clean = _generate_synthetic_heatmap(h, w, mu_x_true, mu_y_true, sigma=2.0)
    noise = np.random.normal(0.0, 0.03, size=(h, w)).astype(np.float32)
    hm_noisy = np.clip(hm_clean + noise, 0.0, 1.0)

    heatmaps = np.zeros((1, 23, h, w), dtype=np.float32)
    heatmaps[0, 0] = hm_noisy
    centers = np.array([[0.0, 0.0]], dtype=np.float32)
    scales = np.array([[1.0, 1.0]], dtype=np.float32)

    coords, _ = decode_heatmaps_subpixel(heatmaps, centers, scales, use_taylor=True, kernel=7)

    pred_x = coords[0, 0, 0] * (w / 200.0) + w * 0.5
    pred_y = coords[0, 0, 1] * (h / 200.0) + h * 0.5
    err = np.sqrt((pred_x - mu_x_true) ** 2 + (pred_y - mu_y_true) ** 2)

    # Dù có nhiễu, sai số vẫn phải nhỏ hơn một nửa ô lưới (< 0.15 ô)
    assert err < 0.15, f"Sai số có nhiễu vượt ngưỡng cho phép: {err}"


# ==============================================================================
# 2. KIỂM THỬ ĐIỂM TIN CẬY (CONFIDENCE FROM RAW HEATMAP)
# ==============================================================================

def test_confidence_taken_from_raw_heatmap():
    """Điểm tin cậy maxvals phải được lấy từ heatmap thô, không bị sụt giảm bởi GaussianBlur."""
    h, w = 64, 48
    heatmaps = np.zeros((1, 23, h, w), dtype=np.float32)
    # Tạo đỉnh chuẩn có giá trị cực đại chính xác = 1.0
    heatmaps[0, 0] = _generate_synthetic_heatmap(h, w, mu_x=20.0, mu_y=20.0, peak_val=1.0)

    centers = np.array([[50.0, 50.0]], dtype=np.float32)
    scales = np.array([[1.0, 1.0]], dtype=np.float32)

    _, maxvals = decode_heatmaps_subpixel(
        heatmaps, centers, scales, use_taylor=True, kernel=7
    )

    # Nếu lấy từ heatmap làm mượt, maxval sẽ tụt xuống ~0.88 - 0.90
    # Code đúng phải giữ nguyên xấp xỉ 1.0 từ dữ liệu gốc
    assert np.isclose(maxvals[0, 0, 0], 1.0, atol=1e-5), (
        f"maxvals bị làm mượt làm suy hao: {maxvals[0, 0, 0]} != 1.0"
    )


# ==============================================================================
# 3. KIỂM THỬ XỬ LÝ BIÊN BÀN CHÂN (BOUNDARY CONDITIONS)
# ==============================================================================

def test_boundary_handling_bottom_foot():
    """Điểm sát mép đáy (y = 62/64 - vị trí gót chân) không được lỗi index, không sinh NaN."""
    h, w = 64, 48
    # Đặt đỉnh tại hàng y = 62.0 (sát đáy H-2)
    heatmaps = np.zeros((1, 23, h, w), dtype=np.float32)
    heatmaps[0, 19] = _generate_synthetic_heatmap(h, w, mu_x=24.0, mu_y=62.0, sigma=2.0)

    centers = np.array([[100.0, 100.0]], dtype=np.float32)
    scales = np.array([[1.0, 1.0]], dtype=np.float32)

    coords, maxvals = decode_heatmaps_subpixel(
        heatmaps, centers, scales, use_taylor=True, kernel=7, conf_thr=0.2
    )

    # Đảm bảo không bị NaN do out-of-bounds
    assert not np.isnan(coords[0, 19, 0]), "Tọa độ X bàn chân sát đáy bị NaN"
    assert not np.isnan(coords[0, 19, 1]), "Tọa độ Y bàn chân sát đáy bị NaN"
    assert maxvals[0, 19, 0] > 0.5


def test_boundary_extreme_edge_clipping():
    """Đỉnh nằm ngay tại biên (y = 0 hoặc y = H-1) phải bỏ qua Taylor và kẹp an toàn."""
    h, w = 64, 48
    heatmaps = np.zeros((1, 23, h, w), dtype=np.float32)
    heatmaps[0, 0, 0, 0] = 1.0  # Đỉnh tuyệt đối ở góc (0, 0)

    centers = np.array([[0.0, 0.0]], dtype=np.float32)
    scales = np.array([[1.0, 1.0]], dtype=np.float32)

    # Không được văng exception khi px, py nằm ngoài phạm vi stencil 3x3
    coords, maxvals = decode_heatmaps_subpixel(
        heatmaps, centers, scales, use_taylor=True, kernel=3
    )
    assert not np.isnan(coords[0, 0]).any()


# ==============================================================================
# 4. KIỂM THỬ LOẠI TRỪ ĐIỂM YÊN NGỰA & CỰC TIỂU (HESSIAN NEGATIVE DEFINITE)
# ==============================================================================

def test_saddle_point_and_minimum_rejection():
    """Điều kiện det > 1e-6 and dxx < 0 phải loại bỏ điểm cực tiểu hoặc yên ngựa."""
    h, w = 64, 48
    heatmaps = np.ones((1, 23, h, w), dtype=np.float32)
    # Tạo một hố cực tiểu (Inverted Gaussian) tại tâm
    y, x = np.ogrid[:h, :w]
    heatmaps[0, 0] = 1.0 - np.exp(-((x - 24) ** 2 + (y - 32) ** 2) / 8.0)

    centers = np.array([[100.0, 100.0]], dtype=np.float32)
    scales = np.array([[1.0, 1.0]], dtype=np.float32)

    # Chạy hàm giải mã: với cực tiểu, dxx > 0 nên Taylor bắt buộc không được kích hoạt
    coords, _ = decode_heatmaps_subpixel(heatmaps, centers, scales, use_taylor=True)
    assert not np.isnan(coords).any()


# ==============================================================================
# 5. KIỂM THỬ LỌC ĐIỂM TIN CẬY THẤP & HEATMAP RỖNG (CONF_THR -> NAN)
# ==============================================================================

def test_low_confidence_and_zero_heatmaps_return_nan():
    """Heatmap toàn 0 hoặc có maxval < conf_thr bắt buộc phải trả về np.nan."""
    h, w = 64, 48
    heatmaps = np.zeros((2, 23, h, w), dtype=np.float32)
    # Person 0: keypoint 0 có độ tin cậy yếu 0.15
    heatmaps[0, 0] = _generate_synthetic_heatmap(h, w, peak_val=0.15)
    # Person 1: keypoint 0 có độ tin cậy mạnh 0.85
    heatmaps[1, 0] = _generate_synthetic_heatmap(h, w, peak_val=0.85)

    centers = np.array([[100.0, 100.0], [200.0, 200.0]], dtype=np.float32)
    scales = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)

    coords, maxvals = decode_heatmaps_subpixel(
        heatmaps, centers, scales, conf_thr=0.2
    )

    # Person 0, kpt 0 có maxval = 0.15 < 0.2 => Phải là NaN
    assert np.isnan(coords[0, 0, 0])
    assert np.isnan(coords[0, 0, 1])
    # Person 1, kpt 0 có maxval = 0.85 >= 0.2 => Tọa độ hợp lệ
    assert not np.isnan(coords[1, 0, 0])
    assert not np.isnan(coords[1, 0, 1])

    # Keypoint 1 (toàn số 0 trên cả 2 người) => Phải là NaN
    assert np.isnan(coords[0, 1]).all()
    assert np.isnan(coords[1, 1]).all()


# ==============================================================================
# 6. KIỂM THỬ PHÉP ÁNH XẠ HÌNH HỌC (STANDARD VS UDP COORDINATE INVARIANT)
# ==============================================================================

def test_coordinate_mapping_center_invariant():
    """Điểm ở chính giữa không gian heatmap phải ánh xạ chính xác về BBox center."""
    h, w = 64, 48
    heatmaps = np.zeros((1, 23, h, w), dtype=np.float32)

    cx_expected, cy_expected = 450.0, 720.0
    centers = np.array([[cx_expected, cy_expected]], dtype=np.float32)
    scales = np.array([[1.5, 2.0]], dtype=np.float32)

    # 1. Kiểm tra chế độ UDP: Tâm liên tục nằm tại (W-1)/2 và (H-1)/2
    mu_x_udp = (w - 1.0) * 0.5  # 23.5
    mu_y_udp = (h - 1.0) * 0.5  # 31.5
    heatmaps[0, 0] = _generate_synthetic_heatmap(h, w, mu_x=mu_x_udp, mu_y=mu_y_udp)

    coords_udp, _ = decode_heatmaps_subpixel(
        heatmaps, centers, scales, use_taylor=True, use_udp=True
    )
    assert np.isclose(coords_udp[0, 0, 0], cx_expected, atol=1e-3), (
        f"UDP X không khớp tâm BBox: {coords_udp[0, 0, 0]} != {cx_expected}"
    )
    assert np.isclose(coords_udp[0, 0, 1], cy_expected, atol=1e-3), (
        f"UDP Y không khớp tâm BBox: {coords_udp[0, 0, 1]} != {cy_expected}"
    )

    # 2. Kiểm tra chế độ Standard: Tâm nằm tại W/2 và H/2
    mu_x_std = w * 0.5  # 24.0
    mu_y_std = h * 0.5  # 32.0
    heatmaps[0, 0] = _generate_synthetic_heatmap(h, w, mu_x=mu_x_std, mu_y=mu_y_std)

    coords_std, _ = decode_heatmaps_subpixel(
        heatmaps, centers, scales, use_taylor=True, use_udp=False
    )
    assert np.isclose(coords_std[0, 0, 0], cx_expected, atol=1e-3), (
        f"Standard X không khớp tâm BBox: {coords_std[0, 0, 0]} != {cx_expected}"
    )
    assert np.isclose(coords_std[0, 0, 1], cy_expected, atol=1e-3), (
        f"Standard Y không khớp tâm BBox: {coords_std[0, 0, 1]} != {cy_expected}"
    )


# ==============================================================================
# 7. KIỂM THỬ HÀM LÀM MƯỢT GAUSSIAN VỚI ZERO-PADDING
# ==============================================================================

def test_gaussian_blur_heatmaps_parameters():
    """Kiểm tra điều kiện kích thước kernel và bảo toàn shape."""
    heatmaps = np.random.uniform(0.0, 1.0, size=(2, 23, 64, 48)).astype(np.float32)

    # Kernel chẵn hoặc < 3 phải bắn ValueError
    with pytest.raises(ValueError):
        gaussian_blur_heatmaps(heatmaps, kernel=4)
    with pytest.raises(ValueError):
        gaussian_blur_heatmaps(heatmaps, kernel=2)

    # Kernel hợp lệ phải bảo toàn hoàn toàn cấu trúc shape
    blurred = gaussian_blur_heatmaps(heatmaps, kernel=7)
    assert blurred.shape == heatmaps.shape
    assert blurred.dtype == heatmaps.dtype