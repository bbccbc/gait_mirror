"""
gait_mirror.pose.heatmap_decoder
Đã test: 20/10/2026
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Giai ma heatmap ViTPose sang toa do 2D sub-pixel (DARK/Taylor).
cung giao dien, them kernel lam mem co zero-padding,
diem tin cay lay tu heatmap tho, tra NaN cho diem tin cay thap, chi xu ly cac kenh can dung.

Luu y quan trong:
 - centers/scales phai tinh bang CUNG cong thuc voi luc cat anh dua vao mang (mmpose: bbox -> center, scale,
   ti le 192:256, he so padding). Giai ma dung nhung cat sai thi toa do van sai.
 - use_udp phai khop voi cach mo hinh duoc huan luyen/danh gia (kiem tra config). Lech se gay sai so
   phu thuoc vi tri trong hop (khoang 1 o heatmap o mep).
 - Neu tu lay heatmap: flip-test (trung binh heatmap lat) can flip_back + shift, hoac tat flip-test.
"""
from typing import Optional, Tuple
import cv2
import numpy as np


def gaussian_blur_heatmaps(heatmaps: np.ndarray, kernel: int = 7) -> np.ndarray:
    """Lam muot tung heatmap bang Gaussian, dem 0 quanh bien (giong mmpose) de dinh khong bi keo vao bien
    nhu khi dung bien phan xa mac dinh cua cv2."""
    if kernel < 3 or kernel % 2 == 0:
        raise ValueError("kernel phai le va >= 3")
    n_, k_, h_, w_ = heatmaps.shape
    b = (kernel - 1) // 2
    out = np.empty_like(heatmaps)
    for n in range(n_):
        for k in range(k_):
            d = np.zeros((h_ + 2 * b, w_ + 2 * b), dtype=heatmaps.dtype)
            d[b:-b, b:-b] = heatmaps[n, k]
            d = cv2.GaussianBlur(d, (kernel, kernel), 0)
            out[n, k] = d[b:-b, b:-b]
    return out


def decode_heatmaps_subpixel(
    heatmaps: np.ndarray,
    centers: np.ndarray,
    scales: np.ndarray,
    use_taylor: bool = True,
    use_udp: bool = False,
    num_kpts_to_keep: int = 23,
    kernel: int = 7,
    conf_thr: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """(N, K, H, W) -> toa do anh goc (N, num_kpts_to_keep, 2) va diem tin cay (N, num_kpts_to_keep, 1).
    Diem co tin cay < conf_thr duoc tra NaN (neu conf_thr khac None)."""
    hm_raw = heatmaps[:, :num_kpts_to_keep]
    N, K, H, W = hm_raw.shape

    # diem tin cay tu heatmap THO (lam muot ha dinh, lam sai lech nguong tin cay)
    maxvals = hm_raw.reshape(N, K, -1).max(axis=2)[..., None]

    hm = gaussian_blur_heatmaps(hm_raw, kernel) if use_taylor else hm_raw
    max_idx = hm.reshape(N, K, -1).argmax(axis=2)
    preds = np.zeros((N, K, 2), dtype=np.float32)
    preds[..., 0] = max_idx % W
    preds[..., 1] = max_idx // W

    if use_taylor:
        for n in range(N):
            for k in range(K):
                px, py = int(preds[n, k, 0]), int(preds[n, k, 1])
                if 1 <= px <= W - 2 and 1 <= py <= H - 2:      # chi can lan can 3x3
                    p = np.log(np.maximum(hm[n, k, py - 1:py + 2, px - 1:px + 2], 1e-10))
                    dx = 0.5 * (p[1, 2] - p[1, 0])
                    dy = 0.5 * (p[2, 1] - p[0, 1])
                    dxx = p[1, 2] - 2 * p[1, 1] + p[1, 0]
                    dyy = p[2, 1] - 2 * p[1, 1] + p[0, 1]
                    dxy = 0.25 * (p[2, 2] - p[2, 0] - p[0, 2] + p[0, 0])
                    det = dxx * dyy - dxy ** 2
                    if det > 1e-6 and dxx < 0:
                        ox = -(dyy * dx - dxy * dy) / det
                        oy = -(-dxy * dx + dxx * dy) / det
                        preds[n, k, 0] += np.clip(ox, -0.5, 0.5)
                        preds[n, k, 1] += np.clip(oy, -0.5, 0.5)

    denom = np.array([(W - 1.0) if use_udp else W, (H - 1.0) if use_udp else H], dtype=np.float64)
    scale_pixel = np.asarray(scales, dtype=np.float64) * 200.0           # (N, 2)
    coords = preds * (scale_pixel / denom)[:, None, :] \
        + (np.asarray(centers, dtype=np.float64) - scale_pixel * 0.5)[:, None, :]
    coords = coords.astype(np.float32)

    if conf_thr is not None:
        coords[maxvals[..., 0] < conf_thr] = np.nan
    return coords, maxvals