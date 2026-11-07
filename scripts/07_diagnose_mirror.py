#!/usr/bin/env python3
"""Chẩn đoán hình học gương từ dữ liệu 2D (không dùng n, d nhập tay).

Hai kiểm tra độc lập với d:

1. Đồng quy (epipole): với gương phẳng, đường thẳng nối điểm thật và điểm ảnh
   gương của cùng một khớp đều đi qua một điểm e = K n. Nếu các đường không
   đồng quy => ghép cặp real/mirror sai (lật trái-phải, nhầm người...). Script
   thử cả hai trường hợp: giữ nguyên và hoán đổi trái-phải của mirror_kpts.
2. Quét hướng gương: n = (sin phi, 0, -cos phi), chọn phi có sai số tái chiếu
   nhỏ nhất; so với n suy từ epipole (cho phép n_y != 0 khi camera chúi).

d chỉ đổi tỉ lệ 3D nên không quan sát được từ sai số tái chiếu; lấy từ ràng
buộc bàn chân ở Y = setup.camera_height.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml
from numpy.typing import NDArray

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from gaitmirror.reconstruction.mirror_camera import MirrorCameraSystem  # noqa: E402
from gaitmirror.reconstruction.triangulate_dlt import (  # noqa: E402
    TriangulationConfig,
    build_sigmas,
    triangulate_points_dlt,
)

LOG = logging.getLogger("diagnose_mirror")
FloatArray = NDArray[np.float64]
FOOT_IDS = (17, 18, 19, 20, 21, 22)
# Cặp trái-phải theo quy ước 23 điểm: COCO-17 + [LBT, LST, LH, RBT, RST, RH]
LR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14),
            (15, 16), (17, 20), (18, 21), (19, 22)]
D0 = 1.0


def lr_permutation(n_joints: int = 23) -> NDArray[np.int64]:
    perm = np.arange(n_joints)
    for a, b in LR_PAIRS:
        perm[a], perm[b] = b, a
    return perm


def normal_from_phi(phi_deg: float) -> FloatArray:
    p = np.deg2rad(phi_deg)
    return np.array([np.sin(p), 0.0, -np.cos(p)])


def fit_epipole(
    pr: FloatArray, pm: FloatArray, center: FloatArray
) -> Tuple[FloatArray, FloatArray]:
    """Điểm đồng quy e (3,) và khoảng cách (px) từ e tới từng đường nối.

    Lặp 4 lần, mỗi lần bỏ các đường có residual > 3 lần median (robust).
    """
    s = 1000.0
    a = np.c_[(pr - center) / s, np.ones(len(pr))]
    b = np.c_[(pm - center) / s, np.ones(len(pm))]
    lines = np.cross(a, b)
    norm = np.linalg.norm(lines[:, :2], axis=1)
    keep = norm > 1e-9
    lines = lines[keep] / norm[keep, None]
    use = np.ones(len(lines), bool)
    e = np.array([0.0, 0.0, 1.0])
    for _ in range(4):
        _, _, vt = np.linalg.svd(lines[use], full_matrices=False)
        e = vt[-1]
        res = np.abs(lines @ e) / max(abs(e[2]), 1e-12)
        use = res <= max(3.0 * np.median(res), 1e-6)
    # đổi về pixel
    e_px = np.array([e[0] * s + center[0] * e[2], e[1] * s + center[1] * e[2], e[2]])
    res_px = np.full(len(pr), np.nan)
    res_px[keep] = np.abs(lines @ e) / max(abs(e[2]), 1e-12) * s
    return e_px, res_px


def point_line_residual(pr: FloatArray, pm: FloatArray, e: FloatArray) -> FloatArray:
    """Khoảng cách (px) từ điểm gương tới đường (epipole -> điểm thật).

    Đo ngay tại vị trí khớp nên so sánh trực tiếp được với nhiễu detector
    (khác khoảng cách tới epipole vốn bị khuếch đại khi epipole ở xa).
    """
    if abs(e[2]) < 1e-9:
        return np.full(len(pr), np.nan)
    e2 = e[:2] / e[2]
    v1 = pr - e2
    v2 = pm - e2
    cross = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return cross / np.maximum(np.linalg.norm(v1, axis=1), 1e-9)


def prepare(
    pts: FloatArray, conf: FloatArray, interp: FloatArray,
    min_conf: float, n_points: int, seed: int,
) -> Tuple[FloatArray, FloatArray, NDArray[np.int64]]:
    good = (
        np.isfinite(pts).all(axis=(2, 3))
        & (np.nan_to_num(conf, nan=0.0) >= min_conf).all(axis=2)
        & ~interp.any(axis=2)
    )
    tt, jj = np.nonzero(good)
    if tt.size < 100:
        raise SystemExit(f"Quá ít điểm hợp lệ ({tt.size}).")
    rng = np.random.default_rng(seed)
    sel = rng.choice(tt.size, size=min(n_points, tt.size), replace=False)
    tt, jj = tt[sel], jj[sel]
    return pts[tt, jj], conf[tt, jj], jj


def evaluate(
    K: FloatArray, n: FloatArray, pts: FloatArray, sig: FloatArray,
    cfg: TriangulationConfig,
) -> Tuple[float, float, FloatArray, FloatArray]:
    system = MirrorCameraSystem(K, n, D0)
    res = triangulate_points_dlt(system.projection_matrices, pts, sig, cfg)
    ok = res.valid & res.in_front
    frac = float(ok.mean())
    if frac < 0.2:
        return 1e3, frac, res.points_3d, ok
    return float(np.median(res.reproj_error[ok].mean(axis=1))), frac, \
        res.points_3d, ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input_npz", type=Path, required=True)
    ap.add_argument("--config_camera", type=Path, required=True)
    ap.add_argument("--config_mirror", type=Path, required=True)
    ap.add_argument("--n_points", type=int, default=4000)
    ap.add_argument("--phi_step", type=float, default=1.0)
    ap.add_argument("--min_confidence", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cam_cfg: Dict[str, Any] = yaml.safe_load(open(args.config_camera, encoding="utf-8"))
    mir_cfg: Dict[str, Any] = yaml.safe_load(open(args.config_mirror, encoding="utf-8"))
    setup = mir_cfg.get("setup", {}) or {}
    cam_height = setup.get("camera_height")
    walkway = setup.get("walkway_to_mirror")

    current = MirrorCameraSystem.from_config(
        cam_cfg, mir_cfg, target_image_size=(1920, 1080)
    )
    K = current.K
    center = K[:2, 2].copy()
    tri_cfg = TriangulationConfig.from_dict(mir_cfg.get("triangulation"))
    tri_cfg.max_iters = 8

    data = np.load(args.input_npz, allow_pickle=False)
    mkey = "mirror_kpts_raw" if "mirror_kpts_raw" in data.files else "mirror_kpts"
    real, mirror = data["real_kpts"], data[mkey]
    t_len, n_j, _ = real.shape
    interp = np.zeros((t_len, n_j, 2), bool)
    for i, key in enumerate(("real_mask", "mirror_mask")):
        if key in data.files:
            interp[..., i] = data[key]
    perm = lr_permutation(n_j)

    def build(swap: bool) -> Tuple[FloatArray, FloatArray, NDArray[np.int64]]:
        m = mirror[:, perm] if swap else mirror
        it = interp.copy()
        if swap:
            it[..., 1] = it[:, perm, 1]
        p = np.stack([real[..., :2], m[..., :2]], axis=2)
        c = np.stack([real[..., 2], m[..., 2]], axis=2)
        return prepare(current.undistort_points(p), c, it,
                       args.min_confidence, args.n_points, args.seed)

    # ---- Kiểm tra 1: đồng quy, thử giữ nguyên / hoán đổi trái-phải ----
    results = {}
    for swap in (False, True):
        p_sub, c_sub, jj = build(swap)
        e, _ = fit_epipole(p_sub[:, 0], p_sub[:, 1], center)
        res = point_line_residual(p_sub[:, 0], p_sub[:, 1], e)
        n_e = np.linalg.solve(K, e)
        n_e /= np.linalg.norm(n_e)
        if n_e[2] > 0:
            n_e = -n_e
        results[swap] = (float(np.nanmedian(res)), float(np.nanpercentile(res, 90)),
                         n_e, p_sub, c_sub, jj)
        LOG.info(
            "Epipole [%s]: median %.1f px, p90 %.1f px | n=%s",
            "hoán đổi trái-phải" if swap else "giữ nguyên",
            results[False if not swap else True][0],
            results[swap][1], np.round(n_e, 3).tolist(),
        )
    swap_best = results[True][0] < results[False][0]
    med, p90, n_epi, p_sub, c_sub, jj = results[swap_best]
    if swap_best and results[True][0] < 0.7 * results[False][0]:
        LOG.warning(
            "Hoán đổi trái-phải của mirror_kpts đồng quy tốt hơn hẳn: bước gán "
            "real/mirror có vẻ lật SAI (chưa lật, hoặc lật hai lần)."
        )
    if med > 5.0:
        LOG.warning(
            "Các đường nối real-mirror KHÔNG đồng quy (median %.1f px tại vị trí "
            "khớp, ngưỡng tốt ~1-3 px): ghép cặp điểm/nhầm người ở nhiều khung "
            "hình, hoặc K/méo sai.", med,
        )

    # ---- Kiểm tra 2: quét phi + n từ epipole ----
    sig = build_sigmas(jj, c_sub, None, tri_cfg)
    phis = np.arange(-85.0, 85.0 + 1e-9, args.phi_step)
    costs = np.array([evaluate(K, normal_from_phi(p), p_sub, sig, tri_cfg)[0]
                      for p in phis])
    phi_best = float(phis[int(np.argmin(costs))])
    try:
        from scipy.optimize import minimize_scalar

        phi_best = float(minimize_scalar(
            lambda p: evaluate(K, normal_from_phi(p), p_sub, sig, tri_cfg)[0],
            bounds=(phi_best - 2.0, phi_best + 2.0), method="bounded").x)
    except ImportError:
        pass
    n_sweep = normal_from_phi(phi_best)
    c_sweep = evaluate(K, n_sweep, p_sub, sig, tri_cfg)
    c_epi = evaluate(K, n_epi, p_sub, sig, tri_cfg)
    c_cfg = evaluate(K, current.normal, p_sub, sig, tri_cfg)
    LOG.info("Reproj median: quét phi=%.1f deg -> %.2f px | n từ epipole -> %.2f px "
             "| cấu hình hiện tại -> %.2f px", phi_best, c_sweep[0], c_epi[0], c_cfg[0])
    n_best, c_best = (n_epi, c_epi) if c_epi[0] < c_sweep[0] else (n_sweep, c_sweep)

    # ---- d: so sánh nhiều nguồn độc lập (sai số tái chiếu không phụ thuộc d) ----
    _, frac, X, ok = c_best
    LOG.info("Valid %.0f%% (điểm trước gương và trước camera).", 100 * frac)
    mean_term = float(np.mean(X[ok] @ n_best + D0))  # khoảng cách tới gương, d=D0
    cands: Dict[str, float] = {}
    foot = np.isin(jj, FOOT_IDS) & ok
    # Mặt sàn: n_g vuông góc n_m, không roll => n_g = (0, cos a, sin a),
    # tan a = -n_y / n_z  (a = độ chúi xuống của camera, giả định gương thẳng đứng)
    alpha = float(np.arctan(-n_best[1] / n_best[2])) if abs(n_best[2]) > 1e-6 else 0.0
    n_g = np.array([0.0, np.cos(alpha), np.sin(alpha)])
    LOG.info("Camera chúi xuống ~%.1f deg (suy từ n_y; giả định gương thẳng "
             "đứng, không roll). Pháp tuyến sàn n_g=%s", np.degrees(alpha),
             np.round(n_g, 3).tolist())
    y_med = float(np.median(X[foot] @ n_g)) if foot.sum() >= 20 else float("nan")
    if cam_height is not None and y_med > 0:
        cands["sàn (có xét chúi)"] = D0 * float(cam_height) / y_med
    # Khớp mặt sàn trực tiếp từ bàn chân (không dựa vào n_y): bàn chân chạm sàn
    # tạo cụm điểm rất hẹp theo pháp tuyến sàn; chọn beta làm đuôi trên của
    # phân bố g = X . n_g(beta) hẹp nhất (p95 - p70), mức sàn = p90.
    beta_fit: Optional[float] = None
    if foot.sum() >= 200:
        gp = X[foot]
        a_deg = float(np.degrees(alpha))
        betas = np.arange(a_deg - 10.0, a_deg + 10.0 + 1e-9, 0.25)

        def _g(b_deg: float) -> FloatArray:
            b = np.deg2rad(b_deg)
            return gp @ np.array([0.0, np.cos(b), np.sin(b)])

        widths = [float(np.percentile(_g(b), 95) - np.percentile(_g(b), 70))
                  for b in betas]
        beta_fit = float(betas[int(np.argmin(widths))])
        h0 = float(np.percentile(_g(beta_fit), 90))
        LOG.info("Khớp sàn từ bàn chân: beta=%.2f deg (so với %.2f deg suy từ n_y; "
                 "chênh %.2f deg).", beta_fit, a_deg, beta_fit - a_deg)
        if cam_height is not None and h0 > 0:
            cands["sàn khớp từ bàn chân"] = D0 * float(cam_height) / h0
    if setup.get("camera_to_mirror") is not None:
        cands["thước dây x |n_z|"] = float(setup["camera_to_mirror"]) * abs(n_best[2])
    if setup.get("walkway_length") is not None:
        m_arr = mirror[:, perm] if swap_best else mirror
        ur = current.undistort_points(real[..., :2])
        um = current.undistort_points(m_arr[..., :2])
        hip = np.stack([ur[:, [11, 12]], um[:, [11, 12]]], axis=2)  # (T,2,2v,2)
        cf = np.stack([real[..., 2], m_arr[..., 2]], axis=2)[:, [11, 12]]
        good_t = np.isfinite(hip).all(axis=(1, 2, 3)) & (
            np.nan_to_num(cf, nan=0.0) >= args.min_confidence).all(axis=(1, 2))
        if good_t.sum() >= 30:
            sys_h = MirrorCameraSystem(K, n_best, D0)
            hp = hip[good_t].reshape(-1, 2, 2)
            rh = triangulate_points_dlt(sys_h.projection_matrices, hp, None, tri_cfg)
            pts_h = rh.points_3d.reshape(-1, 2, 3)
            ok_h = rh.valid.reshape(-1, 2).all(axis=1) & rh.in_front.reshape(-1, 2).all(axis=1)
            mid = pts_h[ok_h].mean(axis=1)
            if len(mid) >= 30:
                c = mid - mid.mean(axis=0)
                axis = np.linalg.svd(c, full_matrices=False)[2][0]
                proj = c @ axis
                extent = float(np.percentile(proj, 98) - np.percentile(proj, 2))
                if extent > 1e-6:
                    cands["chiều dài đường đi"] = D0 * float(setup["walkway_length"]) / extent
    if current.d > 0:
        cands["d trong cấu hình"] = float(current.d)
    LOG.info("%-24s %9s %22s %20s", "Nguồn d", "d (m)", "khoảng cách tới gương", "cao camera (m)")
    for name, dv in cands.items():
        sc = dv / D0
        LOG.info("%-24s %9.3f %22.3f %20.3f", name, dv, sc * mean_term,
                 sc * y_med if np.isfinite(y_med) else float("nan"))
    LOG.info("Kỳ vọng: khoảng cách tới gương %s m, cao camera %s m. Nếu các nguồn "
             "không khớp nhau (±10%%) thì một giả định đang sai.", walkway, cam_height)
    d_est: Optional[float] = None
    if c_best[0] > 3.0:
        LOG.warning("Reproj tối thiểu vẫn %.1f px: chưa tin được n, d bên dưới.",
                    c_best[0])

    print("\n--- Đề xuất cho configs/mirror.yaml ---")
    print(f"# hoán đổi trái-phải mirror_kpts: {'CÓ -> chạy 03 với --hypothesis swap' if swap_best else 'không'}")
    print("mirror:")
    print(f"  normal: [{n_best[0]:.4f}, {n_best[1]:.4f}, {n_best[2]:.4f}]")
    print(f"  d: {current.d:.4f}   # sai số tái chiếu không phụ thuộc d; xem bảng nguồn d ở trên")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())