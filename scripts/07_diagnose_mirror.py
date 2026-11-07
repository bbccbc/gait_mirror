#!/usr/bin/env python3
"""Chẩn đoán hình học gương từ dữ liệu 2D (không dùng n, d nhập tay).

Ý tưởng:
  1. Gương thẳng đứng => n = (sin phi, 0, -cos phi), chỉ còn 1 ẩn: phi.
  2. Quét phi, tam giác hóa một tập điểm con, chọn phi có sai số tái chiếu
     (median, px) nhỏ nhất. Sai số tái chiếu KHÔNG phụ thuộc d (d chỉ đổi tỉ lệ
     3D) nên d lấy từ ràng buộc ngoài: bàn chân ở Y = camera_height.
  3. In so sánh với cấu hình hiện tại và đoạn YAML đề xuất.

Ví dụ::

    python scripts/03a_diagnose_mirror.py \\
        --input_npz data/interim/assigned_2d/IMG_8380_filtered.npz \\
        --config_camera configs/camera.yaml \\
        --config_mirror configs/mirror.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
D0 = 1.0  # d tùy ý: chỉ đổi tỉ lệ, không đổi sai số tái chiếu


def normal_from_phi(phi_deg: float) -> FloatArray:
    p = np.deg2rad(phi_deg)
    return np.array([np.sin(p), 0.0, -np.cos(p)])


def phi_from_normal(n: FloatArray) -> float:
    return float(np.degrees(np.arctan2(n[0], -n[2])))


def evaluate(
    K: FloatArray,
    phi_deg: float,
    pts: FloatArray,
    sig: FloatArray,
    cfg: TriangulationConfig,
) -> Tuple[float, float, FloatArray, FloatArray]:
    """Trả về (median reproj px, valid_frac, X (N,3), valid (N,))."""
    system = MirrorCameraSystem(K, normal_from_phi(phi_deg), D0)
    res = triangulate_points_dlt(system.projection_matrices, pts, sig, cfg)
    ok = res.valid & res.in_front
    frac = float(ok.mean())
    if frac < 0.2:
        return 1e3, frac, res.points_3d, ok
    err = res.reproj_error[ok].mean(axis=1)
    return float(np.median(err)), frac, res.points_3d, ok


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

    with open(args.config_camera, "r", encoding="utf-8") as fh:
        cam_cfg: Dict[str, Any] = yaml.safe_load(fh)
    with open(args.config_mirror, "r", encoding="utf-8") as fh:
        mir_cfg: Dict[str, Any] = yaml.safe_load(fh)
    setup = mir_cfg.get("setup", {}) or {}
    cam_height = setup.get("camera_height")
    walkway = setup.get("walkway_to_mirror")

    current = MirrorCameraSystem.from_config(
        cam_cfg, mir_cfg, target_image_size=(1920, 1080)
    )
    K = current.K
    tri_cfg = TriangulationConfig.from_dict(mir_cfg.get("triangulation"))
    tri_cfg.max_iters = 8

    data = np.load(args.input_npz, allow_pickle=False)
    real, mirror = data["real_kpts"], data["mirror_kpts"]
    t_len, n_j, _ = real.shape
    pts = np.stack([real[..., :2], mirror[..., :2]], axis=2)
    conf = np.stack([real[..., 2], mirror[..., 2]], axis=2)
    interp = np.zeros((t_len, n_j, 2), bool)
    for i, key in enumerate(("real_mask", "mirror_mask")):
        if key in data.files:
            interp[..., i] = data[key]
    pts = current.undistort_points(pts)
    good = (
        np.isfinite(pts).all(axis=(2, 3))
        & (np.nan_to_num(conf, nan=0.0) >= args.min_confidence).all(axis=2)
        & ~interp.any(axis=2)
    )
    tt, jj = np.nonzero(good)
    if tt.size < 100:
        raise SystemExit(f"Quá ít điểm hợp lệ ({tt.size}).")
    rng = np.random.default_rng(args.seed)
    sel = rng.choice(tt.size, size=min(args.n_points, tt.size), replace=False)
    tt, jj = tt[sel], jj[sel]
    p_sub, c_sub = pts[tt, jj], conf[tt, jj]
    sig = build_sigmas(jj, c_sub, None, tri_cfg)
    LOG.info("Dùng %d điểm để quét phi.", tt.size)

    # --- Quét phi ---
    phis = np.arange(-85.0, 85.0 + 1e-9, args.phi_step)
    costs = np.array([evaluate(K, p, p_sub, sig, tri_cfg)[0] for p in phis])
    best = int(np.argmin(costs))
    phi_best = float(phis[best])
    try:
        from scipy.optimize import minimize_scalar

        r = minimize_scalar(
            lambda p: evaluate(K, p, p_sub, sig, tri_cfg)[0],
            bounds=(phi_best - 2.0, phi_best + 2.0),
            method="bounded",
        )
        phi_best = float(r.x)
    except ImportError:
        pass
    cost_best, frac_best, X, ok = evaluate(K, phi_best, p_sub, sig, tri_cfg)

    # Tối ưu cục bộ thứ hai ở phía đối diện dấu để cho thấy độ phân biệt
    other = costs[np.sign(phis) != np.sign(phi_best)]
    LOG.info(
        "phi tốt nhất = %.2f deg: median reproj %.2f px (valid %.0f%%). "
        "Tốt nhất phía dấu ngược lại: %.2f px.",
        phi_best, cost_best, 100 * frac_best, float(other.min()),
    )

    # --- Hiện tại ---
    phi_cfg = phi_from_normal(current.normal)
    c_cfg = evaluate(K, phi_cfg, p_sub, sig, tri_cfg)
    LOG.info(
        "Cấu hình hiện tại: phi=%.2f deg: median reproj %.2f px (valid %.0f%%).",
        phi_cfg, c_cfg[0], 100 * c_cfg[1],
    )

    # --- d từ chiều cao camera (sàn Y = camera_height) ---
    n_best = normal_from_phi(phi_best)
    foot = np.isin(jj, FOOT_IDS) & ok
    d_est = None
    if cam_height is not None and foot.sum() >= 20:
        y_med = float(np.median(X[foot, 1]))
        if y_med > 0:
            scale = float(cam_height) / y_med
            d_est = D0 * scale
            X_s = X[ok] * scale
            dist = X_s @ n_best + d_est
            LOG.info(
                "Tỉ lệ từ sàn: %.4f => d = %.4f m. Khoảng cách TB người-gương: "
                "%.3f m (kỳ vọng walkway_to_mirror = %s).",
                scale, d_est, float(dist.mean()), walkway,
            )
        else:
            LOG.warning("Y bàn chân trung vị <= 0: camera nghiêng hoặc sai hình học.")
    else:
        LOG.warning("Không đủ điểm bàn chân hoặc thiếu setup.camera_height.")

    if cost_best > 3.0:
        LOG.warning(
            "Sai số tái chiếu tối thiểu vẫn %.1f px: vấn đề có thể nằm ở gán "
            "real/mirror, lật trái-phải của mirror_kpts, hoặc K/méo, không chỉ ở n.",
            cost_best,
        )

    print("\n--- Đề xuất cho configs/mirror.yaml ---")
    print("mirror:")
    print(f"  normal: [{n_best[0]:.4f}, 0.0, {n_best[2]:.4f}]")
    if d_est is not None:
        print(f"  d: {d_est:.4f}")
    print("  # xóa image_side / use_setup khi dùng normal + d tường minh")
    print(f"# angle_to_mirror_deg (góc trục-mặt gương) ≈ {90.0 - abs(phi_best):.1f}")
    if d_est is not None:
        print(
            "# camera_to_mirror (dọc trục) ≈ "
            f"{d_est / max(np.sin(np.deg2rad(90 - abs(phi_best))), 1e-6):.2f} m"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())