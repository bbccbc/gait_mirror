#!/usr/bin/env python3
"""Giai đoạn 2A: dựng 3D từ camera thật + camera ảo (gương) bằng DLT + LM.

Ví dụ::

    python scripts/03_reconstruct_dlt.py \\
        --input_npz data/interim/assigned_2d/walk01_filtered.npz \\
        --config_camera configs/camera.yaml \\
        --config_mirror configs/mirror.yaml \\
        --output_dir data/interim/poses_3d/dlt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

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

LOG = logging.getLogger("reconstruct_dlt")
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

N_JOINTS = 23
IMAGE_SIZE = (1920, 1080)


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: YAML phải là mapping ở cấp cao nhất.")
    return data


def fill_short_gaps(
    pose: FloatArray, max_gap: int
) -> Tuple[FloatArray, BoolArray]:
    """Nội suy tuyến tính 3D cho các đoạn NaN ngắn (<= max_gap) có hai đầu hợp lệ.

    Đoạn dài hơn, hoặc ở đầu/cuối chuỗi, giữ NaN.
    """
    out = pose.copy()
    filled = np.zeros(pose.shape[:2], dtype=bool)
    if max_gap <= 0:
        return out, filled
    for j in range(pose.shape[1]):
        valid_idx = np.flatnonzero(np.isfinite(pose[:, j]).all(axis=1))
        for a, b in zip(valid_idx[:-1], valid_idx[1:]):
            gap = b - a - 1
            if 0 < gap <= max_gap:
                t = np.arange(a + 1, b)
                alpha = ((t - a) / (b - a))[:, None]
                out[t, j] = (1 - alpha) * pose[a, j] + alpha * pose[b, j]
                filled[t, j] = True
    return out, filled


def _nanmean(x: FloatArray, axis: int = 0) -> FloatArray:
    cnt = np.isfinite(x).sum(axis=axis)
    tot = np.nansum(x, axis=axis)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def build_summary(
    reproj: FloatArray, valid: BoolArray, filled: BoolArray, fps: float
) -> Dict[str, Any]:
    t = reproj.shape[0]
    per_joint_real = _nanmean(reproj[..., 0])
    per_joint_mirror = _nanmean(reproj[..., 1])
    return {
        "n_frames": int(t),
        "fps": float(fps),
        "mean_reproj_real_px": per_joint_real.tolist(),
        "mean_reproj_mirror_px": per_joint_mirror.tolist(),
        "mean_reproj_real_all_px": float(_nanmean(reproj[..., 0].ravel())),
        "mean_reproj_mirror_all_px": float(_nanmean(reproj[..., 1].ravel())),
        "valid_ratio_per_joint": valid.mean(axis=0).tolist(),
        "gap_filled_ratio_per_joint": filled.mean(axis=0).tolist(),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input_npz", type=Path, required=True)
    p.add_argument("--config_camera", type=Path, required=True)
    p.add_argument("--config_mirror", type=Path, required=True)
    p.add_argument(
        "--output_dir", type=Path, default=Path("data/interim/poses_3d/dlt")
    )
    p.add_argument(
        "--min_confidence", type=float, default=None,
        help="Bỏ điểm (không nội suy) có conf thấp hơn ngưỡng. Mặc định: yaml/0.2",
    )
    p.add_argument(
        "--max_reproj_px", type=float, default=None,
        help="Loại điểm 3D nếu sai số tái chiếu ở bất kỳ view nào vượt ngưỡng.",
    )
    p.add_argument(
        "--max_gap_s", type=float, default=None,
        help="Độ dài tối đa (giây) của đoạn NaN 3D được nội suy lấp.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cam_cfg = load_yaml(args.config_camera)
    mir_cfg = load_yaml(args.config_mirror)
    system = MirrorCameraSystem.from_config(
        cam_cfg, mir_cfg, target_image_size=IMAGE_SIZE
    )
    LOG.info("%r", system)

    # Siêu tham số: ưu tiên mirror.yaml -> section `triangulation`
    tri_dict = dict(mir_cfg.get("triangulation", {}) or {})
    run_cfg = mir_cfg.get("reconstruction", {}) or {}
    tri_cfg = TriangulationConfig.from_dict(tri_dict)
    min_conf = (
        args.min_confidence
        if args.min_confidence is not None
        else float(run_cfg.get("min_confidence", 0.2))
    )
    max_reproj = (
        args.max_reproj_px
        if args.max_reproj_px is not None
        else float(run_cfg.get("max_reproj_px", 20.0))
    )
    max_gap_s = (
        args.max_gap_s
        if args.max_gap_s is not None
        else float(run_cfg.get("max_gap_s", 0.2))
    )

    setup = mir_cfg.get("setup", {}) or {}
    floor_y = run_cfg.get("floor_y_m", setup.get("camera_height"))
    floor_tol = float(run_cfg.get("floor_tol_m", 0.05))
    floor_policy = str(run_cfg.get("floor_policy", "warn"))  # warn|drop|clamp
    if floor_policy not in ("warn", "drop", "clamp"):
        raise ValueError("floor_policy phải là warn | drop | clamp.")
    walkway_expected = setup.get("walkway_to_mirror")

    data = np.load(args.input_npz, allow_pickle=False)
    real = np.asarray(data["real_kpts"], dtype=np.float64)
    mirror = np.asarray(data["mirror_kpts"], dtype=np.float64)
    fps = float(data["fps"])
    if real.shape != mirror.shape or real.ndim != 3 or real.shape[1:] != (N_JOINTS, 3):
        raise ValueError(
            f"Shape không hợp lệ: real {real.shape}, mirror {mirror.shape}; "
            f"cần (T, {N_JOINTS}, 3)."
        )
    t_len = real.shape[0]
    real_mask = (
        np.asarray(data["real_mask"], dtype=bool)
        if "real_mask" in data.files
        else np.zeros((t_len, N_JOINTS), bool)
    )
    mirror_mask = (
        np.asarray(data["mirror_mask"], dtype=bool)
        if "mirror_mask" in data.files
        else np.zeros((t_len, N_JOINTS), bool)
    )

    # --- (T, J, V, ...) với V = [real, mirror] ---
    pts = np.stack([real[..., :2], mirror[..., :2]], axis=2)  # (T,J,2,2)
    conf = np.stack([real[..., 2], mirror[..., 2]], axis=2)  # (T,J,2)
    interp = np.stack([real_mask, mirror_mask], axis=2)  # (T,J,2)

    pts = system.undistort_points(pts)
    # điểm thật (không nội suy) có conf thấp -> coi như thiếu
    low = (~interp) & (np.nan_to_num(conf, nan=0.0) < min_conf)
    pts = np.where(low[..., None], np.nan, pts)

    joint_ids = np.tile(np.arange(N_JOINTS), t_len)
    sig = build_sigmas(
        joint_ids, conf.reshape(-1, 2), interp.reshape(-1, 2), tri_cfg
    )

    LOG.info("Tam giác hóa %d điểm (T=%d, J=%d)...", t_len * N_JOINTS, t_len, N_JOINTS)
    res = triangulate_points_dlt(
        system.projection_matrices, pts.reshape(-1, 2, 2), sig, tri_cfg
    )

    pose = res.points_3d.reshape(t_len, N_JOINTS, 3).copy()
    reproj = res.reproj_error.reshape(t_len, N_JOINTS, 2).copy()
    keep = res.valid & res.in_front
    keep &= np.nan_to_num(res.reproj_error, nan=np.inf).max(axis=1) <= max_reproj
    keep = keep.reshape(t_len, N_JOINTS)
    pose[~keep] = np.nan
    reproj[~keep] = np.nan

    n_behind = int((res.valid & ~res.in_front).sum())
    if n_behind:
        LOG.warning(
            "%d điểm nằm sau camera/gương (kiểm tra dấu n, d hoặc thứ tự view).",
            n_behind,
        )

    n_below_floor = 0
    if floor_y is not None:
        # Hệ camera: Y hướng xuống, sàn ở Y = +camera_height (giả định trục
        # quang học nằm ngang). Điểm có Y > floor_y + tol là xuyên sàn.
        with np.errstate(invalid="ignore"):
            below = pose[..., 1] > float(floor_y) + floor_tol
        n_below_floor = int(below.sum())
        if n_below_floor:
            LOG.warning(
                "%d điểm xuyên sàn (Y > %.3f + %.2f m); policy=%s.",
                n_below_floor, float(floor_y), floor_tol, floor_policy,
            )
            if floor_policy == "drop":
                pose[below] = np.nan
                reproj[below] = np.nan
            elif floor_policy == "clamp":
                with np.errstate(invalid="ignore"):
                    over = pose[..., 1] > float(floor_y)
                pose[..., 1] = np.where(over, float(floor_y), pose[..., 1])

    max_gap = int(round(max_gap_s * fps))
    pose, filled = fill_short_gaps(pose, max_gap)
    valid_final = np.isfinite(pose).all(axis=2)

    summary = build_summary(reproj, keep, filled, fps)
    summary["n_below_floor"] = n_below_floor
    summary["floor_y_m"] = None if floor_y is None else float(floor_y)

    # Kiểm tra nhất quán hình học: khoảng cách có dấu từ người tới gương.
    # Người phải đứng cùng phía với camera (> 0) và gần walkway_to_mirror.
    finite_pts = pose[valid_final]
    if finite_pts.size:
        dist_mirror = finite_pts @ system.normal + system.d
        summary["mean_dist_to_mirror_m"] = float(dist_mirror.mean())
        summary["frac_behind_mirror"] = float((dist_mirror <= 0).mean())
        LOG.info(
            "Khoảng cách tới gương: TB %.3f m (frac sau gương %.1f%%)",
            summary["mean_dist_to_mirror_m"],
            100.0 * summary["frac_behind_mirror"],
        )
        if walkway_expected is not None and (
            abs(summary["mean_dist_to_mirror_m"] - float(walkway_expected)) > 0.2
        ):
            LOG.warning(
                "Khoảng cách TB tới gương %.2f m lệch walkway_to_mirror=%.2f m "
                "-> kiểm tra n, d (dấu, góc 26.6 độ, d vuông góc hay dọc trục).",
                summary["mean_dist_to_mirror_m"], float(walkway_expected),
            )
    LOG.info(
        "Reproj TB: real %.2f px | gương %.2f px | valid %.1f%%",
        summary["mean_reproj_real_all_px"],
        summary["mean_reproj_mirror_all_px"],
        100.0 * valid_final.mean(),
    )

    stem = args.input_npz.stem
    video_name = stem[: -len("_filtered")] if stem.endswith("_filtered") else stem
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{video_name}_3d.npz"
    summary_json = json.dumps(summary, ensure_ascii=False)
    np.savez_compressed(
        out_path,
        pose_3d=pose,
        reproj_error=reproj,
        summary=np.array(summary_json),
        valid_mask=valid_final,
        gap_filled_mask=filled,
        fps=np.float64(fps),
        K=system.K,
        mirror_normal=system.normal,
        mirror_d=np.float64(system.d),
    )
    with open(out_path.with_suffix(".summary.json"), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, ensure_ascii=False, indent=2))
    LOG.info("Đã lưu: %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())