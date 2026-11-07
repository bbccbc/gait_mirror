#!/usr/bin/env python3
"""Giai đoạn 2A: dựng 3D từ camera thật + camera ảo (gương) bằng DLT + LM.

Chọn giả thuyết gán nhãn trái-phải của mirror_kpts THEO TỪNG KHUNG:
  A = giữ nguyên (mirror_kpts_raw), B = hoán đổi các cặp đối xứng.
Cả hai được tam giác hóa; chi phí mỗi khung = median sai số tái chiếu (px) +
phạt khớp không hợp lệ (kể cả điểm nằm sau gương). Bộ giải Viterbi 2 trạng
thái với phạt chuyển đổi chọn chuỗi giả thuyết tối ưu (nhãn thường giữ ổn
định qua nhiều khung rồi mới đổi).

Ví dụ::

    python scripts/03_reconstruct_dlt.py \\
        --input_npz data/interim/assigned_2d/IMG_8380_filtered.npz \\
        --config_camera configs/camera.yaml \\
        --config_mirror configs/mirror.yaml \\
        --output_dir data/interim/poses_3d/dlt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
# Quy ước 23 điểm: COCO-17 + [LBT, LST, LH, RBT, RST, RH] (17..22)
DEFAULT_LR_PAIRS: List[Tuple[int, int]] = [
    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
    (17, 20), (18, 21), (19, 22),
]
COST_CAP_PX = 60.0


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: YAML phải là mapping ở cấp cao nhất.")
    return data


def lr_permutation(pairs: Sequence[Tuple[int, int]]) -> NDArray[np.int64]:
    perm = np.arange(N_JOINTS)
    for a, b in pairs:
        perm[a], perm[b] = b, a
    return perm


def fill_short_gaps(pose: FloatArray, max_gap: int) -> Tuple[FloatArray, BoolArray]:
    """Nội suy tuyến tính 3D cho đoạn NaN ngắn (<= max_gap) có hai đầu hợp lệ."""
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


# ---------------------------------------------------------------------- #
# Một giả thuyết: tam giác hóa toàn chuỗi
# ---------------------------------------------------------------------- #
def run_hypothesis(
    system: MirrorCameraSystem,
    tri_cfg: TriangulationConfig,
    real: FloatArray,
    mirror: FloatArray,
    real_mask: BoolArray,
    mirror_mask: BoolArray,
    und_real: FloatArray,
    min_conf: float,
    max_reproj: float,
    require_front: bool,
) -> Dict[str, FloatArray]:
    """Tam giác hóa (T,J) điểm cho một cấu hình mirror. Trả về dict mảng."""
    t_len = real.shape[0]
    und_mirror = system.undistort_points(mirror[..., :2])
    pts = np.stack([und_real, und_mirror], axis=2)  # (T,J,2,2)
    conf = np.stack([real[..., 2], mirror[..., 2]], axis=2)  # (T,J,2)
    interp = np.stack([real_mask, mirror_mask], axis=2)
    low = (~interp) & (np.nan_to_num(conf, nan=0.0) < min_conf)
    pts = np.where(low[..., None], np.nan, pts)

    joint_ids = np.tile(np.arange(N_JOINTS), t_len)
    sig = build_sigmas(joint_ids, conf.reshape(-1, 2), interp.reshape(-1, 2), tri_cfg)
    res = triangulate_points_dlt(
        system.projection_matrices, pts.reshape(-1, 2, 2), sig, tri_cfg
    )
    keep = res.valid & res.in_front
    keep &= np.nan_to_num(res.reproj_error, nan=np.inf).max(axis=1) <= max_reproj
    pose = res.points_3d.reshape(t_len, N_JOINTS, 3).copy()
    if require_front:
        with np.errstate(invalid="ignore"):
            side = pose @ system.normal + system.d > 0.0
        keep = keep.reshape(t_len, N_JOINTS) & side
    else:
        keep = keep.reshape(t_len, N_JOINTS)
    reproj = res.reproj_error.reshape(t_len, N_JOINTS, 2).copy()
    pose[~keep] = np.nan
    reproj[~keep] = np.nan
    return {"pose": pose, "reproj": reproj, "keep": keep}


def frame_costs(
    reproj: FloatArray, keep: BoolArray, min_joints: int, invalid_penalty: float
) -> FloatArray:
    """Chi phí mỗi khung (px): median sai số + phạt tỉ lệ khớp không hợp lệ."""
    err = np.where(keep, reproj.mean(axis=2), np.nan)
    n_valid = keep.sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(err, axis=1)
    cost = med + invalid_penalty * (1.0 - n_valid / keep.shape[1])
    cost = np.where(n_valid >= min_joints, cost, COST_CAP_PX)
    return np.minimum(np.nan_to_num(cost, nan=COST_CAP_PX), COST_CAP_PX)


def viterbi_two_state(costs: FloatArray, switch_penalty: float) -> BoolArray:
    """Chọn chuỗi trạng thái {0,1} cực tiểu tổng chi phí + phạt mỗi lần đổi.

    costs: (T, 2). Trả về mảng bool (T,): True = trạng thái 1 (hoán đổi).
    """
    t_len = costs.shape[0]
    dp = costs[0].copy()
    back = np.zeros((t_len, 2), dtype=np.int64)
    for t in range(1, t_len):
        new = np.empty(2)
        for h in (0, 1):
            stay, switch = dp[h], dp[1 - h] + switch_penalty
            if switch < stay:
                back[t, h], best = 1 - h, switch
            else:
                back[t, h], best = h, stay
            new[h] = best + costs[t, h]
        dp = new
    state = np.zeros(t_len, dtype=np.int64)
    state[-1] = int(np.argmin(dp))
    for t in range(t_len - 1, 0, -1):
        state[t - 1] = back[t, state[t]]
    return state.astype(bool)


def build_summary(
    reproj: FloatArray, valid: BoolArray, filled: BoolArray, fps: float
) -> Dict[str, Any]:
    return {
        "n_frames": int(reproj.shape[0]),
        "fps": float(fps),
        "mean_reproj_real_px": _nanmean(reproj[..., 0]).tolist(),
        "mean_reproj_mirror_px": _nanmean(reproj[..., 1]).tolist(),
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
    p.add_argument("--output_dir", type=Path,
                   default=Path("data/interim/poses_3d/dlt"))
    p.add_argument("--hypothesis", choices=["dual", "raw", "swap"], default="dual",
                   help="dual: chọn theo khung; raw: luôn giữ nguyên; "
                        "swap: luôn hoán đổi trái-phải.")
    p.add_argument("--mirror_key", choices=["auto", "raw", "mirror_kpts"],
                   default="auto", help="Khóa mirror trong npz (auto ưu tiên raw).")
    p.add_argument("--switch_penalty", type=float, default=None,
                   help="Phạt (px) mỗi lần đổi giả thuyết giữa hai khung liền kề.")
    p.add_argument("--min_confidence", type=float, default=None)
    p.add_argument("--max_reproj_px", type=float, default=None)
    p.add_argument("--max_gap_s", type=float, default=None)
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

    tri_cfg = TriangulationConfig.from_dict(mir_cfg.get("triangulation"))
    run_cfg = mir_cfg.get("reconstruction", {}) or {}
    min_conf = (args.min_confidence if args.min_confidence is not None
                else float(run_cfg.get("min_confidence", 0.2)))
    max_reproj = (args.max_reproj_px if args.max_reproj_px is not None
                  else float(run_cfg.get("max_reproj_px", 20.0)))
    max_gap_s = (args.max_gap_s if args.max_gap_s is not None
                 else float(run_cfg.get("max_gap_s", 0.2)))
    switch_penalty = (args.switch_penalty if args.switch_penalty is not None
                      else float(run_cfg.get("switch_penalty_px", 4.0)))
    min_joints = int(run_cfg.get("min_valid_joints", 8))
    invalid_penalty = float(run_cfg.get("invalid_penalty_px", 15.0))
    require_front = bool(run_cfg.get("require_front_of_mirror", True))
    lr_pairs = [tuple(p) for p in run_cfg.get("lr_pairs", DEFAULT_LR_PAIRS)]
    perm = lr_permutation(lr_pairs)

    setup = mir_cfg.get("setup", {}) or {}
    floor_y = run_cfg.get("floor_y_m", setup.get("camera_height"))
    floor_tol = float(run_cfg.get("floor_tol_m", 0.05))
    floor_policy = str(run_cfg.get("floor_policy", "warn"))
    if floor_policy not in ("warn", "drop", "clamp"):
        raise ValueError("floor_policy phải là warn | drop | clamp.")
    walkway_expected = setup.get("walkway_to_mirror")

    data = np.load(args.input_npz, allow_pickle=False)
    real = np.asarray(data["real_kpts"], dtype=np.float64)
    fps = float(data["fps"])

    key = args.mirror_key
    if key == "auto":
        key = "raw" if "mirror_kpts_raw" in data.files else "mirror_kpts"
    mirror_name = "mirror_kpts_raw" if key == "raw" else "mirror_kpts"
    if mirror_name not in data.files:
        raise KeyError(f"Không có khóa '{mirror_name}' trong {args.input_npz} "
                       f"(có: {data.files}).")
    mirror = np.asarray(data[mirror_name], dtype=np.float64)
    if mirror_name == "mirror_kpts":
        LOG.warning("Dùng 'mirror_kpts' (không phải raw): có thể đã bị lật tĩnh. "
                    "Chế độ dual vẫn chọn đúng theo khung nhưng nên dùng raw.")
    if real.shape != mirror.shape or real.ndim != 3 or real.shape[1:] != (N_JOINTS, 3):
        raise ValueError(f"Shape không hợp lệ: real {real.shape}, mirror "
                         f"{mirror.shape}; cần (T, {N_JOINTS}, 3).")
    t_len = real.shape[0]
    zeros = np.zeros((t_len, N_JOINTS), bool)
    real_mask = np.asarray(data["real_mask"], bool) if "real_mask" in data.files else zeros
    mirror_mask = (np.asarray(data["mirror_mask"], bool)
                   if "mirror_mask" in data.files else zeros)

    und_real = system.undistort_points(real[..., :2])
    common = dict(
        system=system, tri_cfg=tri_cfg, real=real, real_mask=real_mask,
        und_real=und_real, min_conf=min_conf, max_reproj=max_reproj,
        require_front=require_front,
    )
    LOG.info("Tam giác hóa %d điểm x 2 giả thuyết (T=%d, J=%d)...",
             t_len * N_JOINTS, t_len, N_JOINTS)
    hyp_a = run_hypothesis(mirror=mirror, mirror_mask=mirror_mask, **common)
    hyp_b = run_hypothesis(mirror=mirror[:, perm], mirror_mask=mirror_mask[:, perm],
                           **common)

    cost_a = frame_costs(hyp_a["reproj"], hyp_a["keep"], min_joints, invalid_penalty)
    cost_b = frame_costs(hyp_b["reproj"], hyp_b["keep"], min_joints, invalid_penalty)
    costs = np.stack([cost_a, cost_b], axis=1)

    if args.hypothesis == "raw":
        swap = np.zeros(t_len, bool)
    elif args.hypothesis == "swap":
        swap = np.ones(t_len, bool)
    else:
        swap = viterbi_two_state(costs, switch_penalty)
    n_switch = int(np.abs(np.diff(swap.astype(int))).sum())
    chosen_cost = np.where(swap, cost_b, cost_a)
    LOG.info("Chi phí TB/khung (px): A(raw) %.2f | B(hoán đổi) %.2f | đã chọn %.2f",
             cost_a.mean(), cost_b.mean(), chosen_cost.mean())
    LOG.info("Hoán đổi trái-phải ở %.1f%% khung, %d lần đổi giả thuyết.",
             100.0 * swap.mean(), n_switch)

    sw3 = swap[:, None, None]
    pose = np.where(sw3, hyp_b["pose"], hyp_a["pose"])
    reproj = np.where(sw3, hyp_b["reproj"], hyp_a["reproj"])
    keep = np.where(swap[:, None], hyp_b["keep"], hyp_a["keep"])
    mirror_corr = np.where(sw3, mirror[:, perm], mirror)
    mirror_mask_corr = np.where(swap[:, None], mirror_mask[:, perm], mirror_mask)

    n_below_floor = 0
    if floor_y is not None:
        with np.errstate(invalid="ignore"):
            below = pose[..., 1] > float(floor_y) + floor_tol
        n_below_floor = int(below.sum())
        if n_below_floor:
            LOG.warning("%d điểm xuyên sàn (Y > %.3f + %.2f m); policy=%s.",
                        n_below_floor, float(floor_y), floor_tol, floor_policy)
            if floor_policy == "drop":
                pose[below] = np.nan
                reproj[below] = np.nan
            elif floor_policy == "clamp":
                with np.errstate(invalid="ignore"):
                    over = pose[..., 1] > float(floor_y)
                pose[..., 1] = np.where(over, float(floor_y), pose[..., 1])

    pose, filled = fill_short_gaps(pose, int(round(max_gap_s * fps)))
    valid_final = np.isfinite(pose).all(axis=2)

    summary = build_summary(reproj, keep, filled, fps)
    summary.update({
        "hypothesis_mode": args.hypothesis,
        "mirror_key": mirror_name,
        "swap_ratio": float(swap.mean()),
        "n_switches": n_switch,
        "mean_frame_cost_A_px": float(cost_a.mean()),
        "mean_frame_cost_B_px": float(cost_b.mean()),
        "mean_frame_cost_chosen_px": float(chosen_cost.mean()),
        "n_below_floor": n_below_floor,
        "floor_y_m": None if floor_y is None else float(floor_y),
    })
    finite_pts = pose[valid_final]
    if finite_pts.size:
        dist_mirror = finite_pts @ system.normal + system.d
        summary["mean_dist_to_mirror_m"] = float(dist_mirror.mean())
        summary["frac_behind_mirror"] = float((dist_mirror <= 0).mean())
        LOG.info("Khoảng cách tới gương: TB %.3f m (frac sau gương %.1f%%)",
                 summary["mean_dist_to_mirror_m"],
                 100.0 * summary["frac_behind_mirror"])
        if walkway_expected is not None and (
            abs(summary["mean_dist_to_mirror_m"] - float(walkway_expected)) > 0.2
        ):
            LOG.warning("Khoảng cách TB tới gương %.2f m lệch walkway_to_mirror="
                        "%.2f m -> kiểm tra n, d (chạy 03a_diagnose_mirror.py).",
                        summary["mean_dist_to_mirror_m"], float(walkway_expected))
    LOG.info("Reproj TB: real %.2f px | gương %.2f px | valid %.1f%%",
             summary["mean_reproj_real_all_px"],
             summary["mean_reproj_mirror_all_px"], 100.0 * valid_final.mean())
    if summary["mean_reproj_real_all_px"] > 4.0:
        LOG.warning("Reproj vẫn cao (>4 px): ghép cặp còn sai ở các nhóm khớp "
                    "riêng lẻ, hoặc K/n/d chưa đúng. Chạy 03a_diagnose_mirror.py.")

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
        mirror_swap=swap,
        mirror_kpts_corrected=mirror_corr,
        mirror_mask_corrected=mirror_mask_corr,
        frame_cost=costs,
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