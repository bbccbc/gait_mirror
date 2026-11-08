#!/usr/bin/env python3
"""Giai đoạn 2B: tinh chỉnh 3D toàn chuỗi (xương bất biến + mượt) + căn chỉnh sàn.

Ví dụ::

    python scripts/04_refine_3d.py \\
        --input_npz data/interim/poses_3d/dlt/IMG_8380_3d.npz \\
        --config_mirror configs/mirror.yaml \\
        --subject_height_m 1.65 --rescale_to_height --hip_raise_auto

Lưu ý quan trọng: ràng buộc độ dài xương KHÔNG sửa được sai tỉ lệ toàn cục (d).
Dùng --subject_height_m để so sánh; --rescale_to_height áp hệ số tỉ lệ đồng đều.
"""

from __future__ import annotations

import argparse
import json
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

from gaitmirror.reconstruction.refine_skeleton import (  # noqa: E402
    BONES, EAR_IDS, HIP_IDS, SHOULDER_IDS, ANKLE_IDS, RefineConfig,
    bone_length_series, build_world_transform, refine_pose_sequence,
    weights_from_reproj,
)

LOG = logging.getLogger("refine_3d")
FloatArray = NDArray[np.float64]
BONE_IDX = {b[0]: i for i, b in enumerate(BONES)}
KNEE_IDS = (13, 14)


def _nanmed(x: FloatArray) -> float:
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def _nanstd(x: FloatArray) -> float:
    x = x[np.isfinite(x)]
    return float(np.std(x)) if x.size > 1 else float("nan")


def trunk_axis(pose: FloatArray, up_fallback: FloatArray) -> FloatArray:
    """Vector đơn vị hông -> vai theo từng khung (T,3); fallback = hướng lên."""
    ms = np.nanmean(pose[:, list(SHOULDER_IDS)], axis=1)
    mh = np.nanmean(pose[:, list(HIP_IDS)], axis=1)
    u = ms - mh
    n = np.linalg.norm(u, axis=1)
    ok = np.isfinite(n) & (n > 1e-6)
    out = np.tile(up_fallback, (len(pose), 1))
    out[ok] = u[ok] / n[ok, None]
    return out


def raise_hips(pose: FloatArray, delta: float, u: FloatArray) -> FloatArray:
    out = pose.copy()
    for j in HIP_IDS:
        out[:, j] = out[:, j] + delta * u
    return out


def choose_hip_raise(
    pose: FloatArray, u: FloatArray, ratio: float, max_raise: float
) -> Tuple[float, float, float]:
    """Chọn δ chung hai bên để median(đùi) ≈ ratio * median(cẳng chân).

    Trả về (delta, thigh_med_sau, shank_med).
    """
    shank = np.concatenate([
        np.linalg.norm(pose[:, 13] - pose[:, 15], axis=1),
        np.linalg.norm(pose[:, 14] - pose[:, 16], axis=1)])
    shank_med = _nanmed(shank)
    best = (0.0, float("inf"), float("nan"))
    for delta in np.arange(0.0, max_raise + 1e-9, 0.0025):
        th = np.concatenate([
            np.linalg.norm(pose[:, 11] + delta * u - pose[:, 13], axis=1),
            np.linalg.norm(pose[:, 12] + delta * u - pose[:, 14], axis=1)])
        tm = _nanmed(th)
        gap = abs(tm - ratio * shank_med)
        if gap < best[1]:
            best = (float(delta), gap, tm)
    return best[0], best[2], shank_med


def stature_estimate(
    pose_world: FloatArray, lengths: FloatArray, up_idx: int, head_top_m: float
) -> Dict[str, float]:
    """Ước lượng chiều cao từ chuỗi keypoint (xấp xỉ, ±3 cm).

    stature = cao cổ chân + cẳng chân + đùi + thân(giữa hông->giữa vai)
              + (giữa vai->giữa tai) + head_top_m (tai -> đỉnh đầu, hằng số).
    Hông chỉ chia lại đùi/thân nên tổng không phụ thuộc vị trí hông.
    """
    ank_h = pose_world[:, list(ANKLE_IDS), up_idx]
    with np.errstate(all="ignore"):
        ankle = _nanmed(np.nanmin(np.where(np.isfinite(ank_h), ank_h, np.inf), axis=1))
    shank = np.nanmean([lengths[BONE_IDX["shank_L"]], lengths[BONE_IDX["shank_R"]]])
    thigh = np.nanmean([lengths[BONE_IDX["thigh_L"]], lengths[BONE_IDX["thigh_R"]]])
    ms = np.nanmean(pose_world[:, list(SHOULDER_IDS)], axis=1)
    mh = np.nanmean(pose_world[:, list(HIP_IDS)], axis=1)
    me = np.nanmean(pose_world[:, list(EAR_IDS)], axis=1)
    trunk = _nanmed(np.linalg.norm(ms - mh, axis=1))
    neck_head = _nanmed(np.linalg.norm(me - ms, axis=1))
    chain = float(shank + thigh + trunk + neck_head)
    return {"ankle_height": ankle, "shank": float(shank), "thigh": float(thigh),
            "trunk": trunk, "neck_head": neck_head, "head_top": head_top_m,
            "chain_sum": chain, "stature": float(ankle + chain + head_top_m)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input_npz", type=Path, required=True)
    p.add_argument("--config_mirror", type=Path, required=True)
    p.add_argument("--output_dir", type=Path,
                   default=Path("data/interim/poses_3d/optimized"))
    p.add_argument("--up_axis", choices=["y", "z"], default="y")
    p.add_argument("--subject_height_m", type=float, default=None)
    p.add_argument("--rescale_to_height", action="store_true",
                   help="Áp hệ số tỉ lệ đồng đều để chiều cao ước lượng = subject_height_m.")
    p.add_argument("--foot_marker_height_m", type=float, default=0.02,
                   help="Độ cao trung bình keypoint ngón/gót so với mặt sàn (m, xấp xỉ).")
    p.add_argument("--apply_yaml_d", action="store_true",
                   help="Nếu d trong mirror.yaml khác d đã dùng ở bước 03 (mirror_d trong "
                        "npz), nhân tỉ lệ pose cho khớp (pose tỉ lệ thuận với d).")
    p.add_argument("--head_top_m", type=float, default=0.12,
                   help="Khoảng cách trung bình tai -> đỉnh đầu (m).")
    p.add_argument("--hip_raise_m", type=float, default=0.0,
                   help="Nâng hông dọc trục thân (m), chung hai bên (áo dài kéo hông xuống).")
    p.add_argument("--hip_raise_auto", action="store_true",
                   help="Tự chọn nâng hông để đùi ≈ ratio * cẳng chân.")
    p.add_argument("--thigh_shank_ratio", type=float, default=1.0)
    p.add_argument("--hip_raise_max_m", type=float, default=0.12)
    p.add_argument("--sigma_data_m", type=float, default=None)
    p.add_argument("--sigma_bone_m", type=float, default=None)
    p.add_argument("--sigma_smooth_m", type=float, default=None)
    p.add_argument("--feet_smooth_factor", type=float, default=None)
    p.add_argument("--max_nfev", type=int, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg_yaml: Dict[str, Any] = yaml.safe_load(open(args.config_mirror, encoding="utf-8"))
    mir = cfg_yaml.get("mirror", cfg_yaml) or {}
    down = mir.get("vertical_down")
    if down is None:
        raise SystemExit("Thiếu mirror.vertical_down trong mirror.yaml "
                         "(chạy 03a_diagnose_mirror.py với --mirror_edge).")
    down_cam = np.asarray(down, float)
    down_cam = down_cam / np.linalg.norm(down_cam)

    rcfg = RefineConfig()
    for k, v in (cfg_yaml.get("refinement", {}) or {}).items():
        if hasattr(rcfg, k):
            setattr(rcfg, k, type(getattr(rcfg, k))(v))
    for name in ("sigma_data_m", "sigma_bone_m", "sigma_smooth_m",
                 "feet_smooth_factor", "max_nfev"):
        v = getattr(args, name)
        if v is not None:
            setattr(rcfg, name, type(getattr(rcfg, name))(v))
    subj_h = args.subject_height_m or (cfg_yaml.get("subject", {}) or {}).get("height_m")

    data = np.load(args.input_npz, allow_pickle=False)
    pose = np.asarray(data["pose_3d"], float)
    fps = float(data["fps"]) if "fps" in data.files else 60.0
    reproj = np.asarray(data["reproj_error"], float) if "reproj_error" in data.files \
        else np.full(pose.shape[:2] + (2,), 1.0)
    gap = np.asarray(data["gap_filled_mask"], bool) if "gap_filled_mask" in data.files else None
    t_len = pose.shape[0]
    d_npz = float(data["mirror_d"]) if "mirror_d" in data.files else None
    d_yaml = float(mir["d"]) if mir.get("d") is not None else None
    if d_npz and d_yaml and abs(d_yaml / d_npz - 1.0) > 0.01:
        if args.apply_yaml_d:
            k0 = d_yaml / d_npz
            pose = pose * k0
            LOG.info("d trong yaml (%.3f) khác d bước 03 (%.3f): nhân tỉ lệ pose x%.4f.",
                     d_yaml, d_npz, k0)
            d_npz = d_yaml
        else:
            LOG.warning("d trong yaml (%.3f) KHÁC d đã dùng ở bước 03 (%.3f): pose đang "
                        "ở tỉ lệ cũ. Chạy lại bước 03 hoặc thêm --apply_yaml_d.",
                        d_yaml, d_npz)
    d_in = d_npz or d_yaml
    LOG.info("Đọc %s: T=%d, valid %.1f%%", args.input_npz.name, t_len,
             100.0 * np.isfinite(pose).all(2).mean())

    # ---- nâng hông (tùy chọn) ----
    up_cam = -down_cam
    u_trunk = trunk_axis(pose, up_cam)
    delta = float(args.hip_raise_m)
    if args.hip_raise_auto:
        delta, th_med, sh_med = choose_hip_raise(
            pose, u_trunk, args.thigh_shank_ratio, args.hip_raise_max_m)
        LOG.info("Nâng hông tự động δ=%.1f cm (đùi %.3f m, cẳng chân %.3f m; "
                 "ratio mục tiêu %.2f). Đây là GIẢ ĐỊNH, không phải đo trực tiếp.",
                 100 * delta, th_med, sh_med, args.thigh_shank_ratio)
    pose_in = raise_hips(pose, delta, u_trunk) if delta != 0.0 else pose
    len_dlt = bone_length_series(pose_in)

    # ---- tối ưu ----
    w = weights_from_reproj(reproj, pose_in, gap, rcfg)
    res = refine_pose_sequence(pose_in, w, rcfg)
    LOG.info("Tối ưu: cost %.4g -> %.4g, nfev=%d (%s)", res.cost_init,
             res.cost_final, res.nfev, res.message)
    len_ref = bone_length_series(res.pose)

    # ---- hệ World ----
    tf = build_world_transform(res.pose, down_cam, args.up_axis,
                               foot_marker_height_m=args.foot_marker_height_m)
    up_idx = 1 if args.up_axis == "y" else 2
    pose_w = tf.apply(res.pose)
    pose_dlt_w = tf.apply(pose_in)
    lengths = res.lengths.copy()
    st = stature_estimate(pose_w, lengths, up_idx, args.head_top_m)

    scale = 1.0
    d_cfg = mir.get("d")
    stature_pre = st["stature"]
    cam_h_pre = -tf.floor_height_cam
    cfg_cam_h = (cfg_yaml.get("setup", {}) or {}).get("camera_height")
    if d_in:
        print("\n=== d suy ra từ dữ liệu (so với cạnh gương / chiều dài đường đi) ===")
        print(f"  d đang dùng cho pose này              : {d_in:.3f} m")
        if subj_h:
            print(f"  từ chiều cao đối tượng ({float(subj_h):.3f} m)      : "
                  f"{d_in * float(subj_h) / stature_pre:.3f} m   "
                  f"(chiều cao ước lượng hiện tại {stature_pre:.3f} m, công thức ~±3%)")
        if cfg_cam_h:
            print(f"  từ chiều cao camera ({float(cfg_cam_h):.3f} m)       : "
                  f"{d_in * float(cfg_cam_h) / cam_h_pre:.3f} m   "
                  f"(camera cao suy ra hiện tại {cam_h_pre:.3f} m; nhạy với vertical_down)")
        print("  Nếu các nguồn khớp nhau trong ~3% thì sửa d trong mirror.yaml rồi chạy lại 03.\n")
    if subj_h:
        ratio = float(subj_h) / st["stature"]
        LOG.info("Chiều cao ước lượng %.3f m so với đối tượng %.3f m: tỉ lệ %.4f",
                 st["stature"], float(subj_h), ratio)
        if d_cfg:
            LOG.info("=> d tương đương = %.3f m (d trong cấu hình %.3f m).",
                     float(d_cfg) * ratio, float(d_cfg))
        if args.rescale_to_height:
            scale = ratio
            pose_w, pose_dlt_w = pose_w * scale, pose_dlt_w * scale
            lengths, len_dlt, len_ref = lengths * scale, len_dlt * scale, len_ref * scale
            st = stature_estimate(pose_w, lengths, up_idx, args.head_top_m)
            LOG.info("Đã nhân tỉ lệ %.4f; chiều cao ước lượng mới %.3f m.", scale,
                     st["stature"])

    # ---- báo cáo ----
    print("\n=== Chiều dài xương danh định (m) ===")
    print(f"{'xương':<18}{'nhóm':<11}{'L bar':>8}{'DLT med':>9}{'std DLT mm':>12}{'std sau mm':>12}")
    rows = []
    for i, (name, _, _, grp) in enumerate(BONES):
        if not np.isfinite(lengths[i]):
            continue
        s0, s1 = 1000 * _nanstd(len_dlt[:, i]), 1000 * _nanstd(len_ref[:, i])
        rows.append((name, grp, float(lengths[i]), _nanmed(len_dlt[:, i]), s0, s1))
        print(f"{name:<18}{grp:<11}{lengths[i]:8.3f}{_nanmed(len_dlt[:, i]):9.3f}{s0:12.1f}{s1:12.1f}")
    m0 = np.nanmean([r[4] for r in rows])
    m1 = np.nanmean([r[5] for r in rows])
    print(f"{'TRUNG BÌNH':<29}{'':>17}{m0:12.1f}{m1:12.1f}")
    print("\n=== Bất đối xứng trái/phải (giữ nguyên, không ép đối xứng) ===")
    asym: Dict[str, Dict[str, float]] = {}
    for base in ("thigh", "shank", "torso", "ankle_heel", "upper_arm", "forearm"):
        l, r = lengths[BONE_IDX[f"{base}_L"]], lengths[BONE_IDX[f"{base}_R"]]
        if np.isfinite(l) and np.isfinite(r):
            asym[base] = {"L": float(l), "R": float(r), "diff_mm": 1000 * float(l - r),
                          "pct": 100 * float((l - r) / (0.5 * (l + r)))}
            print(f"{base:<12} L={l:.3f}  R={r:.3f}  L-R={1000*(l-r):+6.1f} mm ({100*(l-r)/(0.5*(l+r)):+.1f}%)")
    print("\n=== Chiều cao (ước lượng từ keypoint) ===")
    for k in ("ankle_height", "shank", "thigh", "trunk", "neck_head", "head_top"):
        print(f"  {k:<14}{st[k]:.3f} m")
    print(f"  {'Σ chuỗi (không gồm cổ chân, đỉnh đầu)':<40}{st['chain_sum']:.3f} m")
    print(f"  {'CHIỀU CAO ƯỚC LƯỢNG':<40}{st['stature']:.3f} m"
          + (f"   (đối tượng {float(subj_h):.3f} m)" if subj_h else ""))
    cam_h = -tf.floor_height_cam * scale
    print(f"Hệ World: {tf.axes}; sàn = 0. Chiều cao camera suy ra từ dữ liệu: "
          f"{cam_h:.3f} m"
          + (f" (đo thước: {float(cfg_cam_h):.3f} m)" if cfg_cam_h else "") + "\n")
    # ---- lưu ----
    pose_cam_out = res.pose * scale
    stem = args.input_npz.stem
    vid = stem[:-3] if stem.endswith("_3d") else stem
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{vid}_refined.npz"
    summary = {
        "n_frames": int(t_len), "fps": fps, "up_axis": args.up_axis,
        "world_axes": tf.axes, "scale_applied": scale, "hip_raise_m": delta,
        "stature": st, "subject_height_m": subj_h, "asymmetry": asym,
        "bone_std_mm_before": m0, "bone_std_mm_after": m1,
        "optimizer": {"cost_init": res.cost_init, "cost_final": res.cost_final,
                      "nfev": res.nfev},
        "config": {k: getattr(rcfg, k) for k in vars(rcfg)},
        "bones": [{"name": r[0], "group": r[1], "L": r[2], "dlt_median": r[3],
                   "std_dlt_mm": r[4], "std_refined_mm": r[5]} for r in rows],
    }
    np.savez_compressed(
        out_path,
        pose_3d=pose_w,  # hệ World (sàn = 0), thay cho pose_3d camera của bước 03
        pose_3d_cam=pose_cam_out,
        pose_3d_dlt_world=pose_dlt_w,
        valid_mask=np.isfinite(pose_w).all(axis=2),
        reproj_error=reproj,
        weights=w,
        fps=np.float64(fps),
        bone_names=np.array([b[0] for b in BONES]),
        bone_pairs=np.array([[b[1], b[2]] for b in BONES]),
        bone_lengths=lengths,
        R_world_from_cam=tf.R,
        t_world_from_cam=tf.t,
        world_from_cam_scale=np.float64(scale),
        world_axes=np.array(tf.axes),
        up_axis=np.array(args.up_axis),
        summary=np.array(json.dumps(summary, ensure_ascii=False)),
    )
    with open(out_path.with_suffix(".summary.json"), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, ensure_ascii=False, indent=2))
    LOG.info("Đã lưu: %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())