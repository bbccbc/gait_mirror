"""Tinh chỉnh khung xương 3D toàn chuỗi + căn chỉnh hệ tọa độ sàn.

Hàm mục tiêu (dạng bình phương tối thiểu có trọng số, mỗi số hạng chia cho
độ lệch chuẩn mong đợi sigma, tương đương w = 1 / sigma^2)::

    E = sum_{t,j} w_tj ||p_tj - d_tj||^2 / sd^2                 (bám DLT)
      + sum_{t,b} (||p_t,b1 - p_t,b2|| - L_b)^2 / sb^2          (xương bất biến)
      + sum_{t,j} ||p_{t+1,j} - 2 p_tj + p_{t-1,j}||^2 / ss_j^2 (mượt gia tốc)

Biến: toàn bộ điểm hợp lệ (N_v x 3) và vector độ dài L (n_bones). Trái và phải
được tối ưu ĐỘC LẬP (không ép đối xứng) để giữ được chênh lệch chiều dài chi
(LLD). Điểm NaN ở đầu vào không bị bịa ra: đầu ra vẫn NaN ở đó.

Quy ước hệ camera: OpenCV (x phải, y xuống, z ra trước). `down_cam` là vector
đơn vị chỉ hướng xuống (trọng lực) trong hệ camera (mirror.vertical_down).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.optimize import least_squares

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

N_JOINTS = 23
# Quy ước: COCO-17 + [LBT=17, LST=18, LH=19, RBT=20, RST=21, RH=22]
HIP_IDS = (11, 12)
SHOULDER_IDS = (5, 6)
EAR_IDS = (3, 4)
ANKLE_IDS = (15, 16)
FOOT_IDS = (17, 18, 19, 20, 21, 22)
FEET_SMOOTH_IDS = ANKLE_IDS + FOOT_IDS

# (tên, khớp 1, khớp 2, nhóm). Trái (_L) và phải (_R) là hai biến riêng.
BONES: List[Tuple[str, int, int, str]] = [
    ("pelvis_width", 11, 12, "thân"),
    ("shoulder_width", 5, 6, "thân"),
    ("torso_L", 5, 11, "thân"),
    ("torso_R", 6, 12, "thân"),
    ("thigh_L", 11, 13, "đùi"),
    ("thigh_R", 12, 14, "đùi"),
    ("shank_L", 13, 15, "cẳng chân"),
    ("shank_R", 14, 16, "cẳng chân"),
    ("ankle_heel_L", 15, 19, "bàn chân"),
    ("ankle_heel_R", 16, 22, "bàn chân"),
    ("heel_bigtoe_L", 19, 17, "bàn chân"),
    ("heel_bigtoe_R", 22, 20, "bàn chân"),
    ("heel_smalltoe_L", 19, 18, "bàn chân"),
    ("heel_smalltoe_R", 22, 21, "bàn chân"),
    ("ankle_bigtoe_L", 15, 17, "bàn chân"),
    ("ankle_bigtoe_R", 16, 20, "bàn chân"),
    ("ankle_smalltoe_L", 15, 18, "bàn chân"),
    ("ankle_smalltoe_R", 16, 21, "bàn chân"),
    ("forefoot_width_L", 17, 18, "bàn chân"),
    ("forefoot_width_R", 20, 21, "bàn chân"),
    ("upper_arm_L", 5, 7, "cánh tay"),
    ("upper_arm_R", 6, 8, "cánh tay"),
    ("forearm_L", 7, 9, "cánh tay"),
    ("forearm_R", 8, 10, "cánh tay"),
    ("ear_width", 3, 4, "đầu"),
    ("nose_ear_L", 0, 3, "đầu"),
    ("nose_ear_R", 0, 4, "đầu"),
]


@dataclass
class RefineConfig:
    """Siêu tham số tối ưu hóa cơ sinh học (đơn vị mét)."""

    sigma_data_m: float = 0.03
    sigma_bone_m: float = 0.004
    sigma_smooth_m: float = 0.006
    feet_smooth_factor: float = 2.0  # nới sigma mượt cho cổ chân/bàn chân
    reproj_ref_px: float = 3.0
    gap_fill_weight: float = 0.25
    min_joint_valid_ratio: float = 0.05
    max_nfev: int = 80

    # --- RÀNG BUỘC NHÂN TRẮC HỌC KHẮC PHỤC LỖI ÁO THỤNG ---
    # Khớp háng (11, 12) dễ bị nhiễu khi mặc áo thụng: hạ trọng số bám DLT
    # để optimizer ưu tiên tỉ lệ cơ thể hơn là bám sát tọa độ thô.
    hip_weight_factor: float = 0.15      # Nhân trọng số háng (0 < x <= 1)
    target_shank_thigh_ratio: float = 0.85  # Tỉ lệ cẳng chân / đùi (~0.83–0.86)
    sigma_ratio_m: float = 0.005         # Độ siết ràng buộc tỉ lệ (mét)


@dataclass
class RefineResult:
    pose: FloatArray  # (T, J, 3), NaN ở điểm không có dữ liệu đầu vào
    lengths: FloatArray  # (nb,) độ dài danh định L_b (NaN nếu bỏ xương)
    lengths_init: FloatArray  # (nb,) median độ dài DLT thô
    valid: BoolArray  # (T, J)
    cost_init: float
    cost_final: float
    nfev: int
    message: str


def weights_from_reproj(
    reproj: FloatArray,
    pose: FloatArray,
    gap_filled: Optional[BoolArray],
    cfg: RefineConfig,
) -> FloatArray:
    """Trọng số bám dữ liệu w_tj, nới lỏng riêng cho khớp háng (11, 12).

    Khớp háng thường bị che khuất bởi quần áo rộng (áo thụng) làm detection
    sai lệch. Hạ hệ số ``hip_weight_factor`` để optimizer ưu tiên ràng buộc
    tỉ lệ cơ thể hơn là bám sát tọa độ DLT thô tại hai khớp này.
    """
    err = np.nanmean(reproj, axis=2) if reproj.ndim == 3 else reproj
    w = 1.0 / (1.0 + (err / cfg.reproj_ref_px) ** 2)
    finite = np.isfinite(pose).all(axis=2)
    w = np.where(np.isfinite(w), w, cfg.gap_fill_weight)  # nội suy: tin ít
    if gap_filled is not None:
        w = np.where(gap_filled, cfg.gap_fill_weight, w)

    # Nới lỏng bám DLT ở khớp háng để ưu tiên tỉ lệ cơ thể
    w[:, list(HIP_IDS)] *= cfg.hip_weight_factor

    return np.where(finite, w, 0.0)


def refine_pose_sequence(
    pose: FloatArray,
    weights: FloatArray,
    cfg: Optional[RefineConfig] = None,
    bones: Sequence[Tuple[str, int, int, str]] = BONES,
) -> RefineResult:
    """Tối ưu toàn chuỗi. pose: (T, J, 3) NaN = thiếu; weights: (T, J) >= 0.

    Hàm mục tiêu có 3 nhóm số hạng (+ 2 ràng buộc nhân trắc học mới):
      - Bám DLT:    weighted || p_tj - d_tj || / sd
      - Xương:      (||p_b1 - p_b2|| - L_b) / sbn  (Trái/Phải độc lập cho LLD)
      - Mượt:       (p_{t-1} - 2 p_t + p_{t+1}) / ss_j
      - Tỉ lệ L:    (L_shank_L - ratio * L_thigh_L) / sigma_ratio
      - Tỉ lệ R:    (L_shank_R - ratio * L_thigh_R) / sigma_ratio  (độc lập Trái/Phải)
    """
    cfg = cfg or RefineConfig()
    t_len, n_j, _ = pose.shape
    valid = np.isfinite(pose).all(axis=2) & (weights > 0)
    valid[:, valid.mean(axis=0) < cfg.min_joint_valid_ratio] = False

    pid = -np.ones((t_len, n_j), dtype=np.int64)
    n_v = int(valid.sum())
    pid[valid] = np.arange(n_v)
    x0_pts = pose[valid].astype(np.float64)
    w_pt = np.sqrt(weights[valid])

    # ---- Xương: các cặp (t, b) ----
    b1 = np.array([b[1] for b in bones])
    b2 = np.array([b[2] for b in bones])
    n_b = len(bones)
    pa, pb = pid[:, b1], pid[:, b2]
    tt, bb = np.nonzero((pa >= 0) & (pb >= 0))
    ia, ib = pa[tt, bb], pb[tt, bb]
    d0 = np.linalg.norm(x0_pts[ia] - x0_pts[ib], axis=1)
    lengths0 = np.full(n_b, np.nan)
    for k in range(n_b):
        sel = bb == k
        if sel.sum() >= 10:
            lengths0[k] = float(np.median(d0[sel]))
    keep_b = np.isfinite(lengths0[bb])
    ia, ib, bb = ia[keep_b], ib[keep_b], bb[keep_b]

    # ---- Mượt thời gian ----
    p_a, p_b, p_c = pid[:-2], pid[1:-1], pid[2:]
    ts, js = np.nonzero((p_a >= 0) & (p_b >= 0) & (p_c >= 0))
    sa, sb_, sc = p_a[ts, js], p_b[ts, js], p_c[ts, js]
    feet = np.isin(js, FEET_SMOOTH_IDS)
    sig_s = cfg.sigma_smooth_m * np.where(feet, cfg.feet_smooth_factor, 1.0)

    # ---- Lấy index xương đùi/cẳng để ràng buộc tỉ lệ (Trái/Phải độc lập) ----
    bone_names = [b[0] for b in bones]
    idx_thigh_L = bone_names.index("thigh_L")
    idx_shank_L = bone_names.index("shank_L")
    idx_thigh_R = bone_names.index("thigh_R")
    idx_shank_R = bone_names.index("shank_R")

    sd, sbn = cfg.sigma_data_m, cfg.sigma_bone_m
    sig_ratio = cfg.sigma_ratio_m
    ratio_target = cfg.target_shank_thigh_ratio

    n_x = 3 * n_v
    x_init = np.concatenate([x0_pts.ravel(), np.nan_to_num(lengths0, nan=0.0)])
    n_data, n_bone, n_sm = 3 * n_v, len(ia), 3 * len(sa)

    def fun(x: FloatArray) -> FloatArray:
        pts = x[:n_x].reshape(n_v, 3)
        lens = x[n_x:]
        r_data = (w_pt[:, None] * (pts - x0_pts) / sd).ravel()
        dn = np.linalg.norm(pts[ia] - pts[ib], axis=1)
        r_bone = (dn - lens[bb]) / sbn
        r_sm = ((pts[sa] - 2.0 * pts[sb_] + pts[sc]) / sig_s[:, None]).ravel()
        # Ràng buộc tỉ lệ cẳng/đùi — Trái và Phải độc lập (không ép đối xứng LLD)
        r_ratio_L = (lens[idx_shank_L] - ratio_target * lens[idx_thigh_L]) / sig_ratio
        r_ratio_R = (lens[idx_shank_R] - ratio_target * lens[idx_thigh_R]) / sig_ratio
        r_ratio = np.array([r_ratio_L, r_ratio_R], dtype=np.float64)
        return np.concatenate([r_data, r_bone, r_sm, r_ratio])

    def jac(x: FloatArray) -> sparse.csr_matrix:
        pts = x[:n_x].reshape(n_v, 3)
        rows: List[NDArray] = []
        cols: List[NDArray] = []
        vals: List[NDArray] = []

        # 1. Data jacobian
        idx = np.arange(n_x)
        rows.append(idx)
        cols.append(idx)
        vals.append(np.repeat(w_pt / sd, 3))

        # 2. Bone jacobian
        base = n_data
        dv = pts[ia] - pts[ib]
        dn = np.maximum(np.linalg.norm(dv, axis=1), 1e-9)
        u = dv / dn[:, None] / sbn
        r_idx = base + np.arange(n_bone)
        for k in range(3):
            rows.append(r_idx)
            cols.append(3 * ia + k)
            vals.append(u[:, k])
            rows.append(r_idx)
            cols.append(3 * ib + k)
            vals.append(-u[:, k])
        rows.append(r_idx)
        cols.append(n_x + bb)
        vals.append(np.full(n_bone, -1.0 / sbn))

        # 3. Smooth jacobian
        base = n_data + n_bone
        s_idx = base + np.arange(len(sa))[:, None] * 3 + np.arange(3)[None, :]
        for pidx, coef in ((sa, 1.0), (sb_, -2.0), (sc, 1.0)):
            c_idx = 3 * pidx[:, None] + np.arange(3)[None, :]
            rows.append(s_idx.ravel())
            cols.append(c_idx.ravel())
            vals.append(np.repeat(coef / sig_s, 3))

        # 4. Anthropometric ratio jacobian (2 hàng: Trái, Phải)
        base_ratio = n_data + n_bone + n_sm
        # Hàng 0: shank_L - ratio * thigh_L = 0
        rows.append(np.array([base_ratio, base_ratio]))
        cols.append(np.array([n_x + idx_shank_L, n_x + idx_thigh_L]))
        vals.append(np.array([1.0 / sig_ratio, -ratio_target / sig_ratio]))
        # Hàng 1: shank_R - ratio * thigh_R = 0
        rows.append(np.array([base_ratio + 1, base_ratio + 1]))
        cols.append(np.array([n_x + idx_shank_R, n_x + idx_thigh_R]))
        vals.append(np.array([1.0 / sig_ratio, -ratio_target / sig_ratio]))

        n_rows = n_data + n_bone + n_sm + 2
        return sparse.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_rows, n_x + n_b),
        ).tocsr()

    cost0 = 0.5 * float(np.sum(fun(x_init) ** 2))
    sol = least_squares(
        fun, x_init, jac=jac, method="trf", tr_solver="lsmr", x_scale="jac",
        max_nfev=cfg.max_nfev, ftol=1e-6, xtol=1e-8, gtol=1e-8,
    )
    out = np.full_like(pose, np.nan, dtype=np.float64)
    out[valid] = sol.x[:n_x].reshape(n_v, 3)
    lengths = sol.x[n_x:].copy()
    lengths[~np.isfinite(lengths0)] = np.nan
    return RefineResult(
        pose=out, lengths=lengths, lengths_init=lengths0, valid=valid,
        cost_init=cost0, cost_final=float(sol.cost), nfev=int(sol.nfev),
        message=str(sol.message),
    )


def bone_length_series(
    pose: FloatArray, bones: Sequence[Tuple[str, int, int, str]] = BONES
) -> FloatArray:
    """Độ dài từng xương theo thời gian, (T, nb); NaN nếu thiếu đầu mút."""
    b1 = np.array([b[1] for b in bones])
    b2 = np.array([b[2] for b in bones])
    return np.linalg.norm(pose[:, b1] - pose[:, b2], axis=2)


# ---------------------------------------------------------------------- #
# Căn chỉnh sàn
# ---------------------------------------------------------------------- #
@dataclass
class WorldTransform:
    """X_world = R @ X_cam + t.  Mặt sàn tại thành phần 'up' = 0."""

    R: FloatArray  # (3, 3)
    t: FloatArray  # (3,)
    up_axis: str  # "y" hoặc "z"
    floor_height_cam: float  # độ cao sàn so với camera dọc phương thẳng đứng (m)
    axes: str

    def apply(self, pts: FloatArray) -> FloatArray:
        return pts @ self.R.T + self.t


def build_world_transform(
    pose_cam: FloatArray,
    down_cam: Sequence[float],
    up_axis: str = "y",
    start_frac: float = 0.05,
) -> WorldTransform:
    """Hệ World từ phương thẳng đứng, hướng đi và mặt sàn.

    * up = -down (đơn vị). Độ cao một điểm so với camera: h = -X . down.
    * Sàn: trung vị theo khung của độ cao thấp nhất trong các điểm bàn chân
      (chân đang chống luôn là điểm thấp nhất). Đặt sàn = 0.
    * Hướng tiến (X): trục chính của quỹ đạo ngang của khung chậu, dấu theo
      chiều đi từ đầu đến cuối chuỗi. Gốc ngang = vị trí chậu lúc bắt đầu.
    * up_axis="y": (X tiến, Y lên, Z = X x Y: sang phải đối tượng).
      up_axis="z": (X tiến, Y trái, Z lên).  Cả hai đều là hệ thuận tay phải.
    """
    if up_axis not in ("y", "z"):
        raise ValueError("up_axis phải là 'y' hoặc 'z'.")
    down = np.asarray(down_cam, dtype=np.float64).reshape(3)
    down = down / np.linalg.norm(down)
    up = -down

    height = -(pose_cam @ down)  # (T, J)
    foot_h = height[:, FOOT_IDS]
    with np.errstate(all="ignore"):
        min_foot = np.nanmin(np.where(np.isfinite(foot_h), foot_h, np.inf), axis=1)
    min_foot = min_foot[np.isfinite(min_foot)]
    if min_foot.size < 10:
        raise ValueError("Không đủ khung có bàn chân hợp lệ để xác định mặt sàn.")
    floor = float(np.median(min_foot))

    pel = np.nanmean(pose_cam[:, list(HIP_IDS)], axis=1)  # (T, 3)
    ok = np.isfinite(pel).all(axis=1)
    pel_h = pel - (pel @ down)[:, None] * down
    ph = pel_h[ok]
    if len(ph) < 10:
        raise ValueError("Không đủ khung chậu hợp lệ để xác định hướng đi.")
    centered = ph - ph.mean(axis=0)
    fwd = np.linalg.svd(centered, full_matrices=False)[2][0]
    k = max(int(start_frac * len(ph)), 1)
    if float((ph[-k:].mean(axis=0) - ph[:k].mean(axis=0)) @ fwd) < 0:
        fwd = -fwd
    fwd = fwd - (fwd @ up) * up
    fwd = fwd / np.linalg.norm(fwd)
    p0 = ph[:k].mean(axis=0)

    if up_axis == "y":
        third = np.cross(fwd, up)  # X x Y
        rows = np.stack([fwd, up, third])
        axes = "X=tiến, Y=lên, Z=phải"
    else:
        left = np.cross(up, fwd)  # Z x X
        rows = np.stack([fwd, left, up])
        axes = "X=tiến, Y=trái, Z=lên"
    t = -rows @ p0
    up_row = 1 if up_axis == "y" else 2
    t[up_row] = -floor
    return WorldTransform(R=rows, t=t, up_axis=up_axis,
                          floor_height_cam=floor, axes=axes)

# ====================================================================== #
# CLI, BÁO CÁO NHÂN TRẮC HỌC VÀ LƯU FILE
# ====================================================================== #
import argparse
from pathlib import Path
import yaml


def calculate_anatomical_height(lengths_dict: Dict[str, float]) -> float:
    """Tính tổng chiều cao giải phẫu chuẩn: chân + đùi + thân + cổ/đầu + gót."""
    shank = np.nanmean([lengths_dict.get("shank_L", np.nan), lengths_dict.get("shank_R", np.nan)])
    thigh = np.nanmean([lengths_dict.get("thigh_L", np.nan), lengths_dict.get("thigh_R", np.nan)])
    torso = np.nanmean([lengths_dict.get("torso_L", np.nan), lengths_dict.get("torso_R", np.nan)])
    
    # Cổ + đầu chuẩn nhân trắc học (từ vai lên đỉnh sọ): ~28 cm
    # Mắt cá chân xuống mặt sàn (gót chân + đế dép): ~7 cm
    head_and_neck = 0.28
    ankle_to_floor = 0.07
    
    return float(shank + thigh + torso + head_and_neck + ankle_to_floor)


def main():
    parser = argparse.ArgumentParser(description="Tinh chỉnh 3D Pose cơ sinh học + Căn chỉnh mặt sàn")
    parser.add_argument("--input_npz", type=str, required=True, help="File DLT 3D thô (vd: IMG_8387_3d.npz)")
    parser.add_argument("--config_mirror", type=str, default="configs/mirror.yaml", help="Cấu hình gương")
    parser.add_argument("--output_dir", type=str, default="data/interim/poses_3d/optimized")
    parser.add_argument("--target_height", type=float, default=None, help="Chiều cao đối tượng (m) để cân chỉnh d toàn cục")
    parser.add_argument("--up_axis", type=str, default="y", choices=["y", "z"], help="Trục hướng lên trong hệ World")
    args = parser.parse_args()

    input_path = Path(args.input_npz)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/5] Đang nạp dữ liệu từ {input_path.name}...")
    data = np.load(input_path, allow_pickle=True)
    pose_raw = data["pose_3d"].copy()  # (T, 23, 3)
    reproj_err = data["reproj_err"] if "reproj_err" in data else np.zeros(pose_raw.shape[:2])

    with open(args.config_mirror, "r", encoding="utf-8") as f:
        m_cfg = yaml.safe_load(f)
    down_cam = m_cfg.get("vertical_down", [0.0364, 0.9974, 0.0628])
    d_current = float(m_cfg.get("d", 2.71))

    # ---- 2. Hiệu chỉnh tỉ lệ toàn cục (nếu có target_height) ----
    scale_factor = 1.0
    if args.target_height is not None:
        print(f"\n[2/5] Đang kiểm tra tỉ lệ chiều cao mục tiêu: {args.target_height:.3f} m...")
        # Tính chiều cao sơ bộ ban đầu
        b_series = bone_length_series(pose_raw, BONES)
        init_lengths = {b[0]: float(np.nanmedian(b_series[:, i])) for i, b in enumerate(BONES)}
        h_init = calculate_anatomical_height(init_lengths)
        
        scale_factor = args.target_height / h_init
        pose_raw *= scale_factor
        d_equiv = d_current * scale_factor
        print(f"  -> Chiều cao ước tính thô ban đầu: {h_init:.3f} m")
        print(f"  -> Hệ số co giãn (scale): {scale_factor:.4f}")
        print(f"  -> Scale d tương đương cho configs/mirror.yaml: {d_equiv:.4f} m (gốc {d_current} m)")
    else:
        print("\n[2/5] Bỏ qua hiệu chỉnh scale (chạy trực tiếp scale từ dLT)...")

    # ---- 3. Tối ưu chuỗi pose ----
    print("\n[3/5] Đang tối ưu hóa chiều dài xương (Rigid Bone Optimization)...")
    cfg = RefineConfig()
    weights = weights_from_reproj(reproj_err, pose_raw, None, cfg)
    res = refine_pose_sequence(pose_raw, weights, cfg=cfg, bones=BONES)
    print(f"  -> Hoàn tất tối ưu ({res.nfev} iterations): Cost ban đầu = {res.cost_init:.1f} -> Cost sau tối ưu = {res.cost_final:.1f}")

    # ---- 4. Căn chỉnh hệ trục mặt sàn (World Coordinates) ----
    print(f"\n[4/5] Đang căn chỉnh mặt sàn theo trọng lực (up_axis='{args.up_axis}')...")
    w_trans = build_world_transform(res.pose, down_cam=down_cam, up_axis=args.up_axis)
    pose_world = np.full_like(res.pose, np.nan)
    valid_mask = np.isfinite(res.pose).all(axis=2)
    pose_world[valid_mask] = w_trans.apply(res.pose[valid_mask])

    # ---- 5. Báo cáo nhân trắc học & Lưu file ----
    b_dict = {b[0]: float(res.lengths[i]) for i, b in enumerate(BONES) if np.isfinite(res.lengths[i])}
    final_height = calculate_anatomical_height(b_dict)

    print("\n" + "="*50)
    print("BÁO CÁO NHÂN TRẮC HỌC CHI TIẾT SAU KHI REFINEMENT")
    print("="*50)
    print(f"{'Đoạn xương':<20} | {'Chiều dài (cm)':<15}")
    print("-" * 50)
    for name in ["thigh_L", "thigh_R", "shank_L", "shank_R", "torso_L", "torso_R", "pelvis_width"]:
        if name in b_dict:
            print(f"{name:<20} | {b_dict[name]*100:.2f} cm")
    print("-" * 50)
    print(f"-> TỔNG CHIỀU CAO GIẢI PHẪU: {final_height:.3f} m ({final_height*100:.1f} cm)")
    print("="*50)

    # Lưu kết quả
    stem = input_path.stem.replace("_3d", "")
    out_file = output_dir / f"{stem}_refined.npz"
    np.savez_compressed(
        out_file,
        pose_3d=pose_world,             # Đã đưa về hệ sàn World (Y=0 là mặt sàn)
        pose_cam=res.pose,              # Tọa độ trong hệ camera đã nắn xương
        lengths=res.lengths,            # Độ dài bất biến từng xương
        bone_names=np.array([b[0] for b in BONES]),
        target_height=args.target_height if args.target_height else final_height,
        scale_applied=scale_factor,
        world_R=w_trans.R,
        world_t=w_trans.t,
    )
    print(f"\n[5/5] ĐÃ LƯU THÀNH CÔNG: {out_file}\n")


if __name__ == "__main__":
    main()