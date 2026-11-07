"""Tam giác hóa đa view: Hartley-normalized DLT + Levenberg-Marquardt.

Thứ tự view trong toàn bộ module: view 0 = người thật, view 1 = gương.
Mọi hàm đều vector hóa theo số điểm N.

Cực tiểu hóa::

    min_X  sum_v (x_v - pi(P_v, X))^T Sigma_v^{-1} (x_v - pi(P_v, X)),
    Sigma_v = diag(sigma_u^2, sigma_v^2)
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

N_VIEWS_DEFAULT = 2  # [real, mirror]


@dataclass
class TriangulationConfig:
    """Siêu tham số tam giác hóa. Mọi sigma tính bằng pixel.

    Các tuple sigma có thứ tự (real, mirror).
    `foot_*` áp dụng cho các khớp trong `foot_joints` (chống bias bóng sàn):
    đặt foot_sigma_v > foot_sigma_u để giảm trọng số sai số theo phương dọc ảnh.
    """

    sigma_u: Tuple[float, float] = (1.0, 1.0)
    sigma_v: Tuple[float, float] = (1.0, 1.0)
    foot_sigma_u: Optional[Tuple[float, float]] = None  # None -> dùng sigma_u
    foot_sigma_v: Tuple[float, float] = (3.0, 3.0)
    foot_joints: Tuple[int, ...] = (17, 18, 19, 20, 21, 22)
    use_confidence: bool = True
    conf_floor: float = 0.05
    interp_sigma_scale: float = 2.0  # nhân sigma cho điểm nội suy
    max_iters: int = 30
    lm_lambda0: float = 1e-3
    step_tol: float = 1e-8  # mét
    refine: bool = True

    @classmethod
    def from_dict(cls, cfg: Optional[Mapping[str, Any]]) -> "TriangulationConfig":
        if not cfg:
            return cls()
        kw = dict(cfg)
        if "sigma_v_body" in kw:  # alias: sigma_v của thân trên + chân (0..16)
            kw["sigma_v"] = kw.pop("sigma_v_body")
        names = {f.name for f in fields(cls)}
        unknown = set(kw) - names
        if unknown:
            raise KeyError(f"Khóa triangulation không hợp lệ: {sorted(unknown)}")
        for key in ("sigma_u", "sigma_v", "foot_sigma_u", "foot_sigma_v"):
            if key in kw and kw[key] is not None:
                val = kw[key]
                kw[key] = (
                    (float(val), float(val))
                    if np.isscalar(val)
                    else (float(val[0]), float(val[1]))
                )
        if "foot_joints" in kw:
            kw["foot_joints"] = tuple(int(j) for j in kw["foot_joints"])
        return cls(**kw)


@dataclass
class TriangulationResult:
    """Kết quả tam giác hóa cho N điểm, V view."""

    points_3d: FloatArray  # (N, 3) mét; NaN nếu không dựng được
    residuals: FloatArray  # (N, V, 2) pixel có dấu (quan sát - dự đoán)
    reproj_error: FloatArray  # (N, V) pixel, chuẩn L2
    cost: FloatArray  # (N,) tổng bình phương residual đã whiten
    converged: BoolArray  # (N,)
    in_front: BoolArray  # (N,) độ sâu > 0 ở mọi view
    valid: BoolArray  # (N,) dựng thành công
    extras: dict = field(default_factory=dict)


# ---------------------------------------------------------------------- #
# Sigma
# ---------------------------------------------------------------------- #
def build_sigmas(
    joint_ids: NDArray[np.integer],
    confidences: Optional[FloatArray],
    interpolated: Optional[BoolArray],
    config: TriangulationConfig,
) -> FloatArray:
    """Tạo mảng sigma (N, V, 2) = (sigma_u, sigma_v) cho từng điểm/view.

    Args:
        joint_ids: (N,) chỉ số khớp của mỗi điểm.
        confidences: (N, V) độ tin cậy detector, hoặc None.
        interpolated: (N, V) True nếu điểm do nội suy, hoặc None.
    """
    n = joint_ids.shape[0]
    v = N_VIEWS_DEFAULT
    su = np.broadcast_to(np.asarray(config.sigma_u, float), (n, v)).copy()
    sv = np.broadcast_to(np.asarray(config.sigma_v, float), (n, v)).copy()

    foot = np.isin(joint_ids, np.asarray(config.foot_joints, dtype=int))
    fu = config.foot_sigma_u if config.foot_sigma_u is not None else config.sigma_u
    su[foot] = np.asarray(fu, float)
    sv[foot] = np.asarray(config.foot_sigma_v, float)

    scale = np.ones((n, v))
    if config.use_confidence and confidences is not None:
        conf = np.clip(np.nan_to_num(confidences, nan=0.0), config.conf_floor, 1.0)
        scale = scale / np.sqrt(conf)
    if interpolated is not None:
        scale = np.where(interpolated, scale * config.interp_sigma_scale, scale)
    return np.stack([su * scale, sv * scale], axis=-1)


# ---------------------------------------------------------------------- #
# Hartley normalization + DLT
# ---------------------------------------------------------------------- #
def hartley_normalization(pts: FloatArray) -> FloatArray:
    """Ma trận T (3x3) đưa tập điểm 2D về mean 0, khoảng cách TB = sqrt(2).

    Phải tính trên *tập nhiều điểm* của cùng một view (cả chuỗi video), không
    tính riêng cho từng điểm (khi đó sẽ suy biến).
    """
    mean = pts.mean(axis=0)
    dist = np.linalg.norm(pts - mean, axis=1).mean()
    s = np.sqrt(2.0) / max(dist, 1e-12)
    return np.array(
        [[s, 0.0, -s * mean[0]], [0.0, s, -s * mean[1]], [0.0, 0.0, 1.0]]
    )


def _dlt_linear(
    P_norm: FloatArray, x_norm: FloatArray, sig_norm: FloatArray
) -> Tuple[FloatArray, BoolArray]:
    """DLT tuyến tính (đã chuẩn hóa) cho M điểm. Trả về X (M,3), ok (M,)."""
    m, v, _ = x_norm.shape
    A = np.empty((m, 2 * v, 4))
    for i in range(v):
        p1, p2, p3 = P_norm[i, 0], P_norm[i, 1], P_norm[i, 2]
        A[:, 2 * i] = (x_norm[:, i, 0:1] * p3 - p1) / sig_norm[:, i, 0:1]
        A[:, 2 * i + 1] = (x_norm[:, i, 1:2] * p3 - p2) / sig_norm[:, i, 1:2]
    _, _, vt = np.linalg.svd(A)
    Xh = vt[:, -1, :]
    w = Xh[:, 3]
    ok = np.abs(w) > 1e-12
    X = np.full((m, 3), np.nan)
    X[ok] = Xh[ok, :3] / w[ok, None]
    return X, ok


# ---------------------------------------------------------------------- #
# Levenberg-Marquardt
# ---------------------------------------------------------------------- #
def _eval(
    P: FloatArray,
    X: FloatArray,
    obs: FloatArray,
    sig: FloatArray,
    need_jac: bool,
) -> Tuple[FloatArray, FloatArray, Optional[FloatArray], FloatArray]:
    """Residual đã whiten r (M,V,2), cost (M,), Jacobian (M,V,2,3), depth (M,V)."""
    m = X.shape[0]
    Xh = np.concatenate([X, np.ones((m, 1))], axis=1)
    pr = np.einsum("vij,mj->mvi", P, Xh)
    w = pr[..., 2]
    w_safe = np.where(np.abs(w) < 1e-9, 1e-9, w)
    uv = pr[..., :2] / w_safe[..., None]
    r = (obs - uv) / sig
    cost = np.sum(r * r, axis=(1, 2))
    jac = None
    if need_jac:
        # d(uv)/dX = (P[:2,:3] - uv * P[2,:3]) / w
        jac = (
            P[None, :, :2, :3] - uv[..., None] * P[None, :, None, 2, :3]
        ) / w_safe[..., None, None]
        jac = jac / sig[..., None]
    return r, cost, jac, w


def _refine_lm(
    P: FloatArray,
    obs: FloatArray,
    sig: FloatArray,
    X0: FloatArray,
    max_iters: int,
    lam0: float,
    tol: float,
) -> Tuple[FloatArray, BoolArray]:
    """LM vector hóa theo điểm. Trả về X đã tinh chỉnh và cờ hội tụ."""
    X = X0.copy()
    m = X.shape[0]
    lam = np.full(m, lam0)
    _, cost, _, _ = _eval(P, X, obs, sig, need_jac=False)
    converged = np.zeros(m, dtype=bool)
    eye = np.eye(3)

    for _ in range(max_iters):
        act = ~converged
        if not act.any():
            break
        r, cost_cur, J, _ = _eval(P, X, obs, sig, need_jac=True)
        assert J is not None
        JtJ = np.einsum("mvki,mvkj->mij", J, J)
        g = np.einsum("mvki,mvk->mi", J, r)
        diag = np.einsum("mii->mi", JtJ)
        Aug = JtJ + lam[:, None, None] * (diag[:, :, None] * eye) + 1e-12 * eye
        try:
            delta = np.linalg.solve(Aug, g[..., None])[..., 0]
        except np.linalg.LinAlgError:
            delta = np.zeros_like(g)
        X_new = X + delta
        _, cost_new, _, _ = _eval(P, X_new, obs, sig, need_jac=False)
        accept = act & np.isfinite(cost_new) & (cost_new < cost_cur)
        X[accept] = X_new[accept]
        cost = np.where(accept, cost_new, cost_cur)
        lam = np.where(accept, np.maximum(lam / 10.0, 1e-9), np.minimum(lam * 10.0, 1e8))
        small = np.max(np.abs(delta), axis=1) < tol
        converged |= small
        # Điểm bị từ chối liên tục với lam lớn cũng coi như đã dừng
        converged |= (~accept) & (lam >= 1e8)
    return X, converged


# ---------------------------------------------------------------------- #
# API chính
# ---------------------------------------------------------------------- #
def triangulate_points_dlt(
    projections: Sequence[ArrayLike],
    points_2d: ArrayLike,
    sigmas: Optional[ArrayLike] = None,
    config: Optional[TriangulationConfig] = None,
) -> TriangulationResult:
    """Tam giác hóa N điểm từ V view (thường V=2: real + gương).

    Args:
        projections: V ma trận chiếu 3x4 (pixel, đã gỡ méo).
        points_2d: (N, V, 2) tọa độ pixel đã gỡ méo; NaN = thiếu.
        sigmas: (N, V, 2) = (sigma_u, sigma_v) pixel; None -> 1 px.
        config: siêu tham số LM.

    Returns:
        TriangulationResult. Điểm có NaN ở bất kỳ view nào (hoặc thiếu số
        view tối thiểu) cho X = NaN, valid = False.
    """
    cfg = config or TriangulationConfig()
    P = np.stack([np.asarray(p, dtype=np.float64) for p in projections])
    v = P.shape[0]
    if v < 2:
        raise ValueError("Cần tối thiểu 2 view.")
    pts = np.asarray(points_2d, dtype=np.float64)
    if pts.ndim != 3 or pts.shape[1] != v or pts.shape[2] != 2:
        raise ValueError(f"points_2d phải có shape (N, {v}, 2), nhận {pts.shape}.")
    n = pts.shape[0]
    sig = (
        np.ones((n, v, 2))
        if sigmas is None
        else np.asarray(sigmas, dtype=np.float64)
    )
    if sig.shape != (n, v, 2):
        raise ValueError(f"sigmas phải có shape {(n, v, 2)}, nhận {sig.shape}.")

    use = (
        np.isfinite(pts).all(axis=(1, 2))
        & np.isfinite(sig).all(axis=(1, 2))
        & (sig > 0).all(axis=(1, 2))
    )
    X_out = np.full((n, 3), np.nan)
    resid = np.full((n, v, 2), np.nan)
    cost_out = np.full(n, np.nan)
    conv = np.zeros(n, dtype=bool)
    front = np.zeros(n, dtype=bool)
    valid = np.zeros(n, dtype=bool)

    if use.sum() >= 3:
        idx = np.flatnonzero(use)
        x = pts[idx]
        s = sig[idx]

        # --- Bước 1: Hartley-normalized DLT ---
        T = np.stack([hartley_normalization(x[:, i, :]) for i in range(v)])
        P_n = np.einsum("vij,vjk->vik", T, P)
        x_n = np.empty_like(x)
        sig_n = np.empty_like(s)
        for i in range(v):
            scale = T[i, 0, 0]
            x_n[:, i, :] = (x[:, i, :] - (-T[i, :2, 2] / scale)) * scale
            sig_n[:, i, :] = s[:, i, :] * scale
        X0, ok = _dlt_linear(P_n, x_n, sig_n)

        # --- Bước 2: LM tối thiểu sai số tái chiếu có trọng số ---
        if cfg.refine:
            good = np.flatnonzero(ok)
            X_ref = X0.copy()
            c_ref = np.zeros(idx.size, dtype=bool)
            if good.size:
                Xg, cg = _refine_lm(
                    P, x[good], s[good], X0[good],
                    cfg.max_iters, cfg.lm_lambda0, cfg.step_tol,
                )
                X_ref[good] = Xg
                c_ref[good] = cg
        else:
            X_ref, c_ref = X0, ok.copy()

        r_w, cost, _, depth = _eval(
            P, np.nan_to_num(X_ref, nan=0.0), x, s, need_jac=False
        )
        r_px = r_w * s  # đổi về pixel
        good_pt = ok & np.isfinite(cost)
        X_ref[~good_pt] = np.nan
        X_out[idx] = X_ref
        resid[idx] = np.where(good_pt[:, None, None], r_px, np.nan)
        cost_out[idx] = np.where(good_pt, cost, np.nan)
        conv[idx] = c_ref & good_pt
        front[idx] = (depth > 0).all(axis=1) & good_pt
        valid[idx] = good_pt

    reproj = np.linalg.norm(resid, axis=2)
    return TriangulationResult(
        points_3d=X_out,
        residuals=resid,
        reproj_error=reproj,
        cost=cost_out,
        converged=conv,
        in_front=front,
        valid=valid,
    )