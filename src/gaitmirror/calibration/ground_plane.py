"""Mặt sàn: nghiệm TĨNH toàn cục trong hệ camera thật.

QUY ƯỚC::

    normal_up : vector đơn vị, hướng LÊN (ngược trọng lực), trong hệ camera
    offset    : độ cao camera so với mặt sàn (m, cùng đơn vị với điểm 3D)
    height(X) = normal_up . X + offset   (= 0 trên sàn, > 0 phía trên)

Tương thích mirror.yaml cũ: `mirror.vertical_down` = -normal_up, `setup.camera_height`
= offset. Khối mới `ground:` (nếu có) được ưu tiên.

BA CHẾ ĐỘ::

    1. config    : GroundPlane.from_config / from_vertical_down (vertical_down + chiều cao).
    2. envelope  : fit_ground_plane(..., method="envelope")   <- KHUYẾN NGHỊ
    3. ransac    : fit_ground_plane(..., method="ransac")     (mặt phẳng tự do 3 bậc)

VÌ SAO "envelope" + pháp tuyến gương: đi thẳng một đường thì mặt phẳng tự do bị suy biến
quanh trục đi (các điểm gần như thẳng hàng; chỉ hai bàn chân cách nhau ~0.2 m ràng buộc
độ nghiêng ngang). Nếu gương thẳng đứng thì n_g vuông góc n_m, ràng buộc này triệt tiêu
độ nghiêng ngang khi hướng đi song song gương, chỉ còn độ nghiêng DỌC hướng đi, và đó
chính là thứ đo được từ độ dốc đường bao dưới của độ cao bàn chân theo quãng đường đi.
Mỗi ràng buộc bù cho thành phần mà ràng buộc kia không quan sát được.

GIẢ ĐỊNH (phải kiểm chứng): (a) gương thẳng đứng (mirror_normal vuông góc trọng lực);
(b) sàn phẳng; (c) khi đặt sàn, các điểm bàn chân ở phân vị thấp nhất là điểm chạm sàn
(có >= ~floor_quantile% điểm đang chạm); (d) keypoint ngón/gót cách sàn marker_height_m.
Điểm 3D phải ở TỈ LỆ đúng (cùng d): offset tỉ lệ thuận với d, dùng `rescaled`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

FOOT_IDS = (17, 18, 19, 20, 21, 22)


def _unit(v: ArrayLike) -> FloatArray:
    a = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        raise ValueError("Vector có độ dài ~0.")
    return a / n


@dataclass(frozen=True)
class GroundPlane:
    normal_up: FloatArray
    offset: float
    d_ref: Optional[float] = None  # d dùng khi dựng điểm 3D (để đổi tỉ lệ về sau)
    method: str = "config"
    diagnostics: Dict[str, float] = field(default_factory=dict)

    # ---------------------------------------------------------------- tạo
    @classmethod
    def from_vertical_down(
        cls, down: ArrayLike, camera_height: float, d_ref: Optional[float] = None,
        method: str = "config",
    ) -> "GroundPlane":
        if camera_height <= 0:
            raise ValueError("camera_height phải > 0.")
        return cls(normal_up=-_unit(down), offset=float(camera_height),
                   d_ref=d_ref, method=method)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "GroundPlane":
        """Đọc mirror.yaml: khối `ground:` > (`mirror.vertical_down` + `setup.camera_height`)."""
        g = cfg.get("ground")
        if g:
            return cls(
                normal_up=_unit(g["normal_up"]), offset=float(g["offset"]),
                d_ref=None if g.get("d_ref") is None else float(g["d_ref"]),
                method=str(g.get("method", "config")),
            )
        mir = cfg.get("mirror", cfg) or {}
        setup = cfg.get("setup", {}) or {}
        if mir.get("vertical_down") is None or setup.get("camera_height") is None:
            raise KeyError("Thiếu `ground:` hoặc (mirror.vertical_down + setup.camera_height).")
        d_ref = mir.get("d")
        return cls.from_vertical_down(
            mir["vertical_down"], float(setup["camera_height"]),
            d_ref=None if d_ref is None else float(d_ref),
        )

    def to_yaml_dict(self) -> Dict[str, Any]:
        """Khối `ground:` để dán vào mirror.yaml (kèm nguồn gốc, không ghi đè khóa cũ)."""
        out: Dict[str, Any] = {
            "normal_up": [float(v) for v in self.normal_up],
            "offset": float(self.offset), "method": self.method,
        }
        if self.d_ref is not None:
            out["d_ref"] = float(self.d_ref)
        return {"ground": out}

    def legacy_keys(self) -> Dict[str, Any]:
        """Khóa cũ cho code đang chạy: mirror.vertical_down, setup.camera_height."""
        return {"vertical_down": [float(v) for v in -self.normal_up],
                "camera_height": float(self.offset)}

    # ------------------------------------------------------------ hình học
    @property
    def down(self) -> FloatArray:
        return -self.normal_up

    def height(self, X: ArrayLike) -> FloatArray:
        """Độ cao so với sàn (m). X: (..., 3)."""
        return np.asarray(X, dtype=np.float64) @ self.normal_up + self.offset

    def rescaled(self, d_new: float) -> "GroundPlane":
        """Đổi tỉ lệ khi d đổi (điểm 3D tỉ lệ thuận với d)."""
        if self.d_ref is None:
            raise ValueError("Không có d_ref để đổi tỉ lệ.")
        k = float(d_new) / self.d_ref
        return GroundPlane(self.normal_up, self.offset * k, float(d_new),
                           self.method, dict(self.diagnostics))

    def angle_to_deg(self, other: "GroundPlane") -> float:
        c = float(np.clip(self.normal_up @ other.normal_up, -1.0, 1.0))
        return float(np.degrees(np.arccos(c)))


def contact_candidates(
    pose: ArrayLike, foot_ids: Sequence[int] = FOOT_IDS
) -> FloatArray:
    """Gom điểm bàn chân (N, 3) hợp lệ từ pose (T, J, 3). Không lọc pha chạm sàn."""
    p = np.asarray(pose, dtype=np.float64)[:, list(foot_ids)].reshape(-1, 3)
    return p[np.isfinite(p).all(axis=1)]


# ---------------------------------------------------------------------- #
# Fit
# ---------------------------------------------------------------------- #
def _rotate_about(up0: FloatArray, axis: FloatArray, psi: float) -> FloatArray:
    """Quay up0 quanh axis (axis vuông góc up0) một góc psi (Rodrigues rút gọn)."""
    return up0 * np.cos(psi) + np.cross(axis, up0) * np.sin(psi)


def _principal_walk_axis(pts: FloatArray, up: FloatArray) -> FloatArray:
    q = pts - pts.mean(axis=0)
    q = q - (q @ up)[:, None] * up
    return np.linalg.svd(q, full_matrices=False)[2][0]


def _envelope(
    pts: FloatArray, up: FloatArray, e1: FloatArray, n_bins: int, q: float
) -> Tuple[FloatArray, FloatArray]:
    """Đường bao dưới: phân vị thấp của độ cao trong từng khoảng theo trục đi."""
    h = pts @ up
    x = pts @ e1
    edges = np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1))
    xb, hb = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x >= lo) & (x <= hi)
        if sel.sum() >= 20:
            xb.append(float(np.median(x[sel])))
            hb.append(float(np.percentile(h[sel], q)))
    return np.asarray(xb), np.asarray(hb)


def _slope(pts: FloatArray, up: FloatArray, e1: FloatArray, n_bins: int, q: float) -> float:
    xb, hb = _envelope(pts, up, e1, n_bins, q)
    if len(xb) < 4:
        raise ValueError("Quá ít đoạn có dữ liệu để đo độ dốc sàn.")
    return float(np.polyfit(xb, hb, 1)[0])


def fit_ground_plane(
    points: ArrayLike,
    *,
    up_hint: ArrayLike,
    mirror_normal: Optional[ArrayLike] = None,
    method: str = "envelope",
    marker_height_m: float = 0.02,
    floor_quantile: float = 20.0,
    n_bins: int = 10,
    max_tilt_deg: float = 25.0,
    ransac_iters: int = 800,
    ransac_tol_m: float = 0.03,
    d_ref: Optional[float] = None,
    seed: int = 0,
) -> GroundPlane:
    """Fit mặt sàn tĩnh từ điểm bàn chân (N, 3) của TOÀN chuỗi.

    points: gót/ngón chân, hệ camera, đúng tỉ lệ. up_hint: hướng lên ban đầu (ví dụ
    -vertical_down trong config). mirror_normal: n_m (hướng về camera); nếu có, ép
    n_g vuông góc n_m rồi chỉ tinh chỉnh góc quanh n_m (độ nghiêng dọc hướng đi).
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 200:
        raise ValueError(f"Cần >= 200 điểm bàn chân, có {len(pts)}.")
    up_in = _unit(up_hint)
    diag: Dict[str, float] = {"n_points": float(len(pts))}

    if method == "envelope":
        up0 = up_in
        if mirror_normal is not None:
            nm = _unit(mirror_normal)
            proj = up0 - (up0 @ nm) * nm
            if np.linalg.norm(proj) < 0.3:
                raise ValueError("up_hint gần song song pháp tuyến gương: không hợp lệ.")
            up0 = _unit(proj)
            axis = nm
            diag["orthogonality_correction_deg"] = float(
                np.degrees(np.arccos(np.clip(up0 @ up_in, -1.0, 1.0))))
        else:
            e_tmp = _principal_walk_axis(pts, up0)
            axis = _unit(np.cross(up0, e_tmp))
        e1 = _principal_walk_axis(pts, up0)

        def s_of(psi: float) -> float:
            return _slope(pts, _rotate_about(up0, axis, psi), e1, n_bins, floor_quantile)

        lim = np.deg2rad(max_tilt_deg)
        grid = np.linspace(-lim, lim, 61)
        sv = np.array([s_of(p) for p in grid])
        diag["slope_before_deg"] = float(np.degrees(np.arctan(s_of(0.0))))
        from scipy.optimize import brentq, minimize_scalar

        flips = np.nonzero(np.sign(sv[:-1]) * np.sign(sv[1:]) < 0)[0]
        if len(flips):
            k = flips[np.argmin(np.abs(sv[flips]) + np.abs(sv[flips + 1]))]
            psi = float(brentq(s_of, grid[k], grid[k + 1], xtol=1e-6))
        else:
            k = int(np.argmin(np.abs(sv)))
            res = minimize_scalar(lambda p: abs(s_of(p)), method="bounded",
                                  bounds=(grid[max(k - 1, 0)], grid[min(k + 1, 60)]))
            psi = float(res.x)
            diag["slope_not_bracketed"] = 1.0
        up = _rotate_about(up0, axis, psi)
        xb, hb = _envelope(pts, up, e1, n_bins, floor_quantile)
        floor_level = float(np.median(hb))
        offset = -(floor_level - marker_height_m)
        diag["slope_after_deg"] = float(np.degrees(np.arctan(np.polyfit(xb, hb, 1)[0])))
        diag["tilt_applied_deg"] = float(np.degrees(psi))
        diag["tilt_vs_hint_deg"] = float(np.degrees(np.arccos(np.clip(up @ up_in, -1, 1))))
        h = pts @ up - floor_level
        near = (h > -0.08) & (h < 0.05)
        diag["flatness_rms_mm"] = float(1000 * np.sqrt(np.mean(h[near] ** 2))) if near.any() else float("nan")
        lateral = _unit(np.cross(up, e1))
        diag["lateral_spread_m"] = float(np.std(pts @ lateral))
        return GroundPlane(up, float(offset), d_ref, "envelope", diag)

    if method == "ransac":
        rng = np.random.default_rng(seed)
        e1 = _principal_walk_axis(pts, up_in)
        h0 = pts @ up_in
        x = pts @ e1
        edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
        keep = np.zeros(len(pts), bool)
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (x >= lo) & (x <= hi)
            if sel.sum() >= 20:
                keep |= sel & (h0 <= np.percentile(h0[sel], 35.0))
        cand = pts[keep]
        if len(cand) < 50:
            raise ValueError("Quá ít điểm ứng viên chạm sàn.")
        best_n, best_c, best_cnt = None, 0.0, -1
        cos_lim = np.cos(np.deg2rad(max_tilt_deg))
        for _ in range(ransac_iters):
            i = rng.choice(len(cand), 3, replace=False)
            nrm = np.cross(cand[i[1]] - cand[i[0]], cand[i[2]] - cand[i[0]])
            ln = np.linalg.norm(nrm)
            if ln < 1e-4:
                continue
            nrm = nrm / ln
            if nrm @ up_in < 0:
                nrm = -nrm
            if nrm @ up_in < cos_lim:
                continue
            c0 = -float(nrm @ cand[i[0]])
            cnt = int(np.sum(np.abs(cand @ nrm + c0) < ransac_tol_m))
            if cnt > best_cnt:
                best_n, best_c, best_cnt = nrm, c0, cnt
        if best_n is None:
            raise ValueError("RANSAC không tìm được mặt phẳng hợp lệ.")
        inl = cand[np.abs(cand @ best_n + best_c) < ransac_tol_m]
        cen = inl.mean(axis=0)
        _, sing, vt = np.linalg.svd(inl - cen, full_matrices=False)
        n_fit = vt[-1]
        if n_fit @ up_in < 0:
            n_fit = -n_fit
        diag["inlier_fraction"] = float(len(inl) / len(cand))
        sig = sing / np.sqrt(len(inl))
        diag["weak_axis_spread_m"] = float(sig[1])
        diag["ill_conditioned"] = float(sig[1] < 0.15)
        diag["tilt_vs_hint_deg"] = float(np.degrees(np.arccos(np.clip(n_fit @ up_in, -1, 1))))
        offset = -float(n_fit @ cen) + marker_height_m
        return GroundPlane(n_fit, offset, d_ref, "ransac", diag)

    raise ValueError("method phải là 'envelope' hoặc 'ransac'.")