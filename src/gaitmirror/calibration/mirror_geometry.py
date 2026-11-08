"""Hình học gương phẳng: nguồn duy nhất của quy ước dấu, phản xạ và ước lượng tĩnh.

QUY ƯỚC (hệ camera thật, OpenCV: x phải, y xuống, z ra trước)::

    mặt gương:   n . X + d = 0,   ||n|| = 1,   d > 0 = khoảng cách camera -> gương
    n hướng VỀ PHÍA CAMERA (n_z < 0 để gương nằm phía trước camera)
    signed_distance(X) = n . X + d  > 0 ở phía camera (nơi người đứng)
    phản xạ:     X' = X - 2 (n . X + d) n = (I - 2 n n^T) X - 2 d n
    camera ảo:   tâm C' = -2 d n (ảnh phản xạ của gốc tọa độ)

Nếu một tài liệu viết mặt phẳng dạng n . X - d = 0 với n hướng ra xa camera, ma trận
4x4 của họ có tịnh tiến +2 d n; đó là cùng một mặt phẳng với (n' = -n, d' = d) ở quy
ước này. Dấu của n quyết định dấu tịnh tiến, nên phải thống nhất trước khi dùng.

Gương là vật thể CỐ ĐỊNH trong phòng: mọi ước lượng ở đây là TĨNH TOÀN CỤC (một nghiệm
cho cả chuỗi), không tính theo từng khung hình.

Chỉ dùng NumPy. (Chưa có nhánh PyTorch: chỉ cần khi triển khai tối ưu theo tái chiếu.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _section(cfg: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    sub = cfg.get(key, cfg)
    if not isinstance(sub, Mapping):
        raise TypeError(f"Config section '{key}' phải là mapping.")
    return sub


@dataclass(frozen=True)
class MirrorPlane:
    """Mặt phẳng gương trong hệ camera thật. Dùng `from_plane` để chuẩn hóa."""

    normal: FloatArray  # (3,) đơn vị, hướng về phía camera
    d: float  # > 0, mét

    # ---------------------------------------------------------------- tạo
    @classmethod
    def from_plane(
        cls, normal: ArrayLike, d: float, require_in_front: bool = True
    ) -> "MirrorPlane":
        """Chuẩn hóa (||n||=1, d>0) và kiểm tra gương nằm phía trước camera."""
        n = np.asarray(normal, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(n))
        if norm < 1e-12:
            raise ValueError("Pháp tuyến gương có độ dài ~0.")
        n = n / norm
        d_val = float(d) / norm  # giữ nguyên phương trình mặt phẳng
        if abs(d_val) < 1e-9:
            raise ValueError("d ~ 0: gương đi qua tâm camera, suy biến.")
        if d_val < 0.0:
            n, d_val = -n, -d_val
        if require_in_front and n[2] >= -1e-6:
            raise ValueError(
                "Gương nằm SAU camera: với n.X + d = 0 và d > 0 cần n_z < 0 "
                f"(nhận n={np.round(n, 4).tolist()}, d={d_val:.3f}). Nếu bạn viết "
                "mặt phẳng dạng n.X = d (n hướng ra xa camera), hãy đổi dấu pháp "
                f"tuyến: normal = {np.round(-n, 4).tolist()}."
            )
        return cls(normal=n, d=d_val)

    @classmethod
    def from_setup(
        cls, camera_to_mirror: float, angle_to_mirror_deg: float, image_side: str
    ) -> "MirrorPlane":
        """Từ số đo hiện trường (gương thẳng đứng, camera nằm ngang, KHÔNG chúi).

        camera_to_mirror: AB dọc trục quang học tới điểm trục cắt gương (m).
        angle_to_mirror_deg: góc giữa trục quang học và MẶT gương (độ).
        image_side: "left" nếu ảnh gương nằm bên trái người thật trong khung (n_x > 0).
        Hình học: n = (±cos a, 0, -sin a), d = AB * sin a. Camera thực tế thường chúi
        nên n_y != 0; khi đó dùng ước lượng từ dữ liệu (estimate_normal_*).
        """
        if not 0.0 < angle_to_mirror_deg < 90.0:
            raise ValueError("angle_to_mirror_deg phải nằm trong (0, 90).")
        if camera_to_mirror <= 0.0:
            raise ValueError("camera_to_mirror phải > 0.")
        side = image_side.strip().lower()
        if side not in ("left", "right"):
            raise ValueError("image_side phải là 'left' hoặc 'right'.")
        a = np.deg2rad(angle_to_mirror_deg)
        sign = 1.0 if side == "left" else -1.0
        return cls.from_plane(
            [sign * np.cos(a), 0.0, -np.sin(a)], camera_to_mirror * np.sin(a)
        )

    @classmethod
    def from_config(cls, mirror_cfg: Mapping[str, Any]) -> "MirrorPlane":
        """Đọc mirror.yaml (section `mirror` hoặc top-level).

        Ưu tiên: `plane: [a,b,c,d]` > `normal` + (`d` | `distance_m`) > suy từ `setup`
        (khi thiếu normal/plane hoặc `use_setup: true`, cần `image_side`).
        """
        mir = _section(mirror_cfg, "mirror")
        setup = mirror_cfg.get("setup", {}) or {}
        derive = bool(mir.get("use_setup", False)) or (
            "normal" not in mir and "plane" not in mir
        )
        if derive:
            side = mir.get("image_side", setup.get("mirror_image_side"))
            if side is None:
                raise KeyError("Thiếu mirror.image_side (left|right) để suy ra pháp tuyến.")
            return cls.from_setup(
                float(mir.get("camera_to_mirror", setup.get("camera_to_mirror"))),
                float(mir.get("angle_to_mirror_deg", setup.get("angle_to_mirror_deg"))),
                str(side),
            )
        if "plane" in mir:
            p = np.asarray(mir["plane"], dtype=np.float64).reshape(4)
            return cls.from_plane(p[:3], float(p[3]))
        return cls.from_plane(
            mir["normal"], float(mir.get("d", mir.get("distance_m")))
        )

    def to_yaml_dict(self) -> Dict[str, Any]:
        """Khối `mirror:` tương thích mirror.yaml hiện có (normal, d)."""
        return {"normal": [float(v) for v in self.normal], "d": float(self.d)}

    # ------------------------------------------------------------ hình học
    def signed_distance(self, X: ArrayLike) -> FloatArray:
        """n.X + d (dương ở phía camera). X: (..., 3)."""
        return np.asarray(X, dtype=np.float64) @ self.normal + self.d

    def reflect(self, X: ArrayLike) -> FloatArray:
        """Phản xạ điểm qua gương. X: (..., 3)."""
        return reflect_points(X, self.normal, self.d)

    def reflection_matrix3(self) -> FloatArray:
        return np.eye(3) - 2.0 * np.outer(self.normal, self.normal)

    def reflection_matrix4(self) -> FloatArray:
        return reflection_matrix(self.normal, self.d)

    def virtual_camera_center(self, C: Optional[ArrayLike] = None) -> FloatArray:
        """Tâm camera ảo = phản xạ của tâm camera thật (gốc tọa độ nếu C=None)."""
        c = np.zeros(3) if C is None else np.asarray(C, dtype=np.float64)
        return self.reflect(c)

    @property
    def baseline(self) -> float:
        return 2.0 * self.d


# ---------------------------------------------------------------------- #
# Phản xạ + camera ảo
# ---------------------------------------------------------------------- #
def reflection_matrix(n: ArrayLike, d: float) -> FloatArray:
    """Ma trận phản xạ đồng nhất 4x4: [[I - 2nn^T, -2dn], [0, 1]] (n hướng về camera)."""
    n_arr = np.asarray(n, dtype=np.float64).reshape(3)
    n_arr = n_arr / np.linalg.norm(n_arr)
    M = np.eye(4)
    M[:3, :3] = np.eye(3) - 2.0 * np.outer(n_arr, n_arr)
    M[:3, 3] = -2.0 * float(d) * n_arr
    return M


def reflect_points(X: ArrayLike, n: ArrayLike, d: float) -> FloatArray:
    """X' = X - 2 (n.X + d) n, vector hóa theo (..., 3)."""
    X_arr = np.asarray(X, dtype=np.float64)
    n_arr = np.asarray(n, dtype=np.float64).reshape(3)
    n_arr = n_arr / np.linalg.norm(n_arr)
    s = X_arr @ n_arr + float(d)
    return X_arr - 2.0 * s[..., None] * n_arr


@dataclass(frozen=True)
class VirtualCamera:
    """Camera ảo (hệ tay trái: det(R) = -1, chỉ dùng cho phép chiếu / DLT)."""

    R: FloatArray
    t: FloatArray
    P: FloatArray  # 3x4 = K [R | t]
    center: FloatArray  # tâm camera ảo trong hệ tọa độ tham chiếu của plane


def get_virtual_camera(
    K: ArrayLike,
    plane: MirrorPlane,
    R: Optional[ArrayLike] = None,
    t: Optional[ArrayLike] = None,
) -> Tuple[FloatArray, VirtualCamera]:
    """Ma trận chiếu camera thật và camera ảo.

    `plane` được cho trong HỆ THẾ GIỚI; camera thật có ngoại tại X_c = R X_w + t.
    Mặc định R=I, t=0 (hệ thế giới = hệ camera thật), khi đó
    R_virt = I - 2nn^T, t_virt = -2dn, tâm ảo = -2dn.
    Tổng quát: n_c = R n, d_c = d - n_c.t, R_v = H R, t_v = H t - 2 d_c n_c.
    Trả về (P_real, VirtualCamera).
    """
    K_arr = np.asarray(K, dtype=np.float64).reshape(3, 3)
    R_arr = np.eye(3) if R is None else np.asarray(R, dtype=np.float64).reshape(3, 3)
    t_arr = np.zeros(3) if t is None else np.asarray(t, dtype=np.float64).reshape(3)
    n_c = R_arr @ plane.normal
    d_c = plane.d - float(n_c @ t_arr)
    H = np.eye(3) - 2.0 * np.outer(n_c, n_c)
    R_v = H @ R_arr
    t_v = H @ t_arr - 2.0 * d_c * n_c
    P_real = K_arr @ np.hstack([R_arr, t_arr.reshape(3, 1)])
    P_virt = K_arr @ np.hstack([R_v, t_v.reshape(3, 1)])
    center_w = -R_v.T @ t_v  # trong hệ camera ảo -> về hệ thế giới
    return P_real, VirtualCamera(R=R_v, t=t_v, P=P_virt, center=center_w)


# ---------------------------------------------------------------------- #
# Ước lượng tĩnh từ dữ liệu 2D
# ---------------------------------------------------------------------- #
@dataclass
class EpipoleFit:
    normal: FloatArray  # n hướng về camera, n_z < 0
    epipole_px: FloatArray  # (2,) điểm đồng quy (pixel); nan nếu ở vô cực
    residual_median_px: float
    residual_p90_px: float
    n_used: int


def fit_epipole(
    pr: FloatArray, pm: FloatArray, center: FloatArray
) -> Tuple[FloatArray, FloatArray]:
    """Điểm đồng quy e (3,) (đồng nhất, pixel) của các đường nối pr_i -> pm_i.

    Lặp 4 lần, mỗi lần bỏ đường có residual > 3 lần median (robust).
    Với gương phẳng, mọi đường nối "điểm thật - ảnh gương" của cùng một điểm 3D đi qua
    e = K n, không phụ thuộc d. Trả về (e, khoảng cách tới từng đường, tính ở không gian
    chuẩn hóa).
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
    e_px = np.array([e[0] * s + center[0] * e[2], e[1] * s + center[1] * e[2], e[2]])
    res_px = np.full(len(pr), np.nan)
    res_px[keep] = np.abs(lines @ e) / max(abs(e[2]), 1e-12) * s
    return e_px, res_px


def point_line_residual(pr: FloatArray, pm: FloatArray, e: FloatArray) -> FloatArray:
    """Khoảng cách (px) từ điểm gương tới đường (epipole -> điểm thật).

    Đo ngay tại vị trí khớp nên so sánh được với nhiễu detector (1-3 px), khác khoảng
    cách tới epipole vốn bị khuếch đại khi epipole ở xa.
    """
    if abs(e[2]) < 1e-9:
        return np.full(len(pr), np.nan)
    e2 = e[:2] / e[2]
    v1, v2 = pr - e2, pm - e2
    cross = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return cross / np.maximum(np.linalg.norm(v1, axis=1), 1e-9)


def estimate_normal_from_correspondences(
    pts_real: ArrayLike, pts_mirror: ArrayLike, K: ArrayLike,
    max_points: int = 20000, seed: int = 0,
) -> EpipoleFit:
    """Ước lượng pháp tuyến gương (TĨNH) từ các cặp điểm thật <-> ảnh gương.

    pts_*: (N, 2) pixel ĐÃ GỠ MÉO, ghép đúng khớp (đã lật trái-phải). Không cần d,
    không cần mặt sàn. Cho phép n_y != 0 (camera chúi). Dùng toàn bộ chuỗi một lần.
    """
    pr = np.asarray(pts_real, dtype=np.float64).reshape(-1, 2)
    pm = np.asarray(pts_mirror, dtype=np.float64).reshape(-1, 2)
    ok = np.isfinite(pr).all(axis=1) & np.isfinite(pm).all(axis=1)
    pr, pm = pr[ok], pm[ok]
    if len(pr) < 50:
        raise ValueError(f"Quá ít cặp điểm hợp lệ ({len(pr)}).")
    if len(pr) > max_points:
        sel = np.random.default_rng(seed).choice(len(pr), max_points, replace=False)
        pr, pm = pr[sel], pm[sel]
    K_arr = np.asarray(K, dtype=np.float64).reshape(3, 3)
    e, _ = fit_epipole(pr, pm, K_arr[:2, 2].copy())
    res = point_line_residual(pr, pm, e)
    n = np.linalg.solve(K_arr, e)
    n = n / np.linalg.norm(n)
    if n[2] > 0:
        n = -n
    e_img = e[:2] / e[2] if abs(e[2]) > 1e-9 else np.full(2, np.nan)
    return EpipoleFit(
        normal=n, epipole_px=e_img,
        residual_median_px=float(np.nanmedian(res)),
        residual_p90_px=float(np.nanpercentile(res, 90)), n_used=len(pr),
    )


def d_from_mirror_edge(
    normal: ArrayLike, K: ArrayLike, uv_bottom: ArrayLike, uv_top: ArrayLike,
    length_m: float,
) -> Tuple[float, FloatArray]:
    """d (m) và hướng thẳng đứng (xuống) từ hai điểm trên MỘT cạnh đứng của gương.

    uv_*: pixel đã gỡ méo. Tia qua mỗi điểm cắt mặt phẳng tại độ sâu tỉ lệ với d nên
    độ dài đoạn tỉ lệ với d: d = length / |v1 - v2| với v_i = -r_i / (n . r_i).
    Hướng trả về chỉ đáng tin nếu cạnh THỰC SỰ thẳng đứng (gương không nghiêng).
    """
    K_arr = np.asarray(K, dtype=np.float64).reshape(3, 3)
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    n = n / np.linalg.norm(n)
    uv = np.asarray([uv_bottom, uv_top], dtype=np.float64).reshape(2, 2)
    rays = np.linalg.solve(K_arr, np.c_[uv, np.ones(2)].T).T
    den = rays @ n
    if np.any(den >= 0):
        raise ValueError("Tia qua điểm click không cắt gương phía trước camera.")
    v = -rays / den[:, None]
    seg = v[0] - v[1]
    seg_len = float(np.linalg.norm(seg))
    vec = seg / seg_len
    if vec[1] < 0:
        vec = -vec
    return float(length_m) / seg_len, vec