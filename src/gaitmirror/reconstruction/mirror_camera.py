"""Hệ camera + gương phẳng: camera ảo, ma trận chiếu, undistort.

Quy ước (hệ tọa độ camera thật, OpenCV: x phải, y xuống, z hướng ra trước):

* Mặt phẳng gương:  n^T X + d = 0, với ||n|| = 1.
  Sau chuẩn hóa nội bộ luôn có d > 0 (đổi dấu cả n và d nếu cần; phương trình
  mặt phẳng không đổi). Khi đó d chính là khoảng cách từ tâm camera tới gương.
* Phản xạ điểm:  X' = H X - 2 d n,  H = I - 2 n n^T.
* Camera thật:  P_real = K [I | 0].
* Camera ảo:    P_virt = K [H | -2 d n]; chiếu điểm *thật* X thành điểm ảnh
  trong gương (điểm ảnh bên trong gương thấy được qua camera thật).

Lưu ý: det(H) = -1 nên camera ảo là hệ "tay trái"; chỉ dùng P_virt cho
phép chiếu / DLT, không dùng nó như một tư thế camera thật.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _section(cfg: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Lấy section `key` nếu có, ngược lại coi cfg là chính section đó."""
    sub = cfg.get(key, cfg)
    if not isinstance(sub, Mapping):
        raise TypeError(f"Config section '{key}' phải là mapping.")
    return sub


class MirrorCameraSystem:
    """Camera thật + camera ảo sinh bởi gương phẳng."""

    def __init__(
        self,
        K: ArrayLike,
        normal: ArrayLike,
        d: float,
        dist_coeffs: Optional[ArrayLike] = None,
    ) -> None:
        K_arr = np.asarray(K, dtype=np.float64).reshape(3, 3)
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
        # Sau khi d > 0, pháp tuyến phải hướng về phía camera (n_z < 0) thì
        # trục quang học mới cắt gương ở z = -d / n_z > 0 (gương ở phía trước).
        if n[2] >= -1e-6:
            raise ValueError(
                "Gương nằm SAU camera: với n^T X + d = 0 và d > 0, cần n_z < 0 "
                f"(nhận n={np.round(n, 4).tolist()}, d={d_val:.3f}). Nếu bạn "
                "đang viết mặt phẳng dạng n^T X = d (n hướng ra xa camera), hãy "
                f"đổi dấu pháp tuyến: normal = {np.round(-n, 4).tolist()}."
            )

        self.K: FloatArray = K_arr
        self.normal: FloatArray = n
        self.d: float = d_val
        self.dist_coeffs: Optional[FloatArray] = None
        if dist_coeffs is not None:
            dc = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
            if dc.size > 0 and np.any(np.abs(dc) > 0):
                self.dist_coeffs = dc

        self.H: FloatArray = np.eye(3) - 2.0 * np.outer(n, n)
        self.R_virt: FloatArray = self.H.copy()
        self.t_virt: FloatArray = -2.0 * self.d * n
        self.P_real: FloatArray = K_arr @ np.hstack([np.eye(3), np.zeros((3, 1))])
        self.P_virt: FloatArray = K_arr @ np.hstack(
            [self.R_virt, self.t_virt.reshape(3, 1)]
        )

    # ------------------------------------------------------------------ #
    # Khởi tạo từ config
    # ------------------------------------------------------------------ #
    @staticmethod
    def _rescale_K(
        K: FloatArray,
        calib_size: Sequence[int],
        data_size: Sequence[int],
        aspect_mode: str = "none",
    ) -> FloatArray:
        """Quy đổi K từ độ phân giải hiệu chuẩn sang độ phân giải dữ liệu.

        aspect_mode:
          * "none": chỉ cho phép khi hai độ phân giải cùng tỉ lệ khung hình
            (scale đều); khác tỉ lệ -> ValueError để tránh sai K âm thầm.
          * "crop": ảnh đích là vùng cắt giữa của ảnh gốc sau khi scale đều
            theo cạnh "phủ kín" (cover): s = max(tw/cw, th/ch), cắt phần dư
            ở giữa. Phù hợp ảnh 4:3 -> video 16:9 (cùng chiều ngang cảm biến).
        Hệ số méo không đổi (định nghĩa trên tọa độ chuẩn hóa).
        """
        cw, ch = float(calib_size[0]), float(calib_size[1])
        tw, th = float(data_size[0]), float(data_size[1])
        K = np.asarray(K, dtype=np.float64).copy()
        if cw == tw and ch == th:
            return K
        if aspect_mode == "crop":
            s = max(tw / cw, th / ch)
            ox = (cw * s - tw) / 2.0
            oy = (ch * s - th) / 2.0
            K[0, :] *= s
            K[1, :] *= s
            K[0, 2] -= ox
            K[1, 2] -= oy
            return K
        if aspect_mode != "none":
            raise ValueError(f"aspect_mode không hợp lệ: {aspect_mode!r}")
        if abs(cw / ch - tw / th) > 1e-3:
            raise ValueError(
                f"Tỉ lệ khung hình khác nhau ({cw:.0f}x{ch:.0f} vs "
                f"{tw:.0f}x{th:.0f}); đặt camera.aspect_mode: crop nếu video là "
                "vùng cắt giữa của ảnh hiệu chuẩn."
            )
        K[0, :] *= tw / cw
        K[1, :] *= th / ch
        return K

    @staticmethod
    def plane_from_setup(
        camera_to_mirror: float, angle_to_mirror_deg: float, image_side: str
    ) -> Tuple[FloatArray, float]:
        """Suy ra (n, d) từ số đo hiện trường (gương thẳng đứng, trục y // gương).

        Args:
            camera_to_mirror: AB, khoảng cách đo dọc trục quang học từ camera
                tới điểm trục cắt mặt gương (m).
            angle_to_mirror_deg: alpha, góc giữa trục quang học và *mặt phẳng*
                gương (độ). Góc giữa trục và pháp tuyến là 90 - alpha.
            image_side: "left" nếu ảnh trong gương nằm bên trái người thật
                trong khung hình (=> n_x > 0), "right" nếu bên phải (n_x < 0).

        Hình học: n = (±cos a, 0, -sin a),  d = AB * sin a  (vuông góc).
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
        n = np.array([sign * np.cos(a), 0.0, -np.sin(a)])
        return n, float(camera_to_mirror * np.sin(a))

    @classmethod
    def from_config(
        cls,
        camera_cfg: Mapping[str, Any],
        mirror_cfg: Mapping[str, Any],
        target_image_size: Optional[Sequence[int]] = None,
    ) -> "MirrorCameraSystem":
        """Dựng hệ thống từ dict YAML.

        camera.yaml (section `camera` hoặc top-level)::

            camera:
              K: [[fx,0,cx],[0,fy,cy],[0,0,1]]   # hoặc fx, fy, cx, cy
              dist: [k1, k2, p1, p2, k3]         # tùy chọn
              calib_image_size: [3840, 2160]     # độ phân giải lúc hiệu chuẩn
              image_size: [1920, 1080]           # độ phân giải dữ liệu 2D

        mirror.yaml (section `mirror` hoặc top-level)::

            mirror:
              normal: [nx, ny, nz]
              d: 2.35            # mét;  hoặc  plane: [a, b, c, d]

        Hoặc suy tự động từ số đo hiện trường (khi thiếu normal/plane, hoặc
        `use_setup: true`)::

            mirror:
              image_side: left   # ảnh trong gương nằm bên trái người thật
            setup:
              camera_to_mirror: 3.49      # AB, dọc trục quang học (m)
              angle_to_mirror_deg: 26.6   # góc giữa trục quang học và MẶT gương
        """
        cam = _section(camera_cfg, "camera")
        if "K" in cam:
            K = np.asarray(cam["K"], dtype=np.float64).reshape(3, 3)
        else:
            K = np.array(
                [
                    [cam["fx"], 0.0, cam["cx"]],
                    [0.0, cam["fy"], cam["cy"]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
        calib_size = cam.get("calib_image_size", cam.get("image_size"))
        data_size = target_image_size or cam.get("image_size", calib_size)
        if calib_size is not None and data_size is not None:
            K = cls._rescale_K(
                K, calib_size, data_size, str(cam.get("aspect_mode", "none"))
            )

        mir = _section(mirror_cfg, "mirror")
        setup = mirror_cfg.get("setup", {}) or {}
        derive = bool(mir.get("use_setup", False)) or (
            "normal" not in mir and "plane" not in mir
        )
        if derive:
            side = mir.get("image_side", setup.get("mirror_image_side"))
            if side is None:
                raise KeyError(
                    "Thiếu mirror.image_side (left|right) để suy ra pháp tuyến."
                )
            normal, d = cls.plane_from_setup(
                camera_to_mirror=float(
                    mir.get("camera_to_mirror", setup.get("camera_to_mirror"))
                ),
                angle_to_mirror_deg=float(
                    mir.get(
                        "angle_to_mirror_deg", setup.get("angle_to_mirror_deg")
                    )
                ),
                image_side=str(side),
            )
        elif "plane" in mir:
            plane = np.asarray(mir["plane"], dtype=np.float64).reshape(4)
            normal, d = plane[:3], float(plane[3])
        else:
            normal = np.asarray(mir["normal"], dtype=np.float64)
            d = float(mir.get("d", mir.get("distance_m")))
        dist = cam.get("dist", cam.get("dist_coeffs"))
        return cls(K=K, normal=normal, d=d, dist_coeffs=dist)

    # ------------------------------------------------------------------ #
    # Tiện ích hình học
    # ------------------------------------------------------------------ #
    @property
    def projection_matrices(self) -> Tuple[FloatArray, FloatArray]:
        """(P_real, P_virt), thứ tự khớp với [real, mirror] trong pipeline."""
        return self.P_real, self.P_virt

    @property
    def virtual_camera_center(self) -> FloatArray:
        """Tâm camera ảo = ảnh phản xạ của gốc tọa độ = -2 d n."""
        return -2.0 * self.d * self.normal

    @property
    def baseline(self) -> float:
        """Baseline giữa camera thật và ảo = 2 d (mét)."""
        return 2.0 * self.d

    def reflect_points(self, X: ArrayLike) -> FloatArray:
        """Phản xạ điểm 3D qua gương: X' = H X - 2 d n.  X: (..., 3)."""
        X_arr = np.asarray(X, dtype=np.float64)
        return X_arr @ self.H.T + self.t_virt

    def _project(self, P: FloatArray, X: ArrayLike) -> FloatArray:
        X_arr = np.asarray(X, dtype=np.float64)
        Xh = np.concatenate([X_arr, np.ones(X_arr.shape[:-1] + (1,))], axis=-1)
        pr = Xh @ P.T
        return pr[..., :2] / pr[..., 2:3]

    def project_real(self, X: ArrayLike) -> FloatArray:
        """Chiếu điểm 3D lên ảnh của người thật (pixel, không méo)."""
        return self._project(self.P_real, X)

    def project_virtual(self, X: ArrayLike) -> FloatArray:
        """Chiếu điểm 3D *thật* lên vị trí ảnh trong gương (pixel)."""
        return self._project(self.P_virt, X)

    def undistort_points(self, pts: ArrayLike) -> FloatArray:
        """Gỡ méo ống kính, trả về pixel theo mô hình pinhole với cùng K.

        Giữ nguyên NaN. Nếu không có hệ số méo thì trả về bản sao.
        """
        arr = np.asarray(pts, dtype=np.float64)
        out = arr.copy()
        if self.dist_coeffs is None:
            return out
        import cv2  # import muộn: chỉ cần khi có méo

        flat = out.reshape(-1, 2)
        ok = np.isfinite(flat).all(axis=1)
        if ok.any():
            und = cv2.undistortPoints(
                flat[ok].reshape(-1, 1, 2), self.K, self.dist_coeffs, P=self.K
            )
            flat[ok] = und.reshape(-1, 2)
        return flat.reshape(arr.shape)

    def __repr__(self) -> str:
        return (
            f"MirrorCameraSystem(n={np.round(self.normal, 4).tolist()}, "
            f"d={self.d:.4f} m, baseline={self.baseline:.4f} m)"
        )