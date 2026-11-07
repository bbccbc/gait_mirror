"""Hieu chuan noi tai camera bang ban co (M1).

Quy trinh dung cho pipeline:
  1. Quay ban co BANG CHINH camera va CHE DO se dung de quay du lieu (cung do phan giai, cung ong kinh/zoom,
     khoa lay net va tieu cu, tat chong rung). Trich anh tu video quay o che do do.
  2. calibrate_from_images -> IntrinsicsResult; luu bang save_intrinsics.
  3. Voi du lieu: cho mang keypoint chay tren anh GOC (con meo), roi chinh toa do keypoint bang undistort_points.
     Hinh hoc guong (Mirror-Aware) gia dinh camera pinhole khong meo, nen BAT BUOC dung toa do da khu meo.
Don vi: board_size la so GOC TRONG (cot, hang), square_size_mm la canh o vuong; K khong phu thuoc don vi nay.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class IntrinsicsResult:
    rms: float                         # sai so chieu lai RMS (pixel) do cv2.calibrateCamera tra ve
    K: np.ndarray                      # 3x3
    dist: np.ndarray                   # (k1, k2, p1, p2, k3)
    image_size: Tuple[int, int]        # (rong, cao)
    per_image_rms: List[float]         # RMS (pixel) tung anh, cung thang do voi rms
    used_paths: List[str] = field(default_factory=list)
    rejected_paths: List[str] = field(default_factory=list)
    board_size: Tuple[int, int] = (7, 6)
    square_size_mm: float = 25.0


def _object_points(board_size: Tuple[int, int], square_size_mm: float) -> np.ndarray:
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    grid = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp[:, :2] = grid * square_size_mm
    return objp


def _per_image_rms(objpoints, imgpoints, rvecs, tvecs, K, dist) -> List[float]:
    """RMS (pixel) tung anh = sqrt(trung binh binh phuong khoang cach diem). Cung thang do voi rms cua OpenCV.
    (cv2.norm(...)/N, nhu trong tai lieu OpenCV, nho hon RMS khoang sqrt(N) lan.)"""
    out = []
    for o, i, r, t in zip(objpoints, imgpoints, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, K, dist)
        d = (i.reshape(-1, 2) - proj.reshape(-1, 2))
        out.append(float(np.sqrt(np.mean(np.sum(d ** 2, axis=1)))))
    return out


def calibrate_from_images(
    image_paths: List[str],
    board_size: Tuple[int, int] = (7, 6),
    square_size_mm: float = 25.0,
    min_images: int = 10,
    max_image_rms_px: Optional[float] = None,
    fix_k3: bool = False,
) -> IntrinsicsResult:
    """Tim K va he so meo tu tap anh ban co (tat ca anh phai cung do phan giai).

    max_image_rms_px: neu dat, loai cac anh co RMS lon hon nguong roi hieu chuan lai mot lan (nguong do nhom dat).
    fix_k3: co dinh k3 = 0, huu ich khi it anh de tranh qua khop he so meo bac cao.
    """
    objp = _object_points(board_size, square_size_mm)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    find_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    imgpoints, used, rejected = [], [], []
    size: Optional[Tuple[int, int]] = None
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            rejected.append(p)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        this_size = gray.shape[::-1]
        if size is None:
            size = this_size
        elif this_size != size:                       # khac do phan giai: khong tron lan
            rejected.append(p)
            continue
        ok, corners = cv2.findChessboardCorners(gray, board_size, find_flags)
        if not ok:
            rejected.append(p)
            continue
        imgpoints.append(cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria))
        used.append(p)

    def _run(points, paths):
        if len(points) < min_images:
            raise ValueError(
                f"Chi co {len(points)} anh hop le (ban co {board_size[0]}x{board_size[1]} goc trong), "
                f"can it nhat {min_images}. Anh bi loai: {len(rejected)}.")
        objpoints = [objp] * len(points)
        flags = cv2.CALIB_FIX_K3 if fix_k3 else 0
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, points, size, None, None, flags=flags)
        per = _per_image_rms(objpoints, points, rvecs, tvecs, K, dist)
        return rms, K, dist, per

    rms, K, dist, per = _run(imgpoints, used)

    if max_image_rms_px is not None:
        keep = [i for i, e in enumerate(per) if e <= max_image_rms_px]
        if len(keep) < len(per):
            rejected += [used[i] for i in range(len(per)) if i not in keep]
            imgpoints = [imgpoints[i] for i in keep]
            used = [used[i] for i in keep]
            rms, K, dist, per = _run(imgpoints, used)

    return IntrinsicsResult(float(rms), K, dist, (int(size[0]), int(size[1])), per, used, rejected,
                            tuple(board_size), float(square_size_mm))


def summary(res: IntrinsicsResult) -> Dict[str, float]:
    """Cac so de nhom tu danh gia (khong tu phan xu). fx/fy gan 1 va tam anh gan giua la dau hieu hop ly,
    khong bao dam chat luong. Mirror-Aware dung mot tieu cu f: ban co the dung trung binh fx, fy."""
    w, h = res.image_size
    fx, fy, cx, cy = res.K[0, 0], res.K[1, 1], res.K[0, 2], res.K[1, 2]
    return {
        "n_images": len(res.used_paths), "rms_px": res.rms,
        "per_image_rms_mean": float(np.mean(res.per_image_rms)), "per_image_rms_max": float(np.max(res.per_image_rms)),
        "fx": float(fx), "fy": float(fy), "fx_over_fy": float(fx / fy),
        "cx_offset_frac": float((cx - w / 2) / w), "cy_offset_frac": float((cy - h / 2) / h),
    }


def undistort_points(pts: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Khu meo toa do pixel (..., 2) -> toa do pixel cua anh pinhole khong meo, cung K. Giu nguyen NaN.
    Nen dung ham nay tren KEYPOINT thay vi khu meo ca video (re hon, va khong cat bien)."""
    pts = np.asarray(pts, dtype=np.float64)
    flat = pts.reshape(-1, 2)
    out = np.full_like(flat, np.nan)
    ok = np.isfinite(flat).all(axis=1)
    if ok.any():
        u = cv2.undistortPoints(flat[ok].reshape(-1, 1, 2), K, dist, P=K)
        out[ok] = u.reshape(-1, 2)
    return out.reshape(pts.shape)


def undistort_frame(frame: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Khu meo mot khung (de kiem tra bang mat). Khong nen dung cho ca video 60 fps: ton tinh toan,
    va neu dung thi KHONG khu meo keypoint them lan nua."""
    return cv2.undistort(frame, K, dist, None, K)


def save_intrinsics(output_path: str, res: IntrinsicsResult, meta: Optional[dict] = None) -> None:
    """Luu ket qua kem thong tin de tai lap. meta: may quay, ong kinh/zoom, fps, che do, chong rung, ngay."""
    data = {
        "rms_error": res.rms,
        "image_width": res.image_size[0], "image_height": res.image_size[1],
        "fx": float(res.K[0, 0]), "fy": float(res.K[1, 1]), "cx": float(res.K[0, 2]), "cy": float(res.K[1, 2]),
        "K": res.K.tolist(), "dist_coeff": res.dist.flatten().tolist(),
        "per_image_rms": res.per_image_rms, "used_paths": res.used_paths, "rejected_paths": res.rejected_paths,
        "board_size": list(res.board_size), "square_size_mm": res.square_size_mm,
        "meta": meta or {},
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_intrinsics(config_path: str) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Doc K, dist va do phan giai luc hieu chuan."""
    with open(config_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return (np.array(d["K"], dtype=np.float64), np.array(d["dist_coeff"], dtype=np.float64),
            (int(d["image_width"]), int(d["image_height"])))


def assert_same_resolution(calib_size: Tuple[int, int], video_size: Tuple[int, int]) -> None:
    """K chi dung cho dung do phan giai/che do luc hieu chuan."""
    if tuple(calib_size) != tuple(video_size):
        raise ValueError(f"Do phan giai hieu chuan {calib_size} khac video {video_size}; hieu chuan lai o che do quay.")