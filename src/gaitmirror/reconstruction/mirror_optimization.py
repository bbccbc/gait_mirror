"""
Hiện thực hóa correct_virt và compute_orientation_flag theo kiến trúc gốc
Ajisafe et al., "Mirror-Aware Neural Humans", 3DV 2024.

Ranh giới kiến trúc:
  - Hàm compute_orientation_flag(): xác định hướng nhìn per-frame từ tọa độ 2D.
  - Hàm correct_virt(): so sánh hướng detection vs hướng projection từ bộ xương 3D,
    kích hoạt hoán đổi đối xứng per-frame khi hai hướng nghịch nhau.
  - Lớp DecoupledLegLengths: quản lý 4 tham số bone lengths độc lập cho LLD.

Không thực hiện flip mù quáng (blanket flip) tại tầng 2D — tầng này chỉ
so sánh hình học và hoán đổi nhãn khi cần thiết tại tầng tối ưu hóa 3D.
"""

from typing import List, Tuple, Optional
import numpy as np

from gaitmirror.pose.keypoint_mapping import (
    NOSE,
    L_EAR, R_EAR,
    L_SHOULDER, R_SHOULDER,
    WHOLEBODY_FLIP_PAIRS,
)


# ---------------------------------------------------------------------------
# Hàm 1: Xác định hướng nhìn 2D per-frame
# ---------------------------------------------------------------------------

def compute_orientation_flag(
    kpts: np.ndarray,
    conf_thresh: float = 0.3,
) -> np.ndarray:
    """
    Xác định vector hướng nhìn per-frame theo logic gốc của Ajisafe et al.:
      - Ưu tiên: So sánh x_right_shoulder vs x_left_shoulder.
        - flag = True  nếu x_right_shoulder > x_left_shoulder
                        (người nhìn về bên trái ảnh / vai phải xuất hiện bên phải pixel).
        - flag = False nếu x_right_shoulder <= x_left_shoulder.
      - Fallback (khi vai bị che hoặc confidence thấp hơn conf_thresh):
        So sánh x_right_ear vs x_nose.
        - flag = True  nếu x_right_ear > x_nose.

    Args:
        kpts: Mảng (T, K, 3) với K >= 7, channel 3: [x, y, conf].
              K phải đủ lớn để truy cập L_SHOULDER (5), R_SHOULDER (6),
              L_EAR (3), R_EAR (4), NOSE (0).
        conf_thresh: Ngưỡng confidence tối thiểu để tin dùng keypoint.

    Returns:
        flags: Mảng boolean shape (T,).
               True  = x_right_shoulder > x_left_shoulder (hoặc fallback tương đương).
               False = ngược lại.
               np.nan không được trả về — frame không có đủ keypoints trả về False.
    """
    kpts = np.asarray(kpts, dtype=np.float64)
    if kpts.ndim == 2:
        kpts = kpts[np.newaxis]  # (1, K, 3)
    T = kpts.shape[0]
    flags = np.zeros(T, dtype=bool)

    for t in range(T):
        frame = kpts[t]  # (K, 3)

        # Kiểm tra confidence vai
        ls_conf = frame[L_SHOULDER, 2] if frame.shape[1] > 2 else 1.0
        rs_conf = frame[R_SHOULDER, 2] if frame.shape[1] > 2 else 1.0

        ls_x = frame[L_SHOULDER, 0]
        rs_x = frame[R_SHOULDER, 0]

        shoulder_valid = (
            np.isfinite(ls_x) and np.isfinite(rs_x)
            and float(ls_conf) >= conf_thresh
            and float(rs_conf) >= conf_thresh
        )

        if shoulder_valid:
            flags[t] = rs_x > ls_x
        else:
            # Fallback: dùng tai phải vs mũi
            nose_x = frame[NOSE, 0]
            re_x = frame[R_EAR, 0]
            nose_conf = frame[NOSE, 2] if frame.shape[1] > 2 else 1.0
            re_conf = frame[R_EAR, 2] if frame.shape[1] > 2 else 1.0

            ear_nose_valid = (
                np.isfinite(nose_x) and np.isfinite(re_x)
                and float(nose_conf) >= conf_thresh
                and float(re_conf) >= conf_thresh
            )
            if ear_nose_valid:
                flags[t] = re_x > nose_x
            else:
                flags[t] = False  # Không đủ thông tin, giữ nguyên không flip

    return flags


# ---------------------------------------------------------------------------
# Hàm 2: correct_virt — so sánh per-frame hướng detection vs hướng projection
# ---------------------------------------------------------------------------

def correct_virt(
    virt_detection_kpts: np.ndarray,
    virt_projected_kpts: np.ndarray,
    flip_pairs: List[Tuple[int, int]] = WHOLEBODY_FLIP_PAIRS,
    conf_thresh: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    So sánh where_virt vs where_virt_proj từng frame t theo Ajisafe et al.:
      - where_virt[t]       = compute_orientation_flag(virt_detection_kpts[t])
      - where_virt_proj[t]  = compute_orientation_flag(virt_projected_kpts[t])
      - flip_mask[t] = True nếu where_virt[t] != where_virt_proj[t]

    Khi flip_mask[t] == True, hoán đổi các cặp keypoints đối xứng
    và cột confidence weights tương ứng trong frame t.

    Điều này KHÔNG làm biến dạng mảng raw detection ban đầu — chỉ
    tác động lên bản sao output được trả về.

    Args:
        virt_detection_kpts: (T, K, 3) raw detection từ ảnh gương.
            Channel: [x, y, conf].
        virt_projected_kpts: (T, K, 2) hoặc (T, K, 3) hình chiếu từ skeleton 3D
            qua virtual camera. Nếu chỉ có 2 channel thì conf mặc định = 1.0.
        flip_pairs: Danh sách tuple (l_idx, r_idx) để hoán đổi đối xứng.
        conf_thresh: Ngưỡng confidence tối thiểu để tin dùng keypoint
            khi xác định hướng nhìn.

    Returns:
        kpts_aligned:   (T, K, 3) — detection đã được căn chỉnh hướng.
        weights_aligned: (T, K)   — confidence weights sau căn chỉnh.
        flipped_mask:   (T,) bool — True tại frame t nếu đã hoán đổi.
    """
    det = np.asarray(virt_detection_kpts, dtype=np.float64)
    proj = np.asarray(virt_projected_kpts, dtype=np.float64)

    if det.ndim == 2:
        det = det[np.newaxis]
    if proj.ndim == 2:
        proj = proj[np.newaxis]

    # Nếu projected chỉ có 2 channel (x, y) — thêm conf = 1.0
    if proj.shape[-1] == 2:
        proj = np.concatenate(
            [proj, np.ones(proj.shape[:2] + (1,), dtype=np.float64)], axis=-1
        )

    T = det.shape[0]
    K = det.shape[1]
    assert proj.shape[0] == T, (
        f"Số frame không khớp: det={T}, proj={proj.shape[0]}"
    )

    # Lọc flip_pairs để chỉ giữ các cặp có index hợp lệ cho K keypoints thực tế.
    # Đây là trường hợp pipeline lưu 23 điểm (COCO-17 + 6 foot) thay vì 133 đầy đủ.
    valid_pairs = [(l, r) for l, r in flip_pairs if l < K and r < K]

    # Xác định hướng nhìn per-frame
    where_virt = compute_orientation_flag(det, conf_thresh=conf_thresh)
    where_virt_proj = compute_orientation_flag(proj, conf_thresh=conf_thresh)

    # Khởi tạo output là bản sao để không làm biến dạng input
    kpts_aligned = det.copy()
    weights_aligned = det[..., 2].copy()   # (T, K)
    flipped_mask = (where_virt != where_virt_proj)  # (T,) bool

    # Swap per-frame cho các frame cần flip
    flip_indices = np.where(flipped_mask)[0]
    for t in flip_indices:
        for l_idx, r_idx in valid_pairs:
            # Hoán đổi tọa độ
            kpts_aligned[t, l_idx, :], kpts_aligned[t, r_idx, :] = (
                kpts_aligned[t, r_idx, :].copy(),
                kpts_aligned[t, l_idx, :].copy(),
            )
            # Hoán đổi weights tương ứng
            weights_aligned[t, l_idx], weights_aligned[t, r_idx] = (
                weights_aligned[t, r_idx],
                weights_aligned[t, l_idx],
            )

    return kpts_aligned, weights_aligned, flipped_mask


# ---------------------------------------------------------------------------
# Lớp 3: DecoupledLegLengths — bone lengths độc lập cho LLD
# ---------------------------------------------------------------------------

class DecoupledLegLengths:
    """
    Quản lý 4 tham số bone lengths độc lập cho bệnh nhi LLD.

    Theo yêu cầu bài toán LLD (Leg Length Discrepancy):
      - CẤM dùng chung tham số đối xứng (L_femur_left == L_femur_right).
      - 4 biến hoàn toàn độc lập để đo chính xác mức độ chênh lệch chiều dài chi.

    Attributes:
        bone_length_femur_l: Độ dài xương đùi trái (mm).
        bone_length_femur_r: Độ dài xương đùi phải (mm).
        bone_length_tibia_l: Độ dài xương cẳng chân trái (mm).
        bone_length_tibia_r: Độ dài xương cẳng chân phải (mm).
    """

    def __init__(
        self,
        femur_l: float = 400.0,
        femur_r: float = 400.0,
        tibia_l: float = 350.0,
        tibia_r: float = 350.0,
    ) -> None:
        """
        Args:
            femur_l: Độ dài xương đùi trái (mm). Mặc định 400 mm.
            femur_r: Độ dài xương đùi phải (mm). Mặc định 400 mm.
            tibia_l: Độ dài xương cẳng chân trái (mm). Mặc định 350 mm.
            tibia_r: Độ dài xương cẳng chân phải (mm). Mặc định 350 mm.
        """
        self.bone_length_femur_l = float(femur_l)
        self.bone_length_femur_r = float(femur_r)
        self.bone_length_tibia_l = float(tibia_l)
        self.bone_length_tibia_r = float(tibia_r)

    @property
    def lld_femur_mm(self) -> float:
        """Chênh lệch chiều dài xương đùi Trái - Phải (mm). Dương = trái dài hơn."""
        return self.bone_length_femur_l - self.bone_length_femur_r

    @property
    def lld_tibia_mm(self) -> float:
        """Chênh lệch chiều dài xương cẳng chân Trái - Phải (mm)."""
        return self.bone_length_tibia_l - self.bone_length_tibia_r

    @property
    def lld_total_mm(self) -> float:
        """Tổng chênh lệch chiều dài chi (đùi + cẳng chân) Trái - Phải (mm)."""
        return self.lld_femur_mm + self.lld_tibia_mm

    def as_array(self) -> np.ndarray:
        """Trả về 4 tham số dạng mảng [femur_l, femur_r, tibia_l, tibia_r]."""
        return np.array([
            self.bone_length_femur_l,
            self.bone_length_femur_r,
            self.bone_length_tibia_l,
            self.bone_length_tibia_r,
        ], dtype=np.float64)

    def update_from_array(self, params: np.ndarray) -> None:
        """Cập nhật từ mảng [femur_l, femur_r, tibia_l, tibia_r] (kết quả optimizer)."""
        params = np.asarray(params, dtype=np.float64).ravel()
        assert len(params) == 4, f"Cần đúng 4 tham số, nhận {len(params)}"
        self.bone_length_femur_l = float(params[0])
        self.bone_length_femur_r = float(params[1])
        self.bone_length_tibia_l = float(params[2])
        self.bone_length_tibia_r = float(params[3])

    def is_symmetric(self, tol_mm: float = 1.0) -> bool:
        """Kiểm tra hai chân có gần đối xứng không (sai số < tol_mm)."""
        return (
            abs(self.lld_femur_mm) < tol_mm
            and abs(self.lld_tibia_mm) < tol_mm
        )

    def __repr__(self) -> str:
        return (
            f"DecoupledLegLengths("
            f"femur_l={self.bone_length_femur_l:.1f}mm, "
            f"femur_r={self.bone_length_femur_r:.1f}mm, "
            f"tibia_l={self.bone_length_tibia_l:.1f}mm, "
            f"tibia_r={self.bone_length_tibia_r:.1f}mm | "
            f"LLD_total={self.lld_total_mm:+.1f}mm)"
        )
