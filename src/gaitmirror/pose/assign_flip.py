"""
Module phân định Người thật - Gương và xử lý đảo tính đối xứng (Chirality).
Hiện thực hóa theo nguyên lý hình học từ bài báo:
Ajisafe et al., "Mirror-Aware Neural Humans", 3DV 2024.
Tối ưu hóa cho phân tích dáng đi (Gait Analysis / LLD).

Kiến trúc ranh giới rõ ràng:
  - Tầng 2D (module này): Lưu trữ trung thực 100% tọa độ raw detection từ ViTPose.
    KHÔNG thực hiện flip_wholebody() tại đây.
  - Tầng 3D (mirror_optimization.py): Thực hiện correct_virt() so sánh per-frame
    giữa hướng detection 2D và hướng bộ xương 3D chiếu ngược, kích hoạt hoán đổi
    đối xứng chỉ khi cần thiết.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import numpy as np

from gaitmirror.pose.keypoint_mapping import (
    WHOLEBODY_FLIP_PAIRS,
    compute_neck_and_root,
)


@dataclass
class AssociatedFrame:
    """Kết quả phân định Thật - Gương kèm toàn bộ cờ kiểm soát chất lượng (QC).

    Lưu ý kiến trúc: mirror_kpts_raw chứa tọa độ RAW detection từ ảnh gương,
    chưa qua bất kỳ hoán đổi Trái/Phải nào. Trường mirror_kpts được giữ lại
    để tương thích ngược nhưng BÁO KHÔNG NÊN dùng ở tầng 3D (dùng raw thay thế).
    """
    real_kpts: Optional[np.ndarray] = None             # (133, 3) người thật
    mirror_kpts_raw: Optional[np.ndarray] = None       # (133, 3) bóng gương RAW (không flip)
    real_bbox: Optional[np.ndarray] = None             # [x1, y1, x2, y2]
    mirror_bbox: Optional[np.ndarray] = None           # [x1, y1, x2, y2]
    boxes_overlap: bool = False                        # True nếu hai bounding box đè lên nhau
    iou: float = 0.0                                   # Chỉ số IoU
    torso_ratio: float = float("nan")                  # min(dist)/max(dist), đo mức độ mơ hồ
    flags: List[str] = field(default_factory=list)      # Danh sách cờ QC cảnh báo


def flip_wholebody(
    pose: np.ndarray,
    flip_pairs: List[Tuple[int, int]] = WHOLEBODY_FLIP_PAIRS,
) -> np.ndarray:
    """
    Đảo nhãn các cặp khớp đối xứng Trái <-> Phải cho khung xương trong gương.
    Hoạt động trên NumPy thuần, bảo toàn tọa độ pixel, chỉ đổi vị trí index.
    
    Args:
        pose: Mảng shape (..., 133, C) với C >= 2 (x, y, [conf]).
    """
    mirror_pose = pose.copy()
    for l_idx, r_idx in flip_pairs:
        mirror_pose[..., l_idx, :] = pose[..., r_idx, :]
        mirror_pose[..., r_idx, :] = pose[..., l_idx, :]
    return mirror_pose


def compute_bbox_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Tính Intersection over Union (IoU) giữa 2 bounding box [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 1e-6:
        return 0.0
    return float(inter_area / union_area)


def drop_bystanders_frame(
    kpts_list: List[np.ndarray],
    bboxes: np.ndarray,
    top_k: int = 2,
) -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
    """
    Lọc người ngoài cuộc dựa trên chiều dài thân (Ajisafe et al., 3DV 2024).
    Xử lý triệt để lỗi NaN và cảnh báo người thứ ba.
    """
    flags: List[str] = []
    bboxes = np.asarray(bboxes)

    if len(kpts_list) < top_k:
        flags.append("THIEU_NGUOI")
        return kpts_list, bboxes, flags

    if len(kpts_list) > top_k:
        flags.append("CO_NGUOI_THU_BA")

    # 1. Tính chiều dài thân và loại bỏ các pose dính NaN
    valid_indices = []
    torso_dists = []
    for idx, kpts in enumerate(kpts_list):
        neck, root = compute_neck_and_root(kpts)
        if np.isnan(neck).any() or np.isnan(root).any():
            continue
        dist = np.linalg.norm(neck - root)
        if np.isnan(dist) or dist <= 0.0:
            continue
        valid_indices.append(idx)
        torso_dists.append(dist)

    if len(valid_indices) < top_k:
        flags.append("THIEU_DIEM_THAN")
        return [], np.empty((0, 4)), flags

    torso_dists = np.array(torso_dists)
    valid_indices = np.array(valid_indices)

    # 2. Chọn top_k người có thân dài nhất từ tập hợp hợp lệ (không chứa NaN)
    top_k_sub_idx = np.argpartition(torso_dists, -top_k)[-top_k:]
    # Sắp xếp theo thứ tự giảm dần
    sorted_sub_idx = top_k_sub_idx[np.argsort(-torso_dists[top_k_sub_idx])]
    chosen_indices = valid_indices[sorted_sub_idx]

    selected_kpts = [kpts_list[i] for i in chosen_indices]
    selected_bboxes = bboxes[chosen_indices]

    return selected_kpts, selected_bboxes, flags


def clean_and_associate(
    kpts_a: np.ndarray,
    kpts_b: np.ndarray,
    box_a: np.ndarray,
    box_b: np.ndarray,
    iou_thresh: float,
    ratio_thresh: Optional[float] = None,
) -> Tuple[Optional[AssociatedFrame], List[str]]:
    """
    Phân định Thật - Gương theo chiều dài thân ||neck - root||.
    Không đoán mò: yêu cầu truyền rõ iou_thresh và ratio_thresh từ cấu hình.
    """
    flags: List[str] = []
    box_a = np.asarray(box_a)
    box_b = np.asarray(box_b)

    neck_a, root_a = compute_neck_and_root(kpts_a)
    neck_b, root_b = compute_neck_and_root(kpts_b)

    # Kiểm tra NaN trước khi thực hiện so sánh
    if (np.isnan(neck_a).any() or np.isnan(root_a).any() or
        np.isnan(neck_b).any() or np.isnan(root_b).any()):
        flags.append("THIEU_DIEM_THAN")
        return None, flags

    dist_a = float(np.linalg.norm(neck_a - root_a))
    dist_b = float(np.linalg.norm(neck_b - root_b))

    if np.isnan(dist_a) or np.isnan(dist_b) or dist_a <= 0.0 or dist_b <= 0.0:
        flags.append("THIEU_DIEM_THAN")
        return None, flags

    # Kiểm tra trường hợp hòa (chiều dài thân bằng nhau tuyệt đối)
    if dist_a == dist_b:
        flags.append("THAN_BANG_NHAU")

    # Tính tỷ lệ thân để bắt pha mơ hồ
    max_dist = max(dist_a, dist_b)
    min_dist = min(dist_a, dist_b)
    ratio = float(min_dist / max_dist) if max_dist > 0 else float("nan")

    if ratio_thresh is not None and ratio >= ratio_thresh:
        flags.append("THAN_GAN_BANG_NHAU")

    # Phân định Thật - Gương
    if dist_a >= dist_b:
        real_kpts, mirror_kpts_raw = kpts_a, kpts_b
        real_box, mirror_box = box_a, box_b
    else:
        real_kpts, mirror_kpts_raw = kpts_b, kpts_a
        real_box, mirror_box = box_b, box_a

    # Kiểm tra đè hộp bounding box
    iou = compute_bbox_iou(real_box, mirror_box)
    boxes_overlap = iou > iou_thresh
    if boxes_overlap:
        flags.append("HOP_CHONG")

    # KIẾN TRÚC: Lưu nguyên trạng mirror_kpts_raw (133, 3) vào AssociatedFrame.
    # KHÔNG thực hiện flip_wholebody() tại đây — tuân thủ ranh giới tầng 2D/3D
    # theo Ajisafe et al. (3DV 2024). Tầng 3D sẽ tự gọi correct_virt() per-frame.
    frame_result = AssociatedFrame(
        real_kpts=real_kpts,
        mirror_kpts_raw=mirror_kpts_raw,
        real_bbox=real_box,
        mirror_bbox=mirror_box,
        boxes_overlap=boxes_overlap,
        iou=iou,
        torso_ratio=ratio,
        flags=flags,
    )
    return frame_result, flags