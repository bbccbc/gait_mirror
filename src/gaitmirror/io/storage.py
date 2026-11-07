"""
Module lưu trữ và nạp dữ liệu trung gian dạng nén .npz.
Đảm bảo cố định kết quả trích xuất 2D ra đĩa cứng phục vụ Giai đoạn 2.
"""

import json
import os
from typing import Any, Dict, List, Optional
import numpy as np


def save_assigned_2d(
    output_path: str,
    real_kpts: np.ndarray,
    mirror_kpts: np.ndarray,
    real_bboxes: np.ndarray,
    mirror_bboxes: np.ndarray,
    flags_per_frame: List[List[str]],
    torso_ratios: np.ndarray,
    metadata: Dict[str, Any],
) -> None:
    """
    Lưu toàn bộ mảng dữ liệu 2D đã phân định và chuẩn hóa vào file .npz nén.

    Args:
        output_path: Đường dẫn file lưu (.npz).
        real_kpts: Mảng (T, 133, 3) tọa độ người thật [x, y, conf].
        mirror_kpts: Mảng (T, 133, 3) tọa độ bóng gương đã swap Trái <-> Phải.
        real_bboxes: Mảng (T, 4) bounding box người thật.
        mirror_bboxes: Mảng (T, 4) bounding box bóng gương.
        flags_per_frame: Danh sách cờ QC cho từng frame.
        torso_ratios: Mảng (T,) tỷ lệ độ dài thân.
        metadata: Từ điển chứa fps, video_name, độ phân giải,...
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Chuyển đổi flags_per_frame và metadata thành chuỗi JSON để lưu gọn gàng
    flags_json = json.dumps(flags_per_frame)
    meta_json = json.dumps(metadata)

    np.savez_compressed(
        output_path,
        real_kpts=real_kpts.astype(np.float32),
        mirror_kpts=mirror_kpts.astype(np.float32),
        real_bboxes=real_bboxes.astype(np.float32),
        mirror_bboxes=mirror_bboxes.astype(np.float32),
        torso_ratios=torso_ratios.astype(np.float32),
        flags_json=flags_json,
        metadata_json=meta_json,
    )


def load_assigned_2d(input_path: str) -> Dict[str, Any]:
    """
    Đọc dữ liệu 2D đã lưu từ file .npz.

    Returns:
        Dictionary chứa các mảng numpy và metadata giải mã.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Không tìm thấy file: {input_path}")

    with np.load(input_path, allow_pickle=True) as data:
        return {
            "real_kpts": data["real_kpts"],
            "mirror_kpts": data["mirror_kpts"],
            "real_bboxes": data["real_bboxes"],
            "mirror_bboxes": data["mirror_bboxes"],
            "torso_ratios": data["torso_ratios"],
            "flags": json.loads(str(data["flags_json"])),
            "metadata": json.loads(str(data["metadata_json"])),
        }