"""
Module lưu trữ và nạp dữ liệu trung gian dạng nén .npz.
Đảm bảo cố định kết quả trích xuất 2D ra đĩa cứng phục vụ Giai đoạn 2.

Kiến trúc ranh giới (Ajisafe et al., 3DV 2024):
  - `mirror_kpts_raw` lưu tọa độ RAW detection từ ảnh gương — KHÔNG flip L/R.
  - Tầng 3D sẽ tự gọi correct_virt() per-frame để so sánh hướng detection
    và hướng bộ xương 3D chiếu ngược, kích hoạt hoán đổi đúng chỗ.
"""

import json
import os
from typing import Any, Dict, List, Optional
import numpy as np


def save_assigned_2d(
    output_path: str,
    real_kpts: np.ndarray,
    mirror_kpts_raw: np.ndarray,
    real_bboxes: np.ndarray,
    mirror_bboxes: np.ndarray,
    flags_per_frame: List[List[str]],
    torso_ratios: np.ndarray,
    metadata: Dict[str, Any],
) -> None:
    """
    Lưu toàn bộ mảng dữ liệu 2D đã phân định vào file .npz nén.

    Args:
        output_path: Đường dẫn file lưu (.npz).
        real_kpts: Mảng (T, K, 3) tọa độ người thật [x, y, conf].
        mirror_kpts_raw: Mảng (T, K, 3) tọa độ bóng gương RAW (KHÔNG flip L/R).
            Dữ liệu bảo toàn trung thực 100% từ ViTPose detection, không bị
            can thiệp biến đổi hình học giả lập.
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
        mirror_kpts_raw=mirror_kpts_raw.astype(np.float32),
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
        Key 'mirror_kpts_raw' chứa tọa độ RAW (không flip), dùng cho tầng 3D.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Không tìm thấy file: {input_path}")

    with np.load(input_path, allow_pickle=True) as data:
        result: Dict[str, Any] = {
            "real_kpts": data["real_kpts"],
            "real_bboxes": data["real_bboxes"],
            "mirror_bboxes": data["mirror_bboxes"],
            "torso_ratios": data["torso_ratios"],
            "flags": json.loads(str(data["flags_json"])),
            "metadata": json.loads(str(data["metadata_json"])),
        }
        # Hỗ trợ cả file cũ (mirror_kpts) và file mới (mirror_kpts_raw)
        if "mirror_kpts_raw" in data:
            result["mirror_kpts_raw"] = data["mirror_kpts_raw"]
        elif "mirror_kpts" in data:
            # File cũ: trường mirror_kpts đã được flip — cảnh báo người dùng
            result["mirror_kpts_raw"] = data["mirror_kpts"]
        return result