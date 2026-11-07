"""
ViTPose WholeBody inference (133 keypoints) cho pipeline gait_mirror.

Bọc (wrap) thư viện `easy_ViTPose` (JunkyByte) để cung cấp giao diện ổn định:

    estimator = ViTPoseInference(model_weight, yolo_weight, device)
    kpts_dict = estimator.infer_frame(frame_bgr)
    # -> {person_idx: np.ndarray (133, 3) [x, y, conf]}

Các điểm cần lưu ý (đã đối chiếu với mã nguồn easy_ViTPose):
  * easy_ViTPose nhận ảnh RGB, còn OpenCV/VideoReader trả BGR -> đổi kênh ở đây.
  * easy_ViTPose trả keypoints theo thứ tự (y, x, conf) -> đảo thành (x, y, conf).
  * Hàm tự suy dataset từ tên file cần dấu '-' (vd. vitpose-b-wholebody.pth);
    tên có gạch dưới sẽ lỗi -> ta truyền dataset="wholebody" tường minh.
  * Chạy ở chế độ ảnh đơn (is_video=False): không dùng tracker SORT, mỗi frame
    độc lập; person_idx chỉ là thứ tự trong frame đó (không bền vững giữa các
    frame). Việc phân định thật/gương do `clean_and_associate` đảm nhiệm.

Cài đặt (easy_ViTPose không có trên PyPI):
    pip install git+https://github.com/JunkyByte/easy_ViTPose.git
    pip install ultralytics   # cần bản hỗ trợ YOLO26 cho yolo26n.pt
"""

from __future__ import annotations

import os
import re
from typing import Dict, Optional

import cv2
import numpy as np
import torch

NUM_WHOLEBODY_KPTS = 133


def _infer_model_size(weight_path: str) -> str:
    """
    Suy ra kích thước backbone ('s', 'b', 'l', 'h') từ tên file checkpoint.
    Ví dụ: vitpose-b-wholebody.pth / vitpose_b_wholebody.pth -> 'b'.
    Mặc định 'b' nếu không nhận ra.
    """
    name = os.path.basename(weight_path).lower()
    m = re.search(r"vitpose[-_]?([sblh])(?:[-_.]|$)", name)
    return m.group(1) if m else "b"


class ViTPoseInference:
    """
    Suy luận 133 keypoints COCO-WholeBody cho từng người trong một frame.

    Args:
        model_weight: đường dẫn checkpoint ViTPose WholeBody (.pth / .onnx / .engine).
        yolo_weight: đường dẫn trọng số YOLO dùng để phát hiện người.
        device: "cuda:0", "cuda" hoặc "cpu". Tự chuyển sang CPU nếu CUDA không có.
        model_size: 's' | 'b' | 'l' | 'h'. None -> suy từ tên file.
        yolo_size: kích thước ảnh đưa vào YOLO (imgsz). Mặc định 640.
    """

    def __init__(
        self,
        model_weight: str,
        yolo_weight: str,
        device: str = "cuda:0",
        model_size: Optional[str] = None,
        yolo_size: int = 640,
    ) -> None:
        if not os.path.isfile(model_weight):
            raise FileNotFoundError(f"Không tìm thấy trọng số ViTPose: {model_weight}")
        if not os.path.isfile(yolo_weight):
            raise FileNotFoundError(f"Không tìm thấy trọng số YOLO: {yolo_weight}")

        if device.startswith("cuda") and not torch.cuda.is_available():
            print("[ViTPoseInference] CUDA không khả dụng, chuyển sang CPU.")
            device = "cpu"
        self.device = device

        try:
            from easy_ViTPose import VitInference
        except ImportError as e:  # pragma: no cover - phụ thuộc môi trường
            raise ImportError(
                "Chưa cài easy_ViTPose. Cài bằng:\n"
                "  pip install git+https://github.com/JunkyByte/easy_ViTPose.git"
            ) from e

        self.model_size = model_size or _infer_model_size(model_weight)
        self._engine = VitInference(
            model=model_weight,
            yolo=yolo_weight,
            model_name=self.model_size,
            dataset="wholebody",  # bắt buộc tường minh (xem docstring đầu file)
            det_class="human",
            yolo_size=yolo_size,
            device=device,
            is_video=False,  # không tracker: mỗi frame độc lập
        )

    def infer_frame(self, frame_bgr: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Suy luận một frame.

        Args:
            frame_bgr: ảnh BGR uint8 (H, W, 3) từ OpenCV/VideoReader.

        Returns:
            Dict {person_idx: ndarray float32 (133, 3)} với cột [x, y, conf]
            theo toạ độ pixel của ảnh gốc. Dict rỗng nếu không phát hiện ai.
        """
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(
                f"frame phải là ảnh BGR (H, W, 3), nhận được "
                f"{None if frame_bgr is None else frame_bgr.shape}"
            )

        # easy_ViTPose yêu cầu RGB và mảng liên tục trong bộ nhớ
        frame_rgb = np.ascontiguousarray(frame_bgr[..., ::-1])
        raw = self._engine.inference(frame_rgb)

        result: Dict[int, np.ndarray] = {}
        for idx, kpts in raw.items():
            kpts = np.asarray(kpts, dtype=np.float32)
            if kpts.shape != (NUM_WHOLEBODY_KPTS, 3):
                raise RuntimeError(
                    f"Kỳ vọng keypoints shape ({NUM_WHOLEBODY_KPTS}, 3), "
                    f"nhận được {kpts.shape}. Kiểm tra checkpoint có phải WholeBody."
                )
            # (y, x, conf) -> (x, y, conf)
            result[int(idx)] = np.stack(
                [kpts[:, 1], kpts[:, 0], kpts[:, 2]], axis=1
            ).astype(np.float32)
        return result