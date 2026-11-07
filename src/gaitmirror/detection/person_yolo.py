"""
gait_mirror.detection.person_yolo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Module phát hiện người (thật và gương) sử dụng YOLO (Ultralytics).
Tích hợp Bottom Margin Expansion để bảo vệ gót chân 
và ngón chân khỏi hiệu ứng co biên khi làm mượt heatmap Gaussian.
gemini viết ngày 20/10/2026 nào check lại sau
"""

from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class PersonDetectorYOLO:
    """
    Bộ phát hiện đối tượng người dùng YOLO tối ưu cho bài toán camera - gương.
    """

    def __init__(
        self,
        model_weight: str = "models/weights/yolo26n.pt",
        device: str = "cuda:0",
        conf_thr: float = 0.4,
        iou_thr: float = 0.5,
        bottom_margin_ratio: float = 0.08,
    ):
        """
        Khởi tạo detector.

        Args:
            model_weight: Đường dẫn file weights (.pt) hoặc tên mô hình (vd 'yolov8n.pt', 'yolov8m.pt').
            device: Thiết bị chạy ('cuda:0' hoặc 'cpu').
            conf_thr: Ngưỡng tin cậy tối thiểu của bbox người.
            iou_thr: Ngưỡng NMS IoU.
            bottom_margin_ratio: Tỉ lệ nới đáy hộp bao để bảo vệ gót/ngón chân (mặc định 0.08 = 8%).
        """
        if YOLO is None:
            raise ImportError(
                "Chưa cài đặt ultralytics. Hãy chạy: pip install ultralytics"
            )

        self.device = device
        self.conf_thr = conf_thr
        self.iou_thr = iou_thr
        self.bottom_margin_ratio = bottom_margin_ratio

        # Nạp mô hình YOLO
        self.model = YOLO(model_weight)

    def detect(
        self,
        frame_bgr: np.ndarray,
        max_persons: int = 2,
    ) -> np.ndarray:
        """
        Phát hiện người trong khung hình và trả về danh sách các bounding box chuẩn hóa.

        Args:
            frame_bgr: Khung hình video OpenCV dạng mảng NumPy (H, W, 3).
            max_persons: Số lượng người tối đa cần lấy (mặc định 2: thật và gương).

        Returns:
            bboxes: Mảng NumPy shape (M, 5) gồm [x1, y1, x2, y2, score], với M <= max_persons.
                    Nếu không phát hiện được ai, trả về mảng rỗng shape (0, 5).
        """
        img_h, img_w = frame_bgr.shape[:2]

        # Chạy inference YOLO chỉ lọc class 0 (person)
        results = self.model(
            frame_bgr,
            conf=self.conf_thr,
            iou=self.iou_thr,
            classes=[0],
            device=self.device,
            verbose=False,
        )

        if len(results) == 0 or len(results[0].boxes) == 0:
            return np.empty((0, 5), dtype=np.float32)

        boxes_data = results[0].boxes.data.cpu().numpy()  # [x1, y1, x2, y2, conf, cls]

        # 1. Sắp xếp các boxes theo diện tích hộp bao giảm dần
        # Trong setup gương phẳng, người thật và ảnh gương là 2 đối tượng người lớn nhất trong Target Zone
        areas = (boxes_data[:, 2] - boxes_data[:, 0]) * (boxes_data[:, 3] - boxes_data[:, 1])
        sorted_indices = np.argsort(areas)[::-1]
        selected_boxes = boxes_data[sorted_indices[:max_persons]]

        # 2. Xử lý nới biên đáy (Bottom Margin Expansion) để bảo vệ gót và ngón chân
        expanded_boxes = []
        for box in selected_boxes:
            x1, y1, x2, y2, conf = box[:5]
            h = y2 - y1

            # Nới cạnh đáy y2 xuống một lượng = bottom_margin_ratio * chiều cao hộp
            y2_safe = y2 + self.bottom_margin_ratio * h

            # Kẹp an toàn trong phạm vi khung hình
            x1 = np.clip(x1, 0, img_w - 1)
            y1 = np.clip(y1, 0, img_h - 1)
            x2 = np.clip(x2, 0, img_w - 1)
            y2_safe = np.clip(y2_safe, 0, img_h - 1)

            expanded_boxes.append([x1, y1, x2, y2_safe, conf])

        return np.array(expanded_boxes, dtype=np.float32)