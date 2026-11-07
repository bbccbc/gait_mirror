"""
Module đọc và trích xuất khung hình từ video thực nghiệm.
Kiểm tra tính nhất quán của tốc độ khung hình (CFR).
"""

from dataclasses import dataclass
from typing import Generator, Optional, Tuple
import cv2
import numpy as np


@dataclass
class VideoMetadata:
    """Thông số kỹ thuật của video đầu vào."""
    fps: float
    total_frames: int
    width: int
    height: int
    duration_s: float
    is_cfr_standard: bool  # True nếu đạt xấp xỉ chuẩn 60 FPS (dung sai +-2 FPS)


class VideoReader:
    """Bộ đọc video tuần tự qua OpenCV."""

    def __init__(self, video_path: str, target_fps: float = 60.0, fps_tol: float = 2.0):
        self.video_path = video_path
        self.target_fps = target_fps
        self.fps_tol = fps_tol
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Không thể mở file video: {video_path}")

        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration_s = (
            float(self.total_frames / self.fps) if self.fps > 0 else 0.0
        )
        self.is_cfr_standard = abs(self.fps - self.target_fps) <= self.fps_tol

    def get_metadata(self) -> VideoMetadata:
        """Trả về thông tin metadata của video."""
        return VideoMetadata(
            fps=self.fps,
            total_frames=self.total_frames,
            width=self.width,
            height=self.height,
            duration_s=self.duration_s,
            is_cfr_standard=self.is_cfr_standard,
        )

    def frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Generator sinh từng khung hình (BGR) kèm chỉ số frame index.
        Giúp tiết kiệm bộ nhớ RAM khi xử lý video dài.
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                break
            yield frame_idx, frame
            frame_idx += 1

    def release(self):
        """Giải phóng tài nguyên VideoCapture."""
        if self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()