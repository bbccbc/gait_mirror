"""
Kịch bản trích xuất 2D tự động (Perception & Association Pipeline)
Ghép nối toàn bộ chuỗi mắt xích Giai đoạn 1:
Video (.mp4) -> VideoReader -> YOLO26n -> ViTPose WholeBody (133 kpts)
             -> drop_bystanders -> clean_and_associate
             -> cắt lát 23 điểm giải phẫu (COCO-17 + 6 foot) -> Storage (.npz)

Kiến trúc ranh giới (Ajisafe et al., 3DV 2024):
  - clean_and_associate() KHÔNG thực hiện flip_wholebody() — bảo toàn 100% raw.
  - Lưu mirror_kpts_raw (tọa độ RAW detection, không hoán đổi L/R).
  - Tầng 3D sẽ gọi correct_virt() per-frame để so sánh hướng & quyết định flip.

Ghi chú: ViTPose vẫn suy luận đủ 133 điểm để các module phía sau (đặc biệt là
logic đảo chirality trái/phải) hoạt động ổn định. Việc thu hẹp xuống 23 điểm
chỉ diễn ra ở bước gán dữ liệu vào mảng lưu trữ.
"""

import argparse
import os
from pathlib import Path
from typing import List
import numpy as np
import torch
from tqdm import tqdm

from gaitmirror.detection.person_yolo import PersonDetectorYOLO
from gaitmirror.io.storage import save_assigned_2d
from gaitmirror.io.video_reader import VideoReader
from gaitmirror.pose.assign_flip import clean_and_associate, drop_bystanders_frame
from gaitmirror.pose.vitpose_infer import ViTPoseInference

# ---------------------------------------------------------------------------
# Cấu hình bộ keypoints đầu ra
# ---------------------------------------------------------------------------
NUM_KPTS_FULL = 133  # COCO-WholeBody đầy đủ (đầu ra của ViTPose)
NUM_KPTS_OUT = 23  # COCO-17 + 6 điểm bàn chân (indices 0..22)
KEYPOINT_FORMAT = "coco17_plus_feet"

# Thứ tự 23 điểm (tham chiếu / ghi vào metadata để pipeline 3D đối chiếu)
KEYPOINT_NAMES_23 = [
    "nose",  # 0
    "left_eye", "right_eye",  # 1, 2
    "left_ear", "right_ear",  # 3, 4
    "left_shoulder", "right_shoulder",  # 5, 6
    "left_elbow", "right_elbow",  # 7, 8
    "left_wrist", "right_wrist",  # 9, 10
    "left_hip", "right_hip",  # 11, 12
    "left_knee", "right_knee",  # 13, 14
    "left_ankle", "right_ankle",  # 15, 16
    "left_big_toe", "left_small_toe", "left_heel",  # 17, 18, 19
    "right_big_toe", "right_small_toe", "right_heel",  # 20, 21, 22
]
assert len(KEYPOINT_NAMES_23) == NUM_KPTS_OUT


def _slice_kpts(kpts: np.ndarray) -> np.ndarray:
    """
    Cắt mảng keypoints (K, 3) xuống 23 điểm đầu tiên -> (23, 3).
    Chấp nhận cả mảng 133 điểm lẫn mảng đã được thu hẹp sẵn 23 điểm.
    """
    if kpts.ndim != 2 or kpts.shape[0] < NUM_KPTS_OUT or kpts.shape[1] < 3:
        raise ValueError(
            f"Mảng keypoints có shape {kpts.shape}, cần tối thiểu ({NUM_KPTS_OUT}, 3)"
        )
    return kpts[:NUM_KPTS_OUT, :3]


def _match_kpts_to_bboxes(
    kpts_list: List[np.ndarray], yolo_bboxes: np.ndarray
) -> np.ndarray:
    """
    Ghép cặp từng bộ keypoints trích xuất được với bounding box tương ứng từ YOLO.
    Nếu không có bbox YOLO khớp, tính bounding box trực tiếp từ cụm keypoints.
    Hoạt động với mảng keypoints (K, 3) bất kỳ K (133 hoặc 23 điểm).
    """
    if len(kpts_list) == 0:
        return np.empty((0, 4), dtype=np.float32)

    matched_boxes = []
    for kpts in kpts_list:
        # Lọc các điểm hữu hạn có độ tin cậy > 0.2 để xác định tâm người
        finite = np.isfinite(kpts[:, :2]).all(axis=1)
        valid_pts = kpts[finite & (kpts[:, 2] > 0.2), :2]
        if len(valid_pts) < 4:
            valid_pts = kpts[finite, :2]

        if len(valid_pts) == 0:
            # Không có điểm hợp lệ nào: không thể dựng bbox từ keypoints
            if len(yolo_bboxes) > 0:
                matched_boxes.append(yolo_bboxes[0][:4])
            else:
                matched_boxes.append(np.full(4, np.nan, dtype=np.float32))
            continue

        x1_k, y1_k = np.min(valid_pts, axis=0)
        x2_k, y2_k = np.max(valid_pts, axis=0)
        center_k = np.array([(x1_k + x2_k) / 2.0, (y1_k + y2_k) / 2.0])

        best_box = None
        min_dist = float("inf")

        if len(yolo_bboxes) > 0:
            for b in yolo_bboxes:
                center_b = np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])
                dist = np.linalg.norm(center_k - center_b)
                if dist < min_dist:
                    min_dist = dist
                    best_box = b[:4]

        # Fallback: tự dựng bbox từ keypoints với margin đáy 8% bảo vệ bàn chân
        if best_box is None:
            w_k = x2_k - x1_k
            h_k = y2_k - y1_k
            best_box = np.array(
                [x1_k - 0.05 * w_k, y1_k - 0.05 * h_k, x2_k + 0.05 * w_k, y2_k + 0.08 * h_k],
                dtype=np.float32,
            )

        matched_boxes.append(best_box)

    return np.array(matched_boxes, dtype=np.float32)


def run_pipeline(
    video_path: str,
    output_dir: str,
    yolo_weight: str = "yolo26n.pt",
    vitpose_weight: str = "vitpose_b_wholebody.pth",
    device: str = "cuda:0",
    iou_thresh: float = 0.15,
    ratio_thresh: float = 0.95,
) -> str:
    """Thực thi toàn bộ luồng trích xuất 2D và lưu kết quả .npz (23 keypoints)."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Không tìm thấy video: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Kiểm tra thiết bị phần cứng
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA không khả dụng, chuyển sang chạy trên CPU.")
        device = "cpu"

    print(f"--- KHỞI TẠO PIPELINE 2D ({device}) | Đầu ra: {NUM_KPTS_OUT} điểm ---")
    detector = PersonDetectorYOLO(model_weight=yolo_weight, device=device)
    pose_estimator = ViTPoseInference(
        model_weight=vitpose_weight, yolo_weight=yolo_weight, device=device
    )

    with VideoReader(video_path) as reader:
        meta = reader.get_metadata()
        T = meta.total_frames
        print(f"Video: {Path(video_path).name} | Số khung: {T} | FPS: {meta.fps:.2f}")
        if not meta.is_cfr_standard:
            print("CẢNH BÁO: Tốc độ khung hình lệch khỏi chuẩn 60 FPS!")

        # Cấp phát sẵn mảng cố định (T, 23, 3) khởi tạo bằng NaN
        real_kpts_all = np.full((T, NUM_KPTS_OUT, 3), np.nan, dtype=np.float32)
        mirror_kpts_raw_all = np.full((T, NUM_KPTS_OUT, 3), np.nan, dtype=np.float32)
        real_bboxes_all = np.full((T, 4), np.nan, dtype=np.float32)
        mirror_bboxes_all = np.full((T, 4), np.nan, dtype=np.float32)
        torso_ratios_all = np.full((T,), np.nan, dtype=np.float32)
        flags_all: List[List[str]] = []

        for frame_idx, frame in tqdm(reader.frames(), total=T, desc="Trích xuất 2D"):
            # 1. Phát hiện BBoxes từ YOLO (lấy tối đa 5 người để phát hiện người lạ)
            yolo_boxes = detector.detect(frame, max_persons=5)

            # 2. Suy luận đủ 133 điểm khớp từ ViTPose (giữ nguyên để chirality ổn định)
            kpts_dict = pose_estimator.infer_frame(frame)
            kpts_list = [v for v in kpts_dict.values() if isinstance(v, np.ndarray)]

            # Khớp nối BBoxes tương ứng với từng Pose
            bboxes = _match_kpts_to_bboxes(kpts_list, yolo_boxes)

            # 3. Kiểm tra số lượng người phát hiện
            if len(kpts_list) < 2:
                flags_all.append(["THIEU_NGUOI"])
                continue

            # 4. Lọc người thứ ba (Bystander Filter)
            selected_kpts, selected_bboxes, drop_flags = drop_bystanders_frame(
                kpts_list, bboxes, top_k=2
            )

            if len(selected_kpts) < 2:
                flags_all.append(drop_flags + ["THIEU_NGUOI"])
                continue

            # 5. Phân định Thật - Gương (theo chiều dài thân) và kiểm tra che khuất.
            #    KHÔNG flip: mirror_kpts_raw được lưu nguyên trạng (vẫn làm việc trên 133 điểm).
            #    Tầng 3D sẽ gọi correct_virt() per-frame để so sánh hướng & quyết định flip.
            frame_res, assoc_flags = clean_and_associate(
                selected_kpts[0],
                selected_kpts[1],
                selected_bboxes[0],
                selected_bboxes[1],
                iou_thresh=iou_thresh,
                ratio_thresh=ratio_thresh,
            )

            current_flags = drop_flags + assoc_flags
            flags_all.append(current_flags)

            if frame_res is not None:
                # 6. Cắt lát xuống 23 điểm giải phẫu NGAY TẠI bước gán dữ liệu
                # Lưu mirror_kpts_raw (tọa độ RAW, không flip) — tầng 3D tự quyết định
                real_kpts_all[frame_idx] = _slice_kpts(frame_res.real_kpts)
                mirror_kpts_raw_all[frame_idx] = _slice_kpts(frame_res.mirror_kpts_raw)
                real_bboxes_all[frame_idx] = frame_res.real_bbox
                mirror_bboxes_all[frame_idx] = frame_res.mirror_bbox
                torso_ratios_all[frame_idx] = frame_res.torso_ratio

    # 7. Đóng gói và lưu mảng nén ra đĩa
    video_stem = Path(video_path).stem
    out_file = os.path.join(output_dir, f"{video_stem}.npz")
    metadata_dict = {
        "video_path": video_path,
        "video_name": video_stem,
        "fps": meta.fps,
        "total_frames": T,
        "width": meta.width,
        "height": meta.height,
        "duration_s": meta.duration_s,
        "is_cfr_standard": meta.is_cfr_standard,
        "iou_thresh": iou_thresh,
        "ratio_thresh": ratio_thresh,
        "num_keypoints": NUM_KPTS_OUT,
        "keypoint_format": KEYPOINT_FORMAT,
    }

    save_assigned_2d(
        output_path=out_file,
        real_kpts=real_kpts_all,
        mirror_kpts_raw=mirror_kpts_raw_all,
        real_bboxes=real_bboxes_all,
        mirror_bboxes=mirror_bboxes_all,
        flags_per_frame=flags_all,
        torso_ratios=torso_ratios_all,
        metadata=metadata_dict,
    )

    # 8. Báo cáo kiểm soát chất lượng (QC Report)
    valid_frames = int(np.sum(~np.isnan(real_kpts_all[:, 0, 0])))
    overlap_frames = sum(1 for f in flags_all if "HOP_CHONG" in f)
    bystander_frames = sum(1 for f in flags_all if "CO_NGUOI_THU_BA" in f)
    ambiguous_frames = sum(1 for f in flags_all if "THAN_GAN_BANG_NHAU" in f)
    missing_frames = sum(1 for f in flags_all if "THIEU_NGUOI" in f or "THIEU_DIEM_THAN" in f)

    print("\n" + "=" * 50)
    print("BÁO CÁO KIỂM SOÁT CHẤT LƯỢNG (QC SUMMARY)")
    print("=" * 50)
    print(f"• Tổng số khung hình:         {T}")
    print(f"• Khung gán thành công:       {valid_frames} ({valid_frames/T*100:.1f}%)")
    print(f"• Khung bị chồng lấn (IoU):   {overlap_frames}")
    print(f"• Khung có người thứ ba:      {bystander_frames}")
    print(f"• Khung mơ hồ tỉ lệ thân:     {ambiguous_frames}")
    print(f"• Khung mất dấu (NaN/Thiếu):  {missing_frames}")
    print(f"• Định dạng keypoints:        {KEYPOINT_FORMAT} ({NUM_KPTS_OUT} điểm)")
    print(f"File kết quả đã lưu:       {out_file}")
    print("=" * 50 + "\n")

    return out_file


def main():
    parser = argparse.ArgumentParser(description="Trích xuất 2D Pipeline gait_mirror")
    parser.add_argument("--video_path", type=str, required=True, help="Đường dẫn file video đầu vào")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/interim/assigned_2d",
        help="Thư mục lưu trữ file .npz",
    )
    parser.add_argument("--yolo_weight", type=str, default="yolo26n.pt")
    parser.add_argument("--vitpose_weight", type=str, default="vitpose_b_wholebody.pth")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--iou_thresh", type=float, default=0.15)
    parser.add_argument("--ratio_thresh", type=float, default=0.95)

    args = parser.parse_args()
    run_pipeline(
        video_path=args.video_path,
        output_dir=args.output_dir,
        yolo_weight=args.yolo_weight,
        vitpose_weight=args.vitpose_weight,
        device=args.device,
        iou_thresh=args.iou_thresh,
        ratio_thresh=args.ratio_thresh,
    )


if __name__ == "__main__":
    main()