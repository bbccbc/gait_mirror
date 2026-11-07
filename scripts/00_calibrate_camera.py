"""Script thuc thi hieu chuan noi tai camera tu tap anh thuc te (M1).

Chay:
    python scripts/00_calibrate_camera.py
"""
import glob
from pathlib import Path
from gaitmirror.calibration.intrinsics import (
    calibrate_from_images,
    save_intrinsics,
    summary,
)

# 1. Cấu hình thông số bàn cờ
# Lưu ý: board_size là số GIAO ĐIỂM BÊN TRONG (cột, hàng)
BOARD_SIZE = (9, 6)        # Bàn cờ cầm ngang: 8 ô ngang x 9 ô dọc -> 7x8 góc trong
SQUARE_SIZE_MM = 28.0      # Kích thước cạnh 1 ô vuông đo bằng thước dây (mm)

INPUT_DIR = "data/raw/calibration"
OUTPUT_CONFIG = "configs/camera_intrinsics.json"
MAX_ALLOWED_RMS_PX = 1.0   # Ngưỡng RMS tối đa cho phép lưu file cấu hình chuẩn


def main():
    # 2. Quét toàn bộ ảnh trong thư mục
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(f"{INPUT_DIR}/{ext}"))
    
    # [ĐÃ SỬA 1]: Khử trùng lặp đường dẫn do Windows không phân biệt hoa/thường (từ 44 về 22 ảnh)
    image_paths = sorted(list(set(image_paths)))
    
    if not image_paths:
        print(f"[!] Khong tim thay anh nao trong: {INPUT_DIR}")
        return

    print(f"[*] Tim thay {len(image_paths)} anh. Bat dau hieu chuan...")

    # 3. Chạy thuật toán hiệu chuẩn
    try:
        res = calibrate_from_images(
            image_paths=image_paths,
            board_size=BOARD_SIZE,
            square_size_mm=SQUARE_SIZE_MM,
            min_images=10,
            fix_k3=True,                 # Khóa k3=0 để tránh overfitting
        )
    except ValueError as e:
        print(f"[ERROR] {e}")
        return

    # 4. In báo cáo đánh giá nhanh
    s = summary(res)
    print("\n" + "=" * 40)
    print("KET QUA HIEU CHUAN (INTRINSICS SUMMARY)")
    print("=" * 40)
    print(f"- So anh hop le su dung: {s['n_images']}/{len(image_paths)}")
    print(f"- Sai so reprojection RMS: {s['rms_px']:.3f} px (Muc tieu: < {MAX_ALLOWED_RMS_PX} px)")
    print(f"- RMS trung binh tung anh: {s['per_image_rms_mean']:.3f} px")
    print(f"- Tieu cu fx, fy: ({s['fx']:.1f}, {s['fy']:.1f}) | Ty le fx/fy: {s['fx_over_fy']:.4f}")
    print(f"- Do lech quang tam: cx_offset={s['cx_offset_frac']*100:.2f}%, cy_offset={s['cy_offset_frac']*100:.2f}%")
    print(f"- He so meo (k1, k2, p1, p2, k3): {res.dist.ravel().round(5)}")

    if res.rejected_paths:
        print(f"\n[!] Co {len(res.rejected_paths)} anh bi loai bo:")
        for p in res.rejected_paths:
            print(f"  + {Path(p).name}")

    # [ĐÃ SỬA 2]: Kiểm tra chất lượng RMS trước khi cho phép ghi đè cấu hình
    if s["rms_px"] >= MAX_ALLOWED_RMS_PX:
        print(f"\n[CANH BAO] RMS = {s['rms_px']:.3f} px vuot nguong muc tieu (< {MAX_ALLOWED_RMS_PX} px)!")
        print("[!] Khong ghi de len file chuan de tranh lam sai lech pipeline dung 3D.")
        debug_failed_config = "configs/camera_intrinsics_failed.json"
        Path(debug_failed_config).parent.mkdir(parents=True, exist_ok=True)
        save_intrinsics(debug_failed_config, res)
        print(f"[*] Da luu ket qua vao file debug: {debug_failed_config}")
        return

    # 5. Lưu ra configs/camera_intrinsics.json khi đã đạt chuẩn
    Path(OUTPUT_CONFIG).parent.mkdir(parents=True, exist_ok=True)
    save_intrinsics(
        OUTPUT_CONFIG,
        res,
        meta={
            "description": "Camera calibration data for LLD gait analysis",
            "device": "Phone/Camera RGB",
            "fps": 60,
        },
    )
    print(f"\n[OK] Da luu cau hinh hop le vao: {OUTPUT_CONFIG}")


if __name__ == "__main__":
    main()