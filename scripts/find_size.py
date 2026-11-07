import cv2

# Đọc thử 1 tấm ảnh đầu tiên trong tập dữ liệu
img_path = "data/raw/calibration/IMG_8367.JPG"
img = cv2.imread(img_path)

if img is None:
    print(f"[!] Không đọc được ảnh từ: {img_path}")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("Đang tự động dò tìm cấu hình BOARD_SIZE chuẩn...")

found = False
# Quét toàn bộ các tổ hợp kích thước góc phổ biến từ 4 đến 10
for h in range(4, 11):
    for w in range(4, 11):
        if w == h:
            continue
        ret, corners = cv2.findChessboardCorners(
            gray, (w, h),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if ret:
            print("\n" + "=" * 45)
            print(f">>> TÌM THẤY CHUẨN XÁC: BOARD_SIZE = ({w}, {h}) <<<")
            print("=" * 45)
            found = True
            break
    if found:
        break

if not found:
    print("\n[!] Không tìm thấy với ảnh gốc. Đang thử thu nhỏ ảnh (resize 50%)...")
    h_orig, w_orig = gray.shape[:2]
    gray_resized = cv2.resize(gray, (w_orig // 2, h_orig // 2))
    for h in range(4, 11):
        for w in range(4, 11):
            if w == h:
                continue
            ret, _ = cv2.findChessboardCorners(
                gray_resized, (w, h),
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            if ret:
                print("\n" + "=" * 45)
                print(f">>> TÌM THẤY KHI RESIZE: BOARD_SIZE = ({w}, {h}) <<<")
                print("=" * 45)
                found = True
                break
        if found:
            break

if not found:
    print("\n[!] Vẫn chưa tìm thấy góc nào. Nguyên nhân có thể do nền có quần caro hoặc tấm caro gương gây nhiễu.")