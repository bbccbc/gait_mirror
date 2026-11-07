"""
Tiền xử lý & làm mượt quỹ đạo 2D keypoints cho phân tích dáng đi (gait_mirror).

Vị trí trong pipeline:
    01_extract_2d.py  ->  (T, 23, 3) [x, y, conf]  ->  filters.py
                      ->  tam giác đạc 3D (DLT)  ->  Zeni (Heel-Strike / Toe-Off)

Các thành phần:
    1. interpolate_missing_keypoints   : loại điểm kém tin cậy / outlier, nội suy PCHIP
    2. butterworth_lowpass_filter      : Butterworth thông thấp zero-lag (filtfilt)
    3. OneEuroFilter                   : bộ lọc thích nghi theo vận tốc
    4. filter_keypoints_pipeline       : API tổng hợp, trả (kpts_lọc, mask_nội_suy)

Quy ước chung
-------------
* Trục thời gian luôn là axis 0.
* Mọi phép tính nội bộ dùng float64; kết quả được trả về cùng dtype float với đầu vào.
* NaN = "không có dữ liệu". Đoạn mất dấu dài hơn `max_gap` KHÔNG bị nội suy; các bộ lọc
  xử lý từng đoạn liên tục hợp lệ riêng biệt nên NaN không lan sang phần còn lại.
* `mask_interpolated[t, k] = True` nghĩa là tọa độ do nội suy/điền biên, không phải đo trực tiếp.
  Tọa độ còn NaN (mất dấu quá dài) có mask = False; pipeline 3D nên kiểm tra thêm np.isfinite.

Phụ thuộc: numpy, scipy.
"""

from __future__ import annotations

import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, filtfilt

__all__ = [
    "interpolate_missing_keypoints",
    "interpolate_missing_keypoints_with_mask",
    "butterworth_lowpass_filter",
    "OneEuroFilter",
    "filter_keypoints_pipeline",
]


# =============================================================================
# 0. Tiện ích chung
# =============================================================================
def _as_tkc(kpts: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """
    Đưa mảng đầu vào về dạng chuẩn (T, K, C) với C in {2, 3}.

    Chấp nhận:  (T, K, 2|3)  hoặc  (T, 2|3) [một điểm duy nhất -> K = 1].
    Trả về (mảng float64 chuẩn hoá, shape gốc) để khôi phục sau.
    """
    arr = np.asarray(kpts)
    orig_shape = arr.shape
    if arr.ndim == 2 and arr.shape[1] in (2, 3):
        arr = arr[:, None, :]
    elif arr.ndim == 3 and arr.shape[2] in (2, 3):
        pass
    else:
        raise ValueError(
            f"kpts phải có shape (T, K, 2|3) hoặc (T, 2|3), nhận được {orig_shape}"
        )
    return arr.astype(np.float64, copy=True), orig_shape


def _out_dtype(kpts: np.ndarray) -> np.dtype:
    dt = np.asarray(kpts).dtype
    return dt if np.issubdtype(dt, np.floating) else np.dtype(np.float64)


def _rolling_nanmedian(arr: np.ndarray, window: int, min_valid: int = 3) -> np.ndarray:
    """
    Trung vị trượt theo axis 0, bỏ qua NaN (vectorized bằng sort, không sinh cảnh báo).

    arr: (T, ...) ; trả về cùng shape. Ô nào có < `min_valid` mẫu hợp lệ trong cửa sổ -> NaN.
    Biên được đệm NaN (cửa sổ thu hẹp tự nhiên ở hai đầu).
    """
    half = window // 2
    pad = np.full((half,) + arr.shape[1:], np.nan)
    padded = np.concatenate([pad, arr, pad], axis=0)
    win = sliding_window_view(padded, window, axis=0)  # (T, ..., window)
    srt = np.sort(win, axis=-1)  # NaN dồn về cuối
    n = np.sum(~np.isnan(win), axis=-1)
    lo_i = np.clip((n - 1) // 2, 0, window - 1)[..., None]
    hi_i = np.clip(n // 2, 0, window - 1)[..., None]
    lo = np.take_along_axis(srt, lo_i, axis=-1)[..., 0]
    hi = np.take_along_axis(srt, hi_i, axis=-1)[..., 0]
    med = 0.5 * (lo + hi)
    med[n < min_valid] = np.nan
    return med


# =============================================================================
# 1. Khử outlier & nội suy bù khuyết
# =============================================================================
def _interpolate_core(
    coords: np.ndarray,
    conf: Optional[np.ndarray],
    conf_thresh: float,
    max_gap: int,
    jump_thresh_px: Optional[float],
    min_jump_px: float,
    hampel_k: float,
    hampel_window: int,
    fill_edges: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lõi thuật toán. coords: (T, K, 2) float64; conf: (T, K) hoặc None.
    Trả về (coords_đã_điền (T, K, 2), mask_interpolated (T, K)).
    """
    T, K, _ = coords.shape

    # --- (a) Điểm khuyết "cứng": NaN hoặc confidence thấp -------------------
    missing = ~np.isfinite(coords).all(axis=-1)
    if conf is not None:
        missing |= ~(conf >= conf_thresh)  # NaN conf cũng bị loại

    # --- (b) Outlier vận tốc/nhảy vọt (Hampel trên vị trí) -------------------
    # Mỗi điểm được so với trung vị trượt (±hampel_window//2 frame) của chính quỹ đạo nó.
    # Chuyển động nhanh nhưng đơn điệu (vung chân) có trung vị bám sát nên KHÔNG bị loại;
    # chỉ các "gai" nhảy đi rồi quay lại (hoán đổi trái/phải thoáng qua, đứt tracking)
    # mới có phần dư lớn. Ngưỡng thích nghi theo độ ồn của từng khớp.
    work = np.where(missing[..., None], np.nan, coords)
    med = _rolling_nanmedian(work, window=hampel_window)
    resid = np.linalg.norm(work - med, axis=-1)  # (T, K), NaN nếu không đánh giá được

    if jump_thresh_px is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            scale = np.nanmedian(resid, axis=0)  # (K,) độ ồn nền của từng khớp
        thr = np.fmax(min_jump_px, hampel_k * 1.4826 * scale)  # fmax: NaN -> min_jump_px
    else:
        thr = np.full(K, float(jump_thresh_px))
    with np.errstate(invalid="ignore"):
        outlier = resid > thr[None, :]  # NaN -> False
    valid = ~missing & ~outlier  # (T, K) điểm đo được, đáng tin

    # --- (c) Nội suy PCHIP trong các khoảng khuyết ≤ max_gap ------------------
    out = np.where(valid[..., None], coords, np.nan)
    mask_interp = np.zeros((T, K), dtype=bool)

    for k in range(K):
        idx = np.flatnonzero(valid[:, k])
        if idx.size == 0:
            continue

        fill_idx: List[np.ndarray] = []
        # Khoảng khuyết nằm giữa hai điểm hợp lệ liên tiếp
        if idx.size >= 2:
            gaps = idx[1:] - idx[:-1] - 1
            for j in np.flatnonzero((gaps > 0) & (gaps <= max_gap)):
                fill_idx.append(np.arange(idx[j] + 1, idx[j + 1]))
        if fill_idx:
            ti = np.concatenate(fill_idx)
            interp = PchipInterpolator(idx, coords[idx, k, :], axis=0)
            out[ti, k, :] = interp(ti)
            mask_interp[ti, k] = True

        # Biên đầu/cuối: forward/backward fill (chỉ khi đoạn rìa ≤ max_gap)
        if fill_edges:
            head = int(idx[0])
            if 0 < head <= max_gap:
                out[:head, k, :] = coords[idx[0], k, :]
                mask_interp[:head, k] = True
            tail = T - 1 - int(idx[-1])
            if 0 < tail <= max_gap:
                out[idx[-1] + 1 :, k, :] = coords[idx[-1], k, :]
                mask_interp[idx[-1] + 1 :, k] = True

    return out, mask_interp


def interpolate_missing_keypoints_with_mask(
    kpts: np.ndarray,
    conf_thresh: float = 0.3,
    max_gap: int = 15,
    *,
    jump_thresh_px: Optional[float] = None,
    min_jump_px: float = 12.0,
    hampel_k: float = 6.0,
    hampel_window: int = 7,
    fill_edges: bool = True,
    interp_conf: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Giống `interpolate_missing_keypoints` nhưng trả thêm mặt nạ nội suy.

    Returns:
        (kpts_out, mask_interpolated)
        kpts_out: cùng shape/dtype với đầu vào.
        mask_interpolated: bool, shape (T, K) (hoặc (T,) nếu đầu vào là (T, 2|3)).
    """
    if max_gap < 0:
        raise ValueError("max_gap phải >= 0")
    if hampel_window < 3 or hampel_window % 2 == 0:
        raise ValueError("hampel_window phải là số lẻ >= 3")

    arr, orig_shape = _as_tkc(kpts)
    coords = arr[..., :2]
    conf = arr[..., 2] if arr.shape[-1] == 3 else None

    filled, mask = _interpolate_core(
        coords, conf, conf_thresh, max_gap, jump_thresh_px, min_jump_px,
        hampel_k, hampel_window, fill_edges,
    )

    out = np.empty_like(arr)
    out[..., :2] = filled
    if conf is not None:
        ic = conf_thresh if interp_conf is None else interp_conf
        new_conf = np.where(np.isfinite(filled).all(-1), conf, 0.0)
        # Điểm bị loại do outlier nhưng conf gốc cao: conf cũng bị thay bởi ic khi được nội suy.
        new_conf = np.where(mask, ic, new_conf)
        # Điểm còn NaN (mất dấu dài): conf = 0
        new_conf = np.where(np.isfinite(filled).all(-1), new_conf, 0.0)
        out[..., 2] = new_conf

    dtype = _out_dtype(kpts)
    out = out.astype(dtype, copy=False).reshape(orig_shape)
    mask_out = mask if len(orig_shape) == 3 else mask[:, 0]
    return out, mask_out


def interpolate_missing_keypoints(
    kpts: np.ndarray,
    conf_thresh: float = 0.3,
    max_gap: int = 15,
    *,
    jump_thresh_px: Optional[float] = None,
    min_jump_px: float = 12.0,
    hampel_k: float = 6.0,
    hampel_window: int = 7,
    fill_edges: bool = True,
    interp_conf: Optional[float] = None,
) -> np.ndarray:
    """
    Khử outlier và nội suy bù khuyết quỹ đạo keypoints bằng PCHIP.

    Một điểm (t, k) bị coi là KHUYẾT (gán NaN) nếu:
      1. toạ độ chứa NaN, hoặc
      2. confidence < `conf_thresh` (chỉ khi mảng có cột conf), hoặc
      3. là outlier nhảy vọt: khoảng cách tới trung vị trượt quỹ đạo của chính nó vượt ngưỡng
         (tức vận tốc pixel đổi đột ngột phi giải phẫu — thường do hoán đổi trái/phải
         thoáng qua hoặc ViTPose "nhảy" sang vật thể khác).

    Các khoảng khuyết có độ dài ≤ `max_gap` được nội suy PCHIP dọc trục thời gian
    (bảo toàn tính đơn điệu/độ dốc, không vọt lố như cubic spline). Khoảng dài hơn giữ NaN.

    Args:
        kpts: (T, K, 3) [x, y, conf], (T, K, 2), hoặc một điểm (T, 2|3).
        conf_thresh: ngưỡng confidence tối thiểu để tin một điểm.
        max_gap: số frame mất dấu liên tiếp tối đa được phép nội suy
            (15 frame ≈ 0.25 s ở 60 Hz — vượt quá thì khớp có thể đã đổi động tác).
        jump_thresh_px: ngưỡng cố định (pixel) cho phần dư outlier. None -> thích nghi
            theo từng khớp: max(min_jump_px, hampel_k * 1.4826 * độ ồn nền).
        min_jump_px: sàn của ngưỡng thích nghi (pixel). Cần chỉnh theo độ phân giải video
            (mặc định hợp lý cho 1080p; tăng lên cho 4K).
        hampel_k: hệ số nhân độ ồn nền khi tính ngưỡng thích nghi.
        hampel_window: cửa sổ trung vị trượt (số lẻ, frame). 7 ≈ 0.12 s ở 60 Hz.
        fill_edges: điền forward/backward cho đoạn khuyết ở đầu/cuối chuỗi
            (chỉ khi đoạn đó ≤ max_gap; dài hơn vẫn NaN thay vì đoán mò).
        interp_conf: confidence gán cho điểm nội suy. None -> dùng `conf_thresh`
            (mức tối thiểu "chấp nhận được"). Dùng mask để phân biệt chính xác điểm nội suy.

    Returns:
        Mảng cùng shape/dtype với `kpts`. Điểm còn khuyết: toạ độ NaN, conf = 0.
    """
    out, _ = interpolate_missing_keypoints_with_mask(
        kpts, conf_thresh, max_gap,
        jump_thresh_px=jump_thresh_px, min_jump_px=min_jump_px,
        hampel_k=hampel_k, hampel_window=hampel_window,
        fill_edges=fill_edges, interp_conf=interp_conf,
    )
    return out


# =============================================================================
# 2. Butterworth thông thấp zero-lag
# =============================================================================
def _winter_cutoff_factor(order: int) -> float:
    """
    Hệ số C của Winter cho lọc hai chiều: sau forward-backward, đáp ứng biên độ bình phương
    nên điểm -3 dB dịch xuống C·fc, với  C = (√2 − 1)^(1/(2·order)).
    (order=2 -> 0.802 như bảng của Winter; order=4 -> 0.896.)
    """
    return (np.sqrt(2.0) - 1.0) ** (1.0 / (2.0 * order))


def _filtfilt_segments(
    y: np.ndarray, b: np.ndarray, a: np.ndarray, padlen: int, min_segment: int
) -> np.ndarray:
    """filtfilt một chuỗi 1-D có NaN: lọc từng đoạn liên tục hữu hạn riêng biệt."""
    out = y.copy()
    finite = np.isfinite(y)
    if not finite.any():
        return out
    # Tìm các đoạn True liên tục
    d = np.diff(np.concatenate([[0], finite.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    for s, e in zip(starts, ends):
        L = e - s
        if L < min_segment:
            continue  # quá ngắn để lọc đáng tin: giữ nguyên số đo
        out[s:e] = filtfilt(b, a, y[s:e], padtype="odd", padlen=min(padlen, L - 1))
    return out


def butterworth_lowpass_filter(
    data: np.ndarray,
    cutoff: float = 6.0,
    fs: float = 60.0,
    order: int = 4,
    *,
    compensate_filtfilt: bool = True,
    padlen: Optional[int] = None,
    min_segment: Optional[int] = None,
) -> np.ndarray:
    """
    Butterworth thông thấp zero-lag (forward-backward) cho tín hiệu cơ sinh học.

    Lọc hai chiều bằng `scipy.signal.filtfilt` triệt tiêu độ lệch pha nên đỉnh
    Heel-Strike / Toe-Off không bị dịch thời gian (điều bắt buộc cho Zeni).

    Lưu ý về tần số cắt (Winter, *Biomechanics and Motor Control of Human Movement*):
    chạy bộ lọc hai lần làm biên độ bị bình phương, nên điểm -3 dB thực tế thấp hơn `cutoff`
    thiết kế (≈ 0.896·fc với bậc 4). Với `compensate_filtfilt=True` (mặc định), bộ lọc được
    thiết kế ở `cutoff / 0.896` để điểm -3 dB sau lọc hai chiều đúng bằng `cutoff`.
    Đặt False nếu muốn dùng nguyên `cutoff` làm tham số thiết kế như `butter()` thông thường.

    Xử lý biên: `padtype='odd'` (phản xạ lẻ) để giảm hiện tượng văng rìa. Dữ liệu chứa NaN
    được lọc theo từng đoạn liên tục hợp lệ, NaN được giữ nguyên.

    Args:
        data: mảng (T, ...) — trục thời gian là axis 0. Mọi cột đều bị lọc, nên chỉ truyền toạ độ
            (không truyền cột confidence).
        cutoff: tần số cắt (Hz). 6 Hz là chuẩn cho quỹ đạo khớp khi đi bộ.
        fs: tần số lấy mẫu (Hz), ≈ 60 cho video lab.
        order: bậc bộ lọc thiết kế (4 theo yêu cầu; hiệu dụng sau filtfilt là bậc 8 về độ dốc).
        compensate_filtfilt: bù hệ số Winter như mô tả ở trên.
        padlen: số mẫu đệm hai đầu. None -> max(3·(order+1), 1.5·fs/cutoff), tự giới hạn theo độ dài đoạn.
        min_segment: đoạn liên tục ngắn hơn giá trị này (mẫu) được giữ nguyên không lọc.
            None -> fs/cutoff (một chu kỳ tần số cắt, ≈ 10 mẫu).

    Returns:
        Mảng cùng shape với `data` (float).
    """
    if fs <= 0 or cutoff <= 0:
        raise ValueError("fs và cutoff phải > 0")
    if order < 1:
        raise ValueError("order phải >= 1")

    x = np.asarray(data, dtype=np.float64)
    if x.ndim < 1:
        raise ValueError("data phải có ít nhất 1 chiều (thời gian ở axis 0)")

    nyq = fs / 2.0
    design_fc = cutoff / _winter_cutoff_factor(order) if compensate_filtfilt else cutoff
    if design_fc >= nyq:
        raise ValueError(
            f"Tần số cắt thiết kế {design_fc:.2f} Hz phải < Nyquist {nyq:.2f} Hz "
            f"(giảm cutoff hoặc tăng fs)"
        )
    b, a = butter(order, design_fc / nyq, btype="low")

    if padlen is None:
        padlen = max(3 * (order + 1), int(round(1.5 * fs / cutoff)))
    if min_segment is None:
        min_segment = max(int(round(fs / cutoff)), 4)

    T = x.shape[0]
    flat = x.reshape(T, -1)  # (T, M)
    out = flat.copy()

    finite_all = np.isfinite(flat).all(axis=0)  # (M,)
    # Cột không NaN: lọc hàng loạt trong một lần gọi (nhanh); phải đủ dài hơn padlen.
    full_cols = np.flatnonzero(finite_all)
    if full_cols.size and T > min_segment:
        out[:, full_cols] = filtfilt(
            b, a, flat[:, full_cols], axis=0, padtype="odd", padlen=min(padlen, T - 1)
        )
    # Cột có NaN: lọc từng đoạn
    for m in np.flatnonzero(~finite_all):
        out[:, m] = _filtfilt_segments(flat[:, m], b, a, padlen, min_segment)

    return out.reshape(x.shape)


# =============================================================================
# 3. One-Euro Filter
# =============================================================================
class OneEuroFilter:
    """
    One-Euro Filter (Casiez et al., CHI 2012) — lọc thông thấp bậc 1 với tần số cắt thích nghi:

        fc = min_cutoff + beta * |vận tốc đã làm mượt|

    * Đứng yên (Stance): vận tốc ≈ 0 -> fc ≈ min_cutoff (thấp) -> khử mạnh jitter.
    * Vung nhanh (Swing): vận tốc lớn -> fc tăng -> trễ nhỏ, bám đúng đỉnh gia tốc.

    Đơn vị: tọa độ pixel, vận tốc pixel/giây (nên `beta` phụ thuộc độ phân giải/khoảng cách
    camera). Gợi ý chỉnh: (1) đặt beta = 0, hạ `min_cutoff` đến khi jitter lúc đứng yên đủ êm;
    (2) tăng `beta` đến khi pha vung hết bị trễ/bẹt đỉnh.

    Args:
        freq: tần số lấy mẫu (Hz).
        min_cutoff: tần số cắt cơ bản (Hz) khi đứng yên.
        beta: độ nhạy theo vận tốc (Hz trên mỗi pixel/giây).
        d_cutoff: tần số cắt (Hz) của bước làm mượt đạo hàm.
        vector_last_axis: True -> axis cuối là vector toạ độ (x, y); vận tốc được tính là độ
            lớn vector nên x và y dùng chung một fc (tránh méo hướng). False -> từng phần tử độc lập.

    Cách dùng:
        f = OneEuroFilter(freq=60)
        y = f.filter_batch(traj)          # offline, (T, ...)
        y_t = f(x_t)                      # streaming từng frame
    NaN ở đầu vào cho ra NaN và đặt lại trạng thái, nên sau khoảng mất dấu bộ lọc khởi động lại
    thay vì kéo dữ liệu cũ sang.
    """

    class _State:
        __slots__ = ("x_hat", "dx_hat")

        def __init__(self) -> None:
            self.x_hat: Optional[np.ndarray] = None
            self.dx_hat: Optional[np.ndarray] = None

    def __init__(
        self,
        freq: float = 60.0,
        min_cutoff: float = 4.0,
        beta: float = 0.01,
        d_cutoff: float = 1.0,
        vector_last_axis: bool = False,
    ) -> None:
        if freq <= 0 or min_cutoff <= 0 or d_cutoff <= 0 or beta < 0:
            raise ValueError("freq, min_cutoff, d_cutoff phải > 0 và beta >= 0")
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.vector_last_axis = bool(vector_last_axis)
        self._alpha_d = self._alpha(self.d_cutoff)
        self._state = self._State()

    # ---- nội bộ -------------------------------------------------------------
    def _alpha(self, cutoff: np.ndarray | float) -> np.ndarray | float:
        # alpha = 1 / (1 + tau/Te), tau = 1/(2π·fc), Te = 1/freq
        return 1.0 / (1.0 + self.freq / (2.0 * np.pi * cutoff))

    def _step(self, x: np.ndarray, st: "_State") -> np.ndarray:
        vec = self.vector_last_axis
        finite = np.isfinite(x)
        valid = finite.all(axis=-1, keepdims=True) if vec else finite

        if st.x_hat is None:
            st.x_hat = np.full(x.shape, np.nan)
            st.dx_hat = np.zeros(x.shape)

        prev_fin = np.isfinite(st.x_hat)
        have = prev_fin.all(axis=-1, keepdims=True) if vec else prev_fin
        use = valid & have  # có cả mẫu mới lẫn trạng thái trước

        x0 = np.where(valid, x, 0.0)
        p0 = np.where(have, st.x_hat, 0.0)

        dx = (x0 - p0) * self.freq
        dx_hat = self._alpha_d * dx + (1.0 - self._alpha_d) * st.dx_hat
        dx_hat = np.where(use, dx_hat, 0.0)  # mẫu khởi động/sau mất dấu: vận tốc 0

        speed = np.linalg.norm(dx_hat, axis=-1, keepdims=True) if vec else np.abs(dx_hat)
        a = self._alpha(self.min_cutoff + self.beta * speed)

        filtered = a * x0 + (1.0 - a) * p0
        x_hat = np.where(valid, np.where(have, filtered, x), np.nan)

        st.x_hat, st.dx_hat = x_hat, dx_hat
        return x_hat

    # ---- API công khai ------------------------------------------------------
    def reset(self) -> None:
        """Xoá trạng thái streaming."""
        self._state = self._State()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Lọc một frame (streaming, nhân quả). Trạng thái được giữ giữa các lần gọi."""
        return self._step(np.asarray(x, dtype=np.float64), self._state)

    def _run_fast(self, data: np.ndarray) -> np.ndarray:
        """Đường nhanh cho dữ liệu không có NaN (cùng công thức với `_step`)."""
        vec = self.vector_last_axis
        ad, k = self._alpha_d, self.freq / (2.0 * np.pi)
        out = np.empty_like(data)
        x_hat = data[0].copy()
        dx_hat = np.zeros_like(x_hat)
        out[0] = x_hat
        for t in range(1, data.shape[0]):
            xt = data[t]
            dx_hat = ad * (xt - x_hat) * self.freq + (1.0 - ad) * dx_hat
            if vec:
                speed = np.sqrt(np.sum(dx_hat * dx_hat, axis=-1, keepdims=True))
            else:
                speed = np.abs(dx_hat)
            a = 1.0 / (1.0 + k / (self.min_cutoff + self.beta * speed))
            x_hat = x_hat + a * (xt - x_hat)
            out[t] = x_hat
        return out

    def _run(self, data: np.ndarray) -> np.ndarray:
        if data.shape[0] == 0:
            return data.copy()
        if np.isfinite(data).all():
            return self._run_fast(data)
        st = self._State()
        out = np.empty_like(data)
        for t in range(data.shape[0]):
            out[t] = self._step(data[t], st)
        return out

    def filter_batch(self, data: np.ndarray, zero_phase: bool = True) -> np.ndarray:
        """
        Lọc cả chuỗi offline. Không ảnh hưởng trạng thái streaming.

        Args:
            data: (T, ...) trục thời gian là axis 0 (vd. (T, K, 2) với vector_last_axis=True).
            zero_phase: True -> chạy thêm một lượt ngược thời gian trên kết quả (forward-backward),
                triệt độ trễ pha của bộ lọc nhân quả để không làm lệch thời điểm HS/TO.
                False -> chỉ lượt xuôi (giống chạy thời gian thực, có trễ nhỏ).

        Returns:
            Mảng cùng shape (float64). NaN được giữ nguyên.
        """
        x = np.asarray(data, dtype=np.float64)
        if x.ndim < 1:
            raise ValueError("data phải có ít nhất 1 chiều")
        if self.vector_last_axis and x.ndim < 2:
            raise ValueError("vector_last_axis=True cần data có >= 2 chiều")
        y = self._run(x)
        if zero_phase:
            y = self._run(y[::-1])[::-1]
        return y


# =============================================================================
# 4. Master API
# =============================================================================
def filter_keypoints_pipeline(
    kpts: np.ndarray,
    fs: float = 60.0,
    cutoff: float = 6.0,
    conf_thresh: float = 0.3,
    method: str = "butterworth",
    *,
    max_gap: int = 15,
    interp_conf: Optional[float] = None,
    one_euro_params: Optional[Dict[str, float]] = None,
    compensate_filtfilt: bool = True,
    **interp_kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pipeline tiền xử lý hoàn chỉnh cho keypoints 2D.

    Bước 1  Loại điểm confidence thấp/NaN/outlier, nội suy PCHIP các khoảng ≤ `max_gap`.
    Bước 2  Làm mượt toạ độ (x, y) của từng khớp bằng Butterworth zero-lag hoặc One-Euro.
    Bước 3  Gộp lại confidence: điểm đo được giữ conf gốc; điểm nội suy nhận `interp_conf`
            (mặc định = conf_thresh); điểm còn NaN có conf = 0.

    Args:
        kpts: (T, K, 3) [x, y, conf], K = 23. (Cũng chấp nhận (T, K, 2) không có conf — khi đó
            chỉ lọc theo NaN/outlier và trả về mảng (T, K, 2).)
        fs: tần số lấy mẫu (Hz).
        cutoff: tần số cắt Butterworth (Hz). Bị bỏ qua khi method="one_euro".
        conf_thresh: ngưỡng confidence.
        method: "butterworth" | "one_euro".
        max_gap: khoảng mất dấu tối đa được nội suy (frame).
        interp_conf: xem `interpolate_missing_keypoints`.
        one_euro_params: dict ghi đè tham số OneEuroFilter (min_cutoff, beta, d_cutoff).
        compensate_filtfilt: bù hệ số Winter cho Butterworth (xem `butterworth_lowpass_filter`).
        **interp_kwargs: chuyển tiếp tới bước nội suy (jump_thresh_px, min_jump_px,
            hampel_k, hampel_window, fill_edges).

    Returns:
        (filtered_kpts, mask_interpolated)
        filtered_kpts: cùng shape/dtype float với đầu vào; điểm mất dấu quá dài vẫn NaN (conf = 0).
        mask_interpolated: bool (T, K); True = tọa độ do nội suy/điền biên, không phải đo trực tiếp.
    """
    if method not in ("butterworth", "one_euro"):
        raise ValueError("method phải là 'butterworth' hoặc 'one_euro'")
    arr_in = np.asarray(kpts)
    if arr_in.ndim != 3 or arr_in.shape[2] not in (2, 3):
        raise ValueError(f"kpts phải có shape (T, K, 2|3), nhận được {arr_in.shape}")

    # Bước 1
    interp, mask = interpolate_missing_keypoints_with_mask(
        arr_in, conf_thresh, max_gap, interp_conf=interp_conf, **interp_kwargs
    )
    interp = interp.astype(np.float64)
    coords = interp[..., :2]

    # Bước 2
    if method == "butterworth":
        smooth = butterworth_lowpass_filter(
            coords, cutoff=cutoff, fs=fs, order=4, compensate_filtfilt=compensate_filtfilt
        )
    else:
        params = {"min_cutoff": 4.0, "beta": 0.01, "d_cutoff": 1.0}
        if one_euro_params:
            params.update(one_euro_params)
        smooth = OneEuroFilter(freq=fs, vector_last_axis=True, **params).filter_batch(
            coords, zero_phase=True
        )

    # Bước 3
    out = np.empty(arr_in.shape, dtype=np.float64)
    out[..., :2] = smooth
    if arr_in.shape[2] == 3:
        finite = np.isfinite(smooth).all(axis=-1)
        out[..., 2] = np.where(finite, interp[..., 2], 0.0)  # đo được: conf gốc; nội suy: interp_conf

    return out.astype(_out_dtype(arr_in), copy=False), mask


# =============================================================================
# Kiểm thử đơn vị (chạy: python filters.py)
# =============================================================================
def _make_synthetic(T: int = 1800, fs: float = 60.0, seed: int = 0):
    """
    Sinh quỹ đạo giả lập của 23 khớp (chu kỳ bước 1.0 s) + nhiễu đo.
    Trả (truth (T, K, 2), noisy (T, K, 3), t). Bàn chân có pha đứng yên phẳng (stance) và
    pha vung (swing) để kiểm tra One-Euro.
    """
    rng = np.random.default_rng(seed)
    K = 23
    t = np.arange(T) / fs
    phase = 2 * np.pi * t / 1.0
    truth = np.zeros((T, K, 2))
    for k in range(K):
        ph = phase + 0.2 * k
        x = 500 + 15 * k + 120 * np.sin(ph) * (0.3 + 0.7 * (k >= 11))
        y = 300 + 25 * k - 70 * np.maximum(0.0, np.sin(ph)) ** 2  # phẳng khi sin<0
        truth[:, k, 0], truth[:, k, 1] = x, y
    noisy = np.concatenate([truth + rng.normal(0, 2.0, truth.shape), np.full((T, K, 1), 0.9)], -1)
    return truth, noisy, t


def _run_self_tests() -> None:
    fs = 60.0
    truth, noisy, t = _make_synthetic()
    T, K, _ = noisy.shape
    rng = np.random.default_rng(1)
    bad = noisy.copy()

    # Sự cố được cấy vào khớp 15 (cổ chân trái) và khớp 3
    k = 15
    bad[200:208, k, :2] = np.nan  # gap ngắn (8 frame)  -> phải nội suy
    bad[500:540, k, :2] = np.nan  # gap dài (40 frame)  -> phải giữ NaN
    bad[800:803, k, 2] = 0.05  # conf thấp 3 frame     -> nội suy
    bad[1000, k, :2] += [180.0, -150.0]  # spike 1 frame    -> loại rồi nội suy
    bad[1200:1202, k, :2] += [-160.0, 120.0]  # spike 2 frame
    bad[:5, 3, :2] = np.nan  # rìa đầu
    bad[-4:, 3, :2] = np.nan  # rìa cuối

    # --- Test 1: nội suy ----------------------------------------------------
    out, mask = interpolate_missing_keypoints_with_mask(bad, 0.3, 15)
    assert out.shape == bad.shape and mask.shape == (T, K)
    assert mask[200:208, k].all() and not mask[500:540, k].any()
    assert np.isnan(out[500:540, k, :2]).all(), "gap dài phải giữ NaN"
    assert np.isfinite(out[200:208, k, :2]).all()
    err = np.abs(out[200:208, k, :2] - truth[200:208, k]).max()
    assert err < 12, f"sai số nội suy gap ngắn quá lớn: {err:.1f}px"
    assert mask[800:803, k].all()
    for s in (1000, 1200, 1201):
        assert mask[s, k], f"spike frame {s} chưa bị loại"
        assert np.abs(out[s, k, :2] - truth[s, k]).max() < 15
    # Không đụng chạm dữ liệu sạch: false-positive rất thấp ở khớp không bị cấy lỗi
    assert mask[:, [0, 5, 10, 20]].mean() < 0.005, "báo outlier nhầm quá nhiều trên dữ liệu sạch"
    assert mask[:5, 3].all() and mask[-4:, 3].all(), "đầu/cuối phải được điền"
    assert np.allclose(out[:5, 3, :2], out[5, 3, :2])
    assert (out[mask, 2] == 0.3).all(), "conf điểm nội suy phải = conf_thresh"
    # (T, 2) và (T, K, 2)
    o2 = interpolate_missing_keypoints(bad[:, k, :2], max_gap=15)
    assert o2.shape == (T, 2)
    o3 = interpolate_missing_keypoints(bad[:, :, :2])
    assert o3.shape == (T, K, 2)
    print("[OK] Nội suy + khử outlier")

    # --- Test 2: Butterworth ----------------------------------------------------
    n = 2400
    tt = np.arange(n) / fs
    for f_hz, lo, hi in [(2.0, 0.95, 1.05), (6.0, 0.66, 0.75), (15.0, 0.0, 0.02)]:
        sig = np.sin(2 * np.pi * f_hz * tt)
        y = butterworth_lowpass_filter(sig, cutoff=6.0, fs=fs, order=4)
        core = slice(300, n - 300)
        gain = np.sqrt(np.mean(y[core] ** 2) / np.mean(sig[core] ** 2))
        assert lo <= gain <= hi, f"độ lợi {f_hz} Hz = {gain:.3f} ngoài [{lo}, {hi}]"
        if f_hz == 2.0:  # zero-lag: lệch pha ≈ 0
            lag = np.argmax(np.correlate(y[core], sig[core], "full")) - (len(sig[core]) - 1)
            assert lag == 0, f"filtfilt phải zero-lag, lệch {lag} mẫu"
    # Nhiễu giảm, đỉnh (swing apex) không dịch quá 1 frame
    sm = butterworth_lowpass_filter(noisy[..., :2], 6.0, fs, 4)
    rmse_raw = np.sqrt(np.mean((noisy[..., :2] - truth) ** 2))
    rmse_flt = np.sqrt(np.mean((sm - truth) ** 2))
    assert rmse_flt < 0.7 * rmse_raw, (rmse_raw, rmse_flt)
    from scipy.signal import find_peaks

    pk_true = find_peaks(-truth[:, 19, 1], distance=40, prominence=30)[0]  # y nhỏ nhất = nhấc cao nhất
    pk_flt = find_peaks(-sm[:, 19, 1], distance=40, prominence=30)[0]
    m = min(len(pk_true), len(pk_flt))
    assert m > 20 and np.abs(pk_true[:m] - pk_flt[:m]).max() <= 1, "đỉnh bị dịch thời gian"
    # NaN: không lan, đoạn dài giữ nguyên NaN
    sm_nan = butterworth_lowpass_filter(out[..., :2], 6.0, fs, 4)
    assert np.isnan(sm_nan[500:540, k]).all() and np.isfinite(sm_nan[:480, k]).all()
    assert np.isfinite(sm_nan[560:, k]).all()
    print(f"[OK] Butterworth: RMSE {rmse_raw:.2f} -> {rmse_flt:.2f}px, zero-lag, -3dB@6Hz")

    # --- Test 3: One-Euro -------------------------------------------------------
    f = OneEuroFilter(freq=fs, vector_last_axis=True)  # tham số mặc định
    oe = f.filter_batch(noisy[..., :2], zero_phase=True)
    stance = truth[:, 19, 1] > truth[:, 19, 1].max() - 1.0  # y phẳng = đứng yên
    # Chọn các frame đứng yên "sâu" (loại 3 frame sát biên pha vung)
    stance = np.convolve(stance, np.ones(7), "same") == 7
    j_raw = np.std(noisy[stance, 19, 1] - truth[stance, 19, 1])
    j_flt = np.std(oe[stance, 19, 1] - truth[stance, 19, 1])
    assert j_flt < 0.55 * j_raw, f"jitter lúc đứng yên chưa giảm đủ: {j_raw:.2f}->{j_flt:.2f}"
    rmse_oe = np.sqrt(np.mean((oe - truth) ** 2))
    # One-Euro làm phẳng nhẹ đỉnh quỹ đạo (vận tốc ~0 -> fc thấp), nên không kỳ vọng thắng
    # Butterworth trên tín hiệu tuần hoàn; chỉ yêu cầu không méo quá mức so với nhiễu đo.
    assert rmse_oe < 1.5 * rmse_raw and rmse_flt < rmse_oe, (rmse_raw, rmse_flt, rmse_oe)
    swing_pk = find_peaks(-oe[:, 19, 1], distance=40, prominence=30)[0]
    m = min(len(pk_true), len(swing_pk))
    assert np.abs(pk_true[:m] - swing_pk[:m]).max() <= 1, "One-Euro zero-phase làm lệch đỉnh"
    # Streaming nhất quán với batch một chiều & xử lý NaN
    fs_ = OneEuroFilter(freq=fs, vector_last_axis=True)
    stream = np.stack([fs_(noisy[i, :, :2]) for i in range(200)])
    ref = OneEuroFilter(freq=fs, vector_last_axis=True).filter_batch(noisy[:200, :, :2], zero_phase=False)
    assert np.allclose(stream, ref)
    nan_in = noisy[:, :, :2].copy()
    nan_in[100:110, 19] = np.nan
    r = f.filter_batch(nan_in)
    assert np.isnan(r[100:110, 19]).all() and np.isfinite(r[:100, 19]).all() and np.isfinite(r[110:, 19]).all()
    print(f"[OK] One-Euro: jitter stance {j_raw:.2f} -> {j_flt:.2f}px, RMSE {rmse_raw:.2f} -> {rmse_oe:.2f}px")

    # --- Test 4: pipeline + tốc độ ---------------------------------------------
    for method in ("butterworth", "one_euro"):
        t0 = time.perf_counter()
        fk, mk = filter_keypoints_pipeline(bad, fs=fs, cutoff=6.0, conf_thresh=0.3, method=method)
        dt = time.perf_counter() - t0
        assert fk.shape == (T, K, 3) and mk.shape == (T, K) and mk.dtype == bool
        assert fk.dtype == bad.dtype
        assert np.isnan(fk[500:540, k, :2]).all() and (fk[500:540, k, 2] == 0).all()
        assert mk[200:208, k].all() and not mk[:100, 0].any()
        meas = ~mk & np.isfinite(fk[..., 0])
        assert (fk[meas, 2] == bad[meas, 2]).all(), "conf điểm đo được phải được bảo toàn"
        print(f"[OK] pipeline[{method}]: {dt * 1000:.0f} ms cho T={T}, K={K}")
        if method == "butterworth":
            assert dt < 0.1, f"Butterworth pipeline quá chậm: {dt:.3f}s"

    # Đầu vào (T, K, 2) không có conf
    f2, m2 = filter_keypoints_pipeline(bad[..., :2])
    assert f2.shape == (T, K, 2) and m2.shape == (T, K)
    # Đầu vào float32 -> đầu ra float32
    fk32, _ = filter_keypoints_pipeline(bad.astype(np.float32))
    assert fk32.dtype == np.float32
    # Tham số sai
    for bad_call in (
        lambda: filter_keypoints_pipeline(bad, method="kalman"),
        lambda: butterworth_lowpass_filter(noisy[..., :2], cutoff=40.0, fs=60.0),
        lambda: interpolate_missing_keypoints(np.zeros((10,))),
    ):
        try:
            bad_call()
            raise AssertionError("đáng lẽ phải báo ValueError")
        except ValueError:
            pass
    print("[OK] dtype / validate tham số")
    print("\nTẤT CẢ KIỂM THỬ ĐÃ PASS")


if __name__ == "__main__":
    _run_self_tests()