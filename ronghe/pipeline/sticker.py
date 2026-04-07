"""
混凝土试块造假识别 Pipeline — 贴纸检测
=======================================
蓝色圆形甬砼码贴纸检测 (HSV → HSV亮色 → HoughCircles，无SAM)
"""

import cv2
import numpy as np


def _refine_ellipse_by_gradient(img_bgr, cx, cy, r_hsv, ell_hsv):
    """HSV 定位中心后，用径向梯度幅值找物理阴影边缘，拟合更准确的椭圆。
    搜索范围: r_hsv*0.85 ~ r_hsv*1.5，72 个方向取梯度幅值峰值，
    IQR 过滤后 fitEllipse。失败时返回 None。"""
    import math
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.5)
    sobelx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)

    h, w = img_bgr.shape[:2]
    n_angles = 72
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    r_min = int(r_hsv * 0.85)
    r_max = int(r_hsv * 1.5)

    edge_radii, edge_points = [], []
    for angle in angles:
        best_r, best_g = None, 0
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        for r in range(max(r_min, 3), min(r_max, int(min(w, h) / 2))):
            xi, yi = int(cx + r * cos_a), int(cy + r * sin_a)
            if 0 <= xi < w and 0 <= yi < h:
                g = grad_mag[yi, xi]
                if g > best_g:
                    best_g = g
                    best_r = r
        if best_r is not None and best_g > 15:
            edge_radii.append(best_r)
            edge_points.append([cx + best_r * cos_a, cy + best_r * sin_a])

    if len(edge_radii) < 20:
        return None

    arr = np.array(edge_radii)
    q25, q75 = np.percentile(arr, [25, 75])
    iqr = q75 - q25
    keep = (arr >= q25 - 0.5 * iqr) & (arr <= q75 + 0.5 * iqr)
    pts = np.array(edge_points)[keep].reshape(-1, 1, 2).astype(np.float32)
    if len(pts) < 5:
        return None
    return cv2.fitEllipse(pts)


def _ellipse_iou(contour, ellipse, img_shape):
    """计算轮廓 mask 与拟合椭圆 mask 的 IoU。"""
    h, w = img_shape[:2]
    m_cnt = np.zeros((h, w), np.uint8)
    cv2.drawContours(m_cnt, [contour], -1, 255, -1)
    m_ell = np.zeros((h, w), np.uint8)
    cv2.ellipse(m_ell, ellipse, 255, -1)
    inter = np.count_nonzero(m_cnt & m_ell)
    union = np.count_nonzero(m_cnt | m_ell)
    return inter / union if union > 0 else 0.0


def _hsv_candidates(img_bgr, hsv, h_lo, s_lo, v_lo, h_hi=135):
    """在给定 HSV 范围内找圆形候选轮廓。
    返回 (scored_list, mask)，scored_list 元素为 (circ*aspect score, contour)，
    已按得分降序排列（最圆的在前）。
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.inRange(hsv, (h_lo, s_lo, v_lo), (h_hi, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = img_bgr.shape[0] * img_bgr.shape[1]
    AREA_MIN = img_area * 0.005
    AREA_MAX = img_area * 0.65
    scored = []
    for c in contours:
        a = cv2.contourArea(c)
        if a < AREA_MIN or a > AREA_MAX or len(c) < 5:
            continue
        perim = cv2.arcLength(c, True)
        circ  = 4 * np.pi * a / (perim ** 2) if perim > 0 else 0
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0
        score = circ * aspect
        scored.append((score, c))
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored, mask


def _hough_fallback(img_bgr):
    """HoughCircles 兜底：纯形状检测，不依赖颜色。"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    h, w = img_bgr.shape[:2]
    img_area = h * w
    min_r = int((img_area * 0.005) ** 0.5)
    max_r = int((img_area * 0.25) ** 0.5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=min_r * 2,
        param1=50, param2=35,
        minRadius=min_r, maxRadius=max_r)
    if circles is None:
        return np.zeros((h, w), dtype=np.uint8), None, None
    circles = np.round(circles[0]).astype(int)
    # 取最圆（半径最接近图像短边 1/8）的
    target_r = min(h, w) / 8
    circles = sorted(circles, key=lambda c: abs(c[2] - target_r))
    cx, cy, r = circles[0]
    # 构造椭圆（正圆）
    ellipse = ((float(cx), float(cy)), (float(r * 2), float(r * 2)), 0.0)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r + 5, 255, -1)
    return mask, (cx, cy), ellipse


_last_pass = 'FAILED'   # 上一次使用的 Pass 标签
_last_sam_masks = None  # 保留兼容旧代码引用，不再使用


def detect_blue_sticker(img_bgr):
    """
    检测蓝色圆形甬砼码贴纸（纯CV，无SAM）：
      Pass-1: HSV 标准范围 (H=88-135, S≥50, V≥50) + IoU 验证
      Pass-2: HSV 高亮低饱和 (S≥15, V≥200) — 褪色/过曝贴纸
      Pass-3: HoughCircles 纯形状检测 — 终极兜底
    返回: (sticker_mask, center, ellipse)；无贴纸时返回 (全零mask, None, None)
    """
    global _last_pass
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    def _try_hsv(scored):
        for best_score, sticker in scored:
            if best_score < 0.15:
                break
            img_area = img_bgr.shape[0] * img_bgr.shape[1]
            if cv2.contourArea(sticker) < img_area * 0.005 or len(sticker) < 5:
                continue
            ell_hsv = cv2.fitEllipse(sticker)
            iou = _ellipse_iou(sticker, ell_hsv, img_bgr.shape)
            if iou < 0.75:
                continue
            cx, cy = ell_hsv[0]
            r_hsv = max(ell_hsv[1]) / 2
            ell_grad = _refine_ellipse_by_gradient(img_bgr, cx, cy, r_hsv, ell_hsv)
            if ell_grad is not None:
                hsv_ratio  = min(ell_hsv[1]) / max(ell_hsv[1])
                grad_ratio = min(ell_grad[1]) / max(ell_grad[1])
                r_grad = max(ell_grad[1]) / 2
                size_ratio = r_grad / r_hsv
                # 梯度椭圆比 HSV 椭圆更不圆、或者明显偏大 → 回退用 HSV
                if grad_ratio < hsv_ratio - 0.01 or \
                   size_ratio > 1.15 or size_ratio < 0.8:
                    ell_grad = None
            ellipse = ell_grad if ell_grad is not None else ell_hsv
            center = (int(ellipse[0][0]), int(ellipse[0][1]))
            sticker_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
            cv2.ellipse(sticker_mask, ellipse, 255, -1)
            sticker_mask = cv2.dilate(sticker_mask, kernel, iterations=1)
            return sticker_mask, center, ellipse
        return None

    # ── Pass-1: HSV 标准范围 ──────────────────────────────────────
    scored, _ = _hsv_candidates(img_bgr, hsv, 88, 50, 50)
    result = _try_hsv(scored)
    if result:
        _last_pass = 'HSV'
        return result

    # ── Pass-2: HSV 高亮低饱和 ────────────────────────────────────
    scored, _ = _hsv_candidates(img_bgr, hsv, 86, 15, 200)
    result = _try_hsv(scored)
    if result:
        _last_pass = 'HSV-bright'
        return result

    # ── Pass-3: HoughCircles ──────────────────────────────────────
    result = _hough_fallback(img_bgr)
    _last_pass = 'Hough' if result[1] is not None else 'FAILED'
    return result
