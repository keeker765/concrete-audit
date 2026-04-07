"""
混凝土试块造假识别 Pipeline — 透视矫正
=======================================
椭圆→圆 Homography 迭代矫正
"""

import cv2
import numpy as np

from .sticker import detect_blue_sticker


def _compute_homography(ellipse):
    """计算椭圆→圆的透视变换矩阵 H"""
    import math
    (cx, cy), (ax1, ax2), angle = ellipse
    r1 = ax1 / 2
    r2 = ax2 / 2
    r_target = max(r1, r2)
    a = np.radians(angle)

    src = np.float32([
        [cx + math.cos(a) * r1, cy + math.sin(a) * r1],
        [cx - math.cos(a) * r1, cy - math.sin(a) * r1],
        [cx + math.cos(a + math.pi/2) * r2, cy + math.sin(a + math.pi/2) * r2],
        [cx - math.cos(a + math.pi/2) * r2, cy - math.sin(a + math.pi/2) * r2],
    ])
    dst = np.float32([
        [cx + math.cos(a) * r_target, cy + math.sin(a) * r_target],
        [cx - math.cos(a) * r_target, cy - math.sin(a) * r_target],
        [cx + math.cos(a + math.pi/2) * r_target, cy + math.sin(a + math.pi/2) * r_target],
        [cx - math.cos(a + math.pi/2) * r_target, cy - math.sin(a + math.pi/2) * r_target],
    ])

    return cv2.getPerspectiveTransform(src, dst)


def _apply_homography_rectify(img_bgr, ellipse):
    """单次椭圆→圆透视变换 (Homography, 8 DOF)，比仿射 (6 DOF) 更精确。
    取椭圆4个端点→圆的4个端点，用 getPerspectiveTransform 计算。
    注意: OpenCV fitEllipse 返回 (center, (width, height), angle)，
    width 沿 angle 方向，height 垂直于 angle 方向，不保证 width > height。"""
    (cx, cy), (ax1, ax2), angle = ellipse
    if max(ax1, ax2) == 0:
        return img_bgr, 1.0
    ratio = min(ax1, ax2) / max(ax1, ax2)
    if ratio > 0.99:
        return img_bgr, ratio

    H = _compute_homography(ellipse)
    h, w = img_bgr.shape[:2]
    return cv2.warpPerspective(img_bgr, H, (w, h)), ratio


def rectify_perspective(img_bgr, ellipse, max_iter=3):
    """
    迭代式椭圆→圆透视矫正 (Homography)。
    复合变换: 每轮计算增量 H，累积到 H_total，始终从原图变换（避免多次插值质量退化）。
    带 oscillation detection: 如果ratio下降则回退到最优结果。
    返回: (corrected_img, H_total) — H_total 可用于变换原始 mask，避免重新检测
    """
    if ellipse is None:
        return img_bgr, np.eye(3, dtype=np.float64)

    (_, _), (ma, mi), _ = ellipse
    if max(ma, mi) == 0:
        return img_bgr, np.eye(3, dtype=np.float64)
    init_ratio = min(ma, mi) / max(ma, mi)
    if init_ratio > 0.99:
        return img_bgr, np.eye(3, dtype=np.float64)

    print(f"    [透视矫正] ratio={init_ratio:.3f}, angle={ellipse[2]:.1f}°", end='')

    h, w = img_bgr.shape[:2]
    H_total = np.eye(3, dtype=np.float64)
    current_ell = ellipse
    best_H = H_total.copy()
    best_ratio = init_ratio

    for it in range(max_iter):
        H_inc = _compute_homography(current_ell)
        H_new = np.float64(H_inc) @ H_total
        # 始终从原图变换，避免多次插值退化
        corrected = cv2.warpPerspective(img_bgr, H_new, (w, h))
        _, _, ell_new = detect_blue_sticker(corrected)
        if ell_new is None:
            if it == 0:
                best_H = H_new
            break
        (_, _), (ma2, mi2), _ = ell_new
        if max(ma2, mi2) == 0:
            break
        new_ratio = min(ma2, mi2) / max(ma2, mi2)
        print(f" → r{it+1}={new_ratio:.3f}", end='')

        if new_ratio > best_ratio:
            best_H = H_new.copy()
            best_ratio = new_ratio
        elif new_ratio < best_ratio - 0.002:
            print(f"(↓revert)", end='')
            break

        if new_ratio > 0.99:
            break

        H_total = H_new
        current_ell = ell_new

    print()
    return cv2.warpPerspective(img_bgr, best_H, (w, h)), best_H
