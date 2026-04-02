"""
混凝土试块旋转对齐模块
=====================
新策略：基于矩形外形摆正 + 贴纸象限匹配（不依赖 QR 角度）

流程:
  1. 从 SAM mask 获取 minAreaRect → 矩形旋转角度
  2. 旋转图像使矩形边缘与轴对齐（量化到最近 90°）
  3. 根据贴纸在图像中的相对位置（象限）确定 0/90/180/270 旋转
"""
import cv2
import numpy as np
import torch
from .config import rotate_cv2


def detect_rect_angle(sam_mask):
    """从 SAM mask 检测混凝土面矩形的旋转角度。
    使用 minAreaRect 拟合，返回需要旋转的角度使矩形轴对齐。
    Returns: angle_deg (float) — 旋转角度，使矩形边与坐标轴平行
    """
    if sam_mask is None:
        return 0.0
    mask_u8 = sam_mask if sam_mask.dtype == np.uint8 else sam_mask.astype(np.uint8)
    if mask_u8.max() == 0:
        return 0.0
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    # rect = ((cx, cy), (w, h), angle)
    # OpenCV minAreaRect angle: [-90, 0)
    # We want to rotate so edges align with axes
    angle = rect[2]
    w, h = rect[1]
    # If width < height, the "long side" is vertical, angle refers to the shorter side
    # We want the smallest rotation to align
    if w < h:
        angle = angle + 90
    # Now angle is the rotation of the long side from horizontal
    # We want to counter-rotate by this angle
    return -angle


def get_sticker_quadrant(img_h, img_w, cx, cy):
    """判断贴纸中心在图像中的象限。

    象限定义（图像坐标系，Y轴向下）:
      Q0 = 左上 (x < 0.5, y < 0.5)
      Q1 = 右上 (x >= 0.5, y < 0.5)
      Q2 = 右下 (x >= 0.5, y >= 0.5)
      Q3 = 左下 (x < 0.5, y >= 0.5)

    Returns: int (0-3)
    """
    rx = cx / img_w if img_w > 0 else 0.5
    ry = cy / img_h if img_h > 0 else 0.5
    if rx < 0.5:
        return 0 if ry < 0.5 else 3  # 左上 or 左下
    else:
        return 1 if ry < 0.5 else 2  # 右上 or 右下


def quadrant_rotation(q_wet, q_dry):
    """根据湿态/干态贴纸象限差异计算旋转量。

    旋转量 = (q_dry - q_wet) mod 4 × 90°
    即：需要将湿态图旋转多少个 90° 才能使贴纸象限与干态一致。

    Returns: rot_k (int, 0-3), rot_deg (float)
    """
    rot_k = (q_dry - q_wet) % 4
    rot_deg = rot_k * 90.0
    return rot_k, rot_deg


def rectify_to_axis(img_bgr, sam_mask, sticker_center, sticker_ellipse):
    """将图像旋转到矩形边缘与轴对齐。

    1. 从 SAM mask 计算矩形角度
    2. 旋转图像（量化到最近的 90° 以避免插值损失）
    3. 同步变换贴纸中心和椭圆

    Args:
        img_bgr: BGR 图像
        sam_mask: SAM 分割 mask（可以为 None）
        sticker_center: (cx, cy) 贴纸中心
        sticker_ellipse: ((cx,cy), (ma,mi), angle) 贴纸椭圆

    Returns: (img_rotated, new_center, new_ellipse, rot_k)
    """
    angle = detect_rect_angle(sam_mask)
    # 量化到最近的 90°
    rot_k = round(angle / 90) % 4
    if rot_k == 0:
        return img_bgr, sticker_center, sticker_ellipse, 0

    img_rot = rotate_cv2(img_bgr, rot_k)
    h, w = img_bgr.shape[:2]

    # 变换贴纸中心
    if sticker_center is not None:
        cx, cy = sticker_center
        if rot_k == 1:  # 90° CCW
            new_cx, new_cy = cy, w - 1 - cx
        elif rot_k == 2:  # 180°
            new_cx, new_cy = w - 1 - cx, h - 1 - cy
        elif rot_k == 3:  # 270° CCW (90° CW)
            new_cx, new_cy = h - 1 - cy, cx
        else:
            new_cx, new_cy = cx, cy
        new_center = (int(new_cx), int(new_cy))
    else:
        new_center = None

    # 变换椭圆
    if sticker_ellipse is not None:
        (ecx, ecy), (ma, mi), eangle = sticker_ellipse
        if rot_k == 1:
            new_ecx, new_ecy = ecy, w - 1 - ecx
            new_angle = (eangle + 90) % 360
        elif rot_k == 2:
            new_ecx, new_ecy = w - 1 - ecx, h - 1 - ecy
            new_angle = (eangle + 180) % 360
        elif rot_k == 3:
            new_ecx, new_ecy = h - 1 - ecy, ecx
            new_angle = (eangle + 270) % 360
        else:
            new_ecx, new_ecy, new_angle = ecx, ecy, eangle
        new_ellipse = ((new_ecx, new_ecy), (ma, mi), new_angle)
    else:
        new_ellipse = None

    return img_rot, new_center, new_ellipse, rot_k


def align_pair(img_wet, img_dry, center_wet, center_dry,
               mask_wet=None, mask_dry=None,
               ellipse_wet=None, ellipse_dry=None,
               sam_mask_wet=None, sam_mask_dry=None):
    """完整的配对对齐流程。

    1. 分别矩形摆正（用 SAM mask）
    2. 计算贴纸象限
    3. 根据象限差异旋转湿态图
    4. 如果象限不可靠（贴纸在中心），回退到 4 候选暴力搜索

    Returns: dict with alignment results
    """
    # Step 1: 矩形摆正
    img_wet_r, center_wet_r, ell_wet_r, rk_wet = rectify_to_axis(
        img_wet, sam_mask_wet, center_wet, ellipse_wet)
    img_dry_r, center_dry_r, ell_dry_r, rk_dry = rectify_to_axis(
        img_dry, sam_mask_dry, center_dry, ellipse_dry)

    # 同步旋转 mask
    mask_wet_r = rotate_cv2(mask_wet, rk_wet) if mask_wet is not None and rk_wet > 0 else mask_wet
    mask_dry_r = rotate_cv2(mask_dry, rk_dry) if mask_dry is not None and rk_dry > 0 else mask_dry

    h_w, w_w = img_wet_r.shape[:2]
    h_d, w_d = img_dry_r.shape[:2]

    # Step 2: 始终使用 brute4 尝试全部 4 个旋转方向
    # 象限匹配在 DINO 裁切尺寸差异大时不可靠（湿态框大、干态框紧），
    # brute4 每个方向多花 ~0.1s 但永远不会因旋转判错而漏掉正确匹配
    method = 'brute4'
    candidates = [(k, k * 90.0) for k in range(4)]

    # 如果象限可靠，把预测方向排到最前（优先尝试，可提前终止）
    if center_wet_r is not None and center_dry_r is not None:
        q_wet = get_sticker_quadrant(h_w, w_w, center_wet_r[0], center_wet_r[1])
        q_dry = get_sticker_quadrant(h_d, w_d, center_dry_r[0], center_dry_r[1])
        rot_k_pred, _ = quadrant_rotation(q_wet, q_dry)
        # 把预测方向排最前
        candidates = [(rot_k_pred, rot_k_pred * 90.0)] + \
                     [(k, k * 90.0) for k in range(4) if k != rot_k_pred]

    return {
        'img_wet': img_wet_r,
        'img_dry': img_dry_r,
        'mask_wet': mask_wet_r,
        'mask_dry': mask_dry_r,
        'center_wet': center_wet_r,
        'center_dry': center_dry_r,
        'candidates': candidates,
        'method': method,
        'rect_rot_wet': rk_wet * 90,
        'rect_rot_dry': rk_dry * 90,
    }
