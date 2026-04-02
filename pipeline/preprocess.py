"""预处理流水线：加载 → 缩放 → 贴纸检测 → ROI裁切 → 透视矫正 → CLAHE"""
import cv2
import numpy as np
import torch
from pathlib import Path

from .config import DEVICE, ROI_SCALE
from .sticker import detect_blue_sticker
from .roi import (crop_roi_around_sticker, crop_roi_square_border,
                  _detect_concrete_face, _find_face_by_edge_scan,
                  _find_face_by_dino)
from .rectify import rectify_perspective


def load_and_preprocess(img_path, max_size=1024, roi_mode='dino', rectify=False):
    """
    加载 → 缩放 → 贴纸检测 → ROI裁切 → 透视矫正(可选) → CLAHE
    roi_mode:
      'dino'    — Grounding DINO 检测混凝土块（推荐，最准确）
      'face'    — GrabCut 检测混凝土面矩形边界
      'sticker' — 贴纸半径×2.5 裁切（旧方法）
      'square'  — 以贴纸为中心取最大正方形，靠图像边界
    rectify: 是否执行透视矫正（默认关闭）
    Returns: (img_bgr, tensor, exclude_mask, center, sticker_found, meta)
    Note: NO qr_angle in return — QR is only for pairing
    """
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")

    orig_h, orig_w = img_bgr.shape[:2]
    h, w = orig_h, orig_w
    if max(h, w) > max_size:
        s = max_size / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * s), int(h * s)))

    # 全图贴纸检测
    _, center, ellipse = detect_blue_sticker(img_bgr)

    # 保存ROI裁切前的全图（用于生成对比图）
    img_pre_roi = img_bgr.copy()
    roi_bounds = None

    # ROI 裁切
    sam_mask = None
    refined_sticker_ell = None
    if roi_mode == 'dino':
        if center is not None and ellipse is not None:
            (_, _), (ma, mi), _ = ellipse
            sr = max(ma, mi) / 2

            x1, y1, x2, y2 = _find_face_by_dino(
                img_bgr, int(center[0]), int(center[1]), sr)
            sam_mask = None
            refined_sticker_ell = None

            roi_bounds = (x1, y1, x2, y2)
            (ecx, ecy), (ema, emi), eangle = ellipse
            img_roi = img_bgr[y1:y2, x1:x2].copy()
            center_roi = (int(ecx - x1), int(ecy - y1))
            ellipse_roi = ((ecx - x1, ecy - y1), (ema, emi), eangle)
        else:
            img_roi, center_roi, ellipse_roi = img_bgr, center, ellipse
    elif roi_mode == 'face':
        if center is not None and ellipse is not None:
            (_, _), (ma, mi), _ = ellipse
            sr = max(ma, mi) / 2

            # 纯CV边缘扫描找混凝土面，不依赖SAM predictor
            x1, y1, x2, y2 = _find_face_by_edge_scan(
                img_bgr, int(center[0]), int(center[1]), sr)
            sam_mask = None
            refined_sticker_ell = None

            roi_bounds = (x1, y1, x2, y2)
            (ecx, ecy), (ema, emi), eangle = ellipse
            img_roi = img_bgr[y1:y2, x1:x2].copy()
            center_roi = (int(ecx - x1), int(ecy - y1))
            ellipse_roi = ((ecx - x1, ecy - y1), (ema, emi), eangle)
        else:
            img_roi, center_roi, ellipse_roi = img_bgr, center, ellipse
    elif roi_mode == 'square':
        img_roi, center_roi, ellipse_roi = crop_roi_square_border(
            img_bgr, center, ellipse)
    else:
        img_roi, center_roi, ellipse_roi = crop_roi_around_sticker(
            img_bgr, center, ellipse, scale=ROI_SCALE)

    # 计算透视矫正元信息
    persp_ratio = 1.0
    persp_applied = False
    if ellipse_roi is not None:
        (_, _), (ma, mi), _ = ellipse_roi
        if max(ma, mi) > 0:
            persp_ratio = min(ma, mi) / max(ma, mi)
            persp_applied = rectify and persp_ratio <= 0.99

    # 透视矫正（仅在 rectify=True 时执行）
    if rectify:
        img_rect, H_total = rectify_perspective(img_roi, ellipse_roi)
    else:
        img_rect = img_roi
        H_total = np.eye(3, dtype=np.float64)

    # 通过变换原始 mask 获取排除区域（避免在变形图上重新检测导致梯度找错边缘）
    # 使用 SAM 精化后的贴纸椭圆（如果有的话）以获得更准确的排除区域
    ellipse_for_mask = ellipse_roi
    if refined_sticker_ell is not None and ellipse_roi is not None:
        (recx, recy), (rema, remi), reangle = refined_sticker_ell
        x1_roi, y1_roi = roi_bounds[0], roi_bounds[1]
        ellipse_for_mask = ((recx - x1_roi, recy - y1_roi), (rema, remi), reangle)
    if ellipse_for_mask is not None:
        roi_h, roi_w = img_roi.shape[:2]
        pre_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        cv2.ellipse(pre_mask, ellipse_for_mask, 255, -1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        pre_mask = cv2.dilate(pre_mask, kernel, iterations=1)
        exclude_mask = cv2.warpPerspective(pre_mask, H_total,
                                           (img_rect.shape[1], img_rect.shape[0]))
    else:
        exclude_mask = np.zeros(img_rect.shape[:2], dtype=np.uint8)

    # 计算矫正后的长短轴比（从变换后的 mask 轮廓拟合椭圆）
    persp_ratio_after = 1.0
    contours_rect, _ = cv2.findContours(exclude_mask, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
    if contours_rect:
        c_rect = max(contours_rect, key=cv2.contourArea)
        if len(c_rect) >= 5:
            ell_rect = cv2.fitEllipse(c_rect)
            (_, _), (ma2, mi2), _ = ell_rect
            if max(ma2, mi2) > 0:
                persp_ratio_after = min(ma2, mi2) / max(ma2, mi2)

    # 处理 SAM mask 到与 ROI+透视矫正后的图像匹配的坐标系
    sam_mask_processed = None
    if sam_mask is not None and roi_bounds is not None:
        x1, y1, x2, y2 = roi_bounds
        sam_crop = sam_mask[y1:y2, x1:x2].copy()
        sam_mask_processed = cv2.warpPerspective(
            sam_crop.astype(np.uint8), H_total,
            (img_rect.shape[1], img_rect.shape[0]))

    # CLAHE
    gray = cv2.cvtColor(img_rect, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced_3ch = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    tensor = torch.from_numpy(enhanced_3ch).permute(2, 0, 1).float() / 255.0

    roi_h, roi_w = img_rect.shape[:2]
    persp_str = (f"persp={persp_ratio:.3f}→{persp_ratio_after:.3f}"
                 if persp_applied else "persp=N/A")
    print(f"  Preprocess: {Path(img_path).name} → "
          f"sticker={'yes' if center else 'no'}, "
          f"ROI={roi_w}×{roi_h}({roi_mode}), {persp_str}")

    meta = {
        'orig_size': f"{orig_w}x{orig_h}",
        'roi_size': f"{roi_w}x{roi_h}",
        'roi_mode': roi_mode,
        'persp_ratio': persp_ratio,
        'persp_ratio_after': persp_ratio_after,
        'persp_applied': persp_applied,
        'img_pre_roi': img_pre_roi,
        'roi_bounds': roi_bounds,
        'center_orig': center,
        'ellipse_orig': ellipse,
        'sam_mask': sam_mask,
        'sam_mask_processed': sam_mask_processed,
        'refined_sticker_ell': refined_sticker_ell,
    }

    return img_rect, tensor, exclude_mask, center_roi, center is not None, meta
