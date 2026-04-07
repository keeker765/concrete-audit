"""
混凝土试块造假识别 Pipeline — ROI 裁切
=======================================
圆形/正方形/SAM面 三种 ROI 裁切 + debug 可视化
"""

import cv2
import numpy as np

from .config import _get_sam_predictor, _get_gdino, DEVICE


# Grounding DINO 检测参数
_DINO_PROMPT = "concrete block."
_DINO_SCORE_THRESH = 0.15
_DINO_NMS_IOU = 0.5
_DINO_MAX_SIZE = 1024
_DINO_MIN_AREA_RATIO = 0.08  # 检测框面积 < 图像8% → fallback


def _find_face_by_dino(img_bgr, cx, cy, sticker_r):
    """
    用 Grounding DINO 检测混凝土面矩形边界。
    NMS 去重后按 score 选最佳框。
    返回 (x1, y1, x2, y2)
    """
    import torch
    from PIL import Image
    from torchvision.ops import nms as tv_nms

    h, w = img_bgr.shape[:2]
    processor, model = _get_gdino()

    # 缩放到工作尺寸
    scale = 1.0
    if max(h, w) > _DINO_MAX_SIZE:
        scale = _DINO_MAX_SIZE / max(h, w)
        resized = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    else:
        resized = img_bgr
    hr, wr = resized.shape[:2]

    img_pil = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    inputs = processor(images=img_pil, text=_DINO_PROMPT,
                       return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        text_threshold=_DINO_SCORE_THRESH,
        target_sizes=[(hr, wr)])
    r = results[0]

    inv = 1.0 / scale
    img_area = w * h

    if len(r["boxes"]) == 0:
        # fallback：贴纸中心 ± 4×r
        fb = int(sticker_r * 4)
        x1 = max(0, cx - fb); y1 = max(0, cy - fb)
        x2 = min(w, cx + fb); y2 = min(h, cy + fb)
        print(f"    [face-dino] no detection → fallback bbox=({x1},{y1},{x2},{y2})")
        return x1, y1, x2, y2

    # NMS
    nms_keep = tv_nms(r["boxes"], r["scores"], iou_threshold=_DINO_NMS_IOU)
    kept_boxes = r["boxes"][nms_keep]
    kept_scores = r["scores"][nms_keep]

    best_nms = int(kept_scores.argmax())
    score = float(kept_scores[best_nms])
    x1, y1, x2, y2 = [int(v * inv) for v in kept_boxes[best_nms].tolist()]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    box_area = (x2 - x1) * (y2 - y1)
    if box_area < img_area * _DINO_MIN_AREA_RATIO:
        # 检测框太小（只框到了贴纸），fallback
        fb = int(sticker_r * 4)
        x1 = max(0, cx - fb); y1 = max(0, cy - fb)
        x2 = min(w, cx + fb); y2 = min(h, cy + fb)
        print(f"    [face-dino] area too small ({box_area/img_area*100:.0f}%) → fallback")
    else:
        print(f"    [face-dino] bbox=({x1},{y1},{x2},{y2}) score={score:.2f} "
              f"area={box_area/img_area*100:.0f}%")

    return int(x1), int(y1), int(x2), int(y2)


def crop_roi_dino(img_bgr, center, ellipse):
    """用 Grounding DINO 检测混凝土块并裁切 ROI。
    返回: (cropped_img, new_center, new_ellipse)"""
    if center is None or ellipse is None:
        return img_bgr, center, ellipse

    cx, cy = center
    (ecx, ecy), (ma, mi), angle = ellipse
    sticker_r = max(ma, mi) / 2.0

    x1, y1, x2, y2 = _find_face_by_dino(img_bgr, cx, cy, int(sticker_r))

    cropped = img_bgr[y1:y2, x1:x2].copy()
    new_center = (int(ecx - x1), int(ecy - y1))
    new_ellipse = ((ecx - x1, ecy - y1), (ma, mi), angle)
    return cropped, new_center, new_ellipse


def _find_face_by_edge_scan(img_bgr, cx, cy, sticker_r):
    """
    GrabCut 分割混凝土面。

    原理：
    1. 用贴纸位置估算 GrabCut 初始矩形 hint（贴纸中心 ± 4×r）
    2. GrabCut 迭代分离前景（混凝土）与背景（压机/桌面）
    3. 找包含贴纸中心的最大前景轮廓，取外接矩形
    4. 若 GrabCut 失败或结果异常，fallback 到固定比例裁切

    纯 OpenCV，~20-40ms，不依赖任何模型。
    返回 (x1, y1, x2, y2)
    """
    h, w = img_bgr.shape[:2]
    sr = max(int(sticker_r), 10)
    img_area = h * w

    # GrabCut rect hint：贴纸中心 ± face_half，贴纸通常占面边长 1/5~1/4
    face_half = max(int(sr * 3.8), 80)
    rx1 = max(2, cx - face_half)
    ry1 = max(2, cy - face_half)
    rx2 = min(w - 2, cx + face_half)
    ry2 = min(h - 2, cy + face_half)
    rect_w = rx2 - rx1
    rect_h = ry2 - ry1

    x1, y1, x2, y2 = rx1, ry1, rx2, ry2   # 默认就用估算矩形

    if rect_w > 20 and rect_h > 20:
        try:
            # 下采样到 512px 加速 GrabCut
            scale = min(512.0 / max(h, w), 1.0)
            if scale < 1.0:
                small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
                srx1 = int(rx1 * scale); sry1 = int(ry1 * scale)
                srw  = int(rect_w * scale); srh = int(rect_h * scale)
                scx  = int(cx * scale);     scy = int(cy * scale)
                sh, sw = small.shape[:2]
            else:
                small = img_bgr
                srx1, sry1, srw, srh = rx1, ry1, rect_w, rect_h
                scx, scy = cx, cy
                sh, sw = h, w

            bgd = np.zeros((1, 65), np.float64)
            fgd = np.zeros((1, 65), np.float64)
            gc_mask = np.zeros((sh, sw), np.uint8)
            cv2.grabCut(small, gc_mask, (srx1, sry1, srw, srh),
                        bgd, fgd, 2, cv2.GC_INIT_WITH_RECT)
            fg = np.where((gc_mask == 1) | (gc_mask == 3), 255, 0).astype(np.uint8)

            # 形态学闭运算，填充贴纸孔洞
            k = max(int(sr * scale) | 1, 9)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=1)

            cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                s_area = sh * sw
                if area < s_area * 0.04 or area > s_area * 0.92:
                    continue
                if cv2.pointPolygonTest(cnt, (float(scx), float(scy)), False) >= 0:
                    bx, by, bw_c, bh_c = cv2.boundingRect(cnt)
                    # 映射回原始坐标
                    inv = 1.0 / scale if scale < 1.0 else 1.0
                    x1 = int(bx * inv); y1 = int(by * inv)
                    x2 = int((bx + bw_c) * inv); y2 = int((by + bh_c) * inv)
                    break
        except Exception:
            pass   # GrabCut 失败用 fallback

    # 合理性校验：边长至少 3×r
    fb = int(sr * 4)
    if (x2 - x1) < sr * 3:
        x1, x2 = max(0, cx - fb), min(w, cx + fb)
    if (y2 - y1) < sr * 3:
        y1, y2 = max(0, cy - fb), min(h, cy + fb)

    print(f"    [face-grabcut] bbox=({x1},{y1},{x2},{y2}) sr={sr}")
    return int(x1), int(y1), int(x2), int(y2)


def crop_roi_around_sticker(img_bgr, center, ellipse, scale=2.5):
    """
    以贴纸为中心裁切 ROI，消除背景（压力机、模具、桌面）。
    scale: ROI 半径 = 贴纸半径 × scale
    返回: (cropped_img, new_center, new_ellipse)
    """
    if center is None or ellipse is None:
        return img_bgr, center, ellipse

    h, w = img_bgr.shape[:2]
    (cx, cy), (ma, mi), angle = ellipse
    radius = max(ma, mi) / 2.0
    roi_half = int(radius * scale)

    x1 = max(0, int(cx - roi_half))
    y1 = max(0, int(cy - roi_half))
    x2 = min(w, int(cx + roi_half))
    y2 = min(h, int(cy + roi_half))

    cropped = img_bgr[y1:y2, x1:x2].copy()
    new_center = (int(cx - x1), int(cy - y1))
    new_ellipse = ((cx - x1, cy - y1), (ma, mi), angle)
    return cropped, new_center, new_ellipse


def crop_roi_square_border(img_bgr, center, ellipse):
    """
    正方形边框裁切法：找混凝土面的最大正方形区域。
    思路:
      1. 以贴纸中心为锚点
      2. 向四周扩展到图像边界，取最大正方形
      3. 不限制为贴纸半径的倍数，尽量保留更多混凝土面
    如果贴纸偏角上，正方形会靠边但包含更多有效区域。
    返回: (cropped_img, new_center, new_ellipse)
    """
    if center is None or ellipse is None:
        return img_bgr, center, ellipse

    h, w = img_bgr.shape[:2]
    cx, cy = center
    (ecx, ecy), (ma, mi), angle = ellipse

    # 贴纸到四边的距离
    dist_left = cx
    dist_right = w - cx
    dist_top = cy
    dist_bottom = h - cy

    # 能取到的最大正方形半边长 = 四方向最小距离
    half = min(dist_left, dist_right, dist_top, dist_bottom)

    # 但至少要比贴纸大一些（贴纸半径的 1.5 倍）
    sticker_r = max(ma, mi) / 2.0
    half = max(half, int(sticker_r * 1.5))

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)

    # 强制正方形（取短边）
    side = min(x2 - x1, y2 - y1)
    # 重新居中
    mid_x = (x1 + x2) // 2
    mid_y = (y1 + y2) // 2
    x1 = max(0, mid_x - side // 2)
    y1 = max(0, mid_y - side // 2)
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)

    cropped = img_bgr[y1:y2, x1:x2].copy()
    new_center = (int(ecx - x1), int(ecy - y1))
    new_ellipse = ((ecx - x1, ecy - y1), (ma, mi), angle)
    return cropped, new_center, new_ellipse


def _detect_concrete_face(img_bgr, cx, cy, sticker_r):
    """使用 SAM 分割模型检测混凝土面的矩形边界。
    1. 先用混凝土纹理（低饱和度灰色区域）找到面中心
    2. 用 point+box 组合 prompt + 形态学 opening 清理
    返回: (x1, y1, x2, y2, sam_mask, refined_sticker_ell)"""
    h, w = img_bgr.shape[:2]
    predictor = _get_sam_predictor()

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)

    # === 混凝土纹理检测：找面中心 ===
    # 混凝土特征: 低饱和度(S<50)、中等亮度(V=40-220)、灰色
    hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    concrete_mask = ((hsv_img[:, :, 1] < 50) &
                     (hsv_img[:, :, 2] > 40) &
                     (hsv_img[:, :, 2] < 220)).astype(np.uint8) * 255
    # 形态学闭合填补小孔 + 开运算去噪
    ck = max(7, int(sticker_r * 0.2)) | 1
    c_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
    concrete_mask = cv2.morphologyEx(concrete_mask, cv2.MORPH_CLOSE, c_kernel)
    concrete_mask = cv2.morphologyEx(concrete_mask, cv2.MORPH_OPEN, c_kernel)

    # 找包含贴纸中心的混凝土连通区域
    n_cc, cc_labels = cv2.connectedComponents(concrete_mask)
    cy_s, cx_s = min(cy, h-1), min(cx, w-1)
    lbl_at_stk = cc_labels[cy_s, cx_s]
    face_cx, face_cy = cx, cy  # 默认用贴纸中心

    if lbl_at_stk > 0:
        region = (cc_labels == lbl_at_stk)
        region_area = region.sum()
        # 仅当混凝土区域足够大（>2×贴纸面积）时采用其质心
        if region_area > np.pi * sticker_r**2 * 2:
            ys_r, xs_r = np.where(region)
            face_cx, face_cy = int(xs_r.mean()), int(ys_r.mean())
    else:
        # 贴纸中心不在混凝土上，找最大混凝土区域
        best_area, best_lbl = 0, 0
        for i in range(1, n_cc):
            a = int((cc_labels == i).sum())
            if a > best_area:
                best_area, best_lbl = a, i
        if best_area > np.pi * sticker_r**2 * 2:
            ys_r, xs_r = np.where(cc_labels == best_lbl)
            face_cx, face_cy = int(xs_r.mean()), int(ys_r.mean())

    print(f"    [SAM] prompt: sticker=({cx},{cy}) → face_center=({face_cx},{face_cy})")

    # 组合 prompt: point（面中心为前景点）+ box（预期面边界）
    box_half = int(sticker_r * 3.0)
    box = np.array([max(0, face_cx - box_half), max(0, face_cy - box_half),
                    min(w, face_cx + box_half), min(h, face_cy + box_half)])
    masks, scores, _ = predictor.predict(
        point_coords=np.array([[face_cx, face_cy]]),
        point_labels=np.array([1]),
        box=box,
        multimask_output=True)

    # 从候选 mask 中选最佳：面积 > 1.3× 贴纸直径 且最方正
    sticker_area = np.pi * sticker_r ** 2
    best_mask = None
    best_quality = -1

    for mask, score in zip(masks, scores):
        area = mask.sum()
        ratio = np.sqrt(area / sticker_area)
        if ratio < 1.3 or ratio > 6.0:
            continue
        ys, xs = np.where(mask)
        bw, bh = xs.max() - xs.min(), ys.max() - ys.min()
        ar = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0
        quality = float(score) * ar * ar
        if quality > best_quality:
            best_quality = quality
            best_mask = mask

    if best_mask is None:
        best_mask = masks[np.argmax(scores)]

    # 形态学 opening 清理机器/网格与面的连接
    raw_mask = (best_mask > 0).astype(np.uint8) * 255
    ksz = max(5, int(sticker_r * 0.15)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    opened = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)

    # 颜色过滤：去除高饱和度区域（红/绿标记等非混凝土色）
    # hsv_img 已在上方混凝土纹理检测中计算
    sat = hsv_img[:, :, 1]
    hue = hsv_img[:, :, 0]
    # 蓝色色相范围 90-135（对应OpenCV H=0-180），保留蓝色贴纸
    is_blue = (hue >= 90) & (hue <= 135) & (sat > 50)
    high_sat_nonblue = (sat > 100) & ~is_blue  # 红/绿/黄等高彩度标记
    # 仅去除mask内的大块高饱和区域（面积>贴纸面积×0.3），忽略小噪点
    color_mask = np.zeros_like(opened)
    color_mask[high_sat_nonblue] = 255
    color_mask = cv2.bitwise_and(color_mask, opened)  # 仅mask内
    n_cc, cc_labels = cv2.connectedComponents(color_mask)
    min_blob = int(sticker_area * 0.3)
    for i in range(1, n_cc):
        blob_area = int((cc_labels == i).sum())
        if blob_area > min_blob:
            opened[cc_labels == i] = 0

    # 保留包含贴纸中心的连通区域
    n_labels, labels = cv2.connectedComponents(opened)
    cy_safe = min(cy, labels.shape[0] - 1)
    cx_safe = min(cx, labels.shape[1] - 1)
    lbl = labels[cy_safe, cx_safe]
    if lbl > 0:
        cleaned = (labels == lbl).astype(np.uint8) * 255
    else:
        cleaned = opened

    # 如果 opening 后面积太小（< 4× 贴纸面积），回退到原始 SAM mask
    if cleaned.sum() / 255 < sticker_area * 4:
        cleaned = raw_mask

    # minAreaRect 补全不规则边缘（填补被 opening 侵蚀的角/边）
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        box_pts = cv2.boxPoints(rect).astype(np.int32)
        filled = cleaned.copy()
        cv2.fillPoly(filled, [box_pts], 255)
        ys, xs = np.where(filled > 0)
    else:
        ys, xs = np.where(cleaned > 0)

    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    margin = max(5, int(sticker_r * 0.05))
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)

    # 保证贴纸中心在 bbox 内
    pad = 10
    x1 = min(x1, max(0, cx - pad))
    y1 = min(y1, max(0, cy - pad))
    x2 = max(x2, min(w, cx + pad))
    y2 = max(y2, min(h, cy + pad))

    # ROI 大小上限：防止湿态图把模框/背景截入
    # 最大半径 = sticker_r × MAX_FACE_HALF（默认 3.5）
    MAX_FACE_HALF = 3.5
    max_half = int(sticker_r * MAX_FACE_HALF)
    x1 = max(x1, cx - max_half)
    y1 = max(y1, cy - max_half)
    x2 = min(x2, cx + max_half)
    y2 = min(y2, cy + max_half)
    # 再次确保贴纸中心不被裁出
    x1 = min(x1, max(0, cx - pad))
    y1 = min(y1, max(0, cy - pad))
    x2 = max(x2, min(w, cx + pad))
    y2 = max(y2, min(h, cy + pad))

    # 保存清理后的 mask 用于 debug（从 cleaned 轮廓生成，不含 fillPoly 补全）
    debug_mask = cleaned

    # === SAM-based sticker boundary refinement ===
    # Point-only prompt gives 3 granularity levels: QR code / sticker / face
    # The sticker-level mask is cleaner than HSV (no protrusions from text/concrete)
    refined_sticker_ell = None
    stk_masks, stk_scores, _ = predictor.predict(
        point_coords=np.array([[cx, cy]]),
        point_labels=np.array([1]),
        multimask_output=True)

    best_stk_q = -1
    for smk, ssc in zip(stk_masks, stk_scores):
        mask_area = int(smk.sum())
        r_area = np.sqrt(mask_area / np.pi)
        ratio = r_area / sticker_r if sticker_r > 0 else 0
        ys_s, xs_s = np.where(smk)
        if len(ys_s) < 10:
            continue
        bw_s = xs_s.max() - xs_s.min()
        bh_s = ys_s.max() - ys_s.min()
        ar_s = min(bw_s, bh_s) / max(bw_s, bh_s) if max(bw_s, bh_s) > 0 else 0
        # Filter: sticker-sized (0.5-1.4× HSV radius), circular (ar≥0.85)
        # 上限从 2.0 收紧到 1.4，防止 SAM 返回含文字的大标签区域
        if ar_s < 0.85 or ratio < 0.5 or ratio > 1.4:
            continue
        q = float(ssc) * ar_s * ar_s
        if q > best_stk_q:
            best_stk_q = q
            mask_u8_s = (smk > 0).astype(np.uint8) * 255
            conts_s, _ = cv2.findContours(mask_u8_s, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if conts_s:
                c_s = max(conts_s, key=cv2.contourArea)
                if len(c_s) >= 5:
                    refined_sticker_ell = cv2.fitEllipse(c_s)

    return x1, y1, x2, y2, debug_mask, refined_sticker_ell


def _generate_detection_debug(img_bgr, cx, cy, sticker_r, roi_bounds, sam_mask=None,
                              refined_sticker_ell=None, hsv_ellipse=None):
    """生成中间过程诊断图：
    1. 标签检测边缘图（HSV mask + 梯度边缘）
    2. 方框检测诊断图（SAM 分割 mask + 检测到的边界线）
    返回: (sticker_debug_img, face_debug_img) — 都是 BGR 图
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # === 标签检测中间图 ===
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv, (90, 50, 50), (135, 255, 255))
    blur_s = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 1.5)
    sx = cv2.Sobel(blur_s, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(blur_s, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sx**2 + sy**2)
    grad_norm = np.clip(grad_mag / grad_mag.max() * 255, 0, 255).astype(np.uint8)
    sticker_vis = np.zeros((h, w, 3), dtype=np.uint8)
    sticker_vis[:, :, 0] = (gray * 0.3).astype(np.uint8)
    sticker_vis[:, :, 1] = grad_norm
    sticker_vis[:, :, 2] = hsv_mask
    # 画贴纸椭圆（优先SAM精化，否则用HSV椭圆，最后用圆形）
    draw_ell = refined_sticker_ell or hsv_ellipse
    if draw_ell is not None:
        cv2.ellipse(sticker_vis, draw_ell, (255, 255, 0), 2)
    else:
        cv2.circle(sticker_vis, (cx, cy), int(sticker_r), (255, 255, 0), 2)

    # === 面检测诊断图（SAM mask + 补全矩形 + 外接矩形）===
    face_vis = img_bgr.copy()
    if sam_mask is not None:
        # 半透明绿色覆盖 SAM mask 区域
        green_overlay = np.zeros_like(face_vis)
        green_overlay[:, :, 1] = 255
        mask_bool = sam_mask > 0 if sam_mask.dtype != bool else sam_mask
        face_vis[mask_bool] = cv2.addWeighted(
            face_vis[mask_bool], 0.6, green_overlay[mask_bool], 0.4, 0)
        # 描出 SAM mask 边缘轮廓（绿色）
        mask_u8 = (mask_bool.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(face_vis, contours, -1, (0, 255, 0), 2)
        # 画 minAreaRect 补全矩形（黄色虚线）
        if contours:
            c = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(c)
            box_pts = cv2.boxPoints(rect).astype(np.int32)
            cv2.drawContours(face_vis, [box_pts], 0, (0, 255, 255), 2)
    # 画外接矩形（白色）
    if roi_bounds is not None:
        x1, y1, x2, y2 = roi_bounds[:4]
        cv2.rectangle(face_vis, (x1, y1), (x2, y2), (255, 255, 255), 2)
    # 画贴纸检测边界（优先SAM精化，否则HSV）
    if refined_sticker_ell is not None:
        cv2.ellipse(face_vis, refined_sticker_ell, (255, 0, 0), 2)
    elif hsv_ellipse is not None:
        cv2.ellipse(face_vis, hsv_ellipse, (255, 0, 0), 2)
    else:
        cv2.circle(face_vis, (cx, cy), int(sticker_r), (255, 0, 0), 2)

    return sticker_vis, face_vis


def crop_roi_concrete_face(img_bgr, center, ellipse):
    """
    混凝土面矩形ROI裁切：使用 SAM 检测混凝土块面的边界，裁切到该矩形。
    返回: (cropped_img, new_center, new_ellipse)
    """
    if center is None or ellipse is None:
        return img_bgr, center, ellipse

    h, w = img_bgr.shape[:2]
    cx, cy = center
    (ecx, ecy), (ma, mi), angle = ellipse
    sticker_r = max(ma, mi) / 2.0

    result = _detect_concrete_face(img_bgr, cx, cy, int(sticker_r))
    x1, y1, x2, y2 = result[0], result[1], result[2], result[3]

    cropped = img_bgr[y1:y2, x1:x2].copy()
    new_center = (int(ecx - x1), int(ecy - y1))
    new_ellipse = ((ecx - x1, ecy - y1), (ma, mi), angle)
    return cropped, new_center, new_ellipse
