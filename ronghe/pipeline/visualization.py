"""匹配可视化与热力图"""
import cv2
import numpy as np
from .config import _async_save


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
        green_overlay = np.zeros_like(face_vis)
        green_overlay[:, :, 1] = 255
        mask_bool = sam_mask > 0 if sam_mask.dtype != bool else sam_mask
        face_vis[mask_bool] = cv2.addWeighted(
            face_vis[mask_bool], 0.6, green_overlay[mask_bool], 0.4, 0)
        mask_u8 = (mask_bool.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(face_vis, contours, -1, (0, 255, 0), 2)
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


def draw_matches_vis(img0_bgr, img1_bgr, kpts0, kpts1, matches, inliers_mask,
                     mask0, mask1, title, save_path, rot_deg=0,
                     meta0=None, meta1=None):
    """匹配可视化：绿线=内点，红线=外点，蓝色=贴纸区域，标注旋转+透视+QR"""
    img0_rgb = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2RGB)
    img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)

    def _ellipse_from_mask(mask):
        """从 mask 轮廓拟合椭圆，返回 ellipse 或 None"""
        contours, _ = cv2.findContours(
            mask if mask.dtype == np.uint8 else mask.astype(np.uint8),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        if len(c) >= 5:
            return cv2.fitEllipse(c)
        return None

    mask0_vis = mask0
    mask1_vis = mask1
    ell0_vis = _ellipse_from_mask(mask0)
    ell1_vis = _ellipse_from_mask(mask1)

    for img, mask, ell in [(img0_rgb, mask0_vis, ell0_vis), (img1_rgb, mask1_vis, ell1_vis)]:
        if mask.max() > 0:
            blue = np.zeros_like(img)
            blue[:, :, 2] = 255
            ov = mask > 0
            img[ov] = cv2.addWeighted(img[ov], 0.6, blue[ov], 0.4, 0)
        if ell is not None:
            cv2.ellipse(img, ell, (0, 0, 180), 2)

    h0, w0 = img0_rgb.shape[:2]
    h1, w1 = img1_rgb.shape[:2]
    gap = 10
    h_max = max(h0, h1)
    canvas = np.zeros((h_max, w0 + w1 + gap, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0_rgb
    canvas[:h1, w0 + gap:] = img1_rgb

    canvas_bgr = np.zeros((h_max, w0 + w1 + gap, 3), dtype=np.uint8)
    canvas_bgr[:h0, :w0] = cv2.cvtColor(img0_rgb, cv2.COLOR_RGB2BGR)
    canvas_bgr[:h1, w0 + gap:] = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2BGR)

    pts0 = kpts0[matches[:, 0]].cpu().numpy()
    pts1 = kpts1[matches[:, 1]].cpu().numpy()

    overlay = canvas_bgr.copy()
    for is_inlier_pass in [False, True]:
        for i, (p0, p1) in enumerate(zip(pts0, pts1)):
            if bool(inliers_mask[i]) != is_inlier_pass:
                continue
            color = (0, 255, 0) if inliers_mask[i] else (0, 0, 255)
            pt_a = (int(p0[0]), int(p0[1]))
            pt_b = (int(p1[0] + w0 + gap), int(p1[1]))
            cv2.line(overlay, pt_a, pt_b, color, 1, cv2.LINE_AA)
            cv2.circle(overlay, pt_a, 6, color, 1, cv2.LINE_AA)
            cv2.circle(overlay, pt_b, 6, color, 1, cv2.LINE_AA)
    alpha_blend = 0.7
    canvas_bgr = cv2.addWeighted(overlay, alpha_blend, canvas_bgr, 1 - alpha_blend, 0)

    def _put_text_bg(img, text, pos, scale, color, bg_color=(0,0,0), bg_alpha=0.6, thickness=2):
        """带背景框的文字"""
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x, y = pos
        pad = 4
        sub = img[max(0,y-th-pad):y+pad+baseline, max(0,x-pad):x+tw+pad]
        if sub.size > 0:
            bg = np.full_like(sub, bg_color, dtype=np.uint8)
            cv2.addWeighted(bg, bg_alpha, sub, 1-bg_alpha, 0, sub)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    # WET / DRY 标签
    rot_note = f" rot={rot_deg:.1f}d" if rot_deg != 0 else ""
    wet_label = f"WET{rot_note}"
    _put_text_bg(canvas_bgr, wet_label, (w0//2 - 60, 25), 0.7, (255,255,0))
    _put_text_bg(canvas_bgr, "DRY", (w0 + gap + w1//2 - 20, 25), 0.7, (0,255,255))

    # --- QR 角度箭头可视化 ---
    def _rotate_qr_pts(qr_pts, rot_deg, orig_h, orig_w):
        """将 QR 角点变换到旋转后的图像坐标系（含画布扩大）。"""
        if qr_pts is None or abs(rot_deg) < 0.1:
            return qr_pts
        pts = qr_pts.copy()
        nearest_90 = round(rot_deg / 90) * 90
        if abs(rot_deg - nearest_90) < 3.0:
            rot_k = (4 - int(nearest_90 / 90) % 4) % 4
            result = np.zeros_like(pts)
            for i, (x, y) in enumerate(pts):
                if rot_k == 1:
                    result[i] = [y, orig_w - 1 - x]
                elif rot_k == 2:
                    result[i] = [orig_w - 1 - x, orig_h - 1 - y]
                elif rot_k == 3:
                    result[i] = [orig_h - 1 - y, x]
            return result
        else:
            cx, cy = orig_w / 2, orig_h / 2
            M = cv2.getRotationMatrix2D((cx, cy), -rot_deg, 1.0)
            cos_a = abs(M[0, 0])
            sin_a = abs(M[0, 1])
            new_w = int(orig_h * sin_a + orig_w * cos_a)
            new_h = int(orig_h * cos_a + orig_w * sin_a)
            M[0, 2] += (new_w - orig_w) / 2
            M[1, 2] += (new_h - orig_h) / 2
            ones = np.ones((len(pts), 1))
            pts_hom = np.hstack([pts, ones])
            return (M @ pts_hom.T).T

    def _draw_qr_arrow_cv(img, qr_pts, qr_angle, x_offset, color):
        """用 OpenCV 绘制 QR 角点 + 方向箭头 + X轴参考线 + 角度弧"""
        if qr_pts is None or qr_angle is None:
            return
        colors_bgr = [(0,0,255), (0,128,255), (255,255,255), (255,255,255)]
        for i, pt in enumerate(qr_pts):
            px, py = int(pt[0] + x_offset), int(pt[1])
            cv2.circle(img, (px, py), 5, colors_bgr[i], -1, cv2.LINE_AA)
            cv2.putText(img, str(i), (px+6, py-6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors_bgr[i], 1, cv2.LINE_AA)
        p0 = qr_pts[0]
        p1 = qr_pts[1]
        pt_from = (int(p0[0] + x_offset), int(p0[1]))
        pt_to = (int(p1[0] + x_offset), int(p1[1]))
        cv2.arrowedLine(img, pt_from, pt_to, color, 2, cv2.LINE_AA, tipLength=0.15)
        arrow_len = int(np.sqrt((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2))
        ref_len = max(arrow_len, 40)
        x_end = (pt_from[0] + ref_len, pt_from[1])
        for d in range(0, ref_len, 8):
            sx = pt_from[0] + d
            ex = min(pt_from[0] + d + 4, x_end[0])
            cv2.line(img, (sx, pt_from[1]), (ex, pt_from[1]), (160,160,160), 1, cv2.LINE_AA)
        arc_r = max(20, ref_len // 3)
        ang = qr_angle % 360
        if ang > 180:
            start_a, end_a = ang, 360
        else:
            start_a, end_a = 0, ang
        cv2.ellipse(img, pt_from, (arc_r, arc_r), 0, start_a, end_a, color, 1, cv2.LINE_AA)
        mid_ang = (start_a + end_a) / 2
        rad = np.deg2rad(mid_ang)
        tx = int(pt_from[0] + (arc_r + 12) * np.cos(rad))
        ty = int(pt_from[1] + (arc_r + 12) * np.sin(rad))
        _put_text_bg(img, f"{qr_angle:+.1f}", (tx-20, ty), 0.45, color)

    # WET QR 箭头
    if meta0 and meta0.get('qr_pts') is not None:
        roi_size_str = meta0.get('roi_size', '')
        if 'x' in roi_size_str:
            ow, oh = map(int, roi_size_str.split('x'))
        else:
            ow, oh = w0, h0
        wet_qr_pts = _rotate_qr_pts(meta0['qr_pts'], rot_deg, oh, ow)
        wet_qr_angle = meta0.get('qr_angle')
        if wet_qr_angle is not None:
            wet_qr_angle_rotated = wet_qr_angle + rot_deg
        else:
            wet_qr_angle_rotated = None
        _draw_qr_arrow_cv(canvas_bgr, wet_qr_pts, wet_qr_angle_rotated, 0, (0,255,0))
    # DRY QR 箭头
    if meta1 and meta1.get('qr_pts') is not None:
        _draw_qr_arrow_cv(canvas_bgr, meta1['qr_pts'], meta1.get('qr_angle'), w0 + gap, (0,255,255))

    # WET 底部信息条
    if meta0:
        info0_parts = []
        if meta0.get('persp_applied'):
            r_before = meta0['persp_ratio']
            r_after = meta0.get('persp_ratio_after', r_before)
            info0_parts.append(f"persp={r_before:.3f}->{r_after:.3f}")
        else:
            info0_parts.append("persp=N/A")
        qr0 = meta0.get('qr_angle')
        info0_parts.append(f"QR={qr0:+.1f}d" if qr0 is not None else "QR=N/A")
        info0_parts.append(f"ROI={meta0.get('roi_size','?')}")
        info0 = " | ".join(info0_parts)
        _put_text_bg(canvas_bgr, info0, (10, h0 - 10), 0.4, (255,255,255))

    # DRY 底部信息条
    if meta1:
        info1_parts = []
        if meta1.get('persp_applied'):
            r_before = meta1['persp_ratio']
            r_after = meta1.get('persp_ratio_after', r_before)
            info1_parts.append(f"persp={r_before:.3f}->{r_after:.3f}")
        else:
            info1_parts.append("persp=N/A")
        qr1 = meta1.get('qr_angle')
        info1_parts.append(f"QR={qr1:+.1f}d" if qr1 is not None else "QR=N/A")
        info1_parts.append(f"ROI={meta1.get('roi_size','?')}")
        qr0_val = meta0.get('qr_angle') if meta0 else None
        if qr0_val is not None and qr1 is not None:
            diff = (qr1 - qr0_val) % 360
            info1_parts.append(f"diff={diff:.1f}")
        info1 = " | ".join(info1_parts)
        _put_text_bg(canvas_bgr, info1, (w0 + gap + 10, h1 - 10), 0.4, (255,255,255))

    # 顶部标题
    _put_text_bg(canvas_bgr, title, (10, canvas_bgr.shape[0] - 5), 0.45, (255,255,255))

    cv2.imwrite(str(save_path), canvas_bgr)


def generate_heatmap(img_bgr, kpts, confs, save_path):
    """特征关注热力图"""
    heatmap = np.zeros(img_bgr.shape[:2], dtype=np.float32)
    for kp, c in zip(kpts, confs):
        x, y = int(kp[0]), int(kp[1])
        if 0 <= y < heatmap.shape[0] and 0 <= x < heatmap.shape[1]:
            cv2.circle(heatmap, (x, y), radius=12, color=float(c), thickness=-1)
    heatmap = cv2.GaussianBlur(heatmap, (41, 41), 0)
    if heatmap.max() > 0:
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    colored = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.6, colored, 0.4, 0)
    _async_save(save_path, overlay)
