import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'best(1).pt'
OUTPUT_DIR = BASE_DIR / 'train_compare_outputs'
SAMPLE_DIR = Path(r'C:\Users\Lenovo\Desktop\1\sample') # ===== 修改读取位置，轮询该目录下的所有子目录=====
IMG_SIZE = 640

# 取消 GPU 加速：本脚本全程使用 CPU
# 精细度控制：fast / medium / deep
QUALITY_LEVEL = 'deep'  # 可选 'fast', 'medium', 'deep'

WEIGHT_COSINE = 0.40
WEIGHT_EUCLIDEAN = 0.20
WEIGHT_MASK_AREA = 0.16
WEIGHT_ASPECT_RATIO = 0.12
WEIGHT_CONTOUR_AREA = 0.12
VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# ===== 统计阈值（对比度低的判定阈值）=====
LOW_MATCH_THRESHOLD = 0.80


def load_model():
    model = YOLO(str(MODEL_PATH))
    model.model.eval()
    model.model.to('cpu')
    return model


def load_image(image_path):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f'无法读取图片: {image_path}')
    return img_bgr


def create_run_output_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)
    run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = OUTPUT_DIR / run_name
    run_dir.mkdir(exist_ok=True)
    return run_dir


def _rectify_transform_from_qr(img_shape_hw, pts):
    """给定二维码四点，返回透视矩阵 M=shift@H 以及输出尺寸 (out_w,out_h)。"""
    src = np.array(pts).reshape(4, 2).astype(np.float32)
    side = float(np.mean([
        np.linalg.norm(src[1] - src[0]),
        np.linalg.norm(src[2] - src[1]),
        np.linalg.norm(src[3] - src[2]),
        np.linalg.norm(src[0] - src[3]),
    ]))
    dst = np.array([[0, 0], [side, 0], [side, side], [0, side]], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, dst)
    h, w = img_shape_hw
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

    x_min, y_min = mapped.min(axis=0)
    x_max, y_max = mapped.max(axis=0)

    shift = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    out_w = int(np.ceil(x_max - x_min))
    out_h = int(np.ceil(y_max - y_min))
    M = shift @ H
    return M, (out_w, out_h)


def rectify_by_qr(img, pts):
    h, w = img.shape[:2]
    M, (out_w, out_h) = _rectify_transform_from_qr((h, w), pts)
    return cv2.warpPerspective(img, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def warp_mask_with_M(mask_u8, M, out_size_wh):
    """mask 用同一个透视矩阵同步拉正（最近邻，保持0/1）。"""
    out_w, out_h = out_size_wh
    warped = cv2.warpPerspective(mask_u8.astype(np.uint8) * 255, M, (out_w, out_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return (warped > 127).astype(np.uint8)


def detect_qr_robust(img):
    detector = cv2.QRCodeDetector()
    text, pts, _ = detector.detectAndDecode(img)
    if pts is not None and text:
        return text, pts, img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    enhanced = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    text, pts, _ = detector.detectAndDecode(enhanced)
    if pts is not None and text:
        return text, pts, enhanced

    h, w = img.shape[:2]
    for scale in [0.5, 0.75, 1.5, 0.25]:
        resized = cv2.resize(img, (int(w * scale), int(h * scale)))
        text, pts, _ = detector.detectAndDecode(resized)
        if pts is not None and text:
            return text, pts / scale, img
    return '', None, img


def rectify_and_read_qr(img_bgr, mask_u8=None):
    """对输入图做二维码检测并拉正；若提供 mask，则用同一矩阵同步拉正 mask。"""
    qr_text, points, _ = detect_qr_robust(img_bgr)
    if points is None:
        return img_bgr, False, '', None, None

    h, w = img_bgr.shape[:2]
    M, (out_w, out_h) = _rectify_transform_from_qr((h, w), points[0])
    rectified = cv2.warpPerspective(img_bgr, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    rectified_mask = None
    if mask_u8 is not None:
        rectified_mask = warp_mask_with_M(mask_u8, M, (out_w, out_h))

    qr_text_after, _, _ = detect_qr_robust(rectified)
    return rectified, True, (qr_text_after or qr_text), points, rectified_mask


def segment_main_object(model, img_bgr):
    results = model.predict(source=img_bgr, device='cpu', verbose=False)
    result = results[0]
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return img_bgr, None, None, None
    scores = result.boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(scores))
    mask = result.masks.data[best_idx].cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8)
    mask = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    box_xyxy = None
    if result.boxes.xyxy is not None and len(result.boxes.xyxy) > best_idx:
        box_xyxy = result.boxes.xyxy[best_idx].cpu().numpy().astype(int)
    if box_xyxy is None:
        return img_bgr, None, None, None
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    h, w = img_bgr.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return img_bgr, None, box_xyxy, None
    crop_xyxy = (x1, y1, x2, y2)
    cropped = img_bgr[y1:y2 + 1, x1:x2 + 1].copy()
    cropped_mask = mask[y1:y2 + 1, x1:x2 + 1]
    cropped[cropped_mask == 0] = 0
    return cropped, mask, box_xyxy, crop_xyxy


def draw_qr_polygon(img_bgr, qr_points, color=(0, 255, 255), thickness=3):
    canvas = img_bgr.copy()
    if qr_points is None:
        return canvas
    pts = np.array(qr_points).reshape(-1, 2).astype(int)
    if len(pts) >= 4:
        cv2.polylines(canvas, [pts.reshape((-1, 1, 2))], True, color, thickness)
    return canvas


def draw_mask_overlay(img_bgr, mask, color=(0, 0, 255), alpha=0.35):
    canvas = img_bgr.copy()
    if mask is None:
        return canvas
    overlay = canvas.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)


def draw_detection_visual(img_bgr, mask=None, box_xyxy=None, qr_points=None, title='', qr_text=''):
    canvas = draw_mask_overlay(img_bgr, mask)
    canvas = draw_qr_polygon(canvas, qr_points)
    if box_xyxy is not None:
        x1, y1, x2, y2 = [int(v) for v in box_xyxy]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if title:
        cv2.putText(canvas, title, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    if qr_text:
        qr_show = f'QR: {qr_text}'
        cv2.rectangle(canvas, (12, 52), (min(canvas.shape[1] - 12, 12 + max(420, len(qr_show) * 16)), 96), (0, 0, 0), -1)
        cv2.putText(canvas, qr_show, (20, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return canvas


def make_side_by_side(img1, img2, label1='Image 1', label2='Image 2'):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    target_h = max(h1, h2)

    def resize_keep(img, target_h):
        h, w = img.shape[:2]
        scale = target_h / max(h, 1)
        new_w = max(1, int(round(w * scale)))
        return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LINEAR)

    r1 = resize_keep(img1, target_h)
    r2 = resize_keep(img2, target_h)
    pad = np.full((target_h, 30, 3), 30, dtype=np.uint8)
    merged = np.hstack([r1, pad, r2])
    cv2.putText(merged, label1, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(merged, label2, (r1.shape[1] + 50, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return merged


def save_visualizations(pair_dir, image1_path, image2_path, orig1, orig2, rectified1, rectified2, mask1, mask2, rect_mask1, rect_mask2, box1, box2, crop1_xyxy, crop2_xyxy, qr_points1, qr_points2, qr_text1, qr_text2, fitness, decision_note):
    vis_orig1 = draw_detection_visual(orig1, mask1, box1, None, '原图1 / YOLO', qr_text1)
    vis_orig2 = draw_detection_visual(orig2, mask2, box2, None, '原图2 / YOLO', qr_text2)
    # 在拉正图上重新检测一次二维码点，用于可视化框（避免坐标系错位）
    _, rect_pts1, _ = detect_qr_robust(rectified1)
    _, rect_pts2, _ = detect_qr_robust(rectified2)
    vis_rect1 = draw_detection_visual(rectified1, rect_mask1, None, rect_pts1, '裁剪后拉正图1 / QR', qr_text1)
    vis_rect2 = draw_detection_visual(rectified2, rect_mask2, None, rect_pts2, '裁剪后拉正图2 / QR', qr_text2)

    compare_orig = make_side_by_side(vis_orig1, vis_orig2, '原图1', '原图2')
    compare_rect = make_side_by_side(vis_rect1, vis_rect2, '拉正图1', '拉正图2')

    # NOTE: 不再在下方贴 ROI。最终 vis_compare_rectified 的主画面就是 YOLO 绿框裁剪区域（crop/rectified）。

    cv2.rectangle(compare_rect, (10, compare_rect.shape[0] - 74), (min(compare_rect.shape[1] - 10, 980), compare_rect.shape[0] - 10), (0, 0, 0), -1)
    cv2.putText(compare_rect, f'{decision_note} | 匹配度={fitness:.4f}', (20, compare_rect.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 255, 255), 4, cv2.LINE_AA)

    cv2.imwrite(str(pair_dir / f'vis_original_1_{Path(image1_path).stem}.jpg'), vis_orig1)
    cv2.imwrite(str(pair_dir / f'vis_original_2_{Path(image2_path).stem}.jpg'), vis_orig2)
    cv2.imwrite(str(pair_dir / f'vis_mask_1_{Path(image1_path).stem}.jpg'), vis_rect1)
    cv2.imwrite(str(pair_dir / f'vis_mask_2_{Path(image2_path).stem}.jpg'), vis_rect2)
    cv2.imwrite(str(pair_dir / 'vis_compare_original.jpg'), compare_orig)
    cv2.imwrite(str(pair_dir / 'vis_compare_rectified.jpg'), compare_rect)
    return compare_orig, compare_rect


def preprocess_cropped_image(img_bgr, img_size=IMG_SIZE):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (img_size, img_size))
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0)


def preprocess_cropped_image_multiscale(img_bgr, img_size=IMG_SIZE, scales=[0.9, 1.0, 1.1]):
    """多尺度预处理，生成不同尺度的图像张量 - 使用更小范围和更少尺度"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    tensors = []
    
    for scale in scales:
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 中心裁剪或填充到目标尺寸
        if new_h < img_size or new_w < img_size:
            # 需要填充
            pad_h = max(0, img_size - new_h)
            pad_w = max(0, img_size - new_w)
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left
            resized = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        # 中心裁剪
        start_h = (resized.shape[0] - img_size) // 2
        start_w = (resized.shape[1] - img_size) // 2
        cropped = resized[start_h:start_h + img_size, start_w:start_w + img_size]
        
        tensor = torch.from_numpy(cropped).permute(2, 0, 1).float() / 255.0
        tensors.append(tensor.unsqueeze(0))
    
    return tensors


def extract_feature_from_cropped(model, cropped_bgr):
    """提取深度特征 - 使用多尺度特征融合"""
    # 多尺度特征提取
    multi_scale_tensors = preprocess_cropped_image_multiscale(cropped_bgr, img_size=IMG_SIZE)
    
    all_features = []
    spatial_features_list = []
    
    with torch.no_grad():
        for x in multi_scale_tensors:
            outputs = model.model(x)
            tensors = []
            spatial_tensors = []

            def collect_tensors(obj, keep_spatial=False):
                if isinstance(obj, torch.Tensor):
                    if obj.ndim == 4:
                        if keep_spatial and len(spatial_tensors) == 0:  # 只保留第一个空间特征
                            spatial_tensors.append(obj)
                        # 全局池化特征
                        if obj.shape[2] >= 4 and obj.shape[3] >= 4:  # 只保留足够大的特征图
                            tensors.append(F.adaptive_avg_pool2d(obj, (1, 1)).flatten(1))
                    elif obj.ndim == 3:
                        tensors.append(obj.mean(dim=1))
                    elif obj.ndim == 2:
                        tensors.append(obj)
                elif isinstance(obj, (list, tuple)):
                    for item in obj:
                        collect_tensors(item, keep_spatial=keep_spatial)
                elif isinstance(obj, dict):
                    for item in obj.values():
                        collect_tensors(item, keep_spatial=keep_spatial)

            collect_tensors(outputs, keep_spatial=True)
            
            if not tensors:
                raise RuntimeError('未能从模型输出中提取到有效特征')
            
            # 合并当前尺度的特征（限制特征维度以避免过大）
            if len(tensors) > 4:
                tensors = tensors[:4]  # 只取前4个特征张量
            scale_embedding = torch.cat(tuple(tensors), dim=1)
            all_features.append(scale_embedding)
            
            # 保存空间特征用于局部匹配
            if spatial_tensors:
                spatial_features_list.append(spatial_tensors[0])
    
    # 多尺度特征融合（加权平均）- 简化权重
    weights = [0.2, 0.6, 0.2]  # 小、中、大尺度的权重，中间权重更高
    fused_features = []
    for i, feat in enumerate(all_features):
        w = weights[i] if i < len(weights) else 1.0 / len(all_features)
        fused_features.append(feat * w)
    
    embedding = torch.sum(torch.stack(fused_features), dim=0)
    embedding = F.normalize(embedding, p=2, dim=1)
    
    # 计算局部特征相似度
    local_similarity = None
    if len(spatial_features_list) >= 2:
        # 使用中间尺度的空间特征进行局部匹配
        mid_spatial = spatial_features_list[1]  # 1.0x 尺度
        local_similarity = mid_spatial
    
    return embedding.squeeze(0).cpu().numpy(), local_similarity


def compute_local_feature_similarity(spatial_feat1, spatial_feat2):
    """计算局部特征相似度 - 简化版本"""
    if spatial_feat1 is None or spatial_feat2 is None:
        return 0.5  # 默认值
    
    # 确保空间维度一致
    if spatial_feat1.shape != spatial_feat2.shape:
        # 使用插值调整到相同尺寸
        target_h = min(spatial_feat1.shape[2], spatial_feat2.shape[2])
        target_w = min(spatial_feat1.shape[3], spatial_feat2.shape[3])
        spatial_feat1 = F.interpolate(spatial_feat1, size=(target_h, target_w), mode='bilinear', align_corners=False)
        spatial_feat2 = F.interpolate(spatial_feat2, size=(target_h, target_w), mode='bilinear', align_corners=False)
    
    # 简化版：直接计算余弦相似度（不使用复杂的注意力机制）
    feat1_flat = F.normalize(spatial_feat1.flatten(2), p=2, dim=1)
    feat2_flat = F.normalize(spatial_feat2.flatten(2), p=2, dim=1)
    
    # 计算平均余弦相似度
    similarity = torch.mean(torch.sum(feat1_flat * feat2_flat, dim=1))
    
    # 转换到0-1范围
    local_sim = (similarity.item() + 1) / 2
    return max(0.0, min(1.0, local_sim))


def compute_structural_similarity(img1, img2):
    """计算结构相似性 (SSIM) - 感知相似度"""
    # 转换为灰度图
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1
    
    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2
    
    # 确保尺寸一致
    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
    
    # 计算SSIM
    mu1 = cv2.GaussianBlur(gray1.astype(np.float32), (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(gray2.astype(np.float32), (11, 11), 1.5)
    
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(gray1.astype(np.float32) ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(gray2.astype(np.float32) ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(gray1.astype(np.float32) * gray2.astype(np.float32), (11, 11), 1.5) - mu1_mu2
    
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    
    return float(np.mean(ssim_map))


def compute_ssim_fast(img1, img2):
    """快速SSIM - 使用降采样"""
    # 转换为灰度并降采样
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1
    
    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2
    
    # 降采样到128x128以加速
    gray1 = cv2.resize(gray1, (128, 128))
    gray2 = cv2.resize(gray2, (128, 128))
    
    # 简化版SSIM - 使用均值滤波代替高斯滤波
    mu1 = cv2.blur(gray1.astype(np.float32), (7, 7))
    mu2 = cv2.blur(gray2.astype(np.float32), (7, 7))
    
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.blur(gray1.astype(np.float32)**2, (7, 7)) - mu1_sq
    sigma2_sq = cv2.blur(gray2.astype(np.float32)**2, (7, 7)) - mu2_sq
    sigma12 = cv2.blur(gray1.astype(np.float32)*gray2.astype(np.float32), (7, 7)) - mu1_mu2
    
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8)
    
    return float(np.mean(ssim_map))


def compute_histogram_similarity(img1, img2):
    """计算颜色直方图相似度"""
    # 转换为HSV颜色空间
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    
    # 计算H和S通道的直方图
    hist1_h = cv2.calcHist([hsv1], [0], None, [180], [0, 180])
    hist1_s = cv2.calcHist([hsv1], [1], None, [256], [0, 256])
    hist2_h = cv2.calcHist([hsv2], [0], None, [180], [0, 180])
    hist2_s = cv2.calcHist([hsv2], [1], None, [256], [0, 256])
    
    # 归一化
    cv2.normalize(hist1_h, hist1_h, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist1_s, hist1_s, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2_h, hist2_h, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2_s, hist2_s, 0, 1, cv2.NORM_MINMAX)
    
    # 计算相关性
    corr_h = cv2.compareHist(hist1_h, hist2_h, cv2.HISTCMP_CORREL)
    corr_s = cv2.compareHist(hist1_s, hist2_s, cv2.HISTCMP_CORREL)
    
    return (max(0, corr_h) + max(0, corr_s)) / 2


def extract_texture_features(img):
    """快速纹理特征提取 - 使用积分图和降采样"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 降采样以加速
    small = cv2.resize(gray, (64, 64))
    
    # 使用Sobel算子提取梯度特征（比LBP快得多）
    sobelx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    
    # 计算梯度幅值和方向直方图
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    direction = np.arctan2(sobely, sobelx) * 180 / np.pi
    
    # 分块统计
    features = []
    h, w = small.shape
    blocks = 4
    bh, bw = h // blocks, w // blocks
    
    for i in range(blocks):
        for j in range(blocks):
            block_mag = magnitude[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            block_dir = direction[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            
            # 梯度幅值均值
            features.append(np.mean(block_mag))
            # 梯度方向直方图（8个bin）
            hist, _ = np.histogram(block_dir, bins=8, range=(-180, 180))
            features.extend(hist / (np.sum(hist) + 1e-8))
    
    return np.array(features, dtype=np.float32)


def compute_texture_similarity(img1, img2):
    """快速纹理相似度计算"""
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    feat1 = extract_texture_features(img1)
    feat2 = extract_texture_features(img2)
    
    # 余弦相似度
    dot = np.dot(feat1, feat2)
    norm1 = np.linalg.norm(feat1)
    norm2 = np.linalg.norm(feat2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.5
    
    similarity = dot / (norm1 * norm2)
    return (similarity + 1) / 2


def cosine_similarity(vec1, vec2):
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


def euclidean_distance(vec1, vec2):
    return float(np.linalg.norm(vec1 - vec2))


def extract_mask_features(mask):
    if mask is None:
        return {'area_ratio': None, 'aspect_ratio': None, 'contour_area_ratio': None}
    binary = (mask > 0).astype(np.uint8)
    h, w = binary.shape[:2]
    area_ratio = float(binary.sum() / (h * w)) if h > 0 and w > 0 else None
    ys, xs = np.where(binary > 0)
    if len(xs) == 0 or len(ys) == 0:
        return {'area_ratio': area_ratio, 'aspect_ratio': None, 'contour_area_ratio': None}
    bw = max(1, int(xs.max() - xs.min() + 1))
    bh = max(1, int(ys.max() - ys.min() + 1))
    aspect_ratio = float(bw / bh)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_area = max((cv2.contourArea(c) for c in contours), default=0.0)
    contour_area_ratio = float(contour_area / (h * w)) if h > 0 and w > 0 else None
    return {'area_ratio': area_ratio, 'aspect_ratio': aspect_ratio, 'contour_area_ratio': contour_area_ratio}


def safe_abs_diff(v1, v2):
    if v1 is None or v2 is None:
        return None
    return float(abs(v1 - v2))


def qr_match_score(qr_text1, qr_text2, qr_found1, qr_found2):
    if qr_found1 and qr_found2:
        t1 = (qr_text1 or '').strip()
        t2 = (qr_text2 or '').strip()
        if not t1 or not t2:
            return -1.0
        return 1.0 if t1 == t2 else 0.0
    return -1.0


def build_feature_distances(feat1, feat2, mask1, mask2, qr_text1, qr_text2, qr_found1, qr_found2, 
                           spatial_feat1=None, spatial_feat2=None, img1=None, img2=None):
    """构建特征距离 - 包含深度特征、局部特征、结构特征、纹理特征"""
    m1 = extract_mask_features(mask1)
    m2 = extract_mask_features(mask2)
    
    # 基础深度特征相似度
    cosine_sim = cosine_similarity(feat1, feat2)
    euclid_dist = euclidean_distance(feat1, feat2)
    
    # 局部特征相似度
    local_sim = compute_local_feature_similarity(spatial_feat1, spatial_feat2)
    
    # 结构相似度 (SSIM) - 使用快速版本
    ssim = 0.5
    if img1 is not None and img2 is not None:
        try:
            ssim = compute_ssim_fast(img1, img2)
        except:
            pass
    
    # 颜色直方图相似度
    hist_sim = 0.5
    if img1 is not None and img2 is not None:
        try:
            hist_sim = compute_histogram_similarity(img1, img2)
        except:
            pass
    
    # 纹理相似度
    texture_sim = 0.5
    if img1 is not None and img2 is not None:
        try:
            texture_sim = compute_texture_similarity(img1, img2)
        except:
            pass
    
    return {
        'qr_match_score': qr_match_score(qr_text1, qr_text2, qr_found1, qr_found2),
        'cosine_similarity': cosine_sim,
        'euclidean_distance': euclid_dist,
        'cnn_local_similarity': local_sim,
        'ssim': ssim,
        'histogram_similarity': hist_sim,
        'texture_similarity': texture_sim,
        'mask_area_diff': safe_abs_diff(m1['area_ratio'], m2['area_ratio']),
        'aspect_ratio_diff': safe_abs_diff(m1['aspect_ratio'], m2['aspect_ratio']),
        'contour_area_diff': safe_abs_diff(m1['contour_area_ratio'], m2['contour_area_ratio']),
    }


def build_fitness_score(distances):
    """构建综合匹配度分数 - 使用多维度特征融合"""
    qr_score = distances['qr_match_score']
    
    # 二维码识别不清时，仍然提示重新拍摄
    if qr_score == -1.0:
        return 0.0, '二维码识别不清，请重新拍摄'
    
    # 二维码不一致时，继续计算相似度，但标记状态
    qr_note = '二维码不一致' if qr_score == 0.0 else '二维码一致'

    # 深度特征相似度（全局 + 局部）
    cosine_score = max(0.0, min(1.0, distances['cosine_similarity']))
    local_score = max(0.0, min(1.0, distances.get('cnn_local_similarity', cosine_score)))
    
    # 结构相似度
    ssim_score = max(0.0, min(1.0, distances.get('ssim', 0.5)))
    
    # 颜色相似度
    hist_score = max(0.0, min(1.0, distances.get('histogram_similarity', 0.5)))
    
    # 纹理相似度
    texture_score = max(0.0, min(1.0, distances.get('texture_similarity', 0.5)))
    
    # 几何特征
    euclidean_score = 1.0 / (1.0 + distances['euclidean_distance'])
    area_score = 1.0 if distances['mask_area_diff'] is None else 1.0 / (1.0 + distances['mask_area_diff'] * 10)
    aspect_score = 1.0 if distances['aspect_ratio_diff'] is None else 1.0 / (1.0 + distances['aspect_ratio_diff'] * 5)
    contour_score = 1.0 if distances['contour_area_diff'] is None else 1.0 / (1.0 + distances['contour_area_diff'] * 10)

    # 新的权重分配（更均衡的多维度融合）
    # 深度特征：全局25% + 局部15% = 40%
    # 感知特征：结构15% + 颜色10% + 纹理10% = 35%
    # 几何特征：欧氏10% + 面积5% + 宽高比5% + 轮廓5% = 25%
    
    fitness = (
        0.10 * cosine_score +      # 深度全局特征（余弦相似度）
        0.10 * local_score +       # 深度局部特征
        0.30 * ssim_score +        # 结构相似度(SSIM)
        0.05 * hist_score +        # 颜色直方图相似度
        0.10 * texture_score +     # 纹理特征（保留，未单独指定时分到剩余权重）
        0.10 * euclidean_score +   # 欧氏距离（几何整体差异）
        0.10 * area_score +        # 主目标面积差
        0.05 * aspect_score +      # 宽高比差
        0.10 * contour_score       # 轮廓面积差
    )
    
    return float(fitness), qr_note


def save_compare_outputs(pair_dir, image1_path, image2_path, crop1, crop2, rectified1, rectified2):
    out1 = pair_dir / f"crop_1_{Path(image1_path).name}"
    out2 = pair_dir / f"crop_2_{Path(image2_path).name}"
    rect1 = pair_dir / f"rectified_1_{Path(image1_path).name}"
    rect2 = pair_dir / f"rectified_2_{Path(image2_path).name}"
    cv2.imwrite(str(out1), crop1)
    cv2.imwrite(str(out2), crop2)
    cv2.imwrite(str(rect1), rectified1)
    cv2.imwrite(str(rect2), rectified2)
    return out1, out2, rect1, rect2


def save_compare_report(pair_dir, image1_path, image2_path, rect1, rect2, out1, out2, distances, fitness, decision_note, qr1, qr2, qr_text1, qr_text2, mask1, mask2):
    report_path = pair_dir / 'compare_report.txt'
    
    # 判断是否显示匹配度数值
    qr_score = distances['qr_match_score']
    if qr_score == -1.0:
        fitness_display = "N/A (二维码识别不清)"
    else:
        # 二维码不一致时也显示匹配度
        fitness_display = f"{fitness:.4f}"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('=== 批量图片比较结果报告 ===\n\n')
        f.write(f'原图1: {image1_path}\n')
        f.write(f'原图2: {image2_path}\n\n')
        f.write(f'拉正图1: {rect1}\n')
        f.write(f'拉正图2: {rect2}\n')
        f.write(f'裁剪图1: {out1}\n')
        f.write(f'裁剪图2: {out2}\n\n')
        f.write(f'二维码检测1: {"成功" if qr1 else "失败"}\n')
        f.write(f'二维码检测2: {"成功" if qr2 else "失败"}\n')
        f.write(f'二维码内容1: {qr_text1 if qr_text1 else "N/A"}\n')
        f.write(f'二维码内容2: {qr_text2 if qr_text2 else "N/A"}\n')
        f.write(f'二维码一致性: {decision_note}\n\n')
        f.write(f'主目标分割1: {"成功" if mask1 is not None else "失败"}\n')
        f.write(f'主目标分割2: {"成功" if mask2 is not None else "失败"}\n\n')
        
        # 详细计算值（只要二维码识别成功就显示，无论是否一致）
        if qr_score != -1.0:
            f.write('=== 深度特征 ===\n')
            f.write(f'余弦相似度: {distances["cosine_similarity"]:.4f} (权重10%)\n')
            f.write(f'局部特征相似度: {distances.get("cnn_local_similarity", 0):.4f} (权重10%)\n')
            f.write(f'欧氏距离: {distances["euclidean_distance"]:.4f} -> 转换分: {1.0/(1.0+distances["euclidean_distance"]):.4f} (权重10%)\n\n')
            
            f.write('=== 感知特征 ===\n')
            f.write(f'结构相似度(SSIM): {distances.get("ssim", 0):.4f} (权重30%)\n')
            f.write(f'颜色直方图相似度: {distances.get("histogram_similarity", 0):.4f} (权重5%)\n')
            f.write(f'纹理相似度: {distances.get("texture_similarity", 0):.4f} (权重10%)\n\n')
            
            f.write('=== 几何特征 ===\n')
            f.write(f'主目标面积差: {distances["mask_area_diff"] if distances["mask_area_diff"] is not None else "N/A"} (权重10%)\n')
            f.write(f'宽高比差: {distances["aspect_ratio_diff"] if distances["aspect_ratio_diff"] is not None else "N/A"} (权重5%)\n')
            f.write(f'轮廓面积差: {distances["contour_area_diff"] if distances["contour_area_diff"] is not None else "N/A"} (权重10%)\n\n')
        
        f.write(f'最终匹配度: {fitness_display}\n')
    return report_path


def process_pair(model, image1_path, image2_path, pair_dir):
    img1 = load_image(image1_path)
    img2 = load_image(image2_path)
    crop1, mask1_full, box1, crop1_xyxy = segment_main_object(model, img1)
    crop2, mask2_full, box2, crop2_xyxy = segment_main_object(model, img2)

    crop1_mask = None
    if mask1_full is not None and crop1_xyxy is not None:
        x1, y1, x2, y2 = crop1_xyxy
        crop1_mask = mask1_full[y1:y2 + 1, x1:x2 + 1]
    crop2_mask = None
    if mask2_full is not None and crop2_xyxy is not None:
        x1, y1, x2, y2 = crop2_xyxy
        crop2_mask = mask2_full[y1:y2 + 1, x1:x2 + 1]

    rectified1, qr1, qr_text1, qr_points1, rect_mask1 = rectify_and_read_qr(crop1, crop1_mask)
    rectified2, qr2, qr_text2, qr_points2, rect_mask2 = rectify_and_read_qr(crop2, crop2_mask)

    out1, out2, rect1, rect2 = save_compare_outputs(pair_dir, image1_path, image2_path, crop1, crop2, rectified1, rectified2)
    
    # 使用新的特征提取方法（返回特征和空间特征）
    feat1, spatial_feat1 = extract_feature_from_cropped(model, rectified1)
    feat2, spatial_feat2 = extract_feature_from_cropped(model, rectified2)
    
    # 构建特征距离（包含多维度特征）
    distances = build_feature_distances(
        feat1, feat2, rect_mask1, rect_mask2, 
        qr_text1, qr_text2, qr1, qr2,
        spatial_feat1, spatial_feat2,
        rectified1, rectified2
    )
    
    fitness, decision_note = build_fitness_score(distances)
    save_compare_report(pair_dir, image1_path, image2_path, rect1, rect2, out1, out2, distances, fitness, decision_note, qr1, qr2, qr_text1, qr_text2, rect_mask1, rect_mask2)
    save_visualizations(pair_dir, image1_path, image2_path, img1, img2, rectified1, rectified2, mask1_full, mask2_full, rect_mask1, rect_mask2, box1, box2, crop1_xyxy, crop2_xyxy, qr_points1, qr_points2, qr_text1, qr_text2, fitness, decision_note)
    return fitness, decision_note


def find_pairs_in_folder(folder: Path):
    """在单个子文件夹内决定哪些图片要互相比对。

    默认规则（当前启用）：
      - 只看文件名形如 A1 / B1 / A2 / B2 ...（扩展名任意，只要是图片）
      - 跳过所有 C*（例如 C1/C2/C3 都不参与）
      - 按相同编号配对：A1 对 B1，A2 对 B2，A3 对 B3 ...

    如果你想手动改规则（例如 A1 与 B2 比对）：
      1) 先在下面的 MANUAL_PAIR_MAP 里写映射（A编号 -> B编号）
         例如：
             MANUAL_PAIR_MAP = {
                 '1': '2',  # A1 -> B2
                 '2': '1',  # A2 -> B1
             }
      2) 然后把 USE_MANUAL_MAP 改成 True

    注意：
      - 只会对“映射存在且文件真实存在”的配对进行比对
      - 仍然会跳过 C*
    """

    USE_MANUAL_MAP = True
    MANUAL_PAIR_MAP = {
        '1': '2',  # A1 -> B2
        '2': '3',  # A2 -> B3
        '3': '1',  # A3 -> B1
    }

    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
    a_map = {}
    b_map = {}

    for p in files:
        stem = p.stem
        m = re.match(r'^([ABCabc])(\d+)$', stem)
        if not m:
            continue
        prefix = m.group(1).upper()
        idx = m.group(2)
        if prefix == 'C':
            continue
        if prefix == 'A':
            a_map[idx] = p
        elif prefix == 'B':
            b_map[idx] = p

    if USE_MANUAL_MAP:
        pairs = []
        # A编号 -> B编号
        for a_idx, b_idx in MANUAL_PAIR_MAP.items():
            if a_idx in a_map and b_idx in b_map:
                # 返回的第一个 idx 仍然用 A 的编号，便于命名
                pairs.append((a_idx, a_map[a_idx], b_map[b_idx]))
        return pairs

    common = sorted(set(a_map.keys()) & set(b_map.keys()), key=lambda x: int(x))
    return [(idx, a_map[idx], b_map[idx]) for idx in common]


def main():
    if not MODEL_PATH.exists():
        print('模型文件不存在。')
        return
    if not SAMPLE_DIR.exists():
        print(f'样本目录不存在: {SAMPLE_DIR}')
        return

    run_dir = create_run_output_dir()
    summary_lines = []
    model = load_model()

    folders = sorted([p for p in SAMPLE_DIR.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not folders:
        print('样本目录下没有子文件夹。')
        return

    for folder in folders:
        pairs = find_pairs_in_folder(folder)
        if not pairs:
            continue
        folder_out = run_dir / folder.name
        folder_out.mkdir(exist_ok=True)
        print(f'处理文件夹: {folder.name}')
        for idx, a_path, b_path in pairs:
            pair_name = f'A{idx}_vs_B{idx}'
            pair_dir = folder_out / pair_name
            pair_dir.mkdir(exist_ok=True)
            try:
                fitness, decision_note = process_pair(model, a_path, b_path, pair_dir)
                summary = f'{folder.name} | {pair_name} | {decision_note} | 匹配度={fitness:.4f}'
                print(summary)
                summary_lines.append(summary)
            except Exception as e:
                summary = f'{folder.name} | {pair_name} | 处理失败 | {e}'
                print(summary)
                summary_lines.append(summary)

    # ===== 统计汇总（扫描 vs 对比度低） =====
    total = 0
    scan_fail = 0          # 处理失败 或 二维码识别不清
    scan_ok = 0

    # 在扫描成功中进一步拆分
    qr_mismatch = 0        # 二维码不一致
    qr_mismatch_low = 0    # 二维码不一致且对比度低
    qr_mismatch_ok = 0     # 二维码不一致但对比正常
    low_match = 0          # 对比度低（匹配度 < LOW_MATCH_THRESHOLD，不含二维码不一致）
    ok_match = 0           # 对比正常

    ge90 = 0               # 匹配度 >= 0.90（统计所有识别成功的情况）
    ge80 = 0               # 匹配度 >= 0.80（统计所有识别成功的情况）
    ge70 = 0               # 匹配度 >= 0.70（统计所有识别成功的情况）
    ge60 = 0               # 匹配度 >= 0.60（统计所有识别成功的情况）
    ge50 = 0               # 匹配度 >= 0.50（统计所有识别成功的情况）

    for line in summary_lines:
        if not line.strip():
            continue
        total += 1

        if '处理失败' in line or '识别不清' in line or '请重新拍摄' in line:
            scan_fail += 1
            continue

        # 提取匹配度数值
        m = re.search(r'匹配度=([0-9]*\.?[0-9]+)', line)
        v = None
        if m:
            v = float(m.group(1))
            if v >= 0.90:
                ge90 += 1
            if v >= 0.80:
                ge80 += 1
            if v >= 0.70:
                ge70 += 1
            if v >= 0.60:
                ge60 += 1
            if v >= 0.50:
                ge50 += 1

        # 二维码不一致的情况
        if '二维码不一致' in line:
            scan_ok += 1
            qr_mismatch += 1
            # 使用相同阈值判断是否对比度低
            if v is not None and v < LOW_MATCH_THRESHOLD:
                qr_mismatch_low += 1
            else:
                qr_mismatch_ok += 1
            continue

        # 到这里是二维码一致的情况
        scan_ok += 1

        if v is not None and v < LOW_MATCH_THRESHOLD:
            low_match += 1
        else:
            ok_match += 1

    fail_rate = (scan_fail / total * 100.0) if total else 0.0
    qr_mismatch_rate = (qr_mismatch / total * 100.0) if total else 0.0
    low_match_rate = (low_match / scan_ok * 100.0) if scan_ok else 0.0
    ok_match_rate = (ok_match / scan_ok * 100.0) if scan_ok else 0.0
    qr_mismatch_low_rate = (qr_mismatch_low / qr_mismatch * 100.0) if qr_mismatch else 0.0

    ge90_rate = (ge90 / scan_ok * 100.0) if scan_ok else 0.0
    ge80_rate = (ge80 / scan_ok * 100.0) if scan_ok else 0.0
    ge70_rate = (ge70 / scan_ok * 100.0) if scan_ok else 0.0
    ge60_rate = (ge60 / scan_ok * 100.0) if scan_ok else 0.0
    ge50_rate = (ge50 / scan_ok * 100.0) if scan_ok else 0.0

    header = [
        '=== 统计汇总 ===',
        f'总扫描次数: {total}',
        f'',
        f'【二维码识别问题】',
        f'  扫描失败次数: {scan_fail} (占比: {fail_rate:.2f}%)  [二维码识别不清]',
        f'  二维码不一致次数: {qr_mismatch} (占比: {qr_mismatch_rate:.2f}%)  [二维码内容不同]',
        f'    - 其中对比度低: {qr_mismatch_low} (占比: {qr_mismatch_low_rate:.2f}%) [匹配度<{LOW_MATCH_THRESHOLD}]',
        f'    - 其中对比正常: {qr_mismatch_ok}',
        f'',
        f'【匹配度分析（仅统计二维码一致的情况）】',
        f'  扫描成功次数: {scan_ok}',
        f'  对比度低次数: {low_match} (占比: {low_match_rate:.2f}%)  [匹配度<{LOW_MATCH_THRESHOLD}]',
        f'  对比正常次数: {ok_match} (占比: {ok_match_rate:.2f}%)',
        f'',
        f'【高分匹配统计（统计所有识别成功的情况）】',
        f'  匹配度>=0.90: {ge90_rate:.2f}% ({ge90}/{scan_ok})',
        f'  匹配度>=0.80: {ge80_rate:.2f}% ({ge80}/{scan_ok})',
        f'  匹配度>=0.70: {ge70_rate:.2f}% ({ge70}/{scan_ok})',
        f'  匹配度>=0.60: {ge60_rate:.2f}% ({ge60}/{scan_ok})',
        f'  匹配度>=0.50: {ge50_rate:.2f}% ({ge50}/{scan_ok})',
        '=== 明细 ===',
    ]

    summary_path = run_dir / 'summary.txt'
    with open(summary_path, 'w', encoding='utf-8-sig') as f:
        f.write('\n'.join(header))
        if summary_lines:
            f.write('\n')
            f.write('\n'.join(summary_lines))
        else:
            f.write('没有找到可比较的 A/B 配对。')

    print('')
    print(f'批量处理完成，结果目录: {run_dir}')
    print(f'汇总文件: {summary_path}')


if __name__ == '__main__':
    main()
