import os
from datetime import datetime
from pathlib import Path
from tkinter import Tk, filedialog

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'best(1).pt'
DEFAULT_DIR = BASE_DIR
OUTPUT_DIR = BASE_DIR / 'compare_outputs'
IMG_SIZE = 640

# 检测GPU可用性
# 取消 GPU 加速：强制使用 CPU
DEVICE = torch.device('cpu')
print(f'使用设备: {DEVICE}')

# 精细度控制：fast / medium / deep
QUALITY_LEVEL = 'deep'

WEIGHT_COSINE = 0.40
WEIGHT_EUCLIDEAN = 0.20
WEIGHT_MASK_AREA = 0.16
WEIGHT_ASPECT_RATIO = 0.12
WEIGHT_CONTOUR_AREA = 0.12


def choose_two_images(initial_dir=None):
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_paths = filedialog.askopenfilenames(
        title='请选择两张要比对的图片',
        initialdir=str(initial_dir or DEFAULT_DIR),
        filetypes=[('图片文件', '*.jpg *.jpeg *.png *.bmp *.webp'), ('所有文件', '*.*')]
    )
    root.destroy()
    file_paths = list(file_paths)
    if len(file_paths) != 2:
        print('请一次恰好选择两张图片。')
        return None, None
    return file_paths[0], file_paths[1]


def load_model():
    model = YOLO(str(MODEL_PATH))
    model.model.eval()
    model.model.to('cpu')
    return model


def load_image(image_path):
    img_bgr = cv2.imread(image_path)
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
    return cv2.warpPerspective(
        img,
        M,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


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
            pts_orig = pts / scale
            return text, pts_orig, img

    return '', None, img


def rectify_and_read_qr(img_bgr, mask_u8=None):
    """对输入图做二维码检测并拉正；若提供 mask，则用同一矩阵同步拉正 mask。"""
    qr_text, points, _ = detect_qr_robust(img_bgr)
    if points is None:
        return img_bgr, False, '', None, None

    h, w = img_bgr.shape[:2]
    M, (out_w, out_h) = _rectify_transform_from_qr((h, w), points[0])
    rectified = cv2.warpPerspective(
        img_bgr,
        M,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    rectified_mask = None
    if mask_u8 is not None:
        rectified_mask = warp_mask_with_M(mask_u8, M, (out_w, out_h))

    qr_text_after, _, _ = detect_qr_robust(rectified)
    final_text = qr_text_after if qr_text_after else qr_text
    return rectified, True, final_text, points, rectified_mask


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


def save_visualizations(run_dir, image1_path, image2_path, orig1, orig2, rectified1, rectified2, mask1, mask2, rect_mask1, rect_mask2, box1, box2, crop1_xyxy, crop2_xyxy, qr_points1, qr_points2, qr_text1, qr_text2, fitness, decision_note):
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

    cv2.imwrite(str(run_dir / f'vis_original_1_{Path(image1_path).stem}.jpg'), vis_orig1)
    cv2.imwrite(str(run_dir / f'vis_original_2_{Path(image2_path).stem}.jpg'), vis_orig2)
    cv2.imwrite(str(run_dir / f'vis_mask_1_{Path(image1_path).stem}.jpg'), vis_rect1)
    cv2.imwrite(str(run_dir / f'vis_mask_2_{Path(image2_path).stem}.jpg'), vis_rect2)
    cv2.imwrite(str(run_dir / 'vis_compare_original.jpg'), compare_orig)
    cv2.imwrite(str(run_dir / 'vis_compare_rectified.jpg'), compare_rect)
    return compare_orig, compare_rect



def preprocess_cropped_image(img_bgr, img_size=IMG_SIZE):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (img_size, img_size))
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0)
    return tensor  # CPU


def preprocess_cropped_image_multiscale(img_bgr, img_size=IMG_SIZE, scales=[0.85, 0.925, 1.0, 1.075, 1.15]):
    """多尺度预处理 - CPU版本"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    tensors = []
    
    for scale in scales:
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 中心裁剪或填充到目标尺寸
        if new_h < img_size or new_w < img_size:
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
    """提取深度特征 - GPU加速多尺度特征融合"""
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
                        if keep_spatial and len(spatial_tensors) == 0:
                            spatial_tensors.append(obj)
                        if obj.shape[2] >= 4 and obj.shape[3] >= 4:
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
            
            # 限制特征数量
            if len(tensors) > 4:
                tensors = tensors[:4]
            scale_embedding = torch.cat(tuple(tensors), dim=1)
            all_features.append(scale_embedding)
            
            if spatial_tensors:
                spatial_features_list.append(spatial_tensors[0])
    
    # 多尺度特征融合 - 5个尺度的权重
    weights = [0.1, 0.2, 0.4, 0.2, 0.1]
    fused_features = []
    for i, feat in enumerate(all_features):
        w = weights[i] if i < len(weights) else 1.0 / len(all_features)
        fused_features.append(feat * w)
    
    embedding = torch.sum(torch.stack(fused_features), dim=0)
    embedding = F.normalize(embedding, p=2, dim=1)
    
    # 获取局部特征
    local_similarity = None
    if len(spatial_features_list) >= 3:
        # 使用中间尺度的空间特征
        mid_idx = len(spatial_features_list) // 2
        local_similarity = spatial_features_list[mid_idx]
    
    return embedding.squeeze(0).cpu().numpy(), local_similarity


def compute_local_feature_similarity_gpu(spatial_feat1, spatial_feat2):
    """计算局部特征相似度 - GPU加速"""
    if spatial_feat1 is None or spatial_feat2 is None:
        return 0.5
    
    # 确保在GPU上
    if not spatial_feat1.is_cuda:
        spatial_feat1 = spatial_feat1.to(DEVICE)
    if not spatial_feat2.is_cuda:
        spatial_feat2 = spatial_feat2.to(DEVICE)
    
    # 确保空间维度一致
    if spatial_feat1.shape != spatial_feat2.shape:
        target_h = min(spatial_feat1.shape[2], spatial_feat2.shape[2])
        target_w = min(spatial_feat1.shape[3], spatial_feat2.shape[3])
        spatial_feat1 = F.interpolate(spatial_feat1, size=(target_h, target_w), mode='bilinear', align_corners=False)
        spatial_feat2 = F.interpolate(spatial_feat2, size=(target_h, target_w), mode='bilinear', align_corners=False)
    
    # 计算余弦相似度
    feat1_flat = F.normalize(spatial_feat1.flatten(2), p=2, dim=1)
    feat2_flat = F.normalize(spatial_feat2.flatten(2), p=2, dim=1)
    
    similarity = torch.mean(torch.sum(feat1_flat * feat2_flat, dim=1))
    
    local_sim = (similarity.item() + 1) / 2
    return max(0.0, min(1.0, local_sim))


def compute_ssim_fast(img1, img2):
    """快速SSIM - GPU加速版本"""
    import torch
    
    # 转换为灰度并降采样
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1
    
    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2
    
    # 降采样到128x128
    gray1 = cv2.resize(gray1, (128, 128))
    gray2 = cv2.resize(gray2, (128, 128))
    
    # 转移到GPU计算
    tensor1 = torch.from_numpy(gray1.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
    tensor2 = torch.from_numpy(gray2.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        # 使用平均池化模拟均值滤波
        kernel_size = 7
        mu1 = F.avg_pool2d(tensor1, kernel_size, stride=1, padding=kernel_size//2)
        mu2 = F.avg_pool2d(tensor2, kernel_size, stride=1, padding=kernel_size//2)
        
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(tensor1 * tensor1, kernel_size, stride=1, padding=kernel_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(tensor2 * tensor2, kernel_size, stride=1, padding=kernel_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(tensor1 * tensor2, kernel_size, stride=1, padding=kernel_size//2) - mu1_mu2
        
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8)
        
        return float(ssim_map.mean().cpu().item())


def extract_texture_features_fast(img):
    """快速纹理特征提取 - GPU加速"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 降采样
    small = cv2.resize(gray, (64, 64))
    
    # 转移到GPU
    tensor = torch.from_numpy(small.astype(np.float32)).to(DEVICE)
    
    with torch.no_grad():
        # Sobel算子（GPU版本）
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=DEVICE).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=DEVICE).view(1, 1, 3, 3)
        
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        
        sobelx = F.conv2d(tensor, sobel_x, padding=1)
        sobely = F.conv2d(tensor, sobel_y, padding=1)
        
        # 梯度幅值和方向
        magnitude = torch.sqrt(sobelx**2 + sobely**2)
        direction = torch.atan2(sobely, sobelx) * 180 / 3.14159
        
        # 分块统计
        features = []
        blocks = 4
        bh, bw = 64 // blocks, 64 // blocks
        
        for i in range(blocks):
            for j in range(blocks):
                block_mag = magnitude[0, 0, i*bh:(i+1)*bh, j*bw:(j+1)*bw]
                block_dir = direction[0, 0, i*bh:(i+1)*bh, j*bw:(j+1)*bw]
                
                # 梯度幅值均值
                features.append(block_mag.mean().item())
                
                # 梯度方向直方图
                hist = torch.histc(block_dir, bins=8, min=-180, max=180)
                hist = hist / (hist.sum() + 1e-8)
                features.extend(hist.cpu().numpy().tolist())
    
    return np.array(features, dtype=np.float32)


def compute_texture_similarity_fast(img1, img2):
    """快速纹理相似度计算"""
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    feat1 = extract_texture_features_fast(img1)
    feat2 = extract_texture_features_fast(img2)
    
    dot = np.dot(feat1, feat2)
    norm1 = np.linalg.norm(feat1)
    norm2 = np.linalg.norm(feat2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.5
    
    similarity = dot / (norm1 * norm2)
    return (similarity + 1) / 2


def compute_histogram_similarity(img1, img2):
    """计算颜色直方图相似度"""
    # 降采样以加速
    img1_small = cv2.resize(img1, (128, 128))
    img2_small = cv2.resize(img2, (128, 128))
    
    # 转换为HSV颜色空间
    hsv1 = cv2.cvtColor(img1_small, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2_small, cv2.COLOR_BGR2HSV)
    
    # 计算H和S通道的直方图
    hist1_h = cv2.calcHist([hsv1], [0], None, [90], [0, 180])  # 减少bin数量
    hist1_s = cv2.calcHist([hsv1], [1], None, [64], [0, 256])
    hist2_h = cv2.calcHist([hsv2], [0], None, [90], [0, 180])
    hist2_s = cv2.calcHist([hsv2], [1], None, [64], [0, 256])
    
    # 归一化
    cv2.normalize(hist1_h, hist1_h, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist1_s, hist1_s, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2_h, hist2_h, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2_s, hist2_s, 0, 1, cv2.NORM_MINMAX)
    
    # 计算相关性
    corr_h = cv2.compareHist(hist1_h, hist2_h, cv2.HISTCMP_CORREL)
    corr_s = cv2.compareHist(hist1_s, hist2_s, cv2.HISTCMP_CORREL)
    
    return (max(0, corr_h) + max(0, corr_s)) / 2


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
    """构建特征距离 - GPU加速多维度特征"""
    m1 = extract_mask_features(mask1)
    m2 = extract_mask_features(mask2)

    # 基础深度特征相似度
    cosine_sim = cosine_similarity(feat1, feat2)
    euclid_dist = euclidean_distance(feat1, feat2)

    # 局部特征相似度 - GPU加速
    local_sim = compute_local_feature_similarity_gpu(spatial_feat1, spatial_feat2)

    # 结构相似度 (SSIM) - GPU加速
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

    # 纹理相似度 - GPU加速
    texture_sim = 0.5
    if img1 is not None and img2 is not None:
        try:
            texture_sim = compute_texture_similarity_fast(img1, img2)
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
    """构建综合匹配度分数 - 多维度特征融合"""
    qr_score = distances['qr_match_score']
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

    # 多维度权重融合
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


def save_compare_outputs(run_dir, image1_path, image2_path, crop1, crop2, rectified1, rectified2):
    out1 = run_dir / f"crop_1_{Path(image1_path).name}"
    out2 = run_dir / f"crop_2_{Path(image2_path).name}"
    rect1 = run_dir / f"rectified_1_{Path(image1_path).name}"
    rect2 = run_dir / f"rectified_2_{Path(image2_path).name}"
    cv2.imwrite(str(out1), crop1)
    cv2.imwrite(str(out2), crop2)
    cv2.imwrite(str(rect1), rectified1)
    cv2.imwrite(str(rect2), rectified2)
    return out1, out2, rect1, rect2


def save_compare_report(run_dir, image1_path, image2_path, rect1, rect2, out1, out2, distances, fitness, decision_note, qr1, qr2, qr_text1, qr_text2, mask1, mask2):
    report_path = run_dir / 'compare_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('=== 图片比较结果报告 ===\n\n')
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
        f.write(f'二维码一致性分数: {distances.get("qr_match_score", 0):.4f}\n')
        f.write(f'提示: {decision_note}\n')
        f.write(f'主目标分割1: {"成功" if mask1 is not None else "失败"}\n')
        f.write(f'主目标分割2: {"成功" if mask2 is not None else "失败"}\n\n')

        f.write('=== 深度特征 ===\n')
        f.write(f'余弦相似度: {distances.get("cosine_similarity", 0):.4f} (权重10%)\n')
        f.write(f'局部特征相似度: {distances.get("cnn_local_similarity", 0):.4f} (权重10%)\n')
        f.write(f'欧氏距离: {distances.get("euclidean_distance", 0):.4f} -> 转换分: {1.0/(1.0+distances.get("euclidean_distance", 0)):.4f} (权重10%)\n\n')

        f.write('=== 感知特征 ===\n')
        f.write(f'结构相似度(SSIM): {distances.get("ssim", 0):.4f} (权重30%)\n')
        f.write(f'颜色直方图相似度: {distances.get("histogram_similarity", 0):.4f} (权重5%)\n')
        f.write(f'纹理相似度: {distances.get("texture_similarity", 0):.4f} (权重10%)\n\n')

        f.write('=== 几何特征 ===\n')
        f.write(f'主目标面积差: {distances.get("mask_area_diff", "N/A")} (权重10%)\n')
        f.write(f'主目标宽高比差: {distances.get("aspect_ratio_diff", "N/A")} (权重5%)\n')
        f.write(f'轮廓面积差: {distances.get("contour_area_diff", "N/A")} (权重10%)\n\n')

        f.write(f'最终匹配度: {fitness:.4f}\n')
    return report_path


def main():
    if not MODEL_PATH.exists():
        print('模型文件不存在。')
        return

    image1_path, image2_path = choose_two_images(DEFAULT_DIR)
    if not image1_path or not image2_path:
        return

    run_dir = create_run_output_dir()

    try:
        model = load_model()
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

        out1, out2, rect1, rect2 = save_compare_outputs(run_dir, image1_path, image2_path, crop1, crop2, rectified1, rectified2)

        feat1, spatial_feat1 = extract_feature_from_cropped(model, rectified1)
        feat2, spatial_feat2 = extract_feature_from_cropped(model, rectified2)

        distances = build_feature_distances(
            feat1, feat2, rect_mask1, rect_mask2,
            qr_text1, qr_text2, qr1, qr2,
            spatial_feat1, spatial_feat2,
            rectified1, rectified2
        )
        fitness, decision_note = build_fitness_score(distances)

        save_compare_report(run_dir, image1_path, image2_path, rect1, rect2, out1, out2, distances, fitness, decision_note, qr1, qr2, qr_text1, qr_text2, rect_mask1, rect_mask2)
        save_visualizations(run_dir, image1_path, image2_path, img1, img2, rectified1, rectified2, mask1_full, mask2_full, rect_mask1, rect_mask2, box1, box2, crop1_xyxy, crop2_xyxy, qr_points1, qr_points2, qr_text1, qr_text2, fitness, decision_note)

        if decision_note == '二维码识别不清，请重新拍摄':
            print('二维码识别不清，请重新拍摄')
        else:
            print('二维码识别成功')
            print(f'匹配度：{fitness:.4f}')
            print(f'可视化结果已保存到：{run_dir}')

    except Exception:
        print('二维码识别不清，请重新拍摄')


if __name__ == '__main__':
    main()
