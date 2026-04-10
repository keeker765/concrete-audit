"""
混凝土试块造假识别 Pipeline — 配置与工具
=========================================
imports, constants, async save, SAM singleton, rotation utils
"""

import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import torch
from pathlib import Path

# segment_anything 仅在 _get_sam_predictor 内部惰性加载

# ============================================================
# 配置
# ============================================================
DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

_PROJECT_ROOT = Path(__file__).parent.parent
SAM_CHECKPOINT = Path(os.environ.get('SAM_CHECKPOINT', str(_PROJECT_ROOT / "models" / "sam_vit_b_01ec64.pth")))
SAMPLES_DIR = _PROJECT_ROOT / "samples"
DATA_DIR    = _PROJECT_ROOT / "data"
OUTPUT_ROOT = _PROJECT_ROOT / "output_v2"
OUTPUT_ROOT.mkdir(exist_ok=True)

MAX_KEYPOINTS = 2048
MATCH_THRESHOLD = 0.46
MIN_MATCHES = 20
SCORE_W_CONF = 0.5       # confidence 权重
SCORE_W_IR   = 0.5       # inlier_ratio 权重
ROI_SCALE = 2.5          # ROI 半径 = 贴纸半径 × ROI_SCALE
MAX_SIZE_WET = 1024      # 湿态图最大边
MAX_SIZE_DRY = 1536      # 干态图最大边（分辨率更高，因 ROI 更小）

# 异步图片保存线程池
_save_pool = ThreadPoolExecutor(max_workers=4)
_save_futures = []

def _async_save(path, img):
    """异步保存图片，不阻塞主流程"""
    fut = _save_pool.submit(cv2.imwrite, str(path), img)
    _save_futures.append(fut)

def _wait_saves():
    """等待所有异步保存完成"""
    for f in _save_futures:
        f.result()
    _save_futures.clear()

print(f"Device: {DEVICE}")
print(f"Output:  {OUTPUT_ROOT}")

# SAM 模型单例（惰性加载）
_sam_predictor = None

def _get_sam_predictor():
    """惰性加载 SAM 模型，全局复用"""
    global _sam_predictor
    if _sam_predictor is None:
        from segment_anything import sam_model_registry, SamPredictor  # 惰性导入，不用SAM时不加载
        print("[SAM] Loading vit_b model...")
        sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
        sam.to(DEVICE)
        _sam_predictor = SamPredictor(sam)
        print("[SAM] Model loaded.")
    return _sam_predictor


# Grounding DINO 单例（惰性加载）
GDINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
_gdino_cache = None

def _get_gdino():
    """惰性加载 Grounding DINO 模型，全局复用。返回 (processor, model)"""
    global _gdino_cache
    if _gdino_cache is None:
        import time
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        print(f"[DINO] Loading {GDINO_MODEL_ID}...")
        t0 = time.time()
        processor = AutoProcessor.from_pretrained(GDINO_MODEL_ID)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL_ID).to(DEVICE)
        model.eval()
        print(f"[DINO] Model loaded in {time.time()-t0:.1f}s")
        _gdino_cache = (processor, model)
    return _gdino_cache


def get_output_dir(method_name):
    """按方法名组织输出子目录"""
    d = OUTPUT_ROOT / method_name
    d.mkdir(exist_ok=True)
    return d

# ============================================================
# 旋转工具
# ============================================================
_CV2_ROT = {
    1: cv2.ROTATE_90_COUNTERCLOCKWISE,
    2: cv2.ROTATE_180,
    3: cv2.ROTATE_90_CLOCKWISE,
}

def rotate_cv2(img, rot_k):
    """旋转 BGR 图像 rot_k×90° 逆时针"""
    if rot_k == 0:
        return img
    return cv2.rotate(img, _CV2_ROT[rot_k])


def rotate_by_angle(img_bgr, angle_deg):
    """以图像中心为原点旋转任意角度，扩大画布保留完整内容，边缘填充黑色"""
    h, w = img_bgr.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    # 计算旋转后的新尺寸（包含完整内容）
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    # 调整平移，使旋转中心仍在新画布中心
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(img_bgr, M, (new_w, new_h),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
