"""
贴纸弧形文字 OCR — 极坐标展开 + EasyOCR
=========================================
作为 QR 码解码失败时的备用识别方案。
贴纸顶部弧形文字格式: "0111XXXXXXXXX-N" (N = 试块编号 1/2/3)
"""

import os
import re
import json
import cv2
import numpy as np
from pathlib import Path

# 懒加载 EasyOCR（首次调用时初始化，约3-5秒）
_reader = None

# OCR 磁盘缓存（与 qr_cache.json 分开，避免冲突）
_OCR_CACHE: dict = {}
_OCR_CACHE_FILE = Path(__file__).parent.parent / "output_v2" / "ocr_cache.json"


def load_ocr_cache():
    global _OCR_CACHE
    if _OCR_CACHE_FILE.exists():
        with open(_OCR_CACHE_FILE, 'r', encoding='utf-8') as f:
            _OCR_CACHE = json.load(f)


def save_ocr_cache():
    _OCR_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_OCR_CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(_OCR_CACHE, f, ensure_ascii=False)


def _get_reader():
    global _reader
    if _reader is None:
        os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
        import easyocr
        _reader = easyocr.Reader(['en'], gpu=True, verbose=False)
    return _reader


def _arc_unwrap(img_bgr, cx, cy, r):
    """极坐标展开: 贴纸顶部弧 (200°~340°, 外圈在上)"""
    # 上采样确保贴纸半径 ≥ 400px（OCR清晰度要求）
    if r < 400:
        s = 400.0 / r
        img_bgr = cv2.resize(img_bgr, (int(img_bgr.shape[1] * s), int(img_bgr.shape[0] * s)),
                             interpolation=cv2.INTER_CUBIC)
        cx, cy, r = int(cx * s), int(cy * s), int(r * s)

    r_inner = int(r * 0.60)
    r_outer = int(r * 0.82)
    N_STEPS, N_ROWS = 1200, r_outer - r_inner

    angles = np.linspace(np.deg2rad(200), np.deg2rad(340), N_STEPS)
    radii  = np.linspace(r_outer, r_inner, N_ROWS)
    ang_grid, r_grid = np.meshgrid(angles, radii)
    map_x = (cx + r_grid * np.cos(ang_grid)).astype(np.float32)
    map_y = (cy + r_grid * np.sin(ang_grid)).astype(np.float32)

    unwrapped = cv2.remap(img_bgr, map_x, map_y, cv2.INTER_LANCZOS4)

    # 锐化
    blur = cv2.GaussianBlur(unwrapped, (0, 0), 1.0)
    return cv2.addWeighted(unwrapped, 2.5, blur, -1.5, 0)


def _enhance_for_ocr(arc_img):
    """B-R 差值增强（蓝底灰字 → 高对比灰度图），缩到字符高度 ~80px"""
    b = arc_img[:, :, 0].astype(np.int16)
    r = arc_img[:, :, 2].astype(np.int16)
    diff = np.clip(255 - (b - r), 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(8, 4))
    diff = clahe.apply(diff)

    h, w = diff.shape
    target_h = 80
    return cv2.resize(diff, (int(w * target_h / h), target_h), interpolation=cv2.INTER_AREA)


def _fix_chars(text: str) -> str:
    """OCR 常见误识别纠正"""
    table = str.maketrans('oOlIiBbSsqQZzDdUu', '00111155509909000')
    cleaned = text.translate(table)
    return re.sub(r'[^0-9\-]', '', cleaned)


def ocr_sticker_text(img_bgr, cx, cy, r):
    """
    对贴纸图像进行弧形文字识别。
    返回: 识别文本字符串（如 "0111250005334-2"）或 None
    """
    try:
        arc = _arc_unwrap(img_bgr, cx, cy, r)
        small = _enhance_for_ocr(arc)

        reader = _get_reader()
        results = reader.readtext(small, detail=1, text_threshold=0.3, low_text=0.2)
        if not results:
            return None

        texts = []
        for box, t, c in results:
            clean = _fix_chars(t)
            if clean:
                x_left = min(p[0] for p in box)
                texts.append((x_left, clean, c))

        texts.sort(key=lambda x: x[0])
        joined = ''.join(t for _, t, _ in texts)

        # 尝试匹配标准格式 (10~13位数字)-N
        m = re.search(r'(\d{10,13})-?(\d)', joined)
        if m:
            return f"{m.group(1)}-{m.group(2)}"

        return joined if len(joined) > 5 else None

    except Exception as e:
        return None


def ocr_specimen_id(img_bgr, filepath=None):
    """
    完整 OCR 流程: 检测贴纸 → 弧形展开 → OCR → 提取试块编号 (1/2/3)。
    filepath 可选，用于磁盘缓存。
    返回: 试块编号字符串 (如 '2') 或 None
    """
    cache_key = str(filepath) if filepath else None
    if cache_key and cache_key in _OCR_CACHE:
        return _OCR_CACHE[cache_key]

    def _store(val):
        if cache_key:
            _OCR_CACHE[cache_key] = val
        return val

    try:
        from .sticker import detect_blue_sticker
        _, center, ellipse = detect_blue_sticker(img_bgr)
        if center is None or ellipse is None:
            return _store(None)

        cx, cy = center
        r = max(ellipse[1]) / 2

        text = ocr_sticker_text(img_bgr, cx, cy, r)
        if not text:
            return _store(None)

        m = re.search(r'-(\d)$', text)
        result = m.group(1) if m else None
        return _store(result)

    except Exception:
        return _store(None)
