"""
混凝土试块造假识别 Pipeline — QR 码检测
========================================
QR 码方向检测 + 内容解码
"""

import json
import cv2
import numpy as np
from pathlib import Path

# 磁盘QR缓存 — key=图片绝对路径, value=解码内容(str或null)
_QR_DISK_CACHE = {}
_QR_CACHE_FILE = Path(__file__).parent.parent / "output_v2" / "qr_cache.json"


def load_qr_cache():
    """从磁盘加载QR缓存，程序启动时调用一次"""
    global _QR_DISK_CACHE
    if _QR_CACHE_FILE.exists():
        with open(_QR_CACHE_FILE, 'r', encoding='utf-8') as f:
            _QR_DISK_CACHE = json.load(f)


def save_qr_cache():
    """将QR缓存写回磁盘"""
    _QR_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_QR_CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(_QR_DISK_CACHE, f, ensure_ascii=False)


def _detect_finder_patterns(gray):
    """检测 QR 码的 3 个 finder pattern（定位图案）。
    Finder pattern 特征: 嵌套正方形，黑白比例 1:1:3:1:1。
    返回: [(cx, cy), ...] 最多 3 个中心点，按面积从大到小。
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []

    hier = hierarchy[0]  # [next, prev, child, parent]
    finder_centers = []

    for i, cnt in enumerate(contours):
        # finder pattern 有 >= 2 层嵌套子轮廓
        child = hier[i][2]
        if child < 0:
            continue
        grandchild = hier[child][2]
        if grandchild < 0:
            continue

        # 检查外层轮廓是否近似正方形
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        ar = min(w, h) / max(w, h) if max(w, h) > 0 else 0
        if ar < 0.7:
            continue

        # 检查面积比例: 外/中/内 ≈ 49:25:9 (7²:5²:3²)
        inner_area = cv2.contourArea(contours[child])
        if inner_area < 10:
            continue
        ratio = area / inner_area
        if 1.2 < ratio < 6.0:
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                finder_centers.append((cx, cy, area))

    # 去重：合并距离过近的中心点
    merged = []
    for cx, cy, a in sorted(finder_centers, key=lambda x: -x[2]):
        too_close = False
        for mx, my, _ in merged:
            if np.sqrt((cx-mx)**2 + (cy-my)**2) < np.sqrt(a) * 0.5:
                too_close = True
                break
        if not too_close:
            merged.append((cx, cy, a))
    return [(x, y) for x, y, _ in merged[:3]]


def _identify_finder_corners(gray, qr_pts):
    """从 cv2 检测的 4 个 QR 角点中识别 3 个 finder pattern 角。
    Finder pattern 角有更多高对比度结构（嵌套方框）。
    返回: 3 个 finder 角点的中心 [(x,y), ...] 或 None。
    """
    h, w = gray.shape[:2]
    # QR 尺寸估算
    side = max(np.linalg.norm(qr_pts[1] - qr_pts[0]),
               np.linalg.norm(qr_pts[2] - qr_pts[1]))
    patch_r = int(side * 0.25)  # finder pattern 约占 QR 宽度的 ~1/4

    scores = []
    for pt in qr_pts:
        px, py = int(pt[0]), int(pt[1])
        x1 = max(0, px - patch_r)
        y1 = max(0, py - patch_r)
        x2 = min(w, px + patch_r)
        y2 = min(h, py + patch_r)
        patch = gray[y1:y2, x1:x2]
        if patch.size == 0:
            scores.append(0)
            continue
        # Finder pattern 有很高的局部对比度（黑白交替方框）
        # 用标准差 + 梯度幅值衡量
        std_val = float(np.std(patch))
        sx = cv2.Sobel(patch.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(patch.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        grad_mean = float(np.mean(np.sqrt(sx**2 + sy**2)))
        scores.append(std_val + grad_mean)

    # 得分最低的角没有 finder pattern（bottom-right）
    min_idx = int(np.argmin(scores))
    finder_corners = [qr_pts[i] for i in range(4) if i != min_idx]
    return finder_corners


def _qr_angle_from_finder_patterns(centers):
    """从 3 个 finder pattern 中心计算 QR 方向角。
    QR 布局:  TL --- TR
              |
              BL    (no pattern)
    三个 finder 中，直角顶点是 TL。
    返回: top-left → top-right 边的角度（度）。
    """
    if len(centers) != 3:
        return None
    pts = np.array(centers, dtype=np.float64)

    # 找直角顶点（TL）: 到其他两点距离之和最小的点
    # 或者: 三角形中，与其他两点距离的平方和 ≈ 第三边距离的平方（勾股定理）
    dists = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            dists[i, j] = np.sqrt(((pts[i] - pts[j])**2).sum())

    # TL 是使得 d(TL,TR)² + d(TL,BL)² ≈ d(TR,BL)² 的那个点
    best_tl = 0
    best_err = float('inf')
    for i in range(3):
        j, k = [x for x in range(3) if x != i]
        err = abs(dists[i, j]**2 + dists[i, k]**2 - dists[j, k]**2)
        if err < best_err:
            best_err = err
            best_tl = i

    tl = pts[best_tl]
    others = [pts[x] for x in range(3) if x != best_tl]

    # TR 和 BL 区分: cross product 判断方向
    # TL→TR 和 TL→BL 构成右手坐标系（图像坐标中 cross > 0 表示顺时针）
    v0 = others[0] - tl
    v1 = others[1] - tl
    cross = v0[0] * v1[1] - v0[1] * v1[0]
    if cross > 0:
        tr, bl = others[0], others[1]
    else:
        tr, bl = others[1], others[0]

    # TL → TR 方向角
    dx, dy = tr[0] - tl[0], tr[1] - tl[1]
    return np.degrees(np.arctan2(dy, dx))


def _qr_angle_from_pts(pts):
    """从 QR 4 角点计算方向角 (top-left → top-right 边的角度)"""
    dx = pts[1][0] - pts[0][0]
    dy = pts[1][1] - pts[0][1]
    return np.degrees(np.arctan2(dy, dx))


def detect_qr_angle(img_bgr):
    """
    检测 QR 码方向角度，用于精确旋转对齐。
    依次尝试多种预处理方式 + 多种检测器（QRCodeDetector → Aruco → pyzbar）。
    返回: (angle_degrees, qr_corners_4x2) 或 (None, None)
    """
    det = cv2.QRCodeDetector()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_strong = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4, 4))
    sharp_k = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])

    # === Phase 1: cv2.QRCodeDetector 原始分辨率 ===
    preprocess_list = [
        img_bgr,
        cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR),
        cv2.filter2D(img_bgr, -1, sharp_k),
        cv2.cvtColor(clahe_strong.apply(gray), cv2.COLOR_GRAY2BGR),
    ]
    for prep in preprocess_list:
        ok, pts = det.detect(prep)
        if ok and pts is not None:
            return _qr_angle_from_pts(pts[0]), pts[0].copy()

    # === Phase 2: cv2.QRCodeDetector 放大 2× ===
    big = cv2.resize(img_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    big_gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    preprocess_big = [
        big,
        cv2.cvtColor(clahe.apply(big_gray), cv2.COLOR_GRAY2BGR),
        cv2.filter2D(big, -1, sharp_k),
        cv2.cvtColor(clahe_strong.apply(big_gray), cv2.COLOR_GRAY2BGR),
    ]
    for prep in preprocess_big:
        ok, pts = det.detect(prep)
        if ok and pts is not None:
            return _qr_angle_from_pts(pts[0]), (pts[0] / 2.0).copy()

    # === Phase 3: cv2.QRCodeDetector 放大 3× + clahe_strong ===
    big3 = cv2.resize(img_bgr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    big3_gray = cv2.cvtColor(big3, cv2.COLOR_BGR2GRAY)
    prep_3x = cv2.cvtColor(clahe_strong.apply(big3_gray), cv2.COLOR_GRAY2BGR)
    ok, pts = det.detect(prep_3x)
    if ok and pts is not None:
        return _qr_angle_from_pts(pts[0]), (pts[0] / 3.0).copy()

    # === Phase 4: QRCodeDetectorAruco (更鲁棒，同角点序) ===
    det_aruco = cv2.QRCodeDetectorAruco()
    for prep in [img_bgr, big]:
        scale = 1.0 if prep is img_bgr else 2.0
        ok, pts = det_aruco.detect(prep)
        if ok and pts is not None:
            return _qr_angle_from_pts(pts[0]), (pts[0] / scale).copy()

    # === Phase 5: pyzbar 放大 2× (最后备选，角点序不同需要转换) ===
    try:
        from pyzbar import pyzbar as pzb
        codes = pzb.decode(big)
        qr_codes = [c for c in codes if c.type == 'QRCODE']
        if qr_codes:
            poly = qr_codes[0].polygon
            if len(poly) >= 4:
                # pyzbar 角点序: [TL-ish, BL-ish, BR-ish, TR-ish]
                # cv2 角点序:   [TL, TR, BR, BL]
                # pyzbar[0]≈cv2[1], pyzbar[1]≈cv2[0], pyzbar[2]≈cv2[3], pyzbar[3]≈cv2[2]
                pz_pts = np.array([(p.x / 2.0, p.y / 2.0) for p in poly[:4]])
                # 重排为 cv2 顺序: [pzb[1], pzb[0], pzb[3], pzb[2]]
                cv2_order = np.array([pz_pts[1], pz_pts[0], pz_pts[3], pz_pts[2]])
                return _qr_angle_from_pts(cv2_order), cv2_order.copy()
    except ImportError:
        pass

    return None, None


def decode_qr_content(img_bgr, filepath=None):
    """解码QR码内容（文本）。尝试多种预处理+检测器。
    filepath: 可选，传入图片路径可启用磁盘缓存，避免重复解码。
    返回: str 或 None"""
    # 磁盘缓存命中
    cache_key = str(filepath) if filepath else None
    if cache_key and cache_key in _QR_DISK_CACHE:
        return _QR_DISK_CACHE[cache_key]

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe_strong = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4, 4))
    det = cv2.QRCodeDetector()

    def _hit(val):
        if cache_key:
            _QR_DISK_CACHE[cache_key] = val
        return val

    # Phase 1: 原图 + clahe
    for prep in [img_bgr, cv2.cvtColor(clahe_strong.apply(gray), cv2.COLOR_GRAY2BGR)]:
        val, pts, _ = det.detectAndDecode(prep)
        if val and val.strip():
            return _hit(val.strip())

    # Phase 2: 2x放大
    big = cv2.resize(img_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    big_gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    for prep in [big, cv2.cvtColor(clahe_strong.apply(big_gray), cv2.COLOR_GRAY2BGR)]:
        val, pts, _ = det.detectAndDecode(prep)
        if val and val.strip():
            return _hit(val.strip())

    # Phase 3: Aruco
    det_aruco = cv2.QRCodeDetectorAruco()
    for prep in [img_bgr, big]:
        val, pts, _ = det_aruco.detectAndDecode(prep)
        if val and val.strip():
            return _hit(val.strip())

    # Phase 4: pyzbar
    try:
        from pyzbar import pyzbar as pzb
        codes = pzb.decode(big)
        qr_codes = [c for c in codes if c.type == 'QRCODE']
        if qr_codes:
            result = qr_codes[0].data.decode('utf-8', errors='replace').strip()
            if cache_key:
                _QR_DISK_CACHE[cache_key] = result
            return result
    except ImportError:
        pass

    if cache_key:
        _QR_DISK_CACHE[cache_key] = None
    return None
