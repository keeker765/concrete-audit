"""配对解析 + 主匹配流程 + 跨批次测试"""
import os
import re
import time
import cv2
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

from lightglue.utils import rbd

from .config import (DEVICE, OUTPUT_ROOT, MIN_MATCHES, MATCH_THRESHOLD,
                     MAX_SIZE_WET, MAX_SIZE_DRY, MAX_KEYPOINTS,
                     get_output_dir, _async_save, _wait_saves, rotate_cv2)
from .qr import decode_qr_content, load_qr_cache, save_qr_cache
from .ocr_fallback import ocr_specimen_id, load_ocr_cache, save_ocr_cache

# ============================================================
# 手动配对缓存 (manual_pairs.json)
# ============================================================
import json as _json
_MANUAL_PAIRS: dict = {}
_MANUAL_PAIRS_FILE = Path(__file__).parent.parent / "output_v2" / "manual_pairs.json"


def load_manual_pairs():
    global _MANUAL_PAIRS
    if _MANUAL_PAIRS_FILE.exists():
        with open(_MANUAL_PAIRS_FILE, 'r', encoding='utf-8') as f:
            _MANUAL_PAIRS = _json.load(f)


def save_manual_pairs():
    _MANUAL_PAIRS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_MANUAL_PAIRS_FILE, 'w', encoding='utf-8') as f:
        _json.dump(_MANUAL_PAIRS, f, ensure_ascii=False, indent=2)
from .preprocess import load_and_preprocess
from .align import align_pair
from .matching import filter_matches, compute_score, filter_points_by_mask, compute_score_from_points
from .visualization import draw_matches_vis, generate_heatmap, _generate_detection_debug


# ============================================================
# 解析配对
# ============================================================

def parse_sample_pairs(samples_dir):
    """从 samples/ 目录解析干湿配对，优先用QR码配对，失败则按文件名回退。"""
    wet_files = sorted(samples_dir.glob("*_wet*.jpeg"))
    dry_files = sorted(samples_dir.glob("*_dry*.jpg"))

    batches = defaultdict(lambda: {'wet': [], 'dry': []})
    for f in wet_files:
        parts = f.stem.rsplit('_wet', 1)
        batches[parts[0]]['wet'].append(f)
    for f in dry_files:
        parts = f.stem.rsplit('_dry', 1)
        batches[parts[0]]['dry'].append(f)

    pairs = []
    for batch_id, files in batches.items():
        wet_sorted = sorted(files['wet'])
        dry_sorted = sorted(files['dry'])

        def _get_specimen_id(filepath):
            img = cv2.imread(str(filepath))
            if img is None:
                return None
            content = decode_qr_content(img, filepath=filepath)
            if content is not None:
                m2 = re.search(r'-(\d+)$', content)
                return m2.group(1) if m2 else None
            # QR 失败 → 贴纸弧形 OCR 备用
            return ocr_specimen_id(img, filepath=filepath)

        wet_ids = {f: _get_specimen_id(f) for f in wet_sorted}
        dry_ids = {f: _get_specimen_id(f) for f in dry_sorted}

        # 按QR编号配对
        dry_by_id = {}
        for f, sid in dry_ids.items():
            if sid is not None:
                dry_by_id[sid] = f

        paired_dry = set()
        qr_pairs = []
        for wf in wet_sorted:
            wid = wet_ids[wf]
            if wid is not None and wid in dry_by_id:
                qr_pairs.append((wf, dry_by_id[wid]))
                paired_dry.add(dry_by_id[wid])

        if len(qr_pairs) >= len(wet_sorted):
            # 全部QR配对成功
            print(f"  {batch_id}: QR配对成功 ({len(qr_pairs)}对)")
            for i, (w, d) in enumerate(qr_pairs):
                pairs.append((batch_id, w, d, i + 1))
        else:
            # QR配对不完整，回退到文件名顺序
            if qr_pairs:
                print(f"  {batch_id}: QR仅配对{len(qr_pairs)}对，回退文件名顺序")
            for i, (w, d) in enumerate(zip(wet_sorted, dry_sorted)):
                pairs.append((batch_id, w, d, i + 1))

    return pairs


def parse_data_pairs(data_dir):
    """从 data/ 目录结构解析干湿配对。
    通过解码QR码内容获取试块编号（如 0111250005329-3 → 编号3），
    按编号匹配同一批次的干湿图片。
    QR解码失败的图片标记为INVALID。
    """
    pairs = []
    company_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    for company_dir in company_dirs:
        short_name = company_dir.name[:4]
        batch_dirs = sorted([d for d in company_dir.iterdir() if d.is_dir()])
        for batch_dir in batch_dirs:
            m = re.search(r'压-(.+)$', batch_dir.name)
            batch_num = m.group(1) if m else batch_dir.name
            batch_id = f"{short_name}_{batch_num}"

            wet_files = sorted(batch_dir.glob("*.jpeg"))
            dry_files = sorted(batch_dir.glob("*.jpg"))
            if len(wet_files) != 3 or len(dry_files) != 3:
                print(f"  ⚠️ 跳过 {batch_id}: jpeg={len(wet_files)}, jpg={len(dry_files)}")
                continue

            def _get_specimen_id(filepath):
                img = cv2.imread(str(filepath))
                if img is None:
                    return None
                content = decode_qr_content(img, filepath=filepath)
                if content is not None:
                    m2 = re.search(r'-(\d+)$', content)
                    return m2.group(1) if m2 else content
                # QR 失败 → 手动配对表
                key = str(filepath)
                if key in _MANUAL_PAIRS:
                    return _MANUAL_PAIRS[key]
                # 最后备用: 贴纸弧形 OCR
                return ocr_specimen_id(img, filepath=filepath)

            wet_ids = {f: _get_specimen_id(f) for f in wet_files}
            dry_ids = {f: _get_specimen_id(f) for f in dry_files}

            print(f"  {batch_id}: WET={[wet_ids[f] for f in wet_files]}, "
                  f"DRY={[dry_ids[f] for f in dry_files]}")

            # 按QR编号配对
            dry_by_id = {}
            for f, sid in dry_ids.items():
                if sid is not None:
                    dry_by_id[sid] = f

            specimen_no = 0
            for wf in wet_files:
                specimen_no += 1
                wid = wet_ids[wf]
                if wid is not None and wid in dry_by_id:
                    pairs.append((batch_id, wf, dry_by_id[wid], specimen_no))
                else:
                    pairs.append((batch_id, wf, None, specimen_no))

    return pairs


# ============================================================
# 辅助函数
# ============================================================

def _ellipse_from_mask(mask):
    """从 mask 轮廓拟合椭圆，返回 ellipse 或 None"""
    if mask is None:
        return None
    mask_u8 = mask if mask.dtype == np.uint8 else mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if len(c) >= 5:
        return cv2.fitEllipse(c)
    return None


def _make_tensor(img_bgr):
    """从 BGR 图像创建 CLAHE 增强后的 float32 tensor"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enh3 = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return torch.from_numpy(enh3).permute(2, 0, 1).float() / 255.0


# ============================================================
# 旋转对齐 + 主匹配流程
# ============================================================

def run_method(method_name, extractor, matcher, pairs, roi_mode='dino', rectify=False):
    """对所有配对跑一种方法，返回结果列表。"""
    results = []
    out_dir = get_output_dir(method_name)

    for batch_id, wet_path, dry_path, specimen_no in pairs:
        label = f"{batch_id}#{specimen_no}"
        print(f"\n--- {method_name} | {label} ---")

        # QR配对失败（dry_path=None）→ INVALID
        if dry_path is None:
            print(f"  INVALID: QR配对失败（未找到匹配的DRY图）")
            results.append({
                'batch': batch_id, 'specimen': specimen_no,
                'method': method_name,
                'raw_matches': 0, 'filtered_matches': 0,
                'inliers': 0, 'mean_conf': 0, 'inlier_ratio': 0,
                'final_score': 0, 'verdict': 'INVALID',
                'best_rotation': 0, 'rot_method': 'N/A',
                'sticker_wet': False, 'sticker_dry': False,
                'time_sec': 0, 'invalid_reason': 'QR配对失败',
                'wet_path': str(wet_path), 'dry_path': str(dry_path) if dry_path else None,
            })
            continue

        try:
            img0_bgr, tensor0, mask0, center0, stk0, meta0 = load_and_preprocess(
                wet_path, max_size=MAX_SIZE_WET, roi_mode=roi_mode, rectify=rectify)
            img1_bgr, tensor1, mask1, center1, stk1, meta1 = load_and_preprocess(
                dry_path, max_size=MAX_SIZE_DRY, roi_mode=roi_mode, rectify=rectify)

            # INVALID: 贴纸检测失败 → 跳过匹配（不再检查 QR 角度）
            invalid_reason = None
            if not stk0:
                invalid_reason = "WET贴纸未检测到"
            elif not stk1:
                invalid_reason = "DRY贴纸未检测到"

            if invalid_reason:
                print(f"  INVALID: {invalid_reason}")
                results.append({
                    'batch': batch_id, 'specimen': specimen_no,
                    'method': method_name,
                    'raw_matches': 0, 'filtered_matches': 0,
                    'inliers': 0, 'mean_conf': 0, 'inlier_ratio': 0,
                    'final_score': 0, 'verdict': 'INVALID',
                    'best_rotation': 0, 'rot_method': 'N/A',
                    'sticker_wet': stk0, 'sticker_dry': stk1,
                    'time_sec': 0, 'invalid_reason': invalid_reason,
                    'wet_path': str(wet_path), 'dry_path': str(dry_path) if dry_path else None,
                })
                continue

            t_start = time.time()

            # --- 新对齐策略：矩形摆正 + 贴纸象限匹配 ---
            ell0 = _ellipse_from_mask(mask0)
            ell1 = _ellipse_from_mask(mask1)
            align_result = align_pair(
                img0_bgr, img1_bgr,
                center0, center1,
                mask_wet=mask0, mask_dry=mask1,
                ellipse_wet=ell0, ellipse_dry=ell1,
                sam_mask_wet=meta0.get('sam_mask_processed'),
                sam_mask_dry=meta1.get('sam_mask_processed'),
            )

            img_wet_a = align_result['img_wet']
            img_dry_a = align_result['img_dry']
            mask_wet_a = align_result['mask_wet']
            mask_dry_a = align_result['mask_dry']

            # 确保 mask 不为 None
            if mask_wet_a is None:
                mask_wet_a = np.zeros(img_wet_a.shape[:2], dtype=np.uint8)
            if mask_dry_a is None:
                mask_dry_a = np.zeros(img_dry_a.shape[:2], dtype=np.uint8)

            # 尺度归一化：若两张 ROI 面积比 > 1.5，将较大图等比缩放
            # 保持宽高比，避免拉伸导致圆变椭圆
            def _normalize_scale(img_wet, img_dry, m_wet, m_dry):
                h0, w0 = img_wet.shape[:2]
                h1, w1 = img_dry.shape[:2]
                area_ratio = (h0 * w0) / max(h1 * w1, 1)
                if area_ratio > 1.5:
                    # 湿态图更大 → 等比缩放到干态面积级别
                    scale = (h1 * w1 / (h0 * w0)) ** 0.5
                    new_w, new_h = int(w0 * scale), int(h0 * scale)
                    img_wet = cv2.resize(img_wet, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    if m_wet is not None and m_wet.size > 0:
                        m_wet = cv2.resize(m_wet, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                    print(f"    [尺度归一化] WET {w0}×{h0} → {new_w}×{new_h} (ratio={area_ratio:.2f})")
                elif area_ratio < 1 / 1.5:
                    # 干态图更大 → 等比缩放到湿态面积级别
                    scale = (h0 * w0 / (h1 * w1)) ** 0.5
                    new_w, new_h = int(w1 * scale), int(h1 * scale)
                    img_dry = cv2.resize(img_dry, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    if m_dry is not None and m_dry.size > 0:
                        m_dry = cv2.resize(m_dry, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                    print(f"    [尺度归一化] DRY {w1}×{h1} → {new_w}×{new_h} (ratio={area_ratio:.2f})")
                return img_wet, img_dry, m_wet, m_dry

            img_wet_a, img_dry_a, mask_wet_a, mask_dry_a = _normalize_scale(
                img_wet_a, img_dry_a, mask_wet_a, mask_dry_a)

            # 干态特征提取（只做一次）
            tensor_dry = _make_tensor(img_dry_a)
            feats1 = extractor.extract(tensor_dry.to(DEVICE))

            best_score = -1
            best_rot = 0.0
            best = None

            n_cands = len(align_result['candidates'])
            print(f"    [{align_result['method']}] {n_cands} candidate(s)")

            for rot_k, rot_deg in align_result['candidates']:
                # 旋转湿态图像
                if rot_k > 0:
                    img_r = rotate_cv2(img_wet_a, rot_k)
                    m_r = np.ascontiguousarray(np.rot90(mask_wet_a, rot_k))
                else:
                    img_r = img_wet_a
                    m_r = mask_wet_a

                # 特征提取 + 匹配
                tensor_wet = _make_tensor(img_r)
                feats0 = extractor.extract(tensor_wet.to(DEVICE))
                result = matcher({'image0': feats0, 'image1': feats1})
                f0, f1, res = [rbd(x) for x in [feats0, feats1, result]]

                matches_raw = res['matches']
                scores_raw = res.get('scores', None)
                kpts0, kpts1 = f0['keypoints'], f1['keypoints']

                matches_f, scores_f = filter_matches(
                    kpts0, kpts1, matches_raw, scores_raw, m_r, mask_dry_a)
                sc, cf, ir, inl, n_scored = compute_score(
                    kpts0, kpts1, matches_f, scores_f)

                if sc > best_score:
                    best_score = sc
                    best_rot = rot_deg
                    best = {
                        'kpts0': kpts0, 'kpts1': kpts1,
                        'matches': matches_f, 'scores': scores_f,
                        'inliers': inl, 'conf': cf, 'inlier_ratio': ir,
                        'n_raw': len(matches_raw), 'n_filtered': len(matches_f),
                        'mask0': m_r, 'rot_method': align_result['method'],
                        'img0_vis': img_r,
                    }

            elapsed = time.time() - t_start
            final_score = best_score
            rot_deg = best_rot
            rot_method = best['rot_method']
            n_matches = best['n_filtered']
            n_inliers = int(best['inliers'].sum()) if len(best['inliers']) > 0 else 0

            print(f"    → 选择 {rot_deg:.1f}° ({rot_method}, "
                  f"score={final_score:.3f})")

            if n_matches < MIN_MATCHES:
                verdict = "INSUFFICIENT"
            elif final_score > MATCH_THRESHOLD:
                verdict = "SAME"
            else:
                verdict = "DIFFERENT"

            print(f"  Raw: {best['n_raw']} -> Filtered: {n_matches} -> "
                  f"Inliers: {n_inliers}")
            print(f"  Conf: {best['conf']:.3f} | InlierR: {best['inlier_ratio']:.3f} | "
                  f"Score: {final_score:.3f} | {verdict} | "
                  f"rot={rot_deg:.1f}° ({rot_method})")
            print(f"  Time: {elapsed:.2f}s")

            # 可视化
            img0_vis = best['img0_vis']
            kpts0, kpts1 = best['kpts0'], best['kpts1']
            matches_f, scores_f = best['matches'], best['scores']

            # 每批次一个子目录
            batch_dir = out_dir / batch_id
            batch_dir.mkdir(exist_ok=True)

            if n_matches >= 4:
                vis_title = (f"{method_name} | #{specimen_no} | "
                             f"Score={final_score:.2f} | {verdict} | "
                             f"rot={rot_deg:.1f}({rot_method})")
                vis_path = batch_dir / f"match_{specimen_no}.png"
                draw_matches_vis(img0_vis, img_dry_a, kpts0, kpts1,
                                 matches_f, best['inliers'],
                                 best['mask0'], mask_dry_a, vis_title, vis_path,
                                 rot_deg=rot_deg,
                                 meta0=meta0, meta1=meta1)
                print(f"  Saved: {vis_path.relative_to(OUTPUT_ROOT)}")

                hm_path = batch_dir / f"heatmap_{specimen_no}.png"
                mk = kpts0[matches_f[:, 0]].detach().cpu().numpy()
                mc = (scores_f.detach().cpu().float().numpy()
                      if scores_f is not None else np.ones(len(mk)))
                generate_heatmap(img0_vis, mk, mc, hm_path)

            # 生成 ROI 对比图 + 中间过程图 → debug子目录
            debug_dir = batch_dir / 'debug'
            debug_dir.mkdir(exist_ok=True)
            for tag, meta, img_path_cur in [('wet', meta0, wet_path),
                                             ('dry', meta1, dry_path)]:
                pre_img = meta.get('img_pre_roi')
                bounds = meta.get('roi_bounds')
                ctr = meta.get('center_orig')
                ell = meta.get('ellipse_orig')
                if pre_img is not None and bounds is not None and ctr is not None:
                    x1r, y1r, x2r, y2r = bounds
                    fw_r, fh_r = x2r - x1r, y2r - y1r
                    ar_r = min(fw_r, fh_r) / max(fw_r, fh_r) if max(fw_r, fh_r) > 0 else 0
                    sr_r = max(ell[1][0], ell[1][1]) / 2 if ell else 0
                    vis_orig = pre_img.copy()
                    sam_m = meta.get('sam_mask')
                    if sam_m is not None:
                        mask_u8 = ((sam_m > 0).astype(np.uint8)
                                   if sam_m.dtype != bool
                                   else sam_m.astype(np.uint8)) * 255
                        contours, _ = cv2.findContours(
                            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(vis_orig, contours, -1, (0, 255, 0), 2)
                        if contours:
                            c_max = max(contours, key=cv2.contourArea)
                            rect = cv2.minAreaRect(c_max)
                            box_pts = cv2.boxPoints(rect).astype(np.int32)
                            cv2.drawContours(vis_orig, [box_pts], 0, (0, 255, 255), 2)
                    cv2.rectangle(vis_orig, (x1r, y1r), (x2r, y2r), (255, 255, 255), 2)
                    refined_ell_vis = meta.get('refined_sticker_ell')
                    if ctr and ell:
                        draw_ell = refined_ell_vis if refined_ell_vis is not None else ell
                        cv2.ellipse(vis_orig, draw_ell, (255, 0, 0), 2)
                    cropped_r = pre_img[y1r:y2r, x1r:x2r]
                    ph, pw = pre_img.shape[:2]
                    th = max(ph, fh_r)
                    o_rs = cv2.resize(vis_orig, (int(pw * th / ph), th))
                    c_rs = cv2.resize(cropped_r, (int(fw_r * th / fh_r), th))
                    combo = np.hstack([o_rs, c_rs])
                    cv2.putText(combo, f'Original {pw}x{ph}', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(combo, f'ROI {fw_r}x{fh_r} ar={ar_r:.2f}',
                                (o_rs.shape[1] + 10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    _async_save(debug_dir / f'roi_{tag}{specimen_no}.jpg', combo)

                    # 中间过程图：带框原图 | 标签边缘 | 面检测（三联对比）
                    if ctr and ell:
                        sam_m = meta.get('sam_mask')
                        refined_ell = meta.get('refined_sticker_ell')
                        stk_debug, face_debug = _generate_detection_debug(
                            pre_img, ctr[0], ctr[1], int(sr_r), bounds,
                            sam_mask=sam_m, refined_sticker_ell=refined_ell,
                            hsv_ellipse=ell)
                        th2 = ph
                        orig_r = cv2.resize(vis_orig, (int(pw * th2 / ph), th2))
                        stk_r = cv2.resize(stk_debug, (int(pw * th2 / ph), th2))
                        face_r = cv2.resize(face_debug, (int(pw * th2 / ph), th2))
                        triple = np.hstack([orig_r, stk_r, face_r])
                        cv2.putText(triple, 'Original', (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(triple, 'Sticker Detection', (orig_r.shape[1] + 10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(triple, 'Face Detection (SAM)', (orig_r.shape[1]*2 + 10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        _async_save(debug_dir / f'process_{tag}{specimen_no}.jpg',
                                    triple)

            results.append({
                'batch': batch_id,
                'specimen': specimen_no,
                'method': method_name,
                'raw_matches': best['n_raw'],
                'filtered_matches': n_matches,
                'inliers': n_inliers,
                'mean_conf': round(best['conf'], 4),
                'inlier_ratio': round(best['inlier_ratio'], 4),
                'final_score': round(final_score, 4),
                'verdict': verdict,
                'best_rotation': round(float(rot_deg), 1),
                'rot_method': rot_method,
                'sticker_wet': stk0,
                'sticker_dry': stk1,
                'time_sec': round(elapsed, 2),
                'wet_path': str(wet_path), 'dry_path': str(dry_path) if dry_path else None,
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({
                'batch': batch_id, 'specimen': specimen_no,
                'method': method_name, 'error': str(e),
                'wet_path': str(wet_path), 'dry_path': str(dry_path) if dry_path else None,
            })

    _wait_saves()
    return results


def run_method_loftr(method_name, loftr_model, pairs, roi_mode='dino', rectify=False):
    """用 LoFTR 跑所有配对。LoFTR 是 dense matcher，输入灰度图，直接输出匹配点对。"""
    results = []
    out_dir = get_output_dir(method_name)

    for batch_id, wet_path, dry_path, specimen_no in pairs:
        label = f"{batch_id}#{specimen_no}"
        print(f"\n--- {method_name} | {label} ---")

        if dry_path is None:
            print(f"  INVALID: QR配对失败")
            results.append({
                'batch': batch_id, 'specimen': specimen_no,
                'method': method_name,
                'raw_matches': 0, 'filtered_matches': 0,
                'inliers': 0, 'mean_conf': 0, 'inlier_ratio': 0,
                'final_score': 0, 'verdict': 'INVALID',
                'best_rotation': 0, 'rot_method': 'N/A',
                'sticker_wet': False, 'sticker_dry': False,
                'time_sec': 0, 'invalid_reason': 'QR配对失败',
            })
            continue

        try:
            img0_bgr, tensor0, mask0, center0, stk0, meta0 = load_and_preprocess(
                wet_path, max_size=MAX_SIZE_WET, roi_mode=roi_mode, rectify=rectify)
            img1_bgr, tensor1, mask1, center1, stk1, meta1 = load_and_preprocess(
                dry_path, max_size=MAX_SIZE_DRY, roi_mode=roi_mode, rectify=rectify)

            invalid_reason = None
            if not stk0:
                invalid_reason = "WET贴纸未检测到"
            elif not stk1:
                invalid_reason = "DRY贴纸未检测到"

            if invalid_reason:
                print(f"  INVALID: {invalid_reason}")
                results.append({
                    'batch': batch_id, 'specimen': specimen_no,
                    'method': method_name,
                    'raw_matches': 0, 'filtered_matches': 0,
                    'inliers': 0, 'mean_conf': 0, 'inlier_ratio': 0,
                    'final_score': 0, 'verdict': 'INVALID',
                    'best_rotation': 0, 'rot_method': 'N/A',
                    'sticker_wet': stk0, 'sticker_dry': stk1,
                    'time_sec': 0, 'invalid_reason': invalid_reason,
                })
                continue

            t_start = time.time()

            # LoFTR 对齐：同样使用矩形+象限
            ell0 = _ellipse_from_mask(mask0)
            ell1 = _ellipse_from_mask(mask1)
            align_result = align_pair(
                img0_bgr, img1_bgr,
                center0, center1,
                mask_wet=mask0, mask_dry=mask1,
                ellipse_wet=ell0, ellipse_dry=ell1,
                sam_mask_wet=meta0.get('sam_mask_processed'),
                sam_mask_dry=meta1.get('sam_mask_processed'),
            )

            img_wet_a = align_result['img_wet']
            img_dry_a = align_result['img_dry']
            mask_wet_a = align_result['mask_wet']
            mask_dry_a = align_result['mask_dry']

            if mask_wet_a is None:
                mask_wet_a = np.zeros(img_wet_a.shape[:2], dtype=np.uint8)
            if mask_dry_a is None:
                mask_dry_a = np.zeros(img_dry_a.shape[:2], dtype=np.uint8)

            # 干态灰度 tensor
            gray_dry = cv2.cvtColor(img_dry_a, cv2.COLOR_BGR2GRAY)
            t_dry = torch.from_numpy(gray_dry).float()[None, None] / 255.0

            best_score = -1
            best_rot = 0.0
            best = None

            n_cands = len(align_result['candidates'])
            print(f"    [{align_result['method']}] {n_cands} candidate(s)")

            for rot_k, rot_deg in align_result['candidates']:
                if rot_k > 0:
                    img_r = rotate_cv2(img_wet_a, rot_k)
                    m_r = np.ascontiguousarray(np.rot90(mask_wet_a, rot_k))
                else:
                    img_r = img_wet_a
                    m_r = mask_wet_a

                gray_wet = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
                t_wet = torch.from_numpy(gray_wet).float()[None, None] / 255.0

                with torch.no_grad():
                    res = loftr_model({
                        'image0': t_wet.to(DEVICE),
                        'image1': t_dry.to(DEVICE),
                    })

                kpts0_raw = res['keypoints0']
                kpts1_raw = res['keypoints1']
                conf_raw = res['confidence']
                n_raw = len(kpts0_raw)

                # 过滤低置信度
                high_conf = conf_raw > 0.3
                kpts0_hc = kpts0_raw[high_conf]
                kpts1_hc = kpts1_raw[high_conf]
                conf_hc = conf_raw[high_conf]

                # 过滤贴纸区域
                kpts0_f, kpts1_f, conf_f = filter_points_by_mask(
                    kpts0_hc, kpts1_hc, conf_hc, m_r, mask_dry_a)

                n_filtered = len(kpts0_f)
                if n_filtered >= 4:
                    pts0_np = kpts0_f.cpu().numpy()
                    pts1_np = kpts1_f.cpu().numpy()
                    conf_np = conf_f.cpu().numpy()
                    sc, cf, ir, inl, n_scored = compute_score_from_points(
                        pts0_np, pts1_np, conf_np)
                else:
                    sc, cf, ir = 0.0, 0.0, 0.0
                    inl = np.array([])
                    n_scored = n_filtered

                if sc > best_score:
                    best_score = sc
                    best_rot = rot_deg
                    best = {
                        'kpts0': kpts0_f, 'kpts1': kpts1_f,
                        'conf': cf, 'inlier_ratio': ir,
                        'inliers': inl,
                        'n_raw': n_raw, 'n_filtered': n_filtered,
                        'mask0': m_r, 'rot_method': align_result['method'],
                        'img0_vis': img_r,
                    }

            elapsed = time.time() - t_start
            final_score = best_score
            rot_deg = best_rot
            rot_method = best['rot_method']
            n_matches = best['n_filtered']
            n_inliers = int(best['inliers'].sum()) if len(best['inliers']) > 0 else 0

            print(f"    → 选择 {rot_deg:.1f}° ({rot_method}, score={final_score:.3f})")

            if n_matches < MIN_MATCHES:
                verdict = "INSUFFICIENT"
            elif final_score > MATCH_THRESHOLD:
                verdict = "SAME"
            else:
                verdict = "DIFFERENT"

            print(f"  Raw: {best['n_raw']} -> Filtered: {n_matches} -> "
                  f"Inliers: {n_inliers}")
            print(f"  Conf: {best['conf']:.3f} | InlierR: {best['inlier_ratio']:.3f} | "
                  f"Score: {final_score:.3f} | {verdict} | "
                  f"rot={rot_deg:.1f}° ({rot_method})")
            print(f"  Time: {elapsed:.2f}s")

            results.append({
                'batch': batch_id, 'specimen': specimen_no,
                'method': method_name,
                'raw_matches': best['n_raw'],
                'filtered_matches': n_matches,
                'inliers': n_inliers,
                'mean_conf': round(best['conf'], 4),
                'inlier_ratio': round(best['inlier_ratio'], 4),
                'final_score': round(final_score, 4),
                'verdict': verdict,
                'best_rotation': round(float(rot_deg), 1),
                'rot_method': rot_method,
                'sticker_wet': stk0,
                'sticker_dry': stk1,
                'time_sec': round(elapsed, 2),
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({
                'batch': batch_id, 'specimen': specimen_no,
                'method': method_name, 'error': str(e),
            })

    _wait_saves()
    return results


def run_method_roma(method_name, roma_model, pairs, roi_mode='dino', rectify=False):
    """用 RoMa 稠密匹配器跑所有配对。RoMa 输出像素级warp + certainty。"""
    results = []
    out_dir = get_output_dir(method_name)

    for batch_id, wet_path, dry_path, specimen_no in pairs:
        label = f"{batch_id}#{specimen_no}"
        print(f"\n--- {method_name} | {label} ---")

        if dry_path is None:
            print(f"  INVALID: QR配对失败")
            results.append({
                'batch': batch_id, 'specimen': specimen_no,
                'method': method_name,
                'raw_matches': 0, 'filtered_matches': 0,
                'inliers': 0, 'mean_conf': 0, 'inlier_ratio': 0,
                'final_score': 0, 'verdict': 'INVALID',
                'best_rotation': 0, 'rot_method': 'N/A',
                'sticker_wet': False, 'sticker_dry': False,
                'time_sec': 0, 'invalid_reason': 'QR配对失败',
            })
            continue

        try:
            img0_bgr, tensor0, mask0, center0, stk0, meta0 = load_and_preprocess(
                wet_path, max_size=MAX_SIZE_WET, roi_mode=roi_mode, rectify=rectify)
            img1_bgr, tensor1, mask1, center1, stk1, meta1 = load_and_preprocess(
                dry_path, max_size=MAX_SIZE_DRY, roi_mode=roi_mode, rectify=rectify)

            invalid_reason = None
            if not stk0:
                invalid_reason = "WET贴纸未检测到"
            elif not stk1:
                invalid_reason = "DRY贴纸未检测到"

            if invalid_reason:
                print(f"  INVALID: {invalid_reason}")
                results.append({
                    'batch': batch_id, 'specimen': specimen_no,
                    'method': method_name,
                    'raw_matches': 0, 'filtered_matches': 0,
                    'inliers': 0, 'mean_conf': 0, 'inlier_ratio': 0,
                    'final_score': 0, 'verdict': 'INVALID',
                    'best_rotation': 0, 'rot_method': 'N/A',
                    'sticker_wet': stk0, 'sticker_dry': stk1,
                    'time_sec': 0, 'invalid_reason': invalid_reason,
                })
                continue

            t_start = time.time()

            ell0 = _ellipse_from_mask(mask0)
            ell1 = _ellipse_from_mask(mask1)
            align_result = align_pair(
                img0_bgr, img1_bgr,
                center0, center1,
                mask_wet=mask0, mask_dry=mask1,
                ellipse_wet=ell0, ellipse_dry=ell1,
                sam_mask_wet=meta0.get('sam_mask_processed'),
                sam_mask_dry=meta1.get('sam_mask_processed'),
            )

            img_wet_a = align_result['img_wet']
            img_dry_a = align_result['img_dry']
            mask_wet_a = align_result['mask_wet']
            mask_dry_a = align_result['mask_dry']

            if mask_wet_a is None:
                mask_wet_a = np.zeros(img_wet_a.shape[:2], dtype=np.uint8)
            if mask_dry_a is None:
                mask_dry_a = np.zeros(img_dry_a.shape[:2], dtype=np.uint8)

            best_score = -1
            best_rot = 0.0
            best = None

            n_cands = len(align_result['candidates'])
            print(f"    [{align_result['method']}] {n_cands} candidate(s)")

            # RoMa 需要 RGB PIL Image 或路径，这里用临时文件
            import tempfile
            from PIL import Image as PILImage

            # 保存 dry 图像一次
            dry_rgb = cv2.cvtColor(img_dry_a, cv2.COLOR_BGR2RGB)
            dry_pil = PILImage.fromarray(dry_rgb)
            H_dry, W_dry = img_dry_a.shape[:2]

            for rot_k, rot_deg in align_result['candidates']:
                if rot_k > 0:
                    img_r = rotate_cv2(img_wet_a, rot_k)
                    m_r = np.ascontiguousarray(np.rot90(mask_wet_a, rot_k))
                else:
                    img_r = img_wet_a
                    m_r = mask_wet_a

                wet_rgb = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
                wet_pil = PILImage.fromarray(wet_rgb)
                H_wet, W_wet = img_r.shape[:2]

                # RoMa match
                with torch.no_grad():
                    warp, certainty = roma_model.match(wet_pil, dry_pil, device=DEVICE)
                    matches, cert = roma_model.sample(warp, certainty)
                    kptsA, kptsB = roma_model.to_pixel_coordinates(
                        matches, H_wet, W_wet, H_dry, W_dry)

                kpts0_t = kptsA
                kpts1_t = kptsB
                conf_t = cert
                n_raw = len(kpts0_t)

                # 过滤贴纸区域
                kpts0_f, kpts1_f, conf_f = filter_points_by_mask(
                    kpts0_t, kpts1_t, conf_t, m_r, mask_dry_a)
                n_filtered = len(kpts0_f)

                if n_filtered >= 4:
                    pts0_np = kpts0_f.cpu().numpy() if torch.is_tensor(kpts0_f) else kpts0_f
                    pts1_np = kpts1_f.cpu().numpy() if torch.is_tensor(kpts1_f) else kpts1_f
                    conf_np = conf_f.cpu().numpy() if torch.is_tensor(conf_f) else conf_f
                    sc, cf, ir, inl, n_scored = compute_score_from_points(
                        pts0_np, pts1_np, conf_np)
                else:
                    sc, cf, ir = 0.0, 0.0, 0.0
                    inl = np.array([])
                    n_scored = n_filtered

                if sc > best_score:
                    best_score = sc
                    best_rot = rot_deg
                    best = {
                        'kpts0': kpts0_f, 'kpts1': kpts1_f,
                        'conf': cf, 'inlier_ratio': ir,
                        'inliers': inl,
                        'n_raw': n_raw, 'n_filtered': n_filtered,
                        'mask0': m_r, 'rot_method': align_result['method'],
                        'img0_vis': img_r,
                    }

            elapsed = time.time() - t_start
            final_score = best_score
            rot_deg = best_rot
            rot_method = best['rot_method']
            n_matches = best['n_filtered']
            n_inliers = int(best['inliers'].sum()) if len(best['inliers']) > 0 else 0

            print(f"    → 选择 {rot_deg:.1f}° ({rot_method}, score={final_score:.3f})")

            if n_matches < MIN_MATCHES:
                verdict = "INSUFFICIENT"
            elif final_score > MATCH_THRESHOLD:
                verdict = "SAME"
            else:
                verdict = "DIFFERENT"

            print(f"  Raw: {best['n_raw']} -> Filtered: {n_matches} -> "
                  f"Inliers: {n_inliers}")
            print(f"  Conf: {best['conf']:.3f} | InlierR: {best['inlier_ratio']:.3f} | "
                  f"Score: {final_score:.3f} | {verdict} | "
                  f"rot={rot_deg:.1f}° ({rot_method})")
            print(f"  Time: {elapsed:.2f}s")

            results.append({
                'batch': batch_id, 'specimen': specimen_no,
                'method': method_name,
                'raw_matches': best['n_raw'],
                'filtered_matches': n_matches,
                'inliers': n_inliers,
                'mean_conf': round(best['conf'], 4),
                'inlier_ratio': round(best['inlier_ratio'], 4),
                'final_score': round(final_score, 4),
                'verdict': verdict,
                'best_rotation': round(float(rot_deg), 1),
                'rot_method': rot_method,
                'sticker_wet': stk0,
                'sticker_dry': stk1,
                'time_sec': round(elapsed, 2),
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({
                'batch': batch_id, 'specimen': specimen_no,
                'method': method_name, 'error': str(e),
            })

    _wait_saves()
    return results


def run_cross_batch_test(extractor, matcher, pairs, method_name, rectify=False):
    """
    跨批次测试：取不同批次的 wet/dry 做匹配，预期 DIFFERENT。
    """
    print(f"\n{'='*60}")
    print(f"Cross-batch test ({method_name})")
    print(f"{'='*60}")

    results = []
    batch_ids = list(set(p[0] for p in pairs))[:4]
    first_of_batch = {}
    for b, w, d, n in pairs:
        if b in batch_ids and n == 1 and b not in first_of_batch:
            first_of_batch[b] = (w, d)

    items = list(first_of_batch.items())
    cross_dir = get_output_dir("cross_batch")
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            bid_i, (wi, _) = items[i]
            bid_j, (_, dj) = items[j]
            label = f"CROSS_{bid_i}_vs_{bid_j}"
            print(f"\n--- {label} ---")

            try:
                img0_bgr, tensor0, mask0, center0, _, meta0 = load_and_preprocess(
                    wi, max_size=MAX_SIZE_WET, rectify=rectify)
                img1_bgr, tensor1, mask1, center1, _, meta1 = load_and_preprocess(
                    dj, max_size=MAX_SIZE_DRY, rectify=rectify)

                # 对齐
                ell0 = _ellipse_from_mask(mask0)
                ell1 = _ellipse_from_mask(mask1)
                align_result = align_pair(
                    img0_bgr, img1_bgr,
                    center0, center1,
                    mask_wet=mask0, mask_dry=mask1,
                    ellipse_wet=ell0, ellipse_dry=ell1,
                    sam_mask_wet=meta0.get('sam_mask_processed'),
                    sam_mask_dry=meta1.get('sam_mask_processed'),
                )

                img_wet_a = align_result['img_wet']
                img_dry_a = align_result['img_dry']
                mask_wet_a = align_result['mask_wet']
                mask_dry_a = align_result['mask_dry']
                if mask_wet_a is None:
                    mask_wet_a = np.zeros(img_wet_a.shape[:2], dtype=np.uint8)
                if mask_dry_a is None:
                    mask_dry_a = np.zeros(img_dry_a.shape[:2], dtype=np.uint8)

                tensor_dry = _make_tensor(img_dry_a)
                feats1 = extractor.extract(tensor_dry.to(DEVICE))

                best_score = -1
                best_rot = 0.0
                best_d = None

                for rot_k, rot_deg in align_result['candidates']:
                    if rot_k > 0:
                        img_r = rotate_cv2(img_wet_a, rot_k)
                        m_r = np.ascontiguousarray(np.rot90(mask_wet_a, rot_k))
                    else:
                        img_r = img_wet_a
                        m_r = mask_wet_a

                    tensor_wet = _make_tensor(img_r)
                    feats0 = extractor.extract(tensor_wet.to(DEVICE))
                    result = matcher({'image0': feats0, 'image1': feats1})
                    f0, f1, res = [rbd(x) for x in [feats0, feats1, result]]

                    matches_raw = res['matches']
                    scores_raw = res.get('scores', None)
                    kpts0, kpts1 = f0['keypoints'], f1['keypoints']

                    matches_f, scores_f = filter_matches(
                        kpts0, kpts1, matches_raw, scores_raw, m_r, mask_dry_a)
                    sc, cf, ir, inl, _ = compute_score(
                        kpts0, kpts1, matches_f, scores_f)

                    if sc > best_score:
                        best_score = sc
                        best_rot = rot_deg
                        best_d = {
                            'n_filtered': len(matches_f),
                            'rot_method': align_result['method'],
                        }

                n_filt = best_d['n_filtered']
                if n_filt < MIN_MATCHES:
                    verdict = "INSUFFICIENT"
                elif best_score > MATCH_THRESHOLD:
                    verdict = "SAME"
                else:
                    verdict = "DIFFERENT"

                print(f"  Score: {best_score:.3f} | {verdict} | "
                      f"rot={best_rot:.1f}° ({best_d.get('rot_method','?')}) | "
                      f"matches={n_filt}")

                results.append({
                    'pair': label,
                    'matches': n_filt,
                    'score': round(best_score, 4),
                    'verdict': verdict,
                    'expected': 'DIFFERENT',
                })

            except Exception as e:
                print(f"  ERROR: {e}")

    return results


def run_intra_batch_cross_test(extractor, matcher, main_results, method_name, rectify=False):
    """
    同批交叉验证：对全部3对都是SAME的批次，做同批内交叉匹配。
    例如 batch 有 (wet1,dry1),(wet2,dry2),(wet3,dry3) 都是 SAME，
    则交叉测试 wet1-dry2, wet1-dry3, wet2-dry1, wet2-dry3, wet3-dry1, wet3-dry2，
    预期全部 DIFFERENT。
    """
    # 找出全部3对都是 SAME 的批次
    from collections import defaultdict
    batch_verdicts = defaultdict(list)
    for r in main_results:
        if r.get('verdict') and r.get('batch'):
            batch_verdicts[r['batch']].append(r['verdict'])

    all_same_batches = [b for b, vlist in batch_verdicts.items()
                        if len(vlist) >= 3 and all(v == 'SAME' for v in vlist)]

    if not all_same_batches:
        print("\n[intra-batch cross] 没有全部SAME的批次，跳过")
        return []

    print(f"\n{'='*60}")
    print(f"Intra-batch cross test ({method_name})")
    print(f"全部SAME批次: {len(all_same_batches)} 个")
    print(f"{'='*60}")

    # 按批次分组，从 main_results 中提取 wet/dry 路径
    batch_pairs = defaultdict(list)
    for r in main_results:
        if r.get('batch') in all_same_batches and r.get('wet_path') and r.get('dry_path'):
            batch_pairs[r['batch']].append((r['wet_path'], r['dry_path'], r['specimen']))

    results = []
    cross_dir = get_output_dir("intra_cross")

    for batch_id, pair_list in batch_pairs.items():
        if len(pair_list) < 2:
            continue
        print(f"\n  === {batch_id} ({len(pair_list)} pairs) ===")

        # 预处理所有图像
        processed = {}
        for wet_path, dry_path, spec_no in pair_list:
            try:
                img_w, t_w, mask_w, center_w, stk_w, meta_w = load_and_preprocess(
                    wet_path, max_size=MAX_SIZE_WET, rectify=rectify)
                img_d, t_d, mask_d, center_d, stk_d, meta_d = load_and_preprocess(
                    dry_path, max_size=MAX_SIZE_DRY, rectify=rectify)
                processed[spec_no] = {
                    'wet': (img_w, mask_w, center_w, meta_w),
                    'dry': (img_d, mask_d, center_d, meta_d),
                }
            except Exception as e:
                print(f"    预处理失败 #{spec_no}: {e}")

        spec_nos = sorted(processed.keys())

        # 交叉匹配：每个 wet_i vs 每个 dry_j (i≠j)
        for i in spec_nos:
            for j in spec_nos:
                if i == j:
                    continue
                label = f"{batch_id}#wet{i}_vs_dry{j}"
                try:
                    img_w, mask_w, center_w, meta_w = processed[i]['wet']
                    img_d, mask_d, center_d, meta_d = processed[j]['dry']

                    ell_w = _ellipse_from_mask(mask_w)
                    ell_d = _ellipse_from_mask(mask_d)
                    align_result = align_pair(
                        img_w, img_d, center_w, center_d,
                        mask_wet=mask_w, mask_dry=mask_d,
                        ellipse_wet=ell_w, ellipse_dry=ell_d,
                        sam_mask_wet=meta_w.get('sam_mask_processed'),
                        sam_mask_dry=meta_d.get('sam_mask_processed'),
                    )

                    img_wet_a = align_result['img_wet']
                    img_dry_a = align_result['img_dry']
                    mw_a = align_result['mask_wet']
                    md_a = align_result['mask_dry']
                    if mw_a is None:
                        mw_a = np.zeros(img_wet_a.shape[:2], dtype=np.uint8)
                    if md_a is None:
                        md_a = np.zeros(img_dry_a.shape[:2], dtype=np.uint8)

                    # 等比尺度归一化
                    h0, w0 = img_wet_a.shape[:2]
                    h1, w1 = img_dry_a.shape[:2]
                    area_ratio = (h0 * w0) / max(h1 * w1, 1)
                    if area_ratio > 1.5:
                        scale = (h1 * w1 / (h0 * w0)) ** 0.5
                        nw, nh = int(w0 * scale), int(h0 * scale)
                        img_wet_a = cv2.resize(img_wet_a, (nw, nh), interpolation=cv2.INTER_AREA)
                        if mw_a.size > 0:
                            mw_a = cv2.resize(mw_a, (nw, nh), interpolation=cv2.INTER_NEAREST)
                    elif area_ratio < 1 / 1.5:
                        scale = (h0 * w0 / (h1 * w1)) ** 0.5
                        nw, nh = int(w1 * scale), int(h1 * scale)
                        img_dry_a = cv2.resize(img_dry_a, (nw, nh), interpolation=cv2.INTER_AREA)
                        if md_a.size > 0:
                            md_a = cv2.resize(md_a, (nw, nh), interpolation=cv2.INTER_NEAREST)

                    tensor_dry = _make_tensor(img_dry_a)
                    feats1 = extractor.extract(tensor_dry.to(DEVICE))

                    best_score = -1
                    best_rot = 0.0
                    best_n = 0
                    best_conf = 0.0
                    best_ir = 0.0
                    best_inliers = 0

                    for rot_k, rot_deg in align_result['candidates']:
                        if rot_k > 0:
                            img_r = rotate_cv2(img_wet_a, rot_k)
                            m_r = rotate_cv2(mw_a, rot_k) if mw_a is not None else None
                        else:
                            img_r = img_wet_a
                            m_r = mw_a

                        tensor_wet = _make_tensor(img_r)
                        feats0 = extractor.extract(tensor_wet.to(DEVICE))
                        res = matcher({'image0': feats0, 'image1': feats1})
                        f0, f1, res = [rbd(x) for x in [feats0, feats1, res]]

                        matches_raw = res['matches']
                        scores_raw = res.get('scores', None)
                        kpts0, kpts1 = f0['keypoints'], f1['keypoints']

                        matches_f, scores_f = filter_matches(
                            kpts0, kpts1, matches_raw, scores_raw,
                            m_r if m_r is not None else np.zeros(img_r.shape[:2], np.uint8),
                            md_a)
                        sc, cf, ir, inl, _ = compute_score(
                            kpts0, kpts1, matches_f, scores_f)

                        if sc > best_score:
                            best_score = sc
                            best_rot = rot_deg
                            best_n = len(matches_f)
                            best_conf = cf
                            best_ir = ir
                            best_inliers = int(inl.sum()) if hasattr(inl, 'sum') else 0

                    if best_n < 4:
                        verdict = "INSUFFICIENT"
                    elif best_score >= MATCH_THRESHOLD:
                        verdict = "SAME"
                    else:
                        verdict = "DIFFERENT"

                    status = "✅" if verdict == "DIFFERENT" else "❌"
                    print(f"    {label}: score={best_score:.3f} {verdict} {status}")

                    results.append({
                        'pair': label,
                        'batch': batch_id,
                        'wet_spec': i, 'dry_spec': j,
                        'matches': best_n,
                        'inliers': best_inliers,
                        'mean_conf': round(best_conf, 4),
                        'inlier_ratio': round(best_ir, 4),
                        'score': round(best_score, 4),
                        'verdict': verdict,
                        'expected': 'DIFFERENT',
                        'correct': verdict != 'SAME',
                    })

                except Exception as e:
                    print(f"    {label}: ERROR {e}")

    # 汇总
    if results:
        n_correct = sum(1 for r in results if r['correct'])
        n_total = len(results)
        n_false_same = sum(1 for r in results if r['verdict'] == 'SAME')
        print(f"\n  === Intra-batch cross summary ===")
        print(f"  Total: {n_total} | Correct(DIFF): {n_correct} | "
              f"FalseSAME: {n_false_same} | Accuracy: {n_correct/n_total*100:.1f}%")

    return results


# ============================================================
# 单对匹配（供 web /inspect 页使用）
# ============================================================

_CACHED_MODELS: dict = {}   # method_name → (extractor, matcher)


def _get_models(method: str = 'sp'):
    """懒加载并缓存匹配模型，避免 web 请求每次重新加载。"""
    if method in _CACHED_MODELS:
        return _CACHED_MODELS[method]
    from lightglue import LightGlue, SuperPoint, ALIKED, SIFT, DoGHardNet
    if method == 'sp':
        ext = SuperPoint(max_num_keypoints=MAX_KEYPOINTS).eval().to(DEVICE)
        mat = LightGlue(features='superpoint').eval().to(DEVICE)
    elif method == 'aliked':
        ext = ALIKED(max_num_keypoints=MAX_KEYPOINTS).eval().to(DEVICE)
        mat = LightGlue(features='aliked').eval().to(DEVICE)
    elif method == 'sift':
        ext = SIFT(max_num_keypoints=MAX_KEYPOINTS).eval().to(DEVICE)
        mat = LightGlue(features='sift').eval().to(DEVICE)
    elif method == 'hardnet':
        ext = DoGHardNet(max_num_keypoints=MAX_KEYPOINTS).eval().to(DEVICE)
        mat = LightGlue(features='doghardnet').eval().to(DEVICE)
    else:
        raise ValueError(f"Unknown method: {method}")
    _CACHED_MODELS[method] = (ext, mat)
    return ext, mat


def _make_preprocess_vis(meta: dict, label: str = '') -> np.ndarray:
    """
    根据 meta 生成预处理中间可视化图：
      - 原图（resize后）
      - 叠加 SAM mask（半透明绿色）
      - 画混凝土面 ROI 框（青色矩形）
      - 画贴纸椭圆（绿色）
    """
    img = meta.get('img_pre_roi')
    if img is None:
        return None
    vis = img.copy()

    # SAM mask 半透明叠加
    sam_m = meta.get('sam_mask')
    if sam_m is not None:
        mask_bool = (sam_m > 0) if sam_m.dtype != bool else sam_m
        green = np.zeros_like(vis)
        green[:, :, 1] = 180
        vis[mask_bool] = cv2.addWeighted(
            vis[mask_bool], 0.55, green[mask_bool], 0.45, 0)
        mask_u8 = mask_bool.astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 255, 128), 2)

    # 混凝土面 ROI 框
    bounds = meta.get('roi_bounds')
    if bounds is not None:
        x1, y1, x2, y2 = bounds
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 255), 3)
        cv2.putText(vis, 'ROI', (x1 + 4, y1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)

    # 贴纸椭圆
    ell = meta.get('refined_sticker_ell') or meta.get('ellipse_orig')
    if ell is not None:
        cv2.ellipse(vis, ell, (0, 255, 0), 3)
        (cx, cy) = meta.get('center_orig') or (int(ell[0][0]), int(ell[0][1]))
        cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)

    if label:
        cv2.putText(vis, label, (8, vis.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(vis, label, (8, vis.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return vis


def run_single_pair(wet_path: str, dry_path: str,
                    method: str = 'sp', roi_mode: str = 'dino', rectify=False) -> dict:
    """
    对单对图片运行完整匹配流水线，返回中间结果字典（供 web /inspect 页使用）。

    返回 keys:
      wet_roi, dry_roi  — 预处理后的 ROI 图（BGR ndarray）
      match_vis         — 匹配可视化图（BGR ndarray or None）
      score             — float
      verdict           — 'SAME' / 'DIFFERENT' / 'INSUFFICIENT' / 'INVALID'
      n_raw, n_filtered, n_inliers  — int
      rot_deg           — float
      elapsed           — float
      error             — str or None
    """
    import time as _time
    import tempfile

    out = dict(wet_roi=None, dry_roi=None, match_vis=None,
               wet_previs=None, dry_previs=None,
               wet_aligned=None, dry_aligned=None,
               score=0.0, verdict='INVALID', n_raw=0,
               n_filtered=0, n_inliers=0, rot_deg=0.0,
               elapsed=0.0, error=None)
    try:
        img0_bgr, tensor0, mask0, center0, stk0, meta0 = load_and_preprocess(
            wet_path, max_size=MAX_SIZE_WET, roi_mode=roi_mode, rectify=rectify)
        img1_bgr, tensor1, mask1, center1, stk1, meta1 = load_and_preprocess(
            dry_path, max_size=MAX_SIZE_DRY, roi_mode=roi_mode, rectify=rectify)

        out['wet_roi'] = img0_bgr
        out['dry_roi'] = img1_bgr
        # 预处理中间图（SAM mask + ROI框 + 贴纸椭圆）
        out['wet_previs'] = _make_preprocess_vis(meta0, 'WET 预处理')
        out['dry_previs'] = _make_preprocess_vis(meta1, 'DRY 预处理')

        if not stk0:
            out['verdict'] = 'INVALID'
            out['error'] = 'WET 贴纸未检测到'
            return out
        if not stk1:
            out['verdict'] = 'INVALID'
            out['error'] = 'DRY 贴纸未检测到'
            return out

        t0 = _time.time()

        ell0 = _ellipse_from_mask(mask0)
        ell1 = _ellipse_from_mask(mask1)
        align_result = align_pair(
            img0_bgr, img1_bgr, center0, center1,
            mask_wet=mask0, mask_dry=mask1,
            ellipse_wet=ell0, ellipse_dry=ell1,
            sam_mask_wet=meta0.get('sam_mask_processed'),
            sam_mask_dry=meta1.get('sam_mask_processed'),
        )

        img_wet_a = align_result['img_wet']
        img_dry_a = align_result['img_dry']
        mw = align_result['mask_wet']
        md = align_result['mask_dry']
        if mw is None:
            mw = np.zeros(img_wet_a.shape[:2], dtype=np.uint8)
        if md is None:
            md = np.zeros(img_dry_a.shape[:2], dtype=np.uint8)

        # 尺度归一化
        h0, w0 = img_wet_a.shape[:2]
        h1, w1 = img_dry_a.shape[:2]
        ar = (h0 * w0) / max(h1 * w1, 1)
        if ar > 1.5:
            img_wet_a = cv2.resize(img_wet_a, (w1, h1), interpolation=cv2.INTER_AREA)
            mw = cv2.resize(mw, (w1, h1), interpolation=cv2.INTER_NEAREST)
        elif ar < 1 / 1.5:
            img_dry_a = cv2.resize(img_dry_a, (w0, h0), interpolation=cv2.INTER_AREA)
            md = cv2.resize(md, (w0, h0), interpolation=cv2.INTER_NEAREST)

        extractor, matcher = _get_models(method)
        feats1 = extractor.extract(_make_tensor(img_dry_a).to(DEVICE))

        best_score, best_rot, best = -1.0, 0.0, None
        for rot_k, rot_deg in align_result['candidates']:
            img_r = rotate_cv2(img_wet_a, rot_k) if rot_k > 0 else img_wet_a
            m_r = np.ascontiguousarray(np.rot90(mw, rot_k)) if rot_k > 0 else mw

            feats0 = extractor.extract(_make_tensor(img_r).to(DEVICE))
            res_raw = matcher({'image0': feats0, 'image1': feats1})
            f0, f1, res = [rbd(x) for x in [feats0, feats1, res_raw]]

            matches_raw = res['matches']
            scores_raw = res.get('scores')
            kpts0, kpts1 = f0['keypoints'], f1['keypoints']

            matches_f, scores_f = filter_matches(
                kpts0, kpts1, matches_raw, scores_raw, m_r, md)
            sc, cf, ir, inl, _ = compute_score(kpts0, kpts1, matches_f, scores_f)

            if sc > best_score:
                best_score = sc
                best_rot = rot_deg
                best = dict(kpts0=kpts0, kpts1=kpts1,
                            matches=matches_f, scores=scores_f,
                            inliers=inl, conf=cf, inlier_ratio=ir,
                            n_raw=len(matches_raw), img0_vis=img_r, mask0=m_r,
                            rot_method=align_result['method'])

        elapsed = _time.time() - t0
        n_f = len(best['matches'])
        n_inl = int(best['inliers'].sum()) if len(best['inliers']) > 0 else 0

        if n_f < MIN_MATCHES:
            verdict = 'INSUFFICIENT'
        elif best_score >= MATCH_THRESHOLD:
            verdict = 'SAME'
        else:
            verdict = 'DIFFERENT'

        # 匹配可视化图
        match_vis = None
        if n_f >= 4:
            tmp = tempfile.mktemp(suffix='.png')
            title = (f"{method.upper()} | score={best_score:.3f} | {verdict} | "
                     f"rot={best_rot:.1f}° | {n_f} matches / {n_inl} inliers")
            draw_matches_vis(best['img0_vis'], img_dry_a,
                             best['kpts0'], best['kpts1'],
                             best['matches'], best['inliers'],
                             best['mask0'], md, title, tmp,
                             rot_deg=best_rot, meta0=meta0, meta1=meta1)
            if os.path.exists(tmp):
                match_vis = cv2.imread(tmp)
                os.remove(tmp)

        out.update(dict(
            wet_roi=best['img0_vis'], dry_roi=img_dry_a,
            wet_aligned=img_wet_a, dry_aligned=img_dry_a,
            match_vis=match_vis, score=best_score, verdict=verdict,
            n_raw=best['n_raw'], n_filtered=n_f,
            n_inliers=n_inl, rot_deg=best_rot, elapsed=elapsed,
            mean_conf=best['conf'], inlier_ratio=best['inlier_ratio'],
        ))

        # 匹配点坐标（供前端交互）
        if n_f >= 1:
            kp0 = best['kpts0'][best['matches'][:, 0]].detach().cpu().numpy()
            kp1 = best['kpts1'][best['matches'][:, 1]].detach().cpu().numpy()
            inl_mask = best['inliers']
            if hasattr(best['scores'], 'detach'):
                sc_arr = best['scores'].detach().cpu().float().numpy()
            else:
                sc_arr = np.zeros(n_f)
            out['match_points'] = []
            for i in range(n_f):
                out['match_points'].append({
                    'x0': float(kp0[i, 0]), 'y0': float(kp0[i, 1]),
                    'x1': float(kp1[i, 0]), 'y1': float(kp1[i, 1]),
                    'inlier': bool(inl_mask[i]) if len(inl_mask) > i else False,
                    'conf': float(sc_arr[i]) if i < len(sc_arr) else 0,
                })

    except Exception:
        import traceback
        out['error'] = traceback.format_exc()

    return out
