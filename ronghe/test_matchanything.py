"""
MatchAnything 测试脚本 — 接入完整预处理管线
绕过 PyTorch Lightning，直接加载 EfficientLoFTR 模型推理
"""
import sys, os, time
os.environ['LOGURU_LEVEL'] = 'WARNING'

MA_ROOT = os.path.join(os.path.dirname(__file__), 'MatchAnything_hf', 'imcui', 'third_party', 'MatchAnything')
sys.path.insert(0, MA_ROOT)

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from yacs.config import CfgNode as CN

from pipeline.config import (DEVICE, SAMPLES_DIR, DATA_DIR, OUTPUT_ROOT,
                              MAX_SIZE_WET, MAX_SIZE_DRY, MIN_MATCHES, MATCH_THRESHOLD,
                              get_output_dir, rotate_cv2, _wait_saves)
from pipeline.runner import parse_sample_pairs, parse_data_pairs, _ellipse_from_mask
from pipeline.preprocess import load_and_preprocess
from pipeline.align import align_pair
from pipeline.matching import compute_score_from_points
from pipeline.visualization import draw_matches_vis

# ── 1. Load config & model ──────────────────────────────────────
from src.config.default import get_cfg_defaults

def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}

config = get_cfg_defaults()
config.merge_from_file(os.path.join(MA_ROOT, 'configs', 'models', 'eloftr_model.py'))
config.METHOD = 'matchanything_eloftr'
config.LOFTR.MATCH_COARSE.THR = 0.1
if config.DATASET.NPE_NAME is not None:
    config.LOFTR.COARSE.NPE = [832, 832, 640, 640]

_config = lower_config(config)

from src.loftr import LoFTR

print("Building MatchAnything (EfficientLoFTR) model...")
ma_model = LoFTR(config=_config['loftr'])

ckpt_path = os.path.join(MA_ROOT, 'weights', 'matchanything_eloftr.ckpt')
state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
load_result = ma_model.load_state_dict(state_dict, strict=False)
print(f"Weights loaded: missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)}")

ma_model.eval().to(DEVICE)
print(f"Model on {DEVICE}")


# ── 2. MatchAnything 推理（接受预处理后的灰度图） ──────────────────
MA_MAX_SIZE = 640  # 限制输入尺寸防止 OOM（8GB VRAM）

def _prepare_ma_tensor(gray_img):
    """将灰度图 resize+pad 为 MatchAnything 要求的格式（32 的倍数，正方形 padding）。"""
    h, w = gray_img.shape[:2]
    if max(h, w) > MA_MAX_SIZE:
        scale = MA_MAX_SIZE / max(h, w)
        w_new, h_new = int(w * scale), int(h * scale)
        gray_img = cv2.resize(gray_img, (w_new, h_new), interpolation=cv2.INTER_LANCZOS4)
        h, w = h_new, w_new

    # 对齐到 32 的倍数
    w32 = (w // 32) * 32
    h32 = (h // 32) * 32
    if (w32, h32) != (w, h):
        gray_img = cv2.resize(gray_img, (w32, h32), interpolation=cv2.INTER_LANCZOS4)

    h_scale = h / h32
    w_scale = w / w32

    # 正方形 padding
    pad_size = max(h32, w32)
    padded = np.zeros((pad_size, pad_size), dtype=np.float32)
    padded[:h32, :w32] = gray_img.astype(np.float32)
    valid_mask = np.zeros((pad_size, pad_size), dtype=bool)
    valid_mask[:h32, :w32] = True

    tensor = torch.from_numpy(padded)[None, None] / 255.0
    return tensor, valid_mask, [h_scale, w_scale], (h32, w32)


def run_ma_on_gray_pair(gray0, gray1):
    """对两张灰度图运行 MatchAnything，返回匹配点对和置信度（坐标已映射回输入图尺寸）。"""
    t0, m0, scale0, hw0 = _prepare_ma_tensor(gray0)
    t1, m1, scale1, hw1 = _prepare_ma_tensor(gray1)

    batch = {'image0': t0.to(DEVICE), 'image1': t1.to(DEVICE)}

    # 分别处理 padding mask（两张图 pad 尺寸可能不同）
    tm0 = torch.from_numpy(m0).to(DEVICE).float()
    tm1 = torch.from_numpy(m1).to(DEVICE).float()
    mask0_ds = F.interpolate(tm0[None, None], scale_factor=0.125, mode='nearest',
                             recompute_scale_factor=False)[0, 0].bool()
    mask1_ds = F.interpolate(tm1[None, None], scale_factor=0.125, mode='nearest',
                             recompute_scale_factor=False)[0, 0].bool()
    batch['mask0'] = mask0_ds[None]
    batch['mask1'] = mask1_ds[None]

    with torch.no_grad():
        ma_model(batch)

    mkpts0 = batch['mkpts0_f'].cpu().numpy()
    mkpts1 = batch['mkpts1_f'].cpu().numpy()
    mconf = batch['mconf'].cpu().numpy()

    del batch
    torch.cuda.empty_cache()

    # 映射回裁切后图像的坐标（不是原图，是 ROI 图像坐标）
    if len(mkpts0) > 0:
        mkpts0[:, 0] *= (hw0[1] / ((hw0[1] // 32) * 32)) if hw0[1] > 0 else 1
        mkpts0[:, 1] *= (hw0[0] / ((hw0[0] // 32) * 32)) if hw0[0] > 0 else 1
        mkpts1[:, 0] *= (hw1[1] / ((hw1[1] // 32) * 32)) if hw1[1] > 0 else 1
        mkpts1[:, 1] *= (hw1[0] / ((hw1[0] // 32) * 32)) if hw1[0] > 0 else 1

    return mkpts0, mkpts1, mconf


def _filter_points_np(kpts0, kpts1, conf, mask0, mask1):
    """用 numpy 过滤落在排除区域（贴纸）内的匹配点。"""
    if len(kpts0) == 0:
        return kpts0, kpts1, conf
    keep = []
    for i in range(len(kpts0)):
        y0, x0 = int(round(kpts0[i, 1])), int(round(kpts0[i, 0]))
        y1, x1 = int(round(kpts1[i, 1])), int(round(kpts1[i, 0]))
        in0 = (0 <= y0 < mask0.shape[0] and 0 <= x0 < mask0.shape[1] and mask0[y0, x0] > 0)
        in1 = (0 <= y1 < mask1.shape[0] and 0 <= x1 < mask1.shape[1] and mask1[y1, x1] > 0)
        if not in0 and not in1:
            keep.append(i)
    if not keep:
        return kpts0[:0], kpts1[:0], conf[:0]
    idx = np.array(keep)
    return kpts0[idx], kpts1[idx], conf[idx]


# ── 3. 主测试流程（接入完整管线） ───────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="MatchAnything 混凝土试块匹配测试")
    parser.add_argument('--limit', type=int, default=0, help='只跑前 N 对（0=全部）')
    parser.add_argument('--data', action='store_true', help='使用 data/ 目录')
    parser.add_argument('--conf-thr', type=float, default=0.2, help='置信度过滤阈值')
    args = parser.parse_args()

    method_name = "MatchAnything_v2"

    if args.data:
        pairs = parse_data_pairs(DATA_DIR)
    else:
        pairs = parse_sample_pairs(SAMPLES_DIR)
    if args.limit > 0:
        pairs = pairs[:args.limit]

    print(f"\n{'='*60}")
    print(f"MatchAnything (EfficientLoFTR) — 完整管线测试")
    print(f"  Pairs: {len(pairs)} | conf_thr: {args.conf_thr}")
    print(f"  Threshold={MATCH_THRESHOLD}  MIN_MATCHES={MIN_MATCHES}")
    print(f"{'='*60}")

    results = []
    out_dir = get_output_dir(method_name)

    for batch_id, wet_path, dry_path, specimen_no in pairs:
        label = f"{batch_id}#{specimen_no}"
        print(f"\n--- {method_name} | {label} ---")

        if dry_path is None:
            print(f"  INVALID: QR配对失败")
            results.append({
                'batch': batch_id, 'specimen': specimen_no,
                'method': method_name, 'final_score': 0,
                'verdict': 'INVALID', 'invalid_reason': 'QR配对失败',
            })
            continue

        try:
            # 预处理：ROI 裁切 + 透视矫正 + 贴纸检测
            img0_bgr, tensor0, mask0, center0, stk0, meta0 = load_and_preprocess(
                wet_path, max_size=MAX_SIZE_WET, roi_mode='dino')
            img1_bgr, tensor1, mask1, center1, stk1, meta1 = load_and_preprocess(
                dry_path, max_size=MAX_SIZE_DRY, roi_mode='dino')

            if not stk0:
                print(f"  INVALID: WET贴纸未检测到")
                results.append({'batch': batch_id, 'specimen': specimen_no,
                    'method': method_name, 'final_score': 0, 'verdict': 'INVALID'})
                continue
            if not stk1:
                print(f"  INVALID: DRY贴纸未检测到")
                results.append({'batch': batch_id, 'specimen': specimen_no,
                    'method': method_name, 'final_score': 0, 'verdict': 'INVALID'})
                continue

            t_start = time.time()

            # 对齐：矩形摆正 + 贴纸象限匹配
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

            # 干态灰度（固定）
            gray_dry = cv2.cvtColor(img_dry_a, cv2.COLOR_BGR2GRAY)

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

                # MatchAnything 推理
                mkpts0, mkpts1, mconf = run_ma_on_gray_pair(gray_wet, gray_dry)
                n_raw = len(mkpts0)

                # 过滤低置信度
                if n_raw > 0:
                    high_conf = mconf > args.conf_thr
                    mkpts0 = mkpts0[high_conf]
                    mkpts1 = mkpts1[high_conf]
                    mconf = mconf[high_conf]

                # 过滤贴纸区域
                mkpts0, mkpts1, mconf = _filter_points_np(mkpts0, mkpts1, mconf, m_r, mask_dry_a)
                n_filtered = len(mkpts0)

                if n_filtered >= 4:
                    sc, cf, ir, inl, n_scored = compute_score_from_points(mkpts0, mkpts1, mconf)
                else:
                    sc, cf, ir = 0.0, 0.0, 0.0
                    inl = np.array([])

                if sc > best_score:
                    best_score = sc
                    best_rot = rot_deg
                    best = {
                        'kpts0': mkpts0, 'kpts1': mkpts1, 'conf': cf,
                        'inlier_ratio': ir, 'inliers': inl,
                        'n_raw': n_raw, 'n_filtered': n_filtered,
                        'mask0': m_r, 'rot_method': align_result['method'],
                        'img0_vis': img_r,
                    }

            elapsed = time.time() - t_start
            final_score = best_score
            n_matches = best['n_filtered']
            n_inliers = int(best['inliers'].sum()) if len(best['inliers']) > 0 else 0

            if n_matches < 4:
                verdict = "INSUFFICIENT"
            elif final_score >= MATCH_THRESHOLD:
                verdict = "SAME"
            else:
                verdict = "DIFFERENT"

            print(f"    → 选择 {best_rot:.1f}° ({best['rot_method']}, score={final_score:.3f})")
            print(f"  Raw: {best['n_raw']} -> Filtered: {n_matches} -> Inliers: {n_inliers}")
            print(f"  Conf: {best['conf']:.3f} | InlierR: {best['inlier_ratio']:.3f} | "
                  f"Score: {final_score:.3f} | {verdict} | rot={best_rot:.1f}° ({best['rot_method']})")
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
                'best_rotation': round(float(best_rot), 1),
                'rot_method': best['rot_method'],
                'time_sec': round(elapsed, 2),
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({
                'batch': batch_id, 'specimen': specimen_no,
                'method': method_name, 'error': str(e),
                'final_score': 0, 'verdict': 'ERROR',
            })

    _wait_saves()

    # ── 汇总 ──
    scores = [r['final_score'] for r in results if 'final_score' in r]
    verdicts = [r.get('verdict', '') for r in results]
    same_n = sum(1 for v in verdicts if v == 'SAME')
    diff_n = sum(1 for v in verdicts if v == 'DIFFERENT')
    insuf_n = sum(1 for v in verdicts if v == 'INSUFFICIENT')
    invalid_n = sum(1 for v in verdicts if v == 'INVALID')

    print(f"\n{'='*60}")
    print(f"SUMMARY — {method_name}")
    print(f"{'='*60}")
    print(f"  Pairs: {len(results)} | SAME: {same_n} | DIFFERENT: {diff_n} | "
          f"INSUFFICIENT: {insuf_n} | INVALID: {invalid_n}")
    if scores:
        print(f"  Score: mean={np.mean(scores):.3f}, "
              f"min={np.min(scores):.3f}, max={np.max(scores):.3f}")

    print(f"\n{'Pair':<40} {'Score':>6} {'Raw':>5} {'Filt':>5} {'Verdict':<12}")
    print(f"{'-'*70}")
    for r in results:
        label = f"{r['batch']} #{r['specimen']}"
        print(f"{label:<40} {r['final_score']:>6.3f} "
              f"{r.get('raw_matches',0):>5} {r.get('filtered_matches',0):>5} "
              f"{r['verdict']:<12}")
    print(f"{'-'*70}")
    print(f"SAME: {same_n}/{len(results)} ({100*same_n/len(results):.0f}%)")
