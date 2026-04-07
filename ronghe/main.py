"""混凝土试块造假识别 — 入口"""
import sys
import json
import shutil
import numpy as np
from pathlib import Path

from pipeline.config import (DEVICE, OUTPUT_ROOT, SAMPLES_DIR, DATA_DIR,
                              MATCH_THRESHOLD, MIN_MATCHES, MAX_KEYPOINTS)
from pipeline.runner import (parse_sample_pairs, parse_data_pairs,
                              run_method, run_method_loftr, run_method_roma,
                              run_cross_batch_test, run_intra_batch_cross_test,
                              load_manual_pairs, save_manual_pairs)
from pipeline.qr import load_qr_cache, save_qr_cache
from pipeline.ocr_fallback import load_ocr_cache, save_ocr_cache
from lightglue import LightGlue, SuperPoint, ALIKED, SIFT, DoGHardNet


def _clean_dir(d):
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


def _print_summary(name, res_list):
    scores = [r['final_score'] for r in res_list if 'final_score' in r]
    verdicts = [r.get('verdict', '') for r in res_list if 'verdict' in r]
    same_n = sum(1 for v in verdicts if v == 'SAME')
    diff_n = sum(1 for v in verdicts if v == 'DIFFERENT')
    insuf_n = sum(1 for v in verdicts if v == 'INSUFFICIENT')
    invalid_n = sum(1 for v in verdicts if v == 'INVALID')
    print(f"\n{'='*60}")
    print(f"SUMMARY — {name}")
    print(f"{'='*60}")
    print(f"  Pairs: {len(scores)} | SAME: {same_n} | DIFFERENT: {diff_n} | "
          f"INSUFFICIENT: {insuf_n} | INVALID: {invalid_n}")
    if scores:
        print(f"  Score: mean={np.mean(scores):.3f}, "
              f"min={np.min(scores):.3f}, max={np.max(scores):.3f}")
    return same_n, diff_n, insuf_n, invalid_n


def main():
    import argparse
    parser = argparse.ArgumentParser(description="混凝土试块造假识别 Demo v2")
    parser.add_argument('--limit', type=int, default=0,
                        help='只跑前 N 对（0=全部）')
    parser.add_argument('--cross', action='store_true',
                        help='同时跑跨批次对照测试')
    parser.add_argument('--data', action='store_true',
                        help='使用 data/ 目录（全量数据）而非 samples/')
    parser.add_argument('--methods', type=str, default='all',
                        help='要跑的方法，逗号分隔: sp,aliked,loftr,roma,all')
    parser.add_argument('--batch', type=str, default='',
                        help='只跑指定批次，支持模糊匹配，如 250059 或 宁波大目_250059')
    parser.add_argument('--specimen', type=int, default=0,
                        help='与 --batch 配合，只跑指定编号（1/2/3），0=全部')
    args = parser.parse_args()

    methods = set(args.methods.lower().split(','))
    run_all = 'all' in methods

    print("=" * 60)
    print("混凝土试块造假识别 — 多方法对比")
    print(f"  Threshold={MATCH_THRESHOLD}  MIN_MATCHES={MIN_MATCHES}")
    print("=" * 60)

    if args.data:
        print(f"Samples: {DATA_DIR} (全量)")
        load_qr_cache(); load_ocr_cache(); load_manual_pairs()
        pairs = parse_data_pairs(DATA_DIR)
        save_qr_cache(); save_ocr_cache()
    else:
        print(f"Samples: {SAMPLES_DIR}")
        load_qr_cache(); load_ocr_cache(); load_manual_pairs()
        pairs = parse_sample_pairs(SAMPLES_DIR)
        save_qr_cache(); save_ocr_cache()
    if args.limit > 0:
        pairs = pairs[:args.limit]
    if args.batch:
        pairs = [p for p in pairs if args.batch in p[0]]
        if args.specimen > 0:
            pairs = [p for p in pairs if p[3] == args.specimen]
        if not pairs:
            print(f"⚠️ 未找到匹配 --batch={args.batch} 的配对")
            return
    print(f"\nRunning {len(pairs)} wet/dry pairs:")
    for b, w, d, n in pairs:
        d_name = d.name if d is not None else "INVALID(无匹配)"
        print(f"  {b} #{n}: {w.name} <-> {d_name}")

    all_results = {}
    summaries = {}

    # === 1. SuperPoint + LightGlue ===
    if run_all or 'sp' in methods:
        print(f"\n{'='*60}")
        print("1) SuperPoint + LightGlue")
        print(f"{'='*60}")
        _clean_dir(OUTPUT_ROOT / "SP_LG_v2")
        ext_sp = SuperPoint(max_num_keypoints=MAX_KEYPOINTS).eval().to(DEVICE)
        mat_lg_sp = LightGlue(features='superpoint').eval().to(DEVICE)
        results_sp = run_method("SP_LG_v2", ext_sp, mat_lg_sp, pairs)
        all_results['SuperPoint_LightGlue'] = results_sp
        summaries['SP+LG'] = _print_summary("SuperPoint+LG", results_sp)
        del ext_sp, mat_lg_sp

    # === 2. ALIKED + LightGlue ===
    if run_all or 'aliked' in methods:
        print(f"\n{'='*60}")
        print("2) ALIKED + LightGlue")
        print(f"{'='*60}")
        _clean_dir(OUTPUT_ROOT / "ALIKED_LG_v2")
        ext_ak = ALIKED(max_num_keypoints=MAX_KEYPOINTS).eval().to(DEVICE)
        mat_lg_ak = LightGlue(features='aliked').eval().to(DEVICE)
        results_ak = run_method("ALIKED_LG_v2", ext_ak, mat_lg_ak, pairs)
        all_results['ALIKED_LightGlue'] = results_ak
        summaries['ALIKED+LG'] = _print_summary("ALIKED+LG", results_ak)
        del ext_ak, mat_lg_ak

    # === 3. SIFT + LightGlue ===
    if run_all or 'sift' in methods:
        print(f"\n{'='*60}")
        print("3) SIFT + LightGlue")
        print(f"{'='*60}")
        _clean_dir(OUTPUT_ROOT / "SIFT_LG_v2")
        ext_sift = SIFT(max_num_keypoints=MAX_KEYPOINTS).eval().to(DEVICE)
        mat_lg_sift = LightGlue(features='sift').eval().to(DEVICE)
        results_sift = run_method("SIFT_LG_v2", ext_sift, mat_lg_sift, pairs)
        all_results['SIFT_LightGlue'] = results_sift
        summaries['SIFT+LG'] = _print_summary("SIFT+LG", results_sift)
        del ext_sift, mat_lg_sift

    # === 4. DoGHardNet + LightGlue ===
    if run_all or 'hardnet' in methods:
        print(f"\n{'='*60}")
        print("4) DoGHardNet + LightGlue")
        print(f"{'='*60}")
        _clean_dir(OUTPUT_ROOT / "HardNet_LG_v2")
        ext_hn = DoGHardNet(max_num_keypoints=MAX_KEYPOINTS).eval().to(DEVICE)
        mat_lg_hn = LightGlue(features='doghardnet').eval().to(DEVICE)
        results_hn = run_method("HardNet_LG_v2", ext_hn, mat_lg_hn, pairs)
        all_results['DoGHardNet_LightGlue'] = results_hn
        summaries['HardNet+LG'] = _print_summary("DoGHardNet+LG", results_hn)
        del ext_hn, mat_lg_hn

    # === 5. LoFTR ===
    if run_all or 'loftr' in methods:
        print(f"\n{'='*60}")
        print("5) LoFTR (dense matcher)")
        print(f"{'='*60}")
        _clean_dir(OUTPUT_ROOT / "LoFTR_v2")
        from kornia.feature import LoFTR as KorniaLoFTR
        loftr_model = KorniaLoFTR(pretrained='outdoor').eval().to(DEVICE)
        results_loftr = run_method_loftr("LoFTR_v2", loftr_model, pairs)
        all_results['LoFTR'] = results_loftr
        summaries['LoFTR'] = _print_summary("LoFTR", results_loftr)
        del loftr_model

    # === 6. RoMa ===
    if run_all or 'roma' in methods:
        print(f"\n{'='*60}")
        print("6) RoMa (dense warp matcher)")
        print(f"{'='*60}")
        _clean_dir(OUTPUT_ROOT / "RoMa_v2")
        from romatch import roma_outdoor
        roma_model = roma_outdoor(device=DEVICE)
        results_roma = run_method_roma("RoMa_v2", roma_model, pairs)
        all_results['RoMa'] = results_roma
        summaries['RoMa'] = _print_summary("RoMa", results_roma)
        del roma_model

    # === 对比总表 ===
    print(f"\n{'='*60}")
    print("对比总表")
    print(f"{'='*60}")
    print('{:<15} {:>6} {:>6} {:>6} {:>6}'.format(
        'Method', 'SAME', 'DIFF', 'INSUF', 'INVLD'))
    print('-'*50)
    for name, (s, d, i, inv) in summaries.items():
        print('{:<15} {:>6} {:>6} {:>6} {:>6}'.format(name, s, d, i, inv))

    # 逐对对比
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("逐对对比")
        print(f"{'='*60}")
        method_names = list(all_results.keys())
        header = '{:<35}'.format('Pair')
        for mn in method_names:
            short = mn.replace('_LightGlue', '+LG').replace('SuperPoint', 'SP')
            header += ' {:>8} {:>8}'.format(short[:8], 'verdict')
        print(header)
        print('-' * len(header))

        n_pairs = len(list(all_results.values())[0])
        for idx in range(n_pairs):
            row_data = []
            for mn in method_names:
                r = all_results[mn][idx]
                row_data.append((r.get('final_score', 0), r.get('verdict', 'ERR')))
            label = '{}#{}'.format(row_data[0] and all_results[method_names[0]][idx].get('batch','?'),
                                   all_results[method_names[0]][idx].get('specimen','?'))
            line = '{:<35}'.format(label)
            for sc, v in row_data:
                line += ' {:>8.4f} {:>8}'.format(sc, v)
            print(line)

    # === Cross-batch (optional) ===
    if args.cross and 'SuperPoint_LightGlue' in all_results:
        ext_sp = SuperPoint(max_num_keypoints=MAX_KEYPOINTS).eval().to(DEVICE)
        mat_lg = LightGlue(features='superpoint').eval().to(DEVICE)
        cross = run_cross_batch_test(ext_sp, mat_lg, pairs, "SP_LG_v2")
        all_results['cross_batch'] = cross
        if cross:
            cs = [r['score'] for r in cross]
            cross_verdicts = [r['verdict'] for r in cross]
            false_pos = sum(1 for v in cross_verdicts if v == 'SAME')
            print(f"\nCross-batch:")
            print(f"  Score: mean={np.mean(cs):.3f}, "
                  f"min={np.min(cs):.3f}, max={np.max(cs):.3f}")
            print(f"  False positives: {false_pos}/{len(cs)}")

    # 同批交叉验证（对全部SAME的批次）
    if 'SuperPoint_LightGlue' in all_results:
        sp_results = all_results['SuperPoint_LightGlue']
        ext_sp2 = SuperPoint(max_num_keypoints=MAX_KEYPOINTS).eval().to(DEVICE)
        mat_lg2 = LightGlue(features='superpoint').eval().to(DEVICE)
        intra_cross = run_intra_batch_cross_test(
            ext_sp2, mat_lg2, sp_results, "SP_LG_v2")
        if intra_cross:
            all_results['intra_batch_cross'] = intra_cross

    # 保存结果
    rpath = OUTPUT_ROOT / "results_v2.json"
    if args.batch:
        # --batch 模式：只更新匹配的记录，不覆盖其余结果
        existing = {}
        if rpath.exists():
            try:
                existing = json.loads(rpath.read_text(encoding='utf-8'))
            except Exception:
                pass
        for method_key, new_list in all_results.items():
            if method_key not in existing:
                existing[method_key] = []
            # 用新结果替换 existing 中对应 batch+specimen 的记录
            new_index = {(r['batch'], r['specimen']): r for r in new_list}
            merged = []
            replaced = set()
            for old_r in existing[method_key]:
                key = (old_r['batch'], old_r['specimen'])
                if key in new_index:
                    merged.append(new_index[key])
                    replaced.add(key)
                else:
                    merged.append(old_r)
            # 新增不在 existing 中的记录
            for key, r in new_index.items():
                if key not in replaced:
                    merged.append(r)
            existing[method_key] = merged
        with open(rpath, 'w', encoding='utf-8') as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        print(f"\nResults (merged) → {rpath}")
    else:
        with open(rpath, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nResults → {rpath}")

    # 自动生成 report 目录（按判定分类 match 图片）
    _generate_report(all_results, OUTPUT_ROOT)


def _generate_report(all_results, output_root):
    """将 SP_LG_v2 的 match 图片按判定结果分类到 report/ 目录"""
    import shutil
    sp = all_results.get('SuperPoint_LightGlue', [])
    if not sp:
        return
    report_dir = output_root / "report"
    if report_dir.exists():
        shutil.rmtree(report_dir)
    src_root = output_root / "SP_LG_v2"
    copied = 0
    for r in sp:
        if 'error' in r or 'verdict' not in r:
            continue
        batch = r['batch']
        spec = r['specimen']
        verdict = r['verdict']
        src_file = src_root / batch / f"match_{spec}.png"
        if src_file.exists():
            dst = report_dir / verdict / batch
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst / src_file.name)
            copied += 1
    print(f"\nReport: {copied} files → {report_dir}")


if __name__ == '__main__':
    main()
