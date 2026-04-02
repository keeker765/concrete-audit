"""
混凝土试块图片标注工具 — 本地 HTTP 服务
启动后浏览器访问 http://localhost:8765
功能：
  - 展示所有干湿配对的原图 + 匹配结果图（QR码配对）
  - 勾选标记每对的质量（好/一般/差）
  - 标注保存到 labels.json
"""
import http.server
import json
import os
import re
import sys
import urllib.parse
import base64
import mimetypes
from pathlib import Path
from collections import defaultdict

PORT = 8765
PROJECT_ROOT = Path(__file__).parent
SAMPLES_DIR = PROJECT_ROOT / "samples"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output_v2"
RESULTS_FILE = OUTPUT_DIR / "results_v2.json"
LABELS_FILE = PROJECT_ROOT / "labels.json"

# 全局缓存（进程内），避免每次请求重新扫描
_PAIRS_CACHE = None
_PAIRS_LOCK = __import__('threading').Lock()
_MATCH_INDEX = None   # {batch_id: {specimen_no: {method: path}}}

# 磁盘持久化缓存文件
_PAIRS_DISK_CACHE = OUTPUT_DIR / "pairs_cache.json"
_MATCH_DISK_CACHE = OUTPUT_DIR / "match_index_cache.json"


def _results_mtime():
    """返回 results_v2.json 的修改时间戳，用于缓存失效判断"""
    try:
        return RESULTS_FILE.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def _load_pairs_from_disk():
    """从磁盘读取 pairs_cache，若 results 已更新则返回 None（需重建）"""
    if not _PAIRS_DISK_CACHE.exists():
        return None
    try:
        with open(_PAIRS_DISK_CACHE, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        if obj.get('results_mtime') != _results_mtime():
            return None          # results 已更新，缓存失效
        return obj['pairs']
    except Exception:
        return None


def _save_pairs_to_disk(pairs):
    try:
        OUTPUT_DIR.mkdir(exist_ok=True)
        with open(_PAIRS_DISK_CACHE, 'w', encoding='utf-8') as f:
            json.dump({'results_mtime': _results_mtime(), 'pairs': pairs},
                      f, ensure_ascii=False)
    except Exception as e:
        print(f"[warn] pairs_cache 写盘失败: {e}")


def _load_match_index_from_disk():
    """从磁盘读取 match_index_cache"""
    if not _MATCH_DISK_CACHE.exists():
        return None
    try:
        with open(_MATCH_DISK_CACHE, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        # match_index 的 key 是 bname(str)，value 是 {str: {str: str}}
        # json 序列化后 specimen_no 变 str，需转回 int
        index = {}
        for bname, specs in obj.items():
            index[bname] = {int(k): v for k, v in specs.items()}
        return index
    except Exception:
        return None


def _save_match_index_to_disk(index):
    try:
        OUTPUT_DIR.mkdir(exist_ok=True)
        # specimen_no (int) → str for JSON
        serializable = {bname: {str(k): v for k, v in specs.items()}
                        for bname, specs in index.items()}
        with open(_MATCH_DISK_CACHE, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, ensure_ascii=False)
    except Exception as e:
        print(f"[warn] match_index_cache 写盘失败: {e}")

# 导入 pipeline 的 QR 解码（含磁盘缓存）
sys.path.insert(0, str(PROJECT_ROOT))
import cv2
from pipeline.qr import decode_qr_content, load_qr_cache, save_qr_cache

# 加载已有标注
def load_labels():
    if LABELS_FILE.exists():
        with open(LABELS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_labels(labels):
    with open(LABELS_FILE, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

def _get_specimen_id(filepath):
    """解码图片QR码，提取试块编号（优先走磁盘缓存，缓存命中无需读图）"""
    from pipeline.qr import _QR_DISK_CACHE
    cache_key = str(filepath)
    if cache_key in _QR_DISK_CACHE:
        content = _QR_DISK_CACHE[cache_key]
    else:
        img = cv2.imread(cache_key)
        if img is None:
            return None
        content = decode_qr_content(img, filepath=filepath)
    if content is None:
        return None
    m = re.search(r'-(\d+)$', content)
    return m.group(1) if m else None

def _get_qr_content(filepath):
    """返回图片的完整 QR 内容字符串（优先走磁盘缓存，缓存命中无需读图）"""
    from pipeline.qr import _QR_DISK_CACHE
    cache_key = str(filepath)
    if cache_key in _QR_DISK_CACHE:
        return _QR_DISK_CACHE[cache_key]
    img = cv2.imread(cache_key)
    if img is None:
        return None
    return decode_qr_content(img, filepath=filepath)


# 解析配对 — 从 results_v2.json 获取已跑的配对，再找回原图
def get_pairs():
    global _PAIRS_CACHE
    if _PAIRS_CACHE is not None:
        return _PAIRS_CACHE
    with _PAIRS_LOCK:         # 防止多线程重复 build
        if _PAIRS_CACHE is None:
            cached = _load_pairs_from_disk()
            if cached is not None:
                print(f"[cache] pairs_cache 磁盘命中 ({len(cached)} 对)")
                _PAIRS_CACHE = cached
            else:
                _PAIRS_CACHE = _build_pairs()
                _save_pairs_to_disk(_PAIRS_CACHE)
    return _PAIRS_CACHE

def _build_pairs():
    # 先加载 QR 磁盘缓存，避免对每张图重复解码（提速关键）
    try:
        load_qr_cache()
    except Exception:
        pass

    results = load_results()
    if not results:
        return []
    
    # 取第一个方法的结果作为配对列表
    first_method = list(results.keys())[0]
    method_results = results[first_method]
    if not isinstance(method_results, list):
        return []
    
    # 构建 batch→specimen 列表
    seen = set()
    pair_keys = []
    for r in method_results:
        key = (r['batch'], r['specimen'])
        if key not in seen:
            seen.add(key)
            pair_keys.append(key)
    
    # 查找原图：先在 data/ 再在 samples/ 中搜索
    def _find_images_for_batch(batch_id):
        """找到某个 batch_id 对应的所有湿/干图片"""
        wet_files, dry_files = [], []
        
        # 在 data/ 目录中搜索
        if DATA_DIR.exists():
            for company_dir in DATA_DIR.iterdir():
                if not company_dir.is_dir():
                    continue
                for batch_dir in company_dir.iterdir():
                    if not batch_dir.is_dir():
                        continue
                    # 匹配 batch_id（可能是简写）
                    short_name = company_dir.name[:4]
                    m = re.search(r'压-(.+)$', batch_dir.name)
                    batch_num = m.group(1) if m else batch_dir.name
                    dir_batch_id = f"{short_name}_{batch_num}"
                    if dir_batch_id == batch_id:
                        wet_files = sorted(batch_dir.glob("*.jpeg"))
                        dry_files = sorted(batch_dir.glob("*.jpg"))
                        return wet_files, dry_files
        
        # 在 samples/ 目录中搜索
        if SAMPLES_DIR.exists():
            wet_files = sorted(SAMPLES_DIR.glob(f"{batch_id}_wet*.jpeg"))
            dry_files = sorted(SAMPLES_DIR.glob(f"{batch_id}_dry*.jpg"))
        
        return wet_files, dry_files
    
    # 缓存 batch → (wet_by_qr, dry_by_qr, wet_sorted, dry_sorted)
    batch_images = {}
    unique_batches = list(dict.fromkeys(b for b, _ in pair_keys))
    
    def _get_batch_images(batch_id):
        if batch_id in batch_images:
            return batch_images[batch_id]
        
        wet_files, dry_files = _find_images_for_batch(batch_id)
        
        # QR 配对（利用磁盘缓存，已解码的不会重复解码）
        wet_by_qr = {}
        dry_by_qr = {}
        for f in wet_files:
            sid = _get_specimen_id(f)
            if sid is not None:
                wet_by_qr[sid] = f
        for f in dry_files:
            sid = _get_specimen_id(f)
            if sid is not None:
                dry_by_qr[sid] = f
        
        result = (wet_by_qr, dry_by_qr, wet_files, dry_files)
        batch_images[batch_id] = result
        return result
    
    # 预先处理所有 batch（显示进度）
    for i, batch_id in enumerate(unique_batches):
        print(f"  [{i+1}/{len(unique_batches)}] {batch_id}", flush=True)
        _get_batch_images(batch_id)
    
    pairs = []
    for batch_id, specimen_no in pair_keys:
        wet_by_qr, dry_by_qr, wet_sorted, dry_sorted = _get_batch_images(batch_id)

        # 湿态：按文件名顺序取（与 runner.py 的 specimen_no 逻辑一致）
        wet_idx = specimen_no - 1
        wet_path = str(wet_sorted[wet_idx]) if wet_idx < len(wet_sorted) else ''

        # 从湿态图的 QR 获取真实 QR ID，再用它找干态（QR 匹配成功才配对）
        wet_actual_qr = _get_specimen_id(wet_sorted[wet_idx]) if wet_path else None
        if wet_actual_qr is not None and wet_actual_qr in dry_by_qr:
            dry_path = str(dry_by_qr[wet_actual_qr])
            tag = f'QR-{wet_actual_qr}'
        else:
            # QR 配对失败 → 干态显示缺失，不用索引兜底
            dry_path = ''
            tag = 'QR配对失败'

        # 读取完整 QR 字符串供显示（走磁盘缓存，不额外解码）
        wet_qr = _get_qr_content(wet_path) if wet_path else None
        dry_qr = _get_qr_content(dry_path) if dry_path else None

        pairs.append({
            'batch': batch_id, 'specimen': specimen_no,
            'wet_path': wet_path,
            'dry_path': dry_path,
            'wet_name': Path(wet_path).name if wet_path else '缺失',
            'dry_name': Path(dry_path).name if dry_path else '缺失（QR配对失败）',
            'pair_method': tag,
            'wet_qr': wet_qr or '未解码',
            'dry_qr': dry_qr or '未解码',
        })
    
    return pairs

# 查找匹配结果图
def _build_match_index():
    """扫描 output_v2/ 一次，建立 {batch_id: {specimen_no: {method: path}}} 索引"""
    cached = _load_match_index_from_disk()
    if cached is not None:
        print(f"[cache] match_index 磁盘命中 ({len(cached)} 个 batch)")
        return cached
    index = {}
    if not OUTPUT_DIR.exists():
        return index
    for method_dir in OUTPUT_DIR.iterdir():
        if not method_dir.is_dir() or method_dir.name in ('debug',):
            continue
        for batch_dir in method_dir.iterdir():
            if not batch_dir.is_dir():
                continue
            for f in batch_dir.glob("match_*.png"):
                try:
                    spec_no = int(f.stem.split('_')[1])
                except (IndexError, ValueError):
                    continue
                bname = batch_dir.name
                if bname not in index:
                    index[bname] = {}
                if spec_no not in index[bname]:
                    index[bname][spec_no] = {}
                index[bname][spec_no][method_dir.name] = str(f)
    _save_match_index_to_disk(index)
    return index

def find_match_images(batch_id, specimen_no):
    global _MATCH_INDEX
    if _MATCH_INDEX is None:
        _MATCH_INDEX = _build_match_index()
    results = {}
    for bname, specs in _MATCH_INDEX.items():
        if batch_id in bname:
            if specimen_no in specs:
                results.update(specs[specimen_no])
    return results

# 加载结果数据
def load_results():
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def img_to_base64(path):
    """将图片转为 base64 data URI"""
    if not os.path.exists(path):
        return ""
    mime = mimetypes.guess_type(path)[0] or 'image/png'
    with open(path, 'rb') as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{data}"

def build_html(page=1, page_size=15, filter_verdict='all', filter_labeled='all', filter_invalid_reason='all'):
    all_pairs = get_pairs()
    results = load_results()
    labels = load_labels()

    # 构建结果查找表（只使用 SuperPoint_LightGlue）
    DISPLAY_METHOD = 'SuperPoint_LightGlue'
    score_lookup = {}
    for method_name, method_results in results.items():
        if method_name != DISPLAY_METHOD:
            continue
        if not isinstance(method_results, list):
            continue
        for r in method_results:
            key = f"{r.get('batch','')}__{r.get('specimen','')}"
            if key not in score_lookup:
                score_lookup[key] = {}
            score_lookup[key][method_name] = r

    # 过滤
    def _verdict(pair):
        pid = f"{pair['batch']}__{pair['specimen']}"
        scores = score_lookup.get(pid, {})
        for r in scores.values():
            return r.get('verdict', 'UNKNOWN')
        return 'UNKNOWN'

    def _invalid_reason(pair):
        pid = f"{pair['batch']}__{pair['specimen']}"
        scores = score_lookup.get(pid, {})
        for r in scores.values():
            return r.get('invalid_reason', '')
        return ''

    filtered_pairs = all_pairs
    if filter_verdict != 'all':
        filtered_pairs = [p for p in filtered_pairs if _verdict(p) == filter_verdict]
    if filter_invalid_reason != 'all':
        filtered_pairs = [p for p in filtered_pairs if _invalid_reason(p) == filter_invalid_reason]
    if filter_labeled == 'unlabeled':
        filtered_pairs = [p for p in filtered_pairs if f"{p['batch']}__{p['specimen']}" not in labels]
    elif filter_labeled == 'labeled':
        filtered_pairs = [p for p in filtered_pairs if f"{p['batch']}__{p['specimen']}" in labels]

    total = len(filtered_pairs)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = max(1, min(page, total_pages))
    start = (page - 1) * page_size
    pairs = filtered_pairs[start:start + page_size]
    labeled_count = sum(1 for p in all_pairs if f"{p['batch']}__{p['specimen']}" in labels)

    cards_html = ""
    for pair in pairs:
        pair_id = f"{pair['batch']}__{pair['specimen']}"
        label = labels.get(pair_id, {})
        quality = label.get('quality', '')
        notes = label.get('notes', '')
        
        # 分数信息
        scores_info = score_lookup.get(pair_id, {})
        scores_html = ""
        for method, r in scores_info.items():
            score = r.get('final_score', 0)
            verdict = r.get('verdict', '?')
            short_method = method.replace('_LightGlue', '+LG').replace('SuperPoint', 'SP')
            color = '#4CAF50' if verdict == 'SAME' else ('#f44336' if verdict == 'DIFFERENT' else '#ff9800')
            badge_text = f'{short_method}: {score:.3f} ({verdict})'
            if verdict == 'INVALID':
                reason = r.get('invalid_reason', '')
                if reason:
                    badge_text = f'{short_method}: INVALID — {reason}'
            scores_html += f'<span class="score-badge" style="background:{color}">{badge_text}</span> '
        
        # 匹配结果图
        match_imgs = find_match_images(pair['batch'], pair['specimen'])
        match_html = ""
        for method, path in match_imgs.items():
            match_html += f'''
            <div class="match-img-container">
                <div class="method-label">{method}</div>
                <img src="/img?path={urllib.parse.quote(path)}" class="match-img" loading="lazy"
                     onclick="showLb(this.src)">
            </div>'''
        
        checked = {
            'good': 'checked' if quality == 'good' else '',
            'ok': 'checked' if quality == 'ok' else '',
            'bad': 'checked' if quality == 'bad' else '',
        }
        wet_has_text = label.get('wet_has_text', '')  # 'yes'/'no'/''
        dry_has_text = label.get('dry_has_text', '')
        
        def _text_btn(side, current_val):
            """生成有字/无字按钮组"""
            yes_cls = 'active' if current_val == 'yes' else ''
            no_cls = 'active' if current_val == 'no' else ''
            side_label = '湿态' if side == 'wet' else '干态'
            return f'''<div class="text-btn-group">
                <span class="text-btn-label">有字?</span>
                <button class="text-btn yes {yes_cls}" onclick="saveTextFlag('{pair_id}','{side}','yes',this)">✍ 有字</button>
                <button class="text-btn no {no_cls}" onclick="saveTextFlag('{pair_id}','{side}','no',this)">◻ 无字</button>
            </div>'''
        
        wet_qr_display = pair.get('wet_qr', '?')
        pair_method = pair.get('pair_method', '?')
        tag_class = 'pair-tag warn' if '⚠' in pair_method else 'pair-tag'
        dry_qr_display = pair.get('dry_qr', '?')
        wet_img_html = f'''<img src="/img?path={urllib.parse.quote(pair['wet_path'])}" class="orig-img" loading="lazy"
                         onclick="showLb(this.src)">''' if pair['wet_path'] else '<div class="missing-img">图片缺失</div>'
        dry_img_html = f'''<img src="/img?path={urllib.parse.quote(pair['dry_path'])}" class="orig-img" loading="lazy"
                         onclick="showLb(this.src)">''' if pair['dry_path'] else '<div class="missing-img">图片缺失</div>'
        
        cards_html += f'''
        <div class="pair-card" data-pair-id="{pair_id}">
            <div class="pair-header">
                <h3>{pair['batch']} #{pair['specimen']} <span class="{tag_class}">{pair_method}</span></h3>
                <div class="scores">{scores_html}</div>
            </div>
            <div class="images-row">
                <div class="orig-img-container">
                    <div class="img-label">WET (湿态)</div>
                    {wet_img_html}
                    <div class="filename">{pair['wet_name']}</div>
                    <div class="qr-content" title="QR内容">📷 {wet_qr_display}</div>
                    {_text_btn('wet', wet_has_text)}
                </div>
                <div class="orig-img-container">
                    <div class="img-label">DRY (干态)</div>
                    {dry_img_html}
                    <div class="filename">{pair['dry_name']}</div>
                    <div class="qr-content" title="QR内容">📷 {dry_qr_display}</div>
                    {_text_btn('dry', dry_has_text)}
                </div>
            </div>
            <div class="match-results">{match_html}</div>
            <div class="label-row">
                <label class="radio-label good"><input type="radio" name="q_{pair_id}" value="good" {checked['good']} onchange="saveLabel('{pair_id}', this.value)"> ✅ 好（同块，匹配正确）</label>
                <label class="radio-label ok"><input type="radio" name="q_{pair_id}" value="ok" {checked['ok']} onchange="saveLabel('{pair_id}', this.value)"> ⚠️ 一般（同块，但匹配有问题）</label>
                <label class="radio-label bad"><input type="radio" name="q_{pair_id}" value="bad" {checked['bad']} onchange="saveLabel('{pair_id}', this.value)"> ❌ 差（不确定/匹配失败）</label>
                <input type="text" class="notes-input" placeholder="备注..." value="{notes}" 
                       onchange="saveNote('{pair_id}', this.value)">
            </div>
        </div>'''
    
    # 分页导航（携带过滤参数）
    def _qs(p):
        return f"/?page={p}&verdict={filter_verdict}&labeled={filter_labeled}&invalid_reason={urllib.parse.quote(filter_invalid_reason)}"

    def _page_btn(p, label):
        active = 'style="background:#00d2ff;color:#000"' if p == page else ''
        return f'<a href="{_qs(p)}" class="page-btn" {active}>{label}</a>'
    
    page_nav = '<div class="pagination">'
    if page > 1:
        page_nav += _page_btn(1, '«') + _page_btn(page-1, '‹')
    for p in range(max(1, page-2), min(total_pages, page+2)+1):
        page_nav += _page_btn(p, str(p))
    if page < total_pages:
        page_nav += _page_btn(page+1, '›') + _page_btn(total_pages, '»')
    page_nav += '</div>'

    # 统计各 invalid_reason 数量（全量，不受当前filter影响）
    def _get_invalid_reason_all(pair):
        pid = f"{pair['batch']}__{pair['specimen']}"
        scores = score_lookup.get(pid, {})
        for r in scores.values():
            return r.get('invalid_reason', '')
        return ''
    n_inv_wet  = sum(1 for p in all_pairs if _get_invalid_reason_all(p) == 'WET贴纸未检测到')
    n_inv_dry  = sum(1 for p in all_pairs if _get_invalid_reason_all(p) == 'DRY贴纸未检测到')
    n_inv_qr   = sum(1 for p in all_pairs if _get_invalid_reason_all(p) == 'QR配对失败')

    # 过滤控件 HTML
    def _flink(verdict='all', labeled='all', label='', inv_reason='all'):
        active = ('class="filter-btn active"'
                  if verdict == filter_verdict and labeled == filter_labeled and inv_reason == filter_invalid_reason
                  else 'class="filter-btn"')
        enc_r = urllib.parse.quote(inv_reason)
        return f'<a href="/?page=1&verdict={verdict}&labeled={labeled}&invalid_reason={enc_r}" {active}>{label}</a>'

    filter_bar = f'''<div class="filter-bar">
        <span style="color:#888;font-size:13px">筛选：</span>
        {_flink('all','all','全部')}
        {_flink('SAME','all','SAME')}
        {_flink('DIFFERENT','all','DIFFERENT')}
        {_flink('INSUFFICIENT','all','INSUFFICIENT')}
        {_flink('INVALID','all','INVALID')}
        &nbsp;&nbsp;
        <span style="color:#888;font-size:13px">INVALID原因：</span>
        {_flink('INVALID','all',f'WET贴纸未检测 ({n_inv_wet})','WET贴纸未检测到')}
        {_flink('INVALID','all',f'DRY贴纸未检测 ({n_inv_dry})','DRY贴纸未检测到')}
        {_flink('INVALID','all',f'QR配对失败 ({n_inv_qr})','QR配对失败')}
        &nbsp;|&nbsp;
        {_flink('all','unlabeled','未标注')}
        {_flink('all','labeled','已标注')}
    </div>'''

    return f'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>混凝土试块标注工具</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, 'Microsoft YaHei', sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 20px; }}
h1 {{ text-align: center; margin-bottom: 20px; color: #00d2ff; }}
.stats {{ text-align: center; margin-bottom: 20px; color: #888; }}
.pair-tag {{ font-size: 12px; color: #4CAF50; background: #1a3a1e; padding: 2px 8px; border-radius: 6px; margin-left: 8px; }}
.pair-card {{ background: #16213e; border-radius: 12px; padding: 20px; margin-bottom: 24px; border: 1px solid #0f3460; }}
.pair-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px; }}
.pair-header h3 {{ color: #00d2ff; font-size: 18px; }}
.scores {{ display: flex; gap: 6px; flex-wrap: wrap; }}
.score-badge {{ padding: 4px 10px; border-radius: 12px; color: white; font-size: 13px; font-weight: bold; }}
.images-row {{ display: flex; gap: 16px; margin-bottom: 12px; flex-wrap: wrap; }}
.orig-img-container {{ flex: 1; min-width: 200px; text-align: center; }}
.img-label {{ font-weight: bold; margin-bottom: 4px; color: #aaa; }}
.orig-img {{ max-width: 100%; max-height: 300px; border-radius: 8px; cursor: zoom-in; transition: all 0.3s; }}
.orig-img.expanded {{ max-height: 800px; max-width: 100%; }}
.filename {{ font-size: 11px; color: #666; margin-top: 4px; }}
.missing-img {{ width: 200px; height: 150px; background: #2a2a3e; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #666; margin: 0 auto; }}
.match-results {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }}
.match-img-container {{ flex: 1; min-width: 300px; text-align: center; }}
.method-label {{ font-size: 12px; color: #aaa; margin-bottom: 4px; }}
.match-img {{ max-width: 100%; max-height: 250px; border-radius: 8px; cursor: zoom-in; transition: all 0.3s; }}
.match-img.expanded {{ max-height: 800px; }}
.label-row {{ display: flex; gap: 16px; align-items: center; flex-wrap: wrap; padding-top: 12px; border-top: 1px solid #0f3460; }}
.radio-label {{ cursor: pointer; padding: 6px 14px; border-radius: 8px; background: #1a1a3e; transition: all 0.2s; }}
.radio-label:hover {{ background: #2a2a5e; }}
.radio-label.good input:checked ~ {{ }} 
.radio-label input {{ margin-right: 6px; }}
.notes-input {{ flex: 1; min-width: 150px; padding: 6px 12px; border-radius: 8px; border: 1px solid #0f3460; background: #1a1a3e; color: #e0e0e0; font-size: 14px; }}
.notes-input:focus {{ outline: none; border-color: #00d2ff; }}
.text-btn-group {{ display: flex; align-items: center; gap: 6px; margin-top: 8px; justify-content: center; }}
.text-btn-label {{ font-size: 12px; color: #888; }}
.text-btn {{ padding: 4px 12px; border-radius: 6px; border: 1px solid #0f3460; background: #1a1a3e; color: #aaa; cursor: pointer; font-size: 13px; transition: all 0.2s; }}
.text-btn:hover {{ background: #2a2a5e; }}
.text-btn.yes.active {{ background: #e65100; color: white; border-color: #e65100; }}
.text-btn.no.active {{ background: #2e7d32; color: white; border-color: #2e7d32; }}
.qr-content {{ font-size: 11px; color: #6af; margin: 4px 0; font-family: monospace; word-break: break-all; }}
.pair-tag.warn {{ background: #3a2000; color: #ffb74d; border: 1px solid #ffb74d; }}
.save-status {{ position: fixed; top: 20px; right: 20px; background: #4CAF50; color: white; padding: 8px 16px; border-radius: 8px; display: none; z-index: 999; }}
#lb {{ display:none; position:fixed; inset:0; background:rgba(0,0,0,.88); z-index:9999; align-items:center; justify-content:center; cursor:zoom-out; }}
#lb.show {{ display:flex; }}
#lb img {{ max-width:92vw; max-height:92vh; object-fit:contain; border-radius:6px; transform-origin:center; transition:transform .12s; }}
#lb-close {{ position:fixed; top:14px; right:20px; color:#fff; font-size:38px; cursor:pointer; user-select:none; line-height:1; z-index:10000; }}
#lb-hint {{ position:fixed; bottom:14px; left:50%; transform:translateX(-50%); color:#aaa; font-size:12px; pointer-events:none; }}
.pagination {{ display: flex; justify-content: center; gap: 8px; margin: 20px 0; flex-wrap: wrap; }}
.page-btn {{ padding: 8px 14px; border-radius: 8px; background: #16213e; color: #e0e0e0; text-decoration: none; border: 1px solid #0f3460; transition: all 0.2s; }}
.page-btn:hover {{ background: #0f3460; }}
.filter-bar {{ display: flex; justify-content: center; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; align-items: center; }}
.filter-btn {{ padding: 6px 14px; border-radius: 8px; background: #16213e; color: #aaa; text-decoration: none; border: 1px solid #0f3460; font-size: 13px; transition: all 0.2s; }}
.filter-btn:hover {{ background: #0f3460; color: white; }}
.filter-btn.active {{ background: #00d2ff; color: #000; border-color: #00d2ff; font-weight: bold; }}
</style>
</head>
<body>
<!-- Lightbox -->
<div id="lb" onclick="if(event.target===this)hideLb()">
  <span id="lb-close" onclick="hideLb()">&#x2715;</span>
  <img id="lb-img" src="" draggable="false">
  <div id="lb-hint">滚轮缩放 · Esc 或点击空白关闭</div>
</div>
<h1>🧱 混凝土试块匹配标注工具</h1>
<div class="stats">共 {len(all_pairs)} 对（当前筛选: {total}）| 已标注 {labeled_count} 对 | 第 {page}/{total_pages} 页 | 点击图片可放大</div>
{filter_bar}
{page_nav}
<div class="save-status" id="saveStatus">✓ 已保存</div>
{cards_html}
{page_nav}
<script>
function saveLabel(pairId, quality) {{
    fetch('/save', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{pair_id: pairId, quality: quality}})
    }}).then(r => r.json()).then(d => {{
        let s = document.getElementById('saveStatus');
        s.style.display = 'block';
        setTimeout(() => s.style.display = 'none', 1500);
    }});
}}
function saveNote(pairId, notes) {{
    fetch('/save', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{pair_id: pairId, notes: notes}})
    }}).then(r => r.json()).then(d => {{
        let s = document.getElementById('saveStatus');
        s.style.display = 'block';
        setTimeout(() => s.style.display = 'none', 1500);
    }});
}}
function saveTextFlag(pairId, side, value, btn) {{
    // side = 'wet' or 'dry', value = 'yes' or 'no'
    let field = side + '_has_text';
    let data = {{pair_id: pairId}};
    data[field] = value;
    fetch('/save', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(data)
    }}).then(r => r.json()).then(d => {{
        // 更新按钮状态
        let group = btn.parentElement;
        group.querySelectorAll('.text-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        let s = document.getElementById('saveStatus');
        s.style.display = 'block';
        setTimeout(() => s.style.display = 'none', 1500);
    }});
}}
var _lbScale = 1;
function showLb(src) {{
  _lbScale = 1;
  var img = document.getElementById('lb-img');
  img.style.transform = 'scale(1)';
  img.src = src;
  document.getElementById('lb').classList.add('show');
}}
function hideLb() {{
  document.getElementById('lb').classList.remove('show');
}}
document.getElementById('lb').addEventListener('wheel', function(e) {{
  e.preventDefault();
  _lbScale = Math.min(10, Math.max(0.3, _lbScale * (e.deltaY < 0 ? 1.15 : 0.87)));
  document.getElementById('lb-img').style.transform = 'scale(' + _lbScale + ')';
}}, {{passive:false}});
document.addEventListener('keydown', function(e) {{ if(e.key==='Escape') hideLb(); }});
</script>
</body>
</html>'''


class LabelHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # 静默日志
    
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        
        if parsed.path == '/' or parsed.path == '':
            params = urllib.parse.parse_qs(parsed.query)
            page = int(params.get('page', ['1'])[0])
            filter_verdict = params.get('verdict', ['all'])[0]
            filter_labeled = params.get('labeled', ['all'])[0]
            filter_invalid_reason = params.get('invalid_reason', ['all'])[0]
            html = build_html(page=page, filter_verdict=filter_verdict, filter_labeled=filter_labeled, filter_invalid_reason=filter_invalid_reason)
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))
        
        elif parsed.path == '/img':
            params = urllib.parse.parse_qs(parsed.query)
            img_path = params.get('path', [''])[0]
            if os.path.exists(img_path):
                mime = mimetypes.guess_type(img_path)[0] or 'application/octet-stream'
                self.send_response(200)
                self.send_header('Content-Type', mime)
                self.send_header('Cache-Control', 'max-age=3600')
                self.end_headers()
                with open(img_path, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
        
        elif parsed.path == '/labels':
            labels = load_labels()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(labels, ensure_ascii=False).encode('utf-8'))
        
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/save':
            length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(length))
            
            labels = load_labels()
            pair_id = body.get('pair_id', '')
            if pair_id not in labels:
                labels[pair_id] = {}
            
            if 'quality' in body:
                labels[pair_id]['quality'] = body['quality']
            if 'notes' in body:
                labels[pair_id]['notes'] = body['notes']
            if 'wet_has_text' in body:
                labels[pair_id]['wet_has_text'] = body['wet_has_text']
            if 'dry_has_text' in body:
                labels[pair_id]['dry_has_text'] = body['dry_has_text']
            
            save_labels(labels)
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == '__main__':
    # 加载共享磁盘QR缓存（main.py跑过后已有大量缓存）
    load_qr_cache()
    
    # 启动时预计算配对
    print("正在配对图片...", flush=True)
    _CACHED_PAIRS = get_pairs()
    
    # 保存新增的QR解码到磁盘
    save_qr_cache()
    print(f"配对完成: {len(_CACHED_PAIRS)} 对", flush=True)
    
    # 替换 get_pairs 为缓存版本
    get_pairs = lambda: _CACHED_PAIRS
    
    server = http.server.HTTPServer(('0.0.0.0', PORT), LabelHandler)
    print(f"标注工具已启动: http://localhost:{PORT}", flush=True)
    print(f"标注保存位置: {LABELS_FILE}", flush=True)
    print("按 Ctrl+C 停止", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
        server.server_close()
