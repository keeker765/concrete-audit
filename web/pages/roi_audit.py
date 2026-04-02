"""
ROI 审计页面
============
/roi_audit   — 主页面（分页列表 + 当前图审核）
/roi_audit/save  POST — 保存标注
/roi_audit/image GET  — 返回图片（带或不带边框）
"""
import base64, json, os, sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).parent.parent.parent
DETECTIONS_FILE = ROOT / 'output_v2' / 'roi_detections.json'
LABELS_FILE = ROOT / 'output_v2' / 'roi_labels.json'

PAGE_SIZE = 20

_detections_cache = None
_labels_cache = None


def _load_detections():
    global _detections_cache
    if not _detections_cache:   # 空时重新读取（支持扫描中刷新）
        if DETECTIONS_FILE.exists():
            _detections_cache = json.loads(DETECTIONS_FILE.read_text(encoding='utf-8'))
        else:
            _detections_cache = {}
    return _detections_cache


def _load_labels():
    global _labels_cache
    if _labels_cache is None:
        if LABELS_FILE.exists():
            _labels_cache = json.loads(LABELS_FILE.read_text(encoding='utf-8'))
        else:
            _labels_cache = {}
    return _labels_cache


def _save_labels():
    LABELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    LABELS_FILE.write_text(
        json.dumps(_labels_cache, ensure_ascii=False, indent=2), encoding='utf-8')


def _img_key_list():
    """返回有 bbox_orig 的图片 key 列表"""
    det = _load_detections()
    return [k for k, v in det.items()
            if isinstance(v, dict) and v.get('bbox_orig') is not None]


def _make_img_b64(rel_path, bbox=None, custom_bbox=None, max_dim=600):
    """读取图片，在上面画出 bbox，返回 base64 JPEG 和 display_size"""
    img_path = ROOT / rel_path.replace('/', os.sep)
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        return None, None

    h0, w0 = bgr.shape[:2]
    if max(h0, w0) > max_dim:
        s = max_dim / max(h0, w0)
        bgr = cv2.resize(bgr, (int(w0 * s), int(h0 * s)))
    dh, dw = bgr.shape[:2]
    scale = dw / w0

    det = _load_detections().get(rel_path, {})
    orig_size = det.get('orig_size', [w0, h0])
    orig_w = orig_size[0]
    sc = dw / orig_w  # orig → display

    # 画检测框（橙色）
    if bbox:
        x1, y1, x2, y2 = [int(c * sc) for c in bbox]
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 140, 255), 2)

    # 画自定义框（绿色）
    if custom_bbox:
        x1, y1, x2, y2 = [int(c * sc) for c in custom_bbox]
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 220, 50), 3)

    _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf.tobytes()).decode()
    return b64, (dw, dh)


def build_roi_audit_page(page: int = 1, search: str = '', show_filter: str = 'all',
                        idx: int = None) -> str:
    """Build ROI audit page HTML. Returns HTML string."""
    keys = _img_key_list()
    if not keys:
        return _page_not_ready()

    labels = _load_labels()

    # 过滤
    filtered = keys
    if search:
        filtered = [k for k in filtered if search.lower() in k.lower()]
    if show_filter == 'ok':
        filtered = [k for k in filtered if labels.get(k, {}).get('label') == 'ok']
    elif show_filter == 'wrong':
        filtered = [k for k in filtered if labels.get(k, {}).get('label') == 'wrong']
    elif show_filter == 'quality':
        filtered = [k for k in filtered if labels.get(k, {}).get('label') == 'quality']
    elif show_filter == 'unlabeled':
        filtered = [k for k in filtered if k not in labels]

    total = len(filtered)
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page = max(1, min(page, total_pages))
    page_keys = filtered[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]

    # 当前选中
    if idx is not None:
        cur_key = filtered[idx] if idx < len(filtered) else None
    elif page_keys:
        cur_key = page_keys[0]
    else:
        cur_key = None

    cur_idx = filtered.index(cur_key) if cur_key in filtered else 0

    # 生成当前图的 base64
    img_b64, img_size = None, (600, 600)
    cur_det = {}
    cur_label = {}
    if cur_key:
        cur_det = _load_detections().get(cur_key, {})
        cur_label = labels.get(cur_key, {})
        bbox = cur_det.get('bbox_orig')
        custom_bbox = cur_label.get('custom_bbox')
        img_b64, img_size = _make_img_b64(cur_key, bbox=bbox,
                                           custom_bbox=custom_bbox, max_dim=600)

    # 统计
    n_ok = sum(1 for v in labels.values() if v.get('label') == 'ok')
    n_wrong = sum(1 for v in labels.values() if v.get('label') == 'wrong')
    n_quality = sum(1 for v in labels.values() if v.get('label') == 'quality')
    n_unlabeled = len(keys) - len(labels)

    return _render_page(
        keys=page_keys, all_keys=filtered,
        cur_key=cur_key, cur_idx=cur_idx,
        img_b64=img_b64, img_size=img_size,
        cur_det=cur_det, cur_label=cur_label,
        page=page, total_pages=total_pages, total=total,
        search=search, show_filter=show_filter,
        stats={'ok': n_ok, 'wrong': n_wrong, 'quality': n_quality, 'unlabeled': n_unlabeled, 'total': len(keys)},
    )


def save_roi_label(body: dict) -> tuple:
    """Save ROI label. Returns (result_dict, status_code)."""
    key = body.get('key')
    label = body.get('label')
    custom_bbox = body.get('custom_bbox')

    if not key or label not in ('ok', 'wrong', 'quality'):
        return {'ok': False, 'error': 'invalid'}, 400

    global _labels_cache
    lbs = _load_labels()
    lbs[key] = {'label': label}
    if custom_bbox:
        lbs[key]['custom_bbox'] = custom_bbox
    _save_labels()
    return {'ok': True, 'saved': key}, 200


# ── Legacy http.server compat ────────────────────────────────────

def handle_roi_audit_get(handler, qs):
    from web.server import _send_html
    page = int(qs.get('page', ['1'])[0])
    search = qs.get('q', [''])[0].strip()
    show_filter = qs.get('filter', ['all'])[0]
    cur_idx_str = qs.get('idx', [None])[0]
    idx = int(cur_idx_str) if cur_idx_str is not None else None
    html = build_roi_audit_page(page, search, show_filter, idx)
    _send_html(handler, html)


def handle_roi_audit_post(handler, path):
    from web.server import _send_json

    length = int(handler.headers.get('Content-Length', 0))
    body_raw = handler.rfile.read(length).decode('utf-8')

    if path == '/roi_audit/save':
        data = json.loads(body_raw)
        result, code = save_roi_label(data)
        _send_json(handler, result, code)
    else:
        handler.send_error(404)


def _page_not_ready():
    return '''<!DOCTYPE html><html><body style="font-family:sans-serif;padding:40px">
<h2>⏳ 检测数据未就绪</h2>
<p>请先运行：<code>python roi_batch_scan.py</code></p>
<p>或加 <code>--limit 100</code> 快速生成部分结果后刷新此页。</p>
</body></html>'''


def _render_page(keys, all_keys, cur_key, cur_idx, img_b64, img_size,
                 cur_det, cur_label, page, total_pages, total, search,
                 show_filter, stats):
    img_w, img_h = img_size if img_size else (600, 600)
    cur_bbox_orig = cur_det.get('bbox_orig') or []
    cur_custom = cur_label.get('custom_bbox') or []
    cur_lbl = cur_label.get('label', '')

    def _key_short(k):
        parts = k.split('/')
        return '/'.join(parts[-2:]) if len(parts) >= 2 else k

    def _lbl_badge(k):
        lb = _load_labels().get(k, {}).get('label', '')
        if lb == 'ok':     return '<span style="color:#22c55e">✓</span>'
        if lb == 'wrong':  return '<span style="color:#ef4444">✗</span>'
        if lb == 'quality': return '<span style="color:#f59e0b">⚠</span>'
        return '<span style="color:#94a3b8">○</span>'

    list_items = ''
    for i, k in enumerate(keys):
        page_offset = (page - 1) * PAGE_SIZE
        abs_idx = all_keys.index(k) if k in all_keys else page_offset + i
        active = 'background:#1e3a5f;color:#fff;' if k == cur_key else ''
        list_items += (
            f'<div style="padding:4px 8px;cursor:pointer;border-bottom:1px solid #334155;'
            f'font-size:11px;{active}" '
            f'onclick="selectImg({abs_idx})">'
            f'{_lbl_badge(k)} {_key_short(k)}</div>'
        )

    # 分页按钮
    pager = ''
    for p in range(max(1, page - 2), min(total_pages, page + 2) + 1):
        active_style = 'background:#3b82f6;color:white;' if p == page else ''
        pager += (f'<a href="?page={p}&q={search}&filter={show_filter}" '
                  f'style="margin:0 2px;padding:3px 8px;border:1px solid #475569;'
                  f'border-radius:4px;text-decoration:none;{active_style}">{p}</a>')

    # 滤镜按钮
    filters = [('all', f'全部({stats["total"]})'),
               ('unlabeled', f'未标注({stats["unlabeled"]})'),
               ('ok', f'✓正确({stats["ok"]})'),
               ('wrong', f'✗错误({stats["wrong"]})'),
               ('quality', f'⚠质量({stats["quality"]})')]
    filter_btns = ''
    for fv, flabel in filters:
        active_s = 'background:#3b82f6;color:white;' if fv == show_filter else 'background:#1e293b;color:#cbd5e1;'
        filter_btns += (f'<a href="?page=1&q={search}&filter={fv}" '
                        f'style="padding:3px 10px;border:1px solid #475569;border-radius:4px;'
                        f'text-decoration:none;font-size:12px;{active_s}">{flabel}</a> ')

    img_tag = ''
    if img_b64:
        img_tag = f'<img id="mainImg" src="data:image/jpeg;base64,{img_b64}" style="max-width:100%;border-radius:6px;" />'

    # 判断当前框状态
    bbox_info = ''
    if cur_bbox_orig:
        x1, y1, x2, y2 = cur_bbox_orig
        bbox_info = f'自动框: ({x1},{y1}) → ({x2},{y2})  {x2-x1}×{y2-y1}px'
    if cur_custom:
        cx1, cy1, cx2, cy2 = cur_custom
        bbox_info += f' | 手动框: ({cx1},{cy1}) → ({cx2},{cy2})'

    # 当前标注状态颜色
    lbl_color = {'ok': '#22c55e', 'wrong': '#ef4444', 'quality': '#f59e0b'}.get(cur_lbl, '#64748b')
    lbl_text = {'ok': '✓ 已标注：正确', 'wrong': '✗ 已标注：错误',
                'quality': '⚠ 已标注：质量问题'}.get(cur_lbl, '○ 未标注')

    return f'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>ROI 审计</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0f172a;color:#e2e8f0;font-family:system-ui,sans-serif;display:flex;height:100vh;overflow:hidden}}
#sidebar{{width:260px;min-width:260px;background:#1e293b;display:flex;flex-direction:column;border-right:1px solid #334155}}
#sidebar-head{{padding:10px;border-bottom:1px solid #334155}}
#sidebar-head h3{{font-size:14px;color:#94a3b8;margin-bottom:6px}}
#search{{width:100%;padding:4px 8px;background:#0f172a;border:1px solid #334155;border-radius:4px;color:#e2e8f0;font-size:12px}}
#filter-row{{margin-top:6px;display:flex;flex-wrap:wrap;gap:3px}}
#list{{flex:1;overflow-y:auto}}
#pager{{padding:8px;border-top:1px solid #334155;text-align:center;font-size:12px}}
#main{{flex:1;display:flex;flex-direction:column;overflow:hidden}}
#toolbar{{padding:10px 14px;background:#1e293b;border-bottom:1px solid #334155;display:flex;align-items:center;gap:10px;flex-wrap:wrap}}
#img-area{{flex:1;overflow:auto;padding:16px;display:flex;justify-content:center;align-items:flex-start;position:relative}}
#canvas-wrap{{position:relative;display:inline-block}}
canvas#drawCanvas{{position:absolute;top:0;left:0;cursor:crosshair;display:none}}
#info-bar{{padding:6px 14px;background:#0f172a;font-size:11px;color:#94a3b8;border-top:1px solid #1e293b}}
.btn{{padding:6px 14px;border:none;border-radius:5px;cursor:pointer;font-size:13px;font-weight:500}}
.btn-ok{{background:#16a34a;color:white}}
.btn-wrong{{background:#dc2626;color:white}}
.btn-quality{{background:#d97706;color:white}}
.btn-draw{{background:#7c3aed;color:white}}
.btn-cancel{{background:#475569;color:white}}
.btn:hover{{opacity:0.85}}
#lbl-status{{font-size:12px;padding:3px 10px;border-radius:4px;background:#1e293b;color:{lbl_color}}}
</style>
</head>
<body>

<div id="sidebar">
  <div id="sidebar-head">
    <h3>📋 ROI 审计 ({total} 张)</h3>
    <input id="search" type="text" placeholder="搜索路径..." value="{search}"
           onkeydown="if(event.key=='Enter')doSearch()" />
    <div id="filter-row">{filter_btns}</div>
  </div>
  <div id="list">{list_items}</div>
  <div id="pager">{pager}</div>
</div>

<div id="main">
  <div id="toolbar">
    <span id="lbl-status">{lbl_text}</span>
    <button class="btn btn-ok" onclick="saveLabel('ok')">✓ 正确</button>
    <button class="btn btn-wrong" onclick="saveLabel('wrong')">✗ 错误（自动框）</button>
    <button class="btn btn-quality" onclick="saveLabel('quality')">⚠ 质量问题</button>
    <button class="btn btn-draw" id="btnDraw" onclick="startDraw()">✏ 手动框选（拖拽自动保存）</button>
    <button class="btn btn-cancel" id="btnCancel" onclick="cancelDraw()" style="display:none">取消框选</button>
    <span style="font-size:11px;color:#64748b;margin-left:auto">← → 键切换</span>
  </div>
  <div id="img-area">
    <div id="canvas-wrap">
      {img_tag}
      <canvas id="drawCanvas"></canvas>
    </div>
  </div>
  <div id="info-bar">{bbox_info or '（无贴纸检测结果）'}</div>
</div>

<script>
const CUR_KEY = {json.dumps(cur_key)};
const CUR_IDX = {cur_idx};
const ALL_KEYS = {json.dumps(all_keys)};
const CUR_PAGE = {page};
const SEARCH = {json.dumps(search)};
const FILTER = {json.dumps(show_filter)};
const CUR_ORIG_SIZE = {json.dumps(cur_det.get('orig_size', [1,1]))};
const CUR_BBOX_ORIG = {json.dumps(cur_bbox_orig)};

function selectImg(idx) {{
    window.location = '?page=' + Math.floor(idx/{PAGE_SIZE}+1) + '&idx=' + idx + '&q=' + encodeURIComponent(SEARCH) + '&filter=' + FILTER;
}}

function doSearch() {{
    const q = document.getElementById('search').value;
    window.location = '?page=1&q=' + encodeURIComponent(q) + '&filter=' + FILTER;
}}

document.addEventListener('keydown', e => {{
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {{
        const next = Math.min(CUR_IDX + 1, ALL_KEYS.length - 1);
        selectImg(next);
    }} else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {{
        const prev = Math.max(CUR_IDX - 1, 0);
        selectImg(prev);
    }}
}});

// ---- 标注保存 ----
async function saveLabel(label, customBbox) {{
    if (!CUR_KEY) return;
    const body = {{key: CUR_KEY, label}};
    if (customBbox) body.custom_bbox = customBbox;
    const r = await fetch('/roi_audit/save', {{method:'POST', body: JSON.stringify(body)}});
    if ((await r.json()).ok) {{
        // 切到下一张
        const next = Math.min(CUR_IDX + 1, ALL_KEYS.length - 1);
        selectImg(next);
    }}
}}

// ---- 画框工具 ----
let drawing = false, startX, startY, rect = null;
const img = document.getElementById('mainImg');
const canvas = document.getElementById('drawCanvas');
const ctx = canvas ? canvas.getContext('2d') : null;

function startDraw() {{
    if (!img) return;
    canvas.width = img.clientWidth;
    canvas.height = img.clientHeight;
    canvas.style.display = 'block';
    document.getElementById('btnDraw').style.display = 'none';
    document.getElementById('btnCancel').style.display = '';
}}

function cancelDraw() {{
    canvas.style.display = 'none';
    document.getElementById('btnDraw').style.display = '';
    document.getElementById('btnCancel').style.display = 'none';
    rect = null;
}}

canvas && canvas.addEventListener('mousedown', e => {{
    drawing = true;
    const r = canvas.getBoundingClientRect();
    startX = e.clientX - r.left;
    startY = e.clientY - r.top;
    rect = null;
}});

canvas && canvas.addEventListener('mousemove', e => {{
    if (!drawing) return;
    const r = canvas.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#22c55e';
    ctx.lineWidth = 2;
    ctx.strokeRect(startX, startY, x - startX, y - startY);
}});

canvas && canvas.addEventListener('mouseup', e => {{
    drawing = false;
    const r = canvas.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    rect = [Math.min(startX,x), Math.min(startY,y), Math.max(startX,x), Math.max(startY,y)];
    // 画完立即自动保存
    if (rect[2] - rect[0] > 5 && rect[3] - rect[1] > 5) {{
        const scaleX = CUR_ORIG_SIZE[0] / canvas.width;
        const scaleY = CUR_ORIG_SIZE[1] / canvas.height;
        const origBox = [
            Math.round(rect[0] * scaleX), Math.round(rect[1] * scaleY),
            Math.round(rect[2] * scaleX), Math.round(rect[3] * scaleY)
        ];
        saveLabel('wrong', origBox);
    }}
}});
</script>
</body>
</html>'''
