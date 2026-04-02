"""
蓝色标签审计工具 — port 8768
显示每张图片的贴纸检测中间状态:
  原图 | HSV蓝色掩码 | 检测结果(绿圈=成功/橙圈=失败轮廓)
"""
import os, sys, json, io, base64, urllib.parse, threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import cv2, numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from pipeline.sticker import _refine_ellipse_by_gradient

DATA_DIR   = os.path.join(os.path.dirname(__file__), 'data')
CACHE_FILE = os.path.join(os.path.dirname(__file__), 'output_v2', 'sticker_audit_cache.json')
PORT       = 8768
PAGE_SIZE  = 12

# ─── 检测 + 中间状态 ──────────────────────────────────────────────────────────

def _audit_detect(img_bgr):
    """直接调用 detect_blue_sticker，保证与流水线逻辑 100% 一致。
    额外返回 HSV mask 和所有候选轮廓供可视化使用。"""
    from pipeline.sticker import detect_blue_sticker, _hsv_candidates

    # 获取 HSV 候选用于可视化（Pass-1 优先，Pass-2 备用）
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    scored1, mask1 = _hsv_candidates(img_bgr, hsv, 88, 50, 50)
    if scored1:
        vis_mask = mask1
        all_contours = [c for _, c in scored1]
    else:
        scored2, mask2 = _hsv_candidates(img_bgr, hsv, 86, 15, 200)
        vis_mask = mask2
        all_contours = [c for _, c in scored2]

    # 调用真正的检测逻辑
    from pipeline.sticker import _last_pass as _lp_before
    sticker_mask, center, ellipse = detect_blue_sticker(img_bgr)
    import pipeline.sticker as _st_mod
    pass_used = _st_mod._last_pass  # 读取本次使用的 Pass

    if center is not None:
        used_hough = (pass_used == 'Hough')
        status = f"FOUND({pass_used})"
        reason = ""
    else:
        used_hough = False
        status = "FAILED"
        reason = "所有Pass均失败"

    return dict(
        status=status, reason=reason, center=center, ellipse=ellipse,
        mask=vis_mask,
        candidate_contours=all_contours,
        rejected_contours=[],
        img_area=img_bgr.shape[0] * img_bgr.shape[1],
        used_hough=used_hough,
    )


def _to_jpeg_b64(img_bgr, max_w=320, quality=82):
    h, w = img_bgr.shape[:2]
    if w > max_w:
        scale = max_w / w
        img_bgr = cv2.resize(img_bgr, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode() if ok else ''


def _build_thumbnail(img_path, thumb_type):
    """Generate thumbnail image bytes for orig / mask / result."""
    img = cv2.imread(img_path)
    if img is None:
        blank = np.zeros((200, 320, 3), dtype=np.uint8)
        cv2.putText(blank, 'load error', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        return _encode_jpeg(blank)

    # 检测分辨率与流水线保持一致（DRY最大1536，WET最大1024）
    h, w = img.shape[:2]
    is_wet = ('wet' in img_path.lower() or 'image_' in os.path.basename(img_path).lower())
    detect_max = 1024 if is_wet else 1536
    det_scale = min(detect_max / w, detect_max / h, 1.0)
    if det_scale < 1.0:
        det_img = cv2.resize(img, (int(w * det_scale), int(h * det_scale)), interpolation=cv2.INTER_AREA)
    else:
        det_img = img

    info = _audit_detect(det_img)

    # 显示用缩略图：宽度最大 640
    disp_max = 640
    disp_scale = min(disp_max / w, 1.0)
    if disp_scale < 1.0:
        disp_img = cv2.resize(img, (int(w * disp_scale), int(h * disp_scale)), interpolation=cv2.INTER_AREA)
    else:
        disp_img = img.copy()

    # 将检测坐标从 det_img 尺寸映射到 disp_img 尺寸
    coord_scale = disp_scale / det_scale  # det→disp 比例

    def scale_ellipse(ell):
        if ell is None:
            return None
        (cx, cy), (ma, mi), angle = ell
        return ((cx * coord_scale, cy * coord_scale),
                (ma * coord_scale, mi * coord_scale), angle)

    def scale_contours(cnts):
        return [np.round(c * coord_scale).astype(np.int32) for c in cnts]

    if thumb_type == 'original':
        return _encode_jpeg(disp_img)

    if thumb_type == 'mask':
        det_h, det_w = det_img.shape[:2]
        disp_h, disp_w = disp_img.shape[:2]
        mask_resized = cv2.resize(info['mask'], (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
        vis = disp_img.copy()
        vis[mask_resized > 0] = (255, 80, 0)
        blended = cv2.addWeighted(disp_img, 0.5, vis, 0.5, 0)
        return _encode_jpeg(blended)

    if thumb_type == 'result':
        vis = disp_img.copy()
        if info['rejected_contours']:
            cv2.drawContours(vis, scale_contours(info['rejected_contours']), -1, (0, 165, 255), 2)
        if info['candidate_contours']:
            cv2.drawContours(vis, scale_contours(info['candidate_contours']), -1, (0, 255, 255), 2)
        scaled_ell = scale_ellipse(info['ellipse'])
        if scaled_ell is not None:
            ell_color = (0, 220, 255) if info.get('used_hough') else (0, 255, 0)
            cv2.ellipse(vis, scaled_ell, ell_color, 3)
            sc = (int(info['center'][0] * coord_scale), int(info['center'][1] * coord_scale))
            cv2.circle(vis, sc, 6, (0, 0, 255), -1)
        color = (0, 220, 255) if info.get('used_hough') else (0, 200, 0) \
                if info['status'].startswith('FOUND') else (0, 0, 220)
        label = info['status'] if info['status'].startswith('FOUND') else f"FAILED: {info['reason']}"
        cv2.putText(vis, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 4)
        cv2.putText(vis, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return _encode_jpeg(vis)

    return b''


def _encode_jpeg(img, quality=85):
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b''


# ─── Cache ───────────────────────────────────────────────────────────────────

_cache_lock = threading.Lock()
_cache: dict = {}   # path → {status, reason}

def _load_cache():
    global _cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, encoding='utf-8') as f:
                _cache = json.load(f)
        except Exception:
            _cache = {}

def _save_cache():
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(_cache, f, ensure_ascii=False, indent=2)

def _scan_images():
    imgs = []
    for root, dirs, files in os.walk(DATA_DIR):
        for fn in files:
            if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                imgs.append(os.path.join(root, fn))
    return sorted(imgs)

def _ensure_cached(path):
    """Return cache entry for path, computing if missing."""
    with _cache_lock:
        if path in _cache:
            return _cache[path]
    img = cv2.imread(path)
    if img is None:
        entry = {'status': 'ERROR', 'reason': '图片读取失败'}
    else:
        h, w = img.shape[:2]
        max_w = 640
        if w > max_w:
            scale = max_w / w
            img = cv2.resize(img, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
        info = _audit_detect(img)
        entry = {'status': info['status'], 'reason': info['reason']}
    with _cache_lock:
        _cache[path] = entry
    return entry

def _background_scan():
    imgs = _scan_images()
    for path in imgs:
        _ensure_cached(path)
    _save_cache()
    print(f'[sticker_audit] 扫描完成: {len(imgs)} 张图片已缓存')

# ─── HTML ─────────────────────────────────────────────────────────────────────

HTML_TMPL = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>蓝色标签检测审计</title>
<style>
body{{font-family:sans-serif;background:#1a1a2e;color:#eee;margin:0;padding:12px}}
h1{{color:#e0c97f;margin:0 0 10px}}
.controls{{display:flex;gap:10px;margin-bottom:14px;align-items:center;flex-wrap:wrap}}
.btn{{padding:6px 14px;border:none;border-radius:5px;cursor:pointer;font-size:14px;
      background:#2d5a8e;color:#fff;text-decoration:none}}
.btn.active{{background:#e0c97f;color:#1a1a2e}}
.stat{{background:#26264a;border-radius:6px;padding:4px 12px;font-size:14px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(600px,1fr));gap:18px}}
.card{{background:#26264a;border-radius:10px;padding:12px;border:2px solid transparent}}
.card.found{{border-color:#2a7a2a}}
.card.failed{{border-color:#8a2a2a}}
.card.error{{border-color:#555}}
.card-title{{font-size:12px;color:#aaa;margin-bottom:8px;word-break:break-all}}
.card-title b{{color:#e0c97f}}
.thumbs{{display:flex;gap:8px;flex-wrap:wrap}}
.thumb-wrap{{display:flex;flex-direction:column;align-items:center;gap:4px}}
.thumb-label{{font-size:11px;color:#888}}
.thumb-wrap img{{width:180px;height:135px;object-fit:cover;border-radius:6px;cursor:zoom-in;
                  border:1px solid #444}}
.thumb-wrap img:hover{{border-color:#e0c97f}}
.status-badge{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:12px;
               font-weight:bold;margin-top:4px}}
.found .status-badge{{background:#2a7a2a;color:#afffaf}}
.failed .status-badge{{background:#8a2a2a;color:#ffafaf}}
.reason{{font-size:12px;color:#f88;margin-top:2px}}
.pager{{display:flex;gap:6px;justify-content:center;margin-top:20px;flex-wrap:wrap}}
.pager a{{padding:5px 12px;background:#26264a;border-radius:5px;color:#ccc;text-decoration:none}}
.pager a.cur{{background:#e0c97f;color:#1a1a2e;font-weight:bold}}
.pending{{font-size:12px;color:#888;margin-left:8px}}
/* Lightbox */
#lb{{display:none;position:fixed;inset:0;background:rgba(0,0,0,.92);z-index:9999;
     align-items:center;justify-content:center;cursor:zoom-out}}
#lb.on{{display:flex}}
#lb img{{max-width:94vw;max-height:94vh;border-radius:8px;transform-origin:center;
          transition:transform .15s}}
#lb-hint{{position:fixed;bottom:18px;left:50%;transform:translateX(-50%);
          color:#888;font-size:13px;pointer-events:none}}
</style>
</head><body>
<h1>🔵 蓝色标签检测审计</h1>
<div class="controls">
  <a class="btn {fa}" href="/?filter=all&page=1">全部 ({n_all})</a>
  <a class="btn {ff}" href="/?filter=found&page=1">✓ 检测成功 ({n_found})</a>
  <a class="btn {fl}" href="/?filter=failed&page=1">✗ 检测失败 ({n_failed})</a>
  <span class="stat">第 {page}/{total_pages} 页</span>
  {pending_msg}
</div>
<div class="grid">
{cards}
</div>
<div class="pager">{pager}</div>
<div id="lb" onclick="hideLb()"><img id="lb-img" src=""><div id="lb-hint">滚轮缩放 · 点击关闭</div></div>
<script>
function showLb(src){{
  var el=document.getElementById('lb-img');
  el.src=src; el.style.transform='scale(1)';
  document.getElementById('lb').classList.add('on');
}}
function hideLb(){{ document.getElementById('lb').classList.remove('on'); }}
document.getElementById('lb').addEventListener('wheel',function(e){{
  e.preventDefault();
  var img=document.getElementById('lb-img');
  var m=parseFloat(img.style.transform.replace('scale(','').replace(')',''))||1;
  m=Math.max(0.5,Math.min(10,m+(e.deltaY<0?0.2:-0.2)));
  img.style.transform='scale('+m+')';
}},{{passive:false}});
document.addEventListener('keydown',function(e){{if(e.key==='Escape')hideLb();}});
</script>
</body></html>
"""

CARD_TMPL = """
<div class="card {cls}">
  <div class="card-title"><b>{batch}</b> — {fname}</div>
  <div class="thumbs">
    <div class="thumb-wrap">
      <img src="/thumb?path={enc}&type=original" loading="lazy"
           onclick="showLb('/img?path={enc}')">
      <span class="thumb-label">原图</span>
    </div>
    <div class="thumb-wrap">
      <img src="/thumb?path={enc}&type=mask" loading="lazy"
           onclick="showLb('/thumb_full?path={enc}&type=mask')">
      <span class="thumb-label">HSV蓝色掩码</span>
    </div>
    <div class="thumb-wrap">
      <img src="/thumb?path={enc}&type=result" loading="lazy"
           onclick="showLb('/thumb_full?path={enc}&type=result')">
      <span class="thumb-label">检测结果</span>
    </div>
  </div>
  <span class="status-badge">{status}</span>
  {reason_html}
</div>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress logs

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        qs     = urllib.parse.parse_qs(parsed.query)

        if parsed.path == '/':
            self._serve_page(qs)
        elif parsed.path == '/thumb':
            self._serve_thumb(qs, max_w=320)
        elif parsed.path == '/thumb_full':
            self._serve_thumb(qs, max_w=1280)
        elif parsed.path == '/img':
            self._serve_img(qs)
        elif parsed.path == '/refresh':
            self._serve_refresh()
        else:
            self.send_error(404)

    def _serve_page(self, qs):
        filt  = qs.get('filter', ['all'])[0]
        page  = int(qs.get('page', ['1'])[0])

        all_images = _scan_images()
        n_all    = len(all_images)
        n_found  = sum(1 for p in all_images if _cache.get(p, {}).get('status', '').startswith('FOUND'))
        n_failed = sum(1 for p in all_images if _cache.get(p, {}).get('status') == 'FAILED')
        n_cached = sum(1 for p in all_images if p in _cache)
        pending_n = n_all - n_cached

        if filt == 'found':
            subset = [p for p in all_images if _cache.get(p, {}).get('status', '').startswith('FOUND')]
        elif filt == 'failed':
            subset = [p for p in all_images if _cache.get(p, {}).get('status') in ('FAILED', 'ERROR', None)]
        else:
            subset = all_images

        total_pages = max(1, (len(subset) + PAGE_SIZE - 1) // PAGE_SIZE)
        page = max(1, min(page, total_pages))
        items = subset[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]

        cards_html = ''
        for path in items:
            entry = _ensure_cached(path)
            enc   = urllib.parse.quote(path, safe='')
            fname = os.path.basename(path)
            parts = path.replace('\\', '/').split('/')
            batch = parts[-2] if len(parts) >= 2 else ''
            cls   = entry['status'].lower()
            reason_html = f'<div class="reason">{entry["reason"]}</div>' if entry.get('reason') else ''
            cards_html += CARD_TMPL.format(
                cls=cls, batch=batch, fname=fname,
                enc=enc, status=entry['status'], reason_html=reason_html,
            )

        # pager
        pager = ''
        if total_pages > 1:
            for p in range(1, total_pages + 1):
                cls_p = 'cur' if p == page else ''
                pager += f'<a class="{cls_p}" href="/?filter={filt}&page={p}">{p}</a>'

        pending_msg = ''
        if pending_n > 0:
            pending_msg = f'<span class="pending">⏳ 后台扫描中: {n_cached}/{n_all}</span>'

        body = HTML_TMPL.format(
            fa='active' if filt == 'all' else '',
            ff='active' if filt == 'found' else '',
            fl='active' if filt == 'failed' else '',
            n_all=n_all, n_found=n_found, n_failed=n_failed,
            page=page, total_pages=total_pages,
            pending_msg=pending_msg,
            cards=cards_html, pager=pager,
        ).encode('utf-8')

        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def _serve_thumb(self, qs, max_w=320):
        path  = urllib.parse.unquote(qs.get('path', [''])[0])
        ttype = qs.get('type', ['original'])[0]
        if not path or not os.path.exists(path):
            self.send_error(404); return

        img = cv2.imread(path)
        if img is None:
            self.send_error(404); return

        h, w = img.shape[:2]
        work_w = 640
        if w > work_w:
            scale = work_w / w
            img = cv2.resize(img, (work_w, int(h * scale)), interpolation=cv2.INTER_AREA)

        info = _audit_detect(img)

        if ttype == 'original':
            vis = img
        elif ttype == 'mask':
            vis = img.copy()
            vis[info['mask'] > 0] = (255, 80, 0)
            vis = cv2.addWeighted(img, 0.5, vis, 0.5, 0)
        elif ttype == 'result':
            vis = img.copy()
            if info['rejected_contours']:
                cv2.drawContours(vis, info['rejected_contours'], -1, (0, 165, 255), 2)
            if info['candidate_contours']:
                cv2.drawContours(vis, info['candidate_contours'], -1, (0, 255, 255), 2)
            if info['ellipse'] is not None:
                ell_color = (0, 220, 255) if info.get('used_hough') else (0, 255, 0)
                cv2.ellipse(vis, info['ellipse'], ell_color, 3)
                cv2.circle(vis, info['center'], 6, (0, 0, 255), -1)
            if info['status'].startswith('FOUND'):
                color = (0, 220, 255) if info.get('used_hough') else (0, 200, 0)
            else:
                color = (0, 0, 220)
            label = info['status'] if info['status'].startswith('FOUND') else f"FAILED: {info['reason']}"
            cv2.putText(vis, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 4)
            cv2.putText(vis, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            vis = img

        # resize to max_w
        h2, w2 = vis.shape[:2]
        if w2 > max_w:
            scale = max_w / w2
            vis = cv2.resize(vis, (max_w, int(h2 * scale)), interpolation=cv2.INTER_AREA)

        ok, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        data = buf.tobytes() if ok else b''

        self.send_response(200)
        self.send_header('Content-Type', 'image/jpeg')
        self.send_header('Content-Length', len(data))
        self.send_header('Cache-Control', 'max-age=300')
        self.end_headers()
        self.wfile.write(data)

    def _serve_img(self, qs):
        path = urllib.parse.unquote(qs.get('path', [''])[0])
        if not path or not os.path.exists(path):
            self.send_error(404); return
        with open(path, 'rb') as f:
            data = f.read()
        ext  = os.path.splitext(path)[1].lower()
        mime = 'image/jpeg' if ext in ('.jpg', '.jpeg') else 'image/png'
        self.send_response(200)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', len(data))
        self.end_headers()
        self.wfile.write(data)

    def _serve_refresh(self):
        global _cache
        with _cache_lock:
            _cache = {}
        threading.Thread(target=_background_scan, daemon=True).start()
        body = b'OK - refreshing...'
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)


if __name__ == '__main__':
    _load_cache()
    all_imgs = _scan_images()
    n_cached = sum(1 for p in all_imgs if p in _cache)
    print(f'[sticker_audit] 已缓存 {n_cached}/{len(all_imgs)} 张图片')
    if n_cached < len(all_imgs):
        print('[sticker_audit] 后台扫描未缓存图片...')
        threading.Thread(target=_background_scan, daemon=True).start()
    print(f'[sticker_audit] 启动 http://localhost:{PORT}')
    srv = HTTPServer(('0.0.0.0', PORT), Handler)
    srv.serve_forever()
