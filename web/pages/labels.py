"""
结果审核页面 — 委托到 label_tool.py 的逻辑
"""
import os, sys, json, urllib.parse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

_LABELS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'output_v2', 'labels.json'))


def _get_handler_module():
    import label_tool
    return label_tool


def serve_image(fpath: str):
    """Return (bytes, mime) or (None, None)."""
    if fpath and os.path.isfile(fpath):
        ext = os.path.splitext(fpath)[1].lower().lstrip('.')
        mime = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png'}.get(ext, 'image/jpeg')
        with open(fpath, 'rb') as f:
            return f.read(), mime
    return None, None


def get_labels_data() -> dict:
    """Return contents of labels.json."""
    if os.path.exists(_LABELS_FILE):
        with open(_LABELS_FILE, encoding='utf-8') as f:
            return json.load(f)
    return {}


def build_labels_page(verdict: str = "all", labeled: str = "all",
                      page: int = 1, invalid_reason: str = "all"):
    """Build labels page HTML. Returns (html, ready) where ready is False if cache not warm."""
    mod = _get_handler_module()
    if mod._PAIRS_CACHE is None:
        return _loading_html(), False

    html = mod.build_html(
        page=page,
        filter_verdict=verdict,
        filter_labeled=labeled,
        filter_invalid_reason=invalid_reason,
    )
    html = _rewrite_urls(html)
    return html, True


def save_label(body: dict) -> dict:
    """Save a label entry. Returns {'ok': True}."""
    if os.path.exists(_LABELS_FILE):
        with open(_LABELS_FILE, encoding='utf-8') as f:
            labels = json.load(f)
    else:
        labels = {}

    key = body.get('key', '')
    if key:
        labels[key] = {
            'quality': body.get('quality', ''),
            'notes': body.get('notes', ''),
            'wet_has_text': body.get('wet_has_text', False),
            'dry_has_text': body.get('dry_has_text', False),
        }
        with open(_LABELS_FILE, 'w', encoding='utf-8') as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)

    return {'ok': True}


def _rewrite_urls(html: str) -> str:
    """Rewrite label_tool URLs to /labels/* namespace."""
    html = html.replace('href="/?', 'href="/labels?')
    html = html.replace("href='/?", "href='/labels?")
    html = html.replace('/img?path=', '/labels/img?path=')
    html = html.replace("action='/save'", "action='/labels/save'")
    html = html.replace('action="/save"', 'action="/labels/save"')
    html = html.replace("fetch('/save'", "fetch('/labels/save'")
    html = html.replace("fetch('/labels'", "fetch('/labels/data'")
    html = html.replace('<body>', '<body><div style="padding:8px 16px;background:#111">'
                         '<a href="/" style="color:#667eea;text-decoration:none">← 首页</a></div>')
    return html


def _loading_html() -> str:
    return '''<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="4;url=/labels">
<style>body{background:#1a1a2e;color:#eee;display:flex;align-items:center;
justify-content:center;height:100vh;font-family:sans-serif;flex-direction:column;gap:16px}
.spinner{width:48px;height:48px;border:4px solid #333;border-top:4px solid #667eea;
border-radius:50%;animation:spin 0.8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}</style></head>
<body><div class="spinner"></div>
<p>⏳ 正在加载配对数据（90 个批次 QR 解码中）...</p>
<p style="color:#888;font-size:0.9em">4 秒后自动刷新</p>
<a href="/" style="color:#667eea">← 返回首页</a>
</body></html>'''


# ── Legacy http.server compat ────────────────────────────────────

def handle_labels_get(handler, path, qs):
    from web.server import _send_html, _send_json, _send_image

    if path == '/labels/img':
        fpath = qs.get('path', [''])[0]
        img_bytes, mime = serve_image(fpath)
        if img_bytes:
            _send_image(handler, img_bytes, mime)
        else:
            handler.send_error(404)
        return

    if path == '/labels/data':
        _send_json(handler, get_labels_data())
        return

    verdict = qs.get('verdict', ['all'])[0]
    labeled = qs.get('labeled', ['all'])[0]
    page = int(qs.get('page', ['1'])[0])
    invalid_reason = qs.get('invalid_reason', ['all'])[0]

    html, _ready = build_labels_page(verdict, labeled, page, invalid_reason)
    _send_html(handler, html)


def handle_labels_post(handler, path):
    from web.server import _send_json

    if path == '/labels/save':
        length = int(handler.headers.get('Content-Length', 0))
        body = json.loads(handler.rfile.read(length)) if length else {}
        result = save_label(body)
        _send_json(handler, result)
    else:
        handler.send_error(404)
