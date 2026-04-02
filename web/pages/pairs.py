"""
手动 QR 配对页面 — 委托到 manual_pair_tool.py 的逻辑
"""
import os, sys, json, urllib.parse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

_PAIRS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'output_v2', 'manual_pairs.json'))


def serve_image(fpath: str):
    """Return (bytes, mime) or (None, None)."""
    if fpath and os.path.isfile(fpath):
        ext = os.path.splitext(fpath)[1].lower().lstrip('.')
        mime = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png'}.get(ext, 'image/jpeg')
        with open(fpath, 'rb') as f:
            return f.read(), mime
    return None, None


def build_pairs_page(page: int = 1) -> str:
    """Build the pairs page HTML and return it."""
    import manual_pair_tool as mpt
    html = mpt._render_page(page=page)
    html = _rewrite_urls(html)
    return html


def save_pair(body: dict) -> dict:
    """Save a manual pair entry. Returns {'ok': True}."""
    if os.path.exists(_PAIRS_FILE):
        with open(_PAIRS_FILE, encoding='utf-8') as f:
            pairs = json.load(f)
    else:
        pairs = {}

    fpath = body.get('path', '')
    block = body.get('block', '')
    if fpath:
        pairs[fpath] = block
        with open(_PAIRS_FILE, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)

    return {'ok': True}


def _rewrite_urls(html: str) -> str:
    """Rewrite manual_pair_tool URLs to /pairs/* namespace."""
    html = html.replace('href="/?', 'href="/pairs?')
    html = html.replace("href='/?", "href='/pairs?")
    html = html.replace('/img?path=', '/pairs/img?path=')
    html = html.replace("action='/save'", "action='/pairs/save'")
    html = html.replace('action="/save"', 'action="/pairs/save"')
    html = html.replace("fetch('/save'", "fetch('/pairs/save'")
    html = html.replace('<body>', '<body><div style="padding:8px 16px;background:#111">'
                         '<a href="/" style="color:#667eea;text-decoration:none">← 首页</a></div>')
    return html


# ── Legacy http.server compat ────────────────────────────────────

def handle_pairs_get(handler, path, qs):
    from web.server import _send_html, _send_image

    if path == '/pairs/img':
        fpath = qs.get('path', [''])[0]
        img_bytes, mime = serve_image(fpath)
        if img_bytes:
            _send_image(handler, img_bytes, mime)
        else:
            handler.send_error(404)
        return

    page = int(qs.get('page', ['1'])[0])
    html = build_pairs_page(page)
    _send_html(handler, html)


def handle_pairs_post(handler, path):
    from web.server import _send_json

    if path == '/pairs/save':
        length = int(handler.headers.get('Content-Length', 0))
        body = json.loads(handler.rfile.read(length)) if length else {}
        result = save_pair(body)
        _send_json(handler, result)
    else:
        handler.send_error(404)
