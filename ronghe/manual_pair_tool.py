"""
手动配对工具 — QR 解码失败图片的人工试块编号录入
==================================================
启动: python manual_pair_tool.py
访问: http://localhost:8766

功能:
  - 显示 data/ 目录中 QR 解码失败的所有图片
  - 展示贴纸裁剪图 + 顶部文字条带
  - 用户在输入框填入试块编号 (1/2/3)
  - 保存到 output_v2/manual_pairs.json
  - runner.py 在 QR 失败时自动读取该文件
"""
import http.server
import json
import os
import sys
import base64
import urllib.parse
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np

PORT = 8766
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output_v2"
QR_CACHE_FILE = OUTPUT_DIR / "qr_cache.json"
MANUAL_PAIRS_FILE = OUTPUT_DIR / "manual_pairs.json"

sys.path.insert(0, str(PROJECT_ROOT))


# ── 缓存加载 ──────────────────────────────────────────────
def _load_json(path):
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def _save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── 贴纸裁剪 ──────────────────────────────────────────────
def _sticker_crops(img_path: Path):
    """返回 (fullcrop_b64, topstrip_b64) JPEG base64，失败返回 (None, None)"""
    try:
        from pipeline.sticker import detect_blue_sticker
        img = cv2.imread(str(img_path))
        if img is None:
            return None, None
        _, center, ellipse = detect_blue_sticker(img)
        if center is None:
            return None, None
        cx, cy = center
        r = int(max(ellipse[1]) / 2)

        # 全贴纸裁剪
        pad = int(r * 1.05)
        h, w = img.shape[:2]
        x1, y1 = max(0, cx - pad), max(0, cy - pad)
        x2, y2 = min(w, cx + pad), min(h, cy + pad)
        full = img[y1:y2, x1:x2]

        # 顶部文字条带
        ty1 = max(0, cy - int(r * 0.95))
        ty2 = cy - int(r * 0.35)
        tx1 = max(0, cx - int(r * 0.9))
        tx2 = min(w, cx + int(r * 0.9))
        top = img[ty1:ty2, tx1:tx2]

        def _to_b64(arr, max_w=600):
            if arr.size == 0:
                return None
            rh, rw = arr.shape[:2]
            if rw > max_w:
                arr = cv2.resize(arr, (max_w, int(rh * max_w / rw)),
                                 interpolation=cv2.INTER_AREA)
            _, buf = cv2.imencode('.jpg', arr, [cv2.IMWRITE_JPEG_QUALITY, 88])
            return base64.b64encode(buf.tobytes()).decode()

        return _to_b64(full, 600), _to_b64(top, 1200)
    except Exception:
        return None, None


# ── 构建待配对列表 ─────────────────────────────────────────
def _build_pending():
    """返回 QR 解码失败的 (filepath, batch_id, filename) 列表"""
    qr_cache = _load_json(QR_CACHE_FILE)
    manual = _load_json(MANUAL_PAIRS_FILE)
    pending = []
    done = []

    if not DATA_DIR.exists():
        return pending, done

    for company_dir in sorted(DATA_DIR.iterdir()):
        if not company_dir.is_dir():
            continue
        for batch_dir in sorted(company_dir.iterdir()):
            if not batch_dir.is_dir():
                continue
            import re
            m = re.search(r'压-(.+)$', batch_dir.name)
            batch_num = m.group(1) if m else batch_dir.name
            batch_id = f"{company_dir.name[:4]}_{batch_num}"

            # 湿态 (.jpeg) 和 干态 (.jpg) 都需要配对
            all_imgs = sorted(batch_dir.glob("*.jpeg")) + sorted(batch_dir.glob("*.jpg"))
            for img_f in all_imgs:
                key = str(img_f)
                qr_val = qr_cache.get(key)
                manual_val = manual.get(key)
                if qr_val is None:
                    kind = "WET" if img_f.suffix == ".jpeg" else "DRY"
                    entry = (img_f, f"{batch_id}[{kind}]", manual_val)
                    if manual_val is None:
                        pending.append(entry)
                    else:
                        done.append(entry)

    return pending, done


PAGE_SIZE = 15

# ── HTML 渲染 ─────────────────────────────────────────────
def _render_page(page=1):
    pending, done = _build_pending()
    # 所有项目合并展示：未配对在前，已配对在后（全部可编辑）
    all_items = pending + done
    total_pending = len(pending)
    total_pages = max(1, (len(all_items) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = max(1, min(page, total_pages))
    page_items = all_items[(page-1)*PAGE_SIZE : page*PAGE_SIZE]

    rows_html = ""
    for img_path, batch_id, saved_val in page_items:
        key = str(img_path)
        # 延迟加载贴纸裁剪（避免首次渲染调用 SAM）
        sticker_html = '<span style="color:#aaa;font-size:12px">点击原图查看</span>'

        img_url = f'/img?path={urllib.parse.quote(str(img_path))}'
        orig_html = (f'<img src="{img_url}" class="preview-img orig-thumb" '
                     f'style="height:140px;cursor:zoom-in" '
                     f'data-fullsrc="{img_url}" title="点击放大：原图（全分辨率）">')

        # 预填已保存的值
        def opt(v, label, sel):
            s = ' selected' if sel == v else ''
            return f'<option value="{v}"{s}>{label}</option>'

        saved_str = saved_val or ''
        row_bg = 'background:#f0fff0' if saved_val else ''
        safe_key = urllib.parse.quote(key, safe='')
        rows_html += f"""
        <tr style="{row_bg}">
          <td style="padding:6px;white-space:nowrap;font-size:12px">{batch_id}<br><small>{img_path.name}</small>
            {'<br><span style="color:#4CAF50;font-size:11px">✓ 已保存: '+saved_val+'</span>' if saved_val else ''}</td>
          <td style="padding:6px">{orig_html}</td>
          <td style="padding:6px">{sticker_html}</td>
          <td style="padding:6px;text-align:center">
            <select data-key="{safe_key}" class="sel-num" style="font-size:18px;padding:4px">
              {opt('','-- 未知 --',saved_str)}
              {opt('1','1',saved_str)}
              {opt('2','2',saved_str)}
              {opt('3','3',saved_str)}
            </select>
            <span class="save-ok" style="display:none;color:#4CAF50;font-size:13px">✓</span>
          </td>
        </tr>"""

    total = len(all_items)

    # 分页导航
    def page_link(p, label):
        cls = 'page-active' if p == page else ''
        return f'<a href="/?page={p}" class="pg {cls}">{label}</a>'
    pg_links = page_link(1,'«')
    pg_links += page_link(max(1,page-1),'‹')
    for p in range(max(1,page-2), min(total_pages,page+2)+1):
        pg_links += page_link(p, str(p))
    pg_links += page_link(min(total_pages,page+1),'›')
    pg_links += page_link(total_pages,'»')
    page_nav = f'<div class="pgnav">{pg_links}</div>'
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>手动配对工具</title>
<style>
body{{font-family:sans-serif;margin:20px;background:#f5f5f5}}
h2{{color:#333}}
table{{border-collapse:collapse;background:#fff;width:100%;box-shadow:0 1px 4px #ccc}}
tr:nth-child(even){{background:#fafafa}}
th{{background:#4a90d9;color:#fff;padding:8px;text-align:left}}
.stat{{font-size:13px;color:#666;margin-bottom:10px}}
.btn{{background:#4a90d9;color:#fff;border:none;padding:10px 28px;font-size:15px;
      border-radius:4px;cursor:pointer;margin:12px 0}}
.btn:hover{{background:#357abd}}
.pgnav{{text-align:center;margin:10px 0}}
.pg{{display:inline-block;padding:5px 11px;margin:2px;border-radius:6px;
     background:#e8f0fe;color:#1a73e8;text-decoration:none;font-size:14px}}
.pg:hover{{background:#c5d8fb}}
.page-active{{background:#1a73e8;color:#fff;font-weight:bold}}
/* Lightbox */
#lb{{display:none;position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:9999;
     align-items:center;justify-content:center;cursor:zoom-out}}
#lb.show{{display:flex}}
#lb img{{max-width:92vw;max-height:92vh;object-fit:contain;border-radius:6px;
          transform-origin:center;transition:transform .15s}}
#lb-close{{position:fixed;top:16px;right:24px;color:#fff;font-size:36px;
            cursor:pointer;user-select:none;line-height:1}}
#lb-hint{{position:fixed;bottom:16px;left:50%;transform:translateX(-50%);
           color:#aaa;font-size:13px}}
</style>
</head>
<body>
<!-- Lightbox 遮罩 -->
<div id="lb" onclick="closeLb(event)">
  <span id="lb-close" onclick="hideLb()">&#x2715;</span>
  <img id="lb-img" src="" draggable="false">
  <div id="lb-hint">滚轮缩放 · 点击空白关闭</div>
</div>
<h2>🔧 手动配对工具 — QR 失败图片</h2>
<div class="stat">待配对: <b>{total_pending}</b> / 总计: <b>{total}</b> | 第 <b>{page}/{total_pages}</b> 页（每页{PAGE_SIZE}条）</div>
{page_nav}
  <table>
    <tr>
      <th>批次/文件名</th>
      <th>原图</th>
      <th>贴纸裁剪 + 顶部文字</th>
      <th>试块编号（自动保存）</th>
    </tr>
    {rows_html if rows_html else '<tr><td colspan="4" style="padding:20px;text-align:center;color:#888">🎉 所有图片均已配对（QR 或手动）</td></tr>'}
  </table>
  {page_nav}
<script>
var scale = 1;
function showLb(src) {{
  scale = 1;
  document.getElementById('lb-img').style.transform = 'scale(1)';
  document.getElementById('lb-img').src = src;
  document.getElementById('lb').classList.add('show');
}}
function hideLb() {{
  document.getElementById('lb').classList.remove('show');
}}
function closeLb(e) {{
  if (e.target === document.getElementById('lb')) hideLb();
}}
document.getElementById('lb').addEventListener('wheel', function(e) {{
  e.preventDefault();
  scale = Math.min(8, Math.max(0.5, scale * (e.deltaY < 0 ? 1.15 : 0.87)));
  document.getElementById('lb-img').style.transform = 'scale(' + scale + ')';
}}, {{passive:false}});
document.addEventListener('keydown', function(e) {{
  if (e.key === 'Escape') hideLb();
}});
document.querySelectorAll('.preview-img').forEach(function(img) {{
  img.addEventListener('click', function() {{
    showLb(this.getAttribute('data-fullsrc') || this.src);
  }});
}});
// 自动保存：select 变化时立即 POST
document.querySelectorAll('.sel-num').forEach(function(sel) {{
  sel.addEventListener('change', function() {{
    var key = this.getAttribute('data-key');
    var val = this.value;
    var ok = this.parentElement.querySelector('.save-ok');
    var body = encodeURIComponent(key) + '=' + encodeURIComponent(val);
    fetch('/save', {{method:'POST', body:body,
      headers:{{'Content-Type':'application/x-www-form-urlencoded'}}
    }}).then(function(r) {{
      if (r.ok && val) {{
        ok.style.display = 'inline';
        sel.style.borderColor = '#4CAF50';
      }} else {{
        ok.style.display = 'none';
        sel.style.borderColor = '';
      }}
    }});
  }});
}});
</script>
</body>
</html>"""


# ── HTTP 服务 ─────────────────────────────────────────────
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[manual_pair] {fmt % args}")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        # /img?path=... — 直接服务原始图片文件（全分辨率）
        if parsed.path == '/img':
            params = urllib.parse.parse_qs(parsed.query)
            img_path = params.get('path', [''])[0]
            if img_path and os.path.exists(img_path):
                import mimetypes
                mime = mimetypes.guess_type(img_path)[0] or 'application/octet-stream'
                with open(img_path, 'rb') as f:
                    data = f.read()
                self.send_response(200)
                self.send_header('Content-Type', mime)
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'max-age=3600')
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(404)
                self.end_headers()
            return

        params = urllib.parse.parse_qs(parsed.query)
        page = int(params.get('page', ['1'])[0])
        html = _render_page(page=page)
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        params_url = urllib.parse.parse_qs(parsed.query)
        page = params_url.get('page', ['1'])[0]

        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length).decode('utf-8')
        params = urllib.parse.parse_qs(body)

        manual = _load_json(MANUAL_PAIRS_FILE)
        updated = 0
        for key_enc, vals in params.items():
            val = vals[0].strip()
            if val:
                key = urllib.parse.unquote(key_enc)
                manual[key] = val
                updated += 1

        _save_json(MANUAL_PAIRS_FILE, manual)
        print(f"[manual_pair] 保存 {updated} 条记录 → {MANUAL_PAIRS_FILE}")

        # auto-save 返回 JSON，普通 form POST 返回重定向
        accept = self.headers.get('Content-Type', '')
        if 'application/x-www-form-urlencoded' in accept and updated <= 1:
            # AJAX 单条保存 → 返回 JSON
            resp = b'{"ok":true}'
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
        else:
            # 表单提交 → 重定向回当前页
            params_url = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            page = params_url.get('page', ['1'])[0]
            self.send_response(303)
            self.send_header('Location', f'/?page={page}')
            self.end_headers()


if __name__ == '__main__':
    print(f"手动配对工具启动: http://localhost:{PORT}")
    print(f"数据目录: {DATA_DIR}")
    print(f"保存到: {MANUAL_PAIRS_FILE}")
    from pipeline.qr import load_qr_cache
    load_qr_cache()
    server = http.server.HTTPServer(('', PORT), Handler)
    server.serve_forever()
