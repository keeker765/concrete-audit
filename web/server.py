"""
混凝土试块审计系统 — 统一 Web 服务
==================================
端口 8765，路由：
  /             首页导航
  /audit        贴纸+矩形检测审计（交互式）
  /labels       结果审核
  /pairs        手动 QR 配对
"""
import os, sys, json, urllib.parse, threading
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

PORT = 8765


class Router(BaseHTTPRequestHandler):
    """统一路由分发"""

    def log_message(self, fmt, *args):
        pass  # 安静模式

    # ── GET ──────────────────────────────────────────────────────
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip('/')
        qs = urllib.parse.parse_qs(parsed.query)

        if path == '' or path == '/':
            from web.pages.home import handle_home
            handle_home(self)
        elif path.startswith('/audit_single'):
            from web.pages.audit_single import _serve_audit_single
            _serve_audit_single(self, qs)
        elif path.startswith('/audit'):
            from web.pages.audit import handle_audit_get
            handle_audit_get(self, path, qs)
        elif path.startswith('/labels'):
            from web.pages.labels import handle_labels_get
            handle_labels_get(self, path, qs)
        elif path.startswith('/pairs'):
            from web.pages.pairs import handle_pairs_get
            handle_pairs_get(self, path, qs)
        elif path.startswith('/inspect'):
            from web.pages.inspect import _serve_page
            _serve_page(self, qs)
        elif path.startswith('/roi_audit'):
            from web.pages.roi_audit import handle_roi_audit_get
            handle_roi_audit_get(self, qs)
        else:
            self.send_error(404)

    # ── POST ─────────────────────────────────────────────────────
    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip('/')

        if path.startswith('/labels'):
            from web.pages.labels import handle_labels_post
            handle_labels_post(self, path)
        elif path.startswith('/pairs'):
            from web.pages.pairs import handle_pairs_post
            handle_pairs_post(self, path)
        elif path.startswith('/audit_single'):
            from web.pages.audit_single import handle_audit_single_post
            handle_audit_single_post(self, path)
        elif path.startswith('/audit'):
            from web.pages.audit import handle_audit_post
            handle_audit_post(self, path)
        elif path.startswith('/inspect'):
            from web.pages.inspect import handle_inspect_post
            handle_inspect_post(self, path)
        elif path.startswith('/roi_audit'):
            from web.pages.roi_audit import handle_roi_audit_post
            handle_roi_audit_post(self, path)
        else:
            self.send_error(404)


def _send_html(handler, html, code=200):
    handler.send_response(code)
    handler.send_header('Content-Type', 'text/html; charset=utf-8')
    handler.end_headers()
    handler.wfile.write(html.encode('utf-8'))


def _send_json(handler, obj, code=200):
    handler.send_response(code)
    handler.send_header('Content-Type', 'application/json; charset=utf-8')
    handler.end_headers()
    handler.wfile.write(json.dumps(obj, ensure_ascii=False).encode('utf-8'))


def _send_image(handler, img_bytes, mime='image/jpeg'):
    handler.send_response(200)
    handler.send_header('Content-Type', mime)
    handler.send_header('Content-Length', str(len(img_bytes)))
    handler.end_headers()
    handler.wfile.write(img_bytes)


def main():
    # 后台线程预热 label_tool 缓存（避免阻塞服务启动）
    import threading
    def _warmup():
        try:
            import label_tool
            label_tool.get_pairs()
            print(f'[Web] label_tool 缓存就绪（{len(label_tool.get_pairs())} 对）')
        except Exception as e:
            print(f'[Web] label_tool 预热失败: {e}')
    threading.Thread(target=_warmup, daemon=True).start()

    server = HTTPServer(('0.0.0.0', PORT), Router)
    print(f'[Web] 统一审计服务启动: http://localhost:{PORT}')
    print(f'  /        首页')
    print(f'  /audit   贴纸检测审计')
    print(f'  /labels  结果审核（后台预热中...）')
    print(f'  /pairs   手动配对')
    print(f'  /audit_single  单对审计（上传图片）')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n[Web] 已停止')


if __name__ == '__main__':
    main()
