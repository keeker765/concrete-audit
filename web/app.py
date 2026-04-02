"""
混凝土试块审计系统 — FastAPI Web 服务
=====================================
端口 8765，路由：
  /              首页导航
  /audit         贴纸+矩形检测审计（交互式）
  /labels        结果审核
  /pairs         手动 QR 配对
  /inspect       单对匹配审计
  /roi_audit     ROI 框选审计
  /audit_single  单对审计（上传图片）— stub, 待实现
"""
import os, sys, threading
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, Request, Query, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, Response

PORT = 8765


@asynccontextmanager
async def lifespan(app):
    def _warmup():
        try:
            import label_tool
            label_tool.get_pairs()
            print(f'[Web] label_tool 缓存就绪（{len(label_tool.get_pairs())} 对）')
        except Exception as e:
            print(f'[Web] label_tool 预热失败: {e}')
    threading.Thread(target=_warmup, daemon=True).start()
    yield


app = FastAPI(title="混凝土试块审计系统", lifespan=lifespan)


# ── Helper: serve image file ────────────────────────────────────

def _serve_image_file(fpath: str) -> Response:
    """Read an image file and return a Response, or 404."""
    if not fpath or not os.path.isfile(fpath):
        return Response(status_code=404)
    ext = os.path.splitext(fpath)[1].lower().lstrip('.')
    mime = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png'}.get(ext, 'image/jpeg')
    with open(fpath, 'rb') as f:
        return Response(content=f.read(), media_type=mime)


# ══════════════════════════════════════════════════════════════════
#  HOME
# ══════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def home_page():
    from web.pages.home import build_home_page
    return build_home_page()


# ══════════════════════════════════════════════════════════════════
#  AUDIT — 贴纸检测审计
# ══════════════════════════════════════════════════════════════════

@app.get("/audit", response_class=HTMLResponse)
async def audit_page(batch: str = "", page: int = 1):
    from web.pages.audit import build_audit_page
    return build_audit_page(batch, page)


@app.get("/audit/img")
async def audit_img(path: str = ""):
    return _serve_image_file(path)


@app.post("/audit/detect")
async def audit_detect(request: Request):
    from web.pages.audit import run_detect
    body = await request.json()
    result = run_detect(body.get('path', ''))
    code = 400 if result.get('error') else 200
    return JSONResponse(content=result, status_code=code)


# ══════════════════════════════════════════════════════════════════
#  LABELS — 结果审核
# ══════════════════════════════════════════════════════════════════

@app.get("/labels", response_class=HTMLResponse)
async def labels_page(verdict: str = "all", labeled: str = "all",
                      page: int = 1, invalid_reason: str = "all"):
    from web.pages.labels import build_labels_page
    html, _ready = build_labels_page(verdict, labeled, page, invalid_reason)
    return html


@app.get("/labels/img")
async def labels_img(path: str = ""):
    return _serve_image_file(path)


@app.get("/labels/data")
async def labels_data():
    from web.pages.labels import get_labels_data
    return get_labels_data()


@app.post("/labels/save")
async def labels_save(request: Request):
    from web.pages.labels import save_label
    body = await request.json()
    return save_label(body)


# ══════════════════════════════════════════════════════════════════
#  PAIRS — 手动 QR 配对
# ══════════════════════════════════════════════════════════════════

@app.get("/pairs", response_class=HTMLResponse)
async def pairs_page(page: int = 1):
    from web.pages.pairs import build_pairs_page
    return build_pairs_page(page)


@app.get("/pairs/img")
async def pairs_img(path: str = ""):
    return _serve_image_file(path)


@app.post("/pairs/save")
async def pairs_save(request: Request):
    from web.pages.pairs import save_pair
    body = await request.json()
    return save_pair(body)


# ══════════════════════════════════════════════════════════════════
#  INSPECT — 单对匹配审计
# ══════════════════════════════════════════════════════════════════

@app.get("/inspect", response_class=HTMLResponse)
async def inspect_page(page: int = 1, q: str = ""):
    from web.pages.inspect import build_inspect_page
    return build_inspect_page(page, q)


@app.post("/inspect/run")
async def inspect_run(request: Request):
    from web.pages.inspect import run_inspect
    body = await request.json()
    return run_inspect(body.get('wet_path', ''), body.get('dry_path', ''),
                       body.get('method', 'sp'))


# ══════════════════════════════════════════════════════════════════
#  ROI AUDIT — ROI 框选审计
# ══════════════════════════════════════════════════════════════════

@app.get("/roi_audit", response_class=HTMLResponse)
async def roi_audit_page(page: int = 1, q: str = "",
                         filter: str = "all", idx: int = None):
    from web.pages.roi_audit import build_roi_audit_page
    return build_roi_audit_page(page, q, filter, idx)


@app.post("/roi_audit/save")
async def roi_audit_save(request: Request):
    from web.pages.roi_audit import save_roi_label
    body = await request.json()
    result, code = save_roi_label(body)
    return JSONResponse(content=result, status_code=code)


# ══════════════════════════════════════════════════════════════════
#  AUDIT SINGLE — stub routes (to be implemented)
# ══════════════════════════════════════════════════════════════════

@app.get("/audit_single", response_class=HTMLResponse)
async def audit_single_page():
    from web.pages.audit_single import build_page
    return build_page()


@app.post("/audit_single/run")
async def audit_single_run(
    wet_file: UploadFile = File(...),
    dry_file: UploadFile = File(...),
    method: str = Form("sp"),
):
    from web.pages.audit_single import run_match
    wet_bytes = await wet_file.read()
    dry_bytes = await dry_file.read()
    return await run_match(wet_bytes, dry_bytes, method)


# ══════════════════════════════════════════════════════════════════
#  Main entry point
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import uvicorn
    print(f'[Web] 统一审计服务启动: http://localhost:{PORT}')
    print(f'  /        首页')
    print(f'  /audit   贴纸检测审计')
    print(f'  /labels  结果审核（后台预热中...）')
    print(f'  /pairs   手动配对')
    print(f'  /inspect 单对匹配审计')
    print(f'  /roi_audit ROI 框选审计')
    uvicorn.run(app, host="0.0.0.0", port=PORT)
