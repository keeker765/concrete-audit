"""
混凝土试块造假识别 — 纯 JSON API 服务
======================================
端口 8080，无额外页面依赖。

路由:
  GET  /                  嵌入式 audit_single 审计页面 (HTML)
  GET  /health            健康检查 + 模型就绪状态
  POST /match             上传湿/干照片 → 返回 SP+LightGlue 鉴定结果
  POST /audit_single/run  与 /match 等价（字段名兼容 audit_single HTML）

启动:
  cd <项目根目录>
  python api/main.py            # 或
  uvicorn api.main:app --port 8080 --reload

首次启动会自动从 HuggingFace Hub 下载 SuperPoint / LightGlue 权重（约 50 MB）。
"""

import os
import sys
import time
import tempfile
import traceback
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from pipeline.config import DEVICE, MATCH_THRESHOLD, MIN_MATCHES
from pipeline.runner import run_single_pair, _get_models


# ── 图像 → base64 ─────────────────────────────────────────────────────────────

def _b64(img_bgr, quality: int = 95) -> str:
    """BGR ndarray → base64 JPEG"""
    import cv2, base64
    if img_bgr is None:
        return ""
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def _b64_png(img_bgr) -> str:
    """BGR ndarray → base64 PNG（无损）"""
    import cv2, base64
    if img_bgr is None:
        return ""
    ok, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


# ── 启动时预热模型 ────────────────────────────────────────────────────────────

def _warmup():
    print(f"[API] 正在加载 SuperPoint + LightGlue（device={DEVICE}）...")
    t0 = time.time()
    _get_models("sp")
    print(f"[API] 模型就绪，耗时 {time.time() - t0:.1f}s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    import threading
    threading.Thread(target=_warmup, daemon=True).start()
    yield


# ── FastAPI 应用 ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="混凝土试块鉴定 API",
    description="上传湿/干照片，返回 SuperPoint+LightGlue 造假鉴定结果。",
    version="1.0.0",
    lifespan=lifespan,
)

# 允许本地 HTML 文件（file://）及任意前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 公共匹配逻辑 ──────────────────────────────────────────────────────────────

async def _run_match_core(wet_bytes: bytes, dry_bytes: bytes, method: str) -> dict:
    """执行完整流水线并返回与 audit_single/run 格式完全一致的字典。"""
    from pipeline.runner import _CACHED_MODELS
    if method not in _CACHED_MODELS:
        _get_models(method)

    tmp_dir = tempfile.mkdtemp(prefix="api_match_")
    wet_path = os.path.join(tmp_dir, "wet.jpg")
    dry_path = os.path.join(tmp_dir, "dry.jpg")

    try:
        with open(wet_path, "wb") as f:
            f.write(wet_bytes)
        with open(dry_path, "wb") as f:
            f.write(dry_bytes)

        result = run_single_pair(wet_path, dry_path, method=method)

        return {
            "verdict":      result.get("verdict", "INVALID"),
            "score":        round(float(result.get("score", 0)), 4),
            "n_raw":        result.get("n_raw", 0),
            "n_filtered":   result.get("n_filtered", 0),
            "n_inliers":    result.get("n_inliers", 0),
            "rot_deg":      round(float(result.get("rot_deg", 0)), 1),
            "elapsed":      round(float(result.get("elapsed", 0)), 2),
            "error":        result.get("error"),
            "mean_conf":    round(float(result.get("mean_conf", 0)), 4),
            "inlier_ratio": round(float(result.get("inlier_ratio", 0)), 4),
            "wet_previs":   _b64(result.get("wet_previs")),
            "dry_previs":   _b64(result.get("dry_previs")),
            "wet_roi":      _b64_png(result.get("wet_roi")),
            "dry_roi":      _b64_png(result.get("dry_roi")),
            "match_vis":    _b64(result.get("match_vis")),
            "match_points": result.get("match_points", []),
        }

    except Exception:
        return {"error": traceback.format_exc()}

    finally:
        for p in (wet_path, dry_path):
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass


# ── GET / ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, summary="审计页面")
def index_page():
    """嵌入式 audit_single 审计页面，POST 到 /audit_single/run。"""
    # 直接复用 web/pages/audit_single.py 的 build_page()
    try:
        import web.pages.audit_single as _as
        return _as.build_page()
    except ImportError:
        return HTMLResponse("<h2>请从项目根目录启动：<code>python api/main.py</code></h2>")


# ── GET /health ───────────────────────────────────────────────────────────────

@app.get("/health", summary="健康检查")
def health():
    from pipeline.runner import _CACHED_MODELS
    return {
        "status":          "ok",
        "device":          DEVICE,
        "models_ready":    "sp" in _CACHED_MODELS,
        "match_threshold": MATCH_THRESHOLD,
        "min_matches":     MIN_MATCHES,
    }


# ── POST /match ───────────────────────────────────────────────────────────────

@app.post("/match", summary="鉴定一对试块照片（通用接口）")
async def match(
    wet:    UploadFile = File(..., description="湿态照片"),
    dry:    UploadFile = File(..., description="干态照片"),
    method: str = Form("sp", description="sp / aliked / sift / hardnet"),
):
    """
    返回与 `/audit_single/run` 完全相同的 JSON 格式，
    包含 base64 预处理图、ROI 图、匹配可视化图及匹配点坐标。
    """
    data = await _run_match_core(await wet.read(), await dry.read(), method)
    code = 500 if ("error" in data and not data.get("verdict")) else 200
    return JSONResponse(data, status_code=code)


# ── POST /audit_single/run ────────────────────────────────────────────────────

@app.post("/audit_single/run", summary="鉴定（兼容 audit_single HTML）")
async def audit_single_run(
    wet_file: UploadFile = File(..., description="湿态照片"),
    dry_file: UploadFile = File(..., description="干态照片"),
    method:   str = Form("sp"),
):
    """
    字段名与 web/pages/audit_single.py 的 HTML 表单保持一致，
    可直接在浏览器 `http://localhost:8080` 使用可视化审计界面。
    """
    data = await _run_match_core(await wet_file.read(), await dry_file.read(), method)
    code = 500 if ("error" in data and not data.get("verdict")) else 200
    return JSONResponse(data, status_code=code)


# ── 直接运行入口 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    print(f"[API] 启动 http://0.0.0.0:{port}")
    print(f"[API] 审计界面: http://localhost:{port}/")
    print(f"[API] API 文档: http://localhost:{port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=port)
