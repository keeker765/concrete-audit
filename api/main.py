"""
混凝土试块造假识别 — 纯 JSON API 服务
======================================
端口 8080，无 HTML，仅 REST API。

路由:
  GET  /health    健康检查 + 模型就绪状态
  POST /match     上传湿/干照片 → 返回 SP+LightGlue 鉴定结果

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

# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

from pipeline.config import DEVICE, MATCH_THRESHOLD, MIN_MATCHES
from pipeline.runner import run_single_pair, _get_models


# ── 启动时预热模型 ────────────────────────────────────────────────────────────

def _warmup():
    """预加载 SP+LG 模型，触发自动权重下载（首次需联网）。"""
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


# ── GET /health ───────────────────────────────────────────────────────────────

@app.get("/health", summary="健康检查")
def health():
    """
    返回服务状态和模型就绪情况。

    - `models_ready`: true 表示 SP+LG 权重已加载，可以接受请求
    - `device`: 当前推理设备（cuda / mps / cpu）
    """
    from pipeline.runner import _CACHED_MODELS
    return {
        "status": "ok",
        "device": DEVICE,
        "models_ready": "sp" in _CACHED_MODELS,
        "match_threshold": MATCH_THRESHOLD,
        "min_matches": MIN_MATCHES,
    }


# ── POST /match ───────────────────────────────────────────────────────────────

@app.post("/match", summary="鉴定一对试块照片")
async def match(
    wet: UploadFile = File(..., description="湿态照片（制作时拍摄）"),
    dry: UploadFile = File(..., description="干态照片（送检时拍摄）"),
    method: str = Form("sp", description="匹配方法: sp / aliked / sift / hardnet"),
):
    """
    上传湿/干照片，运行特征匹配流水线，返回造假鉴定结果。

    **响应字段说明**

    | 字段 | 类型 | 说明 |
    |------|------|------|
    | verdict | string | SAME（真实）/ DIFFERENT（换块）/ INSUFFICIENT（匹配点不足）/ INVALID（检测失败）|
    | score | float | 综合评分 [0, 1]，≥ threshold 判 SAME |
    | n_raw | int | 原始匹配点数 |
    | n_filtered | int | 过滤贴纸区域后的点数 |
    | n_inliers | int | RANSAC 内点数 |
    | mean_conf | float | 平均匹配置信度 |
    | inlier_ratio | float | RANSAC 内点率 |
    | elapsed | float | 推理耗时（秒）|
    | error | string / null | 错误信息（正常为 null）|
    """
    # 若模型尚未就绪（后台线程还在加载），同步等待
    from pipeline.runner import _CACHED_MODELS
    if method not in _CACHED_MODELS:
        _get_models(method)

    wet_bytes = await wet.read()
    dry_bytes = await dry.read()

    tmp_dir = tempfile.mkdtemp(prefix="api_match_")
    wet_path = os.path.join(tmp_dir, "wet.jpg")
    dry_path = os.path.join(tmp_dir, "dry.jpg")

    try:
        with open(wet_path, "wb") as f:
            f.write(wet_bytes)
        with open(dry_path, "wb") as f:
            f.write(dry_bytes)

        result = run_single_pair(wet_path, dry_path, method=method)

        return JSONResponse({
            "verdict":      result.get("verdict", "INVALID"),
            "score":        round(float(result.get("score", 0)), 4),
            "n_raw":        result.get("n_raw", 0),
            "n_filtered":   result.get("n_filtered", 0),
            "n_inliers":    result.get("n_inliers", 0),
            "mean_conf":    round(float(result.get("mean_conf", 0)), 4),
            "inlier_ratio": round(float(result.get("inlier_ratio", 0)), 4),
            "elapsed":      round(float(result.get("elapsed", 0)), 2),
            "error":        result.get("error"),
        })

    except Exception:
        return JSONResponse({"error": traceback.format_exc()}, status_code=500)

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


# ── 直接运行入口 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    print(f"[API] 启动 http://0.0.0.0:{port}")
    print(f"[API] 文档: http://localhost:{port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=port)
