"""
贴纸检测审计页面 — 交互式单图检测
=================================
路由：
  GET  /audit              列出所有图片（按批次分组）
  GET  /audit/img?path=    原图服务
  POST /audit/detect       运行 SAM 检测，返回 JSON（base64 结果图）
"""
import os, json, base64, urllib.parse, threading
import cv2
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
DATA_DIR = os.path.abspath(DATA_DIR)
PAGE_SIZE = 18

# ── 检测逻辑 ─────────────────────────────────────────────────────

def _run_detection(img_bgr):
    """
    检测贴纸 + 混凝土面（复用 pipeline 逻辑）+ ROI 裁切 + 透视矫正 + SAM 分割图。
    SAM 只跑一次：detect_blue_sticker(Pass-1=SAM) 生成 masks 后缓存到
    sticker._last_sam_masks，_make_sam_visualization 直接复用，不重复推理。
    返回 dict: {found, center, ellipse, pass_used, sam_vis, concrete_box, roi_img, rectified, ...}
    """
    import pipeline.sticker as st_mod

    # 清空上一张图的 SAM 缓存，确保当前图不会复用旧数据
    st_mod._last_sam_masks = None

    # 1. 贴纸检测（Pass-1=SAM，会将 masks 写入 _last_sam_masks）
    sticker_mask, center, ellipse = st_mod.detect_blue_sticker(img_bgr)
    pass_used = st_mod._last_pass

    # 2. 混凝土面检测（复用 pipeline/roi.py 的 _detect_concrete_face）
    concrete_box = None
    if center is not None and ellipse is not None:
        try:
            from pipeline.roi import _detect_concrete_face
            cx, cy = center
            sticker_r = max(ellipse[1]) / 2.0
            x1, y1, x2, y2, _dbg, _ell = _detect_concrete_face(
                img_bgr, cx, cy, int(sticker_r))
            concrete_box = (x1, y1, x2, y2)
        except Exception as e:
            print(f"[audit] _detect_concrete_face failed: {e}")

    # 3. SAM 分割彩色可视化（传入已检测的 concrete_box 一起画出来）
    sam_vis = _make_sam_visualization(img_bgr, concrete_box)

    result = dict(
        found=center is not None,
        center=center,
        ellipse=ellipse,
        pass_used=pass_used,
        sam_vis=sam_vis,
        concrete_box=concrete_box,
        roi_img=None,
        rectified=None,
    )

    if center is None:
        return result

    # 4. ROI 裁切（用混凝土面bbox，fallback 到贴纸圆形裁切）
    from pipeline.roi import crop_roi_concrete_face, crop_roi_around_sticker
    if concrete_box is not None:
        roi_img, roi_center, roi_ellipse = crop_roi_concrete_face(
            img_bgr, center, ellipse)
    else:
        roi_img, roi_center, roi_ellipse = crop_roi_around_sticker(
            img_bgr, center, ellipse, scale=2.5)

    result['roi_img'] = roi_img
    result['roi_center'] = roi_center
    result['roi_ellipse'] = roi_ellipse

    # 透视矫正（椭圆→圆）
    from pipeline.rectify import rectify_perspective
    rect_img, H = rectify_perspective(roi_img, roi_ellipse)
    _, rect_center, rect_ell = st_mod.detect_blue_sticker(rect_img)
    result['rectified'] = rect_img
    result['rect_ellipse'] = rect_ell

    return result


def _make_sam_visualization(img_bgr, concrete_box=None):
    """生成 SAM 自动分割的彩色标注可视化图。
    优先复用 sticker.py 里缓存的 _last_sam_masks，避免重复推理。
    """
    import pipeline.sticker as st_mod

    # 优先使用已有缓存（detect_blue_sticker 跑过 SAM 后会存在这里）
    masks = st_mod._last_sam_masks
    if not masks:
        try:
            mask_gen = st_mod._get_sam_mask_gen()
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            masks = mask_gen.generate(img_rgb)
            st_mod._last_sam_masks = masks
        except Exception:
            return img_bgr.copy()

    if not masks:
        return img_bgr.copy()

    # 按面积降序排列
    masks = sorted(masks, key=lambda m: m['area'], reverse=True)

    vis = img_bgr.copy().astype(np.float32)
    # 半透明彩色叠加
    np.random.seed(42)
    colors = np.random.randint(60, 255, size=(len(masks), 3), dtype=np.uint8)

    for i, m in enumerate(masks):
        seg = m['segmentation']
        color = colors[i].tolist()
        overlay = np.zeros_like(img_bgr)
        overlay[seg] = color
        vis[seg] = vis[seg] * 0.45 + overlay[seg].astype(np.float32) * 0.55

    vis = np.clip(vis, 0, 255).astype(np.uint8)

    # 画轮廓线
    for i, m in enumerate(masks):
        seg = m['segmentation'].astype(np.uint8) * 255
        cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = colors[i].tolist()
        cv2.drawContours(vis, cnts, -1, color, 1)

    # 标注候选贴纸（小-中等圆形区域，排除混凝土面/背景大块）
    h, w = img_bgr.shape[:2]
    img_area = h * w
    for m in masks:
        area = m['area']
        # 面积在 0.5%~20% 之间才考虑（排除混凝土面等大区域）
        if area < img_area * 0.005 or area > img_area * 0.20:
            continue
        seg = m['segmentation'].astype(np.uint8) * 255
        cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        if len(cnt) < 5:
            continue
        perim = cv2.arcLength(cnt, True)
        circ = 4 * np.pi * area / (perim ** 2) if perim > 0 else 0
        bx, by, bw, bh = cv2.boundingRect(cnt)
        aspect = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0
        score = circ * aspect
        # 画检测到的混凝土面正方形框（最像正方形的大区域）
    if concrete_box is not None:
        x1, y1, x2, y2 = concrete_box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 3)  # 亮青色粗框
        cv2.putText(vis, 'concrete face', (x1 + 4, y1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(vis, 'concrete face', (x1 + 4, y1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    return vis


def _encode_jpeg(img, quality=85):
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode() if ok else ''


def _draw_detection(img_bgr, result):
    """在原图上画检测结果：绿椭圆=贴纸，青色框=混凝土面"""
    vis = img_bgr.copy()
    # 先画混凝土面的蓝色正方形框
    box = result.get('concrete_box')
    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 180, 0), 3)
        cv2.putText(vis, 'face', (x1 + 4, y1 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 0), 2)
    # 再画贴纸椭圆
    if result['ellipse'] is not None:
        cv2.ellipse(vis, result['ellipse'], (0, 255, 0), 3)
        cv2.circle(vis, result['center'], 6, (0, 0, 255), -1)
    label = result.get('pass_used', '')
    cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
    cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return vis


# ── 文件扫描 ──────────────────────────────────────────────────────

def _scan_images():
    """扫描 data/ 下所有图片，按批次分组。返回 [(batch_name, [paths...])]"""
    batches = {}
    for root, dirs, files in os.walk(DATA_DIR):
        imgs = sorted([
            os.path.join(root, f)
            for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        if imgs:
            batch = os.path.basename(root)
            batches[batch] = imgs
    return sorted(batches.items())


# ── FastAPI-compatible functions ───────────────────────────────────

def serve_image(fpath: str):
    """Read an image file and return (bytes, mime_type) or (None, None)."""
    if not fpath or not os.path.isfile(fpath):
        return None, None
    ext = os.path.splitext(fpath)[1].lower()
    mime = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png'}.get(ext.lstrip('.'), 'image/jpeg')
    with open(fpath, 'rb') as f:
        return f.read(), mime


def run_detect(img_path: str) -> dict:
    """Run detection on an image path and return JSON-serializable result dict."""
    if not img_path or not os.path.isfile(img_path):
        return {'error': '文件不存在'}

    img = cv2.imread(img_path)
    if img is None:
        return {'error': '无法读取图片'}

    # 按流水线规则缩放
    h, w = img.shape[:2]
    is_wet = 'image_' in os.path.basename(img_path).lower() or 'wet' in img_path.lower()
    max_sz = 1024 if is_wet else 1536
    scale = min(max_sz / w, max_sz / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    result = _run_detection(img)

    resp = {
        'found': result['found'],
        'pass_used': result['pass_used'],
        'original': _encode_jpeg(img, 90),
    }

    if result.get('sam_vis') is not None:
        resp['sam_vis'] = _encode_jpeg(result['sam_vis'], 85)

    if result['found']:
        det_vis = _draw_detection(img, result)
        resp['detection'] = _encode_jpeg(det_vis, 90)

        if result.get('roi_img') is not None:
            roi_vis = result['roi_img'].copy()
            if result.get('roi_ellipse'):
                cv2.ellipse(roi_vis, result['roi_ellipse'], (0, 255, 0), 2)
            resp['roi'] = _encode_jpeg(roi_vis, 90)

        if result.get('rectified') is not None:
            rect_vis = result['rectified'].copy()
            if result.get('rect_ellipse'):
                cv2.ellipse(rect_vis, result['rect_ellipse'], (0, 255, 0), 2)
            resp['rectified'] = _encode_jpeg(rect_vis, 90)

        resp['center'] = list(result['center'])
        resp['ellipse_axes'] = list(result['ellipse'][1]) if result['ellipse'] else None
    else:
        resp['detection'] = _encode_jpeg(img, 90)

    return resp


def build_audit_page(batch: str = "", page: int = 1) -> str:
    """Build the audit page HTML and return it as a string."""
    return _build_page_html(batch, page)


# ── HTTP 处理 (legacy http.server compat) ─────────────────────────

def handle_audit_get(handler, path, qs):
    if path == '/audit/img':
        _serve_img(handler, qs)
    elif path == '/audit':
        _serve_page(handler, qs)
    else:
        handler.send_error(404)


def handle_audit_post(handler, path):
    if path == '/audit/detect':
        _handle_detect(handler)
    else:
        handler.send_error(404)


def _serve_img(handler, qs):
    from web.server import _send_image
    fpath = qs.get('path', [''])[0]
    img_bytes, mime = serve_image(fpath)
    if img_bytes is None:
        handler.send_error(404)
        return
    _send_image(handler, img_bytes, mime)


def _handle_detect(handler):
    from web.server import _send_json
    length = int(handler.headers.get('Content-Length', 0))
    body = json.loads(handler.rfile.read(length)) if length else {}
    resp = run_detect(body.get('path', ''))
    code = 400 if 'error' in resp and resp['error'] else 200
    _send_json(handler, resp, code)


def _serve_page(handler, qs):
    """主审计页面 (legacy http.server compat)"""
    from web.server import _send_html
    batch_filter = qs.get('batch', [''])[0]
    page = int(qs.get('page', ['1'])[0])
    html = _build_page_html(batch_filter, page)
    _send_html(handler, html)


def _build_page_html(batch_filter: str = "", page: int = 1) -> str:
    """Build and return the audit page HTML."""

    all_batches = _scan_images()
    batch_names = [b for b, _ in all_batches]

    # 筛选批次
    if batch_filter:
        images = []
        for b, imgs in all_batches:
            if b == batch_filter:
                images = imgs
                break
    else:
        images = []
        for _, imgs in all_batches:
            images.extend(imgs)

    total = len(images)
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page = max(1, min(page, total_pages))
    start = (page - 1) * PAGE_SIZE
    page_images = images[start:start + PAGE_SIZE]

    # 批次选择器
    batch_options = '<option value="">全部批次</option>'
    for b in batch_names:
        sel = ' selected' if b == batch_filter else ''
        batch_options += f'<option value="{b}"{sel}>{b}</option>'

    # 图片卡片
    cards = ''
    for img_path in page_images:
        fname = os.path.basename(img_path)
        batch = os.path.basename(os.path.dirname(img_path))
        is_wet = 'image_' in fname.lower() or 'wet' in fname.lower()
        tag = 'WET' if is_wet else 'DRY'
        tag_color = '#4CAF50' if is_wet else '#FF9800'
        epath = urllib.parse.quote(img_path, safe='')
        cards += f'''
        <div class="card" id="card-{hash(img_path) & 0xFFFFFFFF}" data-path="{img_path}">
          <div class="card-header">
            <span class="tag" style="background:{tag_color}">{tag}</span>
            <span class="batch">{batch}</span>
          </div>
          <div class="img-row">
            <div class="img-col">
              <div class="img-label">原图</div>
              <img class="thumb" src="/audit/img?path={epath}" loading="lazy"
                   onclick="showLightbox(this.src)">
            </div>
            <div class="img-col sam-col" style="display:none">
              <div class="img-label">SAM 分割</div>
              <img class="thumb sam-img" onclick="showLightbox(this.src)">
            </div>
            <div class="img-col result-col" style="display:none">
              <div class="img-label">检测结果</div>
              <img class="thumb result-img" onclick="showLightbox(this.src)">
            </div>
            <div class="img-col roi-col" style="display:none">
              <div class="img-label">ROI 裁切</div>
              <img class="thumb roi-img" onclick="showLightbox(this.src)">
            </div>
            <div class="img-col rect-col" style="display:none">
              <div class="img-label">透视矫正</div>
              <img class="thumb rect-img" onclick="showLightbox(this.src)">
            </div>
          </div>
          <div class="card-footer">
            <span class="fname">{fname}</span>
            <button class="detect-btn" onclick="detectImage(this)">🔍 检测</button>
            <span class="status-label"></span>
          </div>
        </div>'''

    # 分页
    def page_url(p):
        params = f'page={p}'
        if batch_filter:
            params += f'&batch={urllib.parse.quote(batch_filter)}'
        return f'/audit?{params}'

    pagination = '<div class="pagination">'
    if page > 1:
        pagination += f'<a href="{page_url(page-1)}">◀ 上一页</a>'
    pagination += f'<span>第 {page}/{total_pages} 页（共 {total} 张）</span>'
    if page < total_pages:
        pagination += f'<a href="{page_url(page+1)}">下一页 ▶</a>'
    pagination += '</div>'

    html = f'''<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>贴纸检测审计</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }}
  .topbar {{ display: flex; align-items: center; gap: 16px; margin-bottom: 20px; flex-wrap: wrap; }}
  .topbar a {{ color: #667eea; text-decoration: none; font-size: 1.1em; }}
  .topbar h1 {{ font-size: 1.4em; }}
  select {{ background: #16213e; color: #eee; border: 1px solid #0f3460; padding: 6px 12px;
            border-radius: 6px; font-size: 0.9em; }}
  .cards-grid {{ display: grid; grid-template-columns: 1fr; gap: 16px; max-width: 1200px; margin: 0 auto; }}
  .card {{ background: #16213e; border-radius: 10px; padding: 16px; border: 1px solid #0f3460; }}
  .card-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }}
  .tag {{ padding: 2px 8px; border-radius: 4px; font-size: 0.75em; color: #fff; }}
  .batch {{ color: #888; font-size: 0.85em; }}
  .img-row {{ display: flex; gap: 12px; flex-wrap: wrap; }}
  .img-col {{ flex: 1; min-width: 200px; text-align: center; }}
  .img-label {{ font-size: 0.8em; color: #aaa; margin-bottom: 4px; }}
  .thumb {{ max-width: 100%; max-height: 300px; border-radius: 6px; cursor: pointer;
            border: 1px solid #333; object-fit: contain; background: #111; }}
  .card-footer {{ display: flex; align-items: center; gap: 12px; margin-top: 10px; }}
  .fname {{ color: #888; font-size: 0.8em; flex: 1; overflow: hidden; text-overflow: ellipsis;
            white-space: nowrap; }}
  .detect-btn {{ background: #667eea; color: #fff; border: none; padding: 8px 18px;
                 border-radius: 6px; cursor: pointer; font-size: 0.9em; transition: background 0.2s; }}
  .detect-btn:hover {{ background: #5a6fd6; }}
  .detect-btn:disabled {{ background: #555; cursor: wait; }}
  .status-label {{ font-size: 0.85em; font-weight: bold; }}
  .status-found {{ color: #4CAF50; }}
  .status-failed {{ color: #f44336; }}
  .pagination {{ text-align: center; margin-top: 24px; }}
  .pagination a {{ color: #667eea; text-decoration: none; margin: 0 12px; font-size: 1em; }}
  .pagination span {{ color: #888; }}
  /* Lightbox */
  .lightbox {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
               background: rgba(0,0,0,0.92); z-index: 9999; justify-content: center;
               align-items: center; cursor: zoom-out; }}
  .lightbox.active {{ display: flex; }}
  .lightbox img {{ max-width: 95vw; max-height: 95vh; object-fit: contain;
                   transform-origin: center; transition: transform 0.1s; }}
</style>
</head><body>
<div class="topbar">
  <a href="/">← 首页</a>
  <h1>🎯 贴纸检测审计</h1>
  <select onchange="location.href='/audit?batch='+encodeURIComponent(this.value)">
    {batch_options}
  </select>
  <button class="detect-btn" onclick="detectAll()" style="background:#e91e63">🚀 检测本页全部</button>
</div>

{pagination}

<div class="cards-grid">
  {cards}
</div>

{pagination}

<!-- Lightbox -->
<div class="lightbox" id="lightbox" onclick="closeLightbox()">
  <img id="lb-img" src="">
</div>

<script>
let lbScale = 1;
function showLightbox(src) {{
  const lb = document.getElementById('lightbox');
  const img = document.getElementById('lb-img');
  img.src = src;
  img.style.transform = 'scale(1)';
  lbScale = 1;
  lb.classList.add('active');
}}
function closeLightbox() {{
  document.getElementById('lightbox').classList.remove('active');
}}
document.getElementById('lightbox').addEventListener('wheel', function(e) {{
  e.preventDefault();
  lbScale *= e.deltaY < 0 ? 1.15 : 0.87;
  lbScale = Math.max(0.2, Math.min(lbScale, 10));
  document.getElementById('lb-img').style.transform = 'scale(' + lbScale + ')';
}});

async function detectImage(btn) {{
  const card = btn.closest('.card');
  const path = card.dataset.path;
  const label = card.querySelector('.status-label');
  btn.disabled = true;
  btn.textContent = '⏳ 检测中...';
  label.textContent = '';
  label.className = 'status-label';

  try {{
    const resp = await fetch('/audit/detect', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{path: path}})
    }});
    const data = await resp.json();

    if (data.error) {{
      label.textContent = '❌ ' + data.error;
      label.classList.add('status-failed');
      btn.textContent = '🔍 重试';
      btn.disabled = false;
      return;
    }}

    // SAM 分割
    if (data.sam_vis) {{
      const samCol = card.querySelector('.sam-col');
      const samImg = card.querySelector('.sam-img');
      samImg.src = 'data:image/jpeg;base64,' + data.sam_vis;
      samCol.style.display = '';
    }}

    // 显示检测结果
    const resultCol = card.querySelector('.result-col');
    const resultImg = card.querySelector('.result-img');
    resultImg.src = 'data:image/jpeg;base64,' + data.detection;
    resultCol.style.display = '';

    // ROI
    if (data.roi) {{
      const roiCol = card.querySelector('.roi-col');
      const roiImg = card.querySelector('.roi-img');
      roiImg.src = 'data:image/jpeg;base64,' + data.roi;
      roiCol.style.display = '';
    }}

    // 透视矫正
    if (data.rectified) {{
      const rectCol = card.querySelector('.rect-col');
      const rectImg = card.querySelector('.rect-img');
      rectImg.src = 'data:image/jpeg;base64,' + data.rectified;
      rectCol.style.display = '';
    }}

    label.textContent = data.found
      ? '✅ ' + data.pass_used + (data.center ? ' (' + data.center.join(',') + ')' : '')
      : '❌ FAILED';
    label.classList.add(data.found ? 'status-found' : 'status-failed');
    btn.textContent = '✓ 完成';
    btn.disabled = false;
  }} catch (err) {{
    label.textContent = '❌ 网络错误';
    label.classList.add('status-failed');
    btn.textContent = '🔍 重试';
    btn.disabled = false;
  }}
}}

async function detectAll() {{
  const btns = document.querySelectorAll('.detect-btn:not([disabled])');
  for (const btn of btns) {{
    if (btn.textContent.includes('检测本页')) continue;
    await detectImage(btn);
  }}
}}
</script>
</body></html>'''
    return html
