"""
/audit_single — 单对审计（上传图片）
上传一对湿/干图 → 选择方法 → 跑完整流水线 → 交互式查看匹配结果
FastAPI 版本
"""
import base64
import cv2


def _b64(img_bgr, quality=95):
    """BGR ndarray → base64 JPEG (高质量)"""
    if img_bgr is None:
        return ''
    ok, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode() if ok else ''


def _b64_png(img_bgr):
    """BGR ndarray → base64 PNG (无损)"""
    if img_bgr is None:
        return ''
    ok, buf = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buf.tobytes()).decode() if ok else ''


def build_page() -> str:
    """返回 audit_single 页面 HTML"""
    return _PAGE_HTML


async def run_match(wet_bytes: bytes, dry_bytes: bytes, method: str = 'sp') -> dict:
    """运行匹配流水线，返回 JSON 响应字典"""
    import os
    import tempfile
    import traceback
    import pipeline.runner as runner

    tmp_dir = tempfile.mkdtemp(prefix='audit_single_')
    wet_path = os.path.join(tmp_dir, 'wet.jpg')
    dry_path = os.path.join(tmp_dir, 'dry.jpg')

    try:
        with open(wet_path, 'wb') as f:
            f.write(wet_bytes)
        with open(dry_path, 'wb') as f:
            f.write(dry_bytes)

        result = runner.run_single_pair(wet_path, dry_path, method=method)

        return {
            'verdict':      result.get('verdict', 'INVALID'),
            'score':        float(result.get('score', 0)),
            'n_raw':        result.get('n_raw', 0),
            'n_filtered':   result.get('n_filtered', 0),
            'n_inliers':    result.get('n_inliers', 0),
            'rot_deg':      float(result.get('rot_deg', 0)),
            'elapsed':      float(result.get('elapsed', 0)),
            'error':        result.get('error'),
            'mean_conf':    float(result.get('mean_conf', 0)),
            'inlier_ratio': float(result.get('inlier_ratio', 0)),
            'wet_previs':   _b64(result.get('wet_previs')),
            'dry_previs':   _b64(result.get('dry_previs')),
            'wet_roi':      _b64_png(result.get('wet_roi')),
            'dry_roi':      _b64_png(result.get('dry_roi')),
            'match_vis':    _b64(result.get('match_vis')),
            'match_points': result.get('match_points', []),
        }
    except Exception:
        return {'error': traceback.format_exc()}
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


# ── 页面 HTML ────────────────────────────────────────────────────
_PAGE_HTML = r'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>单对审计</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#e6edf3;font-family:system-ui,sans-serif;display:flex;height:100vh;overflow:hidden}
#left{width:380px;min-width:320px;display:flex;flex-direction:column;border-right:1px solid #30363d;background:#161b22;overflow-y:auto}
#right{flex:1;display:flex;flex-direction:column;overflow:auto;padding:16px}
.topbar{padding:10px 16px;display:flex;align-items:center;gap:10px;background:#161b22;border-bottom:1px solid #30363d}
.topbar a{color:#667eea;text-decoration:none;font-size:.9em}
.topbar h2{font-size:1em;color:#c9d1d9}
.section{padding:12px 16px;border-bottom:1px solid #21262d}
.section-title{font-size:.8em;color:#8b949e;margin-bottom:8px;text-transform:uppercase;letter-spacing:.5px}

/* upload zones */
.drop-zone{border:2px dashed #30363d;border-radius:8px;padding:24px 16px;text-align:center;cursor:pointer;
  transition:border-color .2s,background .2s;position:relative;min-height:120px;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:6px;margin-bottom:8px}
.drop-zone:hover,.drop-zone.drag-over{border-color:#667eea;background:rgba(102,126,234,.06)}
.drop-zone.has-file{border-color:#3fb950;border-style:solid}
.drop-zone .icon{font-size:1.8em;opacity:.5}
.drop-zone .hint{font-size:.8em;color:#8b949e}
.drop-zone .fname{font-size:.8em;color:#3fb950;word-break:break-all;margin-top:4px}
.drop-zone input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer}
.drop-zone .thumb{max-width:100%;max-height:80px;border-radius:4px;margin-top:6px}

/* controls */
select,button{background:#21262d;color:#e6edf3;border:1px solid #30363d;padding:6px 14px;border-radius:6px;cursor:pointer;font-size:.9em}
button.run{background:#1f6feb;border-color:#1f6feb;font-weight:600;width:100%;padding:10px;margin-top:4px;font-size:1em}
button.run:hover{background:#388bfd}
button:disabled{opacity:.5;cursor:default}
#status{font-size:.85em;color:#8b949e;text-align:center;padding:6px 0}

/* results */
.verdict-SAME{color:#3fb950;font-weight:700}
.verdict-DIFFERENT{color:#f85149;font-weight:700}
.verdict-INSUFFICIENT{color:#d29922;font-weight:700}
.verdict-INVALID{color:#8b949e;font-weight:700}
.score-box{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px 20px;display:flex;gap:24px;flex-wrap:wrap;margin-bottom:16px}
.metric{display:flex;flex-direction:column;gap:2px}
.metric .label{font-size:.75em;color:#8b949e}
.metric .val{font-size:1.1em;font-weight:600}
.img-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:12px;margin-bottom:16px}
.img-card{background:#161b22;border:1px solid #30363d;border-radius:8px;overflow:hidden}
.img-card .card-title{padding:8px 12px;font-size:.85em;color:#8b949e;background:#21262d}
.img-card img{width:100%;display:block;cursor:zoom-in}
.error-box{background:#2d1b1b;border:1px solid #f85149;border-radius:8px;padding:14px;
  color:#f85149;font-family:monospace;font-size:.82em;white-space:pre-wrap;word-break:break-all}

/* interactive canvas */
#canvas-wrap{position:relative;background:#161b22;border:1px solid #30363d;border-radius:8px;overflow:hidden;margin-bottom:16px}
#canvas-header{padding:8px 12px;font-size:.85em;color:#8b949e;background:#21262d;display:flex;align-items:center;gap:12px}
#canvas-header label{cursor:pointer;user-select:none;display:flex;align-items:center;gap:4px;font-size:.82em}
#matchCanvas{display:block;cursor:crosshair;width:100%}

/* detail panel below canvas */
#detailPanel{display:none;background:#161b22;border:1px solid #30363d;border-radius:8px;overflow:hidden;margin-bottom:16px}
#detailPanel.visible{display:block}
#detailHeader{padding:8px 12px;font-size:.85em;color:#8b949e;background:#21262d;display:flex;align-items:center;justify-content:space-between}
#detailHeader .close-btn{background:none;border:none;color:#8b949e;cursor:pointer;font-size:1.1em;padding:2px 6px}
#detailHeader .close-btn:hover{color:#e6edf3}
#detailBody{display:flex;gap:16px;padding:16px;align-items:flex-start}
.detail-crop{flex:1;text-align:center;min-width:0}
.detail-crop .crop-title{font-size:.8em;color:#8b949e;margin-bottom:6px}
.detail-crop canvas{display:block;margin:0 auto;border:1px solid #30363d;border-radius:4px;background:#0d1117;cursor:zoom-in;max-width:100%}
#detailInfo{padding:0 16px 12px;font-size:.85em;color:#c9d1d9;line-height:1.6}

/* lightbox */
.lightbox{display:none;position:fixed;inset:0;background:rgba(0,0,0,.92);z-index:9999;align-items:center;justify-content:center;cursor:zoom-out}
.lightbox.active{display:flex}
.lightbox img,.lightbox canvas{max-width:95vw;max-height:90vh;object-fit:contain}

/* spinner */
.spinner{width:40px;height:40px;border:4px solid #21262d;border-top:4px solid #667eea;
  border-radius:50%;animation:spin .8s linear infinite;margin:20px auto}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
</head><body>

<div id="left">
  <div class="topbar">
    <a href="/">← 首页</a>
    <h2>单对审计</h2>
  </div>

  <div class="section">
    <div class="section-title">湿态图片 (WET)</div>
    <div class="drop-zone" id="dropWet">
      <span class="icon">📷</span>
      <span class="hint">点击或拖拽上传湿态图片</span>
      <input type="file" accept="image/*" onchange="onFile(this,'wet')">
    </div>
  </div>

  <div class="section">
    <div class="section-title">干态图片 (DRY)</div>
    <div class="drop-zone" id="dropDry">
      <span class="icon">📷</span>
      <span class="hint">点击或拖拽上传干态图片</span>
      <input type="file" accept="image/*" onchange="onFile(this,'dry')">
    </div>
  </div>

  <div class="section">
    <div class="section-title">匹配方法</div>
    <select id="method" style="width:100%">
      <option value="sp" selected>SuperPoint + LightGlue</option>
      <option value="aliked">ALIKED + LightGlue</option>
      <option value="sift">SIFT + LightGlue</option>
      <option value="hardnet">DoGHardNet + LightGlue</option>
    </select>
    <button class="run" id="runBtn" onclick="runMatch()" disabled>▶ 运行匹配</button>
    <div id="status">← 请上传两张图片</div>
  </div>
</div>

<div id="right">
  <div id="results"></div>
</div>

<div class="lightbox" id="lb" onclick="this.classList.remove('active')">
  <img id="lbImg" src="">
</div>

<script>
/* ── state ── */
let wetFile = null, dryFile = null;
let matchData = null;
let wetRoiImg = null, dryRoiImg = null;
let showOutliers = false;
let canvasScale = 1;
let selectedIdx = -1;
const GAP = 20;
const CROP_SIZE = 480;  // 大裁切尺寸
const CROP_RADIUS = 60; // 原图裁切半径(像素) — 小范围看清纹理

// 根据置信度返回颜色: 绿(高) / 黄(中) / 红(低)
function confColor(conf, alpha) {
  if (conf >= 0.7) return `rgba(63,185,80,${alpha})`;   // 绿
  if (conf >= 0.4) return `rgba(227,179,38,${alpha})`;   // 黄
  return `rgba(248,81,73,${alpha})`;                      // 红
}

/* ── file upload ── */
function onFile(input, which) {
  const f = input.files[0];
  if (!f) return;
  if (which === 'wet') wetFile = f; else dryFile = f;
  const zone = input.closest('.drop-zone');
  zone.classList.add('has-file');
  const reader = new FileReader();
  reader.onload = e => {
    zone.innerHTML = `<span class="icon">✅</span><span class="fname">${f.name}</span>
      <img class="thumb" src="${e.target.result}">
      <input type="file" accept="image/*" onchange="onFile(this,'${which}')">`;
  };
  reader.readAsDataURL(f);
  updateBtn();
}

['dropWet','dropDry'].forEach(id => {
  const zone = document.getElementById(id);
  const which = id === 'dropWet' ? 'wet' : 'dry';
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault(); zone.classList.remove('drag-over');
    const f = e.dataTransfer.files[0];
    if (!f || !f.type.startsWith('image/')) return;
    if (which === 'wet') wetFile = f; else dryFile = f;
    zone.classList.add('has-file');
    const reader = new FileReader();
    reader.onload = ev => {
      zone.innerHTML = `<span class="icon">✅</span><span class="fname">${f.name}</span>
        <img class="thumb" src="${ev.target.result}">
        <input type="file" accept="image/*" onchange="onFile(this,'${which}')">`;
    };
    reader.readAsDataURL(f);
    updateBtn();
  });
});

function updateBtn() {
  document.getElementById('runBtn').disabled = !(wetFile && dryFile);
  if (wetFile && dryFile)
    document.getElementById('status').textContent = '就绪，点击运行';
}

/* ── run match ── */
async function runMatch() {
  const btn = document.getElementById('runBtn');
  const status = document.getElementById('status');
  btn.disabled = true;
  status.textContent = '⏳ 运行中（可能需要 10-30 秒）…';
  document.getElementById('results').innerHTML = '<div class="spinner"></div>';

  const fd = new FormData();
  fd.append('wet_file', wetFile);
  fd.append('dry_file', dryFile);
  fd.append('method', document.getElementById('method').value);

  try {
    const resp = await fetch('/audit_single/run', { method: 'POST', body: fd });
    const data = await resp.json();
    matchData = data;
    renderResults(data);
    status.textContent = data.error && !data.verdict ? '❌ 出错' : `完成 (${(data.elapsed||0).toFixed(1)}s)`;
  } catch(e) {
    status.textContent = '❌ 网络错误: ' + e;
    document.getElementById('results').innerHTML =
      `<div class="error-box"><b>网络错误</b>\n${e}</div>`;
  }
  btn.disabled = false;
}

/* ── render results ── */
function renderResults(d) {
  const res = document.getElementById('results');
  if (d.error && !d.verdict) {
    res.innerHTML = `<div class="error-box"><b>错误</b>\n${d.error}</div>`;
    return;
  }

  const vclass = 'verdict-' + d.verdict;
  const barColor = d.verdict==='SAME'?'#3fb950':d.verdict==='DIFFERENT'?'#f85149':'#d29922';
  const barW = Math.min(100, (d.score||0)*100);

  let h = `
    <div class="score-box">
      <div class="metric"><span class="label">判定</span>
        <span class="val ${vclass}">${d.verdict}</span></div>
      <div class="metric"><span class="label">分数</span>
        <span class="val">${(d.score||0).toFixed(3)}</span></div>
      <div class="metric"><span class="label">置信度</span>
        <span class="val">${(d.mean_conf||0).toFixed(3)}</span></div>
      <div class="metric"><span class="label">内点率</span>
        <span class="val">${(d.inlier_ratio||0).toFixed(3)}</span></div>
      <div class="metric"><span class="label">匹配点</span>
        <span class="val">${d.n_filtered} / ${d.n_raw}</span></div>
      <div class="metric"><span class="label">内点</span>
        <span class="val">${d.n_inliers}</span></div>
      <div class="metric"><span class="label">旋转</span>
        <span class="val">${(d.rot_deg||0).toFixed(1)}°</span></div>
      <div class="metric"><span class="label">耗时</span>
        <span class="val">${(d.elapsed||0).toFixed(1)}s</span></div>
    </div>
    <div style="height:8px;background:#21262d;border-radius:4px;margin-bottom:16px;overflow:hidden">
      <div style="width:${barW}%;height:100%;background:${barColor};border-radius:4px;transition:width .4s"></div>
    </div>`;

  if (d.error) {
    h += `<div class="error-box" style="margin-bottom:16px"><b>警告</b>\n${d.error}</div>`;
  }

  // preprocess images
  const preImgs = [
    ['WET 预处理', d.wet_previs],
    ['DRY 预处理', d.dry_previs],
  ].filter(x => x[1]);
  if (preImgs.length) {
    h += '<div class="img-grid">' + preImgs.map(([t,b]) =>
      `<div class="img-card"><div class="card-title">${t}</div>
       <img src="data:image/jpeg;base64,${b}" onclick="zoom(this)"></div>`
    ).join('') + '</div>';
  }

  // interactive canvas
  if (d.wet_roi && d.dry_roi) {
    h += `<div id="canvas-wrap">
      <div id="canvas-header">
        <span>交互式匹配可视化 — 点击匹配线查看特征点细节</span>
        <label><input type="checkbox" id="cbOutliers" onchange="toggleOutliers(this.checked)"> 显示离群点</label>
        <span id="matchInfo" style="margin-left:auto;font-size:.78em;color:#8b949e"></span>
      </div>
      <canvas id="matchCanvas"></canvas>
    </div>
    <div id="detailPanel">
      <div id="detailHeader">
        <span id="detailTitle">匹配点详情</span>
        <button class="close-btn" onclick="closeDetail()">✕ 关闭</button>
      </div>
      <div id="detailBody">
        <div class="detail-crop">
          <div class="crop-title">WET (湿态)</div>
          <canvas id="cropWet" width="${CROP_SIZE}" height="${CROP_SIZE}"></canvas>
        </div>
        <div class="detail-crop">
          <div class="crop-title">DRY (干态)</div>
          <canvas id="cropDry" width="${CROP_SIZE}" height="${CROP_SIZE}"></canvas>
        </div>
      </div>
      <div id="detailInfo"></div>
    </div>`;
  }

  res.innerHTML = h;

  if (d.wet_roi && d.dry_roi) {
    initCanvas(d);
  }
}

/* ── canvas ── */
function initCanvas(d) {
  const points = d.match_points || [];
  const hi = points.filter(p => p.conf >= 0.7).length;
  const mid = points.filter(p => p.conf >= 0.4 && p.conf < 0.7).length;
  const lo = points.filter(p => p.conf < 0.4).length;
  const info = document.getElementById('matchInfo');
  if (info) info.innerHTML =
    `<span style="color:#3fb950">●</span> 高 ${hi} ` +
    `<span style="color:#e3b326">●</span> 中 ${mid} ` +
    `<span style="color:#f85149">●</span> 低 ${lo} ` +
    `/ ${points.length} 总计`;

  wetRoiImg = new Image();
  dryRoiImg = new Image();
  let loaded = 0;
  const onLoad = () => { if (++loaded === 2) setupCanvas(); };
  wetRoiImg.onload = onLoad;
  dryRoiImg.onload = onLoad;
  wetRoiImg.src = 'data:image/png;base64,' + d.wet_roi;
  dryRoiImg.src = 'data:image/png;base64,' + d.dry_roi;
}

let offWet = null, offDry = null;

function setupCanvas() {
  const canvas = document.getElementById('matchCanvas');
  if (!canvas) return;
  const wrap = document.getElementById('canvas-wrap');

  const wW = wetRoiImg.naturalWidth, wH = wetRoiImg.naturalHeight;
  const dW = dryRoiImg.naturalWidth, dH = dryRoiImg.naturalHeight;
  const totalW = wW + GAP + dW;
  const totalH = Math.max(wH, dH);

  offWet = document.createElement('canvas');
  offWet.width = wW; offWet.height = wH;
  offWet.getContext('2d').drawImage(wetRoiImg, 0, 0);

  offDry = document.createElement('canvas');
  offDry.width = dW; offDry.height = dH;
  offDry.getContext('2d').drawImage(dryRoiImg, 0, 0);

  const containerW = wrap.clientWidth;
  canvasScale = containerW / totalW;
  canvas.width = containerW;
  canvas.height = Math.ceil(totalH * canvasScale);

  drawCanvas();
  canvas.addEventListener('click', onCanvasClick);
}

function drawCanvas() {
  const canvas = document.getElementById('matchCanvas');
  if (!canvas || !offWet || !offDry) return;
  const ctx = canvas.getContext('2d');
  const s = canvasScale;
  const wW = offWet.width, wH = offWet.height;
  const dW = offDry.width, dH = offDry.height;
  const totalH = Math.max(wH, dH);
  const wetY = (totalH - wH) / 2;
  const dryY = (totalH - dH) / 2;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#0d1117';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.drawImage(offWet, 0, wetY * s, wW * s, wH * s);
  ctx.drawImage(offDry, (wW + GAP) * s, dryY * s, dW * s, dH * s);

  const points = (matchData && matchData.match_points) || [];
  const dryOffX = wW + GAP;

  ctx.lineWidth = 1;
  for (let i = 0; i < points.length; i++) {
    const pt = points[i];
    if (!pt.inlier && !showOutliers) continue;
    const isSelected = (i === selectedIdx);
    const x0 = pt.x0 * s, y0 = (pt.y0 + wetY) * s;
    const x1 = (pt.x1 + dryOffX) * s, y1 = (pt.y1 + dryY) * s;

    // 颜色: 选中=紫色, 否则按置信度 绿/黄/红
    const lineAlpha = isSelected ? '0.9' : '0.4';
    const dotAlpha = '0.9';
    const lineColor = isSelected ? `rgba(102,126,234,${lineAlpha})` : confColor(pt.conf, lineAlpha);
    const dotColor = isSelected ? `rgba(102,126,234,${dotAlpha})` : confColor(pt.conf, dotAlpha);

    // line
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = isSelected ? 2.5 : 1;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();

    // circles
    const r = isSelected ? 6 : 4;
    ctx.lineWidth = isSelected ? 2 : 1.5;
    ctx.strokeStyle = dotColor;
    if (isSelected) {
      ctx.fillStyle = `rgba(102,126,234,0.6)`;
      ctx.beginPath(); ctx.arc(x0, y0, r, 0, Math.PI*2); ctx.fill(); ctx.stroke();
      ctx.beginPath(); ctx.arc(x1, y1, r, 0, Math.PI*2); ctx.fill(); ctx.stroke();
    } else {
      ctx.beginPath(); ctx.arc(x0, y0, r, 0, Math.PI*2); ctx.stroke();
      ctx.beginPath(); ctx.arc(x1, y1, r, 0, Math.PI*2); ctx.stroke();
    }
  }
  ctx.lineWidth = 1;
}

function toggleOutliers(checked) {
  showOutliers = checked;
  requestAnimationFrame(drawCanvas);
}

/* ── click to inspect ── */
function onCanvasClick(e) {
  const canvas = document.getElementById('matchCanvas');
  const rect = canvas.getBoundingClientRect();
  const cx = e.clientX - rect.left;
  const cy = e.clientY - rect.top;
  const points = (matchData && matchData.match_points) || [];
  if (!points.length) return;

  const s = canvasScale;
  const wW = offWet.width, wH = offWet.height;
  const dH = offDry.height;
  const totalH = Math.max(wH, dH);
  const wetY = (totalH - wH) / 2;
  const dryY = (totalH - dH) / 2;
  const dryOffX = wW + GAP;

  let bestDist = Infinity, bestIdx = -1;
  for (let i = 0; i < points.length; i++) {
    const pt = points[i];
    if (!pt.inlier && !showOutliers) continue;
    const x0 = pt.x0 * s, y0 = (pt.y0 + wetY) * s;
    const x1 = (pt.x1 + dryOffX) * s, y1 = (pt.y1 + dryY) * s;
    const d = distToSeg(cx, cy, x0, y0, x1, y1);
    if (d < bestDist) { bestDist = d; bestIdx = i; }
  }

  if (bestIdx < 0 || bestDist > 15) { closeDetail(); return; }
  showDetail(points[bestIdx], bestIdx);
}

function distToSeg(px, py, x0, y0, x1, y1) {
  const dx = x1-x0, dy = y1-y0, lenSq = dx*dx+dy*dy;
  if (lenSq === 0) return Math.hypot(px-x0, py-y0);
  const t = Math.max(0, Math.min(1, ((px-x0)*dx+(py-y0)*dy)/lenSq));
  return Math.hypot(px-(x0+t*dx), py-(y0+t*dy));
}

/* ── detail panel (below canvas) ── */
function showDetail(pt, idx) {
  selectedIdx = idx;
  requestAnimationFrame(drawCanvas);

  // draw large crops with reticle
  drawLargeCrop('cropWet', offWet, pt.x0, pt.y0);
  drawLargeCrop('cropDry', offDry, pt.x1, pt.y1);

  // make crops zoomable
  document.getElementById('cropWet').onclick = () => zoomCanvas('cropWet');
  document.getElementById('cropDry').onclick = () => zoomCanvas('cropDry');

  // info
  const inlierStr = pt.inlier
    ? '<span style="color:#3fb950">✓ 内点 (Inlier)</span>'
    : '<span style="color:#f85149">✗ 离群点 (Outlier)</span>';
  document.getElementById('detailTitle').textContent = `匹配点 #${idx+1}`;
  document.getElementById('detailInfo').innerHTML =
    `置信度: <b>${pt.conf.toFixed(4)}</b> &nbsp;|&nbsp; ${inlierStr} &nbsp;|&nbsp; ` +
    `WET: (${pt.x0.toFixed(1)}, ${pt.y0.toFixed(1)}) &nbsp;→&nbsp; DRY: (${pt.x1.toFixed(1)}, ${pt.y1.toFixed(1)})`;

  const panel = document.getElementById('detailPanel');
  panel.classList.add('visible');
  panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function drawLargeCrop(canvasId, srcCanvas, cx, cy) {
  const c = document.getElementById(canvasId);
  if (!c || !srcCanvas) return;
  const ctx = c.getContext('2d');
  const size = CROP_SIZE;
  ctx.clearRect(0, 0, size, size);

  // 从原图裁切区域
  const r = CROP_RADIUS;
  const sx = Math.max(0, Math.round(cx) - r);
  const sy = Math.max(0, Math.round(cy) - r);
  const sw = Math.min(r * 2, srcCanvas.width - sx);
  const sh = Math.min(r * 2, srcCanvas.height - sy);

  // 计算在canvas上的偏移（当靠近边缘时居中显示）
  const dx = (cx - r < 0) ? (r - cx) * (size / (r*2)) : 0;
  const dy = (cy - r < 0) ? (r - cy) * (size / (r*2)) : 0;
  const dw = sw * (size / (r*2));
  const dh = sh * (size / (r*2));

  ctx.fillStyle = '#0d1117';
  ctx.fillRect(0, 0, size, size);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  if (sw > 0 && sh > 0) {
    ctx.drawImage(srcCanvas, sx, sy, sw, sh, dx, dy, dw, dh);
  }

  // 准星（中间留空的十字线）
  const mid = size / 2;
  const gap = 8;   // 中心空隙半径
  const arm = 22;  // 臂长

  ctx.strokeStyle = 'rgba(102,126,234,0.85)';
  ctx.lineWidth = 1.5;
  // 上
  ctx.beginPath(); ctx.moveTo(mid, mid - gap - arm); ctx.lineTo(mid, mid - gap); ctx.stroke();
  // 下
  ctx.beginPath(); ctx.moveTo(mid, mid + gap); ctx.lineTo(mid, mid + gap + arm); ctx.stroke();
  // 左
  ctx.beginPath(); ctx.moveTo(mid - gap - arm, mid); ctx.lineTo(mid - gap, mid); ctx.stroke();
  // 右
  ctx.beginPath(); ctx.moveTo(mid + gap, mid); ctx.lineTo(mid + gap + arm, mid); ctx.stroke();
}

function closeDetail() {
  selectedIdx = -1;
  document.getElementById('detailPanel').classList.remove('visible');
  requestAnimationFrame(drawCanvas);
}

function zoomCanvas(canvasId) {
  const c = document.getElementById(canvasId);
  if (!c) return;
  const lb = document.getElementById('lb');
  const img = document.getElementById('lbImg');
  img.src = c.toDataURL('image/png');
  lb.classList.add('active');
}

function zoom(img) {
  document.getElementById('lbImg').src = img.src;
  document.getElementById('lb').classList.add('active');
}
</script>
</body></html>'''
