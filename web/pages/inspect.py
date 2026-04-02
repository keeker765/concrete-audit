"""
/inspect — 单对图片匹配审计页
选择一对湿/干图 → 选择方法 → 一键跑完整流水线 → 显示所有中间结果
"""
import os
import json
import base64
import cv2
import numpy as np


def _b64(img_bgr, quality=85):
    if img_bgr is None:
        return ''
    ok, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode() if ok else ''


def build_inspect_page(page: int = 1, search: str = '') -> str:
    """Build the inspect page HTML. Returns HTML string."""
    import label_tool

    search = search.lower()
    per_page = 20

    pairs = label_tool.get_pairs()
    if pairs is None:
        return '''<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="4;url=/inspect">
<style>body{background:#1a1a2e;color:#eee;display:flex;align-items:center;
justify-content:center;height:100vh;font-family:sans-serif;flex-direction:column;gap:16px}
.sp{width:48px;height:48px;border:4px solid #333;border-top:4px solid #667eea;
border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}</style></head>
<body><div class="sp"></div><p>⏳ 正在加载配对数据…</p>
<a href="/" style="color:#667eea">← 首页</a></body></html>'''

    if search:
        pairs = [p for p in pairs if search in p['batch'].lower()
                 or search in str(p['specimen'])]

    total = len(pairs)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))
    page_pairs = pairs[(page - 1) * per_page: page * per_page]

    rows = ''
    for p in page_pairs:
        batch = p['batch']
        spec = p['specimen']
        wname = p.get('wet_name', '?')
        dname = p.get('dry_name', '?')
        wet_path = p.get('wet_path', '')
        dry_path = p.get('dry_path', '')
        wet_esc = wet_path.replace('\\', '\\\\').replace("'", "\\'")
        dry_esc = dry_path.replace('\\', '\\\\').replace("'", "\\'")
        rows += f'''
        <tr onclick="selectPair('{batch}',{spec},'{wet_esc}','{dry_esc}')"
            id="row-{batch}-{spec}" class="pair-row">
          <td>{batch}</td>
          <td style="text-align:center">#{spec}</td>
          <td style="color:#aaa;font-size:.85em">{wname}</td>
          <td style="color:#aaa;font-size:.85em">{dname}</td>
        </tr>'''

    def plink(pg, label):
        q_part = f'&q={search}' if search else ''
        return f'<a href="/inspect?page={pg}{q_part}" style="color:#667eea;margin:0 4px">{label}</a>'

    pager = ''
    if total_pages > 1:
        if page > 1:
            pager += plink(1, '«') + plink(page - 1, '‹')
        pager += f' <span style="color:#eee">{page}/{total_pages}</span> '
        if page < total_pages:
            pager += plink(page + 1, '›') + plink(total_pages, '»')

    return _build_inspect_html(rows, pager, search, total)


def run_inspect(wet_path: str, dry_path: str, method: str = 'sp') -> dict:
    """Run the matching pipeline and return a JSON-serializable result dict."""
    import pipeline.runner as runner

    if not wet_path or not dry_path:
        return {'error': '缺少 wet_path 或 dry_path'}

    result = runner.run_single_pair(wet_path, dry_path, method=method)

    return {
        'verdict':    result['verdict'],
        'score':      float(result['score']),
        'n_raw':      result['n_raw'],
        'n_filtered': result['n_filtered'],
        'n_inliers':  result['n_inliers'],
        'rot_deg':    float(result['rot_deg']),
        'elapsed':    float(result['elapsed']),
        'error':      result['error'],
        'wet_previs':  _b64(result.get('wet_previs')),
        'dry_previs':  _b64(result.get('dry_previs')),
        'wet_roi':    _b64(result.get('wet_roi')),
        'dry_roi':    _b64(result.get('dry_roi')),
        'wet_aligned': _b64(result.get('wet_aligned')),
        'dry_aligned': _b64(result.get('dry_aligned')),
        'match_vis':  _b64(result.get('match_vis')),
    }


# ── Legacy http.server compat ────────────────────────────────────

def _serve_page(handler, qs):
    from web.server import _send_html
    page = int(qs.get('page', ['1'])[0])
    search = qs.get('q', [''])[0]
    html = build_inspect_page(page, search)
    _send_html(handler, html)


def handle_inspect_post(handler, path):
    from web.server import _send_json

    length = int(handler.headers.get('Content-Length', 0))
    body = json.loads(handler.rfile.read(length))
    resp = run_inspect(body.get('wet_path', ''), body.get('dry_path', ''),
                       body.get('method', 'sp'))
    _send_json(handler, resp)


def _build_inspect_html(rows, pager, search, total):
    return f'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>配对匹配审计</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d1117;color:#e6edf3;font-family:system-ui,sans-serif;display:flex;height:100vh;overflow:hidden}}
#sidebar{{width:420px;min-width:320px;display:flex;flex-direction:column;border-right:1px solid #30363d;background:#161b22}}
#main{{flex:1;display:flex;flex-direction:column;overflow:auto;padding:16px}}
.topbar{{padding:10px 16px;display:flex;align-items:center;gap:10px;background:#161b22;border-bottom:1px solid #30363d}}
.topbar a{{color:#667eea;text-decoration:none;font-size:.9em}}
.topbar h2{{font-size:1em;color:#c9d1d9}}
#search{{flex:1;background:#0d1117;border:1px solid #30363d;color:#e6edf3;padding:6px 10px;border-radius:6px;font-size:.9em}}
.pair-table{{width:100%;border-collapse:collapse;font-size:.85em}}
.pair-table th{{background:#21262d;padding:8px 10px;text-align:left;color:#8b949e;font-weight:500;position:sticky;top:0}}
.pair-row{{cursor:pointer;border-bottom:1px solid #21262d}}
.pair-row:hover{{background:#1c2128}}
.pair-row.selected{{background:#1f3a6e}}
.pair-row td{{padding:7px 10px}}
#table-wrap{{flex:1;overflow-y:auto}}
#pager{{padding:8px 16px;font-size:.85em;text-align:center;border-top:1px solid #30363d}}
.controls{{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:16px}}
.controls label{{color:#8b949e;font-size:.9em}}
select,button{{background:#21262d;color:#e6edf3;border:1px solid #30363d;padding:6px 14px;border-radius:6px;cursor:pointer;font-size:.9em}}
button.run{{background:#1f6feb;border-color:#1f6feb;font-weight:600}}
button.run:hover{{background:#388bfd}}
button:disabled{{opacity:.5;cursor:default}}
#status{{font-size:.85em;color:#8b949e}}
.verdict-SAME{{color:#3fb950;font-weight:700}}
.verdict-DIFFERENT{{color:#f85149;font-weight:700}}
.verdict-INSUFFICIENT{{color:#d29922;font-weight:700}}
.verdict-INVALID{{color:#8b949e;font-weight:700}}
.score-box{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px 20px;display:flex;gap:24px;flex-wrap:wrap;margin-bottom:16px}}
.metric{{display:flex;flex-direction:column;gap:2px}}
.metric .label{{font-size:.75em;color:#8b949e}}
.metric .val{{font-size:1.1em;font-weight:600}}
.img-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:12px}}
.img-card{{background:#161b22;border:1px solid #30363d;border-radius:8px;overflow:hidden}}
.img-card .card-title{{padding:8px 12px;font-size:.85em;color:#8b949e;background:#21262d}}
.img-card img{{width:100%;display:block;cursor:zoom-in}}
.lightbox{{display:none;position:fixed;inset:0;background:rgba(0,0,0,.92);z-index:9999;align-items:center;justify-content:center}}
.lightbox.active{{display:flex}}
.lightbox img{{max-width:95vw;max-height:90vh;object-fit:contain}}
.error-box{{background:#2d1b1b;border:1px solid #f85149;border-radius:8px;padding:14px;
           color:#f85149;font-family:monospace;font-size:.82em;white-space:pre-wrap;word-break:break-all}}
</style>
</head><body>

<div id="sidebar">
  <div class="topbar">
    <a href="/">← 首页</a>
    <h2>配对匹配审计</h2>
  </div>
  <div style="padding:10px 16px">
    <form method="get" action="/inspect" style="display:flex;gap:8px">
      <input id="search" name="q" placeholder="搜索批次/编号…" value="{search}">
      <button type="submit">搜</button>
    </form>
    <div style="padding:4px 0;font-size:.8em;color:#8b949e">{total} 对</div>
  </div>
  <div id="table-wrap">
    <table class="pair-table">
      <thead><tr>
        <th>批次</th><th>#</th><th>湿态</th><th>干态</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
  <div id="pager">{pager}</div>
</div>

<div id="main">
  <div class="controls">
    <label>方法：</label>
    <select id="method">
      <option value="sp">SuperPoint + LightGlue</option>
      <option value="aliked">ALIKED + LightGlue</option>
      <option value="sift">SIFT + LightGlue</option>
      <option value="hardnet">DoGHardNet + LightGlue</option>
    </select>
    <button class="run" id="runBtn" onclick="runMatch()" disabled>▶ 运行匹配</button>
    <span id="status">← 先从左侧选择一对图片</span>
  </div>
  <div id="results"></div>
</div>

<div class="lightbox" id="lb" onclick="this.classList.remove('active')">
  <img id="lbImg" src="">
</div>

<script>
let _wet='', _dry='';

function selectPair(batch, spec, wet, dry) {{
  document.querySelectorAll('.pair-row').forEach(r=>r.classList.remove('selected'));
  const row = document.getElementById('row-'+batch+'-'+spec);
  if(row) row.classList.add('selected');
  _wet = wet; _dry = dry;
  document.getElementById('runBtn').disabled = !wet || !dry;
  document.getElementById('status').textContent =
    wet && dry ? `已选: ${{batch}} #${{spec}}` : '干/湿路径缺失，无法运行';
}}

async function runMatch() {{
  const method = document.getElementById('method').value;
  const btn = document.getElementById('runBtn');
  const status = document.getElementById('status');
  btn.disabled = true;
  status.textContent = '⏳ 运行中（可能需要 10-30 秒）…';
  document.getElementById('results').innerHTML = '';
  try {{
    const resp = await fetch('/inspect/run', {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{wet_path:_wet, dry_path:_dry, method}})
    }});
    const data = await resp.json();
    renderResults(data);
    status.textContent = `完成 (${{data.elapsed?.toFixed(1)}}s)`;
  }} catch(e) {{
    status.textContent = '❌ 网络错误: ' + e;
  }}
  btn.disabled = false;
}}

function renderResults(d) {{
  const res = document.getElementById('results');
  if(d.error) {{
    res.innerHTML = `<div class="error-box"><b>错误</b>\n${{d.error}}</div>`;
    return;
  }}
  const vclass = 'verdict-' + d.verdict;
  const score_pct = ((d.score||0)*100).toFixed(1);
  const bar_color = d.verdict==='SAME'?'#3fb950':d.verdict==='DIFFERENT'?'#f85149':'#d29922';
  const bar_w = Math.min(100, (d.score||0)*100);

  let scoreHtml = `
    <div class="score-box">
      <div class="metric"><span class="label">判定</span>
        <span class="val ${{vclass}}">${{d.verdict}}</span></div>
      <div class="metric"><span class="label">分数</span>
        <span class="val">${{(d.score||0).toFixed(3)}}</span></div>
      <div class="metric"><span class="label">匹配点</span>
        <span class="val">${{d.n_filtered}} / ${{d.n_raw}}</span></div>
      <div class="metric"><span class="label">内点</span>
        <span class="val">${{d.n_inliers}}</span></div>
      <div class="metric"><span class="label">旋转</span>
        <span class="val">${{(d.rot_deg||0).toFixed(1)}}°</span></div>
      <div class="metric"><span class="label">耗时</span>
        <span class="val">${{(d.elapsed||0).toFixed(1)}}s</span></div>
    </div>
    <div style="height:8px;background:#21262d;border-radius:4px;margin-bottom:16px;overflow:hidden">
      <div style="width:${{bar_w}}%;height:100%;background:${{bar_color}};border-radius:4px;transition:width .4s"></div>
    </div>`;

  const imgs = [
    ['WET 预处理（SAM分割+ROI框+贴纸）', d.wet_previs],
    ['DRY 预处理（SAM分割+ROI框+贴纸）', d.dry_previs],
    ['WET ROI（对齐+旋转后）', d.wet_roi],
    ['DRY ROI', d.dry_roi],
    ['匹配可视化', d.match_vis],
  ].filter(x=>x[1]);

  const imgHtml = imgs.map(([title, b64]) => `
    <div class="img-card">
      <div class="card-title">${{title}}</div>
      <img src="data:image/jpeg;base64,${{b64}}" onclick="zoom(this)">
    </div>`).join('');

  res.innerHTML = scoreHtml + `<div class="img-grid">${{imgHtml}}</div>`;
}}

function zoom(img) {{
  document.getElementById('lbImg').src = img.src;
  document.getElementById('lb').classList.add('active');
}}
</script>
</body></html>'''
