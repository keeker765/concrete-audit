"""首页 — 导航到各子工具"""


def build_home_page() -> str:
    """返回首页 HTML"""
    return """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>混凝土试块审计系统</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; color: #eee;
         display: flex; justify-content: center; align-items: center; min-height: 100vh; }
  .container { max-width: 700px; width: 90%; }
  h1 { text-align: center; font-size: 2em; margin-bottom: 0.3em;
       background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text;
       -webkit-text-fill-color: transparent; }
  .subtitle { text-align: center; color: #888; margin-bottom: 2em; font-size: 0.95em; }
  .cards { display: grid; grid-template-columns: 1fr; gap: 16px; }
  .card { background: #16213e; border-radius: 12px; padding: 24px; text-decoration: none;
          color: #eee; transition: transform 0.15s, box-shadow 0.15s; border: 1px solid #0f3460; }
  .card:hover { transform: translateY(-3px); box-shadow: 0 8px 25px rgba(102,126,234,0.3);
                border-color: #667eea; }
  .card h2 { font-size: 1.3em; margin-bottom: 8px; }
  .card .icon { font-size: 1.8em; float: left; margin-right: 16px; }
  .card p { color: #aaa; font-size: 0.9em; line-height: 1.5; }
  .tag { display: inline-block; background: #667eea; color: #fff; padding: 2px 8px;
         border-radius: 4px; font-size: 0.75em; margin-left: 8px; vertical-align: middle; }
</style>
</head><body>
<div class="container">
  <h1>🔬 混凝土试块审计系统</h1>
  <p class="subtitle">Concrete Specimen Fraud Detection</p>
  <div class="cards">
    <a class="card" href="/audit">
      <span class="icon">🎯</span>
      <h2>贴纸检测审计 <span class="tag">SAM</span></h2>
      <p>交互式检测蓝色标签和混凝土面矩形。逐图点击检测，显示原图→检测结果→透视矫正对比。</p>
    </a>
    <a class="card" href="/labels">
      <span class="icon">📋</span>
      <h2>结果审核</h2>
      <p>审核 SAME/DIFFERENT/INVALID 匹配结果。按判定类型过滤，标注质量和备注。</p>
    </a>
    <a class="card" href="/pairs">
      <span class="icon">🔗</span>
      <h2>手动 QR 配对</h2>
      <p>为 QR 识别失败的图片手动指定试件编号，建立 WET↔DRY 配对关系。</p>
    </a>
    <a class="card" href="/audit_single">
      <span class="icon">📸</span>
      <h2>上传审计 <span class="tag">交互式</span></h2>
      <p>上传任意两张干湿图片，自动运行匹配并生成可交互的匹配可视化。点击连线查看特征点裁切对比。</p>
    </a>
    <a class="card" href="/inspect">
      <span class="icon">🔬</span>
      <h2>单对匹配审计</h2>
      <p>选择任意一对干湿图片，一键运行完整特征匹配流水线，查看 ROI、匹配点、分数和判定结果。</p>
    </a>
    <a class="card" href="/roi_audit">
      <span class="icon">🖼</span>
      <h2>ROI 框选审计</h2>
      <p>逐图检查边缘扫描自动框选结果。标注正确/错误/质量问题，支持在网页上手动拖拽重新框选。</p>
    </a>
  </div>
</div>
</body></html>"""


# Legacy http.server compat
def handle_home(handler):
    from web.server import _send_html
    _send_html(handler, build_home_page())
