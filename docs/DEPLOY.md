# 混凝土试块造假识别 — 部署 & API 文档

> 本文档覆盖 **SuperPoint + LightGlue** 方案的完整部署和 API 说明。

## 目录

- [环境要求](#环境要求)
- [Windows 部署](#windows-部署)
- [macOS 部署](#macos-部署)
- [项目结构](#项目结构)
- [运行方式](#运行方式)
- [API 文档](#api-文档)
- [常见问题](#常见问题)

---

## 环境要求

| 项目 | 最低版本 |
|------|---------|
| Python | 3.10+ |
| CUDA（Windows GPU 可选）| 11.8+ |
| macOS | 12 Monterey+（Apple Silicon 支持 MPS 加速）|

---

## Windows 部署

### 1. 安装 Python

从 [python.org](https://www.python.org/downloads/) 下载 Python 3.10+，安装时勾选 **Add Python to PATH**。

### 2. 获取项目

```bat
git clone <repo_url> concrete_audit
cd concrete_audit
```

### 3. 创建虚拟环境

```bat
python -m venv .venv
.venv\Scripts\activate
```

### 4. 安装 PyTorch

**有 NVIDIA GPU（推荐，CUDA 12.1）：**

```bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**无 GPU / CPU 模式：**

```bat
pip install torch torchvision
```

### 5. 安装其余依赖

```bat
pip install -r requirements.txt
```

---

## macOS 部署

### 1. 安装 Homebrew（如未安装）

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. 安装 Python 3.11

```bash
brew install python@3.11
```

### 3. 获取项目

```bash
git clone <repo_url> concrete_audit
cd concrete_audit
```

### 4. 创建虚拟环境

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 5. 安装 PyTorch（自动适配 MPS）

```bash
pip install torch torchvision
```

Apple Silicon 会自动启用 MPS 加速（代码中优先级：CUDA > MPS > CPU）。  
验证 MPS：`python -c "import torch; print(torch.backends.mps.is_available())"`

### 6. 安装 zbar（QR 识别系统依赖）

```bash
brew install zbar
```

### 7. 安装其余依赖

```bash
pip install -r requirements.txt
```

---

## 项目结构

```
concrete_audit/
├── api/
│   └── main.py        ← 独立 JSON API 服务（端口 8080）
├── data/              ← 全量数据（按供应商/批次组织）
├── samples/           ← 测试样本
├── output_v2/         ← 输出结果（自动创建）
├── pipeline/          ← 核心流水线模块
├── web/               ← 完整 Web 审计系统（端口 8765）
├── main.py            ← 批量命令行入口
└── requirements.txt
```

---

## 运行方式

### 独立 API 服务（推荐）

```bash
# 项目根目录下启动
python api/main.py

# 或 uvicorn 热重载
uvicorn api.main:app --port 8080 --reload
```

打开浏览器 **http://localhost:8080** → 拖拽上传湿/干照片 → 查看交互式匹配可视化。

### 命令行批量处理

```bash
# 用 samples/ 快速测试
python main.py --methods sp

# 只跑前 3 对
python main.py --limit 3 --methods sp

# 全量数据
python main.py --data --methods sp

# 跑指定批次
python main.py --data --batch 250059 --methods sp
```

### 完整 Web 审计系统

```bash
cd web && uvicorn app:app --port 8765 --reload
```

---

## API 文档

### 基础信息

| 项目 | 内容 |
|------|------|
| 默认端口 | 8080 |
| 接口风格 | REST + multipart/form-data |
| 交互文档 | http://localhost:8080/docs（Swagger UI）|
| 首次启动 | 自动下载 SP+LG 模型权重（~50 MB，需联网）|

---

### GET /health — 健康检查

```http
GET /health
```

**响应示例**

```json
{
  "status":          "ok",
  "device":          "cuda",
  "models_ready":    true,
  "match_threshold": 0.46,
  "min_matches":     20
}
```

| 字段 | 说明 |
|------|------|
| `device` | 推理设备：cuda / mps / cpu |
| `models_ready` | SP+LG 权重是否已加载完毕 |

---

### POST /match — 鉴定一对试块

```http
POST /match
Content-Type: multipart/form-data
```

**请求字段**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `wet` | file | ✓ | 湿态照片（制作时拍摄）|
| `dry` | file | ✓ | 干态照片（送检时拍摄）|
| `method` | string | — | 匹配方法，默认 `sp`，可选 `aliked` / `sift` / `hardnet` |

**curl 示例**

```bash
curl -X POST http://localhost:8080/match \
  -F "wet=@wet_photo.jpg" \
  -F "dry=@dry_photo.jpg"
```

**Python requests 示例**

```python
import requests

resp = requests.post(
    "http://localhost:8080/match",
    files={
        "wet": open("wet.jpg", "rb"),
        "dry": open("dry.jpg", "rb"),
    },
    data={"method": "sp"},
)
result = resp.json()
print(result["verdict"], result["score"])
```

**响应字段**

| 字段 | 类型 | 说明 |
|------|------|------|
| `verdict` | string | `SAME` / `DIFFERENT` / `INSUFFICIENT` / `INVALID` |
| `score` | float | 综合评分 [0, 1]，≥ threshold 判 SAME |
| `n_raw` | int | 原始匹配点数 |
| `n_filtered` | int | 过滤贴纸区域后的有效匹配点数 |
| `n_inliers` | int | RANSAC 几何校验内点数 |
| `rot_deg` | float | 推断的旋转角度（°）|
| `mean_conf` | float | 平均匹配置信度 |
| `inlier_ratio` | float | RANSAC 内点率 |
| `elapsed` | float | 推理耗时（秒）|
| `error` | string / null | 错误信息（正常为 null）|
| `wet_previs` | string | base64 JPEG — 湿态预处理中间图 |
| `dry_previs` | string | base64 JPEG — 干态预处理中间图 |
| `wet_roi` | string | base64 PNG — 湿态 ROI 裁切图 |
| `dry_roi` | string | base64 PNG — 干态 ROI 裁切图 |
| `match_vis` | string | base64 JPEG — 匹配可视化叠加图 |
| `match_points` | array | 每个匹配点的坐标和置信度（见下）|

**match_points 元素结构**

```json
{
  "x0": 123.4,    // 湿态图 x 坐标（像素）
  "y0": 56.7,     // 湿态图 y 坐标
  "x1": 234.5,    // 干态图 x 坐标
  "y1": 67.8,     // 干态图 y 坐标
  "inlier": true, // 是否为 RANSAC 内点
  "conf": 0.92    // 匹配置信度 [0, 1]
}
```

**verdict 说明**

| 值 | 含义 |
|----|------|
| `SAME` | 判定为同一试块（真实，score ≥ threshold）|
| `DIFFERENT` | 判定为不同试块（疑似造假，score < threshold）|
| `INSUFFICIENT` | 有效匹配点不足，无法判断 |
| `INVALID` | 贴纸或 QR 码检测失败，流水线无法处理 |

---

### POST /audit_single/run — 审计界面专用

与 `/match` 完全等价，唯一区别是上传字段名为 `wet_file` / `dry_file`，与内置 HTML 页面的表单对应：

```bash
curl -X POST http://localhost:8080/audit_single/run \
  -F "wet_file=@wet.jpg" \
  -F "dry_file=@dry.jpg" \
  -F "method=sp"
```

---

## 常见问题

### Q: Mac 上 `pyzbar` 安装报错

```bash
brew install zbar
pip install pyzbar
```

### Q: `lightglue` 安装失败（网络超时）

```bash
pip install git+https://github.com/cvg/LightGlue.git
```

### Q: MPS 不可用（Apple Silicon Mac）

需要 macOS ≥ 12 且 PyTorch ≥ 2.0（标准 pip 安装即可，无需 nightly）：

```bash
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

### Q: 启动报 `models_ready: false`，一直无法请求

模型在后台线程加载，通常 10~30 秒内完成。可以先调 `/health` 确认 `models_ready: true` 后再发送 `/match` 请求；或者不等，代码会同步等待模型就绪再处理请求。


