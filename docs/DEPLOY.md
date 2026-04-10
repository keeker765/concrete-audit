# 混凝土试块造假识别 — 部署教程

## 目录

- [环境要求](#环境要求)
- [Windows 部署](#windows-部署)
- [macOS 部署](#macos-部署)
- [SAM 模型下载](#sam-模型下载)
- [运行](#运行)
- [常见问题](#常见问题)

---

## 环境要求

| 项目 | 最低版本 |
|------|---------|
| Python | 3.10+ |
| CUDA（Windows GPU）| 11.8+ |
| macOS | 12 Monterey+（Apple Silicon 支持 MPS 加速）|
| 磁盘空间 | ≥ 5 GB（模型文件 + 数据）|

---

## Windows 部署

### 1. 安装 Python

从 [python.org](https://www.python.org/downloads/) 下载 Python 3.10+，安装时勾选 **Add Python to PATH**。

### 2. 克隆 / 解压项目

```bat
cd D:\your\projects
git clone <repo_url> concrete_audit
cd concrete_audit
```

### 3. 创建虚拟环境

```bat
python -m venv .venv
.venv\Scripts\activate
```

### 4. 安装 PyTorch（NVIDIA GPU）

前往 [pytorch.org](https://pytorch.org/get-started/locally/) 选择对应 CUDA 版本，例如 CUDA 12.1：

```bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

如果没有 GPU，安装 CPU 版：

```bat
pip install torch torchvision
```

### 5. 安装其余依赖

```bat
pip install -r requirements.txt
```

> **注意**：`requirements.txt` 里的 `segment-anything` 行需要 git 可用。  
> 如果报错，可手动运行：  
> `pip install git+https://github.com/facebookresearch/segment-anything.git`

### 6. 下载 SAM 模型（见下方）

---

## macOS 部署

### 1. 安装 Homebrew（如未安装）

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. 安装 Python 3.10+

```bash
brew install python@3.11
```

### 3. 克隆 / 解压项目

```bash
cd ~/projects
git clone <repo_url> concrete_audit
cd concrete_audit
```

### 4. 创建虚拟环境

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 5. 安装 PyTorch（Apple Silicon MPS 加速）

```bash
pip install torch torchvision
```

PyTorch 会自动检测 Apple Silicon 并启用 MPS 加速。代码中已适配：

```python
# 优先级: CUDA > MPS > CPU
DEVICE = 'cuda' if torch.cuda.is_available() else \
         ('mps' if torch.backends.mps.is_available() else 'cpu')
```

### 6. 安装 zbar（pyzbar 的系统依赖）

```bash
brew install zbar
```

### 7. 安装其余依赖

```bash
pip install -r requirements.txt
```

### 8. 下载 SAM 模型（见下方）

---

## SAM 模型下载

在项目根目录创建 `models/` 文件夹，下载并放入 SAM 模型文件：

```bash
# 创建目录
mkdir -p models

# 下载（约 375 MB）
curl -L -o models/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

Windows PowerShell：

```powershell
New-Item -ItemType Directory -Force -Path models
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" `
  -OutFile "models\sam_vit_b_01ec64.pth"
```

> 也可通过环境变量指定模型路径（放在项目外部也没问题）：
> ```bash
> export SAM_CHECKPOINT=/path/to/sam_vit_b_01ec64.pth   # Mac/Linux
> set SAM_CHECKPOINT=C:\your\path\sam_vit_b_01ec64.pth  # Windows
> ```

### 项目结构

```
concrete_audit/
├── models/
│   └── sam_vit_b_01ec64.pth   ← SAM 模型（需手动下载）
├── data/                       ← 全量数据（按供应商/批次组织）
├── samples/                    ← 测试样本
├── output_v2/                  ← 输出结果（自动创建）
├── pipeline/
├── main.py
└── requirements.txt
```

---

## 运行

激活虚拟环境后：

```bash
# 用 samples/ 测试（默认，速度快）
python main.py

# 只跑前 3 对，只用 SuperPoint 方法
python main.py --limit 3 --methods sp

# 全量数据
python main.py --data

# 跑指定批次
python main.py --data --batch 250059

# 启动 Web 审计界面（端口 8765）
cd web && uvicorn app:app --port 8765 --reload

# MatchAnything 测试
python test_matchanything.py --limit 3
```

### 可选方法（--methods 参数）

| 值 | 模型 | 说明 |
|----|------|------|
| `sp` | SuperPoint + LightGlue | 默认，速度快 |
| `aliked` | ALIKED + LightGlue | 效果好 |
| `sift` | SIFT + LightGlue | 传统特征 |
| `hardnet` | DoGHardNet + LightGlue | 鲁棒性强 |
| `loftr` | LoFTR | 稠密匹配 |
| `roma` | RoMa | 稠密 warp 匹配 |
| `all` | 全部 | 对比测试 |

---

## 常见问题

### Q: Mac 上 `pyzbar` 安装失败

```bash
brew install zbar
pip install pyzbar
```

### Q: `segment_anything` 安装失败（网络问题）

手动安装：

```bash
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
cd ..
```

### Q: `romatch` 安装失败

```bash
pip install git+https://github.com/Parskatt/RoMa.git
```

### Q: Mac 上 `torch.backends.mps.is_available()` 返回 False

- 确认 macOS ≥ 12 且芯片为 Apple Silicon（M1/M2/M3）
- 确认 PyTorch ≥ 2.0：`python -c "import torch; print(torch.__version__)"`

### Q: `models/sam_vit_b_01ec64.pth` 文件不存在

只有调用 SAM 进行 ROI 分割时才会加载模型（`roi_mode='face'` 或 `'dino'`）。  
如果只用贴纸/正方形 ROI 模式，不需要 SAM 模型。
