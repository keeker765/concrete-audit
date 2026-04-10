# 混凝土试块造假识别 — 部署教程

> 本教程仅覆盖 **SuperPoint + LightGlue** 方案（主流程 `python main.py --methods sp`）。

## 目录

- [环境要求](#环境要求)
- [Windows 部署](#windows-部署)
- [macOS 部署](#macos-部署)
- [运行](#运行)
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

Apple Silicon 会自动启用 MPS 加速，代码中已适配：

```python
DEVICE = 'cuda' if cuda_available else ('mps' if mps_available else 'cpu')
```

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
├── data/          ← 全量数据（按供应商/批次组织）
├── samples/       ← 测试样本
├── output_v2/     ← 输出结果（自动创建）
├── pipeline/
├── main.py
└── requirements.txt
```

---

## 运行

```bash
# 用 samples/ 快速测试
python main.py --methods sp

# 只跑前 3 对
python main.py --limit 3 --methods sp

# 全量数据
python main.py --data --methods sp

# 跑指定批次
python main.py --data --batch 250059 --methods sp

# 启动 Web 审计界面（端口 8765）
cd web && uvicorn app:app --port 8765 --reload
```

---

## 常见问题

### Q: Mac 上 `pyzbar` 安装报错

```bash
brew install zbar
pip install pyzbar
```

### Q: `lightglue` 安装失败

```bash
pip install git+https://github.com/cvg/LightGlue.git
```

### Q: MPS 不可用（Apple Silicon Mac）

```bash
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

确认 macOS ≥ 12 且 PyTorch ≥ 2.0。

