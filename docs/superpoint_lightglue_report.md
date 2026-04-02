# SuperPoint + LightGlue 深度学习原理详解

## 1. 系统概览

**术语说明**：本报告中 `H` = 图像高度（Height，像素数），`W` = 图像宽度（Width，像素数）。例如一张 1024×768 的图，H=768, W=1024。Tensor 形状 `[B, C, H, W]` 表示 Batch × 通道数 × 高 × 宽。

本系统使用 **SuperPoint**（特征提取器）+ **LightGlue**（特征匹配器）两阶段架构，判断两张混凝土试块照片是否为同一块试块。

```
输入图像 (BGR)
    │
    ▼
┌─────────────────────────────────────────────┐
│  预处理 (pipeline 层)                        │
│  BGR → 灰度 → CLAHE 增强 → 灰度转伪RGB      │
│  → float32 tensor [3, H, W] (值域 0~1)      │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  SuperPoint (特征提取)                       │
│  输入: [1, 3, H, W] float32 tensor          │
│  输出: keypoints [M, 2]                     │
│        descriptors [M, 256]                 │
│        scores [M]                           │
└─────────────────┬───────────────────────────┘
                  │  (两张图各提取一组)
                  ▼
┌─────────────────────────────────────────────┐
│  LightGlue (特征匹配)                       │
│  输入: image0 特征 + image1 特征             │
│  输出: matches [K, 2] (匹配对索引)           │
│        scores [K] (置信度)                   │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  后处理 (pipeline 层)                        │
│  过滤贴纸区域 → RANSAC → 评分 → 判定         │
└─────────────────────────────────────────────┘
```

---

## 2. SuperPoint 网络架构

### 2.1 论文信息

> DeTone, Malisiewicz, Rabinovich. "SuperPoint: Self-Supervised Interest Point Detection and Description" (CVPRW 2018)

### 2.2 核心思想

SuperPoint 用一个 **全卷积网络 (FCN)** 同时输出两样东西：
1. **关键点热力图**：图像中哪些像素是"兴趣点"（角点、边缘交叉点等）
2. **描述符图**：每个像素的 256 维特征向量，编码局部外观信息

### 2.3 网络结构

```
输入: [1, 1, H, W] 灰度图 (实际代码中 [1,3,H,W] 会先转灰度)
                │
    ┌───────────┴───────────┐
    │   VGG 风格共享编码器    │   (4 个阶段，总下采样 8×)
    │                       │
    │  Stage 1: Conv 1→64   │   conv1a(1→64, 3×3) → ReLU
    │           Conv 64→64  │   conv1b(64→64, 3×3) → ReLU
    │           MaxPool 2×2 │   → [1, 64, H/2, W/2]
    │                       │
    │  Stage 2: Conv 64→64  │   conv2a, conv2b → ReLU
    │           MaxPool 2×2 │   → [1, 64, H/4, W/4]
    │                       │
    │  Stage 3: Conv 64→128 │   conv3a, conv3b → ReLU
    │           MaxPool 2×2 │   → [1, 128, H/4, W/4]  ← 注意论文里3次pool
    │                       │   实际实现是conv3a(64→128), 然后pool
    │  Stage 4: Conv 128→128│   conv4a, conv4b → ReLU
    │           (无 Pool)   │   → [1, 128, H/8, W/8]
    │                       │
    └───────┬───────┬───────┘
            │       │
    ┌───────┘       └───────┐
    │                       │
    ▼                       ▼
┌─────────────┐     ┌─────────────┐
│ 关键点检测头  │     │ 描述符提取头  │
│             │     │             │
│ convPa:     │     │ convDa:     │
│ 128→256,3×3 │     │ 128→256,3×3 │
│ + ReLU      │     │ + ReLU      │
│             │     │             │
│ convPb:     │     │ convDb:     │
│ 256→65,1×1  │     │ 256→256,1×1 │
│             │     │             │
│ → [1,65,    │     │ → [1,256,   │
│    H/8,W/8] │     │    H/8,W/8] │
└──────┬──────┘     └──────┬──────┘
       │                   │
       ▼                   ▼
  Softmax + 展开        L2 归一化
  → [1, H, W] 热力图    + 双线性插值采样
  → NMS + Top-K          → [M, 256]
  → [M, 2] 关键点        描述符向量
```

### 2.4 关键点检测头：为什么是 65 个通道？

这是 SuperPoint 最精妙的设计之一。

由于共享编码器下采样了 8 倍，输出特征图是 `[H/8, W/8]`。每个特征图像素对应原图的一个 **8×8 区域（cell）**。

- 前 64 个通道：表示这个 8×8 cell 内每个像素是关键点的概率
- 第 65 个通道：**dustbin（垃圾桶）**——表示"这个 cell 内没有关键点"

```
特征图一个像素 → 65 个值 → Softmax → 前 64 个值 → 展开为 8×8
                              ↓
                         第 65 个值（dustbin）丢弃

展开过程：
[1, 64, H/8, W/8] → reshape → [1, H/8, W/8, 8, 8]
                   → permute → [1, H/8, 8, W/8, 8]
                   → reshape → [1, H, W]  ← 回到原始分辨率的热力图
```

**dustbin 的作用**：重新校准 Softmax。如果一个 8×8 区域内真的没有关键点，所有 64 个概率值都应该很低——有了 dustbin，Softmax 可以把大部分概率分给 dustbin，让其他 64 个值都接近 0。

### 2.5 描述符提取头

描述符在 1/8 分辨率的特征图上是 **稠密计算** 的（每个像素都有一个 256 维向量），但最终只在关键点位置 **采样** 描述符：

```python
# 稠密描述符图
desc_dense = L2_normalize(convDb(ReLU(convDa(x))))  # [1, 256, H/8, W/8]

# 在关键点位置做双线性插值采样
for each keypoint (x, y):
    # 将 (x,y) 从原图坐标映射到特征图坐标
    fx, fy = x / 8, y / 8
    # 双线性插值采样 256 维向量
    desc = bilinear_sample(desc_dense, fx, fy)  # [256]
    desc = L2_normalize(desc)
```

**为什么用双线性插值？** 关键点坐标是整数像素，但映射到 1/8 特征图时可能落在非整数位置。双线性插值保证了亚像素级精度。

### 2.6 SuperPoint 的训练（自监督）

训练分三步，不需要人工标注关键点：

```
Step 1: MagicPoint（合成数据预训练检测器）
    - 渲染三角形、矩形、菱形等合成几何图案
    - 角点位置已知 → 用来训练关键点检测头
    - 结果：可以检测简单角点，但在真实图像上表现一般

Step 2: Homographic Adaptation（从合成到真实）
    - 对真实图像做 100 次随机单应性变换
    - 每次变换后用 MagicPoint 检测关键点
    - 把 100 次检测结果聚合 → 生成"伪 GT 关键点"
    - 用伪 GT 重新训练 → 得到在真实图像上更好的检测器

Step 3: 训练描述符
    - 对同一图像做已知的单应性变换
    - 同一物理点在变换前后应该有相同的描述符
    - 损失函数：hinge loss
      - 匹配对：cosine similarity 尽量大
      - 非匹配对：cosine similarity < 0.2
```

### 2.7 为什么 SuperPoint 不具备旋转不变性？

**关键原因：卷积核方向固定。**

卷积核检测水平/垂直边缘的方式是固定的——同一个纹理旋转 90° 后，卷积核的响应完全不同，导致描述符也不同。

传统方法（如 SIFT）通过计算每个关键点的 **主方向** 来解决这个问题：描述符相对于主方向计算，旋转后主方向跟着转，描述符不变。SuperPoint 没有这个机制。

**这就是为什么我们的管线需要 4 候选旋转搜索。**

---

## 3. LightGlue 网络架构

### 3.1 论文信息

> Lindenberger, Sarlin, Pollefeys. "LightGlue: Local Feature Matching at Light Speed" (ICCV 2023)

### 3.2 核心思想

LightGlue 是 SuperGlue 的改进版。它用 **Transformer** 接受两组关键点+描述符，通过自注意力和互注意力学习**全局上下文**，输出匹配对。

关键创新：
1. **深度自适应**：简单图像对可以在前几层就停止，不必跑完全部 9 层
2. **宽度自适应**：中途可以剪枝掉明显不可匹配的点，减少后续计算量
3. **旋转位置编码**（Rotary PE）替代 SuperGlue 的绝对位置编码

### 3.3 网络结构

```
输入: image0 的 M 个关键点+描述符, image1 的 N 个关键点+描述符

┌──────────────────────────────────────────────────────┐
│  输入投影 + 位置编码                                   │
│                                                      │
│  desc0 = Linear(256→256)(descriptors0)  # [M, 256]   │
│  desc1 = Linear(256→256)(descriptors1)  # [N, 256]   │
│                                                      │
│  pos0 = FourierPE(normalize(kpts0))  # 旋转位置编码     │
│  pos1 = FourierPE(normalize(kpts1))                   │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│  Transformer 层 ×9 (每层包含 Self + Cross Attention)  │
│                                                      │
│  ┌────────────────────────────────────────────────┐  │
│  │  Self-Attention Block                          │  │
│  │                                                │  │
│  │  image0 的描述符互相看:                          │  │
│  │  Q, K, V = Linear(desc0)  # 各 [M, 256]        │  │
│  │  Q, K = apply_rotary(Q, K, pos0)  # 注入位置    │  │
│  │  attn = softmax(Q·Kᵀ / √d)  # [M, M]          │  │
│  │  desc0 = attn · V  # 每个点聚合了其他点的信息    │  │
│  │  desc0 = FFN(desc0)  # 前馈网络: 256→512→256    │  │
│  │                                                │  │
│  │  image1 同理 (独立计算)                          │  │
│  └────────────────────────────────────────────────┘  │
│                                                      │
│  ┌────────────────────────────────────────────────┐  │
│  │  Cross-Attention Block (双向)                   │  │
│  │                                                │  │
│  │  image0 → image1:                              │  │
│  │  Q0 = Linear(desc0)  # [M, 256]               │  │
│  │  K1, V1 = Linear(desc1)  # [N, 256]            │  │
│  │  attn = softmax(Q0·K1ᵀ / √d)  # [M, N]        │  │
│  │  msg0 = attn · V1  # image0 的每个点看 image1   │  │
│  │                                                │  │
│  │  image1 → image0: (反向，对称计算)               │  │
│  │  msg1 = attn' · V0                             │  │
│  │                                                │  │
│  │  desc0 += FFN(msg0)                            │  │
│  │  desc1 += FFN(msg1)                            │  │
│  └────────────────────────────────────────────────┘  │
│                                                      │
│  可选: 早停检查 / 点剪枝                               │
└──────────────────┬───────────────────────────────────┘
                   │  (重复 9 层后，或提前停止)
                   ▼
┌──────────────────────────────────────────────────────┐
│  匹配分配 (Match Assignment)                          │
│                                                      │
│  sim = desc0 · desc1ᵀ  # [M, N] 相似度矩阵           │
│  z0 = sigmoid(Linear(desc0))  # [M, 1] 可匹配性      │
│  z1 = sigmoid(Linear(desc1))  # [N, 1] 可匹配性      │
│                                                      │
│  scores = log_double_softmax(sim, z0, z1)            │
│  # [M+1, N+1]  (+1 是 dustbin，表示"无匹配")         │
│                                                      │
│  matches = argmax + 互相一致性检查 + 阈值过滤          │
│  # 输出: [K, 2] 匹配对索引 + [K] 置信度               │
└──────────────────────────────────────────────────────┘
```

### 3.4 Self-Attention 的直觉

**问题**：一个关键点的 256 维描述符只编码了局部 ~16×16 像素的信息。但判断匹配还需要全局上下文——比如"这个角点在建筑的左上角"。

**解决**：Self-Attention 让 image0 的所有关键点互相交流。经过几层后，每个点的描述符不仅包含局部纹理信息，还包含了"我在图像中的相对位置和周围环境"。

### 3.5 Cross-Attention 的直觉

**问题**：两张图的描述符是独立提取的。如果图像有重复纹理（比如混凝土面的多个相似区域），光靠描述符相似度无法正确匹配。

**解决**：Cross-Attention 让 image0 的每个点去"看" image1 的所有点。这使得模型可以学到空间一致性——"我的上方有个蓝色贴纸，对面图像中也只有一个点的上方有蓝色贴纸，所以我们是匹配的。"

### 3.6 Dustbin（垃圾桶）机制

最终的匹配分配矩阵是 `[M+1, N+1]`，多出来的第 M+1 行和第 N+1 列是 dustbin：

```
                   image1 的 N 个关键点     dustbin₁
                 ┌─────────────────────┬──────┐
image0 的 M 个   │                     │      │
关键点           │   sim(i,j) 相似度    │ z0_i │  ← "i 不可匹配"的概率
                 │                     │      │
                 ├─────────────────────┼──────┤
dustbin₀         │      z1_j           │      │  ← "j 不可匹配"的概率
                 └─────────────────────┴──────┘
```

对每一行做 Softmax：如果 image0 的点 i 和 image1 的任何点都不像，z0_i（dustbin 概率）会很高，表示"这个点没有匹配"。

### 3.7 早停与点剪枝

**早停 (Depth Adaptivity)**：
- 每层结束后，用一个小网络预测"当前匹配结果是否足够置信"
- 如果 95% 的点都已经很置信（`depth_confidence=0.95`），提前停止
- 效果：简单图像对（大重叠、少变化）只需 3-4 层；困难图像对跑满 9 层

**点剪枝 (Width Adaptivity)**：
- 中途如果某些点明显不可匹配（比如被遮挡的区域），直接从序列中移除
- 减少后续层的注意力计算量（注意力复杂度 O(M×N)，减少 M 和 N 效果显著）

### 3.8 关键超参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `n_layers` | 9 | Transformer 层数 |
| `num_heads` | 4 | 多头注意力头数 |
| `descriptor_dim` | 256 | 内部特征维度 |
| `detection_threshold` | 0.0005 | 关键点检测阈值 |
| `nms_radius` | 4 | 非极大值抑制半径 |
| `filter_threshold` | 0.1 | 匹配置信度阈值 |
| `depth_confidence` | 0.95 | 早停置信度 |
| `width_confidence` | 0.99 | 剪枝置信度 |
| `max_num_keypoints` | 2048 | 最大关键点数（我们的配置） |

---

## 4. 完整数据流（从我们的代码追踪）

### 4.1 预处理 (`_make_tensor` in `runner.py`)

```python
def _make_tensor(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)      # [H, W] uint8
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)                            # [H, W] uint8, 对比度增强
    enh3 = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)       # [H, W, 3] uint8 (伪 RGB)
    return torch.from_numpy(enh3).permute(2, 0, 1).float() / 255.0
    # → [3, H, W] float32, 值域 [0, 1]
```

**为什么做 CLAHE？** 混凝土面灰度范围窄、对比度低。CLAHE (Contrast Limited Adaptive Histogram Equalization) 在局部区域内做直方图均衡化，让纹理细节更明显，提高关键点检测和描述符质量。

**为什么灰度转伪 RGB？** SuperPoint 内部会再转回灰度。代码中传入 3 通道是为了兼容 LightGlue 的预处理接口。

### 4.2 特征提取 (`extractor.extract`)

```python
feats = extractor.extract(tensor.to(DEVICE))
# tensor: [3, H, W] → 内部加 batch 维 → [1, 3, H, W]
# → 内部转灰度 → [1, 1, H, W]

# 输出:
feats = {
    'keypoints':      [1, M, 2],    # M 个关键点的 (x, y) 坐标
    'keypoint_scores': [1, M],       # M 个关键点的置信度
    'descriptors':    [1, M, 256],   # M 个 256 维描述符（L2 归一化）
    'image_size':     [1, 2],        # (H, W)
}
# M ≤ 2048 (MAX_KEYPOINTS)
```

### 4.3 特征匹配 (`matcher`)

```python
result = matcher({'image0': feats0, 'image1': feats1})

# 输出 (rbd 去掉 batch 维后):
res = {
    'matches':  [K, 2],   # K 对匹配, 每行 [idx_in_img0, idx_in_img1]
    'scores':   [K],      # 每对匹配的置信度 (0~1)
    'stop':     int,      # 在第几层停止 (-1 表示跑满 9 层)
    'prune0':   [M],      # 每个 img0 的点在第几层被剪枝
    'prune1':   [N],      # 每个 img1 的点在第几层被剪枝
}
```

### 4.4 贴纸区域过滤 (`filter_matches`)

```python
matches_f, scores_f = filter_matches(kpts0, kpts1, matches_raw, scores_raw,
                                      mask_wet, mask_dry)
```

对每一对匹配 (i, j)：
- 检查 kpts0[i] 是否落在 mask_wet 的贴纸区域内
- 检查 kpts1[j] 是否落在 mask_dry 的贴纸区域内
- **如果任一点在贴纸区域内 → 丢弃这对匹配**

### 4.5 RANSAC 评分 (`compute_score`)

```python
sc, cf, ir, inl, n = compute_score(kpts0, kpts1, matches_f, scores_f)
```

1. 提取匹配的关键点坐标：
   ```python
   pts0 = kpts0[matches_f[:, 0]]  # [K, 2]
   pts1 = kpts1[matches_f[:, 1]]  # [K, 2]
   ```

2. RANSAC 计算单应性矩阵：
   ```python
   H, inliers_mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)
   # H: 3×3 单应性矩阵（从 image0 到 image1 的透视变换）
   # inliers_mask: [K] 布尔数组，True = 这对匹配符合全局几何一致性
   # 5.0 = RANSAC 重投影误差阈值（像素）
   ```

3. 综合评分：
   ```python
   mean_conf = scores_f.mean()       # LightGlue 置信度均值
   inlier_ratio = inliers / K        # RANSAC 内点率
   raw_score = 0.5 * mean_conf + 0.5 * inlier_ratio

   # 匹配数惩罚
   if K < 15:  # MIN_MATCHES
       final_score = raw_score * (K / 15)   # 线性惩罚
   else:
       final_score = raw_score
   ```

### 4.6 判定

```
final_score ≥ 0.60  →  SAME（同一试块）
final_score < 0.60  →  DIFFERENT（疑似调包）
filtered_matches < 4 → INSUFFICIENT（证据不足）
无贴纸/无QR         → INVALID（无法处理）
```

---

## 5. 完整 Tensor 形状追踪表

> **符号说明**：`B` = batch size（批次大小，通常为 1），`H` = 图像高度，`W` = 图像宽度，`M` = image0 检测到的关键点数，`N` = image1 检测到的关键点数，`K` = 匹配对数

| 阶段 | 操作 | 输入形状 | 输出形状 | 说明 |
|------|------|---------|---------|------|
| **预处理** | BGR→灰度→CLAHE→伪RGB | [H,W,3] uint8 | [3,H,W] float32 | 值域 0~1 |
| **SuperPoint 编码器** | 4 阶段 Conv+Pool | [1,3,H,W] | [1,128,H/8,W/8] | 总步幅 8× |
| **关键点头** | Conv+Softmax+展开 | [1,128,H/8,W/8] | [1,65,H/8,W/8]→[1,H,W] | 65=8×8+dustbin |
| **NMS + Top-K** | 非极大值抑制 | [1,H,W] 热力图 | [1,M,2] 坐标 | M≤2048 |
| **描述符头** | Conv+L2Norm | [1,128,H/8,W/8] | [1,256,H/8,W/8] | 稠密描述符图 |
| **描述符采样** | 双线性插值 | [1,256,H/8,W/8]+[1,M,2] | [1,M,256] | 在关键点位置采样 |
| **LightGlue 位置编码** | Fourier PE | [1,M,2] 归一化坐标 | [1,M,4,2] 旋转嵌入 | cos/sin 对 |
| **Self-Attention ×9** | QKV+Softmax | [1,M,256] | [1,M,256] | 4 头，64 维/头 |
| **Cross-Attention ×9** | 双向注意力 | [1,M,256]+[1,N,256] | [1,M,256]+[1,N,256] | M→N, N→M |
| **匹配分配** | 相似度+双 Softmax | [1,M,256]+[1,N,256] | [1,M+1,N+1] | +1 是 dustbin |
| **匹配过滤** | argmax+阈值 | [1,M+1,N+1] | [K,2]+[K] | K 对匹配+置信度 |
| **贴纸过滤** | mask 检查 | [K,2] + masks | [K',2] | K'≤K |
| **RANSAC** | 单应性估计 | [K',2]×2 | score+inliers | 重投影阈值 5px |

---

## 6. 为什么这套方法适合混凝土试块匹配

### 优势

1. **SuperPoint 的自监督训练**使其对角点和纹理变化有很好的检测能力——混凝土面有丰富的随机纹理（气孔、砂粒、裂纹）
2. **LightGlue 的全局上下文推理**可以处理重复纹理——通过 Self-Attention 编码空间关系，Cross-Attention 建立跨图对应
3. **RANSAC 后验证**有效过滤误匹配——即使有一些错误匹配，只要大多数匹配几何一致，分数就会很高
4. **跨批次区分度好**——不同试块的随机纹理不同，SP+LG 不会产生虚假高分（vs RoMa 的 DINOv2 语义特征会认为"所有混凝土长得差不多"）

### 局限

1. **不具备旋转不变性** → 需要 4 候选旋转搜索
2. **对干湿外观差异敏感** → CLAHE 增强 + ROI 裁切可以缓解，但宁波大目/象山港浦仍然失败
3. **关键点数量受限** → MAX_KEYPOINTS=2048，在大图上可能不够密集
4. **对预处理质量依赖强** → SAM 分割不准确或透视矫正失败会直接影响匹配结果
