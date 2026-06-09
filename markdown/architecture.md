# 架构：草莓点云到 DeepSDF 网格

本文档只描述模型架构本身，即论文中“Method / Architecture”应包含的内容，不涉及脚本入口、目录组织、运行命令、日志路径或实验管理细节。

## 1. 任务定义

目标是从草莓点云预测其对应的形状隐变量，再通过隐式表面解码器恢复连续 SDF 场，并提取三角网格。

整体推理路径为：

```text
输入点云 P
  -> PointNeXt encoder
  -> latent code z
  -> DeepSDF decoder
  -> continuous SDF field f(z, x)
  -> marching cubes
  -> reconstructed mesh
```

其中：

- `P = {p_i}_{i=1}^N` 表示输入点云，`p_i \in R^3`。
- `z \in R^32` 表示形状隐变量。
- `f(z, x)` 表示查询点 `x \in R^3` 处的符号距离值。

## 2. 总体框架

整个方法采用两阶段架构：

1. 先训练一个 DeepSDF decoder，建立“latent code -> 连续 SDF 场”的生成映射。
2. 再训练一个点云编码器，将输入点云直接回归到 DeepSDF latent space。

因此，最终系统可以写成：

```text
z = E(P)
s = F(z, x)
M = MC(F(z, ·))
```

其中：

- `E` 是点云编码器；
- `F` 是 DeepSDF 解码器；
- `MC` 表示 marching cubes 网格提取。

该设计的关键点是：编码器不直接预测体素、点集或网格，而是预测一个可被隐式解码器解释的低维形状表示 `z`。

## 3. 点云编码器

### 3.1 输入与输出

编码器输入为点云 `P \in R^{N x 3}`，当前使用固定点数 `N = 2048`。点云经过中心对齐后送入网络，输出一个 32 维 latent code：

```text
E: R^(2048 x 3) -> R^32
```

### 3.2 主干结构

编码器采用 PointNeXt 风格层级式点云网络，由一个 stem、三个 set abstraction stage、逐层残差特征变换模块以及仅使用最后一级特征的全局聚合头组成。

当前默认配置下，结构概要如下：

```text
Input XYZ, N=2048
  -> Stem Shared MLP, 3 -> 48 -> 48
  -> ResidualMLP1d(48)
  -> SA1, npoint=384, k=24, 48  -> 96
  -> Stage1: InvResMLP + ResidualMLP1d
  -> SA2, npoint=96,  k=24, 96  -> 192
  -> Stage2: InvResMLP + ResidualMLP1d
  -> SA3, npoint=24,  k=24, 192 -> 384
  -> Stage3: InvResMLP + ResidualMLP1d
  -> Stage3-only global descriptor: 384
  -> Global Max/Avg Pool, 384 -> 768
  -> Head MLP, 768 -> 512 -> 256 -> 32
```

### 3.3 Stem

输入点坐标首先通过共享 MLP 进行逐点特征提取：

```text
[B, 3, 2048]
  -> SharedMLP1d(3  -> 48)
  -> SharedMLP1d(48 -> 48)
  -> ResidualMLP1d(48)
  -> [B, 48, 2048]
```

这里的共享 MLP 本质上是 `1 x 1` 卷积加归一化与激活函数，用于把原始几何坐标映射到初始点特征空间。

### 3.4 Set Abstraction 层

每个 set abstraction 模块由三部分组成：

1. 使用 FPS 选择一组中心点。
2. 对每个中心点使用 KNN 收集局部邻域。
3. 对局部相对坐标与邻域特征做共享 MLP 聚合，再通过池化得到中心点特征。

其计算流程可以写为：

```text
输入:
  xyz:      [B, N, 3]
  features: [B, C, N]

1. farthest point sampling -> 中心点
2. k-nearest neighbors     -> 局部邻域
3. grouped_xyz - center_xyz
4. concat(relative_xyz, grouped_features)
5. SharedMLP2d
6. neighborhood max pooling
7. 与中心点 skip 分支相加
```

这种设计同时保留了：

- 稀疏采样后的全局覆盖性；
- 局部邻域内的几何关系；
- 中心点特征的残差传递。

### 3.5 Stage 1

第一层级将点数从 2048 下采样到 384，并把通道数从 48 提升到 96：

```text
SA1:
  npoint = 384
  nsample = 24
  input channels = 48 + 3
  output channels = 96
```

随后接一个 `Stage` 模块。当前默认 `stage_depth = 1`，因此该层级包含一组：

```text
[B, 96, 384]
  -> InvResMLP(96)
  -> ResidualMLP1d(96)
  -> [B, 96, 384]
```

### 3.6 Stage 2

第二层级继续将点数从 384 下采样到 96，并把通道数从 96 提升到 192：

```text
SA2:
  npoint = 96
  nsample = 24
  input channels = 96 + 3
  output channels = 192
```

之后同样接一个 `Stage` 模块：

```text
[B, 192, 96]
  -> InvResMLP(192)
  -> ResidualMLP1d(192)
  -> [B, 192, 96]
```

### 3.7 Stage 3

第三层级进一步将点数从 96 下采样到 24，并把通道数从 192 提升到 384：

```text
SA3:
  npoint = 24
  nsample = 24
  input channels = 192 + 3
  output channels = 384
```

之后再接一个 `Stage` 模块：

```text
[B, 384, 24]
  -> InvResMLP(384)
  -> ResidualMLP1d(384)
  -> [B, 384, 24]
```

### 3.8 Stage3 聚合与 latent 回归

当前采用的配置是 `pointnext_feature_fusion = stage3` 与 `pointnext_global_pool = max_avg`。因此，网络不会再把 `Stage1`、`Stage2`、`Stage3` 的特征做对齐拼接，也不会执行 nearest upsample。全局描述子只来自最后一级 coarse 特征：

```text 
features3: [B, 384, 24]
  -> fused = features3
```

随后对 `features3` 分别做全局最大池化与全局平均池化，并将两者拼接：

```text
[B, 384, 24]
  -> global max pool -> [B, 384]
  -> global avg pool -> [B, 384]
  -> concat          -> [B, 768]
```

再通过一个三层全连接头回归最终 latent：

```text
[B, 768]
  -> Linear(768 -> 512) + LayerNorm + ReLU + Dropout
  -> Linear(512 -> 256) + LayerNorm + ReLU + Dropout
  -> Linear(256 -> 32)
  -> pred_latent [B, 32]
```

因此，点云编码器的核心作用可以概括为：

```text
局部几何编码
  -> 层级式下采样
  -> Stage3 粗尺度形状摘要
  -> 全局统计池化
  -> 低维形状隐变量
```

## 4. 残差点特征变换模块

当前编码器包含两类逐点残差变换模块。

### 4.1 Inverted Residual MLP

每个 `InvResMLP` block 的形式为：

```text
x
  -> Conv1d(C -> 4C, kernel=1)
  -> GroupNorm
  -> ReLU
  -> Conv1d(4C -> C, kernel=1)
  -> GroupNorm
  -> Residual Add
  -> ReLU
```

它通过“先扩张再压缩”的方式增强当前层级的通道表达能力。

### 4.2 ResidualMLP1d

每个 `ResidualMLP1d` block 的形式为：

```text
x
  -> Conv1d(C -> C, kernel=1)
  -> GroupNorm
  -> ReLU
  -> Conv1d(C -> C, kernel=1)
  -> GroupNorm
  -> Residual Add
  -> ReLU
```

它不改变通道宽度，主要作用是稳定特征 refinement，并和 `InvResMLP` 交替堆叠构成每个 stage 的局部建模单元。

## 5. DeepSDF 解码器

### 5.1 输入输出形式

DeepSDF decoder 是一个条件隐式函数网络。给定 latent code `z` 和三维查询点 `x`，网络输出该点的 SDF 值：

```text
F: (z, x) -> s
```

其中：

- `z \in R^32`
- `x \in R^3`
- `s \in R`

因此单次查询的输入维度为：

```text
32 + 3 = 35
```

### 5.2 MLP 结构

解码器主体是一个宽度为 512 的 8 层 MLP，并带有 latent skip connection。结构如下：

```text
input [z, x]: 35
  -> lin0: 35  -> 512
  -> lin1: 512 -> 512
  -> lin2: 512 -> 512
  -> lin3: 512 -> 477
  -> concat original input: 477 + 35 = 512
  -> lin4: 512 -> 512
  -> lin5: 512 -> 512
  -> lin6: 512 -> 512
  -> lin7: 512 -> 512
  -> lin8: 512 -> 1
  -> Tanh
```

其中第 4 层前重新拼接原始输入 `[z, x]`，形成一次中层 skip：

```text
h_4 = concat(h_3, [z, x])
```

这种设计使中层特征仍能直接访问：

- 全局形状编码 `z`；
- 查询点坐标 `x`。

### 5.3 架构作用

该解码器学习的是一个连续隐式表面表示，而不是离散网格模板。因此同一个 latent code `z` 可以在任意空间位置 `x` 上被查询，形成连续 SDF 场：

```text
f_z(x) = F(z, x)
```

随后取零等值面：

```text
{x | F(z, x) = 0}
```

即可恢复对应三维形状的表面。

## 6. 两阶段学习机制

### 6.1 第一阶段：学习形状先验

第一阶段训练 DeepSDF decoder，并为每个训练形状优化一个对应的 latent code。这样，模型建立起：

```text
latent code <-> shape geometry
```

之间的映射关系。经过这一阶段后，解码器已经具备从低维隐变量生成草莓几何体的能力。

### 6.2 第二阶段：学习点云到 latent 的映射

第二阶段固定 DeepSDF decoder，仅训练点云编码器 `E`，使其输出的 `pred_latent` 接近目标 latent。于是整体模型被分解为：

```text
Point cloud -> Encoder -> latent
latent + query points -> Decoder -> SDF
SDF field -> Marching Cubes -> Mesh
```

这种分解比“直接从点云回归网格”更稳定，因为编码器只需学习投影到一个已经成形的几何潜空间，而不必独自承担完整的表面生成任务。

## 7. 损失函数

当前系统的训练分为两条独立链路，因此也对应两套损失：

1. `train_deep_sdf.py` 训练 DeepSDF decoder 以及每个训练样本的 latent embedding。
2. `train.py` 在固定 decoder 的前提下训练 PointNeXt encoder，把输入点云映射到 DeepSDF latent space。

下面只写当前代码与当前配置实际使用的损失形式。

### 7.1 DeepSDF 阶段的损失

在 DeepSDF 预训练阶段，每个样本都会采样一组 SDF 点：

```text
(x_j, s_j),  j = 1, ..., M
```

其中：

- `x_j \in R^3` 是查询点；
- `s_j \in R` 是该点的真实 SDF；
- `z_i \in R^32` 是第 `i` 个训练形状对应的可学习 latent code；
- `F(z_i, x_j)` 是 decoder 的预测值。

当前 `deepsdf/experiments/strawberry/specs.json` 中启用了 `ClampingDistance = 0.1`，因此训练前先对真实值和预测值都做截断：

```text
ŝ_j = clamp(s_j, -delta, delta)
f̂_j = clamp(F(z_i, x_j), -delta, delta)
delta = 0.1
```

主损失是逐点 L1 重建误差：

```text
L_sdf = (1 / M) * Σ_j |f̂_j - ŝ_j|
```

此外，当前配置启用了 `CodeRegularization = true`，并使用 `CodeRegularizationLambda = 1e-4`。代码中的正则项不是平方范数，而是 batch 内 latent 向量的 L2 范数和，并带有前 100 个 epoch 的线性 warm-up：

```text
w(epoch) = min(1, epoch / 100)
L_reg = (lambda_code * w(epoch) / M) * Σ_j ||z_i||_2
```

其中 `lambda_code = 1e-4`。

因此，当前 DeepSDF 的总损失为：

```text
L_deepsdf = L_sdf + L_reg
```

这个目标的含义是：

- `L_sdf` 负责让 decoder 在截断带内准确拟合隐式距离场；
- `L_reg` 负责约束 latent code 的幅值，避免每个样本的隐变量无界增长。

代码里还保留了一个球面正则分支：

```text
|1 - ||z_i||_2|
```

但当前配置并未启用 `CodeRegularizationSphere`，所以它不参与现在的训练。

### 7.2 PointNeXt 阶段的损失

在编码器训练阶段，DeepSDF decoder 固定不更新，PointNeXt 只需要预测一个 latent：

```text
z_pred = E(P)
```

并与预训练得到的目标 latent `z_gt` 对齐。当前 PointNeXt 配置对应 `configs/strawberry.json`，其中：

- `supervised_3d = true`
- `lambda_super = 1.0`
- `lambda_latent_spread = 1.0`
- `lambda_volume = 0.0`
- `3D_loss = false`
- `contrastive = false`
- `kl_divergence = false`
- `reg_latent = false`

因此当前真正参与优化的只有两项。

#### 7.2.1 Latent 回归损失

主监督项 `SuperLoss` 是标准 MSE：

```text
L_super = (1 / Bd) * Σ_{b=1}^B Σ_{k=1}^d (z_pred[b, k] - z_gt[b, k])^2
```

其中：

- `B` 是 batch size；
- `d` 是 latent 维度，当前为 `32`。

这项损失直接迫使编码器输出落到预训练 DeepSDF latent manifold 上，是当前 PointNeXt 训练最核心的监督信号。

#### 7.2.2 Latent spread 损失

为了避免 encoder 只学到“接近均值”的保守输出，代码会比较预测 latent 和目标 latent 在 batch 维度上的逐通道标准差：

```text
sigma_pred[k] = sqrt(Var_b(z_pred[b, k]) + eps)
sigma_gt[k]   = sqrt(Var_b(z_gt[b, k]) + eps)
```

随后对这两个标准差向量做 MSE：

```text
L_spread = (1 / d) * Σ_{k=1}^d (sigma_pred[k] - sigma_gt[k])^2
```

其中 `eps = 1e-6`。这一项不约束单个样本的位置，而是约束整个 batch 的 latent 分布尺度，使预测结果不要发生方差塌缩。

#### 7.2.3 当前 PointNeXt 总损失

因此，当前 PointNeXt 编码器训练的总损失就是：

```text
L_pointnext = lambda_super * L_super
            + lambda_spread * L_spread
```

在当前配置下：

```text
lambda_super = 1.0
lambda_spread = 1.0
```

也就是：

```text
L_pointnext = L_super + L_spread
```

### 7.3 当前未启用但代码保留的损失项

虽然当前配置没有打开，但 `train.py` 中还保留了若干可选项：

- `AttRepLoss`：基于 `HingeEmbeddingLoss` 的同果吸引/异果排斥约束；
- `KLDivLoss`：让 batch latent 分布逼近目标高斯分布的 KL 散度；
- `RegLatentLoss`：约束 `||z_pred||_2` 接近 1；
- `VolumeLoss`：同时结合 `log1p(volume)` 的 `SmoothL1` 与体积相对误差；
- `SDFLoss` / `SDFLoss_new`：把 encoder 输出 latent 送入固定 decoder 后，在规则网格上直接监督 TSDF。

但在 `configs/strawberry.json` 中，这些开关目前都关闭，所以它们不属于“当前 PointNeXt 训练实际使用的损失函数”。

当前默认的 checkpoint 选择准则也与上述目标保持一致：优先依据验证集 `latent_mse` 选择最佳模型，而不是优先按体积误差选模。

## 8. 几何重建过程

在推理阶段，对任意输入点云 `P`，先得到 latent：

```text
z = E(P)
```

再在三维网格上查询解码器得到离散 SDF 体：

```text
s_ijk = F(z, x_ijk)
```

其中 `x_ijk` 表示规则体素网格上的查询坐标。最后通过 marching cubes 在零等值面处提取网格：

```text
M = MC({x | F(z, x) = 0})
```

因此，最终网格并不是编码器直接输出的，而是由：

1. 编码器给出形状 latent；
2. 解码器生成连续隐式场；
3. 等值面提取恢复显式表面。

## 9. 架构总结

该方法本质上是一个“点云编码器 + 隐式形状解码器”的组合框架：

- PointNeXt encoder 负责从不规则点云中提取层级式几何特征，并基于 Stage3 粗尺度特征回归全局形状 latent。
- DeepSDF decoder 负责把 latent 与空间坐标映射为连续 SDF。
- marching cubes 负责把隐式场转换为显式三角网格。

从方法论上看，该架构把“感知”与“生成”分开处理：

- 编码器解决“从观测点云理解形状”的问题；
- 解码器解决“从低维形状表示恢复连续表面”的问题。

这种分工使模型既保留了点云网络对局部几何的建模能力，又继承了隐式表示在连续表面重建上的优势。
