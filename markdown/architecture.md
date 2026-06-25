# 架构：草莓点云到 DeepSDF 网格

本文档描述当前项目中最新、效果最好的整体模型架构，只保留当前有效版本，不再保留旧的 PointNeXt 方案。

## 1. 任务定义

目标是从草莓的残缺点云直接预测其对应的形状隐变量，再通过预训练的 DeepSDF 解码器恢复连续 SDF 场，并进一步提取三角网格。

整体推理路径为：

```text
partial point cloud P
  -> Point-MAE-style encoder E
  -> latent code z
  -> DeepSDF decoder F(z, x)
  -> continuous SDF field
  -> marching cubes
  -> reconstructed mesh M
```

其中：

- `P = {p_i}_{i=1}^N` 表示输入点云，`p_i ∈ R^3`
- `N = 2048`
- `z ∈ R^32` 表示形状隐变量
- `F(z, x)` 表示查询点 `x ∈ R^3` 处的符号距离值

## 2. 总体框架

整个方法采用两阶段框架：

1. 先训练 DeepSDF 解码器，建立 `latent -> SDF field` 的生成映射。
2. 再训练点云编码器，将残缺点云直接回归到 DeepSDF latent space。

因此，整体系统可写为：

```text
z = E(P)
s = F(z, x)
M = MC(F(z, ·))
```

其中：

- `E` 是 Point-MAE 风格点云编码器
- `F` 是 DeepSDF 解码器
- `MC` 表示 marching cubes

这个设计的关键点是：编码器不直接预测体素、网格或点集，而是学习一个可被隐式解码器解释的低维形状表示。

## 3. Point-MAE 风格点云编码器

### 3.1 输入与输出

编码器输入为固定点数的残缺点云：

```text
E: R^(2048 x 3) -> R^32
```

输出为 `32` 维 latent code，用于匹配 DeepSDF latent code。

### 3.2 整体流程

当前编码器的主流程为：

```text
Input point cloud
  -> Patch grouping
  -> Patch embedding
  -> Position embedding
  -> Transformer encoder
  -> attention + max + avg aggregation
  -> MLP head
  -> latent code
```

相对于传统分层点云网络，这一结构先把点云离散成 patch token，再通过 Transformer 建模 token 之间的全局关系。

### 3.3 Patch grouping

编码器首先对输入点云构建局部 patch。具体包含四步：

1. `FPS Sampling`
2. `KNN Query`
3. `Local Normalization`
4. `Local Patch Construction`

对应实现为：

- `num_groups = 64`
- `group_size = 32`

即：

- 从 `2048` 个点中选出 `64` 个 patch center
- 每个 center 查询 `32` 个邻域点
- 每个 patch 用相对坐标表示

形式化地，得到的局部 patch 可写为：

```text
G = {g_j}_{j=1}^{64},  g_j ∈ R^(32 x 3)
```

### 3.4 Patch embedding

每个局部 patch 通过共享 MLP 编码成一个 token。当前 patch embedding 结构为：

```text
3 -> 128 -> 128 -> 384
```

每一层使用：

- `1 x 1 Conv2d`
- `BatchNorm`
- `GELU`

之后在 patch 内对邻域点做 `max pooling`，得到单个 patch token。

因此，每个 patch 最终被映射为：

```text
t_j ∈ R^384
```

整朵草莓对应一个长度为 `64` 的 token 序列。

### 3.5 Position embedding

为了保留 patch 的空间位置信息，每个 patch center 的三维坐标进一步通过一个 MLP 映射到 token 维度：

```text
3 -> 384 -> 384
```

然后与 patch token 相加：

```text
t_j = patch_embed(g_j) + pos_embed(c_j)
```

其中 `c_j` 是第 `j` 个 patch 的中心点坐标。

### 3.6 Transformer encoder

当前主干采用 `8` 层 Transformer block。每个 block 包含：

- `LayerNorm`
- `Multi-Head Attention`
- 残差连接
- `LayerNorm`
- `MLP`
- 残差连接

当前默认超参为：

- `embed_dim = 384`
- `depth = 8`
- `num_heads = 8`
- `mlp_ratio = 4.0`
- `dropout = 0.1`

因此，编码后的 token 表示可写为：

```text
T ∈ R^(64 x 384)
```

### 3.7 Global aggregation

Transformer 输出后，token 被拆分为：

- `patch_tokens`：全部 `64` 个 patch token

然后从 `patch_tokens` 中提取三类全局统计：

1. `attention pooling`
2. `max pooling`
3. `avg pooling`

最后将三者拼接，形成最终全局特征：

```text
global_feat = [attn_pool, max_pool, avg_pool]
```

因此，全局特征维度为：

```text
3 x 384 = 1152
```

这一步是当前版本相较早期实现的关键改进，因为它不再只依赖 `max + avg`，而是额外引入了内容自适应的 `attention pooling`。

消融实验表明，在当前任务中去掉 `CLS token` 后效果更好，因此最终版本不采用 `CLS token`。

### 3.8 Latent regression head

全局特征通过一个 MLP head 映射为最终 latent：

```text
1152 -> 512 -> 256 -> 32
```

每个隐层使用：

- `Linear`
- `LayerNorm`
- `GELU`
- `Dropout`

最终输出：

```text
z_hat ∈ R^32
```

## 4. DeepSDF 解码器

DeepSDF 解码器在训练完成后保持固定，用于把编码器预测的 latent code 解码为连续 SDF 场。

给定 latent code `z` 和空间查询点 `x`，解码器预测：

```text
s = F(z, x)
```

其中 `s` 为该点的符号距离值。

在测试阶段，对三维网格上的点进行密集查询后，使用 marching cubes 提取零水平集，得到最终重建网格。

## 5. 训练目标

当前表现最好的训练目标不是复杂的多重约束，而是一个很轻量的组合：

```text
L = 1.0 * L_super + 0.02 * L_cos
```

其中：

- `L_super`：预测 latent 与目标 latent 的均方误差
- `L_cos`：预测 latent 与目标 latent 的 cosine loss

更具体地，设一个 batch 中第 `i` 个样本的预测 latent 为 `\hat{z}_i \in R^{d_z}`，目标 latent 为 `z_i \in R^{d_z}`，batch 大小为 `B`，则：

```math
\mathcal{L}_{\text{super}}
= \frac{1}{B} \sum_{i=1}^{B} \lVert \hat{z}_i - z_i \rVert_2^2
```

```math
\mathcal{L}_{\text{cos}}
= \frac{1}{B} \sum_{i=1}^{B}
\left(
1 -
\frac{\hat{z}_i^\top z_i}
{\lVert \hat{z}_i \rVert_2 \, \lVert z_i \rVert_2 + \varepsilon}
\right)
```

其中 `\varepsilon` 是数值稳定项。

因此，总损失写为：

```math
\mathcal{L}
= \lambda_{\text{super}} \mathcal{L}_{\text{super}}
+ \lambda_{\text{cos}} \mathcal{L}_{\text{cos}}
```

在当前最优配置下：

```math
\lambda_{\text{super}} = 1.0,\qquad
\lambda_{\text{cos}} = 0.02
```

含义分别为：

- `SuperLoss` 负责约束数值接近
- `LatentCosineLoss` 负责约束方向一致

这版损失的作用不是重塑 latent 分布，而是轻量增强 latent 的判别性，同时避免之前一些强约束带来的塌缩问题。

## 6. 当前有效配置

当前整体架构对应的关键配置为：

```json
{
  "encoder": "point_mae",
  "input_size": 2048,
  "point_mae_num_groups": 64,
  "point_mae_group_size": 32,
  "point_mae_embed_dim": 384,
  "point_mae_depth": 8,
  "point_mae_num_heads": 8,
  "point_mae_mlp_ratio": 4.0,
  "point_mae_dropout": 0.1,
  "lambda_super": 1.0,
  "lambda_latent_cosine": 0.02
}
```

## 7. 当前版本的关键特点

相较早期 PointNeXt 方案，当前架构的优势主要在于：

- 使用 patch token 而不是纯层级局部聚合
- 更适合 partial point cloud 的块状缺失模式
- Transformer 能直接建模远距离 patch 关系
- `attention pooling + max pooling + avg pooling` 提升了全局汇聚质量
- `SuperLoss + 小权重 cosine loss` 能在不破坏 latent 分布的前提下提升预测效果

一句话概括：

当前系统本质上是一个 `Point-MAE-style point cloud encoder + fixed DeepSDF decoder` 的两阶段形状补全框架，其中编码器负责从残缺点云中提取全局形状 token 表示，并将其映射到可被 DeepSDF 解码器解释的 latent space。
