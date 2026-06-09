# Point-MAE Encoder Architecture

## Summary

本文档记录当前仓库中本地接入的 `Point-MAE` 风格点云编码器实现，对应代码：

- [`networks/point_mae.py`](/home/tianqi/corepp2/networks/point_mae.py)

这不是官方完整 `Point-MAE` 预训练框架，而是一个适配当前 `partial point cloud -> DeepSDF latent` 任务的轻量实现。它保留了几个核心思想：

- 将点云划分为局部 patch
- 将每个 patch 编码成 token
- 用 Transformer 在 patch token 之间建模全局关系
- 通过全局池化输出固定长度 latent code

## Overall Pipeline

输入和输出：

- 输入：`x ∈ R^(B, 3, N)`，当前实验中 `N = 2048`
- 输出：`z ∈ R^(B, 32)`，用于回归 DeepSDF latent code

整体流程：

1. 对输入点云做 patch grouping
2. 对每个 patch 做 patch embedding
3. 用 patch center 做 position embedding
4. 将 token 序列送入多层 Transformer block
5. 对 token 做 `max + avg` 全局池化
6. 用 MLP head 输出最终 latent

## Patch Grouping

对应实现：

- [`PointMAEEncoder._group_points()`](/home/tianqi/corepp2/networks/point_mae.py:98)

分组方式：

1. 用 `farthest_point_sample` 选出 `num_groups` 个 patch center
2. 以每个 center 为查询点，用 `knn_point` 找 `group_size` 个邻居
3. 得到形状为 `[B, G, K, 3]` 的局部 patch
4. 对每个 patch 做中心化：`grouped_xyz - center`

当前默认超参：

- `num_groups = 64`
- `group_size = 32`

含义：

- 一朵草莓的点云会被切成 `64` 个局部 patch
- 每个 patch 含 `32` 个邻域点

## Patch Embedding

对应实现：

- [`PointMAEPatchEmbedding`](/home/tianqi/corepp2/networks/point_mae.py:7)

输入：

- `grouped_xyz: [B, G, K, 3]`

处理方式：

- 先转成 `[B, 3, G, K]`
- 经过三层 `1x1 Conv2d + BatchNorm2d + GELU`
- 在 patch 内对 `K` 个点做 `max pooling`

输出：

- `tokens: [B, G, C]`
- 当前 `C = embed_dim = 384`

直观理解：

- 每个局部 patch 会变成一个 `384` 维 token

## Position Embedding

对应实现：

- [`self.pos_embed`](/home/tianqi/corepp2/networks/point_mae.py:70)

做法：

- 使用 patch center 的三维坐标 `[x, y, z]`
- 经过两层 `Linear + GELU` 映射到 `embed_dim`
- 与 patch token 直接相加

公式上可以理解为：

```text
token_i = patch_embed(patch_i) + pos_embed(center_i)
```

这样 Transformer 在建模 token 关系时，不会丢掉 patch 的空间位置信息。

## Transformer Encoder

对应实现：

- [`PointMAETransformerBlock`](/home/tianqi/corepp2/networks/point_mae.py:28)
- [`self.blocks`](/home/tianqi/corepp2/networks/point_mae.py:75)

每个 block 包含：

- `LayerNorm`
- `MultiheadAttention`
- 残差连接
- `LayerNorm`
- `MLP`
- 残差连接

当前默认超参：

- `embed_dim = 384`
- `depth = 8`
- `num_heads = 8`
- `mlp_ratio = 4.0`
- `dropout = 0.1`

作用：

- patch token 之间可以直接交互
- 模型可以从剩余可见局部推断整体形状关系
- 对 partial 点云任务，这比纯局部层级聚合更容易保留全局差异

## Global Pooling And Latent Head

对应实现：

- [`PointMAEEncoder.forward()`](/home/tianqi/corepp2/networks/point_mae.py:108)
- [`self.head`](/home/tianqi/corepp2/networks/point_mae.py:88)

token 经过 Transformer 后，先做：

- `max_pool = tokens.max(dim=1)`
- `avg_pool = tokens.mean(dim=1)`
- 拼接成 `global_feat ∈ R^(B, 768)`

然后送入 MLP head：

```text
768 -> 512 -> 256 -> 32
```

中间层结构：

- `Linear`
- `LayerNorm`
- `GELU`
- `Dropout`

最终输出：

- `latent code ∈ R^(B, 32)`

## Effective Hyperparameters

当前这套实现对应的有效实验设置来自：

- [`point_mae_20260604_dataset_success_config.md`](/home/tianqi/corepp2/markdown/point_mae_20260604_dataset_success_config.md)

其中 Point-MAE 相关超参为：

```json
{
  "encoder": "point_mae",
  "point_mae_num_groups": 64,
  "point_mae_group_size": 32,
  "point_mae_embed_dim": 384,
  "point_mae_depth": 8,
  "point_mae_num_heads": 8,
  "point_mae_mlp_ratio": 4.0,
  "point_mae_dropout": 0.1
}
```

训练设置里，与该 encoder 搭配效果较好的要点是：

- 只启用 `SuperLoss`
- 不启用 `volume`、`latent spread`、`instance contrastive`、`decoder consistency`
- 输入为 partial point cloud
- 输出监督为 DeepSDF latent code

## Why It Worked Better Here

相对当前仓库中的 `pointnext_unet`，这版 `Point-MAE` 更适合当前任务的主要原因是：

- 它先把点云离散成 patch token，再做全局关系建模
- 对 partial 点云的块状缺失更友好
- 更不容易在全局池化前就把实例差异压掉
- 在只用 `SuperLoss` 的情况下，也能保留较好的样本区分度

一句话概括：

`Point-MAE` 在这个项目里更像一个“先理解局部块之间关系，再回归全局形状 latent”的编码器，而不是单纯依赖层级局部聚合和最后一次全局压缩。
