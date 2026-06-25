# Point-MAE Encoder Architecture

## Summary

本文档记录当前仓库中本地接入的 `Point-MAE` 风格点云编码器实现，对应代码：

- [`networks/point_mae.py`](/home/tianqi/corepp2/networks/point_mae.py)

这不是官方完整 `Point-MAE` 预训练框架，而是一个适配当前 `partial point cloud -> DeepSDF latent` 任务的轻量实现。它保留了几个核心思想：

- 将点云划分为局部 patch
- 将每个 patch 编码成 token
- 用 Transformer 在 patch token 之间建模全局关系
- 引入 attention pooling 强化全局汇聚
- 输出固定长度 latent code

## Overall Pipeline

输入和输出：

- 输入：`x ∈ R^(B, 3, N)`，当前实验中 `N = 2048`
- 输出：`z ∈ R^(B, 32)`，用于回归 DeepSDF latent code

整体流程：

1. 对输入点云做 patch grouping
2. 对每个 patch 做 patch embedding
3. 用 patch center 做 position embedding
4. 将 token 序列送入多层 Transformer block
5. 对 patch token 做 `attention + max + avg` 汇聚
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

## Global Aggregation And Latent Head

对应实现：

- [`PointTokenAttentionPooling`](/home/tianqi/corepp2/networks/point_mae.py:8)
- [`PointMAEEncoder.forward()`](/home/tianqi/corepp2/networks/point_mae.py:122)
- [`self.head`](/home/tianqi/corepp2/networks/point_mae.py:103)

token 经过 Transformer 后，先分成：

- `patch_tokens = tokens`

然后对 `patch_tokens` 做三种汇聚：

- `attention pooling`
- `max pooling`
- `avg pooling`

最后拼接成：

```text
global_feat = [attn_pool, max_pool, avg_pool]
```

所以当前全局特征维度是：

- `3 * embed_dim = 1152`

然后送入 MLP head：

```text
1152 -> 512 -> 256 -> 32
```

中间层结构：

- `Linear`
- `LayerNorm`
- `GELU`
- `Dropout`

最终输出：

- `latent code ∈ R^(B, 32)`

这版和早期实现的关键区别是：

- 不再只用 `max + avg pooling`
- 额外加入了 `attention pooling`
- 全局表示更强调 token 之间的内容差异，而不是只依赖统计量
- 消融实验表明 `CLS token` 在当前任务中无益，因此最终版本不采用它

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

当前较优训练设置里，与该 encoder 搭配效果较好的要点是：

- 启用 `SuperLoss`
- 启用小权重 `LatentCosineLoss`
- 不启用 `volume`、`latent spread`、`instance contrastive`、`decoder consistency`
- 输入为 partial point cloud
- 输出监督为 DeepSDF latent code

对应关键配置：

```json
{
  "lambda_super": 1.0,
  "lambda_latent_cosine": 0.02,
  "lambda_volume": 0.0,
  "lambda_latent_spread": 0.0,
  "lambda_instance_contrastive": 0.0,
  "lambda_decoder_consistency": 0.0,
  "lambda_latent_norm": 0.0,
  "lambda_latent_mean": 0.0
}
```

## Current Best Result

当前更优的一版结果来自：

- [`检查点暂存/加入新的损失函数/shape_completion_results_multi_threshold.csv`](/home/tianqi/corepp2/检查点暂存/加入新的损失函数/shape_completion_results_multi_threshold.csv)

相对早期纯 `SuperLoss` 基线，这一版在加入小权重 `LatentCosineLoss` 后指标进一步提升：

- `volume_mae_ml = 1.791514`
- `volume_rmse_ml = 2.133881`
- `chamfer_distance = 0.061642`
- `f1_t0p05 = 46.15188`
- `corr(complete_volume_ml, mesh_volume_ml) = 0.918525`

当前可以把这版理解为：

```text
Total Loss = 1.0 * SuperLoss + 0.02 * LatentCosineLoss
```

这里 `LatentCosineLoss` 的作用不是重新塑形 latent 分布，而是轻量约束：

- `pred latent` 和 `gt latent` 的方向一致
- 在保留 `SuperLoss` 数值监督的同时，提高样本间区分度
- 避免像之前一些强分布约束那样把 latent 拉塌

## Why It Worked Better Here

相对当前仓库中的 `pointnext_unet`，这版 `Point-MAE` 更适合当前任务的主要原因是：

- 它先把点云离散成 patch token，再做全局关系建模
- 对 partial 点云的块状缺失更友好
- 更不容易在全局池化前就把实例差异压掉
- `attention pooling + max pooling + avg pooling` 让全局汇聚更有表达力
- 在 `SuperLoss + 小权重 cosine` 的设置下，能进一步保留较好的样本区分度

一句话概括：

`Point-MAE` 在这个项目里更像一个“先理解局部块之间关系，再回归全局形状 latent”的编码器，而不是单纯依赖层级局部聚合和最后一次全局压缩。
