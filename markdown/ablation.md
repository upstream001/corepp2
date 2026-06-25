# 基于新损失函数的消融实验设计

本文档围绕当前最新主模型的训练目标设计消融实验。重点不是重新探索已经明确失败的强约束损失，而是回答：

1. `LatentCosineLoss` 是否稳定有效？
2. 它的最佳权重范围是什么？
3. `SuperLoss` 与 `LatentCosineLoss` 是互补的，还是只是偶然改善？
4. 是否还需要引入额外 loss 项，或者当前组合已经足够？

## 1. 当前参考模型

当前默认参考模型为：

- encoder: `point_mae`
- global aggregation: `Attention + Max + Avg`
- input points: `2048`
- patch grouping: `64 groups`, `32 neighbors`
- token dim: `384`
- transformer depth: `8`
- latent size: `32`
- DeepSDF checkpoint: `20`

当前参考损失为：

```text
L = 1.0 * L_super + 0.02 * L_cos
```

其中：

- `L_super`: latent MSE
- `L_cos`: latent cosine loss

## 2. 这组 loss 消融要回答什么

这组消融的目标很具体：

### 2.1 Cosine loss 是否真的有用

你已经观察到：

- 纯 `SuperLoss` 可以工作
- 加入小权重 `LatentCosineLoss` 后效果进一步提升

因此需要确认：

- 提升是否稳定
- 提升是否只在 `0.02` 附近出现

### 2.2 最优权重是否可解释

如果 `0.02` 最好，而 `0.05` 退化，说明：

- cosine 约束是有帮助的
- 但它只能做轻量辅助，不能主导训练

这对论文和后续实验都很重要。

### 2.3 是否还值得继续加其它 loss

如果实验表明：

- `SuperLoss + Cosine` 已经明显优于 `SuperLoss only`
- 再增大 cosine 权重反而退化

那说明当前最优方向是：

- 保持 loss 简洁
- 继续优化结构或数据设置

而不是继续叠更多辅助损失。

## 3. 建议的主消融矩阵

### 3.1 主表：Cosine 权重消融

这是最重要的一组实验，建议放在论文主表中。

固定以下条件不变：

- encoder 结构不变
- pooling 结构不变
- dataset 不变
- DeepSDF checkpoint 不变
- batch size / input size / epoch 不变

只改变：

- `lambda_latent_cosine`

建议实验：

1. `SuperLoss only`
   - `lambda_super = 1.0`
   - `lambda_latent_cosine = 0.00`

2. `SuperLoss + Cosine(0.01)`
   - `lambda_super = 1.0`
   - `lambda_latent_cosine = 0.01`

3. `SuperLoss + Cosine(0.02)`
   - `lambda_super = 1.0`
   - `lambda_latent_cosine = 0.02`

4. `SuperLoss + Cosine(0.03)`
   - `lambda_super = 1.0`
   - `lambda_latent_cosine = 0.03`

5. `SuperLoss + Cosine(0.05)`
   - `lambda_super = 1.0`
   - `lambda_latent_cosine = 0.05`

如果计算资源紧张，最少保留这三组：

1. `0.00`
2. `0.02`
3. `0.05`

### 3.2 结果解释

这组实验应重点观察：

- `0.00 -> 0.02` 是否明显提升
- `0.02 -> 0.05` 是否退化

如果结果满足：

- `0.02` 最优
- `0.05` 退化

则可以得到一个很清楚的结论：

> Latent cosine consistency is beneficial only as a light auxiliary objective; overly strong cosine regularization degrades latent regression quality.

## 4. 第二层 loss 消融

如果第一层消融完成后，还想进一步回答 loss 设计问题，可以做下面两类。

### 4.1 MSE 与 Cosine 的互补性

目的：

- 验证 cosine loss 是否只是弱替代 MSE，还是与 MSE 互补

建议实验：

1. `SuperLoss only`
2. `CosineLoss only`
3. `SuperLoss + CosineLoss`

其中：

- `CosineLoss only` 仅作为诊断实验
- 它未必会是强 baseline，但可以回答一个问题：
  - 单独方向一致是否不足以确定正确 latent

预期结论通常会是：

- `Cosine only` 不足
- `SuperLoss + Cosine` 优于 `SuperLoss only`

这能说明两者是互补关系。

### 4.2 主损失与强辅助损失的对比

这组实验不建议一开始就做，但可以作为附表，用来支撑“为什么我们最终只保留轻量 cosine loss”。

建议实验：

1. `SuperLoss only`
2. `SuperLoss + Cosine(0.02)`
3. `SuperLoss + LatentSpread`
4. `SuperLoss + VolumeLoss`
5. `SuperLoss + DecoderConsistency`

注意：

- 这组实验的目的不是重新找最优
- 而是证明之前那些强辅助项会更容易扰乱 latent 学习

如果这些实验结果再次显示退化，就能为最终方法选择提供很强的论据。

## 5. 不建议优先继续扩展的 loss

根据你目前的实验轨迹，以下 loss 不建议作为下一阶段主线：

### 5.1 强分布对齐项

- `LatentSpreadLoss`
- `LatentNormLoss`
- `LatentMeanLoss`

原因：

- 它们会直接塑形 latent 分布
- 过去多次导致 collapse 或均值解

### 5.2 强任务辅助项

- `VolumeLoss`
- `DecoderConsistencyLoss`
- `InstanceContrastiveLoss`

原因：

- 它们在当前任务设定下比主监督更容易主导优化方向
- 历史实验已经给出较强负面信号

### 5.3 多个 loss 同时叠加

例如：

- `Super + Cosine + Spread`
- `Super + Cosine + Volume`
- `Super + Cosine + DecoderConsistency`

不建议当前阶段这么做，因为：

- 无法判断提升或退化来自哪一项
- 会让主结论变得不清楚

## 6. 推荐实验顺序

建议严格按下面顺序做：

1. `Cosine weight ablation`
   - `0.00 / 0.01 / 0.02 / 0.03 / 0.05`

2. `Super vs Cosine vs Super+Cosine`
   - 验证互补性

3. `Super+Cosine` 对比其它强辅助损失
   - 作为附表支撑最终选择

## 7. 每个实验必须记录的指标

建议每次至少记录：

- `volume_mae_ml`
- `volume_rmse_ml`
- `volume_r2`
- `corr(complete_volume_ml, mesh_volume_ml)`
- `chamfer_distance`
- `f1_t0p05`
- `pred volume mean/std`

其中尤其重要的是：

- `corr`
- `pred volume std`

因为它们最能反映 loss 是否在提升区分度，还是又把模型拉回均值解。

## 8. 建议的论文组织方式

### 主表

主表建议放：

- `SuperLoss only`
- `Super + Cos(0.01)`
- `Super + Cos(0.02)`
- `Super + Cos(0.03)`
- `Super + Cos(0.05)`

这组最能直接支撑你的新 loss 设计。

### 附表

附表建议放：

- `Cosine only`
- `Super + Spread`
- `Super + Volume`
- `Super + DecoderConsistency`

这样可以说明：

- 为什么最终只保留 `LatentCosineLoss`
- 为什么没有继续采用更复杂的辅助项

## 9. 当前最值得优先回答的问题

如果下一步只能做一组 loss 实验，我建议优先回答：

> `LatentCosineLoss` 的最优权重是否稳定落在一个小范围内？

因为这能直接决定后续是否还要继续在 loss 上投入：

- 如果 `0.01-0.03` 稳定有效，说明这条路成立
- 如果结果波动很大，说明当前提升可能更多来自结构或随机性，而不是 loss 本身
