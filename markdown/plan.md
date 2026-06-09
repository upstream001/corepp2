# Improvement Plan

## Goal

在当前有效配置基础上，继续提升 `20260604_dataset` 上的 shape completion 效果，同时尽量避免破坏当前已经取得的稳定结果。

当前最佳基线：

- encoder: `point_mae`
- loss: 仅 `SuperLoss`
- decoder checkpoint: `20`
- volume RMSE: `2.602084`
- volume R2: `0.757744`
- chamfer distance: `0.063259`
- f1_t0p05: `45.9`

参考记录：

- [`point_mae_20260604_dataset_success_config.md`](/home/tianqi/corepp2/markdown/point_mae_20260604_dataset_success_config.md)

## Principle

后续实验遵循以下原则：

1. 一次只改一个关键因素。
2. 优先做低风险、低成本、可快速回滚的实验。
3. 暂不重新启用复杂附加损失函数。
4. 所有新实验都与当前基线直接对比。

## Baseline

固定当前基线配置：

- [`configs/20260604_dataset.json`](/home/tianqi/corepp2/configs/20260604_dataset.json)
- DeepSDF experiment:
  [`deepsdf/experiments/20260604_dataset`](/home/tianqi/corepp2/deepsdf/experiments/20260604_dataset/specs.json)
- DeepSDF latent code:
  [`LatentCodes/20.pth`](/home/tianqi/corepp2/deepsdf/experiments/20260604_dataset/LatentCodes/20.pth)
- Encoder checkpoint:
  [`_20260604_dataset_best_model.pt`](/home/tianqi/corepp2/logs/20260604_dataset/checkpoints/_20260604_dataset_best_model.pt)

## Experiment Order

### 1. Compare DeepSDF Checkpoints

目标：

- 判断当前限制是否来自 `decoder / latent manifold`

做法：

- 保持 encoder 配置不变
- 分别测试不同 DeepSDF checkpoint：
  - `20`
  - `40`
  - `60`
  - `100`

命令模板：

```bash
python train.py \
  --cfg ./configs/20260604_dataset.json \
  --experiment ./deepsdf/experiments/20260604_dataset \
  --checkpoint_decoder <CKPT>
```

```bash
python test.py \
  --cfg ./configs/20260604_dataset.json \
  --experiment ./deepsdf/experiments/20260604_dataset \
  --checkpoint_decoder <CKPT>
```

重点观察：

- `volume_rmse_ml`
- `volume_r2`
- `chamfer_distance`
- `f1_t0p05`

### 2. Increase Encoder Epoch

目标：

- 判断 encoder 是否还未充分收敛

做法：

- 保持其余参数不变
- 将 `epoch` 从 `25` 提高到：
  - `40`
  - `60`

仅在第 1 步找到最佳 decoder checkpoint 后进行。

### 3. Increase Input Points

目标：

- 提升 partial point cloud 的几何信息量

做法：

- 保持其余配置不变
- 将 `input_size` 从 `2048` 改为：
  - `3072`
  - `4096`

风险：

- 显存占用上升
- 训练速度变慢

### 4. Tune Point-MAE Capacity

目标：

- 提升 encoder 表达能力

做法：

一次只改一个超参：

- `point_mae_depth: 8 -> 12`
- `point_mae_num_groups: 64 -> 96`
- `point_mae_embed_dim: 384 -> 512`

不建议同时改多个结构超参。

### 5. Batch Structure Tuning

目标：

- 改善训练稳定性和 instance coverage

做法：

- 在不启用复杂损失的前提下调整：
  - `batch_size: 8 -> 12`
  - `instances_per_batch: 4 -> 6`

说明：

- 当前 `group_by_instance_batch = true`
- 即使不启用 contrastive，这种 batch 组织仍可能改善训练分布

## Not Recommended For Now

当前阶段不建议优先重试以下项：

- `lambda_volume > 0`
- `lambda_latent_spread > 0`
- `lambda_instance_contrastive > 0`
- `lambda_decoder_consistency > 0`
- `lambda_latent_norm > 0`
- `lambda_latent_mean > 0`

原因：

- 这些附加项在当前任务中已多次导致 latent collapse 或性能下降
- 当前最稳定的配置是仅保留 `SuperLoss`

## Evaluation Rule

每次实验结束后统一记录：

- `volume_mae_ml`
- `volume_rmse_ml`
- `volume_mape_percent`
- `volume_r2`
- `chamfer_distance`
- `f1_t0p05`

同时记录：

- 使用的 decoder checkpoint
- encoder checkpoint
- config 文件副本或关键 diff

## Recommendation

下一步最推荐立即执行的是：

1. 固定当前 encoder 配置，比较 `checkpoint_decoder = 20 / 40 / 60 / 100`
2. 选出最佳 decoder checkpoint
3. 只提高 encoder `epoch`

这是当前最稳妥、最可能继续涨点的路线。
