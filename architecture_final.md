# 最终架构：草莓点云到 DeepSDF 网格

本文档记录在排查并修复 `mesh_volume_ml` 体积坍塌问题后，当前验证有效的最终架构。

## 目标

任务是从已配准的草莓点云预测 DeepSDF latent code，然后使用预训练 DeepSDF decoder 重建网格，并基于重建几何体估计体积。

最终验证有效的流程是：

```text
已配准完整/残缺点云
    -> PointNeXt encoder
    -> DeepSDF latent code
    -> 预训练 DeepSDF decoder
    -> marching-cubes mesh
    -> mesh volume / Chamfer / F1 指标
```

最终评估使用的体积是 `mesh_volume_ml`，它来自 decoder 生成的 mesh。最终架构中禁用了辅助的 `volume_head`。

## 主要发现

之前的体积坍塌并不是由输入数据的物理尺度缩放导致的。使用 DeepSDF 优化得到的 latent 可以重建出体积分布正确的 mesh：

```text
GT optimized latent mesh volume 与 GT 的相关性: ~0.995
```

真正的问题来自 encoder 输出的 latent 分布：

- 一次失败实验中，encoder latent 被推到了 DeepSDF latent manifold 之外。
- 另一次失败实验中，encoder latent 虽然靠近均值，但样本间方差过小。
- 这两种情况都会让 decoder 生成接近平均尺寸的 mesh。

最终修复方案是：让 encoder 训练主要回到 latent regression，即接近原论文流程，同时显式保持 latent spread。

## 当前关键配置

使用：

```json
{
  "encoder": "pointnext",
  "supervised_3d": true,
  "3D_loss": false,
  "lambda_super": 1.0,
  "lambda_latent_spread": 5.0,
  "lambda_volume": 0.0,
  "validate_mesh_volume": true,
  "grid_density": 30,
  "input_size": 2048,
  "batch_size": 4,
  "epoch": 50
}
```

关键说明：

- `3D_loss` 被关闭，因为低分辨率 SDF/grid loss 会把 encoder 推向平均尺寸的 decoder 行为。
- `lambda_super=1.0` 让 encoder 输出贴近 DeepSDF latent 监督。
- `lambda_latent_spread=5.0` 防止 latent 围绕均值坍塌。
- `lambda_volume=0.0` 禁用辅助 volume head。单独的 volume head 可以学到体积，但不一定能迫使 decoder mesh 体现该体积。
- `validate_mesh_volume=true` 会在验证阶段生成 decoder mesh，并使用 mesh-volume RMSE 来选择 checkpoint。

## 训练监督

对于 train split，encoder 直接读取 DeepSDF 训练阶段产生的 latent codes：

```text
deepsdf/experiments/20260331_dataset/LatentCodes/100.pth
```

对于 validation split，逐样本优化得到的 latent codes 来自：

```text
deepsdf/experiments/20260331_dataset/Reconstructions/100/Codes/complete/
```

如果该 validation 目录不存在，使用以下命令生成：

```bash
/home/tianqi/miniconda3/envs/corepp/bin/python reconstruct_deep_sdf.py \
    --experiment ./deepsdf/experiments/20260331_dataset \
    --data ./data/20260331_dataset \
    --checkpoint_decoder 100 \
    --split ./deepsdf/experiments/splits/20260331_dataset_val.json
```

## 验证策略

当前 validation 使用和最终 test 相同的 mesh 生成路径：

```text
encoder latent
    -> deepsdf.deep_sdf.mesh.create_mesh()
    -> 输出 .ply
    -> ConvexHull(vertices).volume
    -> mesh-volume RMSE
```

validation mesh 会写入：

```text
logs/strawberry/val_output/
```

best checkpoint 优先依据：

```text
Val/mesh_volume_rmse
```

这一点很重要，因为单独的 latent MSE 并不能保证 decoder 生成的 mesh 体积正确。

## 最终测试路径

测试阶段使用：

```text
test.py
    -> encoder prediction
    -> deepsdf.deep_sdf.mesh.create_mesh()
    -> logs/strawberry/output/<frame_id>.ply
    -> mesh_volume_ml
```

测试阶段的 encoder latent 会保存到：

```text
deepsdf/experiments/20260331_dataset/Reconstructions/100/Codes/encoder/
```

生成的测试 mesh 会保存到：

```text
logs/strawberry/output/
```

## 运行命令

训练 encoder：

```bash
/home/tianqi/miniconda3/envs/corepp/bin/python train.py \
    --cfg ./configs/strawberry.json \
    --experiment ./deepsdf/experiments/20260331_dataset \
    --checkpoint_decoder 100
```

运行最终测试：

```bash
/home/tianqi/miniconda3/envs/corepp/bin/python test.py \
    --cfg ./configs/strawberry.json \
    --experiment ./deepsdf/experiments/20260331_dataset \
    --checkpoint_decoder 100
```

主要输出文件：

```text
shape_completion_results_multi_threshold.csv
```

重点检查列：

```text
complete_volume_ml
mesh_volume_ml
chamfer_distance
f1_t0p05
```

在最终架构中，`pred_volume_ml` 预期为空或 NaN，因为 `volume_head` 已禁用。

## 最新验证结果

最新 test set 汇总：

```text
complete_volume_ml:
  mean = 19.86
  std  = 3.87
  min  = 13.71
  max  = 30.36

mesh_volume_ml:
  mean = 20.28
  std  = 4.37
  min  = 11.49
  max  = 30.40

corr(mesh_volume_ml, complete_volume_ml) = 0.963
```

几何指标：

```text
chamfer_distance mean = 0.0376
f1_t0p05 mean        = 72.59
```

validation split 的 mesh-volume 检查：

```text
val mesh-volume corr = 0.954
val mesh-volume RMSE = 1.25 mL
```

最新 encoder latent 分布：

```text
encoder latent norm mean/std = 0.824 / 0.113
encoder latent total variance = 0.566
```

参考分布：

```text
DeepSDF train latent norm mean/std = 0.931 / 0.065
DeepSDF train latent total variance = 0.752

complete optimized latent norm mean/std = 1.492 / 0.204
complete optimized latent total variance = 1.277
```

这说明当前 encoder latent 已接近 DeepSDF latent manifold，并且不再严重坍塌。

## 实用备注

- 不要重新启用 `lambda_volume`，除非 volume head 和 decoder mesh geometry 重新绑定。独立 volume head 可能看起来很准，但 decoder mesh 仍然会坍塌。
- 不要只用 validation latent MSE 选择 checkpoint。mesh-volume validation 能捕捉真实失败模式。
- 如果体积坍塌再次出现，检查：
  - `Debug/Train/LatentNormMean`
  - `Debug/Train/TargetLatentNormMean`
  - test 后 encoder latent total variance
  - `corr(mesh_volume_ml, complete_volume_ml)`
- 下一步有价值的调参 sweep：
  - `lambda_latent_spread = 3.0`
  - `lambda_latent_spread = 5.0`
  - `lambda_latent_spread = 8.0`

