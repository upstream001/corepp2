# 体积估计不准与 `mesh_volume_ml` 坍塌问题复盘

本文档记录本项目中遇到的草莓重建体积估计不准问题，以及最终定位和解决方案。

## 问题现象

在运行 `test.py` 后，输出文件：

```text
shape_completion_results_multi_threshold.csv
```

中的 `mesh_volume_ml` 一开始出现明显的体积坍塌现象：

- 不同样本的 `mesh_volume_ml` 数值过于接近；
- 小果被高估；
- 大果被低估；
- `mesh_volume_ml` 和 `complete_volume_ml` 的相关性不足；
- decoder 生成的 mesh 更像平均尺寸草莓。

早期测试中，GT 体积分布和 mesh 体积分布对比如下：

```text
complete_volume_ml:
  mean = 19.86
  std  = 3.87
  min  = 13.71
  max  = 30.36

mesh_volume_ml:
  mean = 19.99
  std  = 1.41
  min  = 17.17
  max  = 23.15
```

这说明 GT 体积跨度较大，但预测 mesh 体积被压缩到 20 mL 附近。

## 初始假设

最初怀疑问题可能来自：

1. 点云是否做过缩放但重建时没有反缩放；
2. `create_mesh()` 使用固定 `[-3, 3]^3` 采样空间；
3. `mesh_volume_ml` 使用 `ConvexHull(vertices).volume` 而不是真实 watertight mesh volume；
4. decoder 本身是否只能生成平均尺寸；
5. encoder 预测 latent 是否发生 collapse。

经过逐项验证，最终确认主要问题不是数据缩放，也不是 DeepSDF decoder 本身。

## 关键诊断 1：DeepSDF optimized latent 没有体积坍塌

首先使用 DeepSDF 优化得到的 complete/GT latent 直接生成 mesh，并计算体积。

已有 optimized latent 和 mesh 位于：

```text
deepsdf/experiments/20260331_dataset/Reconstructions/100/Codes/complete/
deepsdf/experiments/20260331_dataset/Reconstructions/100/Meshes/complete/
```

诊断结果：

```text
GT complete_volume_ml:
  mean = 19.95
  std  = 3.25
  min  = 13.93
  max  = 30.36

GT optimized latent mesh volume:
  mean = 19.69
  std  = 3.12
  min  = 12.99
  max  = 29.36

corr(GT, GT optimized latent mesh) = 0.995
mesh / GT ratio:
  mean = 0.987
  std  = 0.017
```

结论：

```text
DeepSDF decoder + optimized latent 可以恢复正确体积分布。
```

因此，体积坍塌不是 decoder 本身无法表达尺寸，也不是 `create_mesh()` 固定采样空间必然导致的问题。

## 关键诊断 2：encoder latent 分布异常

随后比较三类 latent：

1. DeepSDF 训练阶段 autodecoder latent：

```text
deepsdf/experiments/20260331_dataset/LatentCodes/100.pth
```

2. complete 点云优化得到的 latent：

```text
deepsdf/experiments/20260331_dataset/Reconstructions/100/Codes/complete/
```

3. encoder 在 test split 上预测出的 latent：

```text
deepsdf/experiments/20260331_dataset/Reconstructions/100/Codes/encoder/
```

失败实验中的一个典型结果：

```text
train_autodecoder:
  norm mean/std = 0.93 / 0.065
  total variance = 0.752

complete_optimized_val:
  norm mean/std = 1.49 / 0.204
  total variance = 1.277

encoder_pred_test:
  norm mean/std = 3.58 / 0.129
  total variance = 0.338
```

这说明 encoder 输出 latent 远离 DeepSDF latent manifold。

另一次修正后又出现了相反问题：

```text
encoder_pred_test:
  norm mean/std = 0.66 / 0.048
  total variance = 0.107
```

这说明 encoder latent 虽然靠近均值，但样本间方差严重不足，仍然会导致 decoder 输出平均形状。

最终确认：

```text
体积坍塌的核心原因是 encoder latent 分布不合适。
```

具体包括：

- latent 跑到 DeepSDF decoder 不熟悉的区域；
- 或 latent 聚集在均值附近，缺少尺寸相关变化；
- volume head 可以学到体积，但 decoder 不一定能把该 latent 解释成正确几何尺寸。

## 误区：volume head 看起来准，但 mesh 仍会塌

早期配置中启用了：

```json
"lambda_volume": 0.5
```

并训练了辅助 `volume_head`。

当时的结果中：

```text
corr(pred_volume_ml, complete_volume_ml) = 0.940
```

看起来体积预测很好。但 `pred_volume_ml` 来自：

```text
volume_head(latent)
```

而最终 mesh 体积来自：

```text
encoder latent -> DeepSDF decoder -> mesh -> ConvexHull volume
```

这两条路径并不等价。volume head 可以学到一个 decoder 不理解的体积方向，从而出现：

```text
volume_head 预测准，但 decoder mesh 体积仍然坍塌
```

因此最终架构中禁用了 volume head：

```json
"lambda_volume": 0.0
```

## 误区：只看 latent MSE 不够

只用 validation latent MSE 选择 checkpoint 也不可靠。

原因是：

- latent MSE 小不一定代表 decoder mesh 体积正确；
- encoder latent 可能靠近均值，但样本间方差不足；
- decoder 输出仍可能接近平均形状。

因此最终加入了 mesh-volume validation。

## 最终解决方案

最终有效配置：

```json
{
  "3D_loss": false,
  "lambda_super": 1.0,
  "lambda_latent_spread": 5.0,
  "lambda_volume": 0.0,
  "validate_mesh_volume": true
}
```

### 1. 关闭 `3D_loss`

```json
"3D_loss": false
```

低分辨率 SDF/grid loss 会把 encoder 推向平均尺寸的 decoder 行为。关闭后，训练更接近原论文中的 latent regression 流程。

### 2. 恢复强 latent 监督

```json
"lambda_super": 1.0
```

让 encoder 输出更贴近 DeepSDF 训练 latent，避免跑出 decoder 熟悉的 latent manifold。

### 3. 提高 latent spread 约束

```json
"lambda_latent_spread": 5.0
```

防止 encoder latent 虽然靠近均值，但样本间方差过小。

### 4. 关闭 volume head

```json
"lambda_volume": 0.0
```

避免 volume head 学到和 decoder mesh 不一致的体积捷径。

### 5. 使用 mesh-volume validation 选 best checkpoint

```json
"validate_mesh_volume": true
```

validation 阶段使用和 test 阶段一致的 mesh 生成路径：

```text
encoder latent
    -> deepsdf.deep_sdf.mesh.create_mesh()
    -> .ply mesh
    -> ConvexHull(vertices).volume
    -> mesh-volume RMSE
```

validation mesh 输出到：

```text
logs/strawberry/val_output/
```

best checkpoint 优先依据：

```text
Val/mesh_volume_rmse
```

## 最终效果

最新测试集结果：

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

体积误差指标：

```text
volume_mae_ml      = 1.062604
volume_rmse_ml     = 1.281944
volume_mape_percent = 5.621
volume_r2          = 0.8886
```

几何指标：

```text
chamfer_distance mean = 0.0376
f1_t0p05 mean        = 72.59
```

validation split 上：

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

说明 encoder latent 已经回到 DeepSDF latent manifold 附近，并且不再严重坍塌。

## 当前输出文件改进

`test.py` 已加入最终统计行。

输出文件：

```text
shape_completion_results_multi_threshold.csv
```

最后一行：

```text
fruit_id = SUMMARY
frame_id = n=<测试样本数>
```

并统计：

```text
volume_mae_ml
volume_rmse_ml
volume_mape_percent
volume_r2
```

这些体积误差指标基于：

```text
GT   = complete_volume_ml
Pred = mesh_volume_ml
```

## 后续建议

如果体积坍塌再次出现，优先检查：

```text
Debug/Train/LatentNormMean
Debug/Train/TargetLatentNormMean
encoder latent total variance
corr(mesh_volume_ml, complete_volume_ml)
```

可继续调参：

```json
"lambda_latent_spread": 3.0
"lambda_latent_spread": 5.0
"lambda_latent_spread": 8.0
```

当前 `lambda_latent_spread=5.0` 是已验证有效的基线。

