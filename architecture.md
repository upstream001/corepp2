# 最终架构：草莓点云到 DeepSDF 网格

本文档记录在排查并修复 `mesh_volume_ml` 体积坍塌问题后，当前验证有效的最终架构。当前仓库的草莓主流程是两阶段系统：

1. `train_deep_sdf.py` / `reconstruct_deep_sdf.py`
   训练 DeepSDF decoder，并通过 latent optimization 获得样本级 latent code。
2. `train.py` / `test.py`
   训练 PointNeXt encoder，将点云直接映射到 DeepSDF latent space，再调用固定 DeepSDF decoder 重建 mesh。

## 1. 总体目标

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

最终评估使用的体积是 `mesh_volume_ml`，它来自 decoder 生成的 mesh。最终架构中禁用了辅助 `volume_head`。

## 2. 当前关键配置

当前有效的 `configs/strawberry.json` 核心配置：

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

- `3D_loss=false`：关闭低分辨率 SDF/grid loss。该 loss 曾把 encoder 推向平均尺寸 decoder 行为。
- `lambda_super=1.0`：保持强 latent MSE 监督，让 encoder 输出贴近 DeepSDF latent space。
- `lambda_latent_spread=5.0`：显式保持 batch 内 latent 方差，防止 latent 围绕均值坍塌。
- `lambda_volume=0.0`：禁用 volume head。单独的 volume head 可以学到体积，但不一定能迫使 decoder mesh 体现该体积。
- `validate_mesh_volume=true`：验证阶段生成 decoder mesh，并使用 mesh-volume RMSE 选择 best checkpoint。

## 3. 模块分层

### 3.1 入口脚本

- [`train_deep_sdf.py`](/home/tianqi/corepp2/train_deep_sdf.py)
  训练 DeepSDF decoder 和训练集 latent embeddings。
- [`reconstruct_deep_sdf.py`](/home/tianqi/corepp2/reconstruct_deep_sdf.py)
  对指定 split 做 latent optimization，输出样本级 optimized latent 和 mesh。
- [`train.py`](/home/tianqi/corepp2/train.py)
  训练外部 encoder，使其预测 latent 对齐 DeepSDF latent space。
- [`test.py`](/home/tianqi/corepp2/test.py)
  加载 encoder 和 DeepSDF decoder，生成 mesh 并输出评估结果。
- [`test_unseen_data.py`](/home/tianqi/corepp2/test_unseen_data.py)
  对 D405 等 unseen 点云目录做直接推理。
- [`compute_reconstruction_metrics.py`](/home/tianqi/corepp2/compute_reconstruction_metrics.py)
  离线指标脚本，用预测 mesh 目录和 GT 点云目录直接计算指标。

### 3.2 数据层

当前草莓主流程使用：

- [`dataloaders/pointcloud_dataset.py`](/home/tianqi/corepp2/dataloaders/pointcloud_dataset.py)

其行为：

1. 从 `complete/` 或 `partial/` 目录读取 `.ply`。
2. 采样或重复采样到固定点数 `input_size`。
3. 对点云做 zero-centering，即减去整云几何中心。
4. 当前默认 `scale = 1.0`，不把点云缩放到单位球。
5. 根据采样点计算带 margin 的局部 `bbox`。
6. 如果存在 `mapping.json` 和 `ground_truth.csv`，返回对应真实体积 `volume_ml`。

因此当前主路径是：

```text
中心对齐，但保留原始尺度
```

这点对体积估计非常重要。此前 D405 unseen 脚本因额外做单位球归一化并把 mesh 乘回 `scale`，导致体积被 `scale^3` 放大，已经修复。

### 3.3 编码器层

当前草莓主实验使用：

- [`networks/pointnext.py`](/home/tianqi/corepp2/networks/pointnext.py)
- `PointNeXtEncoder`

`train.py` / `test.py` 会按配置中的 `param["encoder"]` 动态实例化编码器。当前支持：

- `Encoder` / `EncoderBig` / `EncoderPooled` / `EncoderBigPooled`
- `ERFNetEncoder`
- `DoubleEncoder`
- `PointCloudEncoder` / `PointCloudEncoderLarge`
- `FoldNetEncoder`
- `PointNeXtEncoder`

当前最终架构中，encoder 只负责输出 DeepSDF latent code，不再训练或使用 volume head。

### 3.4 DeepSDF decoder

DeepSDF decoder 来自实验目录中的 `specs.json`，实际类从 `deepsdf.networks.*` 动态导入。标准形式：

```text
输入: [latent_code, x, y, z]
输出: SDF 标量
```

encoder 和 DeepSDF decoder 的边界很明确：

```text
encoder 预测 latent
decoder 根据 latent + xyz 查询 SDF
mesh 由 decoder 生成
```

当前草莓实验使用的 decoder 配置来自：

- [`deepsdf/experiments/20260331_dataset/specs.json`](/home/tianqi/corepp2/deepsdf/experiments/20260331_dataset/specs.json)
- [`deepsdf/networks/deep_sdf_decoder.py`](/home/tianqi/corepp2/deepsdf/networks/deep_sdf_decoder.py)

核心参数：

```json
{
  "NetworkArch": "deep_sdf_decoder",
  "CodeLength": 32,
  "NetworkSpecs": {
    "dims": [512, 512, 512, 512, 512, 512, 512, 512],
    "dropout": [0, 1, 2, 3, 4, 5, 6, 7],
    "dropout_prob": 0.2,
    "norm_layers": [0, 1, 2, 3, 4, 5, 6, 7],
    "latent_in": [4],
    "xyz_in_all": false,
    "use_tanh": false,
    "latent_dropout": false,
    "weight_norm": true
  }
}
```

因此单次 SDF 查询的输入维度是：

```text
latent_code: 32
xyz:          3
input:       35 = 32 + 3
```

网络展开后是一个 8 层宽度为 512 的 MLP，最后输出 1 个 SDF 标量。因为 `latent_in=[4]`，第 4 个 hidden layer 前会把原始 `[latent, xyz]` 再拼接一次，形成 skip connection。

具体维度：

```text
input z,x,y,z: 35
  -> lin0: 35  -> 512, WeightNorm, ReLU, Dropout(0.2)
  -> lin1: 512 -> 512, WeightNorm, ReLU, Dropout(0.2)
  -> lin2: 512 -> 512, WeightNorm, ReLU, Dropout(0.2)
  -> lin3: 512 -> 477, WeightNorm, ReLU, Dropout(0.2)
  -> concat original input: 477 + 35 = 512
  -> lin4: 512 -> 512, WeightNorm, ReLU, Dropout(0.2)
  -> lin5: 512 -> 512, WeightNorm, ReLU, Dropout(0.2)
  -> lin6: 512 -> 512, WeightNorm, ReLU, Dropout(0.2)
  -> lin7: 512 -> 512, WeightNorm, ReLU, Dropout(0.2)
  -> lin8: 512 -> 1
  -> final Tanh
```

这里 `lin3` 输出 477 的原因是源码中当下一层在 `latent_in` 中时，会设置：

```text
out_dim = next_dim - input_dim = 512 - 35 = 477
```

随后 forward 到 `layer=4` 前执行：

```text
x = concat(x, original_input)
```

使得进入 `lin4` 的维度重新变为 512。这个结构来自 DeepSDF 常用的 latent skip MLP，用于让中层仍能直接访问 shape latent 和查询点坐标。

需要注意：

- `weight_norm=true` 时，`norm_layers` 实际表现为对对应 `Linear` 使用 WeightNorm，而不是额外的 LayerNorm。
- `latent_dropout=false`，所以输入 latent 本身不做 dropout。
- `use_tanh=false` 只是不在最后一层线性输出后启用该分支的 tanh；当前实现最后仍执行 `self.th(x)`，因此输出 SDF 会经过 final `Tanh`。
- `xyz_in_all=false`，所以 xyz 不会在每一层都重复拼接，只在输入层和 `latent_in` skip 中出现。

DeepSDF 第一阶段训练时，每个训练样本有一个可学习 latent embedding：

```text
sample_id -> latent embedding z_i, shape [32]
SDF sample point p_j, shape [3]
concat [z_i, p_j], shape [35]
decoder([z_i, p_j]) -> predicted SDF
```

训练目标是 SDF regression，并带 latent code 正则：

```text
CodeRegularization = true
CodeRegularizationLambda = 0.0001
CodeBound = 1.0
ClampingDistance = 0.1
SamplesPerScene = 16384
ScenesPerBatch = 64
```

第二阶段训练 encoder 时，decoder 的角色不同：decoder 参数固定，encoder 学习预测一个可以让该 decoder 生成正确草莓形状的 `z`。

## 4. PointNeXt 分支

如果配置：

```json
"encoder": "pointnext"
```

系统会实例化 [`networks/pointnext.py`](/home/tianqi/corepp2/networks/pointnext.py) 中的 `PointNeXtEncoder`。

默认核心参数：

- `pointnext_width`: 默认 `48`
- `pointnext_nsample`: 默认 `24`
- `pointnext_dropout`: 默认 `0.05`
- `input_size`: 当前为 `2048`
- `latent_size`: 当前来自 DeepSDF `CodeLength=32`

结构概要：

```text
Input XYZ, N=2048
  -> Stem Shared MLP, 3 -> 48
  -> SA1, npoint=512, k=24, 48 -> 96
  -> InvResMLP blocks
  -> SA2, npoint=128, k=24, 96 -> 192
  -> InvResMLP blocks
  -> Global Max/Avg Pool, 192 -> 384
  -> Head MLP, 384 -> 512 -> 256 -> latent_size
```

源码中的实际 forward 维度如下：

```text
输入 partial_pcd: [B, 2048, 3]
permute 后进入 encoder: [B, 3, 2048]

xyz = x.transpose(1, 2): [B, 2048, 3]

Stem:
  SharedMLP1d(3  -> 48): Conv1d(1x1) + GroupNorm + ReLU
  SharedMLP1d(48 -> 48): Conv1d(1x1) + GroupNorm + ReLU
  features: [B, 48, 2048]

SA1:
  FPS 采样 512 个中心点
  每个中心点 KNN 聚合 24 个邻域点
  拼接局部坐标差 grouped_xyz 和 grouped_features
  输入通道 48 + 3 = 51
  SharedMLP2d(51 -> 96)
  SharedMLP2d(96 -> 96)
  邻域 max pooling
  skip: Conv1d(48 -> 96) + GroupNorm
  输出 xyz:      [B, 512, 3]
  输出 features: [B, 96, 512]

Stage1:
  InvResMLP(96), expansion=4: 96 -> 384 -> 96, residual
  InvResMLP(96), expansion=4: 96 -> 384 -> 96, residual
  features: [B, 96, 512]

SA2:
  FPS 采样 128 个中心点
  每个中心点 KNN 聚合 24 个邻域点
  输入通道 96 + 3 = 99
  SharedMLP2d(99  -> 192)
  SharedMLP2d(192 -> 192)
  邻域 max pooling
  skip: Conv1d(96 -> 192) + GroupNorm
  输出 xyz:      [B, 128, 3]
  输出 features: [B, 192, 128]

Stage2:
  InvResMLP(192), expansion=4: 192 -> 768 -> 192, residual
  InvResMLP(192), expansion=4: 192 -> 768 -> 192, residual
  features: [B, 192, 128]

Global pooling:
  adaptive max pool: [B, 192]
  adaptive avg pool: [B, 192]
  concat:            [B, 384]

Head:
  Linear(384 -> 512, bias=False) + LayerNorm + ReLU + Dropout(0.05)
  Linear(512 -> 256, bias=False) + LayerNorm + ReLU + Dropout(0.05)
  Linear(256 -> 32)

输出:
  pred_latent: [B, 32]
```

这个 encoder 没有直接输出 mesh，也不直接输出最终体积。它只输出 DeepSDF latent。最终几何完全由：

```text
pred_latent + DeepSDF decoder + marching cubes
```

决定。

### 4.1 SetAbstraction 细节

`SetAbstraction` 是 PointNeXt 分支中最关键的降采样模块：

```text
输入:
  xyz:      [B, N, 3]
  features: [B, C, N]

1. farthest_point_sample(xyz, npoint)
   从 N 个点中用 FPS 选出 npoint 个中心点。

2. knn_point(nsample, xyz, new_xyz)
   对每个中心点找 nsample 个近邻点。

3. grouped_xyz = grouped_xyz - center_xyz
   使用局部相对坐标，而不是全局坐标。

4. concat(grouped_xyz, grouped_features)
   将几何局部偏移和点特征拼接。

5. SharedMLP2d + max over neighborhood
   得到每个中心点的局部聚合特征。

6. skip(center_features)
   对中心点原始特征做 1x1 投影，并与聚合特征相加。
```

因此 PointNeXt encoder 的归纳偏置是：

- FPS 保留覆盖整个草莓表面的代表点。
- KNN 局部聚合捕捉表面局部几何。
- 相对坐标让局部形状更稳定。
- global max/avg pooling 把局部特征汇总成整果 latent。

### 4.2 InvResMLP 细节

`InvResMLP` 是一个 1D residual MLP block：

```text
x
  -> Conv1d(C -> 4C, kernel=1, bias=False)
  -> GroupNorm
  -> ReLU
  -> Conv1d(4C -> C, kernel=1, bias=False)
  -> GroupNorm
  -> add input x
  -> ReLU
```

它不改变点数，也不改变通道数，只在每个 stage 内增强局部特征表达。

## 5. 训练监督

### 5.1 DeepSDF 训练

DeepSDF 第一阶段由 [`train_deep_sdf.py`](/home/tianqi/corepp2/train_deep_sdf.py) 完成：

```text
SDF samples + sample latent embedding -> decoder -> SDF prediction
```

训练结果包括：

```text
deepsdf/experiments/20260331_dataset/ModelParameters/100.pth
deepsdf/experiments/20260331_dataset/LatentCodes/100.pth
```

### 5.2 Encoder 训练

对于 train split，encoder 直接读取 DeepSDF 训练阶段产生的 latent matrix：

```text
deepsdf/experiments/20260331_dataset/LatentCodes/100.pth
```

对于 validation split，使用逐样本 optimized latent：

```text
deepsdf/experiments/20260331_dataset/Reconstructions/100/Codes/complete/
```

如果该 validation 目录不存在，运行：

```bash
/home/tianqi/miniconda3/envs/corepp/bin/python reconstruct_deep_sdf.py \
    --experiment ./deepsdf/experiments/20260331_dataset \
    --data ./data/20260331_dataset \
    --checkpoint_decoder 100 \
    --split ./deepsdf/experiments/splits/20260331_dataset_val.json
```

### 5.3 当前总损失

当前有效训练主要由两项组成：

```text
L = lambda_super * SuperLoss(pred_latent, gt_latent)
  + lambda_latent_spread * LatentSpreadLoss(pred_latent, gt_latent)
```

其中：

- `SuperLoss` 是 latent MSE。
- `LatentSpreadLoss` 对齐预测 latent 和 GT latent 的逐维标准差，防止 batch 内 latent collapse。

当前禁用：

```json
"3D_loss": false,
"lambda_volume": 0.0
```

因此：

- 不再用低分辨率 SDF/grid loss 直接训练 encoder。
- 不再训练 volume head。
- `pred_volume_ml` 不再作为最终架构输出指标。

## 6. Validation 策略

当前 validation 使用和最终 test 一致的 mesh 生成路径：

```text
encoder latent
    -> deepsdf.deep_sdf.mesh.create_mesh()
    -> 输出 .ply
    -> ConvexHull(vertices).volume
    -> mesh-volume RMSE
```

validation mesh 保存到：

```text
logs/strawberry/val_output/
```

best checkpoint 优先依据：

```text
Val/mesh_volume_rmse
```

原因：单独的 latent MSE 并不能保证 decoder 生成的 mesh 体积正确。

## 7. Test 输出

测试阶段：

```text
test.py
    -> encoder prediction
    -> deepsdf.deep_sdf.mesh.create_mesh()
    -> logs/strawberry/output/<frame_id>.ply
    -> mesh_volume_ml
```

测试阶段 encoder latent 保存到：

```text
deepsdf/experiments/20260331_dataset/Reconstructions/100/Codes/encoder/
```

主要输出：

```text
shape_completion_results.csv
shape_completion_results_multi_threshold.csv
```

`shape_completion_results_multi_threshold.csv` 当前列包括：

```text
fruit_id
frame_id
complete_volume_ml
mesh_volume_ml
volume_mae_ml
volume_rmse_ml
volume_mape_percent
volume_r2
chamfer_distance
precision / recall / f1
precision_t*
recall_t*
f1_t*
```

最后一行是整体汇总：

```text
fruit_id = SUMMARY
frame_id = n=<测试样本数>
```

其中体积误差指标基于：

```text
GT   = complete_volume_ml
Pred = mesh_volume_ml
```

## 8. 运行命令

训练 encoder：

```bash
/home/tianqi/miniconda3/envs/corepp/bin/python train.py \
    --cfg ./configs/strawberry.json \
    --experiment ./deepsdf/experiments/20260331_dataset \
    --checkpoint_decoder 100
```

运行测试：

```bash
/home/tianqi/miniconda3/envs/corepp/bin/python test.py \
    --cfg ./configs/strawberry.json \
    --experiment ./deepsdf/experiments/20260331_dataset \
    --checkpoint_decoder 100
```

运行 D405 unseen 推理：

```bash
/home/tianqi/miniconda3/envs/corepp/bin/python test_unseen_data.py \
    --cfg ./configs/strawberry.json \
    --experiment ./deepsdf/experiments/20260331_dataset \
    --checkpoint_decoder 100 \
    --input_dir ./data/D405_data \
    --output_dir ./unseen_output \
    --input_unit m
```

## 9. 最新验证结果

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

体积误差：

```text
volume_mae_ml        = 1.062604
volume_rmse_ml       = 1.281944
volume_mape_percent  = 5.621
volume_r2            = 0.8886
```

几何指标：

```text
chamfer_distance mean = 0.0376
f1_t0p05 mean        = 72.59
```

validation split：

```text
val mesh-volume corr = 0.954
val mesh-volume RMSE = 1.25 mL
```

latest encoder latent distribution：

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

说明当前 encoder latent 已接近 DeepSDF latent manifold，并且不再严重坍塌。

## 10. 实用备注

- 不要重新启用 `lambda_volume`，除非 volume head 和 decoder mesh geometry 重新绑定。
- 不要只用 validation latent MSE 选择 checkpoint；必须保留 mesh-volume validation。
- 如果体积坍塌再次出现，检查：
  - `Debug/Train/LatentNormMean`
  - `Debug/Train/TargetLatentNormMean`
  - test 后 encoder latent total variance
  - `corr(mesh_volume_ml, complete_volume_ml)`
- D405 unseen 数据是 partial/domain gap。当前已修复尺度放大问题，但若要提高 D405 样本间差异，需要 partial 风格训练或微调。
