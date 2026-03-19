# CoRe++ 当前代码架构说明

本文档按当前仓库代码整理系统结构，而不是按论文或旧版实现描述。现在的仓库本质上由两段式流程组成：

1. `train_deep_sdf.py` / `reconstruct_deep_sdf.py`
   负责训练 DeepSDF 解码器，并通过潜变量优化得到样本级 latent code。
2. `train.py` / `test.py`
   负责训练一个外部编码器，将输入观测直接映射到 DeepSDF latent space，再调用固定的 DeepSDF 解码器完成重建。

## 1. 总体数据流

当前代码支持两类输入分支：

- RGB-D 分支
  来自 [`dataloaders/cameralaser_w_masks.py`](/home/tianqi/corepp2/dataloaders/cameralaser_w_masks.py)，输入是 `rgb + depth`，主要对应原始 CoRe++ 流程。
- 纯点云分支
  来自 [`dataloaders/pointcloud_dataset.py`](/home/tianqi/corepp2/dataloaders/pointcloud_dataset.py)，输入是 `partial_pcd` 或 `complete` 点云，当前很多草莓相关实验走这条分支。

统一推理链路如下：

1. 数据集读取样本，并整理为统一字典格式，例如 `partial_pcd`、`target_pcd`、`latent`、`bbox`、`center`、`scale`。
2. 编码器将输入观测映射为一个固定维度 latent vector。
3. DeepSDF 解码器接收 `latent + xyz` 查询点，输出对应 SDF。
4. 测试阶段通过 `deepsdf.deep_sdf.mesh.create_mesh()` 在规则网格上查询 SDF 并导出网格。
5. 在生成网格后计算体积、Chamfer Distance、Precision/Recall/F1，并写入 CSV。

## 2. 模块分层

### 2.1 入口脚本

- [`train_deep_sdf.py`](/home/tianqi/corepp2/train_deep_sdf.py)
  训练 DeepSDF 解码器和训练集 latent embedding。
- [`reconstruct_deep_sdf.py`](/home/tianqi/corepp2/reconstruct_deep_sdf.py)
  对指定 split 做 latent optimization，输出重建网格和样本级 latent code。
- [`train.py`](/home/tianqi/corepp2/train.py)
  训练外部编码器，使其预测的 latent 对齐 DeepSDF latent space。
- [`test.py`](/home/tianqi/corepp2/test.py)
  加载编码器和 DeepSDF 解码器，生成网格并输出评估结果。
- [`compute_reconstruction_metrics.py`](/home/tianqi/corepp2/compute_reconstruction_metrics.py)
  一个更独立的离线指标脚本，用预测网格目录和 GT 点云目录直接算指标。

### 2.2 数据层

- [`dataloaders/cameralaser_w_masks.py`](/home/tianqi/corepp2/dataloaders/cameralaser_w_masks.py)
  负责 RGB、深度、mask、相机内参和裁剪后的 partial point cloud 处理。
- [`dataloaders/pointcloud_dataset.py`](/home/tianqi/corepp2/dataloaders/pointcloud_dataset.py)
  负责从 `complete/` 或 `partial/` 目录直接读取 `.ply`，并可从 DeepSDF latent matrix 或逐样本 `.pth` 中取监督信号。
- [`dataloaders/transforms.py`](/home/tianqi/corepp2/dataloaders/transforms.py)
  图像分支使用的 Pad、旋转、翻转等增强。

### 2.3 编码器层

编码器实现主要在：

- [`networks/models.py`](/home/tianqi/corepp2/networks/models.py)
- [`networks/pointnext.py`](/home/tianqi/corepp2/networks/pointnext.py)

`train.py` / `test.py` 会按配置中的 `param["encoder"]` 动态实例化编码器。当前支持：

- `Encoder` / `EncoderBig` / `EncoderPooled` / `EncoderBigPooled`
  标准 2D CNN 编码器，输入通常是 `rgb + depth` 组成的 4 通道张量。
- `ERFNetEncoder`
  ERFNet 风格编码器。
- `DoubleEncoder`
  RGB 与 Depth 双分支编码器，内部带多种融合模块。
- `PointCloudEncoder` / `PointCloudEncoderLarge`
  纯点云 MLP 风格编码器。
- `FoldNetEncoder`
  FoldNet 风格点云编码器。
- `PointNeXtEncoder`
  当前代码中最明确、最现代的纯点云编码器实现。

### 2.4 解码器层

DeepSDF 解码器来自实验目录中的 `specs.json` 指定结构，实际类从 `deepsdf.networks.*` 动态导入。典型调用方式见：

- [`train.py`](/home/tianqi/corepp2/train.py)
- [`test.py`](/home/tianqi/corepp2/test.py)
- [`reconstruct_deep_sdf.py`](/home/tianqi/corepp2/reconstruct_deep_sdf.py)

这部分仍然遵循标准 DeepSDF 形式：

- 输入：`[latent_code, x, y, z]`
- 输出：单个 SDF 标量

外部编码器和 DeepSDF 的边界很明确：编码器只预测 latent，不直接输出网格。

## 3. PointNeXt 分支的详细实现与参数

如果配置 `encoder == "pointnext"`，系统将实例化 [`networks/pointnext.py`](/home/tianqi/corepp2/networks/pointnext.py) 中的 `PointNeXtEncoder`。该实现参考了 PointNeXt 论文，但针对当前的潜变量预测任务进行了轻量化定制。

### 3.1 核心参数配置

默认参数如下（若配置文件中未指定则生效）：

- **k 值 (nsample)**: `24`。在 SA 层的局部邻域聚合中使用 KNN 搜索。
- **分组策略**: 仅使用 KNN，未启用基于半径的 Ball Query (Radius=None)。
- **宽度 (width/C)**: `48`。介于标准 PointNext-S (C=32) 和 PointNext-B (C=64) 之间，提供了平衡的计算量与特征精度。
- **输入点数 (N)**: 默认为 `2048`（由 `input_size` 配置项控制）。

### 3.2 网络结构表 (Encoder Details)

| 阶段 (Stage) | 模块 (Module) | 关键参数 | 输入 $\to$ 输出通道 | 输出点数 | 备注 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Input** | 点云数据 | - | - | 2048 | 坐标 (X, Y, Z) |
| **Stem** | Shared MLP | width=48 | 3 $\to$ 48 | 2048 | 2x (Conv1d+GN+ReLU) |
| **SA1** | Set Abstraction | npoint=512, k=24 | 48 $\to$ 96 | 512 | FPS 采样 + KNN 聚合 |
| (Residual) | Stage 1 Blocks | expansion=4 | 96 $\to$ 96 | 512 | 2x InvResMLP |
| **SA2** | Set Abstraction | npoint=128, k=24 | 96 $\to$ 192 | 128 | FPS 采样 + KNN 聚合 |
| (Residual) | Stage 2 Blocks | expansion=4 | 192 $\to$ 192 | 128 | 2x InvResMLP |
| **Global** | Pooled Features | - | 192 $\to$ 384 | 1 | Concat(GlobalMax, GlobalAvg) |
| **Head** | Output MLP | - | 384 $\to$ latent_size | 1 | 384 $\to$ 512 $\to$ 256 $\to$ out |

### 3.3 关键可配参数 (JSON Config)

- `pointnext_width`: 控制 Stem 和后续阶段的基础通道数 (默认 48)。
- `pointnext_nsample`: 控制 KNN 聚合时的邻居数量 (默认 24)。
- `pointnext_dropout`: Head 部分的随机失活率 (默认 0.05)。


## 4. 数据表示与归一化

### 4.1 纯点云训练/测试数据

[`dataloaders/pointcloud_dataset.py`](/home/tianqi/corepp2/dataloaders/pointcloud_dataset.py) 当前做法是：

1. 从 `complete/` 或 `partial/` 目录读取 `.ply`。
2. 采样或重复采样到固定点数 `pad_size`。
3. 对点云做 zero-centering，即减去整云几何中心。
4. 当前默认 `scale = 1.0`，不会再把点云整体缩放到单位球。
5. 根据采样点计算带 margin 的局部 `bbox`，供部分 3D loss 或后处理使用。

这意味着当前纯点云主路径更偏向“中心对齐，但保留原始尺度”。

### 4.2 自定义测试目录

[`test.py`](/home/tianqi/corepp2/test.py) 里的 `CustomTestDataset` 支持直接读取一个 `.ply` 文件或一个目录下的所有 `.ply`。这个分支和 `PointCloudDataset` 不完全一样：

- 会把每个样本移动到自身质心；
- 会按当前样本最大半径归一化到局部单位球；
- 会返回对应的 `center` 和 `scale`。

这条路径主要用于脱离原始训练集组织方式的快速推理。

### 4.3 RGB-D 数据

[`dataloaders/cameralaser_w_masks.py`](/home/tianqi/corepp2/dataloaders/cameralaser_w_masks.py) 会负责：

- 读取 RGB、Depth、Mask 和相机内参；
- 根据 mask 和深度分布裁剪目标；
- 构造 partial point cloud；
- 在需要时准备 SDF supervision 所需的目标数据。

这个模块比纯点云分支复杂得多，仍然保留着原始 CoRe++ 的很多图像侧逻辑。

## 5. 训练架构

### 5.1 DeepSDF 训练

[`train_deep_sdf.py`](/home/tianqi/corepp2/train_deep_sdf.py) 基本沿用官方 DeepSDF 训练范式：

- 学习一个解码器参数集；
- 同时学习训练样本对应的 latent embeddings；
- 训练过程中的 checkpoint、optimizer state、latent codes、logs 都落在实验目录下。

### 5.2 外部编码器训练

[`train.py`](/home/tianqi/corepp2/train.py) 的核心目标是让编码器输出的 latent 接近 DeepSDF latent space。

训练时先根据 `specs.json` 加载一个预训练解码器，然后再训练编码器。当前实现里，解码器是否一起更新由命令行 `--decoder` 决定：

- 默认只更新编码器；
- 指定 `--decoder` 时同时更新编码器和解码器参数。

### 5.3 当前可叠加的损失

`train.py` 中总损失是若干可选项的加和，是否启用由配置决定：

- `SuperLoss`
  使用 MSE 对齐编码器预测 latent 和 GT latent。
- `AttRepLoss`
  在开启 `contrastive` 时按 `fruit_id` 做吸引/分离。
- `KLDivLoss`
  在开启 `kl_divergence` 时约束 batch latent 分布。
- `RegLatentLoss`
  约束 latent 向量范数。
- `SDFLoss`
  在开启 `3D_loss` 时，把编码器输出 latent 与规则体素网格拼接后送入解码器，直接监督 SDF 场。

其中最常见的主监督仍然是 `supervised_3d=True` 时的 latent 回归。

### 5.4 验证与模型保存

训练过程每隔 `validation_frequency` 做一次验证。当前验证指标不是直接看网格质量，而是：

- 在验证集上计算预测 latent 与 GT latent 的 MSE；
- 将其记为 `Mean Validation Latent MSE`；
- 以此决定 best model。

模型保存由 [`utils.py`](/home/tianqi/corepp2/utils.py) 的 `save_model()` 负责，存储内容包括：

- `encoder_state_dict`
- `decoder_state_dict`
- `optimizer_state_dict`
- `epoch`
- `loss`

## 6. 推理与重建架构

### 6.1 DeepSDF 潜变量重建

[`reconstruct_deep_sdf.py`](/home/tianqi/corepp2/reconstruct_deep_sdf.py) 使用标准 latent optimization：

1. 固定训练好的 DeepSDF 解码器；
2. 为每个测试样本初始化 latent；
3. 通过 SDF 样本的重建误差迭代优化 latent；
4. 输出重建网格和对应 latent code。

这一步的主要作用是：

- 为 DeepSDF 本身做 reconstruction benchmark；
- 为后续编码器训练提供样本级 latent 监督。

### 6.2 编码器直接推理

[`test.py`](/home/tianqi/corepp2/test.py) 的推理流程是：

1. 读取配置和 checkpoint。
2. 实例化编码器与 DeepSDF 解码器。
3. 编码器输出 latent。
4. 调用 `deepsdf.deep_sdf.mesh.create_mesh()` 生成三角网格。
5. 把 latent 另外保存为 `.pth`，便于后续分析。
6. 计算体积和几何指标。
7. 将结果持续写入 CSV。

当前实现中，网格默认输出到固定目录：

- `/home/tianqi/corepp2/logs/strawberry/output`

这属于脚本级硬编码，不是通用工作区抽象。

## 7. 体积与评估指标

### 7.1 体积估计

当前体积估计在 [`test.py`](/home/tianqi/corepp2/test.py) 中由 `_compute_volume_ml()` 完成：

1. 先清理重复点、重复面和退化面；
2. 如果网格 watertight，则直接调用 `mesh.get_volume()`；
3. 如果不是 watertight，则退化为 `ConvexHull(vertices).volume`；
4. 按 `volume_unit` 把结果解释为 `cm`、`mm` 或 `m` 对应的体积单位；
5. 默认情况下直接把估计值作为最终体积使用。

需要特别说明：

- 当前实现不再做“线性归一化尺度的三次方换算”这一类后处理描述。
- 代码里保留了 `volume_scale_factor` 这个可选配置项，但默认值是 `1.0`。
- 因此按当前默认路径，体积就是直接基于生成网格估计出来的体积。

### 7.2 几何指标

`test.py` 当前会输出两份表：

- `shape_completion_results.csv`
- `shape_completion_results_multi_threshold.csv`

评估时的关键逻辑是：

1. 先把 GT 点云和预测网格临时平移到 GT 质心。
2. 再按 GT 的最大半径缩放到局部单位球。
3. 在这个归一化空间里计算 Chamfer Distance 与多阈值 Precision/Recall/F1。

默认支持的阈值集合来自配置：

- `metric_threshold`
- `metric_thresholds`

这比旧版只计算单个阈值的流程更完整。

## 8. 当前架构的几个关键结论

基于现有代码，可以把仓库理解为下面这个结构：

- DeepSDF 仍然是唯一的隐式解码后端。
- 外部编码器已经不是单一实现，而是一个可切换的 encoder family。
- 纯点云路径已经是一等公民，不再只是 RGB-D 流程的附属。
- 训练的主监督对象是 latent，对网格的监督目前是可选附加项。
- 测试阶段的体积与指标计算已经集中在 `test.py`，并且比旧文档描述更贴近“直接从生成网格估计”。

如果后续还要继续维护这份文档，建议优先以以下文件为准：

- [`train.py`](/home/tianqi/corepp2/train.py)
- [`test.py`](/home/tianqi/corepp2/test.py)
- [`dataloaders/pointcloud_dataset.py`](/home/tianqi/corepp2/dataloaders/pointcloud_dataset.py)
- [`networks/pointnext.py`](/home/tianqi/corepp2/networks/pointnext.py)
- [`utils.py`](/home/tianqi/corepp2/utils.py)
