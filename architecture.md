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

1. 数据集读取样本，并整理为统一字典格式，例如 `partial_pcd`、`target_pcd`、`latent`、`bbox`、`center`、`scale`、`volume_ml`。
2. 编码器将输入观测映射为一个固定维度 latent vector。
3. 训练阶段可选的 `volume_head` 会从 latent 直接回归体积，用真实 `volume_ml` 做显式体积监督。
4. DeepSDF 解码器接收 `latent + xyz` 查询点，输出对应 SDF。
5. 测试阶段通过 `deepsdf.deep_sdf.mesh.create_mesh()` 在规则网格上查询 SDF 并导出网格。
6. 在生成网格后计算体积、Chamfer Distance、Precision/Recall/F1，并写入 CSV；若 checkpoint 中包含 `volume_head_state_dict`，还会同时输出 latent 回归得到的 `pred_volume_ml`。

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
- [`dataloaders/pointcloud_dataset.py`](/home/tianqi/corepp2/dataloaders/pointcloud_dataset.py)
  负责从 `complete/` 或 `partial/` 目录直接读取 `.ply`，并可从 DeepSDF latent matrix 或逐样本 `.pth` 中取监督信号；如果数据目录存在 `mapping.json` 且仓库根目录存在 `ground_truth.csv`，还会把对应真实体积作为 `volume_ml` 返回。

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

编码器输出 latent 后，当前 `train.py` 和 `test.py` 还会构造一个轻量体积回归头：

$$
\hat{v}_{\log} = h_{\phi}(\hat{\mathbf{z}})
$$

其中 `volume_head` 是 `Linear(latent_size, latent_size) -> ReLU -> Linear(latent_size, 1)`。它不替代 DeepSDF 解码器，只是从同一个 latent 上附加一个体积回归分支，用于缓解只靠 latent MSE 或 SDF 几何监督时体积容易塌缩到均值附近的问题。

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
6. 如果能通过 `mapping.json` 将 partial 文件映射回 `ground_truth.csv` 中的 complete 文件，则返回 `volume_ml` 作为体积监督标签。

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

当前代码中的第一阶段主损失写法与 [`train_deep_sdf.py`](/home/tianqi/corepp2/train_deep_sdf.py) 一致，可写为：

$$
\mathcal{L}_{\text{stage1}}
=
\frac{1}{N}
\sum_{i=1}^{N}
\left|
f_{\theta}(\mathbf{z}_j,\mathbf{x}_i)
-
s_i
\right|
\;+\;
\mathcal{L}_{\text{latent-reg}}
$$

其中：

- $\mathbf{x}_i \in \mathbb{R}^3$ 是采样的空间点；
- $s_i$ 是该点的 GT SDF；
- $\mathbf{z}_j$ 是当前训练样本 $j$ 的 latent code；
- $f_{\theta}(\mathbf{z}_j,\mathbf{x}_i)$ 是 DeepSDF 解码器输出；
- $N$ 是当前样本使用的 SDF 采样点数。

当前实现里的主重建项是 **L1 SDF 重建损失**，不是 MSE。代码中保留了 `enforce_minmax` 分支用于对 GT 和预测 SDF 做 `torch.clamp`，但 [`train_deep_sdf.py`](/home/tianqi/corepp2/train_deep_sdf.py) 当前把 `enforce_minmax` 固定为 `False`，所以默认训练路径不会执行该截断。

如果在 `specs.json` 中开启 `CodeRegularization`，则还会附加 latent 范数正则：

$$
\mathcal{L}_{\text{latent-reg}}
=
\lambda_{\text{code}}
\cdot
\min\!\left(1, \frac{e}{100}\right)
\cdot
\frac{1}{N}
\sum_{i=1}^{N}
\left\|
\mathbf{z}_j
\right\|_2
$$

其中 $e$ 是当前 epoch，代码里用 `min(1, epoch / 100)` 做 warm-up。

如果开启球面正则 `do_code_regularization_sphere`，则正则项替换为：

$$
\mathcal{L}_{\text{sphere-reg}}
=
\lambda_{\text{code}}
\cdot
\min\!\left(1, \frac{e}{100}\right)
\cdot
\frac{1}{N}
\sum_{i=1}^{N}
\left|
1 - \lVert \mathbf{z}_j \rVert_2
\right|
$$

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
- `LatentSpreadLoss`
  在开启 `supervised_3d` 且 `lambda_latent_spread > 0` 时，对齐预测 latent 与 GT latent 的逐维标准差，抑制 batch 内 latent collapse。
- `VolumeLoss`
  在 `lambda_volume > 0` 且样本包含 `volume_ml` 时，用 `volume_head(latent)` 直接回归体积。
- `SDFLoss`
  在开启 `3D_loss` 时，把编码器输出 latent 与规则体素网格拼接后送入解码器，直接监督 SDF 场。

其中最常见的主监督仍然是 `supervised_3d=True` 时的 latent 回归；当前草莓配置中额外叠加了 latent 分布展开和体积回归监督，用来减少预测 latent 与最终体积都向均值收缩的问题。

第二阶段的总损失可统一写成：

$$
\mathcal{L}_{\text{stage2}}
=
\lambda_{\text{att}} \mathcal{L}_{\text{att}}
+
\lambda_{\text{kl}} \mathcal{L}_{\text{kl}}
+
\lambda_{\text{super}} \mathcal{L}_{\text{super}}
+
\lambda_{\text{spread}} \mathcal{L}_{\text{spread}}
+
\lambda_{\text{volume}} \mathcal{L}_{\text{volume}}
+
\mathcal{L}_{\text{reg}}
+
\lambda_{\text{sdf}} \mathcal{L}_{\text{sdf}}
$$

其中只有被配置启用的项才会实际参与优化。

1. `SuperLoss`：latent 监督回归

$$
\mathcal{L}_{\text{super}}
=
\frac{1}{B}
\sum_{b=1}^{B}
\left\|
\hat{\mathbf{z}}_b - \mathbf{z}^{*}_b
\right\|_2^2
$$

这里 $\hat{\mathbf{z}}_b$ 是编码器预测 latent，$\mathbf{z}^{*}_b$ 是来自第一阶段 DeepSDF 的 GT latent。

2. `LatentSpreadLoss`：latent 分布展开约束

该项比较当前 batch 中预测 latent 和 GT latent 的逐维标准差：

$$
\mathcal{L}_{\text{spread}}
=
\frac{1}{D}
\sum_{d=1}^{D}
\left(
\sigma(\hat{\mathbf{z}}_{:,d})
-
\sigma(\mathbf{z}^{*}_{:,d})
\right)^2
$$

它对应 [`loss.py`](/home/tianqi/corepp2/loss.py) 中的 `LatentSpreadLoss`。这不是直接约束单个样本的 latent 取值，而是让一个 batch 内的预测 latent 保留接近 GT latent 的分布宽度，避免编码器把不同大小或不同形状的样本全部压到接近均值的位置。

3. `VolumeLoss`：体积回归监督

当前训练会先从编码器 latent 预测对数体积：

$$
\hat{u}_b = h_{\phi}(\hat{\mathbf{z}}_b),
\qquad
u^{*}_b = \log(1 + v^{*}_b)
$$

其中 $v^{*}_b$ 是数据集返回的真实毫升体积 `volume_ml`。损失由两部分组成：

$$
\mathcal{L}_{\text{volume}}
=
(1-\alpha)
\operatorname{SmoothL1}(\hat{u}, u^{*})
+
\alpha
\frac{1}{B}
\sum_{b=1}^{B}
\frac{|\operatorname{expm1}(\hat{u}_b)-v^{*}_b|}
{|v^{*}_b|+\epsilon}
$$

其中 $\alpha$ 对应配置项 `volume_loss_relative_weight`。第一项在 `log1p(volume_ml)` 空间里做稳定回归，第二项在毫升空间里惩罚相对误差；总项再由 `lambda_volume` 加权加入训练目标。

4. `RegLatentLoss`：latent 单位球约束

$$
\mathcal{L}_{\text{reg}}
=
\lambda_{\text{reg}}
\cdot
\frac{1}{B}
\sum_{b=1}^{B}
\left|
1 - \lVert \hat{\mathbf{z}}_b \rVert_2
\right|
$$

它对应 [`loss.py`](/home/tianqi/corepp2/loss.py) 中的 `RegLatentLoss`，本质上约束预测 latent 的模长接近 1。

5. `SDFLoss`：基于解码器输出的 3D 场监督

代码中会先把预测 SDF 截断到 $[-\tau, \tau]$，再只保留有效体素权重不为 0 且位于窄带内的点（`|s| < 1`），最后做对数变换后的 L1 损失：

$$
\tilde{s} = \operatorname{sign}(s)\log(1 + |s|)
$$

$$
\mathcal{L}_{\text{sdf}}
=
\frac{1}{|\Omega|}
\sum_{i \in \Omega}
\left|
\tilde{\hat{s}}_i - \tilde{s}_i
\right|
$$

其中 $\Omega = \{ i \mid w_i \neq 0,\ |s_i| < 1 \}$，$\hat{s}_i$ 是解码器预测值，$s_i$ 是目标 SDF，$w_i$ 是 TSDF/占据权重。

如果配置 `loss_type == "weighted"`，代码会改用 `SDFLoss_new`。它在同样的窄带和对数变换基础上，额外按 `exp(-alpha * |target|)` 提高近表面 SDF 样本权重，`alpha` 对应配置项 `sdf_alpha`。

6. `KLDivLoss`：batch latent 分布匹配

设当前 batch 预测 latent 的经验均值和协方差分别为：

$$
\boldsymbol{\mu}_b = \frac{1}{B}\sum_{i=1}^{B}\hat{\mathbf{z}}_i,
\qquad
\Sigma_b = \frac{1}{B}\sum_{i=1}^{B}(\hat{\mathbf{z}}_i-\boldsymbol{\mu}_b)(\hat{\mathbf{z}}_i-\boldsymbol{\mu}_b)^{\top}
$$

则代码中构造两个高斯分布
$q = \mathcal{N}(\boldsymbol{\mu}_b, \Sigma_b + 0.001I)$
与
$p = \mathcal{N}(\boldsymbol{\mu}_{\text{target}}, \Sigma_{\text{target}})$，
并最小化：

$$
\mathcal{L}_{\text{kl}} = D_{\mathrm{KL}}(q \parallel p)
$$

7. `AttRepLoss`：同类吸引、异类排斥

对 batch 内任意两个 latent，先计算欧氏距离
$d_{ij} = \lVert \hat{\mathbf{z}}_i - \hat{\mathbf{z}}_j \rVert_2$。
若二者属于同一 `fruit_id`，则目标标签为 $+1$；否则为 $-1$。代码调用 `HingeEmbeddingLoss(margin=\delta)`，因此该项可写为：

$$
\mathcal{L}_{\text{att}}
=
\sum_{i}\sum_{j}
\ell_{\text{hinge}}(d_{ij}, y_{ij})
$$

其中

$$
\ell_{\text{hinge}}(d, y)=
\begin{cases}
d, & y=1 \\
\max(0, \delta - d), & y=-1
\end{cases}
$$

这表示同类样本距离越小越好，异类样本至少要被推开到 margin $\delta$ 之外。

### 5.4 验证与模型保存

训练过程每隔 `validation_frequency` 做一次验证。当前验证指标不是直接看网格质量，而是：

- 在验证集上计算预测 latent 与 GT latent 的 MSE；
- 将其记为 `Mean Validation Latent MSE`；
- 如果验证样本包含 `volume_ml`，同时计算 `Mean Validation Volume Loss`；
- 如果 latent MSE 与 volume loss 都是有限值，best model 选择用 `latent MSE + lambda_volume * volume loss`；否则退化为只用 latent MSE。

模型保存由 [`utils.py`](/home/tianqi/corepp2/utils.py) 的 `save_model()` 负责，存储内容包括：

- `encoder_state_dict`
- `decoder_state_dict`
- `volume_head_state_dict`
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
2. 实例化编码器、DeepSDF 解码器与 `volume_head`。
3. 编码器输出 latent。
4. 如果 checkpoint 包含 `volume_head_state_dict`，则从 `volume_head(latent)` 预测 `pred_volume_ml`；如果配置 `calibrate_volume_head_on_val=True`，还会先用验证集拟合一个线性校准。
5. 调用 `deepsdf.deep_sdf.mesh.create_mesh()` 生成三角网格。
6. 把 latent 另外保存为 `.pth`，便于后续分析。
7. 计算体积和几何指标。
8. 将结果持续写入 CSV。

当前实现中，网格默认输出到固定目录：

- `/home/tianqi/corepp2/logs/strawberry/output`

这属于脚本级硬编码，不是通用工作区抽象。

网格生成的实际实现位于 [`deepsdf/deep_sdf/mesh.py`](/home/tianqi/corepp2/deepsdf/deep_sdf/mesh.py)。当前 `create_mesh()` 使用固定采样范围 `[-3.0, 3.0]^3`，通过 `skimage.measure.marching_cubes()` 提取零等值面；写出 PLY 前会尝试过滤明显贴近采样边界的伪连通域，并执行 Laplacian 平滑。

需要区分两种体积输出：

- `pred_volume_ml`
  来自 `volume_head` 对 latent 的直接回归，属于学习到的体积估计。
- `mesh_volume_ml`
  来自生成网格的几何体积计算，属于重建几何后的后处理估计。

## 7. 体积与评估指标

### 7.1 体积估计

当前体积估计在 [`test.py`](/home/tianqi/corepp2/test.py) 中由 `_compute_volume_ml()` 完成：

1. 先清理重复点、重复面和退化面；
2. 直接对当前网格顶点计算 `ConvexHull(vertices).volume`；
3. 按 `volume_unit` 把结果解释为 `cm`、`mm` 或 `m` 对应的体积单位；
4. 默认情况下再乘以 `volume_scale_factor`，其默认值是 `1.0`。

需要特别说明：

- 当前 CSV 中同时可能包含 `pred_volume_ml` 和 `mesh_volume_ml`；前者来自训练时新增的体积回归头，后者来自网格几何计算。
- `_compute_volume_ml()` 当前不调用 `mesh.get_volume()`，因此不会根据网格是否 watertight 切换计算方式。
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
- 训练阶段已经新增了从 latent 直接回归体积的 `volume_head`，体积不再只依赖测试阶段的网格后处理估计。
- 测试阶段的体积与指标计算已经集中在 `test.py`，并且比旧文档描述更贴近“直接从生成网格估计”。

如果后续还要继续维护这份文档，建议优先以以下文件为准：

- [`train.py`](/home/tianqi/corepp2/train.py)
- [`test.py`](/home/tianqi/corepp2/test.py)
- [`dataloaders/pointcloud_dataset.py`](/home/tianqi/corepp2/dataloaders/pointcloud_dataset.py)
- [`networks/pointnext.py`](/home/tianqi/corepp2/networks/pointnext.py)
- [`utils.py`](/home/tianqi/corepp2/utils.py)
