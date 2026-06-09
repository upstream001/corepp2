# 当前训练参数说明

本文档用于记录当前仓库中草莓点云版本的实际训练参数，主要依据以下来源整理：

- 配置文件：[configs/strawberry.json](/home/tianqi/corepp2/configs/strawberry.json:1)
- 编码器训练脚本：[train.py](/home/tianqi/corepp2/train.py:149)
- DeepSDF 预训练配置：[deepsdf/experiments/strawberry/specs.json](/home/tianqi/corepp2/deepsdf/experiments/strawberry/specs.json:1)
- 原论文 PDF：[High-throughput 3D shape completion of potato tubers on a harvester.pdf](/home/tianqi/corepp2/High-throughput%203D%20shape%20completion%20of%20potato%20tubers%20on%20a%20harvester.pdf)

需要注意：当前实现是“草莓 + 纯点云 + PointNeXt 编码器”版本，而原论文是“马铃薯 + RGB-D 图像 + 卷积编码器”版本。因此，只有部分训练超参数可以直接标记为与原论文一致，网络输入形式与编码器结构本身并不一致。

## 1. 当前任务设置

- 任务对象：草莓数据集，`species = strawberry`
- 数据根目录：`/home/tianqi/corepp2/data/20260331_dataset/`
- 编码器输入：纯点云 partial point cloud
- 编码器结构：`PointNeXt`
- 解码器结构：`DeepSDF decoder`
- 训练目标：将 partial point cloud 编码为 latent vector，并监督其逼近预训练 DeepSDF latent code

## 2. 当前编码器训练参数

### 2.1 数据与输入

- 配置文件：`configs/strawberry.json`
- 输入点数：`input_size = 2048`
- batch size：`4`
- 训练 epoch 数：`50`
- 初始学习率：`1e-4`
- 训练集 DataLoader：`shuffle=True`
- 训练集 DataLoader：`drop_last=True`
- 验证集 batch size：`1`

说明：

- `detection_input = mask`、`normalize_depth = false`、`depth_min = 230`、`depth_max = 350` 在当前纯点云流程中不生效，它们是为原 RGB-D 图像流程保留的兼容字段。

### 2.2 编码器结构参数

当前编码器为 `PointNeXt`，对应配置如下：

- `encoder = pointnext`
- `pointnext_width = 48`
- `pointnext_nsample = 24`
- `pointnext_sa1_npoint = 384`
- `pointnext_sa2_npoint = 96`
- `pointnext_sa3_npoint = 24`
- `pointnext_stage_depth = 1`
- `pointnext_expansion = 4`
- `pointnext_dropout = 0.05`

可写成论文中的结构描述：

- Stem 宽度设为 `48`
- 邻域聚合点数 `nsample = 24`
- 三层 Set Abstraction 下采样点数分别为 `384 / 96 / 24`
- 每个 stage 的残差深度为 `1`
- 倒残差膨胀倍率为 `4`
- Head dropout 为 `0.05`

### 2.3 优化器与学习率调度

由 [train.py](/home/tianqi/corepp2/train.py:265) 可知，当前编码器训练采用：

- 优化器：`Adam`
- weight decay：`1e-6`
- 学习率调度器：`ExponentialLR`
- 衰减系数：`gamma = 0.97`
- 调度频率：每个 epoch 结束后更新一次学习率

### 2.4 当前损失函数设置

当前训练脚本中的总损失由若干可选项组成，但在 `configs/strawberry.json` 下，实际生效的是下列部分：

- 主监督损失：latent MSE
- 权重：`lambda_super = 1.0`
- latent spread loss：开启
- 权重：`lambda_latent_spread = 1.0`

当前关闭的损失项：

- KL divergence：`false`
- contrastive loss：`false`
- rendering loss：`false`
- latent regularization：`false`
- 3D SDF loss：`false`
- volume head loss：`lambda_volume = 0.0`，因此不参与训练

因此，当前编码器训练的主要优化目标可以概括为：

1. 最小化预测 latent 与目标 DeepSDF latent 之间的 MSE。
2. 额外使用 latent spread loss 约束 batch 内 latent 分布不要过于塌缩。

### 2.5 验证与模型选择策略

- `validation_frequency = 10`
- `checkpoint_frequency = 10`
- `validate_mesh_volume = true`
- `selection_metric = latent_mse`
- `grid_density = 30`
- `threshold = 0.0007`
- `normalization_scale = 45.54`

说明：

- 虽然开启了 `validate_mesh_volume = true`，但当前 best checkpoint 的选择依据仍是 `latent_mse`，不是体积误差。
- `grid_density = 30` 主要用于需要通过 decoder 生成网格并计算体积时的离散化分辨率。

## 3. 当前 DeepSDF 预训练参数

当前编码器监督所依赖的 latent code 来自 `deepsdf/experiments/strawberry/specs.json` 中定义的 DeepSDF 预训练。

### 3.1 latent 与网络结构

- latent size：`CodeLength = 32`
- decoder 隐层维度：8 层，每层 `512`
- `dropout_prob = 0.2`
- `latent_in = [4]`
- `xyz_in_all = false`
- `use_tanh = false`
- `latent_dropout = false`
- `weight_norm = true`

### 3.2 DeepSDF 训练设置

- `NumEpochs = 100`
- `SnapshotFrequency = 100`
- 额外快照：`0, 500, 1000`
- 学习率调度：step schedule
- 初始学习率：`5e-4` 与 `1e-3` 两组 schedule
- 衰减间隔：每 `300` epoch
- 衰减因子：`0.5`
- `SamplesPerScene = 16384`
- `ScenesPerBatch = 64`
- `DataLoaderThreads = 16`
- `ClampingDistance = 0.1`
- `CodeRegularization = true`
- `CodeRegularizationLambda = 1e-4`
- `CodeBound = 1.0`

说明：

- `specs.json` 中 `NumEpochs = 100`，但额外快照含 `500` 和 `1000`，这说明该实验目录可能继承了更早期的 DeepSDF 设置模板。若论文中需要严格陈述“实际采用了哪个 checkpoint”，建议以你最终使用的 `--checkpoint_decoder` 为准单独写明。

## 4. 与原论文一致的参数

以下参数可明确标注为“与原论文一致”或“与原论文保持同一设定”。

### 4.1 编码器训练部分

- 编码器初始学习率 `1e-4`
  依据：论文写明 encoder training 的 initial learning rate 为 `1×10^-4`。
- 编码器优化器为 `Adam`
  依据：论文写明 encoder 使用 `Adam optimizer`。
- 编码器主监督损失为 latent MSE
  依据：论文写明 encoder 使用 predicted latent 与 target latent 之间的 `mean squared error loss`。

### 4.2 DeepSDF 预训练部分

- latent size 使用 `32`
  依据：论文指出原始研究中采用 latent size `32`，并且当前 `strawberry` 的 `CodeLength` 也为 `32`。
- DeepSDF 初始学习率 `5e-4`
  依据：论文写明 DeepSDF training started at learning rate `5×10^-4`。
- DeepSDF 学习率每 `300` epoch 衰减一次
  依据：论文明确写明 for every `300 epochs`, learning rate was reduced。
- DeepSDF 学习率衰减因子 `0.5`
  依据：当前 `specs.json` 与论文描述一致。
- DeepSDF 使用 `Adam`
  依据：论文明确写明 decoder training used `Adam optimizer`。

## 5. 与原论文不同的地方

以下内容不应写成“与原论文一致”，因为当前版本已经明显不同。

- 输入模态不同：当前是纯点云输入，原论文是 RGB-D 图像输入。
- 编码器结构不同：当前是 `PointNeXt`，原论文是 7 层卷积编码器。
- 输入尺寸不同：当前是 `2048` 个点，原论文是裁剪并填充后的 `304 × 304` RGB-D 图像。
- 编码器训练 epoch 不同：当前为 `50`，原论文为 `100`。
- 学习率调度不同：当前编码器使用 `ExponentialLR(gamma=0.97)`，原论文编码器使用“每 epoch 衰减 97%”的指数衰减描述，含义接近，但表述上应写为“保持相同衰减比例思想”，不要硬写成完全相同，除非你希望直接按代码实现表述。
- contrastive loss 不同：原论文 encoder loss 中包含 contrastive loss；当前 `configs/strawberry.json` 中 `contrastive = false`，即未启用。
- 验证最优模型准则不同：当前以 `latent_mse` 选 best checkpoint；原论文是在验证集上结合 decoder 推理效果来选择 encoder 权重。

## 6. 建议在论文中的写法

如果你要写“当前实验设置”，建议用下面这种表达：

“在当前草莓点云实验中，我们采用 PointNeXt 作为编码器、DeepSDF 作为解码器。输入为 2048 个点组成的 partial point cloud，batch size 设为 4，训练 50 个 epoch。编码器采用 Adam 优化器，初始学习率为 1e-4，并使用指数学习率衰减。训练时以预训练 DeepSDF latent code 作为监督信号，主损失函数为 latent MSE，同时加入权重为 1.0 的 latent spread loss。DeepSDF 侧的 latent size 设为 32，该设置与原论文保持一致。” 

如果你要写“与原论文一致的部分”，建议用下面这种表达：

“尽管本文采用了不同于原论文的纯点云 PointNeXt 编码器，部分核心训练超参数仍保持与原论文一致，包括编码器优化器使用 Adam、编码器初始学习率设为 1e-4，以及 DeepSDF latent size 设为 32。此外，DeepSDF 预训练阶段同样采用 Adam 优化器，并使用初始学习率 5e-4、每 300 个 epoch 衰减 0.5 的 step learning-rate schedule。” 

## 7. 引用依据

- 当前编码器配置：[configs/strawberry.json](/home/tianqi/corepp2/configs/strawberry.json:12)
- 编码器训练实现：[train.py](/home/tianqi/corepp2/train.py:180)
- 优化器与调度器：[train.py](/home/tianqi/corepp2/train.py:265)
- 验证与 best model 选择：[train.py](/home/tianqi/corepp2/train.py:412)
- DeepSDF 预训练配置：[deepsdf/experiments/strawberry/specs.json](/home/tianqi/corepp2/deepsdf/experiments/strawberry/specs.json:7)

