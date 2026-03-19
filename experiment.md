# 当前已完成实验记录

本文档只记录目前能从仓库代码、配置文件、checkpoint 和输出结果中直接确认的实验，不补写无法从现有文件系统验证的内容。

## 1. 实验主线

当前仓库里的实验基本分为两段：

1. DeepSDF 预训练
   在不同数据集版本上训练隐式解码器，并同步学习样本 latent code。
2. 编码器监督学习
   使用 DeepSDF 训练出的 latent code 作为监督，训练一个点云编码器，做到输入点云后直接预测 latent，再调用 DeepSDF 解码器完成网格重建。

## 2. 已落地的 DeepSDF 预训练实验

仓库中目前可以确认已经完成并保存到 `500 epoch` 的 DeepSDF 预训练实验有 4 组：

### 2.1 `20260301_dataset`

- 实验目录：
  [`deepsdf/experiments/20260301_dataset/specs.json`](/home/tianqi/corepp2/deepsdf/experiments/20260301_dataset/specs.json)
- 数据目录：
  `/home/tianqi/corepp2/data/20260301_dataset`
- split 规模：
  train `224`，val `28`，test `28`
- 训练状态：
  `ModelParameters/500.pth`、`LatentCodes/500.pth`、`OptimizerParameters/500.pth` 已存在
- 结构设置：
  latent dim `32`，8 层 `512` 维 MLP，`dropout_prob=0.2`，`weight_norm=true`
- 训练轮数：
  `NumEpochs = 500`

### 2.2 `20260301_dataset_aug`

- 实验目录：
  [`deepsdf/experiments/20260301_dataset_aug/specs.json`](/home/tianqi/corepp2/deepsdf/experiments/20260301_dataset_aug/specs.json)
- 数据目录：
  `/home/tianqi/corepp2/data/20260301_dataset_aug`
- split 规模：
  train `448`，val `56`，test `56`
- 训练状态：
  `500 epoch` 的 decoder、latent 和 optimizer checkpoint 已存在
- 目的：
  与未增强版本对照，验证数据增强对 DeepSDF latent space 和后续编码器监督的影响

### 2.3 `20260312_dataset`

- 实验目录：
  [`deepsdf/experiments/20260312_dataset/specs.json`](/home/tianqi/corepp2/deepsdf/experiments/20260312_dataset/specs.json)
- 数据目录：
  `/home/tianqi/corepp2/data/20260312_dataset`
- split 规模：
  train `224`，val `28`，test `28`
- 训练状态：
  `500 epoch` 的 decoder、latent 和 optimizer checkpoint 已存在
- 备注：
  这是当前草莓实验最直接对应的一套原始数据版本

### 2.4 `20260312_dataset_aug`

- 实验目录：
  [`deepsdf/experiments/20260312_dataset_aug/specs.json`](/home/tianqi/corepp2/deepsdf/experiments/20260312_dataset_aug/specs.json)
- 数据目录：
  `/home/tianqi/corepp2/data/20260312_dataset_aug`
- split 规模：
  train `448`，val `56`，test `56`
- 训练状态：
  `500 epoch` 的 decoder、latent 和 optimizer checkpoint 已存在
- 目的：
  与 `20260312_dataset` 对照，验证增强数据是否能提升重建或编码器监督效果

## 3. 已存在但当前仓库中未看到完整训练产物的实验配置

### 3.1 `strawberry` DeepSDF 配置

- 配置文件：
  [`deepsdf/experiments/strawberry/specs.json`](/home/tianqi/corepp2/deepsdf/experiments/strawberry/specs.json)
- split 规模：
  train `3136`，val `392`，test `392`
- 计划设置：
  `NumEpochs = 100`
- 当前仓库状态：
  只看到 `specs.json` 和 split 文件，没有看到对应的 `ModelParameters/*.pth` 或 `LatentCodes/*.pth`

因此，`strawberry` 这组 DeepSDF 更像是已经配置好，但训练产物没有保存在当前仓库中，或者尚未在这个工作区完成训练。

## 4. 编码器训练实验

目前能直接确认已经训练完成的一组编码器实验，是 `logs/strawberry` 下的点云编码器训练。

### 4.1 当前主训练配置

配置文件：
[`configs/strawberry.json`](/home/tianqi/corepp2/configs/strawberry.json)

可确认的设置如下：

- 数据目录：
  `/home/tianqi/corepp2/data/20260312_dataset/`
- 编码器：
  `pointnext`
- 输入点数：
  `2048`
- batch size：
  `4`
- epoch：
  `100`
- learning rate：
  `1e-4`
- 主监督：
  `supervised_3d = true`
- 关闭的附加损失：
  `kl_divergence = false`
  `contrastive = false`
  `reg_latent = false`
  `3D_loss = false`
- 验证频率：
  每 `10` 个 epoch
- 保存频率：
  每 `10` 个 epoch

这说明当前主实验不是联合优化复杂损失，而是一个相对干净的 baseline：

- 输入纯点云；
- 使用 PointNeXt 编码器；
- 用 DeepSDF latent code 做 MSE 监督；
- 不额外启用 SDF loss 或对比损失。

### 4.2 已生成的编码器权重

当前已确认存在：

- [`logs/strawberry/checkpoints/_strawberry_best_model.pt`](/home/tianqi/corepp2/logs/strawberry/checkpoints/_strawberry_best_model.pt)
- [`logs/strawberry/checkpoints/_strawberry_checkpoint.pt`](/home/tianqi/corepp2/logs/strawberry/checkpoints/_strawberry_checkpoint.pt)
- [`logs/strawberry/checkpoints/_strawberry_final_model.pt`](/home/tianqi/corepp2/logs/strawberry/checkpoints/_strawberry_final_model.pt)

checkpoint 元信息显示：

- 三个文件的 `epoch` 都是 `99`
- 文件中同时保存了：
  - `encoder_state_dict`
  - `decoder_state_dict`
  - `optimizer_state_dict`
  - `loss`

这说明这组编码器训练已经完整跑完 `100` 个 epoch。

### 4.3 训练日志

当前仓库下可见多份 TensorBoard event 文件：

- 位置：
  [`logs/strawberry/runs`](/home/tianqi/corepp2/logs/strawberry/runs)

说明这组训练至少被重复运行过多次，存在多轮试验或多次中断重跑。

## 5. 当前已保存的推理/测试结果

### 5.1 网格输出

当前可以确认已有一批编码器推理后生成的网格：

- 目录：
  [`logs/strawberry/output`](/home/tianqi/corepp2/logs/strawberry/output)
- 文件范围：
  `00252.ply` 到 `00279.ply`
- 样本数：
  `28`

这与 `20260312_dataset` 的 test split 大小 `28` 一致，说明当前保存下来的这批结果很可能就是在该 test split 上跑出的编码器推理网格。

### 5.2 单阈值指标结果

结果文件：
[`shape_completion_results.csv`](/home/tianqi/corepp2/shape_completion_results.csv)

当前可直接统计出的结果：

- 样本数：
  `28`
- frame 范围：
  `00252` 到 `00279`
- 平均体积：
  `17.379809 ml`
- 平均 Chamfer Distance：
  `0.044110`
- 平均 Precision：
  `0.150000`
- 平均 Recall：
  `9.196429`
- 平均 F1：
  `0.314286`

第一条样本记录：

- `00252`
- `mesh_volume_ml = 16.645374`
- `chamfer_distance = 0.04669`
- `precision = 0.1`
- `recall = 7.1`
- `f1 = 0.3`

最后一条样本记录：

- `00279`
- `mesh_volume_ml = 17.157624`
- `chamfer_distance = 0.04004`
- `precision = 0.2`
- `recall = 11.4`
- `f1 = 0.4`

### 5.3 多阈值指标结果

结果文件：
[`shape_completion_results_multi_threshold.csv`](/home/tianqi/corepp2/shape_completion_results_multi_threshold.csv)

当前统计均值如下：

- `precision_t0p005 = 0.150000`
- `recall_t0p005 = 9.196429`
- `f1_t0p005 = 0.314286`
- `precision_t0p01 = 1.246429`
- `recall_t0p01 = 18.617857`
- `f1_t0p01 = 2.342857`
- `precision_t0p02 = 8.332143`
- `recall_t0p02 = 34.875000`
- `f1_t0p02 = 13.439286`
- `precision_t0p03 = 21.742857`
- `recall_t0p03 = 48.496429`
- `f1_t0p03 = 29.992857`
- `precision_t0p05 = 53.221429`
- `recall_t0p05 = 70.060714`
- `f1_t0p05 = 60.428571`

这些结果说明：

- 在非常严格的 `0.005` 阈值下，当前网格与 GT 的几何吻合度仍然偏弱；
- 阈值放宽后，Precision/Recall/F1 上升明显；
- 当前实验更接近“形状整体恢复可用”，而不是“高精度表面贴合”。

## 6. 批量扫描 checkpoint 的实验脚本

仓库里还保留了两份面向旧版 `potato` 实验的批处理脚本：

- [`run_scripts_reconstruct.sh`](/home/tianqi/corepp2/run_scripts_reconstruct.sh)
- [`run_scripts_metrics.sh`](/home/tianqi/corepp2/run_scripts_metrics.sh)

它们的作用是：

- 遍历 `10` 到 `1000` 的 decoder checkpoint；
- 在验证集上做重建；
- 对每个 checkpoint 计算指标；
- 用于筛选最佳 DeepSDF decoder。

虽然当前主要工作重心已经转到草莓点云实验，但这两份脚本说明仓库里确实保留了“先扫 decoder checkpoint，再确定最佳下游监督源”的实验思路。

## 7. 目前可以明确归纳出的实验结论

从现有文件系统状态看，目前已经做过的实验可以概括为：

1. 在 `20260301_dataset` / `20260301_dataset_aug` / `20260312_dataset` / `20260312_dataset_aug` 上分别完成了 `500 epoch` 的 DeepSDF 预训练。
2. 做过“原始数据 vs augmented 数据”的成对对照实验设计，至少在 DeepSDF 训练层面已经落地。
3. 在 `20260312_dataset` 上使用 `pointnext` 编码器做了纯点云到 latent 的监督学习，训练完成 `100 epoch`，并保存了 best/final checkpoint。
4. 已经对一个 `28` 个样本的测试集做过编码器推理，并导出了网格与两份 CSV 指标结果。
5. 现阶段公开保存在仓库中的结果，最完整的是这套 `20260312_dataset + pointnext + DeepSDF latent supervision` 的实验链路。

## 8. 当前文档未写入的内容

下面这些内容目前没有足够证据，不在本文档中写成既定事实：

- 四组 DeepSDF 预训练之间谁的指标最好；
- `20260312_dataset_aug` 是否已经被用于后续编码器训练；
- `strawberry` 的 DeepSDF 是否曾在别的机器上完整训练过；
- 其他配置文件例如 `foldnet.json`、`pce.json`、`pce_large.json`、`super3d_*.json` 是否已经在当前工作区真正跑过完整实验。

如果后续你要把本文档升级成“论文实验记录”或“组会汇报版本”，建议再补三类信息：

- 每组实验对应的实际运行命令；
- best checkpoint 选择依据；
- 不同编码器、不同数据版本之间的对比表。
