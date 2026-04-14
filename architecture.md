# 架构：草莓点云到 DeepSDF 网格

本文档只描述模型架构本身，即论文中“Method / Architecture”应包含的内容，不涉及脚本入口、目录组织、运行命令、日志路径或实验管理细节。

## 1. 任务定义

目标是从草莓点云预测其对应的形状隐变量，再通过隐式表面解码器恢复连续 SDF 场，并提取三角网格。

整体推理路径为：

```text
输入点云 P
  -> PointNeXt encoder
  -> latent code z
  -> DeepSDF decoder
  -> continuous SDF field f(z, x)
  -> marching cubes
  -> reconstructed mesh
```

其中：

- `P = {p_i}_{i=1}^N` 表示输入点云，`p_i \in R^3`。
- `z \in R^32` 表示形状隐变量。
- `f(z, x)` 表示查询点 `x \in R^3` 处的符号距离值。

## 2. 总体框架

整个方法采用两阶段架构：

1. 先训练一个 DeepSDF decoder，建立“latent code -> 连续 SDF 场”的生成映射。
2. 再训练一个点云编码器，将输入点云直接回归到 DeepSDF latent space。

因此，最终系统可以写成：

```text
z = E(P)
s = F(z, x)
M = MC(F(z, ·))
```

其中：

- `E` 是点云编码器；
- `F` 是 DeepSDF 解码器；
- `MC` 表示 marching cubes 网格提取。

该设计的关键点是：编码器不直接预测体素、点集或网格，而是预测一个可被隐式解码器解释的低维形状表示 `z`。

## 3. 点云编码器

### 3.1 输入与输出

编码器输入为点云 `P \in R^{N x 3}`，当前使用固定点数 `N = 2048`。点云经过中心对齐后送入网络，输出一个 32 维 latent code：

```text
E: R^(2048 x 3) -> R^32
```

### 3.2 主干结构

编码器采用 PointNeXt 风格层级式点云网络，由一个 stem、两个 set abstraction stage、若干局部残差块以及全局聚合头组成。

结构概要如下：

```text
Input XYZ, N=2048
  -> Stem Shared MLP, 3 -> 48
  -> SA1, npoint=512, k=24, 48 -> 96
  -> InvResMLP blocks
  -> SA2, npoint=128, k=24, 96 -> 192
  -> InvResMLP blocks
  -> Global Max/Avg Pool, 192 -> 384
  -> Head MLP, 384 -> 512 -> 256 -> 32
```

### 3.3 Stem

输入点坐标首先通过共享 MLP 进行逐点特征提取：

```text
[B, 3, 2048]
  -> SharedMLP1d(3  -> 48)
  -> SharedMLP1d(48 -> 48)
  -> [B, 48, 2048]
```

这里的共享 MLP 本质上是 `1 x 1` 卷积加归一化与激活函数，用于把原始几何坐标映射到初始点特征空间。

### 3.4 Set Abstraction 层

每个 set abstraction 模块由三部分组成：

1. 使用 FPS 选择一组中心点。
2. 对每个中心点使用 KNN 收集局部邻域。
3. 对局部相对坐标与邻域特征做共享 MLP 聚合，再通过池化得到中心点特征。

其计算流程可以写为：

```text
输入:
  xyz:      [B, N, 3]
  features: [B, C, N]

1. farthest point sampling -> 中心点
2. k-nearest neighbors     -> 局部邻域
3. grouped_xyz - center_xyz
4. concat(relative_xyz, grouped_features)
5. SharedMLP2d
6. neighborhood max pooling
7. 与中心点 skip 分支相加
```

这种设计同时保留了：

- 稀疏采样后的全局覆盖性；
- 局部邻域内的几何关系；
- 中心点特征的残差传递。

### 3.5 Stage 1

第一层级将点数从 2048 下采样到 512，并把通道数从 48 提升到 96：

```text
SA1:
  npoint = 512
  nsample = 24
  input channels = 48 + 3
  output channels = 96
```

随后接两个 Inverted Residual MLP block，在不改变点数与通道数的前提下增强局部表达能力：

```text
[B, 96, 512]
  -> InvResMLP(96)
  -> InvResMLP(96)
  -> [B, 96, 512]
```

### 3.6 Stage 2

第二层级继续将点数从 512 下采样到 128，并把通道数从 96 提升到 192：

```text
SA2:
  npoint = 128
  nsample = 24
  input channels = 96 + 3
  output channels = 192
```

之后同样接两个 Inverted Residual MLP block：

```text
[B, 192, 128]
  -> InvResMLP(192)
  -> InvResMLP(192)
  -> [B, 192, 128]
```

### 3.7 全局聚合与 latent 回归

最后对点级特征进行全局最大池化与全局平均池化，并将两者拼接：

```text
[B, 192, 128]
  -> global max pool -> [B, 192]
  -> global avg pool -> [B, 192]
  -> concat          -> [B, 384]
```

再通过一个三层全连接头回归最终 latent：

```text
[B, 384]
  -> Linear(384 -> 512) + LayerNorm + ReLU + Dropout
  -> Linear(512 -> 256) + LayerNorm + ReLU + Dropout
  -> Linear(256 -> 32)
  -> pred_latent [B, 32]
```

因此，点云编码器的核心作用可以概括为：

```text
局部几何编码
  -> 多尺度特征聚合
  -> 全局形状摘要
  -> 低维形状隐变量
```

## 4. Inverted Residual MLP Block

每个 `InvResMLP` block 是一个逐点残差变换模块，其形式为：

```text
x
  -> Conv1d(C -> 4C, kernel=1)
  -> GroupNorm
  -> ReLU
  -> Conv1d(4C -> C, kernel=1)
  -> GroupNorm
  -> Residual Add
  -> ReLU
```

它不改变点数，也不改变通道数，只在当前层级内提升特征表达能力。相比直接堆叠普通 MLP，这种残差设计更稳定，也更适合层级式点云骨干网络。

## 5. DeepSDF 解码器

### 5.1 输入输出形式

DeepSDF decoder 是一个条件隐式函数网络。给定 latent code `z` 和三维查询点 `x`，网络输出该点的 SDF 值：

```text
F: (z, x) -> s
```

其中：

- `z \in R^32`
- `x \in R^3`
- `s \in R`

因此单次查询的输入维度为：

```text
32 + 3 = 35
```

### 5.2 MLP 结构

解码器主体是一个宽度为 512 的 8 层 MLP，并带有 latent skip connection。结构如下：

```text
input [z, x]: 35
  -> lin0: 35  -> 512
  -> lin1: 512 -> 512
  -> lin2: 512 -> 512
  -> lin3: 512 -> 477
  -> concat original input: 477 + 35 = 512
  -> lin4: 512 -> 512
  -> lin5: 512 -> 512
  -> lin6: 512 -> 512
  -> lin7: 512 -> 512
  -> lin8: 512 -> 1
  -> Tanh
```

其中第 4 层前重新拼接原始输入 `[z, x]`，形成一次中层 skip：

```text
h_4 = concat(h_3, [z, x])
```

这种设计使中层特征仍能直接访问：

- 全局形状编码 `z`；
- 查询点坐标 `x`。

### 5.3 架构作用

该解码器学习的是一个连续隐式表面表示，而不是离散网格模板。因此同一个 latent code `z` 可以在任意空间位置 `x` 上被查询，形成连续 SDF 场：

```text
f_z(x) = F(z, x)
```

随后取零等值面：

```text
{x | F(z, x) = 0}
```

即可恢复对应三维形状的表面。

## 6. 两阶段学习机制

### 6.1 第一阶段：学习形状先验

第一阶段训练 DeepSDF decoder，并为每个训练形状优化一个对应的 latent code。这样，模型建立起：

```text
latent code <-> shape geometry
```

之间的映射关系。经过这一阶段后，解码器已经具备从低维隐变量生成草莓几何体的能力。

### 6.2 第二阶段：学习点云到 latent 的映射

第二阶段固定 DeepSDF decoder，仅训练点云编码器 `E`，使其输出的 `pred_latent` 接近目标 latent。于是整体模型被分解为：

```text
Point cloud -> Encoder -> latent
latent + query points -> Decoder -> SDF
SDF field -> Marching Cubes -> Mesh
```

这种分解比“直接从点云回归网格”更稳定，因为编码器只需学习投影到一个已经成形的几何潜空间，而不必独自承担完整的表面生成任务。

## 7. 训练目标

当前架构中，编码器训练的核心目标是 latent 监督。总损失可写为：

```text
L = lambda_super * L_super
  + lambda_latent_spread * L_spread
```

其中：

- `L_super` 为预测 latent 与目标 latent 之间的均方误差；
- `L_spread` 用于约束预测 latent 的批内分布，避免所有样本坍塌到相近位置。

对应地，当前架构强调：

- 以 latent space 对齐作为主监督；
- 通过分布约束保持形状可分性；
- 由固定的 DeepSDF decoder 负责几何生成。

这意味着模型优化的重点不是直接拟合某个显式几何表示，而是让编码器输出落在一个可被 decoder 正确解释的形状流形上。

## 8. 几何重建过程

在推理阶段，对任意输入点云 `P`，先得到 latent：

```text
z = E(P)
```

再在三维网格上查询解码器得到离散 SDF 体：

```text
s_ijk = F(z, x_ijk)
```

其中 `x_ijk` 表示规则体素网格上的查询坐标。最后通过 marching cubes 在零等值面处提取网格：

```text
M = MC({x | F(z, x) = 0})
```

因此，最终网格并不是编码器直接输出的，而是由：

1. 编码器给出形状 latent；
2. 解码器生成连续隐式场；
3. 等值面提取恢复显式表面。

## 9. 架构总结

该方法本质上是一个“点云编码器 + 隐式形状解码器”的组合框架：

- PointNeXt encoder 负责从不规则点云中提取多尺度几何特征，并回归全局形状 latent。
- DeepSDF decoder 负责把 latent 与空间坐标映射为连续 SDF。
- marching cubes 负责把隐式场转换为显式三角网格。

从方法论上看，该架构把“感知”与“生成”分开处理：

- 编码器解决“从观测点云理解形状”的问题；
- 解码器解决“从低维形状表示恢复连续表面”的问题。

这种分工使模型既保留了点云网络对局部几何的建模能力，又继承了隐式表示在连续表面重建上的优势。
