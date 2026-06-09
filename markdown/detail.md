# 最后一步重建方法说明

本文档说明当前系统在测试阶段最后一步是如何从 encoder 输出的 latent code 重建 mesh，并进一步得到 `mesh_volume_ml` 的。

## 1. 当前使用的方法

当前最后一步重建使用的是：

```text
DeepSDF implicit field sampling + marching cubes mesh extraction
```

也就是：

```text
PointNeXt encoder 输出 latent
    -> 固定的 DeepSDF decoder 查询三维空间 SDF
    -> 在规则 3D 网格上得到 SDF volume
    -> marching cubes 提取 SDF=0 的等值面
    -> 输出 .ply 三角网格
    -> 基于 mesh 顶点计算体积
```

它不是直接预测 mesh，也不是直接预测体积。最终体积 `mesh_volume_ml` 来自生成出来的三角网格。

## 2. 代码入口

测试阶段主入口是：

- [`test.py`](/home/tianqi/corepp2/test.py)

核心调用位置：

```python
deepsdf.deep_sdf.mesh.create_mesh(
    decoder,
    latent,
    mesh_filename,
    start=time.time(),
    N=grid_density,
    max_batch=int(2 ** 18)
)
```

实际 mesh 生成函数在：

- [`deepsdf/deep_sdf/mesh.py`](/home/tianqi/corepp2/deepsdf/deep_sdf/mesh.py)

核心函数：

```python
create_mesh(decoder, latent_vec, filename, start, N=256, max_batch=32 ** 3)
```

当前配置中：

```json
"grid_density": 30
```

因此当前 `test.py` 默认用：

```text
N = 30
```

也就是在 `30 x 30 x 30` 的规则网格上查询 SDF。

## 3. 输入 latent 从哪里来

测试时输入点云先进入 PointNeXt encoder：

```text
partial_pcd: [B, 2048, 3]
    -> permute
encoder_input: [B, 3, 2048]
    -> PointNeXtEncoder
latent: [B, 32]
```

这个 `latent` 是 DeepSDF latent code，不是体积值，也不是 mesh 顶点。

随后 `latent` 被送入固定的 DeepSDF decoder。decoder 查询形式是：

```text
decoder([latent, x, y, z]) -> SDF
```

其中：

```text
latent: 32 维
xyz:     3 维
输入:   35 维
输出:    1 维 SDF 标量
```

## 4. SDF 网格采样

`create_mesh()` 首先构造一个规则三维网格。

当前代码中网格范围是：

```text
x, y, z ∈ [-3.0, 3.0]
```

对应源码参数：

```python
grid_range = 6.0
voxel_origin = [-3.0, -3.0, -3.0]
voxel_size = grid_range / (N - 1)
```

当 `N=30` 时：

```text
voxel_size = 6.0 / 29 ≈ 0.2069
```

然后代码生成 `N^3` 个查询点：

```text
30^3 = 27000 个 xyz 查询点
```

每个查询点都会和同一个 encoder latent 拼接：

```text
[latent, xyz] -> DeepSDF decoder -> SDF
```

为了避免一次性占用过多显存，查询按 batch 分块执行：

```python
max_batch = int(2 ** 18)
```

当前 `30^3 = 27000` 小于 `2^18 = 262144`，所以通常一个 batch 就能完成。

## 5. Marching Cubes 提取 mesh

DeepSDF decoder 对规则网格中每个点输出 SDF 后，会得到一个三维 SDF volume：

```text
sdf_values: [N, N, N]
```

然后调用：

```python
skimage.measure.marching_cubes(
    numpy_3d_sdf_tensor,
    level=0.0,
    spacing=[voxel_size] * 3
)
```

这里的 `level=0.0` 表示提取 SDF 的零等值面：

```text
SDF(x, y, z) = 0
```

在 DeepSDF 中：

- `SDF < 0` 通常表示物体内部。
- `SDF > 0` 通常表示物体外部。
- `SDF = 0` 是物体表面。

因此 marching cubes 得到的三角面片就是当前预测草莓的表面 mesh。

## 6. 坐标变换

`marching_cubes()` 输出的顶点一开始是在 voxel 坐标系中，随后代码将其映射回当前 DeepSDF 查询空间：

```python
mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]
```

因此输出 mesh 的坐标范围仍然对应：

```text
[-3.0, 3.0]^3
```

当前主流程中没有再把 mesh 乘回某个单位球缩放因子。也就是说，当前假设 DeepSDF decoder 和输入点云都已经处在一致的物理尺度坐标中。

`test.py` 中保留了一个可选开关：

```json
"remap_mesh_to_gt_bbox": false
```

默认关闭。只有手动打开时，才会把 `[-3,3]^3` 空间里的 mesh 映射到 GT bbox。

## 7. 后处理

marching cubes 得到初始 mesh 后，当前代码做了两个后处理步骤。

第一步是连通域过滤：

```text
cluster_connected_triangles()
```

目的：

```text
去掉明显贴近采样边界的伪 SDF 零等值面囊泡
```

当前判断逻辑是：如果某个连通域在 x/y/z 三个方向都几乎跨满整个 `[-3,3]` 采样盒，即范围大于 `5.8`，就认为它可能是边界伪壳。

如果存在多个有效连通域，则保留三角面数量最多的那个。

第二步是 Laplacian 平滑：

```python
mesh.filter_smooth_laplacian(
    number_of_iterations=10,
    lambda_filter=0.5
)
```

目的：

```text
减轻 DeepSDF 或 marching cubes 导致的局部表面波纹
```

最后 mesh 被写成 `.ply` 文件：

```text
logs/strawberry/output/<frame_id>.ply
```

## 8. 体积计算方法

`test.py` 重新读取刚刚生成的 `.ply`：

```python
mesh = o3d.io.read_triangle_mesh(mesh_ply_file)
mesh.compute_vertex_normals()
```

随后用 `_compute_volume_ml()` 计算 `mesh_volume_ml`。

当前体积计算不是直接使用 Open3D 的 watertight mesh volume，而是：

```python
ConvexHull(np.asarray(mesh.vertices)).volume
```

也就是基于 mesh 顶点的凸包体积。

单位换算规则：

```text
volume_unit = "cm" 时: 1 cm^3 = 1 mL
volume_unit = "mm" 时: mm^3 / 1000 = mL
volume_unit = "m"  时: m^3 * 1,000,000 = mL
```

当前草莓主配置默认：

```text
volume_unit = "cm"
volume_scale_factor = 1.0
```

所以最终：

```text
mesh_volume_ml = ConvexHull(mesh_vertices).volume
```

## 9. 和 volume head 的区别

当前最终架构已经关闭：

```json
"lambda_volume": 0.0
```

因此测试输出中不再使用 `pred_volume_ml` 作为最终指标。

两者区别是：

```text
pred_volume_ml:
  latent -> volume_head -> 直接回归体积
  不一定和 decoder 生成的 mesh 几何一致

mesh_volume_ml:
  latent -> DeepSDF decoder -> marching cubes mesh -> ConvexHull volume
  来自最终重建几何
```

当前要评估重建结果，应优先看：

```text
mesh_volume_ml
volume_mae_ml
volume_rmse_ml
volume_mape_percent
volume_r2
Chamfer / Precision / Recall / F1
```

## 10. 当前方法的关键影响

这个最后一步重建方法的优点是：

- 保留了 DeepSDF 的连续隐式表面表达。
- encoder 只需预测 32 维 latent，不需要直接生成复杂 mesh。
- mesh 和体积都来自同一个 decoder 几何结果，指标更一致。

需要注意的限制：

- `grid_density=30` 较低，mesh 表面会比较粗；提高 `grid_density` 可以提升细节，但会增加计算量。
- marching cubes 只提取 `SDF=0` 等值面，如果 decoder 输出整体偏移，会直接影响 mesh 尺寸。
- 当前体积用凸包估计，适合草莓这类近似凸形物体；如果物体有明显凹陷，凸包体积会偏大。
- 如果输入点云尺度和 DeepSDF 训练尺度不一致，最后 mesh 体积会系统性偏差。

## 11. PointNeXt encoder 当前详细架构

当前最终架构中使用的 encoder 是：

- [`networks/pointnext.py`](/home/tianqi/corepp2/networks/pointnext.py)
- `PointNeXtEncoder`

它的任务不是直接预测 mesh，也不是直接预测体积，而是把输入点云编码成 DeepSDF latent：

```text
输入 partial/complete point cloud
    -> PointNeXtEncoder
    -> pred_latent, shape [B, 32]
    -> DeepSDF decoder
```

当前默认参数来自 `build_pointnext_encoder()`：

```text
in_channels = 3
out_channels = latent_size = 32
width = pointnext_width = 48
nsample = pointnext_nsample = 24
dropout = pointnext_dropout = 0.05
```

当前输入点数来自配置：

```text
input_size = 2048
```

因此输入张量形状是：

```text
DataLoader partial_pcd: [B, 2048, 3]
train.py/test.py permute 后: [B, 3, 2048]
```

## 12. PointNeXt 总体层级

当前 PointNeXt encoder 的总结构是：

```text
Input [B, 3, 2048]
  -> Stem
       SharedMLP1d(3  -> 48)
       SharedMLP1d(48 -> 48)
       ResidualMLP1d(48)
  -> SA1
       FPS: 2048 -> 384 points
       KNN: k=24
       SharedMLP2d(51 -> 96)
       SharedMLP2d(96 -> 96)
       SharedMLP2d(96 -> 96)
       max over local neighborhood
       skip Conv1d(48 -> 96)
  -> Stage1
       InvResMLP(96), expansion=4
       ResidualMLP1d(96)
  -> SA2
       FPS: 384 -> 96 points
       KNN: k=24
       SharedMLP2d(99 -> 192)
       SharedMLP2d(192 -> 192)
       SharedMLP2d(192 -> 192)
       max over local neighborhood
       skip Conv1d(96 -> 192)
  -> Stage2
       InvResMLP(192), expansion=4
       ResidualMLP1d(192)
  -> SA3
       FPS: 96 -> 24 points
       KNN: k=24
       SharedMLP2d(195 -> 384)
       SharedMLP2d(384 -> 384)
       SharedMLP2d(384 -> 384)
       max over local neighborhood
       skip Conv1d(192 -> 384)
  -> Stage3
       InvResMLP(384), expansion=4
       ResidualMLP1d(384)
  -> Feature fusion
       feature_fusion = stage3
       fused = features3: [B, 384, 24]
  -> Global pooling
       adaptive max pool: [B, 384]
       adaptive avg pool: [B, 384]
       concat: [B, 768]
  -> Head
       Linear(768 -> 512)
       LayerNorm(512)
       ReLU
       Dropout(0.05)
       Linear(512 -> 256)
       LayerNorm(256)
       ReLU
       Dropout(0.05)
       Linear(256 -> 32)
  -> pred_latent [B, 32]
```

## 13. 基础模块

### 13.1 GroupNorm 规则

源码中的 `group_norm(channels)` 会自动选择不超过 8 的最大可整除 group 数：

```text
48  channels -> GroupNorm(8, 48)
96  channels -> GroupNorm(8, 96)
192 channels -> GroupNorm(8, 192)
384 channels -> GroupNorm(8, 384)
768 channels -> GroupNorm(8, 768)
```

这个设计避免了 batch size 较小时 BatchNorm 不稳定的问题。

### 13.2 SharedMLP1d

`SharedMLP1d(in_channels, out_channels)` 是：

```text
Conv1d(in_channels -> out_channels, kernel_size=1, bias=False)
GroupNorm(out_channels)
ReLU(inplace=True)
```

它对每个点独立做 1x1 卷积，不改变点数。

输入输出形状：

```text
[B, C_in, N] -> [B, C_out, N]
```

### 13.3 SharedMLP2d

`SharedMLP2d(in_channels, out_channels)` 是：

```text
Conv2d(in_channels -> out_channels, kernel_size=1, bias=False)
GroupNorm(out_channels)
ReLU(inplace=True)
```

它用于局部邻域特征。输入输出形状：

```text
[B, C_in, npoint, k] -> [B, C_out, npoint, k]
```

其中：

```text
npoint = FPS 采样后的中心点数量
k = 每个中心点的 KNN 邻居数量
```

### 13.4 InvResMLP

`InvResMLP(channels, expansion=4)` 是一个 residual block：

```text
输入 x: [B, C, N]

主分支:
  Conv1d(C -> 4C, kernel_size=1, bias=False)
  GroupNorm(4C)
  ReLU(inplace=True)
  Conv1d(4C -> C, kernel_size=1, bias=False)
  GroupNorm(C)

残差:
  x + main_branch(x)

输出:
  ReLU(x + main_branch(x))
```

它不改变点数，也不改变通道数，只增强每个点的特征表达。

### 13.5 ResidualMLP1d

`ResidualMLP1d(channels)` 的形式是：

```text
输入 x: [B, C, N]

主分支:
  Conv1d(C -> C, kernel_size=1, bias=False)
  GroupNorm(C)
  ReLU(inplace=True)
  Conv1d(C -> C, kernel_size=1, bias=False)
  GroupNorm(C)

残差:
  x + main_branch(x)

输出:
  ReLU(x + main_branch(x))
```

它和 `InvResMLP` 的区别是不会先做通道扩张，而是直接在同一通道数上进一步细化逐点特征。

## 14. 输入与坐标整理

`PointNeXtEncoder.forward(x)` 的第一步：

```python
xyz = x.transpose(1, 2).contiguous()
features = self.stem(x)
```

输入：

```text
x: [B, 3, 2048]
```

得到：

```text
xyz: [B, 2048, 3]
```

这里：

- `xyz` 用于 FPS 采样和 KNN 搜索。
- `features` 是每个点的可学习特征。

## 15. Stem 逐层结构

Stem 定义：

```python
self.stem = nn.Sequential(
    SharedMLP1d(in_channels, width),
    SharedMLP1d(width, width),
    ResidualMLP1d(width),
)
```

当前 `in_channels=3`，`width=48`，所以展开为：

```text
Stem input:
  x: [B, 3, 2048]

Layer stem.0:
  Conv1d(3 -> 48, kernel=1, bias=False)
  GroupNorm(8, 48)
  ReLU
  output: [B, 48, 2048]

Layer stem.1:
  Conv1d(48 -> 48, kernel=1, bias=False)
  GroupNorm(8, 48)
  ReLU
  output: [B, 48, 2048]

Layer stem.2:
  ResidualMLP1d(48)
  output: [B, 48, 2048]
```

Stem 输出：

```text
features: [B, 48, 2048]
xyz:      [B, 2048, 3]
```

## 16. SA1 逐层结构

SA1 定义：

```python
self.sa1 = SetAbstraction(
    in_channels=48,
    out_channels=96,
    npoint=384,
    nsample=24
)
```

### 16.1 FPS 采样

输入：

```text
xyz:      [B, 2048, 3]
features: [B, 48, 2048]
```

FPS：

```text
farthest_point_sample(xyz, npoint=384)
```

输出中心点索引：

```text
fps_idx: [B, 384]
```

取中心点坐标：

```text
new_xyz = index_points(xyz, fps_idx)
new_xyz: [B, 384, 3]
```

取中心点原始特征：

```text
center_features:
  features.transpose(1, 2): [B, 2048, 48]
  index_points(..., fps_idx): [B, 384, 48]
  transpose back: [B, 48, 384]
```

### 16.2 KNN 局部邻域

对每个中心点找 24 个邻居：

```text
group_idx = knn_point(nsample=24, xyz, new_xyz)
group_idx: [B, 384, 24]
```

邻域点坐标：

```text
grouped_xyz = index_points(xyz, group_idx)
grouped_xyz: [B, 384, 24, 3]
```

转成局部相对坐标：

```text
grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)
grouped_xyz: [B, 384, 24, 3]
```

邻域点特征：

```text
features.transpose(1, 2): [B, 2048, 48]
index_points(..., group_idx): [B, 384, 24, 48]
permute: [B, 48, 384, 24]
```

拼接相对坐标和邻域特征：

```text
grouped_xyz.permute: [B, 3, 384, 24]
grouped_features:    [B, 48, 384, 24]
concat:              [B, 51, 384, 24]
```

这里 51 来自：

```text
48 feature channels + 3 relative xyz channels = 51
```

### 16.3 SA1 局部 MLP

SA1 的 `self.mlp`：

```text
SharedMLP2d(51 -> 96)
SharedMLP2d(96 -> 96)
SharedMLP2d(96 -> 96)
```

逐层展开：

```text
Input: [B, 51, 384, 24]

sa1.mlp.0:
  Conv2d(51 -> 96, kernel=1, bias=False)
  GroupNorm(8, 96)
  ReLU
  output: [B, 96, 384, 24]

sa1.mlp.1:
  Conv2d(96 -> 96, kernel=1, bias=False)
  GroupNorm(8, 96)
  ReLU
  output: [B, 96, 384, 24]

sa1.mlp.2:
  Conv2d(96 -> 96, kernel=1, bias=False)
  GroupNorm(8, 96)
  ReLU
  output: [B, 96, 384, 24]
```

对每个中心点的 24 个邻居做 max pooling：

```text
aggregated = mlp_output.max(dim=-1)[0]
aggregated: [B, 96, 384]
```

### 16.4 SA1 skip connection

skip 分支：

```text
center_features: [B, 48, 384]

Conv1d(48 -> 96, kernel=1, bias=False)
GroupNorm(8, 96)

shortcut: [B, 96, 384]
```

融合：

```text
new_features = ReLU(aggregated + shortcut)
new_features: [B, 96, 384]
```

SA1 输出：

```text
xyz:      [B, 384, 3]
features: [B, 96, 384]
```

## 17. Stage1 逐层结构

Stage1 定义：

```python
self.stage1 = nn.Sequential(
    InvResMLP(96),
    ResidualMLP1d(96)
)
```

输入：

```text
features: [B, 96, 384]
```

### 17.1 Stage1 block 0

```text
Input: [B, 96, 384]

Conv1d(96 -> 384, kernel=1, bias=False)
GroupNorm(8, 384)
ReLU
Conv1d(384 -> 96, kernel=1, bias=False)
GroupNorm(8, 96)
Add residual
ReLU

Output: [B, 96, 384]
```

### 17.2 Stage1 block 1

第二个 block 改为同通道数的 `ResidualMLP1d`：

```text
Input:  [B, 96, 384]
96 -> 96 -> 96
Output: [B, 96, 384]
```

Stage1 输出：

```text
xyz:      [B, 384, 3]
features: [B, 96, 384]
```

## 18. SA2 逐层结构

SA2 定义：

```python
self.sa2 = SetAbstraction(
    in_channels=96,
    out_channels=192,
    npoint=96,
    nsample=24
)
```

### 18.1 FPS 采样

输入：

```text
xyz:      [B, 384, 3]
features: [B, 96, 384]
```

FPS：

```text
farthest_point_sample(xyz, npoint=96)
fps_idx: [B, 96]
```

中心点坐标：

```text
new_xyz: [B, 96, 3]
```

中心点特征：

```text
center_features: [B, 96, 96]
```

### 18.2 KNN 局部邻域

```text
group_idx = knn_point(nsample=24, xyz, new_xyz)
group_idx: [B, 96, 24]
```

邻域相对坐标：

```text
grouped_xyz: [B, 96, 24, 3]
grouped_xyz - new_xyz.unsqueeze(2): [B, 96, 24, 3]
```

邻域特征：

```text
grouped_features: [B, 96, 96, 24]
```

拼接：

```text
relative xyz:     [B, 3, 96, 24]
grouped_features: [B, 96, 96, 24]
concat:           [B, 99, 96, 24]
```

这里 99 来自：

```text
96 feature channels + 3 relative xyz channels = 99
```

### 18.3 SA2 局部 MLP

SA2 的 `self.mlp`：

```text
SharedMLP2d(99 -> 192)
SharedMLP2d(192 -> 192)
SharedMLP2d(192 -> 192)
```

逐层展开：

```text
Input: [B, 99, 96, 24]

sa2.mlp.0:
  Conv2d(99 -> 192, kernel=1, bias=False)
  GroupNorm(8, 192)
  ReLU
  output: [B, 192, 96, 24]

sa2.mlp.1:
  Conv2d(192 -> 192, kernel=1, bias=False)
  GroupNorm(8, 192)
  ReLU
  output: [B, 192, 96, 24]

sa2.mlp.2:
  Conv2d(192 -> 192, kernel=1, bias=False)
  GroupNorm(8, 192)
  ReLU
  output: [B, 192, 96, 24]
```

邻域 max pooling：

```text
aggregated = mlp_output.max(dim=-1)[0]
aggregated: [B, 192, 96]
```

### 18.4 SA2 skip connection

skip 分支：

```text
center_features: [B, 96, 96]

Conv1d(96 -> 192, kernel=1, bias=False)
GroupNorm(8, 192)

shortcut: [B, 192, 96]
```

融合：

```text
new_features = ReLU(aggregated + shortcut)
new_features: [B, 192, 96]
```

SA2 输出：

```text
xyz:      [B, 96, 3]
features: [B, 192, 96]
```

## 19. Stage2 逐层结构

Stage2 定义：

```python
self.stage2 = nn.Sequential(
    InvResMLP(192),
    ResidualMLP1d(192)
)
```

输入：

```text
features: [B, 192, 96]
```

### 19.1 Stage2 block 0

```text
Input: [B, 192, 96]

Conv1d(192 -> 768, kernel=1, bias=False)
GroupNorm(8, 768)
ReLU
Conv1d(768 -> 192, kernel=1, bias=False)
GroupNorm(8, 192)
Add residual
ReLU

Output: [B, 192, 96]
```

### 19.2 Stage2 block 1

第二个 block 改为 `ResidualMLP1d(192)`：

```text
Input:  [B, 192, 96]
192 -> 192 -> 192
Output: [B, 192, 96]
```

Stage2 输出：

```text
xyz:      [B, 96, 3]
features: [B, 192, 96]
```

## 20. SA3 与 Stage3 逐层结构

SA3 定义：

```python
self.sa3 = SetAbstraction(
    in_channels=192,
    out_channels=384,
    npoint=24,
    nsample=24
)
```

输入：

```text
xyz:      [B, 96, 3]
features: [B, 192, 96]
```

经过 FPS 后：

```text
fps_idx: [B, 24]
new_xyz: [B, 24, 3]
center_features: [B, 192, 24]
```

KNN 邻域与拼接后：

```text
group_idx: [B, 24, 24]
grouped_xyz: [B, 24, 24, 3]
grouped_features: [B, 192, 24, 24]
concat(relative_xyz, grouped_features): [B, 195, 24, 24]
```

SA3 的局部 MLP：

```text
SharedMLP2d(195 -> 384)
SharedMLP2d(384 -> 384)
SharedMLP2d(384 -> 384)
```

聚合后输出：

```text
aggregated: [B, 384, 24]
shortcut:   [B, 384, 24]
features3:  [B, 384, 24]
```

Stage3 定义：

```python
self.stage3 = nn.Sequential(
    InvResMLP(384),
    ResidualMLP1d(384)
)
```

Stage3 输出仍保持：

```text
xyz3:      [B, 24, 3]
features3: [B, 384, 24]
```

## 21. Global pooling 逐层结构

当前使用的配置是：

```text
pointnext_feature_fusion = stage3
pointnext_global_pool = max_avg
```

因此不会执行多尺度拼接，也不会把 `features2` 或 `features1` nearest upsample 到同一长度。最终全局特征直接取自：

```text
fused = features3: [B, 384, 24]
```

max pooling：

```text
max_feat = adaptive_max_pool1d(fused, 1).squeeze(-1)
max_feat: [B, 384]
```

avg pooling：

```text
avg_feat = adaptive_avg_pool1d(fused, 1).squeeze(-1)
avg_feat: [B, 384]
```

拼接：

```text
global_feat = concat(max_feat, avg_feat)
global_feat: [B, 768]
```

这里：

- max pooling 更关注显著局部结构。
- avg pooling 更保留整体形状分布。
- 拼接后作为最终 latent head 的输入。

## 22. Head MLP 逐层结构

Head 定义：

```python
self.head = nn.Sequential(
    nn.Linear(head_in_dim, 512, bias=False),
    nn.LayerNorm(512),
    nn.ReLU(inplace=True),
    nn.Dropout(dropout),
    nn.Linear(512, 256, bias=False),
    nn.LayerNorm(256),
    nn.ReLU(inplace=True),
    nn.Dropout(dropout),
    nn.Linear(256, out_channels),
)
```

当前：

```text
fused_dim = width * 8 = 384
head_in_dim = fused_dim * 2 = 768
out_channels = 32
dropout = 0.05
```

逐层展开：

```text
Input global_feat:
  [B, 768]

head.0:
  Linear(768 -> 512, bias=False)
  output: [B, 512]

head.1:
  LayerNorm(512)
  output: [B, 512]

head.2:
  ReLU(inplace=True)
  output: [B, 512]

head.3:
  Dropout(p=0.05)
  output: [B, 512]

head.4:
  Linear(512 -> 256, bias=False)
  output: [B, 256]

head.5:
  LayerNorm(256)
  output: [B, 256]

head.6:
  ReLU(inplace=True)
  output: [B, 256]

head.7:
  Dropout(p=0.05)
  output: [B, 256]

head.8:
  Linear(256 -> 32, bias=True)
  output: [B, 32]
```

最终输出：

```text
pred_latent: [B, 32]
```

这个 32 维向量会直接作为 DeepSDF decoder 的 shape code。

## 23. PointNeXt 参数量估算

按当前默认 `width=48`、`latent_size=32`、`stage_depth=1`、`feature_fusion=stage3`、`global_pool=max_avg` 统计，真实模型参数量为：

```text
3,074,256
```

按模块拆开，数量级主要来自：

```text
Stem:
  Conv1d 3->48:    3 * 48 = 144
  GroupNorm 48:    48 gamma + 48 beta = 96
  Conv1d 48->48:   48 * 48 = 2304
  GroupNorm 48:    96
  ResidualMLP1d(48): 约 4.8K

SA1:
  MLP 51->96:      51 * 96 = 4896
  GN 96:           192
  MLP 96->96:      96 * 96 = 9216
  GN 96:           192
  MLP 96->96:      96 * 96 = 9216
  GN 96:           192
  skip 48->96:     48 * 96 = 4608
  skip GN 96:      192

Stage1:
  InvResMLP(96):        约 74.7K
  ResidualMLP1d(96):    约 18.8K

SA2:
  MLP 99->192:     99 * 192 = 19008
  GN 192:          384
  MLP 192->192:    192 * 192 = 36864
  GN 192:          384
  MLP 192->192:    192 * 192 = 36864
  GN 192:          384
  skip 96->192:    96 * 192 = 18432
  skip GN 192:     384

Stage2:
  InvResMLP(192):       约 296.8K
  ResidualMLP1d(192):   约 74.5K

SA3:
  MLP 195->384:    195 * 384 = 74880
  GN 384:          768
  MLP 384->384:    384 * 384 = 147456
  GN 384:          768
  MLP 384->384:    384 * 384 = 147456
  GN 384:          768
  skip 192->384:   192 * 384 = 73728
  skip GN 384:     768

Stage3:
  InvResMLP(384):       约 1.18M
  ResidualMLP1d(384):   约 296.4K

Head:
  Linear 768->512: 768 * 512 = 393216
  LayerNorm 512:   1024
  Linear 512->256: 512 * 256 = 131072
  LayerNorm 256:   512
  Linear 256->32:  256 * 32 + 32 = 8224
```

整体参数规模大约在：

```text
约 3.07M 参数
```

参数增加的主要原因是当前版本比旧文档多了：

- `ResidualMLP1d` stem；
- 第三层 `SA3 + Stage3`；
- `384 -> 768` 的 `max_avg` 全局描述子；
- 更宽的后段通道和更大的 head 输入。

## 24. PointNeXt 数据流总结

完整数据流可以压缩成下面这张表：

```text
阶段          输出点数  输出通道  输出形状
Input         2048     3         [B, 3, 2048]
Stem          2048     48        [B, 48, 2048]
SA1           384      96        [B, 96, 384]
Stage1        384      96        [B, 96, 384]
SA2           96       192       [B, 192, 96]
Stage2        96       192       [B, 192, 96]
SA3           24       384       [B, 384, 24]
Stage3        24       384       [B, 384, 24]
Max pool      1        384       [B, 384]
Avg pool      1        384       [B, 384]
Concat        1        768       [B, 768]
Head FC1      -        512       [B, 512]
Head FC2      -        256       [B, 256]
Head output   -        32        [B, 32]
```

最终：

```text
[B, 32] pred_latent
    -> DeepSDF decoder
    -> SDF field
    -> marching cubes mesh
    -> mesh_volume_ml
```

