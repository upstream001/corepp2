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

