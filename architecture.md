# CoRe++ 纯点云网络架构 (PointNeXt + DeepSDF)

本文档基于当前代码库的实现，详细描述了用于草莓三维重建的端到端框架。整个架构被明确切分为两部分：一是基于 **PointNeXt** 的点云编码器 (Encoder)，二是基于 **DeepSDF** 的隐式表面解码器 (Decoder)。

## 1. 整体架构流程

1. **输入阶段**: 
   - 模型从局部或补全的点云数据中采样得到固定点数（如 2048 个点）。
   - 输入张量尺寸: `(Batch_Size, 3, N)`。
2. **点云特征编码 (Point Cloud Encoder)**:
   - 由 `PointNeXtEncoder` 构成。
   - 输入: 归一化的 3D 点云集合。
   - 输出: 一个紧凑的低维全局隐式特征向量 `Latent Code` (如 32 维)，它隐式表达了水果的全局几何拓扑特征。
3. **隐式空间解码 (SDF Decoder)**:
   - 由 DeepSDF `Decoder` 结构组成。
   - 输入: 上一步生成的 32 维 `Latent Code` 加上任意 3D 查询点的查询坐标 `(x, y, z)`。
   - 输出: 一个标量 `SDF (Signed Distance Function)` 值，表示该点距离物体表面的符号距离（表面内为负数，表面外为正数，表面处为零）。
4. **后聚合与曲面重构 (Post Process)**:
   - 调用官方统一的 `deepsdf.deep_sdf.mesh.create_mesh()` 接口在归一化空间内密集采样形成三维网格。
   - 依赖其内置的 `Marching Cubes` 算法提取 `SDF = 0.0` 的零等值面。由于数据集加载阶段引入了针对性的点云动态去中心化（Centering）和缩放归一化（Scaling），故在此网格生成步骤中会同步执行网格点的逆变换（Inverse Space Mapping），物理无损地映射回现实空间的准确位置。此举彻底移除了之前自研分块处理和额外使用 Taubin 平滑所引入的毛刺畸变（Artifacts）。

---

## 2. 编码器 (Point Cloud Encoder - PointNeXt)

源代码参考路径: `networks/pointnext.py`。
PointNeXt 是一种相比原始 PointNet++ 引入反演残差和更优规范化函数的网络。当前工程的 `PointNeXtEncoder` 实现了一套高效点云提取网络：

### 网络骨干组成：
1. **Stem (先验嵌入层)**
   - 包含一层由 `SharedMLP1d` 构成的序列映射：利用 1x1 连续卷积核将初始的三坐标 `(x, y, z)` 提升为一个多通道高维特征（如 `width=48` 维）。
2. **Stage 1 (第一阶段下采样集抽象 - SA1)**
   - **SetAbstraction (SA)**: 寻找最远点采样 (FPS) 出 `npoint = 512` 个中心点，经 KNN 查询找到周边 `nsample = 24` 个近邻邻居，组成新的局部簇；通过 SharedMLP2d 提取深层特征。
   - **InvResMLP**: 应用含有 GroupNorm 归一化的 **反向残差网络(Inverse Residual MLP)**（两层深度）继续优化第一阶段特征。特征通道翻倍扩宽到 `width * 2`（即 96 维）。
3. **Stage 2 (第二阶段下采样集抽象 - SA2)**
   - 再次利用 FPS 从 512 个点进一步采样核心架构 `128` 个核心中枢点，同样的搜罗 24 个邻居并用特征聚合层叠加特征。
   - 输入第二级的 InvResMLP 层（依旧是两层深度），输出被激增为 `width * 4` 维的张量（即 192 维）。
4. **Global Pooling (特征汇聚层池化)**
   - 舍弃了传统的只求其一的方法，对点云的所有维度同时做自适应的 **Max Pooling** 与 **Average Pooling**，接着将两部分 `192` 维的精华特征在通道轴向 `Cat` (拼接)成一个长达 `384` 维的全局点云状态向量。
5. **Head (全连接聚合投影计算头)**
   - `Linear (384) -> LayerNorm -> ReLU -> Dropout (0.05) -> Linear (512) -> LayerNorm -> ReLU -> Dropout (0.05) -> Linear (256) -> 最终维度(如 32 维)`
   - 末尾输出的结果就是一个干净的隐式编码 `Latent Code`。

---

## 3. 解码器 (Implicit SDF Decoder - DeepSDF)

源代码参考路径: `deepsdf/networks/deep_sdf_decoder.py` 及其超参文件 `specs.json`。
SDF Decoder 采用经典的自动解码网络（AutoDecoder / MLP架构）。整个隐式网络的参数由 DeepSDF 预训练好并被冻结，只为了被 Encoder 的推理输出对标利用。

### 网络骨干组成：
- 输入维度: `Latent Size + 3` (32维隐特征向量 + 1个三维空间点的 `[x,y,z]`)。
- **全连接深度 MLP**:
  网络由连续 8 层的线性隐全连接层 (Linear Layer) 构成。每层的通道映射宽幅（隐藏层单元数）均高达 `512` 维。
- **跳跃连接 (Skip Connection)**:
  为避免深层网络的梯度消失（且基于 DeepSDF 与 NeRF 一致的框架优势），在结构设置 `latent_in=4`：这意味着第 4 层后，初始自带的输入向量会与当前的中间层输出再度做一次 `Concat`（特征级联穿透），将底层信息强制补救到深层感知域里。
- **网络层封装规范**:
  每一层的线性卷积都受控于 `weight_norm` (权重规范化)，并在多层之中加入了丢失概率 `dropout_prob = 0.2` 放水避免局部隐式过拟合。使用的激活函数为原教旨经典的 `ReLU`。

## 4. 关键特性与体积评估对齐机制

**端到端的协同作用与尺度对齐**：
- **动态坐标归一化对齐**：在数据加载端内部会实时对输入的测试点云执行去中心化操作（平移消抵 `center`），接着按其最大分布距离和边缘安全边距（例如 `normalization_margin=1.1`）计算出标量 `scale`，动态将实际测量目标等比重缩放并锁死进网络特化感知的深层球包空间内（约束在 `[-0.5, 0.5]` 基准内）。这确保了传入点云网的坐标分布与训练集毫无误差。
- **无缝网格恢复**：PointNeXt 预测出的 Latent Vector 等价充当 DeepSDF Decoder 的特征空间定位器。调用官方高保真的批量网格推理流（例如设置 `128³ -> 约合 210万` 采样块推断隐式曲面）后，它会将解开的坐标点列经过相反向的 `-offset` 和 `1/scale` 被动重塑还原为原本空间原状下的物理坐标尺度。
- **体积推断 (Volume Estimation):** 为了防止极个别不完全水密或凹坑破损对于体积计算接口的限制，最终框架底层代码层面统一使用极其稳定的 `scipy.spatial.ConvexHull`（凸包体积测算）计算单体包围体数据。结合得出的等价三维网格虚拟空内容，辅以乘上配置文件中测绘所得的 `normalization_scale`（基于真实设备相机视场映射出的总体重塑缩放系数，例如 `45.54`）常数的三次方，便能推演并在统计报表精确推导出具现实物理工程学指导意义的果实用体积（例如 `19.72 ml (毫升)`）。
