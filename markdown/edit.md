# My CoRe++ (图一) vs CoRe++ 2 (图二)：重建策略差异与调整指南

## 1. 核心理论差异：局部优化 vs 全局归纳偏置

这两张图片的巨大差异，本质上是由于两边采用了**截然不同的 3D 重建范式**导致的结果：

### 论文原版流程 (图二, `corepp2` 走的执行路径)
**方法**：Auto-Decoder + 测试期优化 (Test-Time Optimization, 对应被执行的脚本 `reconstruct_deep_sdf.py`)
- **工作原理**：在测试阶段，给定一个残缺或者有噪声的点云，深度网络会初始化一个全零的隐变量 (Latent Code)，然后针对这颗草莓，进行成百上千次（例如 800 次）的反向传播梯度下降，强行去寻找一个能拟合这些离散点云 SDF 数值的特征向量。
- **为什么图二变得坑坑洼洼**：在寻找 Latent Code 的过程中，如果没有极其严苛的正则化（这点在 `corepp2` 代码中被设置为了 `l2reg=False`），优化过程会**严重过拟合**到局部点云的扫描噪声和由于 500 轮训练所带来的网络高频抖动上。模型被强行拉扯，脱离了“正常草莓”应当具备的光滑流形空间，生成了充满奇形怪状的囊泡、突刺和表面褶皱的“恶性表面”。

### my_corepp 的流程 (图一, `my_corepp` 走的执行路径)
**方法**：Encoder-Decoder + 端到端前向推断 (Amortized Inference, 对应你编写的脚本 `test_strawberry_pcd.py` 或最新的 `test.py`)
- **工作原理**：直接**跳过**了耗时且脆弱的测试期反推步骤！你引入了一个前置的特征提取网络（如 `DGCNN` 或 `PointNeXt`）。这个网络在之前的训练中看过了大量的草莓，已经形成了**强大的宏观形状先验（Shape Prior）**。
- **为什么图一整体轮廓圆润平滑**：当包含噪声或残缺的测试点云输入时，前向推断是一步到位的。特征编码器的网络权重就像一张滤网，由于它无法表达那些极度无规律的高频噪声，它会自动忽略点云中的局部瑕疵，强行将输入投影到“完美的草莓形状”所对应的隐空间坐标上。因此，它能够极大程度地对抗高频噪声干扰，保留完美的水滴流线型轮廓。
- **为什么图一有“等高线/阶梯状”波纹**：图一完美的形状表面上带有阶梯纹，仅仅是因为在 `my_corepp` 的 Mesh 提取设定中分辨率较低 (`N=128`)，并且 `convert_sdf_samples_to_ply` 没有附带后处理平滑滤波滤镜（如 Laplacian / Taubin Smoothing）所导致的。

---

## 2. 如何将 corepp2 向 my_corepp 进行修改与进化？

事实上，目前 `corepp2` 项目的代码基建和后处理是优于 `my_corepp` 的（它在 `test.py` 中集成了图一缺乏的平滑逻辑和连通域清理），图二之所以糟糕，单纯是因为在测试时**走错了运行路线**（错误地采用了论文中古老的逐样本优化方案提取测试结果）。

要让 `corepp2` 兼具图一的“完美大体型”加“无缝丝滑外壳”，你需要彻底中止在测试集上运行 `reconstruct_deep_sdf.py`，而是全面转向 `my_corepp` 开辟的端到端编码器体系。具体操作指南如下：

### 第一步：放弃使用 `reconstruct_deep_sdf.py` 提取最终重建网格
**不要再运行 `reconstruct_deep_sdf.py` 来观测测试/验证集的效果！**
该脚本目前仅适用于**准备训练数据**的中间阶段（用来寻找训练集点云所对应的 Latent 真值，供 Encoder 进行对齐学习）。它不是用来输出最终成果的。

### 第二步：在 `corepp2` 中坚持训练前向点云编码器 (Point Cloud Encoder)
确保在 `corepp2` 环境中已经完成了编码器的端到端训练。
```bash
# 在 corepp2 目录下，采用 PointNeXt 或 DGCNN 训练编码器
python train.py --cfg ./configs/strawberry.json --experiment ./deepsdf/experiments/strawberry
```
让前置网络学会如何从草莓点云中一眼“看”出完美平滑的隐向量特征结构。

### 第三步：使用 `corepp2` 的 `test.py` 直接发起测试重建 (代替 `my_corepp` 的 `test_strawberry_pcd.py`)
直接使用集成度极高的 `test.py`，这是 `my_corepp` 中 `test_strawberry_pcd.py` 的究极进化版：
```bash
# 利用 test.py 基于测试集进行前向传播重建与提取
python test.py --cfg ./configs/strawberry.json --experiment ./deepsdf/experiments/strawberry
```

### 第四步：结合 `my_corepp` 的启示，拉满 `corepp2` 的生成参数
图一的阶梯伪影可以通过 `corepp2` 内部的强大引擎来根除。你可以手动修改 `corepp2/configs/strawberry.json` 以及核对 `test.py` 的参数：
1. **调高分辨率（填平阶梯）**：将提取网格的 `grid_density`（或相对应的 `N` 解析度参数）从低精度的 128 改为 **`256`**。这会极大化平滑插值，消除图一的锯齿阶梯边缘。
2. **利用 `corepp2` 中自带的后处理滤波器**：在 `corepp2` 的体系里（无论是 `deep_sdf/mesh.py` 还是 `test.py` 重建中），现在已经加入了去除瑕疵的 Open3D 代码逻辑。当它与你高泛化性的 Point Encoder 结合时，将会得到工业级的表现：
   ```python
   # 确保执行了此类逻辑来消除伪影：
   mesh.remove_degenerate_triangles()
   mesh.remove_non_manifold_edges()
   mesh = mesh.filter_smooth_taubin(number_of_iterations=30)
   ```
   
### 💡 总结
你之所以感觉 `my_corepp` “没做完反而好”，是因为原论文 2019 年的设定（即 `Core++` 作者最开始采纳的机制）是一套没有图像/点云 Encoder 的 `Auto-Decoder` 探索性基线。
你半路引入并依靠的**点云特征提取层**，本质上将整个体系跨时代地升级成了 `Encoder-Decoder`。保持你 `my_corepp` 开创的这个大方向路线，继续在 `corepp2` 里用 `test.py` 走前向推断，你就能收获比原论文完美得多的草莓三维模型了。
