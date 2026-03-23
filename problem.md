1.为什么原test.py中sdf2mesh_cuda不管用？
查看 utils.py里原作者写的 sdf2mesh_cuda函数第 154 行，你会发现它的主要操作是 pcd_gpu.compute_convex_hull()。这意味着它仅仅是把网络预测出 SDF < 0（即在物体内部）的那些空间散点收集起来，然后像套塑料薄膜一样，从最外围紧紧包裹住所有的点。虽然 DeepSDF 网络成功学到了草莓的形状，但深度神经网络不是完美的。在远离草莓的计算空间（也就是 [-]1 到 [+]1 边界框的八个角落及其边缘附近）难免会出现一两个被错误预测为负数（<0）的游离废点。
2.

## 2026-03-24 Latent Collapse Analysis

在 `conda` 的 `corepp` 环境中，对 `20260312_dataset` 的 `val split` 做了一次对齐分析。分析对象限定为同时具备以下三项的 28 个样本：
- Encoder 可直接从点云前向得到预测 latent；
- `deepsdf/experiments/20260312_dataset/Reconstructions/500/Codes/complete/*.pth` 中存在对应的真值 latent；
- `ground_truth.csv` 与 `data/20260312_dataset/mapping.json` 能提供真实体积标签。

分析结果如下：
- 对齐样本数：28
- 真实体积均值 / 标准差：`19.130327 / 3.951213`
- 预测 latent 平均每维方差：`2.887441e-06`
- 真值 latent 平均每维方差：`2.395137e-02`
- 方差比 `pred / true`：`0.000121`
- 预测 latent 总方差 trace：`9.239811e-05`
- 真值 latent 总方差 trace：`7.664440e-01`
- trace 比 `pred / true`：`0.000121`
- 预测 latent 到质心平均距离：`0.009174`
- 真值 latent 到质心平均距离：`0.872293`
- 质心距离比 `pred / true`：`0.010517`
- 预测 latent norm 均值 / 标准差：`0.644269 / 0.001875`
- 真值 latent norm 均值 / 标准差：`0.968491 / 0.073393`
- `corr(pred latent norm, volume)`：`0.183985`
- `corr(true latent norm, volume)`：`-0.413052`
- 预测 latent 各维与体积相关绝对值均值 / 中位数 / 最大值：`0.309455 / 0.279537 / 0.772615`
- 真值 latent 各维与体积相关绝对值均值 / 中位数 / 最大值：`0.374584 / 0.314174 / 0.914575`

结论：
- 当前问题不是 `grid_density` 主导，而是 Encoder 输出的 latent 分布发生了严重收缩。
- 预测 latent 的总体离散程度只有真值 latent 的约 `0.0121%`，已经是明显的 latent collapse。
- 这会直接导致 Decoder 对不同样本解出接近平均形状，从而让预测体积集中在一个很窄的区间内。
- 预测 latent 中并非完全没有体积信息，但动态范围被压得太小，不足以支撑体积的真实变化幅度。
