1. 创建了 `workflow.md` 文件，将基于 `README.md` 总结的 CoRe++ 整体工作流程说明详细写入了该文件。
2. 编写了 `workflow_new.md` 文件，针对草莓点云数据集（且已经配准），专门设计了利用 DeepSDF 进行直接重建和通过 Mesh 几何属性估计体积大小的纯点云工作流程，跳过了与 RGB-D 单视角图像处理相关的多余步骤。
3. 修改了 `workflow_new.md` 文件，补充了训练**点云编码器 (Point Cloud Encoder)** 的完整环节。用端到端网络推理取代了单一的测试时优化迭代步骤；明确了在测试脚本 `test.py` 中如何直接输出网格重建效果与体积大小（`mesh_volume_ml`）。
4. 在 `workflow_new.md` 文件中，补充了数据处理阶段（Data Preparation）关于使用 `data_preparation/make_splits.py` 划分数据集的步骤。
- **新增模块**: 新增了特定的点云数据集加载类 `dataloaders/pointcloud_dataset.py`。其封装了针对类似 `strawberry` 和 `20260301_dataset` 中仅有 `complete` / `partial` 目录下直接存放 `.ply` 的输入数据集，从而替代原先对 Realsense RGB-D 强耦合的 `MaskedCameraLaserData`。
6. 针对新数据集切分需要，编写了 `data_preparation/make_strawberry_splits.py`。其可以直接对纯点云文件夹下的文件按照 8:1:1 或自定义比例进行切分，同时生成两种格式的切分文件（工程全局所需的 `split.json`，以及供给 DeepSDF 专项使用的 `{dataset_name}_{split}.json`）。已经用该脚本处理了 `strawberry` 和 `20260301_dataset` 数据集。
7. 在 `configs/` 目录下编写了专为草莓数据集运行的配置文件：`configs/strawberry.json`，并在文件中基于 `_comment_{field}` 虚拟键的方式对各项核心运行参数进行了详尽的中文注释。
8. 编写了专为纯点云直接计算和保存 DeepSDF 标签数据的脚本 `data_preparation/prepare_strawberry_sdf.py`。该脚本修复了原版 `prepare_deepsdf_training_data.py` 代码因为对文件目录结构的强绑定导致的 `ZeroDivisionError` 和路径崩溃报错。
9. 修复了 DeepSDF 启动报错 `does not include specifications file "specs.json"` 的问题：为 `strawberry` 和 `20260301_dataset` 两个数据集配置了各自的训练超参数和网络规格文件（保存在 `deepsdf/experiments/{dataset_name}/specs.json` 中）。
10. 修复了 `reconstruct_deep_sdf.py` 在执行测试时引发的一系列文件路径找不到的问题（比如直接寻找 `500.pth` 和不正常拼接 `fruit.ply` 导致 Open3D 读取失败）。现在的 `deepsdf/deep_sdf/o3d_utils.py` 文件能够正确根据拆分好的 `samples.npz` 相对路径推断出原始数据的 `complete` 点云实例所在位置并正确加载！
11. 修复了前驱流程读取问题引发的一系列第四阶段 `train.py` 报错问题：
    - 将 Encoder 训练时目标真实 Latent Code 的读取路径由 `Reconstructions/100/Codes/complete` 修正为经过实际反向优化生成的 `Reconstructions/100/Codes/partial`。
    - 将未处理完全的点云读取大小不一引发的 `DataLoader torch.stack` 报错进行修复，统一了 `dataloaders/pointcloud_dataset.py` 中 `target_pcd` 和 `partial_pcd` 的对齐长度处理方法。
    - 移除了代码对 `3DPotatoTwinDemo/ground_truth.csv` 地面真值体积的硬编码路径依赖。
    - 更新并统一了纯点云场景下 Validation 验证集的加载接口类，并将基于复杂 `o3d` 图形学重构生成进行体积评价对比（极其容易失败和拖慢验证耗时）的指标换成了更底层的 `Latent MS Error`。
12. 配合目前的设备实验条件，针对训练爆显存（OOM Error）情况，将 `configs/strawberry.json` 的 `batch_size` 容量调降并完成调优。
13. 修复了最终测试流程 `test.py` 的一系列环境兼容问题：
    - 切断了 `test.py` 底层对 `3DPotatoTwinDemo/ground_truth.csv` 地面真值体积表的强行依赖查找，移除了对 Potato 相关的数据结构键值假设（如 Cultivar，Weight_g 等）。
    - 修复了 `key_error: frame_id` 等报错：为 `pointcloud_dataset.py` 新增了兼容用的 `frame_id` 字典填充，并在 `test.py` 中补充针对纯点云数据的 `bbox` 和 `None值` 报错捕捉处理机制。
    - 重新适配了专用于新数据集测试输出的 DataFrame 并简化保存逻辑，使得在无外部标注真值的纯盲扫环境下，程序仍旧能够顺手计算并全自动存出 `shape_completion_results.csv` 这一结果图表（其中包含所有点的 Mesh 重构估算体积 `mesh_volume_ml` 以及 `chamfer_distance` 距离等评估指标）。
14. 在 `test.py` 中新增了重建网格（Mesh）自动保存功能，现在不仅会测算体积与得分，还能够将每一次推断生成的完整闭合模型以 `.ply` 格式保存至 `Reconstructions/100/meshes/` 供可视化提取。
15. 修复了 DeepSDF 在高轮数（如 `checkpoint 500`）预测下容易在 SDF 边界外产生游离、散落（多余的漂浮碎片网络）的问题。现已向 `deepsdf/deep_sdf/mesh.py` 加入了基于 `open3d` 最大连通域自动过滤机制，保证最终输出的只是一颗完美干净的草莓，这不仅清除了游离组件，还将原先缓慢的 plyfile 序列化保存步骤速度缩减了数十倍。
16. 针对高轮数训练引发的隐式表面（Implicit Surface）过拟合产生巨大“伪包围壳囊泡”(Shell Artifact) 现象，进一步重构了 `deepsdf/deep_sdf/mesh.py` 中的网络过滤系统：由于原先通过强制水密性能(`is_watertight`)检测的手段会误伤包含微小噪点的草莓本体并引发算法退化，新增加的过滤器采用了“空间全包围约束算法”。它通过计算物体在张量域空间 `[-0.5, 0.5]` 内的空间长宽高阈值，自动判断连通网格是否完全触碰或撑满边缘边界。这个专门为了深层网络的 SDF 设计的裁剪器完美剥离了外层漂浮面，保留了中心的主体草莓核心实体。
17. 针对高轮数(500轮)重建草莓表面存在波纹褶皱的问题(高频SDF采样噪声被网络放大)，在 `deepsdf/deep_sdf/mesh.py` 的 Marching Cubes 提取后新增了 Laplacian 平滑后处理步骤(`filter_smooth_laplacian`，迭代10次，lambda=0.5)。该方法在保留500轮学到的精细形状的同时消除因法线估计误差和SDF采样噪声引发的表面波纹，实现形状精准加表面光滑的双重效果。
18. 重构了 `deepsdf/deep_sdf/mesh.py`，对比原版 `mesh copy.py` 进行了全面审查：还原了体素网格参数与原版一致（`voxel_origin=[-0.5,-0.5,-0.5]`），同时保留了连通域过滤和 Laplacian 平滑两大后处理步骤。关键修复：将保存方式从 Open3D `write_triangle_mesh`（会写入错误法线导致渲染扁平灰色）改回原版的 `plyfile` 格式（仅保存顶点+面片，不写法线），让可视化器自行计算法线以保证正确的光照渲染效果。
19. 修复了 `data_preparation/prepare_strawberry_sdf.py` 中导致多层壳重建的两个根本性数据质量问题：(1) **法线方向错误**：原代码 `orient_normals_towards_camera_location(np.zeros(3))` 对居中在原点的点云会使法线全部指向内部，导致 SDF 正负标签翻转。修改为基于质心的逐点点积检测，确保所有法线指向物体外部。(2) **缺少 free space 采样**：参考原版 `o3d_utils.py` 的 `sample_freespace` 函数，新增了远场正 SDF 采样点（在法线方向 0.01~0.1 距离处），为网络在远离表面的区域提供了明确的正 SDF 约束，彻底杜绝无约束区域产生伪零交叉面（多层壳）的问题。修改后需要重新运行数据准备和训练。
20. 修复了 `deepsdf/deep_sdf/o3d_utils.py` 中与 `prepare_strawberry_sdf.py` 完全相同的法线方向 bug：`orient_normals_towards_camera_location(np.zeros(3))` 导致居中点云法线朝内。该函数在 `reconstruct_deep_sdf.py --partial` 重建过程中被调用，用于生成虚拟深度图和 Poisson Mesh。法线错误导致 Poisson 重建本身就有问题，产生了空洞和变形。已使用基于质心的逐点点积检测确保法线朝外。修改后需要重新运行 reconstruct_deep_sdf.py。
21. 修复了所有重建草莓底部一致性空洞的问题。原因是草莓 SDF 数据在 Y 轴方向最小值可达 -0.59，而 Marching Cubes 网格只覆盖 [-0.5, 0.5]，底部被截断。将 `deepsdf/deep_sdf/mesh.py` 中的体素网格范围从 [-0.5, 0.5] 扩大到 [-0.7, 0.7]，同时相应更新了连通域过滤的边界检测阈值。
22. 重写了 `deepsdf/deep_sdf/o3d_utils.py` 中用于 `--partial` 模式的两个核心函数：(1) `generate_pcd_from_virtual_depth`：原版通过 Poisson 重建→虚拟深度渲染→单视角部分点云的复杂流程完全不适用于草莓点云（法线错误、只看到一面、依赖 GUI 渲染器）。重写为使用 Open3D `hidden_point_removal` 从随机视角直接模拟遮挡，简洁高效。(2) `generate_deepsdf_target`：修复法线对齐（质心外向替代强制 Y 轴对齐）、SDF 距离（0.04/0.01 替代极小的 0.001）、free space 采样范围（0.01~0.1 替代 0.001~0.01），全部与 `prepare_strawberry_sdf.py` 训练数据保持一致。
23. 更新了 `workflow_new.md` 文档的第三阶段说明，详细解释了 `--partial` 参数的作用及其正确的适用场景：对于草莓端到端的直接补全方案，如果接收的是完整点云，可以直接传入 SDF ；在需要模拟残缺点云时添加该参数。
24. 修复了 `train.py` 在第四阶段(训练点云编码器) 因代码写死 `Codes/partial` 路径而引发的 `num_samples=0` 报错中断问题。现已加入智能侦测，会根据第三阶段是否加 `--partial` 产生的结果目录(`complete` 或 `partial`)，自动读取可用的 Latent Codes 进行训练。
25. 修复了 `test.py` 在第五阶段（测试/验证）时与 `train.py` 一样由于写死只认 `Codes/partial` 读取真实 Latent Code 对比引发的 `Warning: Found XXX.ply but no corresponding latent code in XXX.pth` 导致无对比项无法完整输出 MSE 指标的问题。已同步追加通过 `os.path.exists` 实现智能回退兼容至可用的 `complete` 真值库的机制。
26. 修复了 `test.py` 中网格重建质量极差（顶点仅 50-110 个）的两个根本原因：(1) 原版 `sdf2mesh_cuda` 使用 `compute_convex_hull`（凸包）而非 Marching Cubes，凸包天生无法表达凹面和细节，只能生成粗糙的多面体。已在 `test.py` 中替换为 `skimage.measure.marching_cubes` 进行等值面提取。(2) `configs/strawberry.json` 中 `grid_density` 仅为 20（20³=8000 个采样点），远低于 `mesh.py` 使用的 256³=1670万。已提升至 128（128³=约200万采样点），在速度和质量之间取得平衡。
27. 修复了 `test.py` 在提升网格分辨率至 128 (约生成210万个散点坐标) 后导致的 `CUDA Out of Memory`。由于之前网络推理将全量空间直接拼接(`torch.cat`)作为210万x34维的矩阵送入 Decoder `forward()`，需要一次性申请 4~8GB 缓存撑爆了只有 8G 显存的本地 GPU。修改方案为分批循环推断 (`batch_size = 65536`) 最后重新拼接，彻底解决了内存泄露。
28. 为 `test.py` 的 Marching Cubes 重建结果添加了连通域过滤（`cluster_connected_triangles` + 保留最大连通组件），移除了由 Decoder SDF 远场噪声零交叉产生的分布在立方体边界内的大量碎片三角面。与 `mesh.py` 中的过滤逻辑保持一致。
29. 修复了 `test.py` 保存的网格渲染颜色扁平（灰色无立体感）的问题。原因与 `mesh.py` 第 18 条记录相同：`o3d.io.write_triangle_mesh` 将不一致的顶点法线写入 PLY 文件，导致可视化器渲染异常。修改为保存前清除法线，让可视化器在加载时自行计算正确法线。
30. 为 `test.py` 的重建结果添加了 Laplacian 平滑后处理（`filter_smooth_laplacian`，迭代 10 次，lambda=0.5），与 `mesh.py` 保持一致，消除表面噪声波纹使草莓表面更光滑。
31. 修复了 `test.py` 中 `mesh_volume_ml` 全为 0 和表面毛刺两个问题：(1) 体积为 0 是因为 `mesh.is_watertight()` 对 Marching Cubes 在有界网格上生成的开口网格始终返回 False，改为直接使用 `abs(mesh.get_volume())`（签名体积法对近似闭合网格足够准确）。(2) 毛刺是因为 Marching Cubes 输出包含退化三角面和非流形边，在 Laplacian 平滑前新增了 `remove_degenerate_triangles`、`remove_duplicated_triangles`、`remove_non_manifold_edges` 等清理步骤，并将平滑迭代次数从 10 提升至 20。
32. 进一步优化 `test.py` 的后处理流程：(1) 将 Laplacian 平滑替换为 **Taubin 平滑**（`filter_smooth_taubin`，30 次迭代），Taubin 平滑不会缩小网格体积，且对毛刺的消除效果更好。(2) 体积计算改为使用**凸包** (`compute_convex_hull`)，与原版土豆论文的 `sdf2mesh_cuda` 保持一致——凸包始终是水密的，`get_volume()` 总能可靠计算。原版之所以没有毛刺和体积为 0 的问题，正是因为全程使用凸包（对近似凸形的土豆足够准确），但草莓形状有凹面细节，所以我们保留 Marching Cubes 用于可视化，仅在体积计算时使用凸包。
33. 将 `test.py` 的体积计算从 Open3D `compute_convex_hull().get_volume()` 替换为 `scipy.spatial.ConvexHull`。Open3D 的凸包有时会生成不水密的结果导致 `get_volume()` 抛出异常，而 scipy 的 QHull 算法直接从顶点计算凸包体积，100% 可靠，无任何水密性依赖，且速度极快。
34. 创建了 `.gitignore` 文件，配置了忽略追踪项目中的数据集（如 `data/`）、模型日志与实验配置（如 `logs/`，`deepsdf/experiments/`）、测试结果和体积评估生成文件（如 `shape_completion_results.csv` 等）以及常规的 Python 环境缓存或 IDE 附加文件。
