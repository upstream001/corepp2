基于 `README.md`，项目中提出的 CoRe++ 方法的整体流程可以分为两个主要阶段：
**第一阶段：训练 DeepSDF（解码器）**用于学习三维形状表示；
**第二阶段：训练 Encoder（编码器）**用于从单视角 RGB-D 图像预测出 DeepSDF 的隐编码（latent code），从而完成 3D 形状补全。

以下是详细流程拆解：

### 阶段一：准备三维真值并训练 DeepSDF (Decoder)

在这一步中，主要是使用通过如 SfM（Structure-from-Motion）技术得到的完整三维点云数据，来训练一个 DeepSDF 网络，使它掌握物体（如土豆）的三维形变空间。

1. **环境与数据就绪**
   - 依照 `INSTALL.md` 配置环境（如依赖项安装）。
   - 下载提供的数据集（Demo Dataset）并放在 `data/3DPotatoTwinDemo` 目录下。

2. **数据预处理 (Data Preparation for DeepSDF)**
   由于 DeepSDF 的训练需要表面（Surface）和空间中的 SDF（符号距离场）采样点和方向信息，因此需要进行转换：
   - **点云提取**：`pcd_from_sfm.py` 从多视角重建生成的 mesh 数据中提取出点云数据，存放入 `./data/potato` 中。
   - **数据增强**：`augment.py` 针对原样本进行数据增强，生成更多不同姿态或变形的点云，并保存在 `./data/potato_augmented` 目录。
   - **SDF 采样**：对原始数据和增强数据分别运行 `prepare_deepsdf_training_data.py`。这一步会在物体表面周围一定空间内随机采样点，并为每个采样点计算出对应表面的 SDF （带符号距离）值，最后存储为 `samples.npz` (Pos/Neg样本分布)，供网络学习。

3. **训练 DeepSDF (Training DeepSDF)**
   - 修改 `deepsdf/experiments/potato/specs.json` 配置文件中的路径为本机实际路径，设置好网络参数、隐向量(latent)维数等。
   - 运行 `train_deep_sdf.py`，模型将针对所有的三维数据去学习一个共享的 Decoder 网络和每个物体独立对应的 latent code。

4. **筛选最优模型 (Reconstruct & Evaluate)**
   - 训练期间会保存多个 Epoch 的模型并产生对应的隐编码（latent code）。
   - 运行测试脚本 `run_scripts_reconstruct.sh` 使用不同阶段的模型来重建所有的点云模型。
   - 运行对比评估脚本 `run_scripts_metrics.sh` 计算重建模型的精度指标，**由此选出表现最好的模型权重（Best weights)**，比如 README 例子里选择的是第 500 个 Epoch 的模型。

5. **生成最终的隐编码标签 (Generating Ground Truth Latent Codes)**
   - 在确定了最佳的 DeepSDF 解码器模型（如 Checkpoint 500）之后，运行对应的 `reconstruct_deep_sdf.py` 脚本（针对普通训练集、验证集和测试集）。该操作使用这个固定的最佳模型来生成/优化数据集中各样本最终对应的专属 latent code，这些隐编码将作为后面编码器训练的**Ground Truth (真实标签)**。

---

### 阶段二：使用单视角 RGB-D 训练图像编码器 (Encoder)

这一阶段的核心是输入采集端（如深度相机）拍到的单边不完整 RGB-D 图像，输出一个 latent code，然后交由上一步练好的 DeepSDF Decoder 解出完整的 3D 结构。

6. **准备 RGB-D 数据集 (Prepare dataset for Encoder)**
   - `organize_data.py` 将提供的 RGB-D 单视角拍摄图像数据（包含 color, depth 等信息）整理并迁移到 `./data/potato` 工作目录下。
   - 然后通过一连串的 `copy_file.py` 脚本，将标定信息（如 RealSense的内参 `intrinsic.json`）、相机的位姿/坐标系变换数据 (`tf/tf.npz`)、物体边界框（`bounding_box.npz`）等关键辅助文件复制到对应的文件夹位置，这些对单视角点的对齐和截取十分重要。

7. **训练图像与点云编码器 (Train the Encoder)**
   - 将配置文件 `configs/super3d.json` （或者其他模型的配置文件）里涉及到数据位置的路径改为对应的本地路径。
   - 运行 `train.py`。在这里它需要指定 `cfg` (超参配置), `experiment` (包含DeepSDF的工作区), 以及**固定的 DeepSDF 解码器 (Checkpoint = 500)**。
   - **学习目标**：Encoder (即 Super3D / 各阶段的网络) 会读取 RGB-D 图片 / 不完整的部分点云，并学习预测出一个潜在向量。训练过程通过对比“预测出的潜在向量”和“阶段一在第5步生成的真实潜在向量”，利用误差反向传播来更新 Encoder 权重。

8. **测试最终模型 (Test the Encoder)**
   - 最后使用 `test.py` 进行效果测试，向 Encoder 送入来自测试集的全新 RGB-D 图片（不完整的单侧视角）。
   - Encoder 计算出预测的 latent code，固定好的 DeepSDF Decoder 则将其还原成了补全后的高保真 3D 形状（例如完整的土豆），从而实现了高通量的 3D 形状补全流程。
