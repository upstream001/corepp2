# CoRe++: High-Throughput 3D Shape Completion of RGB-D Images
## Adjustments are at bottom

![CoRe++](data/git_promo.gif)

### Installation

[INSTALL.md](INSTALL.md)
<br/><br/>

### Dataset
[3DPotatoTwin](https://huggingface.co/datasets/UTokyo-FieldPhenomics-Lab/3DPotatoTwin)
<br/><br/>

### Network weights
[CoRe++ weights](https://drive.google.com/drive/folders/1-i4XYDwQbiJx2x836lqgGCkdotlE7sNC?usp=sharing)
<br/><br/>

### Instructions

1. Download our [demo dataset](https://github.com/UTokyo-FieldPhenomics-Lab/corepp/releases/tag/demo_dataset).
2. Place the zip file in the data folder and unzip the files
3. Prepare the dataset for training DeepSDF
    ```python
    python data_preparation/pcd_from_sfm.py --src ./data/3DPotatoTwinDemo/2_sfm/1_mesh --dst ./data/potato
    python data_preparation/augment.py --json_config_filename ./data_preparation/augment.json --src ./data/potato --dst ./data/potato_augmented
    python data_preparation/prepare_deepsdf_training_data.py --src ./data/potato
    python data_preparation/prepare_deepsdf_training_data.py --src ./data/potato_augmented
    ```
4. Change the file paths in **deepsdf/experiments/potato/specs.json** such that they correspond to your file paths
5. Train DeepSDF
    ```python
    python train_deep_sdf.py --experiment ./deepsdf/experiments/potato
    ```
6. Reconstruct the 3D shapes with DeepSDF
    ```console
    bash run_scripts_reconstruct.sh
    ```
7. Compute the reconstructing metrics and determine the best weights file
    ```console
    bash run_scripts_metrics.sh
    ```
8. For the best weights run the following 3 commands. In this example the best weights are at checkpoint 500.
    ```python
    python reconstruct_deep_sdf.py --experiment ./deepsdf/experiments/potato --data ./data --checkpoint 500 --split ./deepsdf/experiments/splits/potato_train_without_augmentations.json
    python reconstruct_deep_sdf.py --experiment ./deepsdf/experiments/potato --data ./data --checkpoint 500 --split ./deepsdf/experiments/splits/potato_val.json
    python reconstruct_deep_sdf.py --experiment ./deepsdf/experiments/potato --data ./data --checkpoint 500 --split ./deepsdf/experiments/splits/potato_test.json
    ```
9. Prepare the dataset for training the encoder
    ```python
    python data_preparation/organize_data.py --src ./data/3DPotatoTwinDemo/1_rgbd/1_image --dst ./data/potato
    python data_preparation/copy_file.py --src ./data/potato_example/dataset.json --dst ./data/potato --subdir ""
    python data_preparation/copy_file.py --src ./data/potato_example/tf/tf.npz --dst ./data/potato --subdir "tf"
    python data_preparation/copy_file.py --src ./data/potato_example/tf/bounding_box.npz --dst ./data/potato --subdir "tf"
    python data_preparation/copy_file.py --src ./data/potato_example/realsense/intrinsic.json --dst ./data/potato --subdir "realsense"
    ```
10. Change the file paths in **configs/super3d.json** such that they correspond to your file paths
11. Train the encoder
    ```python
    python train.py --cfg ./configs/super3d.json --experiment ./deepsdf/experiments/potato/ --checkpoint_decoder 500
    ```
12. Test the encoder
    ```python
    python test.py --cfg ./configs/super3d.json --experiment ./deepsdf/experiments/potato/ --checkpoint_decoder 500
    ```
<br/>

### Citation
Refer to our research article: 
```BibTeX
@article{BLOK2025109673,
    title = {High-throughput 3D shape completion of potato tubers on a harvester},
    author = {Pieter M. Blok and Federico Magistri and Cyrill Stachniss and Haozhou Wang and James Burridge and Wei Guo},
    journal = {Computers and Electronics in Agriculture},
    volume = {228},
    pages = {109673},
    year = {2025},
    issn = {0168-1699},
    doi = {https://doi.org/10.1016/j.compag.2024.109673},
    url = {https://www.sciencedirect.com/science/article/pii/S0168169924010640},
    keywords = {Potato, Deep learning, RGB-D, 3D shape completion, Structure-from-Motion},
}
```
<br/>

### Acknowledgements
CoRe++ is the updated version of Federico Magistri's original CoRe implementation: <br/>
https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/magistri2022ral-iros.pdf


### My adjustments

本项目在原始 CoRa++ 的基础上进行了重大改进，主要将其扩展为支持**纯点云 (Pure Point Cloud)** 输入的端到端三维补全与尺寸估计架构。

#### 1. 数据准备工具 (Data Preparation)
除了核心训练流程外，本项目还提供了一系列用于构建点云补全数据集的工具：
- **`utils_file/realtime_render.py`**: 高效的正交投影渲染工具，利用 Z-Buffer 技术从随机视角快速生成残缺点云。
- **`utils_file/perspective_render.py`**: 透视投影渲染工具，模拟真实相机（如 Intel RealSense D405）的近距离成像过程，包含深度切片（Depth Slicing）功能。
- **`utils_file/construct_test_datasetV3.py`**: 自动化数据集构建脚本，将不同渲染方式生成的残缺点云与完整点云进行配对，并生成命名统一的训练/测试集。

#### 2. 纯点云输入架构
不同于原始版本依赖 RGB-D 图像，改进后的工作流允许直接使用配准后的单体点云（如草莓点云）作为输入。这在仅有激光扫描数据或点云融测场景下具有更高的适用性。

#### 3. 网络架构
为了高效处理点云特征并预测 DeepSDF 潜变量，引入了多种点云编码器：
- **PointNeXt (推荐)**: 位于 `networks/pointnext.py`，采用多尺度特征提取（Set Abstraction 和 InvResMLP），是目前效果最平衡的编码器。相关详细参数（如 k 值、通道数等）见 [architecture.md](architecture.md)。
- **FoldNet**: 位于 `networks/models.py` 的 `FoldNetEncoder`，基于 Folding 机制进行特征编码。
- **PointCloudEncoder**: 基础的 MLP 风格编码器，适用于快速基准测试。

#### 4. 新版工作流 (`workflow_new.md`)
针对纯点云路径，我们制定了全新的端到端还原流程，详细步骤见 [workflow_new.md](workflow_new.md)。核心阶段包括：
- **数据准备**：包括点云几何增强 (`augment_strawberry.py`)、修复了法线翻转问题的 SDF 采样生成 (`prepare_strawberry_sdf.py`) 以及支持随机种子的数据集切分 (`make_strawberry_splits.py`)。
- **DeepSDF 预训练**：训练解码器（Decoder）以掌握物体的通用隐式表示。
- **编码器训练**：使用 `train.py` 训练编码器，将其预测结果对齐到 DeepSDF 的潜变量空间。
- **物理体积推断**：通过 `test.py` 直接 from 残缺点云生成补全 Mesh，并结合 `normalization_scale` 参数输出具有真实物理意义的体积（ml）。

#### 5. 关键文件说明
- `train_deep_sdf.py` / `reconstruct_deep_sdf.py`: DeepSDF 的核心训练与潜变量优化脚本。
- `train.py` / `test.py`: 外部编码器训练与全流程测试评估。`test.py` 已集成 Marching Cubes 提取、Taubin 平滑滤波、连通域清理及基于凸包的可靠体积计算。
- `dataloaders/pointcloud_dataset.py`: 专为纯点云设计的 Dataset 类，支持中心化、动态边界盒计算及自动匹配 DeepSDF Latent 真值。
- `configs/`: 包含 `strawberry.json` 等配置文件，支持通过参数一键切换编码器类型、损失函数权重以及物理尺度对齐系数。