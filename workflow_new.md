# 草莓数据集：三维重建与大小估计工作流程 (点云数据端到端架构)

因为你手里只有配准好的点云数据，没有多视角的 RGB-D 图片。因此我们可以修改工作流，将原始的“图像编码器(Image Encoder)”替换为特定的**“点云编码器(Point Cloud Encoder)”（例如 FoldNet 或 PCE）**。

这种端到端架构的优势在于，推理阶段比之前单靠 Decoder 进行“测试时优化（Test-time Optimization）”要快得多。整体流程分为：DeepSDF预训练 -> 获取目标隐变量 -> 训练点云编码器 -> 端到端快速补全与大小估计。

---

## 第一阶段：纯点云数据准备 (Data Preparation)

草莓点云已经配准，不用处理相机内参、姿态或变换矩阵 (`tf.npz`, `intrinsic.json`)：

**1. 建立数据文件目录结构**
创建 `./data/strawberry` 目录，并按照以下格式放入已配准的点云（名字必须为 `fruit.ply`）：
```text
data/strawberry/
├── split.json                  # 定义训练、验证与测试集的划分
├── sample_01/
│   ├── dataset.json            # 标记为有效: {"is_useable": true}
│   └── laser/
│       └── fruit.ply           # 草莓点云数据
└── sample_02/
    └── ...
```

**2. （强烈推荐）数据几何增强 (Data Augmentation)**
如果你发现草莓点云的分布尺度和大小趋于一致，网络难以为未见过的大小推断体积（缺乏尺度泛化能力）。强烈推荐运行我专门为你编写的点云增强独立脚本 `augment_strawberry.py`，它会只打乱草莓的大小比例、扭转朝向和做裁切形变（但保留正确的朝外水密法线）：
```bash
python data_preparation/augment_strawberry.py \
    --src /home/tianqi/corepp2/data/20260301_dataset \
    --dst /home/tianqi/corepp2/data/20260301_dataset \
    --json_config data_preparation/augment.json
```
*(注意：扩增完成后，后续所有的第3步、第4步都要换成这个 `_aug` 的新目录，对应的实验目录也要一并建一个新的去重新跑端到端训练！)*

**3. 生成 SDF (符号距离场) 采样训练数据**
运行以下专门针对单体配准点云结构编写的脚本，会在数据集内按 DeepSDF 所需层级生成 `samples.npz`，这是 DeepSDF 网络训练必须的三维外/内表面距离标签。

> **重要说明：** 该脚本已修复了两个影响重建质量的关键问题：
> 1. **法线方向校正**：采用基于质心的逐点点积检测，确保法线全部指向物体外部（原始方法会导致居中点云法线朝内，造成 SDF 内外翻转产生多层壳）。
> 2. **Free Space 远场采样**：在表面外侧 0.01~0.1 距离处额外采样 20000 个正 SDF 点，为网络在远离表面的区域提供约束，防止产生伪零等值面。

```bash
python data_preparation/prepare_strawberry_sdf.py --src /home/tianqi/corepp2/data/20260312_dataset
```

**4. 划分训练/验证/测试集 (Make Splits)**
编写了专门的切分脚本 `data_preparation/make_strawberry_splits.py`。该脚本会自动读取数据集下的所有的实例，并按 `8:1:1` 的比例进行划分：
1. 生成全局的 `split.json` 并放置在数据集根目录
2. 生成专供 DeepSDF 使用的 `{dataset_name}_train.json`, `{dataset_name}_val.json`, `{dataset_name}_test.json` 文件放置于 `deepsdf/experiments/splits` 中以备后用。
```bash
python data_preparation/make_strawberry_splits.py --data_dir /home/tianqi/corepp2/data/20260312_dataset
```

---

## 第二阶段：训练 DeepSDF 解码器 (Decoder)

通过该步骤学习所有草莓样本的 SDF 数据，让 Decoder 掌握草莓物体的通用三维形态，建立起专属这批草莓的 Latent Space。

**1. 准备配置信息**
- 在 `deepsdf/experiments/splits/` 目录下创建 `strawberry_train.json` (对应训练集) 和 `strawberry_test.json`。
- 在 `deepsdf/experiments/strawberry` 目录下创建复制/修改 `specs.json`，确保 `DataSource` 指向 `./data`，`TrainSplit` 指向刚刚的 `strawberry_train.json`。

**2. 启动训练**
```bash
python train_deep_sdf.py --experiment ./deepsdf/experiments/20260312_dataset
```
根据表现选取一个最优的 Checkpoint（例如第 1000 个 Epoch，即 `1000.pth`）。

---

## 第三阶段：训练点云编码器 (Point Cloud Encoder)

这部分就是你关注的编码器训练。因为没有前置的 RGB 图像，我们需要配置一个专用于处理点云特征的网络（比如 Point Cloud Encoder 或 FoldNet）。

> **关于 Latent Codes 真值（Ground Truth）的重要改动：**
> 在原版的测试时优化流程中，需要在训练完了 DeepSDF 之后强行再跑一趟 `reconstruct_deep_sdf.py` 取出重构真值供此时的 Encoder 学习，效率偏低且带来不可靠的噪音。现在 `test.py` 和 Dataset 已**完全重构**，网络会自动从你第二阶段跑完的 `deepsdf/experiments/.../LatentCodes` 中提取生成合并而成的宏观隐式表面权重，**无需手动再跑重构提取！** 

**1. 编写配置文件**
在 `./configs/` 下新建一份 `strawberry.json`，核心设置一定要将 `"encoder"` 指定为点云网络：

```json
{
    "species" : "strawberry",
    "data_dir" : "./data/strawberry/",
    "checkpoint_dir" : "./logs/strawberry/checkpoints/",
    "log_dir": "./logs/strawberry/runs/",
    "encoder" : "foldnet",  // 非常关键：指定为 foldnet、point_cloud 或 point_cloud_large
    "input_size" : 2048,    // 采样点个数
    "batch_size" : 12,
    "epoch" : 100,
    "supervised_3d" : true, //开启3D监督
    "grid_density": 20
    //与 pce.json 类似保持不变
}
```

**2. 启动 Encoder 网络训练**
```bash
python train.py \
    --cfg ./configs/strawberry.json \
    --experiment ./deepsdf/experiments/20260312_dataset \
    --checkpoint_decoder 500
```
在这里点云编码器（Encoder）会接收点云，预测出一组 Latent Code，并直接与 DeepSDF 训练阶段生成的真实 Latent Code 比较误差来更新自身权重。

---

## 第四阶段：端到端推理与草莓大小(体积)估计

现在编码器（Encoder）和解码器（Decoder）均已训练完毕。对于**完全未见过的测试集草莓点云**，我们可以实现直接前向打通：点云 -> Encoder提取编码 -> Decoder重建Mesh网格 -> 计算出真实体积。

**运行直接测试评估脚本：**

```bash
python test.py \
    --cfg ./configs/strawberry.json \
    --experiment ./deepsdf/experiments/20260312_dataset \
    --checkpoint_decoder 500 
```

在默认状态下，测试脚本会自动读取 `data/dataset_name/split.json` 里标注的 `test` 划分集数据进行验证评估。

**在野推断（In-the-wild Inference/自定义测试文件夹）：**
如果你有新采集的散装点云数据，或者刚才利用脚本（诸如投影视角采样渲染脚本带来的残缺点云）生成了一批存放在另一个游离目录的外源 `.ply` 测试包。这时候你只需要挂上 `--test_data_dir` 参数就可以直接对其执行测试推理：
```bash
python test.py \
    --cfg ./configs/strawberry.json \
    --experiment ./deepsdf/experiments/20260301_dataset_aug \
    --checkpoint_decoder 500 \
    --test_data_dir /home/tianqi/corepp2/data/render_output_perspective/size38_16384_normalized
```

`test.py` 内部处理逻辑已经包括了重建出的网格对象的读取，并通过 `mesh.get_volume()` 原生实现了真实空间体积大小计算。
推理结果会自动作为行数据追加到当前主目录下的运行报告中：
✅ **报告位置：** `./shape_completion_results.csv`
在这个输出表中，主要关注 `mesh_volume_ml` 列，它是基于计算出几何物体的体积（Volume）进而估算出的真实草莓大小。