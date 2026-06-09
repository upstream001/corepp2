
# 🍓 如何在新的草莓数据集上训练 CoRe++

本文档详细介绍了如何在一个全新的数据集（例如草莓数据集 `strawberry`）上配置并训练 CoRe++ 模型。

CoRe++ 的训练分为两个主要阶段：
1. **DeepSDF (Decoder) 预训练**：训练模型学习完整的 3D 形状表示 (Latent Space)。
2. **Encoder 训练**：训练模型从 RGB-D 图像映射到预训练好的 Latent Space。

## 1. 准备数据目录结构

首先，你需要确保你的草莓数据集按照 CoRe++ 期望的格式组织。

假设你的项目根目录为 `/data/corepp`。

### 1.1 创建数据根目录
```bash
mkdir -p data/strawberry
```

### 1.2 数据组织规范
每一个草莓样本（例如 `sample_01`, `sample_02`）应该在 `data/strawberry` 下有一个独立的文件夹。目录结构应如下所示：

```
data/strawberry/
├── split.json                  <-- 关键：定义训练/验证/测试集划分
├── sample_01/                  <-- 样本 1
│   ├── dataset.json            <-- 关键：标记该样本是否可用
│   ├── laser/                  <-- 用于 DeepSDF 训练的完整点云
│   │   └── fruit.ply           <-- 必须命名为 fruit.ply (GT Mesh/PCD)
│   ├── realsense/              <-- RGB-D 输入数据
│   │   ├── color/              <-- RGB 图像
│   │   │   ├── 000.png
│   │   │   └── ...
│   │   ├── depth/              <-- 深度图 (.npy 格式)
│   │   │   ├── 000.npy
│   │   │   └── ...
│   │   ├── masks/              <-- 分割掩码 (二值图)
│   │   │   ├── 000.png
│   │   │   └── ...
│   │   └── intrinsic.json      <-- 相机内参
└── sample_02/
    └── ...
```

### 1.3 关键文件示例

**`data/strawberry/split.json`**
```json
{
  "train": ["sample_01", "sample_02", ...],
  "val": ["sample_10", ...],
  "test": ["sample_20", ...]
}
```

**`data/strawberry/sample_XX/dataset.json`**
```json
{
  "is_useable": true
}
```

---

## 2. 第一阶段：DeepSDF (Decoder) 训练

此阶段训练 Decoder 以重建完整的 3D 形状。

### 2.1 准备 DeepSDF 数据集划分文件
在 `deepsdf/experiments/splits/` 目录下创建新的 JSON 文件，指定哪些样本用于 DeepSDF 的训练。

创建 `deepsdf/experiments/splits/strawberry_train.json`:
```json
{
  ".": {
    "strawberry": ["sample_01", "sample_02", "sample_03"] 
  }
}
```
*注意：这里的 key `"strawberry"` 必须要与你的数据文件夹名称一致。*

同样创建 `strawberry_test.json` 用于测试。

### 2.2 准备 DeepSDF 训练数据 (.npz)
DeepSDF 需要预先采样 SDF 值。运行以下命令：

```bash
# 确保你的 python 环境已激活
python data_preparation/prepare_deepsdf_training_data.py --src ./data/strawberry
```
这会在每个样本目录下生成 `SDFSamples` 文件夹。

### 2.3 配置实验参数
1. 创建实验目录：
   ```bash
   mkdir -p deepsdf/experiments/strawberry
   ```
2. 复制并修改配置文件：
   ```bash
   cp deepsdf/experiments/potato/specs.json deepsdf/experiments/strawberry/specs.json
   ```
3. 编辑 `deepsdf/experiments/strawberry/specs.json`，修改以下关键字段：
   ```json
   {
     "Description": "Strawberry DeepSDF Training",
     "DataSource": "./data",  
     "TrainSplit": "deepsdf/experiments/splits/strawberry_train.json", 
     "TestSplit": "deepsdf/experiments/splits/strawberry_test.json",
     "CodeLength": 32,
     ...
   }
   ```
   *注意：`DataSource` 应该是指向包含 `strawberry` 文件夹的父目录（这里是 `./data`），因为 Split 文件中的 key 是 `strawberry`。*

### 2.4 运行 DeepSDF 训练
```bash
python train_deep_sdf.py --experiment ./deepsdf/experiments/strawberry
```
训练完成后，模型权重会保存在 `deepsdf/experiments/strawberry/ModelParameters/`。

---

## 3. 第二阶段：生成 Latent Codes (Reconstruction)

为了训练 Encoder，我们需要为每个训练样本生成对应的 "Ground Truth" Latent Code。这是通过使用训练好的 Decoder 对每个样本进行优化得到的。

假设 DeepSDF 训练了 1000 epoch，我们使用第 1000 轮的权重。

```bash
# 为训练集生成 Latent Codes
python reconstruct_deep_sdf.py \
    --experiment ./deepsdf/experiments/strawberry \
    --data ./data \
    --checkpoint 1000 \
    --split deepsdf/experiments/splits/strawberry_train.json

# 为验证集/测试集生成 Latent Codes (可选，用于评估)
python reconstruct_deep_sdf.py \
    --experiment ./deepsdf/experiments/strawberry \
    --data ./data \
    --checkpoint 1000 \
    --split deepsdf/experiments/splits/strawberry_test.json
```
结果会保存在 `deepsdf/experiments/strawberry/Reconstructions/1000/Codes/` 下。

---

## 4. 第三阶段：Encoder 训练

此阶段训练 CoRe++ 模型（Encoder），使其能从 RGB-D 输入预测 Latent Code。

### 4.1 配置 Encoder 参数
1. 复制配置文件：
   ```bash
   cp configs/super3d.json configs/strawberry.json
   ```
2. 编辑 `configs/strawberry.json`，**必须修改**以下参数以适配新数据集：

   ```json
   {
       "species": "strawberry",  // 必须与数据文件夹名一致 (用于寻找 dataset.json 等)
       "data_dir": "./data/strawberry/", // 指向具体的数据目录
       "checkpoint_dir": "./logs/strawberry/checkpoints/",
       "log_dir": "./logs/strawberry/runs/",
       
       // *** 关键数据参数 ***
       "depth_min": 100,  // 根据草莓数据的实际深度范围调整 (单位通常是 mm)
       "depth_max": 300,  // 同样根据实际情况调整
       "input_size": 256, // 输入图像的裁剪/Padding 尺寸
       
       "encoder": "pool", // 推荐的模型架构
       "batch_size": 16,
       "epoch": 200,
       ...
   }
   ```

### 4.2 运行 Encoder 训练
启动训练脚本，指定刚才生成的 DeepSDF Checkpoint 作为 Decoder 的固定权重。

```bash
python train.py \
    --cfg ./configs/strawberry.json \
    --experiment ./deepsdf/experiments/strawberry/ \
    --checkpoint_decoder 1000
```

### 4.3 监控训练
训练日志会保存到 `log_dir`，你可以使用 TensorBoard 查看：
```bash
tensorboard --logdir ./logs/strawberry/runs/
```

---

## 5. 测试与推理

训练完成后，使用 `test.py` 进行评估：

```bash
python test.py \
    --cfg ./configs/strawberry.json \
    --experiment ./deepsdf/experiments/strawberry/ \
    --checkpoint_decoder 1000
```

## 常见问题 (FAQ)

*   **Q: 为什么 Loss 不下降？**
    *   A: 检查 `depth_min` 和 `depth_max` 是否涵盖了你的深度图的实际值范围。如果归一化出错，深度信息将丢失。
*   **Q: 显存不足 (OOM) 怎么办？**
    *   A: 在 `configs/strawberry.json` 中减小 `batch_size`。
*   **Q: 缺少 Latent Code 报错？**
    *   A: 确保你运行了 **步骤 3 (Reconstruction)**，并且 `split.json` 中的样本 ID 与生成 Latent Code 的样本 ID 一致。

