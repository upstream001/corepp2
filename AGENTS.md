# Repository Guidelines

## 项目结构与模块组织
训练与推理入口脚本位于仓库根目录：
- `train.py` / `test.py`：编码器训练与评估。
- `train_deep_sdf.py` / `reconstruct_deep_sdf.py`：DeepSDF 解码器训练与潜变量重建。
- `compute_reconstruction_metrics.py`：重建指标计算（Chamfer、F1 等）。

核心目录说明：
- `configs/`：实验配置 JSON（如 `configs/super3d.json`）。
- `dataloaders/`、`networks/`、`metrics_3d/`、`sdfrenderer/`：数据处理、模型与评估模块。
- `deepsdf/`：DeepSDF 代码、实验配置与数据划分文件。
- `data_preparation/`：数据转换、增强、切分脚本。
- `data/` 与 `logs/`：本地数据与训练产物（检查点、日志）。

## 构建、测试与开发命令
建议使用 Python 3.10 环境（见 `INSTALL.md`）。

- `python train_deep_sdf.py --experiment ./deepsdf/experiments/potato`
  使用实验目录中的 `specs.json` 训练 DeepSDF 解码器。
- `python reconstruct_deep_sdf.py --experiment ./deepsdf/experiments/potato --data ./data --checkpoint_decoder 500 --split ./deepsdf/experiments/splits/potato_val.json`
  在指定 checkpoint 和 split 上重建网格与潜变量。
- `python train.py --cfg ./configs/super3d.json --experiment ./deepsdf/experiments/potato --checkpoint_decoder 500`
  基于预训练解码器潜变量训练编码器。
- `python test.py --cfg ./configs/super3d.json --experiment ./deepsdf/experiments/potato --checkpoint_decoder 500`
  运行评估并输出 `shape_completion_results.csv`。
- `bash run_scripts_reconstruct.sh` / `bash run_scripts_metrics.sh`
  批量遍历多个 checkpoint 做重建与指标筛选。

## 代码风格与命名规范
- Python 使用 4 空格缩进；函数/变量/文件采用 `snake_case`，类名采用 `PascalCase`。
- 配置键名与 CLI 参数保持稳定，优先扩展现有 JSON 配置，避免硬编码路径。
- 通用逻辑放入模块目录（如 `dataloaders/`、`networks/`、`utils_file/`），入口脚本仅保留流程编排。

## 测试规范
- 当前仓库未配置统一 `pytest` 测试框架，采用脚本级验证。
- 模型改动至少完成：
  1. 使用调试配置进行短轮次训练冒烟测试。
  2. 对同一配置与 checkpoint 运行 `test.py`。
  3. 若修改重建逻辑，在小规模样本上执行 `compute_reconstruction_metrics.py`。
- 临时验证脚本仅在可复现、与数据集强耦合较低时命名为根目录 `test_*.py`。

## 提交与 PR 规范
- 历史提交以简短、任务导向为主，常见前缀：`修复:`、`更新:`、`加入:`。
- 建议格式：`<类型>: <变更内容>`，例如 `修复: 测试管线中的点云归一化`。
- PR 建议包含：
  - 变更内容与动机；
  - 验证所用配置与命令；
  - 涉及模型效果时附前后指标或可视化结果（表格/截图）。
