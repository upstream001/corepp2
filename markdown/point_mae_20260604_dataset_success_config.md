# Point-MAE Encoder On `20260604_dataset`

## Summary

本次实验在 `20260604_dataset` 上取得了当前较好的结果。核心变化是将点云编码器切换为本地接入的 `Point-MAE` 风格 encoder，并将训练损失简化为**只保留 `SuperLoss`**。

对应结果文件：

- [`shape_completion_results_multi_threshold.csv`](/home/tianqi/corepp2/shape_completion_results_multi_threshold.csv)

当前 summary 行指标：

- `num_samples = 132`
- `complete_volume_ml mean = 21.066671`
- `mesh_volume_ml mean = 23.079058`
- `volume_mae_ml = 2.143415`
- `volume_rmse_ml = 2.602084`
- `volume_mape_percent = 11.507`
- `volume_r2 = 0.757744`
- `chamfer_distance = 0.063259`
- `f1_t0p05 = 45.9`

## Effective Config

配置文件：

- [`configs/20260604_dataset.json`](/home/tianqi/corepp2/configs/20260604_dataset.json)

本次有效实验的关键配置如下：

```json
{
  "data_dir": "/home/tianqi/corepp2/data/20260604_dataset/",
  "encoder": "point_mae",
  "batch_size": 8,
  "input_size": 2048,
  "epoch": 25,
  "lr": 0.0001,
  "validation_frequency": 1,
  "use_partial_input": true,
  "group_by_instance_batch": true,
  "instances_per_batch": 4,
  "validate_mesh_volume": true,
  "selection_metric": "combined",
  "lambda_super": 1.0,
  "lambda_volume": 0.0,
  "lambda_latent_spread": 0.0,
  "lambda_instance_contrastive": 0.0,
  "lambda_decoder_consistency": 0.0,
  "lambda_latent_norm": 0.0,
  "lambda_latent_mean": 0.0
}
```

## Training Command

本次实验使用的 DeepSDF checkpoint：

- experiment: [`deepsdf/experiments/20260604_dataset`](/home/tianqi/corepp2/deepsdf/experiments/20260604_dataset/specs.json)
- decoder checkpoint: `20`
- latent code source: [`deepsdf/experiments/20260604_dataset/LatentCodes/20.pth`](/home/tianqi/corepp2/deepsdf/experiments/20260604_dataset/LatentCodes/20.pth)
- encoder best checkpoint: [`logs/20260604_dataset/checkpoints/_20260604_dataset_best_model.pt`](/home/tianqi/corepp2/logs/20260604_dataset/checkpoints/_20260604_dataset_best_model.pt)

DeepSDF decoder / latent code：

```bash
python train_deep_sdf.py \
  --experiment ./deepsdf/experiments/20260604_dataset
```

Encoder training：

```bash
python train.py \
  --cfg ./configs/20260604_dataset.json \
  --experiment ./deepsdf/experiments/20260604_dataset \
  --checkpoint_decoder 20
```

Testing：

```bash
python test.py \
  --cfg ./configs/20260604_dataset.json \
  --experiment ./deepsdf/experiments/20260604_dataset \
  --checkpoint_decoder 20
```

## Notes

- 当前 `Point-MAE` 是本地接入的 `Point-MAE` 风格 encoder，不依赖外部预训练权重。
- 在这次实验里，复杂附加损失（`volume`、`latent spread`、`instance contrastive`、`decoder consistency`、`latent norm`、`latent mean`）全部关闭后，效果反而更好。
- 当前最有效的训练目标是：**仅使用 `SuperLoss` 对齐 DeepSDF latent code**。
