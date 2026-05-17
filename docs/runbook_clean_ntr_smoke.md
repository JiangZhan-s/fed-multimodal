# Clean NTR Smoke Test Runbook

本 runbook 用于验证 UCI-HAR 三路 1 epoch smoke test 链路：

1. `acc_gyro` multimodal
2. `acc` single-modality
3. `gyro` single-modality
4. strict NTR diagnostics

这些命令只用于链路验证，不代表研究结论。

## 1. 运行前检查

```bash
cd /home/wuyi/FedMultimodal/fed-multimodal
git status --short
git branch --show-current
git log -10 --oneline
```

要求：

- 分支为 `research/fedrant`
- 工作区 clean
- 使用新的 `run_id`

## 2. acc_gyro smoke test

```bash
cd /home/wuyi/FedMultimodal/fed-multimodal/fed_multimodal/experiment/uci-har

PYTHONPATH=/home/wuyi/FedMultimodal/fed-multimodal conda run -n zwy1 python train.py \
  --alpha 0.1 \
  --sample_rate 0.02 \
  --learning_rate 0.05 \
  --global_learning_rate 0.025 \
  --num_epochs 1 \
  --test_frequency 1 \
  --fed_alg fed_avg \
  --en_att \
  --att_name fuse_base \
  --hid_size 32 \
  --device auto \
  --fold 1 \
  --modality acc_gyro \
  --save_client_metrics \
  --save_per_client_eval \
  --enable_local_eval_split \
  --local_eval_ratio 0.2 \
  --local_eval_seed 2026 \
  --local_eval_min_samples 1 \
  --per_client_eval_data local_eval \
  --run_id smoke_clean_acc_gyro_f1 \
  --metrics_write_mode error_if_exists
```

## 3. acc-only smoke test

```bash
cd /home/wuyi/FedMultimodal/fed-multimodal/fed_multimodal/experiment/uci-har

PYTHONPATH=/home/wuyi/FedMultimodal/fed-multimodal conda run -n zwy1 python train.py \
  --alpha 0.1 \
  --sample_rate 0.02 \
  --learning_rate 0.05 \
  --global_learning_rate 0.025 \
  --num_epochs 1 \
  --test_frequency 1 \
  --fed_alg fed_avg \
  --hid_size 32 \
  --device auto \
  --fold 1 \
  --modality acc \
  --save_client_metrics \
  --save_per_client_eval \
  --enable_local_eval_split \
  --local_eval_ratio 0.2 \
  --local_eval_seed 2026 \
  --local_eval_min_samples 1 \
  --per_client_eval_data local_eval \
  --run_id smoke_clean_acc_f1 \
  --metrics_write_mode error_if_exists
```

## 4. gyro-only smoke test

```bash
cd /home/wuyi/FedMultimodal/fed-multimodal/fed_multimodal/experiment/uci-har

PYTHONPATH=/home/wuyi/FedMultimodal/fed-multimodal conda run -n zwy1 python train.py \
  --alpha 0.1 \
  --sample_rate 0.02 \
  --learning_rate 0.05 \
  --global_learning_rate 0.025 \
  --num_epochs 1 \
  --test_frequency 1 \
  --fed_alg fed_avg \
  --hid_size 32 \
  --device auto \
  --fold 1 \
  --modality gyro \
  --save_client_metrics \
  --save_per_client_eval \
  --enable_local_eval_split \
  --local_eval_ratio 0.2 \
  --local_eval_seed 2026 \
  --local_eval_min_samples 1 \
  --per_client_eval_data local_eval \
  --run_id smoke_clean_gyro_f1 \
  --metrics_write_mode error_if_exists
```

## 5. NTR diagnostics

```bash
cd /home/wuyi/FedMultimodal/fed-multimodal

PYTHONPATH=/home/wuyi/FedMultimodal/fed-multimodal conda run -n zwy1 python scripts/diagnostics/compute_negative_transfer.py \
  --multi_csv fed_multimodal/result/fed_avg/uci-har/acc_gyro/fuse_base/alpha01_hid32_le1_lr005_bs16_sr002_ep1/runs/smoke_clean_acc_gyro_f1/per_client_eval_metrics.csv \
  --single_a_csv fed_multimodal/result/fed_avg/uci-har/acc/no_att/alpha01_hid32_le1_lr005_bs16_sr002_ep1/runs/smoke_clean_acc_f1/per_client_eval_metrics.csv \
  --single_b_csv fed_multimodal/result/fed_avg/uci-har/gyro/no_att/alpha01_hid32_le1_lr005_bs16_sr002_ep1/runs/smoke_clean_gyro_f1/per_client_eval_metrics.csv \
  --single_a_name acc \
  --single_b_name gyro \
  --output_dir /tmp/fedmultimodal_clean_ntr_smoke \
  --eval_data_type local_eval_split \
  --eps 1.0 \
  --min_eval_samples 5
```

不要使用 `--no_strict_split_check`。

## 6. 输出路径

acc_gyro:

```text
fed_multimodal/result/fed_avg/uci-har/acc_gyro/fuse_base/alpha01_hid32_le1_lr005_bs16_sr002_ep1/runs/smoke_clean_acc_gyro_f1/
```

acc:

```text
fed_multimodal/result/fed_avg/uci-har/acc/no_att/alpha01_hid32_le1_lr005_bs16_sr002_ep1/runs/smoke_clean_acc_f1/
```

gyro:

```text
fed_multimodal/result/fed_avg/uci-har/gyro/no_att/alpha01_hid32_le1_lr005_bs16_sr002_ep1/runs/smoke_clean_gyro_f1/
```

每个 run 目录应包含：

- `result.json`
- `client_metrics.csv`
- `per_client_eval_metrics.csv`
- `local_eval_split_fold1.json`

诊断输出目录应包含：

- `client_level_negative_transfer.csv`
- `diagnosis_summary.csv`
- `diagnosis_config.json`

## 7. 如何判断成功

训练侧：

- 三路命令均成功退出。
- 三路都是 fold1 / 1 epoch。
- 三路 `learning_rate` 一致。
- 三路 `per_client_eval_metrics.csv` 均存在。
- `eval_data_type` 全部为 `local_eval_split`。
- `modality_setting` 分别为 `acc_gyro`、`acc`、`gyro`。
- `split_file` 指向对应 run 目录下的 `local_eval_split_fold1.json`。
- 示例 client 的 train/eval indices 三路一致。

诊断侧：

- strict mode 成功。
- `diagnosis_summary.csv` 中存在 `fold=1` 和 `fold=average`。
- `client_level_negative_transfer.csv` 中存在 `gap` 和 `is_negative_transfer`。
- `diagnosis_config.json` 记录三路 CSV 路径。

最后检查：

```bash
cd /home/wuyi/FedMultimodal/fed-multimodal
git status --short
```

应没有代码修改，也没有未忽略结果文件。

## 8. 常见错误

### run_id 已存在

现象：

```text
Metrics CSV already exists
```

原因：`--metrics_write_mode error_if_exists` 检测到同一 run_id 下已有 CSV。

处理：不要删除旧结果；换一个新的 run_id，例如追加 `_v2` 或时间戳。

### strict split check 失败

可能原因：

- 三路 `local_eval_seed` 不一致。
- 三路 `local_eval_ratio` 不一致。
- 三路 client split 样本数不一致。
- 使用了历史混合 CSV。

处理：检查三路 `diagnosis_config.json` 输入路径和 `local_eval_split_fold1.json`。

### per_client_eval_metrics.csv 混入历史追加记录

可能原因：

- 没有使用 `--run_id`。
- 使用了 `--metrics_write_mode append`。
- 复用了旧 smoke run_id。

处理：正式实验使用新的 run_id，并优先使用 `--metrics_write_mode error_if_exists`。

### learning_rate 不一致

可能原因：三路命令手动复制时遗漏参数。

处理：确认 acc_gyro、acc、gyro 三路都使用相同 `--learning_rate`。

### eval_data_type 不是 local_eval_split

可能原因：

- 未传 `--enable_local_eval_split`。
- 未传 `--per_client_eval_data local_eval`。

处理：重新使用本 runbook 中的命令模板。
