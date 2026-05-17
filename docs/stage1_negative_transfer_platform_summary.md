# 1. 阶段目标

1 阶段的目标不是实现 FedRANT，而是把 FedMultimodal 改造成一个可支持 UCI-HAR 客户端级负迁移诊断的实验平台。当前阶段重点是打通三路实验、客户端级评估、local holdout split、输出隔离和离线诊断链路，为后续 FedRANT-Lite 设计提供可靠数据基础。

# 2. 已实现功能

- `--device`
  - 作用：替代 UCI-HAR `train.py` 中原本硬编码的 `cuda:1`，支持 `auto`、`cpu`、`cuda`、`cuda:N`。
  - 相关文件：`fed_multimodal/experiment/uci-har/train.py`
  - 默认影响：默认 `auto`，不会改变非 CUDA 环境下的官方 CPU 行为；在单 GPU CUDA 环境下默认使用 `cuda:0`。

- `--fold`
  - 作用：支持只运行单个 fold，便于 smoke test 和调试。
  - 相关文件：`fed_multimodal/experiment/uci-har/train.py`
  - 默认影响：默认不传时仍运行 fold1 到 fold5，保持官方兼容。

- `--run_id`
  - 作用：将一次实验的输出隔离到 `runs/{run_id}/` 子目录，避免 smoke test 与正式实验互相污染。
  - 相关文件：`fed_multimodal/experiment/uci-har/train.py`
  - 默认影响：默认 `None` 时保持原有输出路径。

- `--metrics_write_mode`
  - 作用：控制 metrics CSV 写入策略，支持 `append`、`overwrite`、`error_if_exists`。
  - 相关文件：`fed_multimodal/experiment/uci-har/train.py`、`fed_multimodal/trainers/client_metrics.py`、`fed_multimodal/trainers/per_client_eval_metrics.py`
  - 默认影响：默认 `append`，保持原有追加写入行为。

- `--modality acc_gyro / acc / gyro`
  - 作用：支持 multimodal、acc-only、gyro-only 三路实验入口。
  - 相关文件：`fed_multimodal/experiment/uci-har/train.py`、`fed_multimodal/dataloader/local_eval_split.py`
  - 默认影响：默认 `acc_gyro`，保持原双模态流程；`acc` 和 `gyro` 是真单模态，不是置零模态。

- `--save_client_metrics`
  - 作用：保存每轮被采样客户端本地训练后的 train-time metrics。
  - 相关文件：`fed_multimodal/experiment/uci-har/train.py`、`fed_multimodal/trainers/client_metrics.py`
  - 默认影响：默认关闭，不影响官方输出。

- `--save_per_client_eval`
  - 作用：保存 final global model 在每个 client 数据上的 per-client evaluation metrics。
  - 相关文件：`fed_multimodal/experiment/uci-har/train.py`、`fed_multimodal/trainers/per_client_eval_metrics.py`
  - 默认影响：默认关闭，不影响官方输出。

- `--enable_local_eval_split`
  - 作用：将每个 client 内部数据划分为 local_train / local_eval，local_train 用于训练，local_eval 用于客户端级评估。
  - 相关文件：`fed_multimodal/experiment/uci-har/train.py`、`fed_multimodal/dataloader/local_eval_split.py`
  - 默认影响：默认关闭，保持官方原始训练数据使用方式。

- `local_eval_split_fold{fold}.json`
  - 作用：记录每个 client 的 train/eval indices、样本数、split 方法和 warning，保证三路实验可追踪。
  - 相关文件：`fed_multimodal/experiment/uci-har/train.py`、`fed_multimodal/dataloader/local_eval_split.py`
  - 默认影响：仅在启用 local eval split 时生成。

- `per_client_eval_metrics.csv`
  - 作用：记录客户端级 evaluation metrics，支持 `local_train_eval` 和 `local_eval_split`。
  - 相关文件：`fed_multimodal/trainers/per_client_eval_metrics.py`
  - 默认影响：仅在 `--save_per_client_eval` 时生成。

- `client_metrics.csv`
  - 作用：记录 train-time client metrics。
  - 相关文件：`fed_multimodal/trainers/client_metrics.py`
  - 默认影响：仅在 `--save_client_metrics` 时生成。

- `compute_negative_transfer.py`
  - 作用：离线读取三路 per-client eval CSV，计算 NTR、Worst-client F1、Client variance/std，并输出诊断结果。
  - 相关文件：`scripts/diagnostics/compute_negative_transfer.py`、`fed_multimodal/metrics/negative_transfer.py`
  - 默认影响：独立离线脚本，不影响训练代码。

# 3. 当前实验链路

1. 运行 `acc_gyro` multimodal UCI-HAR 实验。
2. 运行 `acc` single-modality UCI-HAR 实验。
3. 运行 `gyro` single-modality UCI-HAR 实验。
4. 三路都生成 `per_client_eval_metrics.csv`。
5. 三路都生成 `local_eval_split_fold1.json`。
6. `compute_negative_transfer.py` 读取三路 CSV，并在 strict split check 下对齐 `dataset`、`fold`、`client_id`、`eval_data_type`、local split metadata。
7. 输出 `client_level_negative_transfer.csv`、`diagnosis_summary.csv`、`diagnosis_config.json`。

# 4. 关键输出文件说明

- `result.json`
  - 官方 fold/global summary 结果文件。
  - 在使用 `--run_id` 时保存到对应 `runs/{run_id}/` 目录。

- `client_metrics.csv`
  - train-time client metrics。
  - 记录的是本地训练后 `client.result` 中已有的指标，不是最终 per-client test/eval 指标。

- `per_client_eval_metrics.csv`
  - per-client evaluation metrics。
  - `eval_data_type=local_train_eval` 表示使用本地训练数据评估，有训练集评估偏差。
  - 论文主指标应使用 `eval_data_type=local_eval_split`。

- `local_eval_split_fold{fold}.json`
  - 每个 fold 的 client local_train/local_eval split 元数据。
  - 用于追踪三路实验是否使用一致的 split indices。

- `client_level_negative_transfer.csv`
  - 每行一个 client，包含 multi F1、single-modality F1、best single F1、gap 和 `is_negative_transfer`。

- `diagnosis_summary.csv`
  - 每个 fold 和 macro average 的 NTR、mean F1、Worst-client F1、variance/std 等 summary。

- `diagnosis_config.json`
  - 记录诊断脚本输入 CSV、阈值、过滤参数和 strict split check 设置。

注意：1 epoch smoke test 不能作为研究结论，只能证明链路可运行。

# 5. NTR 指标说明

对每个 client \(k\)：

```text
best_single_f1(k) = max(F1_acc(k), F1_gyro(k))
gap(k) = F1_multi(k) - best_single_f1(k)
```

如果：

```text
gap(k) < -eps
```

则 client \(k\) 发生 negative transfer。

```text
NTR = negative_transfer_clients / valid_clients
```

当前约定：

- F1 是百分制。
- `eps` 默认值为 `1.0`。
- `min_eval_samples` 默认值为 `5`。
- valid client 必须在 multimodal、acc-only、gyro-only 三路中都有 `eval_data_type=local_eval_split` 的结果。
- 正式分析必须开启 strict split check，确保三路 local split 元数据一致。

# 6. 干净三路 smoke test 结果

本次 1I smoke test 使用：

- acc_gyro run_id：`smoke_clean_acc_gyro_f1`
- acc run_id：`smoke_clean_acc_f1`
- gyro run_id：`smoke_clean_gyro_f1`
- 三路均为 fold1 / 1 epoch。
- 三路 `learning_rate` 均为 `0.05`。
- 示例 client `1-0` 的三路 split indices 一致：
  - train head：`[0, 1, 3, 4, 5, 6, 8, 10, 11, 12]`
  - eval：`[2, 7, 9, 17, 20, 21, 38, 40, 46, 47, 49]`
- strict NTR diagnostics 成功。
- `valid_clients = 100`
- `NTR = 0.000000`
- `mean_multi_f1 = 11.153882`
- `worst20_multi_f1 = 0.000000`
- `std_multi_f1 = 19.820352`

这些数值只用于链路验证，不代表研究结论。

# 7. 标准 smoke test 命令模板

在 `fed_multimodal/experiment/uci-har` 下运行。

## acc_gyro

```bash
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

## acc

```bash
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

## gyro

```bash
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

## NTR diagnostics

```bash
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

# 8. 正式实验注意事项

- 正式实验必须使用新的 `run_id`。
- 不要复用 smoke run_id。
- 不要把 smoke test 数值写成研究结论。
- 正式实验应统一 `alpha`、`fold`、`seed`、`local_eval_ratio`、`learning_rate`、`epochs`。
- single-modality 与 multimodal 必须使用同一 local split。
- `acc` / `gyro` 是真单模态，不是置零模态。
- 单模态模型参数量与多模态不同，需要在论文里说明。
- `local_eval_split` 是我们额外引入的 client-level holdout protocol，不是 FedMultimodal 官方原始设置。

# 9. 下一阶段入口

2 阶段将进入 FedRANT-Lite 设计。

2 阶段建议目标：

- 先用现有平台做正式诊断实验，确认负迁移现象。
- 再实现 FedRANT-Lite。
- FedRANT-Lite 不应直接上复杂模块。
- 第一版先做 reliability-aware aggregation。
- 后续再考虑 FedRANT-Full。
