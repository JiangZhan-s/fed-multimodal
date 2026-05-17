# 1. 实验目的

这是 `alpha=5.0` fold2 sanity check，用于验证 `alpha=5.0` fold1 中观察到的高 NTR 是否可能只是单个 fold 的偶然现象。

该实验不是 FedRANT 实验，不包含 FedRANT-Lite 或 FedRANT-Full 的任何机制。它是 baseline diagnostic，用于检查当前 FedAvg + UCI-HAR clean setting 下的客户端级负迁移是否跨 fold 复现。

该结果是 fold2 sanity check，不是最终论文结论。

# 2. 实验设置

| 项目 | 设置 |
|---|---|
| dataset | UCI-HAR |
| fed_alg | fed_avg |
| alpha | 5.0 |
| fold | 2 |
| num_epochs | 50 |
| sample_rate | 0.1 |
| learning_rate | 0.05 |
| global_learning_rate | 0.025 |
| local_eval_ratio | 0.2 |
| local_eval_seed | 2026 |
| local_eval_min_samples | 5 |
| eval_data_type | local_eval_split |
| modalities | acc_gyro, acc, gyro |

Run IDs:

| Modality | run_id |
|---|---|
| acc_gyro | formal_acc_gyro_alpha50_fold2_ep50_sr01_les2026 |
| acc | formal_acc_alpha50_fold2_ep50_sr01_les2026 |
| gyro | formal_gyro_alpha50_fold2_ep50_sr01_les2026 |

Diagnosis output:

```text
/tmp/fedmultimodal_ntr_formal_alpha50_fold2_ep50_sr01_les2026
```

# 3. 三路模型 result.json 结果

`result.json` average:

| Modality | run_id | F1 | Acc | Top5 Acc |
|---|---|---:|---:|---:|
| acc_gyro | formal_acc_gyro_alpha50_fold2_ep50_sr01_les2026 | 12.848941 | 25.551408 | 84.017645 |
| acc | formal_acc_alpha50_fold2_ep50_sr01_les2026 | 30.301167 | 41.398032 | 91.279267 |
| gyro | formal_gyro_alpha50_fold2_ep50_sr01_les2026 | 18.693623 | 27.349847 | 85.510689 |

# 4. NTR 诊断结果

| Metric | Value |
|---|---:|
| valid_clients | 105 |
| NTR | 1.000000 |
| negative transfer clients | 105 |
| mean_multi_f1 | 5.223903 |
| mean_best_single_f1 | 30.827522 |
| worst20_multi_f1 | 1.956166 |
| worst20_best_single_f1 | 20.159990 |
| variance_multi_f1 | 4.585310 |
| std_multi_f1 | 2.141334 |
| min_multi_f1 | 0.000000 |
| median_multi_f1 | 5.555556 |
| max_multi_f1 | 10.256410 |

# 5. 负迁移 client 明细

Gap 最小前 10 个 client:

| client_id | multi_f1 | acc_f1 | gyro_f1 | best_single_f1 | gap |
|---|---:|---:|---:|---:|---:|
| 11-0 | 3.333333 | 51.111111 | 5.555556 | 51.111111 | -47.777778 |
| 27-0 | 7.407407 | 53.240741 | 30.808081 | 53.240741 | -45.833333 |
| 3-4 | 0.000000 | 45.299145 | 11.428571 | 45.299145 | -45.299145 |
| 27-2 | 6.250000 | 51.388889 | 20.833333 | 51.388889 | -45.138889 |
| 5-2 | 3.333333 | 48.333333 | 16.666667 | 48.333333 | -45.000000 |
| 5-3 | 5.128205 | 49.523810 | 9.259259 | 49.523810 | -44.395604 |
| 27-1 | 4.444444 | 46.666667 | 40.740741 | 46.666667 | -42.222222 |
| 19-1 | 5.128205 | 45.079365 | 16.666667 | 45.079365 | -39.951160 |
| 19-4 | 5.555556 | 45.185185 | 27.994228 | 45.185185 | -39.629630 |
| 1-3 | 2.564103 | 40.277778 | 23.611111 | 40.277778 | -37.713675 |

# 6. 与 alpha=5.0 fold1 的对比

| Metric | fold1 | fold2 |
|---|---:|---:|
| NTR | 0.980952 | 1.000000 |
| negative clients | 103 | 105 |
| mean_multi_f1 | 5.441510 | 5.223903 |
| mean_best_single_f1 | 14.718671 | 30.827522 |
| worst20_multi_f1 | 1.879602 | 1.956166 |
| std_multi_f1 | 2.328803 | 2.141334 |
| median_multi_f1 | 5.882353 | 5.555556 |

# 7. 初步解释

- fold2 复现了 `alpha=5.0` 的高 NTR。
- fold2 中 105/105 clients 都发生 negative transfer。
- fold1 的高 NTR 不是单一偶然现象；至少 fold1 和 fold2 都显示 `acc_gyro` 在 client-level local_eval 上大面积弱于最佳单模态。
- `alpha=5.0` 下 multi client-level F1 在 fold1/fold2 都集中在较低区间。
- fold2 中 acc-only 表现显著强于 `acc_gyro`，提示多模态 fusion/aggregation 在 local_eval 上可能存在系统性劣化。
- 这支持进入 FedRANT-Lite 设计，但仍建议补 fold3 或 missing modality。
- 当前问题更像 client-level negative transfer + global/local 指标不一致，也可能涉及 fusion/attention 配置。
- 不能把这组结果作为最终论文结论。

# 8. 下一步建议

1. 可以进入 FedRANT-Lite 设计。
2. 但建议同时保留 fold3 或 missing modality 作为后续验证。
3. FedRANT-Lite 应聚焦 reliability-aware aggregation，而不是复杂模块堆叠。
4. 设计时要避免只针对 `alpha=5.0` fold1/fold2 过拟合。
