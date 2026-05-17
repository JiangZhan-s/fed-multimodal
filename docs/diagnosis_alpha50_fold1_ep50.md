# 1. 实验目的

这是 `alpha=5.0` clean setting 对照诊断实验，用于和 `alpha=0.1` 比较 Non-IID 强弱下的 client-level negative transfer、Worst-client F1 和 Client variance。

该实验不是 FedRANT 实验，不包含 FedRANT-Lite 或 FedRANT-Full 的任何机制。它是 baseline diagnostic，用于诊断当前 FedAvg + UCI-HAR 三路输入下的客户端级表现差异。

该结果只来自 fold1，不是最终论文结论。

# 2. 实验设置

| 项目 | 设置 |
|---|---|
| dataset | UCI-HAR |
| fed_alg | fed_avg |
| alpha | 5.0 |
| fold | 1 |
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
| acc_gyro | formal_acc_gyro_alpha50_fold1_ep50_sr01_les2026 |
| acc | formal_acc_alpha50_fold1_ep50_sr01_les2026 |
| gyro | formal_gyro_alpha50_fold1_ep50_sr01_les2026 |

Diagnosis output:

```text
/tmp/fedmultimodal_ntr_formal_alpha50_fold1_ep50_sr01_les2026
```

# 3. 三路模型 result.json 结果

`result.json` average:

| Modality | run_id | F1 | Acc | Top5 Acc |
|---|---|---:|---:|---:|
| acc_gyro | formal_acc_gyro_alpha50_fold1_ep50_sr01_les2026 | 19.217786 | 32.575501 | 85.748219 |
| acc | formal_acc_alpha50_fold1_ep50_sr01_les2026 | 18.026400 | 26.942654 | 85.748219 |
| gyro | formal_gyro_alpha50_fold1_ep50_sr01_les2026 | 12.029244 | 23.549372 | 86.664404 |

# 4. NTR 诊断结果

| Metric | Value |
|---|---:|
| valid_clients | 105 |
| NTR | 0.980952 |
| negative transfer clients | 103 |
| mean_multi_f1 | 5.441510 |
| mean_best_single_f1 | 14.718671 |
| worst20_multi_f1 | 1.879602 |
| worst20_best_single_f1 | 6.965231 |
| variance_multi_f1 | 5.423324 |
| std_multi_f1 | 2.328803 |
| min_multi_f1 | 0.000000 |
| median_multi_f1 | 5.882353 |
| max_multi_f1 | 10.256410 |

# 5. 负迁移 client 明细

Gap 最小前 10 个 client:

| client_id | multi_f1 | acc_f1 | gyro_f1 | best_single_f1 | gap |
|---|---:|---:|---:|---:|---:|
| 22-4 | 8.333333 | 34.722222 | 13.095238 | 34.722222 | -26.388889 |
| 17-2 | 4.761905 | 29.761905 | 6.666667 | 29.761905 | -25.000000 |
| 16-3 | 5.555556 | 26.190476 | 8.928571 | 26.190476 | -20.634921 |
| 7-1 | 9.523810 | 30.000000 | 7.870370 | 30.000000 | -20.476190 |
| 21-0 | 8.888889 | 19.528620 | 26.666667 | 26.666667 | -17.777778 |
| 22-2 | 6.666667 | 24.444444 | 18.055556 | 24.444444 | -17.777778 |
| 14-0 | 4.444444 | 21.333333 | 11.428571 | 21.333333 | -16.888889 |
| 6-1 | 6.666667 | 22.857143 | 11.428571 | 22.857143 | -16.190476 |
| 21-2 | 6.666667 | 16.190476 | 22.222222 | 22.222222 | -15.555556 |
| 26-1 | 4.166667 | 19.528620 | 15.079365 | 19.528620 | -15.361953 |

# 6. 与 alpha=0.1 的对比

| Metric | alpha=0.1 | alpha=5.0 |
|---|---:|---:|
| NTR | 0.019048 | 0.980952 |
| negative clients | 2 | 103 |
| mean_multi_f1 | 14.060868 | 5.441510 |
| mean_best_single_f1 | 11.914613 | 14.718671 |
| worst20_multi_f1 | 0.000000 | 1.879602 |
| std_multi_f1 | 20.357985 | 2.328803 |
| median_multi_f1 | 6.250000 | 5.882353 |

# 7. 初步解释

- `alpha=5.0` 下观察到非常高的 NTR，即 103/105 clients 出现 negative transfer。
- 与 `alpha=0.1` 相比，NTR 明显升高。
- `alpha=5.0` 下 `mean_multi_f1 = 5.441510`，低于 `mean_best_single_f1 = 14.718671`，说明多模态在 client-level local_eval 上大面积弱于最佳单模态。
- 但 `result.json` 的 global average 中，`acc_gyro` F1 高于 `acc` 和 `gyro`，因此存在 global summary 与 client-level local_eval 诊断不一致的现象。
- `worst20_multi_f1` 从 `0.000000` 提升到 `1.879602`，但这不代表整体改善，因为 multi 的 client-level F1 集中在低区间，std 也显著降低。
- std 降低更可能意味着整体低性能收缩，而不是公平性改善。
- 这个结果支持继续研究 reliability-aware aggregation，但也提示需要 sanity check：fold1 偶然性、多模态 attention 配置、单模态/多模态模型容量差异、global test 与 local_eval 的差异。
- 不能把这组结果作为最终论文结论。

# 8. 下一步建议

1. 先跑更多 folds，或至少 fold2/fold3，检查 `alpha=5.0` 是否稳定。
2. 进行 missing modality 诊断，观察缺失模态是否进一步加重 NTR。
3. 在设计 FedRANT-Lite 前，需要明确主要问题是负迁移、tail-client，还是 global/local 指标不一致。
4. FedRANT-Lite 可以以 reliability-aware aggregation 为方向，但设计前应避免只根据 fold1 过度拟合。
