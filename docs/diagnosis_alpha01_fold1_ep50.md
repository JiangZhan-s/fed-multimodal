# 1. 实验目的

这是第一组正式 baseline 诊断实验，用于观察 UCI-HAR clean setting 下是否存在 client-level negative transfer。该实验不是 FedRANT 实验，不包含 FedRANT-Lite 或 FedRANT-Full 的任何机制，只用于诊断多模态 `acc_gyro` 与 `acc` / `gyro` 真单模态 baseline 之间的客户端级差异。

# 2. 实验设置

| 项目 | 设置 |
|---|---|
| dataset | UCI-HAR |
| fed_alg | fed_avg |
| alpha | 0.1 |
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
| acc_gyro | formal_acc_gyro_alpha01_fold1_ep50_sr01_les2026 |
| acc | formal_acc_alpha01_fold1_ep50_sr01_les2026 |
| gyro | formal_gyro_alpha01_fold1_ep50_sr01_les2026 |

Diagnosis output:

```text
/tmp/fedmultimodal_ntr_formal_alpha01_fold1_ep50_sr01_les2026
```

# 3. 三路模型结果

`result.json` average:

| Modality | F1 | Acc | Top5 Acc |
|---|---:|---:|---:|
| acc_gyro | 15.498910 | 30.777061 | 82.015609 |
| acc | 15.216362 | 25.449610 | 83.338989 |
| gyro | 12.226031 | 25.279946 | 86.087547 |

# 4. NTR 诊断结果

| Metric | Value |
|---|---:|
| valid_clients | 105 |
| NTR | 0.019048 |
| negative transfer clients | 2 |
| mean_multi_f1 | 14.060868 |
| mean_best_single_f1 | 11.914613 |
| worst20_multi_f1 | 0.000000 |
| worst20_best_single_f1 | 0.000000 |
| variance_multi_f1 | 414.447571 |
| std_multi_f1 | 20.357985 |
| min_multi_f1 | 0.000000 |
| median_multi_f1 | 6.250000 |
| max_multi_f1 | 100.000000 |

# 5. 负迁移 client 明细

| client_id | multi_f1 | acc_f1 | gyro_f1 | best_single_f1 | gap |
|---|---:|---:|---:|---:|---:|
| 23-4 | 14.285714 | 14.285714 | 15.789474 | 15.789474 | -1.503759 |
| 16-2 | 7.500000 | 8.333333 | 8.823529 | 8.823529 | -1.323529 |

# 6. 初步解释

- 已观察到少量 client-level negative transfer。
- `NTR = 0.019048`，约为 1.9%，说明在当前 clean setting 下负迁移比例较低。
- 更明显的问题是 tail-client 表现弱：`worst20_multi_f1 = 0.000000`，`median_multi_f1 = 6.250000`。
- `mean_multi_f1 = 14.060868` 高于 `mean_best_single_f1 = 11.914613`，说明平均性能提升和少量客户端级受损可以同时存在。
- 该结果可以作为 FedRANT-Lite 的初步动机：即使平均表现不差，仍有客户端受损和尾部表现问题。
- 该结果还不够充分，不能作为最终论文结论。它目前只是 fold1、alpha=0.1、clean setting 下的一组 baseline 诊断结果。

# 7. 下一步实验建议

1. 跑 `alpha=5.0` clean setting，对比 Non-IID 强弱变化。
2. 跑更多 folds，确认当前结果不是 fold1 偶然现象。
3. 后续加入 missing modality，观察 NTR 是否升高。
4. 再根据诊断结果设计 FedRANT-Lite，优先考虑 reliability-aware aggregation，而不是直接上复杂模块。
