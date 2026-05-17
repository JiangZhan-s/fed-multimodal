# 1. 实验目的

这是 `acc_gyro` no-att multimodal sanity check，用于判断 `alpha=5.0` fold1 高 NTR 是否主要来自 `fuse_base` attention/fusion 配置。

该实验不是 FedRANT 实验，不包含 FedRANT-Lite 或 FedRANT-Full 的任何机制。它是 baseline sanity check，用于检查在不启用 `fuse_base` attention 时，`acc_gyro` 多模态模型的 client-level negative transfer 是否仍然严重。

该结果不是最终论文结论。

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
| modality | acc_gyro |
| attention setting | no-att |
| local_eval_ratio | 0.2 |
| local_eval_seed | 2026 |
| local_eval_min_samples | 5 |
| eval_data_type | local_eval_split |
| run_id | formal_acc_gyro_noatt_alpha50_fold1_ep50_sr01_les2026 |

Diagnosis output:

```text
/tmp/fedmultimodal_ntr_formal_alpha50_fold1_ep50_noatt_les2026
```

# 3. no-att acc_gyro result.json 结果

`result.json` average:

| Metric | Value |
|---|---:|
| F1 | 36.609368 |
| Acc | 43.942993 |
| Top5 Acc | 94.944011 |

# 4. no-att NTR 诊断结果

| Metric | Value |
|---|---:|
| valid_clients | 105 |
| NTR | 0.085714 |
| negative clients | 9 |
| mean_multi_f1 | 27.439009 |
| mean_best_single_f1 | 14.718671 |
| worst20_multi_f1 | 14.588930 |
| worst20_best_single_f1 | 6.965231 |
| std_multi_f1 | 10.565644 |
| median_multi_f1 | 26.111111 |

# 5. 与 fuse_base fold1 对比

| Metric | fuse_base fold1 | no-att fold1 |
|---|---:|---:|
| NTR | 0.980952 | 0.085714 |
| negative clients | 103 | 9 |
| mean_multi_f1 | 5.441510 | 27.439009 |
| mean_best_single_f1 | 14.718671 | 14.718671 |
| worst20_multi_f1 | 1.879602 | 14.588930 |
| std_multi_f1 | 2.328803 | 10.565644 |
| median_multi_f1 | 5.882353 | 26.111111 |

# 6. 初步解释

- no-att 明显缓解了 `alpha=5.0` fold1 的高 NTR。
- NTR 从 98.1% 降到 8.6%。
- `mean_multi_f1`、`worst20_multi_f1`、`median_multi_f1` 都明显改善。
- 这说明 `fuse_base` attention/fusion 配置可能是高 NTR 的主要来源之一。
- 因此后续不能把问题简单归因于 FedAvg 聚合。
- FedRANT-Lite 仍然可以作为 reliability-aware aggregation 缓解机制，但必须加入 no-att multimodal 作为强 baseline。
- 后续 FedRANT-Lite 的比较对象至少应包括：
  1. FedAvg `acc_gyro` fuse_base
  2. FedAvg `acc_gyro` no-att
  3. FedAvg acc-only
  4. FedAvg gyro-only
  5. FedRANT-Lite `acc_gyro`
- 如果 FedRANT-Lite 只优于 fuse_base 但不优于 no-att，则方法说服力不足。
- 不能把这组结果作为最终论文结论。

# 7. 下一步建议

1. 记录后可以进入 FedRANT-Lite 实现。
2. FedRANT-Lite 的动机要改写为 "reliability-aware aggregation for mitigating client-level degradation under multimodal FL"，而不是单纯 "FedAvg causes negative transfer"。
3. no-att multimodal 必须作为后续实验 baseline。
4. 后续仍可补 fold2 no-att 或 missing modality，但不必阻塞 FedRANT-Lite v1 实现。
