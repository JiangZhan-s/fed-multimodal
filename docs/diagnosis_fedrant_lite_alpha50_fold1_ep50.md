# 1. 实验目的

这是 FedRANT-Lite v1 默认配置的第一组方法诊断实验，用于检验 reliability-aware aggregation 是否能缓解 alpha=5.0 fold1 fuse_base 下的 client-level negative transfer。

这只是方法初测，不是最终实验结论。FedRANT-Lite 不使用 local_eval、test 或 NTR 参与训练；这些信号只用于训练后的诊断分析。当前目标不是击败所有 baseline，而是验证基于训练期 train_loss 和 update_norm 的 reliability-aware aggregation 是否有初步缓解作用。

# 2. 实验设置

| field | value |
|---|---|
| dataset | UCI-HAR |
| fed_alg | fed_rant_lite |
| alpha | 5.0 |
| fold | 1 |
| num_epochs | 50 |
| sample_rate | 0.1 |
| learning_rate | 0.05 |
| global_learning_rate | 0.025 |
| modality | acc_gyro |
| attention | fuse_base |
| local_eval_ratio | 0.2 |
| local_eval_seed | 2026 |
| local_eval_min_samples | 5 |
| eval_data_type | local_eval_split |
| rant_reliability | loss_norm |
| rant_tau_loss | 1.0 |
| rant_tau_norm | 1.0 |
| rant_min_weight | 0.05 |
| rant_max_weight | 5.0 |
| run_id | formal_fedrant_lite_acc_gyro_alpha50_fold1_ep50_sr01_les2026 |
| diagnosis output_dir | /tmp/fedmultimodal_ntr_fedrant_lite_alpha50_fold1_ep50_sr01_les2026 |

# 3. FedRANT-Lite result.json 结果

| metric | value |
|---|---:|
| F1 | 13.983259 |
| Acc | 24.703088 |
| Top5 Acc | 84.017645 |

# 4. FedRANT-Lite NTR 诊断结果

| metric | value |
|---|---:|
| valid_clients | 105 |
| NTR | 0.980952 |
| negative clients | 103 |
| mean_multi_f1 | 5.751975 |
| mean_best_single_f1 | 14.718671 |
| worst20_multi_f1 | 1.982770 |
| worst20_best_single_f1 | 6.965231 |
| variance_multi_f1 | 6.344925 |
| std_multi_f1 | 2.518914 |
| min_multi_f1 | 0.000000 |
| median_multi_f1 | 6.060606 |
| max_multi_f1 | 12.121212 |

# 5. rant_weights 分析

| metric | value |
|---|---:|
| rows | 500 |
| reliability min | 0.183940 |
| reliability mean | 0.463288 |
| reliability max | 1.000000 |
| final_weight min | 0.025944 |
| final_weight mean | 0.100000 |
| final_weight max | 0.214482 |
| update_norm min | 0.017515 |
| update_norm mean | 0.089730 |
| update_norm max | 0.272851 |
| train_loss min | 1.580697 |
| train_loss mean | 1.772751 |
| train_loss max | 1.904002 |
| NaN/inf | none observed |
| corr(final_weight, train_loss) | about -0.725 |
| corr(final_weight, update_norm) | about -0.299 |

The aggregation weights are interpretable: clients with higher train_loss or larger update_norm tend to receive lower final aggregation weights. However, this reliability signal did not significantly reduce NTR under the fuse_base setting.

# 6. 与 baseline 对比

| method | NTR | negative clients | mean_multi_f1 | worst20_multi_f1 | std_multi_f1 | median_multi_f1 | result.json F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FedAvg fuse_base | 0.980952 | 103 | 5.441510 | 1.879602 | 2.328803 | 5.882353 | 19.217786 |
| FedAvg no-att | 0.085714 | 9 | 27.439009 | 14.588930 | 10.565644 | 26.111111 | 36.609368 |
| FedRANT-Lite fuse_base | 0.980952 | 103 | 5.751975 | 1.982770 | 2.518914 | 6.060606 | 13.983259 |

# 7. 初步解释

FedRANT-Lite 默认配置相对 FedAvg fuse_base 只有轻微 client-level 改善。mean_multi_f1、worst20_multi_f1 和 median_multi_f1 略有提升，但 NTR 和 negative clients 没有下降。

The result.json F1 is also lower than FedAvg fuse_base. FedRANT-Lite is far from the no-att baseline, which remains much stronger in this setting.

This suggests that the default loss_norm reliability signal cannot solve the large-scale client-level degradation induced under the fuse_base setting. The result supports a more cautious interpretation: aggregation reliability may provide partial mitigation, but FedRANT-Lite v1 has not solved the problem. The main issue may still come from fusion/attention configuration, or from signals that are too weak to identify harmful updates.

This result should not be used as a final positive conclusion.

# 8. 下一步建议

1. Run a small ablation over reliability modes: loss, norm, and loss_norm.
2. Tune tau_loss and tau_norm to test whether stronger weighting changes NTR or tail performance.
3. Run a FedRANT-Lite no-att control to separate aggregation effects from fuse_base fusion effects.
4. If the ablation still fails to reduce NTR, shift toward fusion reliability or attention-side redesign.
