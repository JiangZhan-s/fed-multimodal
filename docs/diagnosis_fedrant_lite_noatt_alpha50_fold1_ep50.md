# 1. 实验目的

This document records a FedRANT-Lite no-att control experiment. The goal is to test whether reliability-aware aggregation still provides gains after removing the fuse_base attention setting.

This is not a final paper conclusion. It is a method comparison on alpha=5.0 fold1. FedRANT-Lite does not use local_eval, test, or NTR during training. This experiment is used to judge whether aggregation reliability remains useful in a healthier no-att setting.

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
| attention | no-att |
| rant_reliability | loss |
| rant_tau_loss | 1.0 |
| rant_tau_norm | 1.0 |
| rant_min_weight | 0.05 |
| rant_max_weight | 5.0 |
| local_eval_ratio | 0.2 |
| local_eval_seed | 2026 |
| local_eval_min_samples | 5 |
| eval_data_type | local_eval_split |
| run_id | fedrant_loss_noatt_alpha50_fold1_ep50_sr01_les2026 |
| diagnosis output_dir | /tmp/fedmultimodal_ntr_fedrant_loss_noatt_alpha50_fold1_ep50_sr01_les2026 |

# 3. FedRANT-Lite no-att result.json 结果

| metric | value |
|---|---:|
| F1 | 36.637316 |
| Acc | 43.332202 |
| Top5 Acc | 90.736342 |

# 4. FedRANT-Lite no-att NTR 诊断结果

| metric | value |
|---|---:|
| valid_clients | 105 |
| NTR | 0.047619 |
| negative clients | 5 |
| mean_multi_f1 | 32.452982 |
| mean_best_single_f1 | 14.718671 |
| worst20_multi_f1 | 16.501867 |
| worst20_best_single_f1 | 6.965231 |
| variance_multi_f1 | 120.534509 |
| std_multi_f1 | 10.978821 |
| min_multi_f1 | 4.166667 |
| median_multi_f1 | 33.333333 |
| max_multi_f1 | 52.777778 |

# 5. rant_weights 分析

| metric | value |
|---|---:|
| rows | 500 |
| reliability min | 0.367879 |
| reliability mean | 0.629750 |
| reliability max | 1.000000 |
| final_weight min | 0.033811 |
| final_weight mean | 0.100000 |
| final_weight max | 0.239829 |
| update_norm min | 0.017041 |
| update_norm mean | 0.122739 |
| update_norm max | 0.479133 |
| train_loss min | 1.288581 |
| train_loss mean | 1.736237 |
| train_loss max | 1.904596 |
| NaN/inf | none observed |
| final_weight sum per epoch | approximately 1 |

The weight mechanism worked normally. In the no-att setting, reliability-aware aggregation did not break training. The global F1 did not decrease; it slightly improved over the FedAvg no-att baseline.

# 6. 与 baseline 对比

| method | NTR | negative clients | mean_multi_f1 | worst20_multi_f1 | std_multi_f1 | median_multi_f1 | result.json F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FedAvg fuse_base | 0.980952 | 103 | 5.441510 | 1.879602 | 2.328803 | 5.882353 | 19.217786 |
| FedRANT loss fuse_base | 0.876190 | 92 | 7.610062 | 2.350736 | 5.373914 | 6.666667 | 19.581561 |
| FedAvg no-att | 0.085714 | 9 | 27.439009 | 14.588930 | 10.565644 | 26.111111 | 36.609368 |
| FedRANT loss no-att | 0.047619 | 5 | 32.452982 | 16.501867 | 10.978821 | 33.333333 | 36.637316 |

# 7. 初步解释

FedRANT-Lite no-att improves over FedAvg no-att. NTR drops from 0.085714 to 0.047619, and negative clients drop from 9 to 5.

mean_multi_f1, worst20_multi_f1, and median_multi_f1 all improve. The result.json F1 is not harmed; it slightly increases from 36.609368 to 36.637316.

This suggests that reliability-aware aggregation has value in the healthier no-att setting. However, FedRANT-Lite still struggles under fuse_base, which indicates that fuse_base attention/fusion is one of the main bottlenecks. The next research direction should move from pure aggregation repair toward fusion reliability plus aggregation reliability.

FedRANT-Lite loss can be retained as a reliable aggregation component. The no-att baseline must remain a strong baseline in future comparisons. This fold1 result should not be treated as a final paper conclusion.

# 8. 下一步建议

1. Design fusion reliability or attention-side modifications.
2. Keep FedRANT-Lite loss as an aggregation component.
3. Future formal experiments should at least compare FedAvg fuse_base, FedAvg no-att, FedRANT loss no-att, and the new fusion-reliability method.
4. Additional no-att fold2 or missing-modality diagnosis can be added later, but it does not need to block the next design step.
