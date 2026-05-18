# 1. 实验目的

This document records a small FedRANT-Lite reliability/tau ablation. The goal is to check whether the default FedRANT-Lite failure mainly comes from the reliability type or tau setting, not to perform a large hyperparameter search.

This is not a final method conclusion. It is a fold1 / alpha=5.0 / fuse_base ablation. FedRANT-Lite does not use local_eval, test, or NTR during training; these signals are only used for post-training diagnosis.

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
| eval_data_type | local_eval_split |

Four FedRANT-Lite configurations were tested:

| config | rant_reliability | tau_loss | tau_norm |
|---|---|---:|---:|
| loss | loss | 1.0 | 1.0 |
| norm | norm | 1.0 | 1.0 |
| loss_norm tau_loss=2 | loss_norm | 2.0 | 1.0 |
| loss_norm tau_norm=2 | loss_norm | 1.0 | 2.0 |

# 3. rant_weights 概况

| config | reliability min/mean/max | final_weight min/mean/max | update_norm min/mean/max | train_loss min/mean/max | NaN/inf | final_weight sum≈1 |
|---|---:|---:|---:|---:|---|---|
| loss | 0.367879 / 0.622343 / 1.000000 | 0.032428 / 0.100000 / 0.231374 | 0.018853 / 0.091593 / 0.290044 | 1.571396 / 1.771326 / 1.899194 | none | yes |
| norm | 0.500000 / 0.739642 / 1.000000 | 0.049277 / 0.100000 / 0.157169 | 0.014984 / 0.086129 / 0.244348 | 1.622411 / 1.772168 / 1.869742 | none | yes |
| loss_norm tau_loss=2 | 0.067668 / 0.322422 / 1.000000 | 0.011219 / 0.100000 / 0.363218 | 0.018853 / 0.095769 / 0.315557 | 1.549073 / 1.774883 / 1.934509 | none | yes |
| loss_norm tau_norm=2 | 0.122626 / 0.384647 / 1.000000 | 0.020273 / 0.100000 / 0.236323 | 0.016009 / 0.088650 / 0.261561 | 1.586659 / 1.773503 / 1.905845 | none | yes |

# 4. NTR 诊断结果

| config | result.json F1 | NTR | negative clients | mean_multi_f1 | worst20_multi_f1 | std_multi_f1 | median_multi_f1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| loss | 19.581561 | 0.876190 | 92 | 7.610062 | 2.350736 | 5.373914 | 6.666667 |
| norm | 18.743459 | 0.980952 | 103 | 5.454815 | 1.879602 | 2.341077 | 5.882353 |
| loss_norm tau_loss=2 | 14.346725 | 0.904762 | 95 | 6.516228 | 2.313612 | 2.802033 | 6.666667 |
| loss_norm tau_norm=2 | 14.304504 | 0.980952 | 103 | 5.692306 | 1.982770 | 2.469184 | 6.060606 |

# 5. 与 baseline 对比

| method | NTR | negative clients | mean_multi_f1 | worst20_multi_f1 | std_multi_f1 | median_multi_f1 | result.json F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FedAvg fuse_base | 0.980952 | 103 | 5.441510 | 1.879602 | 2.328803 | 5.882353 | 19.217786 |
| FedRANT default loss_norm 1/1 | 0.980952 | 103 | 5.751975 | 1.982770 | 2.518914 | 6.060606 | 13.983259 |
| FedRANT loss | 0.876190 | 92 | 7.610062 | 2.350736 | 5.373914 | 6.666667 | 19.581561 |
| FedRANT norm | 0.980952 | 103 | 5.454815 | 1.879602 | 2.341077 | 5.882353 | 18.743459 |
| FedRANT loss_norm tau_loss=2 | 0.904762 | 95 | 6.516228 | 2.313612 | 2.802033 | 6.666667 | 14.346725 |
| FedRANT loss_norm tau_norm=2 | 0.980952 | 103 | 5.692306 | 1.982770 | 2.469184 | 6.060606 | 14.304504 |
| FedAvg no-att | 0.085714 | 9 | 27.439009 | 14.588930 | 10.565644 | 26.111111 | 36.609368 |

# 6. 初步解释

Loss-only is the best FedRANT-Lite configuration in this ablation. It reduces NTR from 0.980952 to 0.876190 and reduces negative clients from 103 to 92. It also slightly improves mean_multi_f1, worst20_multi_f1, and result.json F1.

Norm-only provides almost no improvement. Increasing tau_norm also does not improve NTR. Increasing tau_loss helps somewhat, but it is still worse than loss-only.

All FedRANT-Lite variants remain far behind the no-att baseline. This suggests that aggregation reliability has a weak but real signal, especially through train_loss, but it is not enough to solve the main client-level degradation under fuse_base. The dominant issue may still come from fusion/attention behavior rather than aggregation alone.

Based on this ablation, it is not recommended to write the current FedRANT-Lite result as significantly effective. It is also not worth launching a broad tau search before checking stronger baselines and the fusion side.

# 7. 下一步建议

1. Run a FedRANT-Lite no-att control.
2. If no-att + FedRANT improves over no-att FedAvg, reliability aggregation may still be useful under a healthier fusion setting.
3. If no-att + FedRANT does not improve, shift toward fusion reliability or attention redesign.
4. Treat loss-only as the better FedRANT-Lite v1 configuration instead of the default loss_norm setting.
