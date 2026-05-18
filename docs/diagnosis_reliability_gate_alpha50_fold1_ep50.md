# Reliability Gate Diagnosis, Alpha 5.0 Fold1 Ep50

## 1. 实验目的

这是 reliability_gate v1 的 ep50 诊断实验，用于判断轻量 learnable modality gate 是否能替代 fuse_base，并接近或超过 no-att baseline。

这是一组方法诊断，不是最终论文结论。reliability_gate 不使用 local_eval、test 或 NTR 参与训练。本实验用于判断 gate 是否值得作为后续主线。

## 2. 实验设置

| item | value |
|---|---|
| dataset | UCI-HAR |
| alpha | 5.0 |
| fold | 1 |
| epochs | 50 |
| sample_rate | 0.1 |
| learning_rate | 0.05 |
| global_learning_rate | 0.025 |
| modality | acc_gyro |
| attention | reliability_gate |
| gate_temperature | 1.0 |
| gate_entropy_reg | 0.0 |
| gate_min_weight | 0.0 |
| local_eval_ratio | 0.2 |
| local_eval_seed | 2026 |
| local_eval_min_samples | 5 |
| eval_data_type | local_eval_split |

| method | run_id | diagnosis output_dir |
|---|---|---|
| FedAvg gate | `formal_fedavg_gate_alpha50_fold1_ep50_sr01_les2026` | `/tmp/fedmultimodal_ntr_fedavg_gate_alpha50_fold1_ep50_sr01_les2026` |
| FedRANT gate | `formal_fedrant_loss_gate_alpha50_fold1_ep50_sr01_les2026` | `/tmp/fedmultimodal_ntr_fedrant_loss_gate_alpha50_fold1_ep50_sr01_les2026` |

## 3. result.json 结果

| method | F1 | Acc | Top5 Acc |
|---|---:|---:|---:|
| FedAvg gate | 11.880765 | 21.886664 | 96.369189 |
| FedRANT gate | 11.987033 | 22.972514 | 96.063794 |

## 4. gate_weights 结果

| method | gate_acc min/mean/max | gate_gyro min/mean/max | gate_entropy min/mean/max |
|---|---:|---:|---:|
| FedAvg gate | 0.553931 / 0.565049 / 0.573122 | 0.426878 / 0.434951 / 0.446069 | 0.682402 / 0.684605 / 0.687312 |
| FedRANT gate | 0.553931 / 0.564303 / 0.574394 | 0.425606 / 0.435697 / 0.446069 | 0.682022 / 0.684794 / 0.687312 |

The gate did not collapse. It slightly favored the accelerometer modality, while entropy stayed close to log(2), suggesting near-balanced fusion. However, this gate behavior did not translate into client-level improvement.

## 5. NTR 诊断结果

| method | valid_clients | NTR | negative clients | mean_multi_f1 | mean_best_single_f1 | worst20_multi_f1 | worst20_best_single_f1 | std_multi_f1 | median_multi_f1 | result.json F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FedAvg gate | 105 | 0.990476 | 104 | 5.322727 | 14.718671 | 1.809645 | 6.965231 | 2.249769 | 5.555556 | 11.880765 |
| FedRANT gate | 105 | 0.990476 | 104 | 5.322727 | 14.718671 | 1.809645 | 6.965231 | 2.249769 | 5.555556 | 11.987033 |

## 6. baseline 对比

| method | NTR | negative clients | mean_multi_f1 | worst20_multi_f1 | std_multi_f1 | median_multi_f1 | result.json F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FedAvg fuse_base | 0.980952 | 103 | 5.441510 | 1.879602 | 2.328803 | 5.882353 | 19.217786 |
| FedAvg no-att | 0.085714 | 9 | 27.439009 | 14.588930 | 10.565644 | 26.111111 | 36.609368 |
| FedRANT no-att | 0.047619 | 5 | 32.452982 | 16.501867 | 10.978821 | 33.333333 | 36.637316 |
| FedAvg gate | 0.990476 | 104 | 5.322727 | 1.809645 | 2.249769 | 5.555556 | 11.880765 |
| FedRANT gate | 0.990476 | 104 | 5.322727 | 1.809645 | 2.249769 | 5.555556 | 11.987033 |

## 7. 初步解释

reliability_gate v1 当前无效。FedAvg gate 和 FedRANT gate 的 NTR 都是 0.990476，negative clients 都是 104。

The gate did not collapse, but performance remained close to or worse than fuse_base. FedRANT aggregation and the gate did not form an effective combined improvement. The no-att baseline remains substantially stronger, and FedRANT no-att is currently the most credible positive method signal.

This result suggests that a simple learnable gate is not sufficient to solve the observed client-level degradation. reliability_gate v1 should be treated as a negative or diagnostic result, not as the main method direction.

If the fusion direction is continued, it likely needs a stronger constraint or a clearer structure, such as starting from stable no-att fusion with reliability-aware aggregation, or designing a more explicit modality-quality estimator.

## 8. 下一步建议

1. 暂时停止 reliability_gate v1 主线。
2. 保留 FedRANT no-att 作为当前主方法候选。
3. 做阶段总结，整理完整证据链。
4. 后续可补 missing modality 或 no-att fold2，但不应继续盲目调 gate.
