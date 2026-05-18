# Current Research Summary

## 1. 当前研究问题

本课题关注多模态联邦学习中的 client-level degradation / negative transfer。当前工作不是简单复现 FedMultimodal，也不是直接堆叠复杂模块，而是先建立诊断平台，确认问题是否真实存在、出现在哪些设置下、以及哪些设计可能缓解或加重问题。

当前诊断对象是 UCI-HAR 的 acc/gyro 多模态联邦学习。核心问题是：多模态 `acc_gyro` 是否总是优于 `acc` 或 `gyro` 单模态？如果 global average 看起来尚可，是否仍然存在局部 client 被多模态融合损害？

## 2. 已建立的诊断平台

当前平台已经支持以下能力：

- `local_eval_split`：在每个 client 内部构造 local_train / local_eval holdout，local_train 用于训练，local_eval 用于严格 per-client evaluation。
- `per_client_eval_metrics.csv`：保存 final global model 在每个 client local_eval 上的 metrics。
- NTR：negative transfer rate，衡量 multi F1 低于 best single F1 的 client 比例。
- Worst-client F1：例如 bottom 20% clients 的平均 F1。
- Client variance/std：衡量 client 间性能差异。
- `acc_gyro / acc / gyro` 三路对照：支持真单模态 baseline，不使用置零模态伪造。
- `run_id` 输出隔离：避免 smoke test 和正式实验输出互相污染。
- strict split check：确保 multi / acc / gyro 三路结果来自一致的 fold、client、local split。
- `docs/runbook_clean_ntr_smoke.md`：记录干净三路 smoke test 操作流程。

这些能力是后续实验可信性的基础。尤其是 local_eval_split、strict split check 和 run_id 输出隔离，使得 per-client 诊断不依赖混杂的历史 CSV 或 train-time metrics。

## 3. 关键诊断发现

### 3.1 alpha=0.1 clean setting

| metric | value |
|---|---:|
| NTR | 0.019048 |
| negative clients | 2 / 105 |
| mean_multi_f1 | 14.060868 |
| worst20_multi_f1 | 0.000000 |

在 alpha=0.1 clean setting 下，观察到少量 client-level negative transfer。NTR 约为 1.9%，负迁移比例较低。但 tail-client 表现很弱，`worst20_multi_f1 = 0.000000`，说明平均表现之外仍存在尾部 client 问题。

### 3.2 alpha=5.0 fuse_base fold1/fold2

| fold | NTR | negative clients | mean_multi_f1 |
|---|---:|---:|---:|
| fold1 | 0.980952 | 103 / 105 | 5.441510 |
| fold2 | 1.000000 | 105 / 105 | 5.223903 |

alpha=5.0 下，高 NTR 在 fold1 和 fold2 都出现，不是单一 fold 偶然现象。`fuse_base` 在 client-level local_eval 上表现出严重 degradation：绝大多数 clients 的 multimodal F1 低于 best single-modality F1。

这也暴露出 global summary 与 client-level local_eval 之间的不一致：某些 result.json global average 并不能充分揭示 per-client degradation。

### 3.3 no-att sanity check

| metric | FedAvg no-att |
|---|---:|
| NTR | 0.085714 |
| negative clients | 9 / 105 |
| mean_multi_f1 | 27.439009 |
| worst20_multi_f1 | 14.588930 |
| result.json F1 | 36.609368 |

去掉 `fuse_base` 后，NTR 从 0.980952 大幅下降到 0.085714，mean/worst20/median client F1 都明显改善。这说明 fusion/attention 是主要风险源之一，`no-att` 是当前必须保留的强 baseline。

## 4. 方法尝试与结果

### 4.1 FedRANT-Lite aggregation

FedRANT-Lite 只在 server aggregation 阶段做 reliability-aware weighting，不改模型结构，不改数据，不使用 local_eval/test/NTR 参与训练。第一版使用训练期 `train_loss` 和 `update_norm` 计算 reliability，并输出 `rant_weights.csv` 用于解释聚合权重。

默认 `loss_norm` 结果：

| metric | value |
|---|---:|
| NTR | 0.980952 |
| mean_multi_f1 | 5.751975 |
| result.json F1 | 13.983259 |

loss-only 最好消融：

| metric | value |
|---|---:|
| NTR | 0.876190 |
| negative clients | 92 |
| mean_multi_f1 | 7.610062 |
| worst20_multi_f1 | 2.350736 |
| result.json F1 | 19.581561 |

结论：aggregation-only 在 fuse_base 下只有弱改善，不能解决主要问题；但 loss reliability 有信号，说明训练期 reliability-aware aggregation 并非完全无效。

### 4.2 FedRANT-Lite loss + no-att

| metric | value |
|---|---:|
| NTR | 0.047619 |
| negative clients | 5 / 105 |
| mean_multi_f1 | 32.452982 |
| worst20_multi_f1 | 16.501867 |
| result.json F1 | 36.637316 |

这是当前最强正向结果。相比 FedAvg no-att，FedRANT-Lite loss + no-att 进一步降低 NTR，并提高 mean/worst20/median client F1，同时不损害 global F1。

当前主方法候选应是 **FedRANT-Lite loss + no-att multimodal**。

### 4.3 reliability_gate v1

| method | NTR | negative clients | mean_multi_f1 | result.json F1 |
|---|---:|---:|---:|---:|
| FedAvg gate | 0.990476 | 104 / 105 | 5.322727 | 11.880765 |
| FedRANT gate | 0.990476 | 104 / 105 | 5.322727 | 11.987033 |

reliability_gate 没有 collapse，gate weights 接近平衡并略偏 acc；但没有带来 client-level 改善。FedRANT aggregation 与 gate 没有形成有效叠加。

结论：简单 learnable gate 不是当前主线，应作为负结果/诊断结果保留。

## 5. 当前主线判断

当前最可信路线：

**FedRANT-Lite loss + no-att multimodal**

原因：

- no-att 避免了 fuse_base 的严重 client-level degradation。
- FedRANT-Lite loss 在 no-att 基础上进一步降低 NTR。
- 训练期只使用 train loss 作为 aggregation reliability 信号。
- 不使用 local_eval/test/NTR 参与训练，避免诊断信号泄漏。
- global F1 未受损，甚至相对 FedAvg no-att 略有提升。

这不是最终结论。目前最强证据主要来自 fold1 / alpha=5.0。仍需更多 folds、missing modality 和其他 alpha 设置验证，才能写成更强论文结论。

## 6. 负结果与边界

需要诚实记录当前边界：

- FedRANT-Lite aggregation 在 fuse_base 下只能弱改善。
- norm/update_norm reliability 基本无效。
- reliability_gate v1 无效。
- 不能声称已经解决所有多模态负迁移。
- 当前结论只支持：在当前 UCI-HAR / FedMultimodal 设置下，no-att + loss reliability 是更稳路线。

这些负结果很重要：它们约束了论文叙事，避免把问题过度归因于 FedAvg 聚合，也避免把简单 gate 写成有效方法。

## 7. 下一步计划

### A. 必做验证

1. FedRANT-Lite loss + no-att fold2。
2. FedRANT-Lite loss + no-att alpha=0.1。
3. missing modality setting 下对比。
4. 至少 3 folds 后再写强结论。

### B. 可选扩展

1. gate_entropy_reg 小范围 sanity。
2. no-att + FedRANT 在 FedProx/FedOpt 下的适配。
3. 对 `rant_weights.csv` 做 client-level 可解释性分析。

### C. 暂不建议

1. 大规模 tau 搜索。
2. 继续盲目调 reliability_gate。
3. 堆复杂 attention 模块。

## 8. 论文叙事草案

我们首先构建 client-level local holdout 评估协议，并提出 NTR 等诊断指标。诊断显示，FedMultimodal 中 fuse_base attention 在部分设置下会导致严重 client-level degradation，而 global average 指标不能充分暴露该问题。进一步 sanity check 表明，去除 attention 的 no-att multimodal 能显著降低 NTR。基于此，我们设计训练期可靠性感知聚合 FedRANT-Lite。实验发现，FedRANT-Lite 在 fuse_base 下只能弱缓解，但与 no-att 结合后能进一步降低 NTR 并改善 worst-client 表现，且不损害 global F1。与此同时，简单 reliability_gate 未能改善，提示后续 fusion reliability 需要更强约束或更明确的模态质量估计。
