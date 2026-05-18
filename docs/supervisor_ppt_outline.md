# Supervisor PPT Outline

## 1. 研究问题与出发点：global average 掩盖 client-level degradation

要点：

- 研究对象是多模态联邦学习中的 client-level degradation / negative transfer。
- 当前不是简单复现 FedMultimodal 的 global average，而是先诊断多模态是否让部分 client 变差。
- 在 UCI-HAR acc/gyro 任务中，global F1/accuracy 可能无法暴露 local client 上的退化。
- 核心问题：`acc_gyro` 是否总是优于 `acc` / `gyro`，还是会在部分 client 上低于最佳单模态。

建议图表/表格：

- 一张示意图：左侧是 global average 指标，右侧是 client-level F1 gap 分布。
- 标出 `gap(k)=F1_multi(k)-max(F1_acc(k),F1_gyro(k))`，说明 gap 为负代表 client-level degradation。

这一页我要讲清楚什么：

- 我的切入点不是“多模态平均效果是否更高”，而是“多模态是否对部分 client 造成损害”。
- 先诊断问题，再设计方法，是当前研究逻辑。

## 2. 实验底座与诊断平台：FedMultimodal + local_eval_split + NTR + worst-client

要点：

- 基于 FedMultimodal KDD 2023 代码，扩展了 device、fold、run_id 和输出隔离。
- 实现 acc_gyro / acc-only / gyro-only 三路对照，且 acc/gyro 是真单模态，不是置零模态。
- 增加 `local_eval_split`，在每个 client 内部构造 local_train / local_eval holdout。
- 输出 `per_client_eval_metrics.csv`，计算 NTR、Worst-20% client F1、Client variance/std。
- strict split check 保证三路 CSV 来自同一 fold、client 和 local split。

建议图表/表格：

- 诊断流程图：三路训练 `acc_gyro / acc / gyro` -> local_eval_split -> per-client eval -> NTR / worst-client / variance。
- 小表格列出关键输出：`result.json`、`per_client_eval_metrics.csv`、`local_eval_split_fold{fold}.json`、diagnosis summary。

这一页我要讲清楚什么：

- 我先搭了一个可复现、可对齐、可追踪的 client-level 诊断平台。
- 后面的结论不是只来自 global result，而是来自三路对齐的 per-client local_eval。

## 3. 关键发现一：fuse_base 在 alpha=5.0 fold1/fold2 下出现高 NTR

要点：

- alpha=5.0 fold1 中，FedAvg fuse_base 的 NTR=0.980952，103/105 clients negative。
- alpha=5.0 fold2 中，NTR=1.000000，105/105 clients negative。
- fold1/fold2 都出现高 NTR，说明这不像是单一 fold 偶然现象。
- fuse_base 的 mean_multi_f1 在 fold1/fold2 都很低，client-level local_eval 显示严重 degradation。

建议图表/表格：

- NTR 对比柱状图：fuse_base fold1=0.980952，fuse_base fold2=1.000000。
- 表格列出 fold、negative clients、mean_multi_f1。

这一页我要讲清楚什么：

- 在 alpha=5.0 clean setting 下，fuse_base 的 client-level 表现非常异常。
- 这个发现为后续方法设计提供了问题动机，但还不能直接说明问题完全来自聚合。

## 4. 关键发现二：no-att sanity check 显著缓解 NTR，说明 fusion/attention 是风险源

要点：

- 去掉 fuse_base attention 后，FedAvg no-att 的 NTR 降到 0.085714。
- negative clients 从 103/105 降到 9/105，mean_multi_f1 提升到 27.439009。
- worst20_multi_f1 提升到 14.588930，result.json F1 为 36.609368。
- 这说明问题不能简单归因于 FedAvg 聚合，fusion/attention 配置是重要风险源之一。
- no-att 是后续必须保留的强 baseline。

建议图表/表格：

- 柱状图对比 FedAvg fuse_base fold1 vs FedAvg no-att fold1 的 NTR、negative clients。
- 方法对比表：NTR、mean_multi_f1、worst20_multi_f1、result.json F1。

这一页我要讲清楚什么：

- no-att sanity check 改变了问题判断：高 NTR 主要与 fuse_base attention/fusion 有关。
- 后续不能只说“FedAvg 聚合导致负迁移”，叙事必须更克制。

## 5. 方法尝试一：FedRANT-Lite aggregation 在 fuse_base 下只有弱改善

要点：

- FedRANT-Lite 只在 server aggregation 阶段做 reliability-aware weighting。
- 训练期 reliability 来自 train loss / update norm，不使用 local_eval、test 或 NTR 参与训练。
- 默认 loss_norm 配置没有降低 NTR；loss-only 是 fuse_base 下最好的消融配置。
- loss-only 将 NTR 从 0.980952 降到 0.876190，negative clients 从 103 降到 92。
- 但 mean_multi_f1 和 worst20_multi_f1 仍然很低，说明 aggregation-only 修复不了 fuse_base 的主要问题。

建议图表/表格：

- 消融表：FedAvg fuse_base、FedRANT default、FedRANT loss、FedRANT norm、FedRANT loss_norm tau variants。
- 重点突出 FedRANT loss 的弱改善，以及它与 no-att baseline 的差距。

这一页我要讲清楚什么：

- FedRANT-Lite aggregation 有一点信号，但在 fuse_base 下不够。
- 这支持“reliability aggregation 有价值”，但不支持“它已经解决 fuse_base degradation”。

## 6. 方法尝试二：FedRANT-Lite loss + no-att 是当前最强正向结果

要点：

- 在 no-att 设置下，FedRANT-Lite loss 相比 FedAvg no-att 进一步降低 NTR。
- NTR 从 0.085714 降到 0.047619，negative clients 从 9/105 降到 5/105。
- mean_multi_f1 提升到 32.452982，worst20_multi_f1 提升到 16.501867。
- result.json F1 为 36.637316，基本没有损害 global F1。
- 当前最可信的主线是 FedRANT-Lite loss + no-att multimodal。

建议图表/表格：

- 方法对比表：FedAvg fuse_base、FedRANT loss fuse_base、FedAvg no-att、FedRANT loss no-att。
- 字段：NTR、negative clients、mean_multi_f1、worst20_multi_f1、median_multi_f1、result.json F1。

这一页我要讲清楚什么：

- 当前最稳的正向结果不是复杂 attention，而是稳定 fusion(no-att) + 训练期 loss reliability aggregation。
- 这个结论仍主要来自 alpha=5.0 fold1，需要后续 folds 验证。

## 7. 负结果与边界：reliability_gate v1 无效，不能夸大结论

要点：

- reliability_gate v1 尝试用轻量 learnable gate 替代 fuse_base。
- gate 没有 collapse，权重大致接近平衡并略偏 acc，但没有带来 client-level 改善。
- FedAvg gate 和 FedRANT gate 的 NTR 都是 0.990476，negative clients 都是 104/105。
- 该结果说明简单 learnable gate 不是当前主线，应作为负结果保留。
- 当前不能声称已经完整解决多模态联邦负迁移。

建议图表/表格：

- gate 行为表：mean_gate_acc、mean_gate_gyro、gate_entropy。
- baseline 对比表：FedAvg fuse_base、FedAvg no-att、FedRANT no-att、FedAvg gate、FedRANT gate。

这一页我要讲清楚什么：

- 我不是只保留正结果，也记录了无效方向。
- 负结果帮助收束主线：暂时不继续盲目调 gate，而是回到 FedRANT no-att 的稳定性验证。

## 8. 当前结论与下一步：主线收束为 FedRANT no-att，后续做 fold2/fold3 和 missing modality

要点：

- 当前主方法候选：FedRANT-Lite loss + no-att multimodal。
- 该路线不使用不稳定 fuse_base，不使用 local_eval/test/NTR 参与训练，只用训练期 loss 做聚合可靠性。
- 当前证据表明它能降低 NTR、改善 worst-client，并保持 global F1 基本不受损。
- 但目前强证据主要来自 UCI-HAR alpha=5.0 fold1，仍需更多 folds 和 missing modality 验证。
- 下一步优先跑 FedRANT no-att fold2/fold3、alpha=0.1、missing modality，并分析 `rant_weights.csv`。

建议图表/表格：

- 下一步实验矩阵：fold2/fold3、alpha=0.1、missing modality、rant weight analysis。
- 用状态标注：已完成、必做验证、可选扩展。

这一页我要讲清楚什么：

- 当前结论是阶段性收束，不是最终论文结论。
- 后续重点是验证 FedRANT no-att 是否稳定，而不是继续堆复杂 attention 或大规模调 gate。
