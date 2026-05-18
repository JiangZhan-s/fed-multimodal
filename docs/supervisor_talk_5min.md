# 1. 开场：我现在做的不是简单复现

老师，我目前是基于 FedMultimodal KDD 2023 的代码在做实验底座，但我没有直接把重点放在简单复现全局平均结果上。我的切入点是：多模态联邦学习里，global average 可能看起来还可以，但某些 client 反而因为多模态融合变差。也就是说，平均指标可能掩盖 client-level degradation 或 negative transfer。

所以我前一阶段主要不是先堆新模块，而是先搭建客户端级诊断平台，确认这个问题是否存在、在哪些设置下严重、以及哪些设计可能缓解。

# 2. 我先搭了什么诊断能力

我现在已经在 UCI-HAR 的 acc/gyro 多模态任务上搭好了几项诊断能力。首先，我加了 local_eval_split，在每个 client 内部把数据分成 local_train 和 local_eval，避免只看训练集指标。然后我支持了 acc_gyro、acc-only、gyro-only 三路对照，并且保存 final global model 在每个 client local_eval 上的 per-client eval。

在指标上，我实现了 NTR，也就是 multimodal F1 低于 best single-modality F1 的 client 比例；同时也看 Worst-client F1 和 client variance/std。再加上 run_id 输出隔离和 strict split check，保证三路实验的 client 和 local split 是对齐的。这样我就能回答一个更细的问题：多模态是不是对某些客户端反而造成了损害。

# 3. 我发现的关键问题

第一个关键发现是，在 alpha=5.0 下，FedAvg + fuse_base 出现了非常严重的 client-level degradation。fold1 里 NTR 是 0.980952，也就是 103/105 个 client 是 negative；fold2 里 NTR 到了 1.0，105/105 个 client 都是 negative。这说明这个现象不是单一 fold 的偶然结果。

第二个关键发现是，no-att sanity check 改变了我对问题来源的判断。去掉 fuse_base attention 后，FedAvg no-att 的 NTR 降到 0.085714，negative clients 只有 9/105，mean_multi_f1 提升到 27.439009。这个结果说明，问题不能简单归因于 FedAvg 聚合本身，fuse_base 这种 fusion/attention 配置是很重要的风险源。

所以现在我的判断是：这个任务里真正危险的地方不是“多模态一定不好”，而是某些 fusion 方式在联邦、Non-IID、client-level local_eval 下会造成局部退化。

# 4. 我做了哪些方法尝试

我做的第一个方法尝试是 FedRANT-Lite aggregation。它只在 server aggregation 阶段做 reliability-aware weighting，信号来自训练期 loss 和 update norm，不使用 local_eval、test 或 NTR 参与训练。这个方法在 fuse_base 下有一点弱改善。比如 loss-only 配置能把 NTR 从 0.980952 降到 0.876190，negative clients 从 103 降到 92，但这个改善还远远不够，说明单纯聚合修复不了 fuse_base 的主要问题。

第二个尝试是把 FedRANT-Lite loss aggregation 和 no-att 结合。这个是目前最好的正向结果。FedAvg no-att 的 NTR 是 0.085714，而 FedRANT loss no-att 进一步降到 0.047619，negative clients 从 9 降到 5，mean_multi_f1 和 worst20_multi_f1 都提升，global F1 也没有下降。这个结果说明，在比较健康的 no-att fusion 设置下，基于训练期 loss 的可靠聚合是有价值的。

第三个尝试是 reliability_gate。我实现了一个轻量 learnable modality gate，想看能不能替代 fuse_base，并自动学习 acc/gyro 权重。结果是 gate 没有 collapse，但效果不好，FedAvg gate 和 FedRANT gate 的 NTR 都是 0.990476，negative clients 都是 104/105。所以简单 learnable gate 目前不是主线，更适合作为负结果记录。

# 5. 当前判断

目前我认为最稳的主线不是复杂 attention，也不是单纯聚合修复，而是：去掉不稳定的 fuse_base attention，采用 no-att 多模态融合，再叠加基于训练期 loss 的可靠聚合。

这个方向的好处是结构简单，不使用测试信息，也不使用 local_eval 或 NTR 作弊；同时有 client-level 指标支持，能降低 NTR、改善 worst-client 表现，并且目前没有损害 global F1。

当然，这还不是最终结论。现在最强结果主要来自 UCI-HAR alpha=5.0 fold1，还需要补 fold2/fold3、alpha=0.1，以及 missing modality 设置，才能确认它是否稳定。

# 6. 下一步计划

接下来我计划先做三件事。第一，跑 FedRANT no-att 的 fold2 和 fold3，检查这个正向结果是不是稳定。第二，跑 alpha=0.1 和 missing modality 设置，看它是否只在 alpha=5.0 下有效，还是在更一般的设置下也能降低 NTR。第三，分析 rant_weights.csv，看哪些 client 被降权、这些 client 是否和 negative clients 有关系。

如果这些结果稳定，我后面会把论文主线整理成“client-level degradation diagnosis + reliable aggregation”。其中 fuse_base 和 reliability_gate 可以作为诊断和负结果，FedRANT no-att 作为当前方法候选。

# 7. 结尾一句话

目前我的工作重点是从客户端级诊断出发，识别多模态联邦中 fusion 引起的局部退化，并用简单可靠的 loss-based aggregation 在稳定 fusion 设置下进行缓解。
