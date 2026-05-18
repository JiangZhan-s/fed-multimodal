# Supervisor Briefing

## 1. 研究问题与动机

当前研究对象是多模态联邦学习中的客户端级退化 / 负迁移。我的目标不是简单复现 FedMultimodal 的全局平均结果，而是先建立一套能观察 client-level 问题的诊断流程，再根据诊断结果设计方法。

目前以 UCI-HAR 的 acc/gyro 多模态联邦学习为主要对象。已有实验显示，global average 指标可能掩盖 client-level degradation：某些设置下 `acc_gyro` 的全局结果看起来并不差，但大量 client 的 local_eval F1 低于其最佳单模态结果。因此，我把研究重点放在“多模态是否对部分客户端造成负迁移，以及如何缓解”上。

## 2. 我已经完成的基础工作

我基于 FedMultimodal KDD 2023 代码做了诊断平台扩展。主要包括：

- 增加 `--device`、`--fold`、`--run_id` 和输出隔离，方便稳定复现实验；
- 增加 `acc_gyro / acc / gyro` 三路实验入口，支持真实单模态对照；
- 增加 `local_eval_split`，在每个 client 内部划分 local_train / local_eval；
- 增加 final global model 的 per-client evaluation 输出；
- 实现 NTR、Worst-client F1、Client variance/std 等离线诊断指标；
- 实现 FedRANT-Lite aggregation，用训练期 loss/update norm 做聚合权重调整；
- 实现 reliability_gate，并记录其负结果。

这些工作是为了让实验不只停留在全局平均 accuracy/F1，而是能够检查每个 client 是否从多模态中受益，尤其是 tail clients 和 negative transfer clients。

## 3. 关键发现

### 3.1 fuse_base 存在严重 client-level degradation

在 alpha=5.0 clean setting 下，FedAvg + fuse_base 的结果非常异常：

| setting | NTR | negative clients | mean_multi_f1 |
|---|---:|---:|---:|
| alpha=5.0 fold1 | 0.980952 | 103/105 | 5.441510 |
| alpha=5.0 fold2 | 1.000000 | 105/105 | 5.223903 |

这说明 `fuse_base` 在 client local_eval 上大面积弱于最佳单模态，并且 fold1/fold2 都出现高 NTR，不像是单个 fold 的偶然现象。

### 3.2 no-att sanity check 改变了问题判断

去掉 fuse_base attention 后，FedAvg no-att 的表现明显改善：

| method | NTR | negative clients | mean_multi_f1 | result.json F1 |
|---|---:|---:|---:|---:|
| FedAvg no-att | 0.085714 | 9/105 | 27.439009 | 36.609368 |

这说明问题不能简单归因于 FedAvg 聚合本身。fusion/attention 配置是重要风险源，no-att 是一个很强、也必须保留的 baseline。

### 3.3 FedRANT-Lite aggregation 有弱信号

FedRANT-Lite 只在 server aggregation 阶段使用训练期 reliability，不使用 local_eval/test/NTR 参与训练。在 fuse_base 下，最好的 loss-only 配置为：

| method | NTR | negative clients | mean_multi_f1 |
|---|---:|---:|---:|
| FedRANT loss fuse_base | 0.876190 | 92 | 7.610062 |

相比 FedAvg fuse_base 有弱改善，但仍无法修复 fuse_base 的主要问题。

### 3.4 当前最强组合是 FedRANT-Lite loss + no-att

当把 FedRANT-Lite loss aggregation 与 no-att multimodal 结合时，目前得到最好的结果：

| method | NTR | negative clients | mean_multi_f1 | worst20_multi_f1 | result.json F1 |
|---|---:|---:|---:|---:|---:|
| FedRANT loss no-att | 0.047619 | 5/105 | 32.452982 | 16.501867 | 36.637316 |

相比 FedAvg no-att，它进一步降低 NTR，提高 mean/worst20/median client F1，而且 global F1 没有受损。这是当前最可信的正向结果。

### 3.5 reliability_gate v1 是负结果

我也实现了一个轻量 learnable modality gate，但结果不好：

| method | NTR | negative clients |
|---|---:|---:|
| FedAvg gate | 0.990476 | 104/105 |
| FedRANT gate | 0.990476 | 104/105 |

gate 没有 collapse，但效果很差，说明简单可学习 gate 不是当前主线，应作为负结果记录。

## 4. 当前方法判断

当前主方法候选是：

**FedRANT-Lite loss + no-att multimodal**

它的含义是：

- 不使用不稳定的 fuse_base attention；
- 保留 acc+gyro 多模态输入；
- 用训练期 loss 估计 client reliability；
- 在 server aggregation 阶段对 client update 重加权；
- 不使用 local_eval/test/NTR 参与训练；
- 用 NTR、Worst-client F1 和 variance/std 做诊断评估。

这个判断还不是最终结论。目前关键正结果主要来自 alpha=5.0 fold1，还需要更多 folds、alpha=0.1 和 missing modality 验证。

## 5. 创新性体现

相对简单复现，当前工作的区别在于：

1. 不只跑 FedAvg/FedProx/FedOpt 的全局平均结果；
2. 构建了 client-level local holdout 评估协议；
3. 定义并计算 NTR；
4. 用 `acc_gyro` 与 `acc/gyro` 对齐比较诊断负迁移；
5. 发现 fusion/attention 可能造成严重客户端级退化；
6. 设计并实现 reliability-aware aggregation；
7. 记录正结果和负结果，并根据诊断逐步修正研究方向。

## 6. 当前局限

目前还需要克制解释：

- 主要实验仍集中在 UCI-HAR；
- 关键正结果主要来自 alpha=5.0 fold1；
- FedRANT no-att 还需要 fold2/fold3 验证；
- missing modality 还没有系统验证；
- reliability_gate v1 失败；
- 当前不能声称已经完整解决多模态联邦负迁移。

## 7. 下一步计划

1. 稳定性验证：
   - 跑 FedRANT no-att fold2 / fold3；
   - 在 alpha=0.1 下验证；
   - 在 missing modality setting 下验证。

2. 结果分析：
   - 分析 `rant_weights.csv`；
   - 查看哪些 client 被降权；
   - 对比 negative clients 是否减少。

3. 论文方向：
   - 以“client-level degradation diagnosis + reliable aggregation”作为主线；
   - 将 fuse_base / reliability_gate 作为诊断和负结果；
   - 将 FedRANT no-att 作为当前方法候选。

## 8. 给老师看的简短结论

目前我不是直接复现 FedMultimodal，而是先构建了客户端级诊断平台，发现 fuse_base attention 在某些设置下会造成严重 client-level degradation。进一步对照发现，no-att 能显著缓解该问题，而基于训练期 loss 的可靠聚合在 no-att 上可以进一步降低 NTR、改善 worst-client 表现。当前最稳的方向是 FedRANT-Lite loss + no-att，但仍需更多 folds 和 missing modality 验证。
