# Figure and Table Plan

This document lists the planned figures and tables for the paper draft. The current emphasis should be on diagnostic evidence. Method claims should remain cautious because FedRANT no-att shows a fold1 positive signal but does not reproduce the gain on fold2.

## Figures

### Figure 1: Overall diagnosis pipeline

- Current file path: `docs/figures/fig1_pipeline.png`.
- Status: draft figure generated.
- Needs polishing: yes; later align typography and notation with the final paper style.
- Data source: implementation design and outputs from `local_eval_split`, `per_client_eval_metrics.csv`, and `compute_negative_transfer.py`.
- Visual design: flowchart.
- Components: FedMultimodal training -> acc_gyro / acc / gyro runs -> local_eval_split -> per-client evaluation -> NTR / worst-client / variance.
- Main conclusion: the paper evaluates client-level negative transfer through aligned multimodal and single-modality local holdout results, not only global averages.
- Current data status: available.
- Missing data: none; this is a conceptual/protocol figure.

### Figure 2: NTR comparison across methods

- Current file path: `docs/figures/fig2_ntr_comparison.png`.
- Status: draft figure generated.
- Needs polishing: yes; add final method labels after fold3/missing-modality results are decided.
- Data source: diagnosis summaries recorded in docs and `/tmp` outputs.
- X-axis: method/setting.
- Y-axis: NTR.
- Candidate bars:
  - FedAvg fuse_base fold1: 0.980952;
  - FedAvg fuse_base fold2: 1.000000;
  - FedAvg no-att fold1: 0.085714;
  - FedRANT loss no-att fold1: 0.047619;
  - FedAvg no-att fold2: 0.980952;
  - FedRANT loss no-att fold2: 0.990476;
  - reliability_gate: 0.990476.
- Main conclusion: fuse_base has severe client-level degradation; no-att fold1 greatly reduces NTR; FedRANT no-att fold1 is promising but fold2 does not reproduce the gain.
- Current data status: mostly available.
- Missing data: fold3 and missing-modality NTR.

### Figure 3: Mean F1 and Worst20 F1 comparison

- Current file path: `docs/figures/fig3_mean_worst20_f1.png`.
- Status: draft figure generated.
- Needs polishing: yes; may need grouped labels by fold and method family.
- Data source: diagnosis summaries.
- X-axis: method/setting.
- Y-axis: F1 percentage.
- Suggested grouped bars: mean_multi_f1 and worst20_multi_f1.
- Candidate settings:
  - FedAvg fuse_base fold1;
  - FedAvg no-att fold1;
  - FedRANT loss no-att fold1;
  - FedAvg no-att fold2;
  - FedRANT loss no-att fold2.
- Main conclusion: no-att improves client-level mean and worst-client metrics over fuse_base, but FedRANT aggregation gain is fold-dependent.
- Current data status: available for listed settings.
- Missing data: fold3, alpha=0.1 FedRANT no-att, missing modality.

### Figure 4: fuse_base fold1/fold2 degradation

- Current file path: `docs/figures/fig4_fuse_base_fold_degradation.png`.
- Status: draft figure generated.
- Needs polishing: minor; likely usable after label and caption cleanup.
- Data source: alpha=5.0 fold1 and fold2 fuse_base diagnosis.
- X-axis: fold.
- Y-axis: NTR or negative clients.
- Suggested visuals:
  - bar chart for NTR;
  - small table or labels for negative clients and mean_multi_f1.
- Main conclusion: high NTR appears in both fold1 and fold2, so the fuse_base degradation is not a single-fold accident.
- Current data status: available.
- Missing data: optional fold3 fuse_base if needed for stronger evidence.

### Figure 5: no-att vs fuse_base sanity check

- Current file path: `docs/figures/fig5_noatt_fold_stability.png`.
- Status: draft figure generated.
- Needs polishing: yes; the caption should clearly state that fold2 does not reproduce low NTR.
- Data source: alpha=5.0 fold1 and fold2 results.
- X-axis: fusion setting.
- Y-axis: NTR, mean_multi_f1, or worst20_multi_f1.
- Suggested visuals:
  - paired bars for FedAvg fuse_base vs FedAvg no-att on fold1;
  - optional fold2 comparison showing no-att remains better than fuse_base but still has high NTR.
- Main conclusion: removing fuse_base attention substantially changes client-level behavior, so fusion/attention is a key risk source.
- Current data status: available.
- Missing data: no-att fold3 for stability.

### Figure 6: FedRANT-Lite ablation

- Current file path: `docs/figures/fig6_fedrant_ablation_ntr.png`.
- Status: draft figure generated.
- Needs polishing: yes; simplify labels and highlight loss-only as the best aggregation-only variant.
- Data source: FedRANT-Lite ablation document.
- X-axis: reliability mode.
- Y-axis: NTR and/or mean_multi_f1.
- Candidate methods:
  - FedAvg fuse_base;
  - FedRANT default loss_norm 1/1;
  - FedRANT loss;
  - FedRANT norm;
  - FedRANT loss_norm tau_loss=2;
  - FedRANT loss_norm tau_norm=2;
  - FedAvg no-att.
- Main conclusion: loss-only is the best aggregation-only variant under fuse_base, but all fuse_base variants remain far behind no-att.
- Current data status: available.
- Missing data: none for the small ablation; avoid expanding into a large hyperparameter search unless necessary.

### Figure 7: reliability_gate negative result

- Current file path: `docs/figures/fig7_reliability_gate_negative_result.png`.
- Status: draft figure generated.
- Needs polishing: yes; pair with optional gate-weight behavior figure if space allows.
- Data source: reliability_gate diagnosis document.
- X-axis: method.
- Y-axis: NTR or mean_multi_f1.
- Candidate methods:
  - FedAvg fuse_base;
  - FedAvg no-att;
  - FedRANT no-att;
  - FedAvg gate;
  - FedRANT gate.
- Optional secondary plot: gate_acc/gate_gyro mean and gate entropy.
- Main conclusion: reliability_gate v1 does not collapse but does not improve client-level metrics; simple learnable gate is not the current main method.
- Current data status: available.
- Missing data: none unless gate variants are revisited.

## Tables

### Table 1: Diagnosis platform components

- Data source: implementation summary and Stage 1 documents.
- Columns:
  - component;
  - purpose;
  - output file;
  - whether used for training;
  - whether affects default behavior.
- Rows:
  - local_eval_split;
  - per_client_eval_metrics.csv;
  - NTR;
  - worst20 F1;
  - client variance/std;
  - run_id output isolation;
  - strict split check.
- Main conclusion: the platform separates diagnosis from training and enables aligned client-level analysis.
- Current data status: available.
- Missing data: none.

### Table 2: Main method comparison

- Data source: diagnosis summary CSVs and docs.
- Columns:
  - method;
  - fold;
  - attention/fusion;
  - NTR;
  - negative clients;
  - mean_multi_f1;
  - worst20_multi_f1;
  - median_multi_f1;
  - result.json F1.
- Candidate rows:
  - FedAvg fuse_base fold1;
  - FedAvg fuse_base fold2;
  - FedAvg no-att fold1;
  - FedRANT loss no-att fold1;
  - FedAvg no-att fold2;
  - FedRANT loss no-att fold2;
  - reliability_gate rows as negative result.
- Main conclusion: no-att is a strong baseline; FedRANT no-att has fold1 signal but fold2 does not reproduce the gain.
- Current data status: available for listed rows.
- Missing data: fold3, alpha=0.1 FedRANT no-att, missing modality.

### Table 3: FedRANT-Lite ablation

- Data source: FedRANT-Lite ablation results.
- Columns:
  - reliability mode;
  - tau_loss;
  - tau_norm;
  - NTR;
  - negative clients;
  - mean_multi_f1;
  - worst20_multi_f1;
  - result.json F1;
  - reliability min/mean/max.
- Main conclusion: loss-only is the strongest current FedRANT-Lite variant under fuse_base, but improvement is limited.
- Current data status: available.
- Missing data: none for current scope.

### Table 4: Fold1 vs fold2 stability

- Data source: fold1 and fold2 diagnostics.
- Columns:
  - method;
  - fold;
  - NTR;
  - negative clients;
  - mean_multi_f1;
  - worst20_multi_f1;
  - std_multi_f1;
  - median_multi_f1;
  - result.json F1.
- Rows:
  - FedAvg no-att fold1;
  - FedRANT no-att fold1;
  - FedAvg no-att fold2;
  - FedRANT no-att fold2;
  - FedAvg fuse_base fold2 as reference.
- Main conclusion: fold2 confirms no-att is better than fuse_base, but does not confirm FedRANT no-att improvement.
- Current data status: available.
- Missing data: fold3.

### Table 5: Missing modality results placeholder

- Data source: future missing-modality experiments.
- Columns:
  - missing setting;
  - method;
  - fold;
  - NTR;
  - negative clients;
  - mean_multi_f1;
  - worst20_multi_f1;
  - result.json F1.
- Candidate methods:
  - FedAvg fuse_base;
  - FedAvg no-att;
  - FedRANT loss no-att.
- Main conclusion: not yet available; this table will test whether missing modality amplifies client-level degradation and whether FedRANT no-att remains useful.
- Current data status: not available.
- Missing data:
  - missing modality runs;
  - strict NTR diagnostics;
  - at least one fold, preferably multiple folds.

## Current figures that can be drawn immediately

1. Figure 1: overall diagnosis pipeline.
2. Figure 2: NTR comparison across existing methods, with a caveat about fold2.
3. Figure 3: mean/worst20 F1 comparison for existing fold1/fold2 results.
4. Figure 4: fuse_base fold1/fold2 degradation.
5. Figure 5: no-att vs fuse_base sanity check.
6. Figure 6: FedRANT-Lite ablation.
7. Figure 7: reliability_gate negative result.

## Data still needed before stronger paper claims

1. FedRANT loss no-att fold3.
2. FedRANT loss no-att under alpha=0.1.
3. Missing-modality diagnostics.
4. Client-level `rant_weights.csv` analysis across folds.
5. Optional no-att fold3 baseline if fold3 FedRANT is run.
