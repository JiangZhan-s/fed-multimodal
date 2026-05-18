# Diagnosis Index

This index tracks formal clean baseline diagnostics for UCI-HAR client-level negative transfer analysis.

| Setting | Document | Notes |
|---|---|---|
| alpha=0.1, fold1, ep50, clean baseline | `docs/diagnosis_alpha01_fold1_ep50.md` | First formal baseline diagnosis. |
| alpha=5.0, fold1, ep50, clean baseline | `docs/diagnosis_alpha50_fold1_ep50.md` | Clean setting comparison against alpha=0.1. |
| alpha=5.0, fold2, ep50, clean baseline | `docs/diagnosis_alpha50_fold2_ep50.md` | Fold2 reproduced high NTR observed in alpha=5.0 fold1. |
| alpha=5.0, fold1, ep50, no-att acc_gyro sanity | `docs/diagnosis_alpha50_fold1_noatt_ep50.md` | No-att acc_gyro substantially reduced NTR compared with fuse_base. |
| FedRANT-Lite default, alpha=5.0, fold1, ep50 | `docs/diagnosis_fedrant_lite_alpha50_fold1_ep50.md` | FedRANT-Lite default loss_norm aggregation was interpretable but did not reduce NTR relative to FedAvg fuse_base. |
| FedRANT-Lite reliability/tau ablation, alpha=5.0, fold1, ep50 | `docs/diagnosis_fedrant_lite_ablation_alpha50_fold1_ep50.md` | Loss-only reliability was the best aggregation-only variant but still far from no-att baseline. |
| FedRANT-Lite loss no-att, alpha=5.0, fold1, ep50 | `docs/diagnosis_fedrant_lite_noatt_alpha50_fold1_ep50.md` | FedRANT-Lite loss improved the no-att baseline, reducing NTR from 0.085714 to 0.047619. |
| reliability_gate, alpha=5.0, fold1, ep50 | `docs/diagnosis_reliability_gate_alpha50_fold1_ep50.md` | reliability_gate did not outperform no-att; FedRANT no-att remains the strongest current method candidate. |
| Current research summary | `docs/current_research_summary.md` | Stage summary for reporting and follow-up paper planning. |

These experiments are clean baseline diagnostics. They are useful for planning follow-up experiments, but they are not final paper conclusions.
