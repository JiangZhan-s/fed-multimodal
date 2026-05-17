# Diagnosis Index

This index tracks formal clean baseline diagnostics for UCI-HAR client-level negative transfer analysis.

| Setting | Document | Notes |
|---|---|---|
| alpha=0.1, fold1, ep50, clean baseline | `docs/diagnosis_alpha01_fold1_ep50.md` | First formal baseline diagnosis. |
| alpha=5.0, fold1, ep50, clean baseline | `docs/diagnosis_alpha50_fold1_ep50.md` | Clean setting comparison against alpha=0.1. |
| alpha=5.0, fold2, ep50, clean baseline | `docs/diagnosis_alpha50_fold2_ep50.md` | Fold2 reproduced high NTR observed in alpha=5.0 fold1. |
| alpha=5.0, fold1, ep50, no-att acc_gyro sanity | `docs/diagnosis_alpha50_fold1_noatt_ep50.md` | No-att acc_gyro substantially reduced NTR compared with fuse_base. |

These experiments are clean baseline diagnostics. They are useful for planning follow-up experiments, but they are not final paper conclusions.
