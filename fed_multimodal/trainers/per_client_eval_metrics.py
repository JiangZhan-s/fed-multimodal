import csv
from pathlib import Path


PER_CLIENT_EVAL_METRICS_FIELDS = [
    "dataset",
    "fed_alg",
    "fold",
    "epoch",
    "eval_stage",
    "client_id",
    "alpha",
    "sample_rate",
    "local_epochs",
    "learning_rate",
    "batch_size",
    "modality_setting",
    "missing_modality_enabled",
    "missing_modailty_rate",
    "eval_data_type",
    "num_samples",
    "eval_loss",
    "eval_acc",
    "eval_f1",
    "eval_uar",
    "eval_top5_acc",
]


def build_per_client_eval_row(
    args,
    fold_idx,
    epoch,
    eval_stage,
    client_id,
    eval_result,
    modality_setting,
    eval_data_type,
):
    return {
        "dataset": args.dataset,
        "fed_alg": args.fed_alg,
        "fold": fold_idx,
        "epoch": epoch,
        "eval_stage": eval_stage,
        "client_id": client_id,
        "alpha": args.alpha,
        "sample_rate": args.sample_rate,
        "local_epochs": args.local_epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "modality_setting": modality_setting,
        "missing_modality_enabled": args.missing_modality,
        "missing_modailty_rate": args.missing_modailty_rate,
        "eval_data_type": eval_data_type,
        "num_samples": eval_result.get("sample"),
        "eval_loss": eval_result.get("loss"),
        "eval_acc": eval_result.get("acc"),
        "eval_f1": eval_result.get("f1"),
        "eval_uar": eval_result.get("uar"),
        "eval_top5_acc": eval_result.get("top5_acc"),
    }


def append_per_client_eval_row(csv_path, row):
    write_per_client_eval_rows(csv_path, [row])


def write_per_client_eval_rows(csv_path, rows):
    if len(rows) == 0:
        return

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with open(str(csv_path), "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PER_CLIENT_EVAL_METRICS_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
