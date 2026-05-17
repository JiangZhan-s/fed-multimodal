import csv
from pathlib import Path


CLIENT_METRICS_FIELDS = [
    "dataset",
    "fed_alg",
    "fold",
    "epoch",
    "client_id",
    "alpha",
    "sample_rate",
    "local_epochs",
    "learning_rate",
    "batch_size",
    "modality_setting",
    "missing_modality_enabled",
    "missing_modailty_rate",
    "num_samples",
    "train_loss",
    "train_acc",
    "train_f1",
    "train_uar",
    "train_top5_acc",
]


def build_client_metrics_row(
    args,
    fold_idx,
    epoch,
    client_id,
    client_result,
    modality_setting,
):
    return {
        "dataset": args.dataset,
        "fed_alg": args.fed_alg,
        "fold": fold_idx,
        "epoch": epoch,
        "client_id": client_id,
        "alpha": args.alpha,
        "sample_rate": args.sample_rate,
        "local_epochs": args.local_epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "modality_setting": modality_setting,
        "missing_modality_enabled": args.missing_modality,
        "missing_modailty_rate": args.missing_modailty_rate,
        "num_samples": client_result.get("sample"),
        "train_loss": client_result.get("loss"),
        "train_acc": client_result.get("acc"),
        "train_f1": client_result.get("f1"),
        "train_uar": client_result.get("uar"),
        "train_top5_acc": client_result.get("top5_acc"),
    }


def append_client_metrics(csv_path, row):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with open(str(csv_path), "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CLIENT_METRICS_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
