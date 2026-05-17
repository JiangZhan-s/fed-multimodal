import csv
from pathlib import Path


_PREPARED_CSV_PATHS = set()


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
    "local_eval_ratio",
    "local_eval_seed",
    "local_train_samples",
    "local_eval_samples",
    "split_file",
    "num_samples",
    "eval_loss",
    "eval_acc",
    "eval_f1",
    "eval_uar",
    "eval_top5_acc",
]


def _prepare_csv_path(csv_path, write_mode):
    if write_mode not in ["append", "overwrite", "error_if_exists"]:
        raise ValueError("write_mode must be one of: append, overwrite, error_if_exists")

    csv_path = Path(csv_path)
    prepared_key = str(csv_path.resolve())
    if prepared_key in _PREPARED_CSV_PATHS:
        return csv_path

    if write_mode == "error_if_exists" and csv_path.exists():
        raise FileExistsError(f"Per-client eval metrics CSV already exists: {csv_path}")
    if write_mode == "overwrite" and csv_path.exists():
        csv_path.unlink()

    _PREPARED_CSV_PATHS.add(prepared_key)
    return csv_path


def build_per_client_eval_row(
    args,
    fold_idx,
    epoch,
    eval_stage,
    client_id,
    eval_result,
    modality_setting,
    eval_data_type,
    split_info=None,
    split_file=None,
):
    split_info = split_info or {}
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
        "local_eval_ratio": getattr(args, "local_eval_ratio", None) if eval_data_type == "local_eval_split" else None,
        "local_eval_seed": getattr(args, "local_eval_seed", None) if eval_data_type == "local_eval_split" else None,
        "local_train_samples": split_info.get("local_train_samples"),
        "local_eval_samples": split_info.get("local_eval_samples"),
        "split_file": split_file,
        "num_samples": eval_result.get("sample"),
        "eval_loss": eval_result.get("loss"),
        "eval_acc": eval_result.get("acc"),
        "eval_f1": eval_result.get("f1"),
        "eval_uar": eval_result.get("uar"),
        "eval_top5_acc": eval_result.get("top5_acc"),
    }


def append_per_client_eval_row(csv_path, row, write_mode="append"):
    write_per_client_eval_rows(csv_path, [row], write_mode=write_mode)


def write_per_client_eval_rows(csv_path, rows, write_mode="append"):
    if len(rows) == 0:
        return

    csv_path = _prepare_csv_path(csv_path, write_mode)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    if not write_header:
        with open(str(csv_path), "r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != PER_CLIENT_EVAL_METRICS_FIELDS:
                existing_rows = list()
                for existing_row in reader:
                    existing_rows.append(
                        {
                            field: existing_row.get(field)
                            for field in PER_CLIENT_EVAL_METRICS_FIELDS
                        }
                    )
                with open(str(csv_path), "w", newline="") as rewrite_f:
                    writer = csv.DictWriter(rewrite_f, fieldnames=PER_CLIENT_EVAL_METRICS_FIELDS)
                    writer.writeheader()
                    writer.writerows(existing_rows)

    with open(str(csv_path), "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PER_CLIENT_EVAL_METRICS_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
