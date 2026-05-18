import csv
from pathlib import Path


_PREPARED_CSV_PATHS = set()


GATE_WEIGHT_FIELDS = [
    "dataset",
    "fed_alg",
    "fold",
    "epoch",
    "client_id",
    "num_samples",
    "mean_gate_acc",
    "mean_gate_gyro",
    "gate_entropy",
    "gate_temperature",
    "gate_entropy_reg",
    "gate_min_weight",
    "train_loss",
    "train_f1",
    "modality_setting",
    "att_name",
    "run_id",
]


def _prepare_csv_path(csv_path, write_mode):
    if write_mode not in ["append", "overwrite", "error_if_exists"]:
        raise ValueError("write_mode must be one of: append, overwrite, error_if_exists")

    csv_path = Path(csv_path)
    prepared_key = str(csv_path.resolve())
    if prepared_key in _PREPARED_CSV_PATHS:
        return csv_path

    if write_mode == "error_if_exists" and csv_path.exists():
        raise FileExistsError(f"Gate weights CSV already exists: {csv_path}")
    if write_mode == "overwrite" and csv_path.exists():
        csv_path.unlink()

    _PREPARED_CSV_PATHS.add(prepared_key)
    return csv_path


def build_gate_weight_row(
    args,
    fold_idx,
    epoch,
    client_id,
    num_samples,
    gate_stats,
    client_result,
    modality_setting,
):
    """Build one client/epoch row of reliability gate statistics."""
    return {
        "dataset": args.dataset,
        "fed_alg": args.fed_alg,
        "fold": fold_idx,
        "epoch": epoch,
        "client_id": client_id,
        "num_samples": num_samples,
        "mean_gate_acc": gate_stats.get("mean_gate_acc"),
        "mean_gate_gyro": gate_stats.get("mean_gate_gyro"),
        "gate_entropy": gate_stats.get("gate_entropy"),
        "gate_temperature": args.gate_temperature,
        "gate_entropy_reg": args.gate_entropy_reg,
        "gate_min_weight": args.gate_min_weight,
        "train_loss": client_result.get("loss"),
        "train_f1": client_result.get("f1"),
        "modality_setting": modality_setting,
        "att_name": args.att_name,
        "run_id": args.run_id,
    }


def append_gate_weight_metrics(csv_path, row, write_mode="append"):
    csv_path = _prepare_csv_path(csv_path, write_mode)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with open(str(csv_path), "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GATE_WEIGHT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
