import csv
from pathlib import Path


_PREPARED_CSV_PATHS = set()


RANT_WEIGHT_FIELDS = [
    "dataset",
    "fed_alg",
    "fold",
    "epoch",
    "client_id",
    "num_samples",
    "base_weight",
    "reliability",
    "final_weight",
    "train_loss",
    "train_acc",
    "train_f1",
    "update_norm",
    "rant_reliability",
    "tau_loss",
    "tau_norm",
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
        raise FileExistsError(f"FedRANT-Lite weights CSV already exists: {csv_path}")
    if write_mode == "overwrite" and csv_path.exists():
        csv_path.unlink()

    _PREPARED_CSV_PATHS.add(prepared_key)
    return csv_path


def build_rant_weight_rows(
    args,
    fold_idx,
    epoch,
    client_ids,
    num_samples,
    base_weights,
    reliabilities,
    final_weights,
    train_results,
    update_norms,
):
    rows = []
    for idx, client_id in enumerate(client_ids):
        result = train_results[idx] if idx < len(train_results) else {}
        rows.append(
            {
                "dataset": args.dataset,
                "fed_alg": args.fed_alg,
                "fold": fold_idx,
                "epoch": epoch,
                "client_id": client_id,
                "num_samples": num_samples[idx],
                "base_weight": base_weights[idx],
                "reliability": reliabilities[idx],
                "final_weight": final_weights[idx],
                "train_loss": result.get("loss"),
                "train_acc": result.get("acc"),
                "train_f1": result.get("f1"),
                "update_norm": update_norms[idx],
                "rant_reliability": args.rant_reliability,
                "tau_loss": args.rant_tau_loss,
                "tau_norm": args.rant_tau_norm,
                "run_id": args.run_id,
            }
        )
    return rows


def write_rant_weight_rows(csv_path, rows, write_mode="append"):
    if len(rows) == 0:
        return

    csv_path = _prepare_csv_path(csv_path, write_mode)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with open(str(csv_path), "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RANT_WEIGHT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
