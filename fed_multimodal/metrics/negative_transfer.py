"""Offline diagnostics for client-level negative transfer."""

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "dataset",
    "fed_alg",
    "fold",
    "client_id",
    "eval_data_type",
    "eval_f1",
    "eval_acc",
    "local_eval_samples",
    "local_eval_seed",
    "local_eval_ratio",
    "split_file",
}

ALIGN_KEYS = ["dataset", "fold", "client_id", "eval_data_type"]


def _is_missing(value):
    return pd.isna(value) or str(value).strip() == ""


def normalize_int_like(value, field_name="value"):
    """Normalize int-like values from CSV, accepting forms like 1, 1.0, or fold1."""
    if _is_missing(value):
        return np.nan

    text = str(value).strip()
    if field_name == "fold" and text.lower().startswith("fold"):
        text = text[4:]

    try:
        numeric = float(text)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be int-like, got {value!r}") from exc

    rounded = int(round(numeric))
    if not math.isclose(numeric, rounded, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"{field_name} must be int-like, got {value!r}")
    return rounded


def normalize_float_like(value):
    """Normalize float-like values from CSV while preserving missing values."""
    if _is_missing(value):
        return np.nan
    return float(str(value).strip())


def values_equal_numeric(a, b, tol=1e-8):
    """Return True when two CSV values are numerically equivalent."""
    if _is_missing(a) and _is_missing(b):
        return True
    if _is_missing(a) or _is_missing(b):
        return False
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)


def normalize_split_check_fields(df):
    """Normalize split metadata used for strict consistency checks."""
    df = df.copy()
    for col in df.columns:
        if col == "fold":
            df[col] = df[col].apply(lambda value: str(normalize_int_like(value, "fold")))
        elif col.endswith("local_eval_seed") or "local_eval_seed_" in col:
            df[col] = df[col].apply(lambda value: normalize_int_like(value, "local_eval_seed"))
        elif col.endswith("local_eval_samples") or "local_eval_samples_" in col:
            df[col] = df[col].apply(lambda value: normalize_int_like(value, "local_eval_samples"))
        elif col.endswith("local_eval_ratio") or "local_eval_ratio_" in col:
            df[col] = df[col].apply(normalize_float_like)
        elif col.endswith("missing_modailty_rate") or "missing_modailty_rate_" in col:
            df[col] = df[col].apply(normalize_float_like)
    return df


def load_per_client_eval_csv(path, eval_data_type="local_eval_split", metric="eval_f1"):
    """Load a per-client eval CSV and filter it to one evaluation data type."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Per-client eval CSV not found: {path}")

    df = pd.read_csv(path)
    required = set(REQUIRED_COLUMNS)
    required.add(metric)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required column(s): {missing}")

    df = df[df["eval_data_type"] == eval_data_type].copy()
    if df.empty:
        raise ValueError(f"{path} has no rows with eval_data_type={eval_data_type!r}")

    numeric_cols = [
        metric,
        "eval_f1",
        "eval_acc",
        "local_eval_samples",
        "local_eval_seed",
        "local_eval_ratio",
        "alpha",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["fold"] = df["fold"].apply(lambda value: str(normalize_int_like(value, "fold")))
    df["client_id"] = df["client_id"].astype(str)

    duplicate_mask = df.duplicated(ALIGN_KEYS, keep=False)
    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())
        warnings.warn(
            f"Found {duplicate_count} duplicate rows for {ALIGN_KEYS}; keeping the last row per key.",
            RuntimeWarning,
        )
        df = df.drop_duplicates(ALIGN_KEYS, keep="last")
    return df


def filter_valid_clients(df, min_eval_samples=5):
    """Filter clients with too few local eval samples and return filtered notes."""
    if "local_eval_samples" not in df.columns:
        raise ValueError("local_eval_samples column is required for filtering.")

    df = df.copy()
    df["local_eval_samples"] = pd.to_numeric(df["local_eval_samples"], errors="coerce")
    valid_mask = df["local_eval_samples"] >= min_eval_samples

    valid_df = df[valid_mask].copy()
    filtered_df = df[~valid_mask].copy()
    if not filtered_df.empty:
        filtered_df["notes"] = f"filtered_low_samples_lt_{min_eval_samples}"

    return valid_df, filtered_df


def _key_set(df):
    return set(map(tuple, df[ALIGN_KEYS].astype(str).to_numpy()))


def _summarize_missing(name, missing_keys, limit=10):
    examples = sorted(missing_keys)[:limit]
    return f"{name}: {len(missing_keys)} missing; examples={examples}"


def _check_split_consistency(merged_df):
    merged_df = normalize_split_check_fields(merged_df)
    comparable_fields = ["local_eval_seed", "local_eval_ratio", "local_eval_samples"]
    for field in comparable_fields:
        multi_col = f"{field}_multi"
        a_col = f"{field}_single_a"
        b_col = f"{field}_single_b"
        mismatch = ~(
            merged_df.apply(lambda row: values_equal_numeric(row[multi_col], row[a_col]), axis=1)
            & merged_df.apply(lambda row: values_equal_numeric(row[multi_col], row[b_col]), axis=1)
        )
        if mismatch.any():
            examples = merged_df.loc[mismatch, ALIGN_KEYS + [multi_col, a_col, b_col]].head(10)
            raise ValueError(
                f"Split mismatch detected for {field}. "
                f"First mismatches: {examples.to_dict(orient='records')}"
            )

    split_file_mismatch = (
        (merged_df["split_file_multi"].astype(str) != merged_df["split_file_single_a"].astype(str))
        | (merged_df["split_file_multi"].astype(str) != merged_df["split_file_single_b"].astype(str))
    )
    if split_file_mismatch.any():
        warnings.warn(
            "split_file paths differ for some aligned clients, but seed/ratio/sample checks passed.",
            RuntimeWarning,
        )


def align_client_results(
    multi_df,
    single_a_df,
    single_b_df,
    strict_split_check=True,
    allow_missing_clients=False,
):
    """Align multimodal and two single-modality client eval results."""
    multi_keys = _key_set(multi_df)
    a_keys = _key_set(single_a_df)
    b_keys = _key_set(single_b_df)
    all_keys = multi_keys | a_keys | b_keys
    common_keys = multi_keys & a_keys & b_keys

    missing_messages = []
    if all_keys != common_keys:
        missing_messages = [
            _summarize_missing("missing_in_multi", all_keys - multi_keys),
            _summarize_missing("missing_in_single_a", all_keys - a_keys),
            _summarize_missing("missing_in_single_b", all_keys - b_keys),
        ]
        if not allow_missing_clients:
            raise ValueError("Unmatched clients across inputs. " + " | ".join(missing_messages))

    merged = multi_df.merge(
        single_a_df,
        on=ALIGN_KEYS,
        how="inner",
        suffixes=("_multi", "_single_a"),
    ).merge(
        single_b_df,
        on=ALIGN_KEYS,
        how="inner",
        suffixes=("", "_single_b"),
    )

    rename_after_second_merge = {}
    for col in list(merged.columns):
        if col not in ALIGN_KEYS and col.endswith("_single_b") is False:
            base = col
            if base in single_b_df.columns and base not in ALIGN_KEYS:
                rename_after_second_merge[base] = f"{base}_single_b"
    merged = merged.rename(columns=rename_after_second_merge)

    if strict_split_check:
        _check_split_consistency(merged)

    aligned = pd.DataFrame(
        {
            "dataset": merged["dataset"],
            "fold": merged["fold"],
            "client_id": merged["client_id"],
            "eval_data_type": merged["eval_data_type"],
            "fed_alg": merged.get("fed_alg_multi"),
            "alpha": merged.get("alpha_multi"),
            "local_eval_samples": merged["local_eval_samples_multi"],
            "split_file": merged["split_file_multi"],
            "multi_f1": merged["eval_f1_multi"],
            "single_a_f1": merged["eval_f1_single_a"],
            "single_b_f1": merged["eval_f1_single_b"],
            "multi_acc": merged["eval_acc_multi"],
            "single_a_acc": merged["eval_acc_single_a"],
            "single_b_acc": merged["eval_acc_single_b"],
        }
    )
    aligned.attrs["unmatched_clients"] = len(all_keys - common_keys)
    aligned.attrs["unmatched_messages"] = missing_messages
    return aligned


def compute_client_negative_transfer(
    aligned_df,
    eps=1.0,
    single_a_name="acc",
    single_b_name="gyro",
):
    """Compute client-level negative-transfer flags and gaps."""
    df = aligned_df.copy()
    df["single_a_name"] = single_a_name
    df["single_b_name"] = single_b_name
    df["best_single_f1"] = df[["single_a_f1", "single_b_f1"]].max(axis=1)
    df["gap"] = df["multi_f1"] - df["best_single_f1"]
    df["is_negative_transfer"] = df["gap"] < -eps
    df["eps"] = eps
    df["notes"] = ""
    df.attrs.update(aligned_df.attrs)
    return df


def compute_worst_client_f1(df, f1_col, worst_ratio=0.2):
    """Compute bottom-ratio mean F1 per fold plus a macro-average row."""
    rows = []
    for fold, fold_df in df.groupby("fold"):
        sorted_df = fold_df.sort_values(f1_col, ascending=True)
        k = max(1, int(math.ceil(len(sorted_df) * worst_ratio)))
        rows.append(
            {
                "fold": fold,
                "worst_client_count": k,
                f"worst{int(worst_ratio * 100)}_{f1_col}": sorted_df.head(k)[f1_col].mean(),
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        average = {"fold": "average"}
        for col in numeric_cols:
            average[col] = result[col].mean()
        result = pd.concat([result, pd.DataFrame([average])], ignore_index=True)
    return result


def compute_fold_summary(client_df, eps=1.0, worst_ratio=0.2):
    """Compute fold-level and macro-average diagnosis summaries."""
    rows = []
    worst_label = f"worst{int(worst_ratio * 100)}"
    unmatched_clients = client_df.attrs.get("unmatched_clients", 0)
    filtered_clients = client_df.attrs.get("filtered_clients", 0)

    for fold, fold_df in client_df.groupby("fold"):
        valid_clients = len(fold_df)
        k = max(1, int(math.ceil(valid_clients * worst_ratio)))
        multi_sorted = fold_df.sort_values("multi_f1", ascending=True)
        best_single_sorted = fold_df.sort_values("best_single_f1", ascending=True)
        rows.append(
            {
                "dataset": fold_df["dataset"].iloc[0],
                "fed_alg": fold_df["fed_alg"].iloc[0],
                "alpha": fold_df["alpha"].iloc[0],
                "fold": fold,
                "eval_data_type": fold_df["eval_data_type"].iloc[0],
                "num_clients": valid_clients + filtered_clients + unmatched_clients,
                "valid_clients": valid_clients,
                "filtered_clients": filtered_clients,
                "unmatched_clients": unmatched_clients,
                "eps": eps,
                "ntr": fold_df["is_negative_transfer"].mean(),
                "mean_multi_f1": fold_df["multi_f1"].mean(),
                "mean_best_single_f1": fold_df["best_single_f1"].mean(),
                f"{worst_label}_multi_f1": multi_sorted.head(k)["multi_f1"].mean(),
                f"{worst_label}_best_single_f1": best_single_sorted.head(k)["best_single_f1"].mean(),
                "variance_multi_f1": fold_df["multi_f1"].var(ddof=0),
                "std_multi_f1": fold_df["multi_f1"].std(ddof=0),
                "min_multi_f1": fold_df["multi_f1"].min(),
                "median_multi_f1": fold_df["multi_f1"].median(),
                "max_multi_f1": fold_df["multi_f1"].max(),
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    numeric_cols = summary.select_dtypes(include=[np.number]).columns
    average = {
        "dataset": summary["dataset"].iloc[0],
        "fed_alg": summary["fed_alg"].iloc[0],
        "alpha": summary["alpha"].iloc[0],
        "fold": "average",
        "eval_data_type": summary["eval_data_type"].iloc[0],
    }
    for col in numeric_cols:
        average[col] = summary[col].mean()
    summary = pd.concat([summary, pd.DataFrame([average])], ignore_index=True)
    return summary


def write_diagnosis_outputs(
    client_df,
    summary_df,
    output_dir,
    write_json_summary=False,
    overwrite=False,
    config=None,
):
    """Write client-level and summary diagnosis outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client_path = output_dir.joinpath("client_level_negative_transfer.csv")
    summary_path = output_dir.joinpath("diagnosis_summary.csv")
    config_path = output_dir.joinpath("diagnosis_config.json")
    output_paths = [client_path, summary_path, config_path]
    if write_json_summary:
        output_paths.append(output_dir.joinpath("diagnosis_summary.json"))

    existing_paths = [path for path in output_paths if path.exists()]
    if existing_paths and not overwrite:
        existing = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(
            f"Diagnosis output file(s) already exist: {existing}. "
            "Use --overwrite to replace them."
        )

    client_df.to_csv(client_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    with open(str(config_path), "w") as f:
        json.dump(config or {}, f, indent=4)

    json_path = None
    if write_json_summary:
        json_path = output_dir.joinpath("diagnosis_summary.json")
        payload = {
            "client_level_negative_transfer": str(client_path),
            "diagnosis_summary": str(summary_path),
            "summary": summary_df.to_dict(orient="records"),
        }
        with open(str(json_path), "w") as f:
            json.dump(payload, f, indent=4)

    return {
        "client_csv": client_path,
        "summary_csv": summary_path,
        "config_json": config_path,
        "summary_json": json_path,
    }
