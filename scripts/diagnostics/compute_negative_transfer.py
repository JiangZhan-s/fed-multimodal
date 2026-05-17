#!/usr/bin/env python
"""Compute offline client-level negative-transfer diagnostics."""

import argparse
import sys
from pathlib import Path

from fed_multimodal.metrics.negative_transfer import (
    align_client_results,
    compute_client_negative_transfer,
    compute_fold_summary,
    filter_valid_clients,
    load_per_client_eval_csv,
    write_diagnosis_outputs,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute client-level negative-transfer diagnostics.")
    parser.add_argument("--multi_csv", required=True, help="Multimodal per_client_eval_metrics.csv")
    parser.add_argument("--single_a_csv", required=True, help="Single-modality A per_client_eval_metrics.csv")
    parser.add_argument("--single_b_csv", required=True, help="Single-modality B per_client_eval_metrics.csv")
    parser.add_argument("--single_a_name", default="acc", help="Name for single-modality A")
    parser.add_argument("--single_b_name", default="gyro", help="Name for single-modality B")
    parser.add_argument("--output_dir", required=True, help="Directory for diagnosis outputs")
    parser.add_argument("--eval_data_type", default="local_eval_split", help="Evaluation data type to use")
    parser.add_argument("--metric", default="eval_f1", help="Metric column to validate and use")
    parser.add_argument("--eps", default=1.0, type=float, help="Negative-transfer margin in F1 points")
    parser.add_argument("--min_eval_samples", default=5, type=int, help="Minimum local eval samples per client")
    parser.add_argument("--worst_ratio", default=0.2, type=float, help="Bottom-client ratio for worst-client F1")
    parser.add_argument(
        "--strict_split_check",
        dest="strict_split_check",
        action="store_true",
        default=True,
        help="Require local split metadata consistency",
    )
    parser.add_argument(
        "--no_strict_split_check",
        dest="strict_split_check",
        action="store_false",
        help="Disable strict local split consistency checks",
    )
    parser.add_argument(
        "--allow_missing_clients",
        action="store_true",
        help="Allow inner join if clients are missing from one input",
    )
    parser.add_argument(
        "--write_json_summary",
        action="store_true",
        help="Also write diagnosis_summary.json",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing diagnosis output files",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.min_eval_samples < 1:
        raise ValueError("--min_eval_samples must be >= 1")
    if args.worst_ratio <= 0 or args.worst_ratio > 1:
        raise ValueError("--worst_ratio must satisfy 0 < worst_ratio <= 1")

    multi_df = load_per_client_eval_csv(args.multi_csv, args.eval_data_type, args.metric)
    single_a_df = load_per_client_eval_csv(args.single_a_csv, args.eval_data_type, args.metric)
    single_b_df = load_per_client_eval_csv(args.single_b_csv, args.eval_data_type, args.metric)

    multi_df, multi_filtered = filter_valid_clients(multi_df, args.min_eval_samples)
    single_a_df, single_a_filtered = filter_valid_clients(single_a_df, args.min_eval_samples)
    single_b_df, single_b_filtered = filter_valid_clients(single_b_df, args.min_eval_samples)

    aligned_df = align_client_results(
        multi_df,
        single_a_df,
        single_b_df,
        strict_split_check=args.strict_split_check,
        allow_missing_clients=args.allow_missing_clients,
    )
    aligned_df.attrs["filtered_clients"] = len(
        set(multi_filtered["client_id"])
        | set(single_a_filtered["client_id"])
        | set(single_b_filtered["client_id"])
    )

    client_df = compute_client_negative_transfer(
        aligned_df,
        eps=args.eps,
        single_a_name=args.single_a_name,
        single_b_name=args.single_b_name,
    )
    summary_df = compute_fold_summary(client_df, eps=args.eps, worst_ratio=args.worst_ratio)
    diagnosis_config = {
        "multi_csv": str(Path(args.multi_csv)),
        "single_a_csv": str(Path(args.single_a_csv)),
        "single_b_csv": str(Path(args.single_b_csv)),
        "single_a_name": args.single_a_name,
        "single_b_name": args.single_b_name,
        "eval_data_type": args.eval_data_type,
        "metric": args.metric,
        "eps": args.eps,
        "min_eval_samples": args.min_eval_samples,
        "worst_ratio": args.worst_ratio,
        "strict_split_check": args.strict_split_check,
        "allow_missing_clients": args.allow_missing_clients,
        "output_dir": str(Path(args.output_dir)),
    }

    paths = write_diagnosis_outputs(
        client_df,
        summary_df,
        args.output_dir,
        write_json_summary=args.write_json_summary,
        overwrite=args.overwrite,
        config=diagnosis_config,
    )

    average = summary_df[summary_df["fold"].astype(str) == "average"]
    report_row = average.iloc[0] if not average.empty else summary_df.iloc[0]
    print("Negative-transfer diagnosis complete.")
    print(f"valid_clients: {int(report_row['valid_clients'])}")
    print(f"NTR: {report_row['ntr']:.6f}")
    print(f"mean_multi_f1: {report_row['mean_multi_f1']:.6f}")
    worst_col = f"worst{int(args.worst_ratio * 100)}_multi_f1"
    print(f"{worst_col}: {report_row[worst_col]:.6f}")
    print(f"client output: {paths['client_csv']}")
    print(f"summary output: {paths['summary_csv']}")
    print(f"config output: {paths['config_json']}")
    if paths["summary_json"] is not None:
        print(f"json output: {paths['summary_json']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
