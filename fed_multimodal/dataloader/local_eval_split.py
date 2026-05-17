import copy
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def maybe_get_labels(records):
    return [record[2] for record in records]


def _subset_records(records, indices):
    return [copy.deepcopy(records[idx]) for idx in indices]


def _stratified_split_indices(labels, num_eval, rng):
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    if any(len(indices) < 2 for indices in label_to_indices.values()):
        raise ValueError("at least one class has fewer than two samples")

    eval_indices = list()
    for indices in label_to_indices.values():
        indices = list(indices)
        rng.shuffle(indices)
        class_eval = int(round(len(indices) * num_eval / len(labels)))
        class_eval = max(1, class_eval)
        class_eval = min(class_eval, len(indices) - 1)
        eval_indices.extend(indices[:class_eval])

    if len(eval_indices) > num_eval:
        rng.shuffle(eval_indices)
        eval_indices = eval_indices[:num_eval]
    elif len(eval_indices) < num_eval:
        remaining = [idx for idx in range(len(labels)) if idx not in set(eval_indices)]
        rng.shuffle(remaining)
        eval_indices.extend(remaining[:num_eval - len(eval_indices)])

    eval_indices = sorted(set(eval_indices))
    train_indices = [idx for idx in range(len(labels)) if idx not in set(eval_indices)]

    if len(train_indices) == 0:
        raise ValueError("stratified split left no training samples")
    return train_indices, eval_indices


def _random_split_indices(num_samples, num_eval, rng):
    indices = list(range(num_samples))
    rng.shuffle(indices)
    eval_indices = sorted(indices[:num_eval])
    train_indices = sorted(indices[num_eval:])
    return train_indices, eval_indices


def split_multimodal_client_records(
    acc_records,
    gyro_records,
    eval_ratio,
    seed,
    min_eval_samples,
    client_id,
):
    if len(acc_records) != len(gyro_records):
        raise ValueError(
            f"Cannot split client {client_id}: acc and gyro sample counts differ "
            f"({len(acc_records)} vs {len(gyro_records)})."
        )

    num_samples = len(acc_records)
    warning = ""
    split_method = "stratified"

    if num_samples <= 1:
        train_indices = list(range(num_samples))
        eval_indices = list()
        split_method = "skipped"
        warning = "not enough samples to create a local eval split"
    else:
        num_eval = int(round(num_samples * eval_ratio))
        num_eval = max(min_eval_samples, num_eval)
        num_eval = min(num_eval, num_samples - 1)

        if num_eval <= 0:
            train_indices = list(range(num_samples))
            eval_indices = list()
            split_method = "skipped"
            warning = "computed local eval size is zero"
        else:
            labels = maybe_get_labels(acc_records)
            rng = np.random.default_rng(seed)
            try:
                train_indices, eval_indices = _stratified_split_indices(labels, num_eval, rng)
            except ValueError as exc:
                split_method = "random"
                warning = f"stratified split failed: {exc}; used random split"
                logging.warning(f"Client {client_id}: {warning}")
                train_indices, eval_indices = _random_split_indices(num_samples, num_eval, rng)

    split_info = {
        "client_id": client_id,
        "num_samples": num_samples,
        "train_indices": train_indices,
        "eval_indices": eval_indices,
        "local_train_samples": len(train_indices),
        "local_eval_samples": len(eval_indices),
        "split_method": split_method,
        "warning": warning,
        "label_counts": dict(Counter(maybe_get_labels(acc_records))),
    }

    return (
        _subset_records(acc_records, train_indices),
        _subset_records(gyro_records, train_indices),
        _subset_records(acc_records, eval_indices),
        _subset_records(gyro_records, eval_indices),
        split_info,
    )


def split_unimodal_client_records(
    records,
    eval_ratio,
    seed,
    min_eval_samples,
    client_id,
):
    if len(records) == 0:
        split_info = {
            "client_id": client_id,
            "num_samples": 0,
            "train_indices": [],
            "eval_indices": [],
            "local_train_samples": 0,
            "local_eval_samples": 0,
            "split_method": "skipped",
            "warning": "no samples to split",
            "label_counts": {},
        }
        return [], [], split_info

    num_samples = len(records)
    warning = ""
    split_method = "stratified"

    if num_samples <= 1:
        train_indices = list(range(num_samples))
        eval_indices = list()
        split_method = "skipped"
        warning = "not enough samples to create a local eval split"
    else:
        num_eval = int(round(num_samples * eval_ratio))
        num_eval = max(min_eval_samples, num_eval)
        num_eval = min(num_eval, num_samples - 1)

        if num_eval <= 0:
            train_indices = list(range(num_samples))
            eval_indices = list()
            split_method = "skipped"
            warning = "computed local eval size is zero"
        else:
            labels = maybe_get_labels(records)
            rng = np.random.default_rng(seed)
            try:
                train_indices, eval_indices = _stratified_split_indices(labels, num_eval, rng)
            except ValueError as exc:
                split_method = "random"
                warning = f"stratified split failed: {exc}; used random split"
                logging.warning(f"Client {client_id}: {warning}")
                train_indices, eval_indices = _random_split_indices(num_samples, num_eval, rng)

    split_info = {
        "client_id": client_id,
        "num_samples": num_samples,
        "train_indices": train_indices,
        "eval_indices": eval_indices,
        "local_train_samples": len(train_indices),
        "local_eval_samples": len(eval_indices),
        "split_method": split_method,
        "warning": warning,
        "label_counts": dict(Counter(maybe_get_labels(records))),
    }

    return (
        _subset_records(records, train_indices),
        _subset_records(records, eval_indices),
        split_info,
    )


def save_local_eval_split_json(split_info_dict, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "w") as f:
        json.dump(split_info_dict, f, indent=4)
