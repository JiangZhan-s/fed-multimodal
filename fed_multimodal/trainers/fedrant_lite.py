import logging
import math

import numpy as np
import torch


def compute_update_norm(global_state, client_state):
    """Return the L2 norm of the client update relative to the global state."""
    total = 0.0
    for key, global_tensor in global_state.items():
        client_tensor = client_state.get(key)
        if client_tensor is None:
            continue
        if not torch.is_floating_point(global_tensor) or not torch.is_floating_point(client_tensor):
            continue
        diff = client_tensor.detach().float().cpu() - global_tensor.detach().float().cpu()
        total += torch.sum(diff * diff).item()
    return float(math.sqrt(total))


def minmax_normalize(values, eps=1e-12):
    """Min-max normalize values; return zeros if the range is degenerate."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    finite_mask = np.isfinite(arr)
    if not finite_mask.all():
        raise ValueError("Cannot normalize non-finite values.")
    value_min = np.min(arr)
    value_max = np.max(arr)
    if abs(value_max - value_min) <= eps:
        return np.zeros_like(arr, dtype=float)
    return (arr - value_min) / (value_max - value_min + eps)


def _fill_missing(values, name):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    finite_mask = np.isfinite(arr)
    if finite_mask.all():
        return arr
    if not finite_mask.any():
        raise ValueError(f"All {name} values are missing or non-finite.")
    fill_value = float(np.median(arr[finite_mask]))
    logging.warning(
        "FedRANT-Lite: filling missing/non-finite %s values with median %.6f",
        name,
        fill_value,
    )
    arr[~finite_mask] = fill_value
    return arr


def compute_reliability_scores(
    losses,
    update_norms,
    mode,
    tau_loss,
    tau_norm,
    min_weight,
    max_weight,
):
    """Compute FedRANT-Lite reliability scores from train loss and update norm."""
    if mode not in ["loss", "norm", "loss_norm"]:
        raise ValueError("mode must be one of: loss, norm, loss_norm")
    if tau_loss < 0 or tau_norm < 0:
        raise ValueError("tau_loss and tau_norm must be non-negative.")
    if min_weight <= 0:
        raise ValueError("min_weight must be positive.")
    if max_weight < min_weight:
        raise ValueError("max_weight must be greater than or equal to min_weight.")

    losses = _fill_missing(losses, "loss")
    update_norms = _fill_missing(update_norms, "update_norm")
    if len(losses) != len(update_norms):
        raise ValueError("losses and update_norms must have the same length.")

    loss_norm = minmax_normalize(losses)
    update_norm_norm = minmax_normalize(update_norms)

    reliability = np.ones_like(loss_norm, dtype=float)
    if mode in ["loss", "loss_norm"]:
        reliability *= np.exp(-tau_loss * loss_norm)
    if mode in ["norm", "loss_norm"]:
        reliability *= 1.0 / (1.0 + tau_norm * update_norm_norm)

    reliability = np.clip(reliability, min_weight, max_weight)
    if not np.isfinite(reliability).all():
        raise ValueError("FedRANT-Lite produced non-finite reliability scores.")
    return reliability.astype(float).tolist()


def compute_fedrant_weights(num_samples, reliabilities):
    """Return normalized FedRANT-Lite aggregation weights."""
    num_samples = np.asarray(num_samples, dtype=float)
    reliabilities = np.asarray(reliabilities, dtype=float)
    if len(num_samples) != len(reliabilities):
        raise ValueError("num_samples and reliabilities must have the same length.")
    if len(num_samples) == 0:
        return []
    if np.any(num_samples < 0):
        raise ValueError("num_samples must be non-negative.")
    if not np.isfinite(num_samples).all() or not np.isfinite(reliabilities).all():
        raise ValueError("num_samples and reliabilities must be finite.")

    raw_weights = num_samples * reliabilities
    raw_sum = float(np.sum(raw_weights))
    if raw_sum <= 0:
        logging.warning("FedRANT-Lite: non-positive raw weights; falling back to FedAvg sample weights.")
        sample_sum = float(np.sum(num_samples))
        if sample_sum <= 0:
            return [1.0 / len(num_samples)] * len(num_samples)
        return (num_samples / sample_sum).astype(float).tolist()
    return (raw_weights / raw_sum).astype(float).tolist()
