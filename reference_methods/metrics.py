"""Count-based metrics for primary-preserving correction."""

from __future__ import annotations

import numpy as np


def action_metrics(base_probabilities: np.ndarray, final_probabilities: np.ndarray, labels: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    base_pred = np.asarray(base_probabilities).argmax(axis=1)
    final_pred = np.asarray(final_probabilities).argmax(axis=1)
    if base_pred.shape != labels.shape or final_pred.shape != labels.shape:
        raise ValueError("prediction rows do not match labels")
    changed_mask = final_pred != base_pred
    rescue_mask = changed_mask & (base_pred != labels) & (final_pred == labels)
    harm_mask = changed_mask & (base_pred == labels) & (final_pred != labels)
    changed = int(changed_mask.sum())
    rescue = int(rescue_mask.sum())
    harm = int(harm_mask.sum())
    rows = int(labels.size)
    net = rescue - harm
    return {
        "rows": rows,
        "overall_percent": 100.0 * float((final_pred == labels).mean()),
        "primary_percent": 100.0 * float((base_pred == labels).mean()),
        "changed_count": changed,
        "changed_percent": 100.0 * changed / rows,
        "rescue_count": rescue,
        "rescue_percent": 100.0 * rescue / rows,
        "harm_count": harm,
        "harm_percent": 100.0 * harm / rows,
        "net_gain_count": net,
        "net_gain_pp": 100.0 * net / rows,
        "conditional_utility_percent": (100.0 * net / changed) if changed else None,
    }
