"""Validation-only retain-or-correct policy selection."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class FrozenActionPolicy:
    threshold: float
    max_change_rate: float
    validation_accuracy: float

    def as_dict(self) -> dict:
        return asdict(self)


def apply_candidate_actions(
    probabilities: np.ndarray,
    candidate_indices: np.ndarray,
    action_scores: np.ndarray,
    threshold: float,
    max_change_rate: float,
) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float32)
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    action_scores = np.asarray(action_scores, dtype=np.float32)
    if probs.ndim != 3 or candidate_indices.shape != (probs.shape[0],) or action_scores.shape != (probs.shape[0],):
        raise ValueError("invalid action array shapes")
    selected = (candidate_indices > 0) & np.isfinite(action_scores) & (action_scores >= threshold)
    budget = int(np.floor(float(max_change_rate) * probs.shape[0] + 1e-9))
    if budget < int(selected.sum()):
        eligible = np.flatnonzero(selected)
        order = eligible[np.argsort(-action_scores[eligible], kind="stable")]
        selected[:] = False
        selected[order[:budget]] = True
    output = probs[:, 0].copy()
    rows = np.flatnonzero(selected)
    output[rows] = probs[rows, candidate_indices[rows]]
    return output


def select_validation_policy(
    probabilities: np.ndarray,
    candidate_indices: np.ndarray,
    action_scores: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray | None = None,
    change_rates: tuple[float, ...] = (0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 1.0),
) -> FrozenActionPolicy:
    labels = np.asarray(labels, dtype=np.int64)
    finite = action_scores[np.isfinite(action_scores) & (candidate_indices > 0)]
    if thresholds is None:
        thresholds = np.unique(
            np.concatenate(
                ((np.asarray([0.0], dtype=np.float32)), np.quantile(finite, np.linspace(0.0, 0.95, 20)))
            )
        ) if finite.size else np.asarray([np.inf], dtype=np.float32)
    best = None
    for rate in change_rates:
        for threshold in thresholds:
            output = apply_candidate_actions(probabilities, candidate_indices, action_scores, float(threshold), rate)
            accuracy = float((output.argmax(axis=1) == labels).mean())
            key = (accuracy, -rate, float(threshold))
            if best is None or key > best[0]:
                best = (key, FrozenActionPolicy(float(threshold), float(rate), accuracy))
    assert best is not None
    return best[1]
