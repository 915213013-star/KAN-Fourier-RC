"""Shared metadata and safeguards for the paper's OOF training protocol.

For every fold, samples in ``outer_holdout`` are excluded from gradient
updates.  Their labels provide the fold-level checkpoint/early-stopping score,
and the selected checkpoint exports the corresponding OOF probabilities.  A
separate policy-validation partition selects correction actions and thresholds;
the official held-out/test partition is evaluated only after the policy is
frozen.

This is the leakage-controlled fold-held-out protocol reported in the paper.
It is not presented as nested cross-validation or as an unbiased nested-CV
performance estimator.
"""

import hashlib
import json
from argparse import Namespace

import numpy as np


PROTOCOL_ID = "fold_holdout_checkpoint_selection_v1"
PROTOCOL_TAG = "fhcsv1"


def add_protocol_args(parser):
    """Keep a common hook across the public neural OOF entry points."""
    return parser


def assert_fold_partition(
    outer_train,
    outer_holdout,
    policy_validation=None,
    official_test=None,
):
    """Fail closed when any optimization/evaluation partition overlaps."""
    train = np.asarray(outer_train, dtype=np.int64)
    holdout = np.asarray(outer_holdout, dtype=np.int64)
    if train.size == 0 or holdout.size == 0:
        raise RuntimeError("OOF train and holdout partitions must both be non-empty.")
    if np.unique(train).size != train.size or np.unique(holdout).size != holdout.size:
        raise RuntimeError("Duplicate sample indices were found inside an OOF partition.")
    if np.intersect1d(train, holdout).size:
        raise RuntimeError("OOF gradient-training and fold-holdout indices overlap.")

    named = [("outer_train", train), ("outer_holdout", holdout)]
    if policy_validation is not None:
        named.append(("policy_validation", np.asarray(policy_validation, dtype=np.int64)))
    if official_test is not None:
        named.append(("official_test", np.asarray(official_test, dtype=np.int64)))
    for i, (name_a, values_a) in enumerate(named):
        for name_b, values_b in named[i + 1 :]:
            if np.intersect1d(values_a, values_b).size:
                raise RuntimeError(f"Partitions overlap: {name_a} and {name_b}.")


def index_digest(indices):
    values = np.ascontiguousarray(np.asarray(indices, dtype="<i8"))
    return hashlib.sha256(values.tobytes()).hexdigest()


_FINGERPRINT_EXCLUDE = {
    "alignment_cache",
    "cache_dir",
    "data_path",
    "force_restart",
    "output_cache",
    "skip_alignment_check",
}


def config_fingerprint(args):
    values = vars(args) if isinstance(args, Namespace) else dict(args)
    canonical = {}
    for key, value in sorted(values.items()):
        if key.startswith("_") or key in _FINGERPRINT_EXCLUDE:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            canonical[key] = value
        elif isinstance(value, (list, tuple)):
            canonical[key] = list(value)
        elif isinstance(value, dict):
            canonical[key] = value
        else:
            canonical[key] = str(value)
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def protocol_metadata(
    args,
    fold,
    phase,
    outer_train_indices,
    outer_holdout_indices,
    selected_epoch=None,
    target_epochs=None,
    policy_validation_indices=None,
    official_test_indices=None,
):
    if phase != "fold_training":
        raise ValueError(f"Unknown OOF phase: {phase}")
    assert_fold_partition(
        outer_train_indices,
        outer_holdout_indices,
        policy_validation_indices,
        official_test_indices,
    )
    meta = {
        "protocol_id": PROTOCOL_ID,
        "fold": int(fold),
        "phase": phase,
        "config_fingerprint": config_fingerprint(args),
        "outer_train_sha256": index_digest(outer_train_indices),
        "outer_holdout_sha256": index_digest(outer_holdout_indices),
        "selected_epoch": None if selected_epoch is None else int(selected_epoch),
        "target_epochs": None if target_epochs is None else int(target_epochs),
        "gradient_fit": "outer_train_only",
        "checkpoint_selection": "outer_holdout_fold",
        "oof_export": "selected_checkpoint_on_same_outer_holdout_fold",
        "policy_selection": "independent_validation_only",
        "official_test_usage": "frozen_evaluation_only",
        "nested_cv_claim": False,
    }
    if policy_validation_indices is not None:
        meta["policy_validation_sha256"] = index_digest(policy_validation_indices)
    if official_test_indices is not None:
        meta["official_test_sha256"] = index_digest(official_test_indices)
    return meta


def checkpoint_matches(checkpoint, expected_metadata):
    actual = checkpoint.get("protocol_metadata")
    if not isinstance(actual, dict):
        return False
    keys = (
        "protocol_id",
        "fold",
        "phase",
        "config_fingerprint",
        "outer_train_sha256",
        "outer_holdout_sha256",
        "target_epochs",
        "gradient_fit",
        "checkpoint_selection",
        "policy_selection",
        "official_test_usage",
    )
    return all(actual.get(key) == expected_metadata.get(key) for key in keys)


def metadata_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def cache_protocol_summary(args, folds):
    return {
        "protocol_id": PROTOCOL_ID,
        "config_fingerprint": config_fingerprint(args),
        "folds": list(folds),
        "gradient_fit": "outer_train_only",
        "epoch_and_checkpoint_selection": "outer_holdout_fold",
        "outer_holdout_usage": "checkpoint_selection_and_oof_export",
        "policy_selection": "independent_validation_only",
        "official_test_usage": "frozen_evaluation_only",
        "interpretation": "leakage_controlled_fold_training_not_nested_cv",
    }
