"""Shared safeguards for leakage-controlled out-of-fold training.

The public training entry points use an inner selection split to choose the
training horizon, then refit a fresh model on the complete outer-training fold.
The outer holdout is consumed only after refitting to create its OOF rows.
"""

import hashlib
import json
from argparse import Namespace

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


PROTOCOL_ID = "inner_select_outer_refit_v1"
PROTOCOL_TAG = "isorfv1"


def add_protocol_args(parser):
    parser.add_argument(
        "--inner_val_fraction",
        type=float,
        default=0.10,
        help="Fraction of each outer-training fold used only for epoch selection.",
    )
    parser.add_argument(
        "--inner_split_seed_offset",
        type=int,
        default=19001,
        help="Deterministic offset used to derive the inner selection split seed.",
    )
    return parser


def _strata(labels, snrs, include_snr=True):
    if include_snr:
        return np.asarray([f"{int(y)}_{int(s)}" for y, s in zip(labels, snrs)])
    return np.asarray([str(int(y)) for y in labels])


def make_inner_selection_split(outer_train_indices, labels_all, snrs_all, seed, fraction):
    outer = np.asarray(outer_train_indices, dtype=np.int64)
    labels = np.asarray(labels_all, dtype=np.int64)[outer]
    snrs = np.asarray(snrs_all, dtype=np.int32)[outer]
    fraction = float(fraction)
    if not 0.0 < fraction < 0.5:
        raise ValueError("inner_val_fraction must be in (0, 0.5).")

    last_error = None
    for include_snr in (True, False):
        try:
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=fraction,
                random_state=int(seed),
            )
            inner_train_pos, inner_select_pos = next(
                splitter.split(np.zeros(len(outer), dtype=np.uint8), _strata(labels, snrs, include_snr))
            )
            inner_train = outer[np.asarray(inner_train_pos, dtype=np.int64)]
            inner_select = outer[np.asarray(inner_select_pos, dtype=np.int64)]
            assert_partition_invariants(outer, inner_train, inner_select)
            return inner_train, inner_select
        except ValueError as exc:
            last_error = exc
    raise ValueError(f"Could not construct a stratified inner selection split: {last_error}")


def assert_partition_invariants(outer_train, inner_train, inner_select, outer_holdout=None):
    outer_train = np.asarray(outer_train, dtype=np.int64)
    inner_train = np.asarray(inner_train, dtype=np.int64)
    inner_select = np.asarray(inner_select, dtype=np.int64)
    if np.intersect1d(inner_train, inner_select).size:
        raise RuntimeError("Inner-train and inner-selection indices overlap.")
    if not np.array_equal(np.sort(np.concatenate([inner_train, inner_select])), np.sort(outer_train)):
        raise RuntimeError("Inner splits do not form an exact partition of the outer-training fold.")
    if outer_holdout is not None:
        outer_holdout = np.asarray(outer_holdout, dtype=np.int64)
        if np.intersect1d(outer_train, outer_holdout).size:
            raise RuntimeError("Outer-training and outer-holdout indices overlap.")


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
    inner_train_indices=None,
    inner_selection_indices=None,
    selected_epoch=None,
    target_epochs=None,
):
    if phase not in {"inner_selection", "outer_refit"}:
        raise ValueError(f"Unknown OOF phase: {phase}")
    meta = {
        "protocol_id": PROTOCOL_ID,
        "fold": int(fold),
        "phase": phase,
        "config_fingerprint": config_fingerprint(args),
        "outer_train_sha256": index_digest(outer_train_indices),
        "outer_holdout_sha256": index_digest(outer_holdout_indices),
        "selected_epoch": None if selected_epoch is None else int(selected_epoch),
        "target_epochs": None if target_epochs is None else int(target_epochs),
    }
    if inner_train_indices is not None:
        meta["inner_train_sha256"] = index_digest(inner_train_indices)
    if inner_selection_indices is not None:
        meta["inner_selection_sha256"] = index_digest(inner_selection_indices)
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
        "inner_train_sha256",
        "inner_selection_sha256",
        "selected_epoch",
        "target_epochs",
    )
    return all(actual.get(key) == expected_metadata.get(key) for key in keys)


def metadata_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def cache_protocol_summary(args, folds):
    return {
        "protocol_id": PROTOCOL_ID,
        "config_fingerprint": config_fingerprint(args),
        "folds": list(folds),
        "outer_holdout_usage": "inference_once_after_fresh_outer_train_refit",
        "epoch_selection": "inner_selection_split_only",
    }
