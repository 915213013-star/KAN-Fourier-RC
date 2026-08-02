"""
RML2016.10B common adapter for the 66.123% RML2016.10A OOF residual-fusion flow.

This module intentionally reuses the stable 10A training helpers, but overrides:
  - NUM_CLASSES / class names for RML2016.10B
  - dataset loader and default data/cache paths
  - aligned split candidates, including the existing 10B historical split files

The inference protocol remains the same:
  train split trains models and OOF probabilities,
  validation selects/calibrates fusion parameters,
  test labels are used only in the final report.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from train_cv_trn_aux_2016 import *  # noqa: F401,F403 - reuse stable helpers


NUM_CLASSES = 10
DEFAULT_MOD_CLASSES = [
    "8PSK",
    "AM-DSB",
    "BPSK",
    "CPFSK",
    "GFSK",
    "PAM4",
    "QAM16",
    "QAM64",
    "QPSK",
    "WBFM",
]

ROOT = Path(__file__).resolve().parent


def _default_10b_root():
    env_root = os.environ.get("RML2016_10B_ROOT") or os.environ.get("KANLIE_10B_ROOT")
    if env_root:
        return Path(env_root)
    candidates = [
        ROOT / "data" / "RML2016.10BGAMC",
        ROOT.parent / "RML2016.10BGAMC",
        Path.cwd() / "RML2016.10BGAMC",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


TENB_ROOT = _default_10b_root()
DEFAULT_10B_DATA_PATH = os.environ.get(
    "RML2016_10B_PATH", str(ROOT / "data" / "RML2016.10b.dat")
)
DEFAULT_10B_CACHE_DIR = os.environ.get("RML2016_10B_CACHE_DIR", str(TENB_ROOT / "feature_cache"))
DEFAULT_10B_ALIGNMENT_CACHE = os.environ.get(
    "RML2016_10B_ALIGNMENT_CACHE",
    str(TENB_ROOT / "results" / "greedy_soup_10b_identity_valtest_probs_for_gamc_fusion.npz"),
)


def _load_10b_dataset_class():
    loader_candidates = [TENB_ROOT / "dataprocessnew4.py", ROOT / "dataprocessnew4.py"]
    loader_path = next((path for path in loader_candidates if path.exists()), None)
    if loader_path is None:
        searched = ", ".join(str(path) for path in loader_candidates)
        raise FileNotFoundError(f"Cannot find the 10B data loader. Searched: {searched}")
    old_path = list(sys.path)
    sys.path.insert(0, str(loader_path.parent))
    try:
        spec = importlib.util.spec_from_file_location("rml2016_10b_dataprocessnew4", loader_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import 10B dataprocess loader: {loader_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.RML2016Dataset
    finally:
        sys.path[:] = old_path


def build_full_dataset(args):
    data_path = getattr(args, "data_path", DEFAULT_10B_DATA_PATH) or DEFAULT_10B_DATA_PATH
    cache_dir = getattr(args, "cache_dir", DEFAULT_10B_CACHE_DIR) or DEFAULT_10B_CACHE_DIR
    RML2016Dataset10B = _load_10b_dataset_class()
    return RML2016Dataset10B(
        data_path=data_path,
        transform=True,
        return_snr=True,
        use_cache=True,
        cache_dir=cache_dir,
    )


def make_aligned_split(full_dataset, split_seed):
    labels = np.asarray(full_dataset.labels, dtype=np.int64)
    snrs = np.asarray(full_dataset.snrs, dtype=np.int32)
    composite = np.array([f"{int(y)}_{int(s)}" for y, s in zip(labels, snrs)])
    indices = np.arange(len(labels))
    train_idx, temp_idx, _, temp_targets = train_test_split(
        indices,
        composite,
        test_size=0.2,
        random_state=split_seed,
        stratify=composite,
    )
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx,
        temp_targets,
        test_size=0.5,
        random_state=split_seed,
        stratify=temp_targets,
    )

    candidate_paths = [
        TENB_ROOT / f"test_indices_model_oracle_distill_split{split_seed}.npy",
        TENB_ROOT / f"test_indices_model_multitf_specialist_split{split_seed}.npy",
        TENB_ROOT / f"test_indices_model_snr_estimator_split{split_seed}.npy",
        TENB_ROOT / f"test_indices_model_transition_specialist_10b_split{split_seed}.npy",
        TENB_ROOT / f"test_indices_model_complex_tcn_distill_split{split_seed}.npy",
        TENB_ROOT / f"test_indices_model_4stream_moe_joint_strat_rml201610b_seed{split_seed}.npy",
        ROOT / f"test_indices_model_4stream_moe_joint_strat_rml201610b_seed{split_seed}.npy",
        ROOT / f"test_indices_fourier_rml10b_seed{split_seed}.npy",
        ROOT / f"test_indices_10b_seed{split_seed}.npy",
        ROOT / "test_indices_10b.npy",
    ]
    for path in candidate_paths:
        if path.exists():
            saved = np.load(path)
            if len(saved) == len(test_idx) and np.array_equal(np.sort(saved), np.sort(test_idx)):
                test_idx = saved.astype(np.int64)
                print(f"[*] Reusing aligned 10B test indices: {path}")
                break
    return train_idx.astype(np.int64), val_idx.astype(np.int64), test_idx.astype(np.int64)


def check_alignment(full_dataset, val_idx, test_idx, cache_path):
    cache_path = cache_path or DEFAULT_10B_ALIGNMENT_CACHE
    if not os.path.exists(cache_path):
        print(f"[!] 10B alignment cache not found, skipped: {cache_path}")
        return
    z = np.load(cache_path, allow_pickle=True)
    labels = np.asarray(full_dataset.labels, dtype=np.int64)
    snrs = np.asarray(full_dataset.snrs, dtype=np.int32)
    pairs = [
        ("labels_val", labels[val_idx], z["labels_val"].astype(np.int64)),
        ("snrs_val", snrs[val_idx], z["snrs_val"].astype(np.int32)),
        ("labels_test", labels[test_idx], z["labels_test"].astype(np.int64)),
        ("snrs_test", snrs[test_idx], z["snrs_test"].astype(np.int32)),
    ]
    for name, cur, ref in pairs:
        if len(cur) != len(ref) or not np.all(cur == ref):
            raise RuntimeError(f"10B split is not aligned with reference cache: {name}")
    print("[*] 10B alignment check passed against reference val/test cache.")
