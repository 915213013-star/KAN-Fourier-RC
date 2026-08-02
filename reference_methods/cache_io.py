"""Portable cache contracts used by the public reference experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class CandidateCache:
    probabilities: np.ndarray
    labels: np.ndarray
    candidate_names: tuple[str, ...]
    sample_ids: np.ndarray
    meta_features: np.ndarray | None = None


@dataclass(frozen=True)
class SignalCache:
    iq: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray
    base_probabilities: np.ndarray | None = None


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.ndim not in (2, 3):
        raise ValueError(f"probabilities must have 2 or 3 dimensions, got {probs.shape}")
    if not np.all(np.isfinite(probs)):
        raise ValueError("probabilities contain NaN or infinity")
    probs = np.clip(probs, 0.0, None)
    denom = probs.sum(axis=-1, keepdims=True)
    if np.any(denom <= 0):
        raise ValueError("each probability vector must have positive mass")
    return probs / denom


def _first_present(archive: np.lib.npyio.NpzFile, keys: Iterable[str]):
    for key in keys:
        if key in archive:
            return archive[key]
    return None


def _labels(archive: np.lib.npyio.NpzFile) -> np.ndarray:
    value = _first_present(archive, ("labels", "y", "targets"))
    if value is None:
        raise KeyError("cache requires labels (accepted keys: labels, y, targets)")
    labels = np.asarray(value, dtype=np.int64).reshape(-1)
    if labels.size == 0 or labels.min(initial=0) < 0:
        raise ValueError("labels must be non-empty non-negative integers")
    return labels


def _sample_ids(archive: np.lib.npyio.NpzFile, rows: int) -> np.ndarray:
    value = _first_present(archive, ("sample_ids", "indices", "idx"))
    if value is None:
        return np.arange(rows, dtype=np.int64)
    sample_ids = np.asarray(value).reshape(-1)
    if sample_ids.shape[0] != rows:
        raise ValueError("sample_ids length does not match the cache rows")
    if np.unique(sample_ids).size != rows:
        raise ValueError("sample_ids must be unique within a split")
    return sample_ids


def load_candidate_cache(path: str | Path) -> CandidateCache:
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        probabilities = _first_present(
            archive, ("candidate_probabilities", "candidate_probs", "probabilities", "probs")
        )
        if probabilities is None and "primary_probabilities" in archive and "auxiliary_probabilities" in archive:
            primary = np.asarray(archive["primary_probabilities"])
            auxiliary = np.asarray(archive["auxiliary_probabilities"])
            if auxiliary.ndim == 2:
                auxiliary = auxiliary[:, None, :]
            probabilities = np.concatenate((primary[:, None, :], auxiliary), axis=1)
        if probabilities is None:
            raise KeyError(
                "candidate cache requires candidate_probabilities/probabilities, or primary_probabilities plus auxiliary_probabilities"
            )
        probabilities = normalize_probabilities(probabilities)
        if probabilities.ndim == 2:
            probabilities = probabilities[:, None, :]
        labels = _labels(archive)
        if probabilities.shape[0] != labels.shape[0]:
            raise ValueError("probability rows do not match labels")
        names_value = _first_present(archive, ("candidate_names", "model_names"))
        if names_value is None:
            names = ("primary",) + tuple(
                f"candidate_{index}" for index in range(1, probabilities.shape[1])
            )
        else:
            names = tuple(str(value) for value in np.asarray(names_value).reshape(-1).tolist())
        if len(names) != probabilities.shape[1]:
            raise ValueError("candidate_names length does not match probability blocks")
        meta = _first_present(archive, ("meta_features", "observable_features", "phi"))
        if meta is not None:
            meta = np.asarray(meta, dtype=np.float32)
            if meta.ndim != 2 or meta.shape[0] != labels.shape[0]:
                raise ValueError("meta_features must have shape [rows, features]")
            if not np.all(np.isfinite(meta)):
                raise ValueError("meta_features contain NaN or infinity")
        sample_ids = _sample_ids(archive, labels.shape[0])
    return CandidateCache(probabilities, labels, names, sample_ids, meta)


def load_signal_cache(path: str | Path) -> SignalCache:
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        iq = _first_present(archive, ("iq", "signals", "x"))
        if iq is None:
            raise KeyError("signal cache requires iq (accepted keys: iq, signals, x)")
        iq = np.asarray(iq, dtype=np.float32)
        if iq.ndim != 3:
            raise ValueError(f"iq must have shape [rows, 2, length] or [rows, length, 2], got {iq.shape}")
        if iq.shape[1] != 2 and iq.shape[2] == 2:
            iq = np.transpose(iq, (0, 2, 1))
        if iq.shape[1] != 2:
            raise ValueError("the I/Q channel dimension must equal 2")
        if not np.all(np.isfinite(iq)):
            raise ValueError("iq contains NaN or infinity")
        labels = _labels(archive)
        if iq.shape[0] != labels.shape[0]:
            raise ValueError("signal rows do not match labels")
        base = _first_present(archive, ("base_probabilities", "primary_probabilities", "primary_probs"))
        if base is not None:
            base = normalize_probabilities(base)
            if base.ndim != 2 or base.shape[0] != labels.shape[0]:
                raise ValueError("base_probabilities must have shape [rows, classes]")
        sample_ids = _sample_ids(archive, labels.shape[0])
    return SignalCache(iq, labels, sample_ids, base)


def assert_candidate_alignment(*caches: CandidateCache) -> None:
    if not caches:
        return
    reference = caches[0]
    for cache in caches[1:]:
        if cache.probabilities.shape[1:] != reference.probabilities.shape[1:]:
            raise ValueError("candidate/class dimensions differ across splits")
        if cache.candidate_names != reference.candidate_names:
            raise ValueError("candidate names differ across splits")


def align_candidate_rows(reference: CandidateCache, cache: CandidateCache) -> CandidateCache:
    """Return ``cache`` in ``reference.sample_ids`` order, failing on any mismatch."""
    if cache.probabilities.shape[2] != reference.probabilities.shape[2]:
        raise ValueError("class dimensions differ across aligned caches")
    lookup = {value: index for index, value in enumerate(cache.sample_ids.tolist())}
    try:
        order = np.asarray([lookup[value] for value in reference.sample_ids.tolist()], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"candidate cache is missing sample_id {error.args[0]!r}") from error
    if len(lookup) != reference.sample_ids.shape[0]:
        extra = set(lookup).difference(reference.sample_ids.tolist())
        raise ValueError(f"candidate cache has a different sample set ({len(extra)} extra rows)")
    labels = cache.labels[order]
    if not np.array_equal(labels, reference.labels):
        raise ValueError("labels differ after sample_id alignment")
    meta = cache.meta_features[order] if cache.meta_features is not None else None
    return CandidateCache(
        probabilities=cache.probabilities[order],
        labels=labels,
        candidate_names=cache.candidate_names,
        sample_ids=cache.sample_ids[order],
        meta_features=meta,
    )


def merge_prediction_caches(
    primary: CandidateCache,
    auxiliaries: Mapping[str, CandidateCache],
) -> CandidateCache:
    """Build one aligned candidate pool from single-predictor probability caches."""
    if primary.probabilities.shape[1] != 1:
        raise ValueError("the primary input must contain exactly one probability block")
    names = ["primary"]
    blocks = [primary.probabilities]
    for name, cache in auxiliaries.items():
        clean_name = str(name).strip()
        if not clean_name or clean_name in names:
            raise ValueError(f"invalid or duplicate candidate name: {name!r}")
        aligned = align_candidate_rows(primary, cache)
        if aligned.probabilities.shape[1] != 1:
            raise ValueError(f"auxiliary {clean_name!r} must contain exactly one probability block")
        names.append(clean_name)
        blocks.append(aligned.probabilities)
    return CandidateCache(
        probabilities=np.concatenate(blocks, axis=1),
        labels=primary.labels.copy(),
        candidate_names=tuple(names),
        sample_ids=primary.sample_ids.copy(),
        meta_features=primary.meta_features.copy() if primary.meta_features is not None else None,
    )


def save_candidate_cache(path: str | Path, cache: CandidateCache) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "candidate_probabilities": normalize_probabilities(cache.probabilities),
        "labels": np.asarray(cache.labels, dtype=np.int64),
        "sample_ids": np.asarray(cache.sample_ids),
        "candidate_names": np.asarray(cache.candidate_names, dtype=np.str_),
    }
    if cache.meta_features is not None:
        payload["meta_features"] = np.asarray(cache.meta_features, dtype=np.float32)
    np.savez_compressed(path, **payload)


def save_prediction_cache(
    path: str | Path,
    probabilities: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    method: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        probabilities=normalize_probabilities(probabilities),
        labels=np.asarray(labels, dtype=np.int64),
        sample_ids=np.asarray(sample_ids),
        method=np.asarray(method),
    )
