"""Fingerprint-validated joblib caches for fitted scikit-learn estimators."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import joblib
import numpy as np


CACHE_VERSION = "kanlie-estimator-cache-v1"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else repr(number)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return repr(value)


def _file_stamp(path):
    if not path:
        return None
    p = Path(path).resolve()
    if not p.exists():
        return {"path": str(p), "missing": True}
    stat = p.stat()
    return {
        "path": str(p),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _sample_digest(x, y, sample_weight=None, max_rows=128):
    n = len(x)
    if n == 0:
        raise ValueError("Cannot cache an estimator trained on an empty array.")
    count = min(int(max_rows), n)
    ids = np.linspace(0, n - 1, num=count, dtype=np.int64)
    h = hashlib.sha256()
    h.update(str(tuple(x.shape)).encode("utf-8"))
    h.update(str(np.asarray(x).dtype).encode("utf-8"))
    h.update(np.ascontiguousarray(np.asarray(x)[ids]).tobytes())
    h.update(np.ascontiguousarray(np.asarray(y)[ids]).tobytes())
    if sample_weight is not None:
        h.update(np.ascontiguousarray(np.asarray(sample_weight)[ids]).tobytes())
    return h.hexdigest()


def _cache_payload(estimator, namespace, x, y, sample_weight, source_paths, context):
    params = estimator.get_params(deep=False) if hasattr(estimator, "get_params") else {}
    return {
        "cache_version": CACHE_VERSION,
        "namespace": str(namespace),
        "estimator_class": f"{type(estimator).__module__}.{type(estimator).__qualname__}",
        "estimator_params": _jsonable(params),
        "x_shape": [int(v) for v in x.shape],
        "y_shape": [int(v) for v in np.asarray(y).shape],
        "sample_digest": _sample_digest(x, y, sample_weight),
        "source_files": [_file_stamp(path) for path in (source_paths or [])],
        "context": _jsonable(context or {}),
    }


def fit_or_load_estimator(
    estimator,
    x,
    y,
    *,
    sample_weight=None,
    cache_dir="",
    reuse=False,
    namespace="model",
    source_paths=None,
    context=None,
):
    """Load a matching fitted estimator or fit and atomically cache it."""

    if not cache_dir:
        estimator.fit(x, y, sample_weight=sample_weight)
        return estimator, False, ""

    payload = _cache_payload(estimator, namespace, x, y, sample_weight, source_paths, context)
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    key = hashlib.sha256(encoded).hexdigest()[:20]
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(namespace)).strip("_") or "model"
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    model_path = cache_root / f"{safe_name}_{key}.joblib"
    manifest_path = cache_root / f"{safe_name}_{key}.json"

    if reuse and model_path.exists() and manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                saved_payload = json.load(f)
            if saved_payload != payload:
                raise RuntimeError("manifest mismatch")
            loaded = joblib.load(model_path)
            expected_features = int(x.shape[1])
            actual_features = int(getattr(loaded, "n_features_in_", expected_features))
            if actual_features != expected_features:
                raise RuntimeError(
                    f"feature mismatch: cached={actual_features}, current={expected_features}"
                )
            print(f"[*] Reusing fitted model: {model_path}", flush=True)
            return loaded, True, str(model_path)
        except Exception as exc:
            print(f"[!] Model cache could not be reused ({model_path}): {exc}", flush=True)

    estimator.fit(x, y, sample_weight=sample_weight)
    model_tmp = Path(str(model_path) + f".tmp-{os.getpid()}")
    manifest_tmp = Path(str(manifest_path) + f".tmp-{os.getpid()}")
    joblib.dump(estimator, model_tmp, compress=3)
    with manifest_tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(model_tmp, model_path)
    os.replace(manifest_tmp, manifest_path)
    print(f"[*] Fitted model saved: {model_path}", flush=True)
    return estimator, False, str(model_path)
