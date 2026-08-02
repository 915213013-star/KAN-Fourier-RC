"""Cross-fitted low-cost auxiliary predictors.

These are compact public reference implementations of the three evidence
families used by KAN-Fourier-RC: higher-order statistics (HCS),
constellation/graph geometry (GAMC-inspired), and pairwise confusion repair.
They provide compact executable implementations under the shared public cache
contract. Dataset-specific search choices remain configuration rather than
being embedded in these estimators.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .cache_io import SignalCache, load_signal_cache, normalize_probabilities, save_prediction_cache
from .features import gamc_geometry_features, hcs_features


def _predict_all_classes(model, features: np.ndarray, class_count: int) -> np.ndarray:
    output = np.zeros((features.shape[0], class_count), dtype=np.float32)
    output[:, np.asarray(model.classes_, dtype=np.int64)] = model.predict_proba(features)
    return normalize_probabilities(output)


def _tree(seed: int, estimators: int, jobs: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=estimators,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=jobs,
        random_state=seed,
    )


def _crossfit_tree(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    evaluation_features: list[np.ndarray],
    *,
    folds: int,
    seed: int,
    estimators: int,
    jobs: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    class_count = int(train_labels.max()) + 1
    oof = np.zeros((train_labels.shape[0], class_count), dtype=np.float32)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (fit_indices, holdout_indices) in enumerate(splitter.split(train_features, train_labels)):
        model = _tree(seed + fold, estimators, jobs)
        model.fit(train_features[fit_indices], train_labels[fit_indices])
        oof[holdout_indices] = _predict_all_classes(model, train_features[holdout_indices], class_count)
    final_model = _tree(seed + folds, estimators, jobs)
    final_model.fit(train_features, train_labels)
    evaluation = [_predict_all_classes(final_model, features, class_count) for features in evaluation_features]
    return oof, evaluation


def _confusion_pairs(base_probabilities: np.ndarray, labels: np.ndarray, top_pairs: int) -> list[tuple[int, int]]:
    predictions = base_probabilities.argmax(axis=1)
    class_count = base_probabilities.shape[1]
    counts = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(counts, (labels, predictions), 1)
    np.fill_diagonal(counts, 0)
    undirected = counts + counts.T
    pairs = []
    for left in range(class_count):
        for right in range(left + 1, class_count):
            pairs.append((int(undirected[left, right]), left, right))
    pairs.sort(reverse=True)
    return [(left, right) for count, left, right in pairs[:top_pairs] if count > 0]


def _fit_pair_models(
    features: np.ndarray,
    labels: np.ndarray,
    base_probabilities: np.ndarray,
    *,
    top_pairs: int,
    seed: int,
) -> list[tuple[int, int, LogisticRegression]]:
    models = []
    for left, right in _confusion_pairs(base_probabilities, labels, top_pairs):
        selected = (labels == left) | (labels == right)
        if selected.sum() < 20 or np.unique(labels[selected]).size < 2:
            continue
        model = LogisticRegression(C=1.0, max_iter=400, solver="lbfgs", random_state=seed)
        model.fit(features[selected], labels[selected])
        models.append((left, right, model))
    return models


def _apply_pair_models(
    models: list[tuple[int, int, LogisticRegression]],
    features: np.ndarray,
    base_probabilities: np.ndarray,
) -> np.ndarray:
    output = np.asarray(base_probabilities, dtype=np.float32).copy()
    top_two = np.argpartition(output, -2, axis=1)[:, -2:]
    for left, right, model in models:
        active = np.any(top_two == left, axis=1) & np.any(top_two == right, axis=1)
        if not np.any(active):
            continue
        pair_mass = output[active, left] + output[active, right]
        pair_probabilities = model.predict_proba(features[active])
        positions = {int(label): index for index, label in enumerate(model.classes_)}
        output[active, left] = pair_mass * pair_probabilities[:, positions[left]]
        output[active, right] = pair_mass * pair_probabilities[:, positions[right]]
    return normalize_probabilities(output)


def _crossfit_pairwise(
    train_features: np.ndarray,
    train: SignalCache,
    evaluation_features: list[np.ndarray],
    evaluation_caches: list[SignalCache],
    *,
    folds: int,
    top_pairs: int,
    seed: int,
) -> tuple[np.ndarray, list[np.ndarray], list[tuple[int, int]]]:
    if train.base_probabilities is None:
        raise ValueError("pairwise reference requires base_probabilities in every signal cache")
    oof = train.base_probabilities.copy()
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (fit_indices, holdout_indices) in enumerate(splitter.split(train_features, train.labels)):
        models = _fit_pair_models(
            train_features[fit_indices],
            train.labels[fit_indices],
            train.base_probabilities[fit_indices],
            top_pairs=top_pairs,
            seed=seed + fold,
        )
        oof[holdout_indices] = _apply_pair_models(
            models,
            train_features[holdout_indices],
            train.base_probabilities[holdout_indices],
        )
    full_models = _fit_pair_models(
        train_features,
        train.labels,
        train.base_probabilities,
        top_pairs=top_pairs,
        seed=seed + folds,
    )
    outputs = []
    for features, cache in zip(evaluation_features, evaluation_caches):
        if cache.base_probabilities is None:
            raise ValueError("pairwise reference requires base_probabilities in every signal cache")
        outputs.append(_apply_pair_models(full_models, features, cache.base_probabilities))
    return oof, outputs, [(left, right) for left, right, _ in full_models]


def run_reference_expert(
    expert: str,
    train_path: str | Path,
    validation_path: str | Path,
    test_path: str | Path | None,
    output_dir: str | Path,
    *,
    folds: int = 5,
    estimators: int = 500,
    top_pairs: int = 16,
    seed: int = 1,
    jobs: int = 8,
) -> dict:
    train = load_signal_cache(train_path)
    validation = load_signal_cache(validation_path)
    test = load_signal_cache(test_path) if test_path else None
    evaluation_caches = [validation] + ([test] if test is not None else [])
    if any(cache.iq.shape[2] != train.iq.shape[2] for cache in evaluation_caches):
        raise ValueError("signal length differs across splits")
    if any(int(cache.labels.max()) != int(train.labels.max()) for cache in evaluation_caches):
        raise ValueError("class count differs across splits")

    if expert == "hcs":
        train_features = hcs_features(train.iq)
        evaluation_features = [hcs_features(cache.iq) for cache in evaluation_caches]
        oof, outputs = _crossfit_tree(
            train_features,
            train.labels,
            evaluation_features,
            folds=folds,
            seed=seed,
            estimators=estimators,
            jobs=jobs,
        )
        metadata = {"feature_family": "higher_order_statistics", "estimators": estimators}
    elif expert == "gamc":
        train_features = gamc_geometry_features(train.iq)
        evaluation_features = [gamc_geometry_features(cache.iq) for cache in evaluation_caches]
        oof, outputs = _crossfit_tree(
            train_features,
            train.labels,
            evaluation_features,
            folds=folds,
            seed=seed,
            estimators=estimators,
            jobs=jobs,
        )
        metadata = {"feature_family": "constellation_graph_geometry", "estimators": estimators}
    elif expert == "pairwise":
        train_features = np.concatenate((hcs_features(train.iq), gamc_geometry_features(train.iq)), axis=1)
        evaluation_features = [
            np.concatenate((hcs_features(cache.iq), gamc_geometry_features(cache.iq)), axis=1)
            for cache in evaluation_caches
        ]
        oof, outputs, pairs = _crossfit_pairwise(
            train_features,
            train,
            evaluation_features,
            evaluation_caches,
            folds=folds,
            top_pairs=top_pairs,
            seed=seed,
        )
        metadata = {"feature_family": "pairwise_confusion", "selected_pairs": pairs}
    else:
        raise ValueError(f"unknown expert: {expert}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_prediction_cache(output_dir / f"{expert}_train_oof.npz", oof, train.labels, train.sample_ids, expert)
    for split_name, cache, probabilities in zip(
        ["validation"] + (["test"] if test is not None else []), evaluation_caches, outputs
    ):
        save_prediction_cache(
            output_dir / f"{expert}_{split_name}.npz",
            probabilities,
            cache.labels,
            cache.sample_ids,
            expert,
        )
    report = {
        "expert": expert,
        "protocol": f"{folds}-fold cross-fit on model-learning data; full fit for validation/test inference",
        "metadata": metadata,
    }
    with (output_dir / f"{expert}_config.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return report
