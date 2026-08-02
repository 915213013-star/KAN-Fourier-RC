"""Matched OOF decision baselines on a shared candidate-probability cache.

The implementations in this module intentionally share the same candidate
pool and observable feature matrix.  They differ only in the decision target:
direct class stacking, candidate competence, or rescue-minus-harm utility.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor

from .cache_io import CandidateCache, assert_candidate_alignment, load_candidate_cache, save_prediction_cache
from .features import probability_meta_features
from .metrics import action_metrics
from .policy import apply_candidate_actions, select_validation_policy


def _features(cache: CandidateCache) -> np.ndarray:
    return probability_meta_features(cache.probabilities, cache.meta_features)


def _xgb_common(seed: int, estimators: int, depth: int, jobs: int, device: str) -> dict:
    return {
        "n_estimators": estimators,
        "max_depth": depth,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "tree_method": "hist",
        "device": device,
        "n_jobs": jobs,
        "random_state": seed,
        "verbosity": 0,
    }


def _constant_or_binary_regressor(
    features: np.ndarray,
    target: np.ndarray,
    *,
    seed: int,
    estimators: int,
    depth: int,
    jobs: int,
    device: str,
):
    target = np.asarray(target, dtype=np.float32)
    if np.all(target == target[0]):
        return float(target[0])
    model = XGBRegressor(
        objective="reg:squarederror",
        **_xgb_common(seed, estimators, depth, jobs, device),
    )
    model.fit(features, target)
    return model


def _predict_regressor(model, features: np.ndarray) -> np.ndarray:
    if isinstance(model, float):
        return np.full(features.shape[0], model, dtype=np.float32)
    return np.asarray(model.predict(features), dtype=np.float32)


def fit_linear_stacking(train: CandidateCache, *splits: CandidateCache, seed: int = 1) -> list[np.ndarray]:
    model = LogisticRegression(
        C=1.0,
        max_iter=600,
        solver="lbfgs",
        random_state=seed,
    )
    model.fit(_features(train), train.labels)
    outputs = []
    class_count = train.probabilities.shape[2]
    for split in splits:
        raw = model.predict_proba(_features(split))
        probabilities = np.zeros((split.labels.shape[0], class_count), dtype=np.float32)
        probabilities[:, model.classes_.astype(np.int64)] = raw
        outputs.append(probabilities)
    return outputs


def fit_xgb_stacking(
    train: CandidateCache,
    *splits: CandidateCache,
    seed: int = 1,
    estimators: int = 520,
    depth: int = 3,
    jobs: int = 8,
    device: str = "cpu",
) -> list[np.ndarray]:
    class_count = train.probabilities.shape[2]
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=class_count,
        eval_metric="mlogloss",
        **_xgb_common(seed, estimators, depth, jobs, device),
    )
    model.fit(_features(train), train.labels)
    return [np.asarray(model.predict_proba(_features(split)), dtype=np.float32) for split in splits]


def fit_candidate_competence(
    train: CandidateCache,
    validation: CandidateCache,
    *evaluation_splits: CandidateCache,
    seed: int = 1,
    estimators: int = 320,
    depth: int = 3,
    jobs: int = 8,
    device: str = "cpu",
) -> tuple[list[np.ndarray], dict]:
    train_features = _features(train)
    all_splits = (validation,) + evaluation_splits
    split_features = [_features(split) for split in all_splits]
    score_blocks = []
    for candidate in range(train.probabilities.shape[1]):
        correct = (train.probabilities[:, candidate].argmax(axis=1) == train.labels).astype(np.float32)
        model = _constant_or_binary_regressor(
            train_features,
            correct,
            seed=seed + candidate,
            estimators=estimators,
            depth=depth,
            jobs=jobs,
            device=device,
        )
        score_blocks.append([_predict_regressor(model, features) for features in split_features])

    outputs = []
    actions = []
    scores = []
    for split_index, split in enumerate(all_splits):
        matrix = np.stack([block[split_index] for block in score_blocks], axis=1)
        candidate_indices = matrix.argmax(axis=1).astype(np.int64)
        action_scores = matrix[np.arange(matrix.shape[0]), candidate_indices] - matrix[:, 0]
        actions.append(candidate_indices)
        scores.append(action_scores)

    policy = select_validation_policy(
        validation.probabilities,
        actions[0],
        scores[0],
        validation.labels,
    )
    for split, candidate_indices, action_scores in zip(all_splits, actions, scores):
        outputs.append(
            apply_candidate_actions(
                split.probabilities,
                candidate_indices,
                action_scores,
                policy.threshold,
                policy.max_change_rate,
            )
        )
    return outputs, {"policy": asdict(policy)}


def fit_isolated_eru(
    train: CandidateCache,
    validation: CandidateCache,
    *evaluation_splits: CandidateCache,
    seed: int = 1,
    estimators: int = 320,
    depth: int = 3,
    jobs: int = 8,
    device: str = "cpu",
) -> tuple[list[np.ndarray], dict]:
    train_features = _features(train)
    primary_prediction = train.probabilities[:, 0].argmax(axis=1)
    primary_correct = primary_prediction == train.labels
    all_splits = (validation,) + evaluation_splits
    split_features = [_features(split) for split in all_splits]
    score_blocks: list[list[np.ndarray]] = []

    for candidate in range(1, train.probabilities.shape[1]):
        candidate_correct = train.probabilities[:, candidate].argmax(axis=1) == train.labels
        utility = candidate_correct.astype(np.float32) - primary_correct.astype(np.float32)
        model = _constant_or_binary_regressor(
            train_features,
            utility,
            seed=seed + candidate,
            estimators=estimators,
            depth=depth,
            jobs=jobs,
            device=device,
        )
        score_blocks.append([_predict_regressor(model, features) for features in split_features])

    actions = []
    scores = []
    for split_index, split in enumerate(all_splits):
        if score_blocks:
            matrix = np.stack([block[split_index] for block in score_blocks], axis=1)
            best_auxiliary = matrix.argmax(axis=1)
            actions.append((best_auxiliary + 1).astype(np.int64))
            scores.append(matrix[np.arange(matrix.shape[0]), best_auxiliary])
        else:
            actions.append(np.zeros(split.labels.shape[0], dtype=np.int64))
            scores.append(np.full(split.labels.shape[0], -np.inf, dtype=np.float32))

    policy = select_validation_policy(
        validation.probabilities,
        actions[0],
        scores[0],
        validation.labels,
    )
    outputs = [
        apply_candidate_actions(
            split.probabilities,
            candidate_indices,
            action_scores,
            policy.threshold,
            policy.max_change_rate,
        )
        for split, candidate_indices, action_scores in zip(all_splits, actions, scores)
    ]
    return outputs, {"policy": asdict(policy)}


def summarize(base: np.ndarray, output: np.ndarray, labels: np.ndarray) -> dict:
    summary = action_metrics(base, output, labels)
    summary["overall_percent"] = 100.0 * float((output.argmax(axis=1) == labels).mean())
    return summary


def run_all_baselines(
    train_path: str | Path,
    validation_path: str | Path,
    test_path: str | Path | None,
    output_dir: str | Path,
    *,
    seed: int = 1,
    estimators: int = 520,
    depth: int = 3,
    jobs: int = 8,
    device: str = "cpu",
) -> dict:
    train = load_candidate_cache(train_path)
    validation = load_candidate_cache(validation_path)
    test = load_candidate_cache(test_path) if test_path else None
    caches = (train, validation) if test is None else (train, validation, test)
    assert_candidate_alignment(*caches)
    evaluation = (validation,) if test is None else (validation, test)

    linear_outputs = fit_linear_stacking(train, *evaluation, seed=seed)
    xgb_outputs = fit_xgb_stacking(
        train, *evaluation, seed=seed, estimators=estimators, depth=depth, jobs=jobs, device=device
    )
    competence_outputs, competence_config = fit_candidate_competence(
        train,
        validation,
        *((test,) if test is not None else ()),
        seed=seed,
        estimators=estimators,
        depth=depth,
        jobs=jobs,
        device=device,
    )
    eru_outputs, eru_config = fit_isolated_eru(
        train,
        validation,
        *((test,) if test is not None else ()),
        seed=seed,
        estimators=estimators,
        depth=depth,
        jobs=jobs,
        device=device,
    )

    methods = {
        "OOF Linear Stacking": linear_outputs,
        "OOF XGBoost Stacking": xgb_outputs,
        "OOF Candidate Competence": competence_outputs,
        "Isolated OOF ERU": eru_outputs,
    }
    configs = {
        "OOF Candidate Competence": competence_config,
        "Isolated OOF ERU": eru_config,
    }
    split_names = ["validation"] + (["test"] if test is not None else [])
    split_caches = [validation] + ([test] if test is not None else [])
    report: dict = {
        "protocol": "train OOF fit; validation-only policy selection; optional test reporting",
        "candidate_names": list(train.candidate_names),
        "methods": {},
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for method, outputs in methods.items():
        report["methods"][method] = {"config": configs.get(method, {}), "splits": {}}
        for split_name, cache, output in zip(split_names, split_caches, outputs):
            report["methods"][method]["splits"][split_name] = summarize(
                cache.probabilities[:, 0], output, cache.labels
            )
            save_prediction_cache(
                output_dir / f"{method.lower().replace(' ', '_')}_{split_name}.npz",
                output,
                cache.labels,
                cache.sample_ids,
                method,
            )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return report
