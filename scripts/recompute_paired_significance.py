"""Recompute or verify the fixed 15-comparison paired significance family.

The public repository ships the aggregate audit ledger, not sample-level
prediction archives.  ``--verify-aggregate`` therefore works immediately.
Authorized reviewers who receive the three controlled NPZ archives can use
``--dataset DATASET=ARCHIVE`` to recompute every count, interval, raw McNemar
p-value, and the joint Holm correction.

Expected NPZ arrays are ``y_true``, ``snr``, and one prediction vector for each
key in ``PREDICTION_KEYS`` below.  HisarMod2019.1 additionally requires
``storage_block``.  Prediction vectors may contain class IDs or class
probabilities with shape ``(N, C)``.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


DATASET_SIZES = {
    "RML2016.10A": 22_000,
    "RML2016.10B": 120_000,
    "HisarMod2019.1": 260_000,
}
PREDICTION_KEYS = {
    "Full_ERU_RC_vs_Primary": "pred_primary",
    "Full_ERU_RC_vs_OOF_Linear_Stacking": "pred_oof_linear_stacking",
    "Full_ERU_RC_vs_OOF_XGBoost_Stacking": "pred_oof_xgboost_stacking",
    "Full_ERU_RC_vs_OOF_Candidate_Competence": "pred_oof_candidate_competence",
    "Full_ERU_RC_vs_Isolated_OOF_ERU": "pred_isolated_oof_eru",
}
FULL_KEY = "pred_full_eru_rc"


def as_labels(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 2:
        values = values.argmax(axis=1)
    if values.ndim != 1:
        raise ValueError(f"Expected a class-ID vector or probability matrix, got {values.shape}")
    return values.astype(np.int64, copy=False)


def make_strata(data: np.lib.npyio.NpzFile, dataset: str) -> np.ndarray:
    labels = as_labels(data["y_true"])
    snr = np.asarray(data["snr"]).reshape(-1)
    if dataset == "HisarMod2019.1":
        block = np.asarray(data["storage_block"]).reshape(-1)
        tokens = np.stack([labels, snr, block], axis=1)
    else:
        tokens = np.stack([labels, snr], axis=1)
    _, strata = np.unique(tokens, axis=0, return_inverse=True)
    return strata


def stratified_paired_bootstrap(
    delta: np.ndarray,
    strata: np.ndarray,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    """Exact within-stratum paired bootstrap using multinomial count draws."""

    rng = np.random.default_rng(seed)
    totals = np.zeros(int(repetitions), dtype=np.int64)
    for stratum in np.unique(strata):
        values = delta[strata == stratum]
        counts = np.asarray([(values == -1).sum(), (values == 0).sum(), (values == 1).sum()])
        draws = rng.multinomial(len(values), counts / len(values), size=int(repetitions))
        totals += draws[:, 2] - draws[:, 0]
    boot_pp = 100.0 * totals / len(delta)
    low, high = np.percentile(boot_pp, [2.5, 97.5])
    return float(low), float(high)


def holm_adjust(raw_values: list[float]) -> list[float]:
    order = np.argsort(raw_values)
    adjusted = np.empty(len(raw_values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(raw_values) - rank) * raw_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def recompute_dataset(path: Path, dataset: str, repetitions: int, seed: int) -> list[dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        required = {"y_true", "snr", FULL_KEY, *PREDICTION_KEYS.values()}
        if dataset == "HisarMod2019.1":
            required.add("storage_block")
        missing = sorted(required - set(data.files))
        if missing:
            raise KeyError(f"{path.name} is missing arrays: {missing}")
        y_true = as_labels(data["y_true"])
        if len(y_true) != DATASET_SIZES[dataset]:
            raise ValueError(f"{dataset}: expected {DATASET_SIZES[dataset]} rows, got {len(y_true)}")
        full = as_labels(data[FULL_KEY])
        full_correct = full == y_true
        strata = make_strata(data, dataset)
        rows = []
        for offset, (comparison, baseline_key) in enumerate(PREDICTION_KEYS.items()):
            baseline = as_labels(data[baseline_key])
            baseline_correct = baseline == y_true
            n10 = int(np.sum(full_correct & ~baseline_correct))
            n01 = int(np.sum(~full_correct & baseline_correct))
            delta = full_correct.astype(np.int8) - baseline_correct.astype(np.int8)
            low, high = stratified_paired_bootstrap(delta, strata, repetitions, seed + offset)
            discordant = n10 + n01
            raw_p = 1.0 if discordant == 0 else float(
                binomtest(min(n10, n01), discordant, 0.5, alternative="two-sided").pvalue
            )
            rows.append({
                "dataset": dataset,
                "comparison": comparison,
                "difference_pp": 100.0 * (n10 - n01) / len(y_true),
                "ci95_low_pp": low,
                "ci95_high_pp": high,
                "n10": n10,
                "n01": n01,
                "mcnemar_p_raw": raw_p,
                "bootstrap_repetitions": repetitions,
                "stratification": (
                    "class_x_snr_x_storage_block" if dataset == "HisarMod2019.1" else "class_x_snr"
                ),
            })
        return rows


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_aggregate(path: Path) -> None:
    rows = read_rows(path)
    expected = {(dataset, comparison) for dataset in DATASET_SIZES for comparison in PREDICTION_KEYS}
    observed = {(row["dataset"], row["comparison"]) for row in rows}
    if len(rows) != 15 or observed != expected:
        raise RuntimeError("Aggregate ledger is not the fixed 15-comparison family.")
    for row in rows:
        size = DATASET_SIZES[row["dataset"]]
        difference = 100.0 * (int(row["n10"]) - int(row["n01"])) / size
        if not math.isclose(difference, float(row["difference_pp"]), abs_tol=1e-6):
            raise RuntimeError(f"Count mismatch: {row['dataset']}/{row['comparison']}")
        if int(row["bootstrap_repetitions"]) != 10_000:
            raise RuntimeError("The released ledger must use 10,000 bootstrap repetitions.")
    print(f"Verified fixed 15-comparison aggregate ledger: {path}")
    print("Sample-level archives are available through controlled academic verification.")


def parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use DATASET=ARCHIVE.npz")
    dataset, path = value.split("=", 1)
    if dataset not in DATASET_SIZES:
        raise argparse.ArgumentTypeError(f"Unknown dataset: {dataset}")
    return dataset, Path(path)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aggregate",
        type=Path,
        default=root / "audit_artifacts" / "paired_significance.csv",
    )
    parser.add_argument("--verify-aggregate", action="store_true")
    parser.add_argument("--dataset", action="append", type=parse_dataset, default=[])
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.verify_aggregate or not args.dataset:
        verify_aggregate(args.aggregate)
    if not args.dataset:
        return
    archives = dict(args.dataset)
    if set(archives) != set(DATASET_SIZES):
        raise RuntimeError("Joint Holm correction requires one controlled archive for each of the three datasets.")
    rows = []
    for dataset in DATASET_SIZES:
        rows.extend(recompute_dataset(archives[dataset], dataset, args.bootstrap_repetitions, args.seed))
    adjusted = holm_adjust([float(row["mcnemar_p_raw"]) for row in rows])
    for row, value in zip(rows, adjusted):
        row["holm_adjusted_p"] = value
    output = args.output or root / "audit_artifacts" / "paired_significance_recomputed.csv"
    fieldnames = list(rows[0])
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Recomputed {len(rows)} pre-specified paired comparisons: {output}")


if __name__ == "__main__":
    main()
