"""Shared command-line interface for low-cost auxiliary experts."""

from __future__ import annotations

import argparse

from .non_neural import run_reference_expert


def run_cli(expert: str) -> None:
    parser = argparse.ArgumentParser(description=f"Fit the {expert} public reference expert.")
    parser.add_argument("--train-cache", required=True, help="Signal cache containing I/Q, labels, and sample IDs.")
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--test-cache")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--estimators", type=int, default=500)
    parser.add_argument("--top-pairs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    report = run_reference_expert(
        expert,
        args.train_cache,
        args.validation_cache,
        args.test_cache,
        args.output_dir,
        folds=args.folds,
        estimators=args.estimators,
        top_pairs=args.top_pairs,
        seed=args.seed,
        jobs=args.jobs,
    )
    print(f"Completed {report['expert']} reference expert: {args.output_dir}")
