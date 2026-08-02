"""Run matched OOF Linear/XGBoost/Competence/isolated-ERU baselines."""

from __future__ import annotations

import argparse
import json

from reference_methods.baselines import run_all_baselines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True, help="Model-learning OOF candidate cache (.npz).")
    parser.add_argument("--validation-cache", required=True, help="Policy-validation candidate cache (.npz).")
    parser.add_argument("--test-cache", help="Optional held-out reporting cache (.npz).")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--estimators", type=int, default=520)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_all_baselines(
        args.train_cache,
        args.validation_cache,
        args.test_cache,
        args.output_dir,
        seed=args.seed,
        estimators=args.estimators,
        depth=args.depth,
        jobs=args.jobs,
        device=args.device,
    )
    for method, record in report["methods"].items():
        values = record["splits"]["test" if "test" in record["splits"] else "validation"]
        print(f"{method:28s} overall={values['overall_percent']:.3f}% changed={values['changed_percent']:.3f}%")


if __name__ == "__main__":
    main()
