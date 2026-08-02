"""Fit isolated OOF residual utility and freeze retain/correct actions on validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reference_methods.baselines import fit_isolated_eru, summarize
from reference_methods.cache_io import (
    assert_candidate_alignment,
    load_candidate_cache,
    save_prediction_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True, help="Model-learning OOF candidate cache.")
    parser.add_argument("--validation-cache", required=True, help="Independent policy-validation cache.")
    parser.add_argument("--test-cache", help="Optional held-out reporting cache.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--estimators", type=int, default=320)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    train = load_candidate_cache(args.train_cache)
    validation = load_candidate_cache(args.validation_cache)
    test = load_candidate_cache(args.test_cache) if args.test_cache else None
    caches = (train, validation) if test is None else (train, validation, test)
    assert_candidate_alignment(*caches)
    outputs, config = fit_isolated_eru(
        train,
        validation,
        *((test,) if test is not None else ()),
        seed=args.seed,
        estimators=args.estimators,
        depth=args.depth,
        jobs=args.jobs,
        device=args.device,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "method": "Isolated OOF ERU",
        "protocol": "train-OOF utility fitting; validation-only policy freeze; optional held-out reporting",
        "candidate_names": list(train.candidate_names),
        "configuration": config,
        "splits": {},
    }
    split_names = ["validation"] + (["test"] if test is not None else [])
    split_caches = [validation] + ([test] if test is not None else [])
    for split_name, cache, output in zip(split_names, split_caches, outputs):
        report["splits"][split_name] = summarize(cache.probabilities[:, 0], output, cache.labels)
        save_prediction_cache(
            output_dir / f"isolated_oof_eru_{split_name}.npz",
            output,
            cache.labels,
            cache.sample_ids,
            "Isolated OOF ERU",
        )
    with (output_dir / "frozen_policy.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    selected = report["splits"]["test" if test is not None else "validation"]
    print(
        f"Isolated OOF ERU overall={selected['overall_percent']:.3f}% "
        f"changed={selected['changed_percent']:.3f}% "
        f"net={selected['net_gain_pp']:+.3f} pp"
    )


if __name__ == "__main__":
    main()
