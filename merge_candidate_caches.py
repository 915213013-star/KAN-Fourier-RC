"""Merge aligned primary and auxiliary probability caches into one candidate pool."""

from __future__ import annotations

import argparse
from collections import OrderedDict

from reference_methods.cache_io import (
    load_candidate_cache,
    merge_prediction_caches,
    save_candidate_cache,
)


def parse_named_cache(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("use NAME=PATH for each auxiliary cache")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("both NAME and PATH are required")
    return name.strip(), path.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True, help="Single-predictor primary cache (.npz).")
    parser.add_argument(
        "--auxiliary",
        action="append",
        default=[],
        type=parse_named_cache,
        metavar="NAME=PATH",
        help="Repeat for each auxiliary predictor.",
    )
    parser.add_argument("--output", required=True, help="Output candidate cache (.npz).")
    args = parser.parse_args()

    auxiliaries = OrderedDict()
    for name, path in args.auxiliary:
        if name in auxiliaries:
            parser.error(f"duplicate auxiliary name: {name}")
        auxiliaries[name] = load_candidate_cache(path)
    merged = merge_prediction_caches(load_candidate_cache(args.primary), auxiliaries)
    save_candidate_cache(args.output, merged)
    print(
        f"Saved {merged.probabilities.shape[0]:,} rows, "
        f"{merged.probabilities.shape[1]} candidates, and "
        f"{merged.probabilities.shape[2]} classes to {args.output}"
    )


if __name__ == "__main__":
    main()
