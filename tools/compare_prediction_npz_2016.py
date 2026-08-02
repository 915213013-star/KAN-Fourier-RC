from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def acc(y, pred, mask=None):
    if mask is None:
        mask = np.ones_like(y, dtype=bool)
    if int(mask.sum()) == 0:
        return float("nan")
    return float((pred[mask] == y[mask]).mean() * 100.0)


def summarize(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    y = d["labels"].astype(int)
    if "pred" in d.files:
        pred = d["pred"].astype(int)
    else:
        pred = d["final_prob"].argmax(axis=1).astype(int)
    snrs = d["snrs"].astype(int)
    out = {
        "name": path.stem.replace("_predictions", ""),
        "overall": acc(y, pred),
        "edge": acc(y, pred, np.isin(snrs, [-18, -16])),
        "ultra_edge": acc(y, pred, np.isin(snrs, [-20, -18])),
        "midlow": acc(y, pred, np.isin(snrs, [-14, -12])),
        "transition": acc(y, pred, np.isin(snrs, [-10, -8, -6, -4, -2])),
        "negative": acc(y, pred, snrs < 0),
        "high": acc(y, pred, snrs >= 0),
    }
    for s in sorted(np.unique(snrs)):
        out[f"snr_{s}"] = acc(y, pred, snrs == s)
    if "use_candidate" in d.files:
        out["use_rate"] = float(d["use_candidate"].mean() * 100.0)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--snrs", nargs="*", type=int, default=[-20, -18, -14, -12, -10, -8, -6, -4, -2, 0])
    args = parser.parse_args()
    rows = [summarize(Path(f)) for f in args.files if Path(f).exists()]
    if not rows:
        raise SystemExit("no files found")

    cols = ["name", "overall", "edge", "ultra_edge", "midlow", "transition", "negative", "high", "use_rate"]
    cols += [f"snr_{s}" for s in args.snrs]
    cols = [c for c in cols if any(c in r for r in rows)]
    widths = {c: max(len(c), max(len(f"{r.get(c, ''):.3f}") if isinstance(r.get(c), float) else len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print(" | ".join(c.ljust(widths[c]) for c in cols))
    print("-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        parts = []
        for c in cols:
            v = r.get(c, "")
            parts.append((f"{v:.3f}" if isinstance(v, float) else str(v)).ljust(widths[c]))
        print(" | ".join(parts))


if __name__ == "__main__":
    main()
