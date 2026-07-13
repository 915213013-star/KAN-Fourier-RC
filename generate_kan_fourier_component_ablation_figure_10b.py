from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

CACHES = {
    "KAN-Fourier": RESULTS / "kan_fourier_10b_ablation_full_mseed361_predictions.npz",
    "Real-valued multiscale conv": RESULTS
    / "kan_fourier_10b_ablation_full_geo_2expert_no_quaternion_mseed361_predictions.npz",
    "First-order pooling": RESULTS
    / "kan_fourier_10b_ablation_full_geo_2expert_no_spd_mseed361_predictions.npz",
    "Parameter-matched MLP": RESULTS
    / "kan_fourier_10b_ablation_full_geo_2expert_no_fourier_kan_mseed361_predictions.npz",
}

OUTPUT_PNG = FIGURES / "component_replacement_sensitivity_snr_10b.png"
OUTPUT_CSV = FIGURES / "component_replacement_sensitivity_snr_10b.csv"


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        return {
            "labels": np.asarray(data["labels"], dtype=np.int64),
            "snrs": np.asarray(data["snrs"], dtype=np.int32),
            "pred": np.asarray(data["pred"], dtype=np.int64),
        }


def check_alignment(caches: dict[str, dict[str, np.ndarray]]) -> None:
    reference = caches["KAN-Fourier"]
    for name, cache in caches.items():
        if not np.array_equal(reference["labels"], cache["labels"]):
            raise ValueError(f"Label alignment mismatch: {name}")
        if not np.array_equal(reference["snrs"], cache["snrs"]):
            raise ValueError(f"SNR alignment mismatch: {name}")


def overall_accuracy(cache: dict[str, np.ndarray]) -> float:
    return float(100.0 * np.mean(cache["pred"] == cache["labels"]))


def accuracy_by_snr(cache: dict[str, np.ndarray]) -> dict[int, float]:
    result: dict[int, float] = {}
    for snr in sorted(np.unique(cache["snrs"]).astype(int)):
        mask = cache["snrs"] == snr
        result[snr] = float(100.0 * np.mean(cache["pred"][mask] == cache["labels"][mask]))
    return result


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11.5,
            "axes.labelsize": 12.5,
            "axes.titlesize": 12.5,
            "legend.fontsize": 9.6,
            "xtick.labelsize": 10.3,
            "ytick.labelsize": 10.3,
            "axes.linewidth": 1.0,
        }
    )


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    configure_style()
    caches = {name: load_predictions(path) for name, path in CACHES.items()}
    check_alignment(caches)

    curves = {name: accuracy_by_snr(cache) for name, cache in caches.items()}
    snrs = sorted(curves["KAN-Fourier"])
    styles = {
        "KAN-Fourier": dict(color="#1F4E79", marker="o", linestyle="-", linewidth=2.65),
        "Real-valued multiscale conv": dict(
            color="#009E73", marker="^", linestyle="--", linewidth=2.0
        ),
        "First-order pooling": dict(color="#D97706", marker="s", linestyle=":", linewidth=2.2),
        "Parameter-matched MLP": dict(color="#B33A3A", marker="D", linestyle="-.", linewidth=2.0),
    }

    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(13.2, 5.25),
        gridspec_kw={"width_ratios": [1.48, 1.0], "wspace": 0.20},
    )

    for name, values in curves.items():
        ax0.plot(
            snrs,
            [values[s] for s in snrs],
            markersize=5.4,
            markerfacecolor="white",
            markeredgewidth=1.35,
            label=f"{name} ({overall_accuracy(caches[name]):.3f}%)",
            **styles[name],
        )

    ax0.set_title("(a) Accuracy across SNRs", loc="left", fontweight="bold")
    ax0.set_xlabel("SNR (dB)")
    ax0.set_ylabel("Classification accuracy (%)")
    ax0.set_xlim(-20.6, 18.6)
    ax0.set_ylim(0, 100)
    ax0.set_xticks([-20, -16, -12, -8, -4, 0, 4, 8, 12, 16, 18])
    ax0.set_yticks(np.arange(0, 101, 20))
    ax0.legend(loc="lower right", frameon=True, framealpha=0.97, borderpad=0.65)

    base_curve = curves["KAN-Fourier"]
    local_deltas = []
    for name in ("Real-valued multiscale conv", "First-order pooling"):
        values = curves[name]
        delta = np.asarray([values[s] - base_curve[s] for s in snrs], dtype=np.float64)
        local_deltas.append(delta)
        ax1.plot(
            snrs,
            delta,
            markersize=5.2,
            markerfacecolor="white",
            markeredgewidth=1.35,
            label=name,
            **styles[name],
        )

    ax1.axhline(0.0, color="#333333", linewidth=1.1)
    ax1.set_title("(b) Local effect of component replacements", loc="left", fontweight="bold")
    ax1.set_xlabel("SNR (dB)")
    ax1.set_ylabel("Accuracy difference from KAN-Fourier (pp)")
    ax1.set_xlim(-20.6, 18.6)
    delta_values = np.concatenate(local_deltas)
    delta_min = min(-1.0, 0.5 * np.floor((float(delta_values.min()) - 0.35) / 0.5))
    delta_max = max(1.0, 0.5 * np.ceil((float(delta_values.max()) + 0.35) / 0.5))
    ax1.set_ylim(delta_min, delta_max)
    ax1.set_xticks([-20, -16, -12, -8, -4, 0, 4, 8, 12, 16, 18])
    tick_step = 1.0 if (delta_max - delta_min) <= 8.0 else 2.0
    ax1.set_yticks(np.arange(np.ceil(delta_min / tick_step) * tick_step, delta_max + 0.01, tick_step))
    ax1.legend(loc="lower right", frameon=True, framealpha=0.97, borderpad=0.65)

    for ax in (ax0, ax1):
        ax.grid(True, axis="y", color="#CCD2D8", linewidth=0.78, alpha=0.78)
        ax.grid(True, axis="x", color="#E4E7EA", linewidth=0.58, alpha=0.62)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("#333333")

    fig.subplots_adjust(left=0.065, right=0.99, top=0.93, bottom=0.13)
    fig.savefig(OUTPUT_PNG, dpi=360, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["snr", *CACHES.keys()])
        for snr in snrs:
            writer.writerow([snr, *[f"{curves[name][snr]:.6f}" for name in CACHES]])

    print("Alignment check passed for all RML2016.10B component predictions.")
    for name, cache in caches.items():
        print(f"{name:32s} overall={overall_accuracy(cache):.3f}%")
    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
