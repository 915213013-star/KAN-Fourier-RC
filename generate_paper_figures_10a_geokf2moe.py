from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

CACHES = {
    "KAN-Fourier": RESULTS
    / "fourier_compressed_full_geo_2expert_budget_ft1_mseed261_split1_valtest_probs_for_soup.npz",
    "IQCC-Former": RESULTS
    / "cv_trn_aux_v2_d64d3_mseed13_split1_valtest_probs_for_fusion.npz",
    "KAN-Fourier-RC": RESULTS
    / "fullgeo2expert_best66318_crossfit_late_microstage_router_split1_predictions.npz",
}

CURVE_PNG = FIGURES / "fig_10a_snr.png"
CURVE_CSV = FIGURES / "fig_10a_snr.csv"
CM_PATHS = {
    -6: FIGURES / "fig_10a_cm_m6.png",
    0: FIGURES / "fig_10a_cm_0.png",
}
def load_test_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        labels = data["labels_test"] if "labels_test" in data else data["labels"]
        snrs = data["snrs_test"] if "snrs_test" in data else data["snrs"]
        if "test_prob" in data:
            pred = data["test_prob"].argmax(axis=1)
        elif "pred" in data:
            pred = data["pred"]
        elif "final_prob" in data:
            pred = data["final_prob"].argmax(axis=1)
        else:
            raise KeyError(f"No test prediction field found in {path}")
        classes = data["mod_classes"] if "mod_classes" in data else None
    return {
        "labels": np.asarray(labels, dtype=np.int64),
        "snrs": np.asarray(snrs, dtype=np.int32),
        "pred": np.asarray(pred, dtype=np.int64),
        "classes": None if classes is None else np.asarray(classes).astype(str),
    }


def check_alignment(caches: dict[str, dict[str, np.ndarray]]) -> None:
    reference_name = "KAN-Fourier-RC"
    reference = caches[reference_name]
    for name, cache in caches.items():
        if not np.array_equal(reference["labels"], cache["labels"]):
            raise ValueError(f"Label alignment mismatch: {name} vs {reference_name}")
        if not np.array_equal(reference["snrs"], cache["snrs"]):
            raise ValueError(f"SNR alignment mismatch: {name} vs {reference_name}")
        if reference["classes"] is not None and cache["classes"] is not None:
            if not np.array_equal(reference["classes"], cache["classes"]):
                raise ValueError(f"Class-order mismatch: {name} vs {reference_name}")


def accuracy_by_snr(cache: dict[str, np.ndarray]) -> dict[int, float]:
    values: dict[int, float] = {}
    for snr in sorted(np.unique(cache["snrs"]).astype(int)):
        mask = cache["snrs"] == snr
        values[snr] = float(100.0 * np.mean(cache["pred"][mask] == cache["labels"][mask]))
    return values


def overall_accuracy(cache: dict[str, np.ndarray]) -> float:
    return float(100.0 * np.mean(cache["pred"] == cache["labels"]))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 12.0,
            "axes.labelsize": 13.0,
            "axes.titlesize": 13.0,
            "legend.fontsize": 10.8,
            "xtick.labelsize": 11.0,
            "ytick.labelsize": 11.0,
            "axes.linewidth": 1.05,
        }
    )


def plot_snr_curve(caches: dict[str, dict[str, np.ndarray]]) -> None:
    style = {
        "KAN-Fourier": {"color": "#355C8A", "marker": "o", "linestyle": "--"},
        "IQCC-Former": {"color": "#D17A22", "marker": "s", "linestyle": ":"},
        "KAN-Fourier-RC": {"color": "#2F7D32", "marker": "D", "linestyle": "-"},
    }
    curves = {name: accuracy_by_snr(cache) for name, cache in caches.items()}
    snrs = sorted(next(iter(curves.values())))

    # A large source canvas keeps labels and close curves legible after IEEE
    # two-column down-scaling. Only sparse major grid lines are drawn.
    fig, ax = plt.subplots(figsize=(9.2, 5.45))
    for name, values in curves.items():
        overall = overall_accuracy(caches[name])
        kwargs = style[name]
        ax.plot(
            snrs,
            [values[snr] for snr in snrs],
            linewidth=2.8 if name == "KAN-Fourier-RC" else 2.15,
            markersize=6.3,
            markerfacecolor="white",
            markeredgewidth=1.55,
            label=f"{name} ({overall:.3f}%)",
            **kwargs,
        )

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Classification accuracy (%)")
    ax.set_xlim(-20.5, 18.5)
    ax.set_ylim(0, 100)
    ax.set_xticks([-20, -16, -12, -8, -4, 0, 4, 8, 12, 16, 18])
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(True, axis="y", which="major", color="#CDD2D8", linewidth=0.85, alpha=0.78)
    ax.grid(True, axis="x", which="major", color="#E3E6E9", linewidth=0.65, alpha=0.58)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#333333")
    ax.legend(loc="lower right", frameon=True, framealpha=0.97, ncol=1, borderpad=0.75)
    fig.tight_layout(pad=1.0)
    fig.savefig(CURVE_PNG, dpi=360, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    with CURVE_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["snr", *caches.keys()])
        for snr in snrs:
            writer.writerow([snr, *[f"{curves[name][snr]:.6f}" for name in caches]])


def plot_confusion_at_snr(cache: dict[str, np.ndarray], target_snr: int) -> float:
    mask = cache["snrs"] == target_snr
    labels = cache["labels"][mask]
    pred = cache["pred"][mask]
    classes = cache["classes"]
    if classes is None:
        classes = np.asarray([str(i) for i in range(int(cache["labels"].max()) + 1)])

    cm = confusion_matrix(labels, pred, labels=np.arange(len(classes))).astype(np.float64)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum > 0) * 100.0
    accuracy = float(100.0 * np.mean(pred == labels))

    colors = LinearSegmentedColormap.from_list(
        "paper_blue", ["#FFFFFF", "#DDEBF7", "#8DB8D8", "#2C6B9A", "#123D63"]
    )
    fig, ax = plt.subplots(figsize=(8.2, 7.35))
    image = ax.imshow(cm_pct, cmap=colors, vmin=0, vmax=100, interpolation="nearest")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Row-normalized percentage (%)", rotation=90, labelpad=9)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=42, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted modulation")
    ax.set_ylabel("True modulation")
    ax.set_title(f"KAN-Fourier-RC at {target_snr} dB (accuracy = {accuracy:.2f}%)", pad=9)

    threshold = 52.0
    for i in range(cm_pct.shape[0]):
        for j in range(cm_pct.shape[1]):
            value = cm_pct[i, j]
            if value < 0.05:
                text = ""
            elif value < 10:
                text = f"{value:.1f}"
            else:
                text = f"{value:.0f}"
            if text:
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8.8,
                    color="white" if value >= threshold else "#202020",
                )

    ax.set_xticks(np.arange(-0.5, len(classes), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(classes), 1), minor=True)
    ax.grid(which="minor", color="#B8C2CC", linewidth=0.62)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout(pad=1.0)
    fig.savefig(CM_PATHS[target_snr], dpi=360, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return accuracy


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    configure_style()
    caches = {name: load_test_cache(path) for name, path in CACHES.items()}
    check_alignment(caches)
    print(f"Alignment check passed for {len(caches)} RML2016.10a caches.")
    for name, cache in caches.items():
        print(f"{name:20s} overall={overall_accuracy(cache):.3f}%")

    plot_snr_curve(caches)
    final_cache = caches["KAN-Fourier-RC"]
    for snr in CM_PATHS:
        accuracy = plot_confusion_at_snr(final_cache, snr)
        print(f"Confusion matrix at {snr:>2d} dB: accuracy={accuracy:.3f}%")

    print(f"Saved: {CURVE_PNG}")
    print(f"Saved: {CURVE_CSV}")
    for path in CM_PATHS.values():
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
