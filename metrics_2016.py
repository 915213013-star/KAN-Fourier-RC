"""Shared RadioML2016 metrics and plotting utilities.

This module intentionally contains reporting helpers only.  It has no access to
training, validation-policy selection, or test-time routing logic.
"""

from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


EPS = 1e-12
NUM_CLASSES = 11
DEFAULT_MOD_CLASSES = [
    "8PSK",
    "AM-DSB",
    "AM-SSB",
    "BPSK",
    "CPFSK",
    "GFSK",
    "PAM4",
    "QAM16",
    "QAM64",
    "QPSK",
    "WBFM",
]
TRANSITION_SNRS = np.asarray([-10, -8, -6, -4, -2], dtype=np.int32)
EDGE_LOW_SNRS = np.asarray([-18, -16], dtype=np.int32)


def normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float32)
    probs = np.clip(probs, EPS, 1.0)
    return probs / (probs.sum(axis=1, keepdims=True) + EPS)


def metrics_from_probs(
    probs: np.ndarray,
    labels: np.ndarray,
    snrs: np.ndarray,
) -> Dict[str, Any]:
    probs = normalize_probs(probs)
    labels = np.asarray(labels, dtype=np.int64)
    snrs = np.asarray(snrs, dtype=np.int32)
    pred = probs.argmax(axis=1).astype(np.int64)

    def accuracy(mask: np.ndarray) -> float:
        mask = np.asarray(mask, dtype=bool)
        if not np.any(mask):
            return 0.0
        return float(np.mean(pred[mask] == labels[mask]) * 100.0)

    by_snr = {
        int(snr): accuracy(snrs == snr)
        for snr in sorted(np.unique(snrs).tolist())
    }
    return {
        "overall_acc": float(np.mean(pred == labels) * 100.0),
        "transition_acc": accuracy(np.isin(snrs, TRANSITION_SNRS)),
        "edge_low_acc": accuracy(np.isin(snrs, EDGE_LOW_SNRS)),
        "negative_acc": accuracy(snrs < 0),
        "high_acc": accuracy(snrs >= 0),
        "by_snr": by_snr,
        "pred": pred,
    }


def print_metrics_line(prefix: str, metrics: Dict[str, Any]) -> None:
    print(
        f"{prefix:<34} "
        f"Overall={metrics['overall_acc']:7.3f}% | "
        f"Trans={metrics['transition_acc']:7.3f}% | "
        f"Edge={metrics['edge_low_acc']:7.3f}% | "
        f"Neg={metrics['negative_acc']:7.3f}% | "
        f"High={metrics['high_acc']:7.3f}%"
    )


def print_snr_table(by_snr: Dict[int, float]) -> None:
    print("\nAccuracy by SNR")
    print("------------------------------------------")
    print(f"{'SNR (dB)':<12} | {'Accuracy (%)':>12}")
    print("------------------------------------------")
    for snr in sorted(by_snr):
        print(f"{snr:<12} | {by_snr[snr]:12.2f}")
    print("------------------------------------------")


def plot_curve(by_snr: Dict[int, float], save_path: str, title: str) -> None:
    xs = sorted(by_snr)
    ys = [by_snr[x] for x in xs]
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.xticks(xs, rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def plot_cm_at_snr(
    labels: np.ndarray,
    pred: np.ndarray,
    snrs: np.ndarray,
    mod_classes: List[str],
    target_snr: int,
    save_path: str,
    title_prefix: str,
) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    snrs = np.asarray(snrs, dtype=np.int32)
    mask = snrs == int(target_snr)
    y_true = labels[mask]
    y_pred = pred[mask]
    accuracy = float(np.mean(y_true == y_pred) * 100.0) if len(y_true) else 0.0
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(mod_classes)),
    ).astype(np.float32)
    cm = cm / (cm.sum(axis=1, keepdims=True) + EPS)

    plt.figure(figsize=(10, 8))
    image = plt.imshow(
        cm,
        interpolation="nearest",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
    )
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.title(f"{title_prefix} at {target_snr} dB (Acc: {accuracy:.2f}%)")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(np.arange(len(mod_classes)), mod_classes, rotation=45, ha="right")
    plt.yticks(np.arange(len(mod_classes)), mod_classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            if value >= 0.005:
                plt.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.55 else "black",
                    fontsize=8,
                )
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()
    return accuracy
