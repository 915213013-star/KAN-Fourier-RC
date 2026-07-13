import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


DEFAULT_10A = os.path.join(
    os.path.dirname(__file__),
    "results",
    "fullgeo2expert_best66318_crossfit_late_microstage_router_split1_predictions.npz",
)
DEFAULT_10B = os.path.join(
    os.path.dirname(__file__),
    "results",
    "fourier_compressed_10b_fullgeo2expert_mseed361_hcs_pairwise_extraaux_meta_split1_predictions.npz",
)
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(__file__), "paper_figures", "fig_probability_tsne_10ab.png"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create class- and SNR-balanced t-SNE plots of final probability vectors."
    )
    parser.add_argument("--cache_10a", default=DEFAULT_10A)
    parser.add_argument("--cache_10b", default=DEFAULT_10B)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--points_per_class_snr", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--perplexity", type=float, default=40.0)
    parser.add_argument("--max_iter", type=int, default=1600)
    return parser.parse_args()


def load_prediction_cache(path):
    cache = np.load(path, allow_pickle=True)
    required = {"labels", "snrs", "final_prob", "mod_classes"}
    missing = sorted(required.difference(cache.files))
    if missing:
        raise KeyError(f"{path} is missing required arrays: {missing}")

    labels = np.asarray(cache["labels"], dtype=np.int64)
    snrs = np.asarray(cache["snrs"], dtype=np.int32)
    probabilities = np.asarray(cache["final_prob"], dtype=np.float32)
    classes = [str(value) for value in cache["mod_classes"].tolist()]
    if probabilities.shape != (len(labels), len(classes)):
        raise ValueError(
            f"Probability shape {probabilities.shape} is incompatible with "
            f"{len(labels)} labels and {len(classes)} classes."
        )
    probabilities = np.clip(probabilities, 0.0, None)
    probabilities /= probabilities.sum(axis=1, keepdims=True).clip(min=1e-12)
    return labels, snrs, probabilities, classes


def balanced_class_snr_indices(labels, snrs, points_per_cell, seed):
    rng = np.random.default_rng(seed)
    selected = []
    for class_id in np.unique(labels):
        for snr in np.unique(snrs):
            candidates = np.flatnonzero((labels == class_id) & (snrs == snr))
            if len(candidates) == 0:
                continue
            take = min(int(points_per_cell), len(candidates))
            selected.extend(rng.choice(candidates, size=take, replace=False).tolist())
    selected = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected)
    return selected


def project(probabilities, seed, perplexity, max_iter):
    # The input remains the model's probability vector. Labels are never passed
    # to t-SNE and are used only to color the completed projection.
    reducer = TSNE(
        n_components=2,
        perplexity=float(perplexity),
        learning_rate="auto",
        init="pca",
        max_iter=int(max_iter),
        random_state=int(seed),
        metric="euclidean",
    )
    return reducer.fit_transform(probabilities).astype(np.float32)


def class_color_map(all_classes):
    palette = plt.get_cmap("tab20")
    color_ids = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1]
    return {
        class_name: palette(color_ids[index % len(color_ids)])
        for index, class_name in enumerate(all_classes)
    }


def draw_panel(ax, embedding, labels, classes, colors, title):
    for class_id, class_name in enumerate(classes):
        mask = labels == class_id
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=8.5,
            alpha=0.76,
            color=colors[class_name],
            label=class_name,
            linewidths=0,
            rasterized=True,
        )
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=8)
    ax.set_xlabel("t-SNE dimension 1", fontsize=9)
    ax.set_ylabel("t-SNE dimension 2", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("#6f7782")
        spine.set_linewidth(0.75)
    ax.legend(
        loc="upper right",
        fontsize=6.8,
        markerscale=1.55,
        frameon=True,
        framealpha=0.92,
        edgecolor="#c8ccd2",
        borderpad=0.45,
        labelspacing=0.25,
        handletextpad=0.35,
        ncol=1,
    )


def main():
    args = parse_args()
    records = []
    for dataset_name, path, seed_offset in (
        ("RML2016.10A", args.cache_10a, 0),
        ("RML2016.10B", args.cache_10b, 1),
    ):
        labels, snrs, probabilities, classes = load_prediction_cache(path)
        indices = balanced_class_snr_indices(
            labels, snrs, args.points_per_class_snr, args.seed + seed_offset
        )
        embedding = project(
            probabilities[indices],
            args.seed + seed_offset,
            args.perplexity,
            args.max_iter,
        )
        accuracy = float((probabilities.argmax(axis=1) == labels).mean() * 100.0)
        records.append(
            {
                "dataset": dataset_name,
                "cache": path,
                "indices": indices,
                "labels": labels[indices],
                "snrs": snrs[indices],
                "probabilities": probabilities[indices],
                "classes": classes,
                "embedding": embedding,
                "accuracy": accuracy,
            }
        )

    all_classes = []
    for record in records:
        for class_name in record["classes"]:
            if class_name not in all_classes:
                all_classes.append(class_name)
    colors = class_color_map(all_classes)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.titlecolor": "#20252b",
            "axes.labelcolor": "#30363d",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.25), constrained_layout=False)
    for panel_id, (ax, record) in enumerate(zip(axes, records)):
        title = (
            f"({chr(ord('a') + panel_id)}) {record['dataset']} "
            f"({record['accuracy']:.3f}% test accuracy)"
        )
        draw_panel(
            ax,
            record["embedding"],
            record["labels"],
            record["classes"],
            colors,
            title,
        )
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.085, top=0.94, wspace=0.12)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    coordinate_path = os.path.splitext(args.output)[0] + "_coords.npz"
    np.savez_compressed(
        coordinate_path,
        embedding_10a=records[0]["embedding"],
        labels_10a=records[0]["labels"],
        snrs_10a=records[0]["snrs"],
        indices_10a=records[0]["indices"],
        classes_10a=np.asarray(records[0]["classes"]),
        embedding_10b=records[1]["embedding"],
        labels_10b=records[1]["labels"],
        snrs_10b=records[1]["snrs"],
        indices_10b=records[1]["indices"],
        classes_10b=np.asarray(records[1]["classes"]),
        cache_10a=np.asarray([records[0]["cache"]]),
        cache_10b=np.asarray([records[1]["cache"]]),
        accuracy_10a=np.asarray([records[0]["accuracy"]], dtype=np.float32),
        accuracy_10b=np.asarray([records[1]["accuracy"]], dtype=np.float32),
        points_per_class_snr=np.asarray([args.points_per_class_snr], dtype=np.int32),
        seed=np.asarray([args.seed], dtype=np.int32),
        perplexity=np.asarray([args.perplexity], dtype=np.float32),
    )
    print(f"Saved figure: {args.output}")
    print(f"Saved coordinates: {coordinate_path}")
    for record in records:
        print(
            f"{record['dataset']}: {len(record['indices'])} plotted samples, "
            f"test accuracy={record['accuracy']:.6f}%"
        )


if __name__ == "__main__":
    main()
