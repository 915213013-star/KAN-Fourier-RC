import argparse
import os
import sys

import numpy as np

import metrics_2016 as base


try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def relpath(*parts):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)


def parse_args():
    p = argparse.ArgumentParser(description="Report test metrics from a Fourier single-model val/test probability cache.")
    p.add_argument(
        "--cache",
        type=str,
        default=relpath("results", "fourier_auxswa_single_mseed231_split1_valtest_probs_for_meta.npz"),
    )
    p.add_argument("--output_suffix", type=str, default="")
    p.add_argument("--cm_snrs", type=int, nargs="+", default=[0, -6])
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.cache):
        raise FileNotFoundError(args.cache)
    z = np.load(args.cache, allow_pickle=True)
    val_prob = z["val_prob"].astype(np.float32)
    test_prob = z["test_prob"].astype(np.float32)
    labels_val = z["labels_val"].astype(np.int64)
    labels_test = z["labels_test"].astype(np.int64)
    snrs_val = z["snrs_val"].astype(np.int32)
    snrs_test = z["snrs_test"].astype(np.int32)
    mod_classes = [str(x) for x in z.get("mod_classes", np.asarray(base.DEFAULT_MOD_CLASSES)).tolist()]

    val_m = base.metrics_from_probs(val_prob, labels_val, snrs_val)
    test_m = base.metrics_from_probs(test_prob, labels_test, snrs_test)
    suffix = args.output_suffix or os.path.splitext(os.path.basename(args.cache))[0]

    print("=" * 120)
    print("Final single-model Fourier report")
    print("=" * 120)
    if "checkpoint" in z:
        print(f"Checkpoint: {str(z['checkpoint'][0])}")
    if "selected_source" in z:
        print(f"Selected source: {str(z['selected_source'][0])}")
    print(f"Cache: {args.cache}")
    base.print_metrics_line("Validation", val_m)
    base.print_metrics_line("Test", test_m)
    print("=" * 120)
    base.print_snr_table(test_m["by_snr"])

    curve_path = relpath("results", f"accuracy_vs_snr_{suffix}.png")
    base.plot_curve(test_m["by_snr"], curve_path, f"{suffix} accuracy vs SNR")
    print(f"[*] SNR curve saved: {curve_path}")

    for snr in args.cm_snrs:
        cm_path = relpath("results", f"confusion_matrix_{snr}dB_{suffix}.png")
        acc = base.plot_cm_at_snr(labels_test, test_m["pred"], snrs_test, mod_classes, int(snr), cm_path, suffix)
        print(f"[*] {snr} dB confusion matrix saved: {cm_path} (Acc={acc:.2f}%)")

    pred_path = relpath("results", f"{suffix}_predictions.npz")
    np.savez_compressed(
        pred_path,
        labels=labels_test.astype(np.int64),
        snrs=snrs_test.astype(np.int32),
        pred=test_m["pred"].astype(np.int64),
        final_prob=test_prob.astype(np.float32),
        mod_classes=np.asarray(mod_classes),
    )
    print(f"[*] Predictions saved: {pred_path}")


if __name__ == "__main__":
    main()
