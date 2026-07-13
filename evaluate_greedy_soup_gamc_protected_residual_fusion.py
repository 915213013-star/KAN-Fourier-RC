# -*- coding: utf-8 -*-
"""
evaluate_greedy_soup_gamc_protected_residual_fusion.py

目的：
    在不改变 RML2016.10A 既有 train/val/test 划分、且推理阶段不使用真实 SNR 的前提下，
    对已有 Greedy Soup neural probability cache 与 GAMC-V2 probability/router cache 做更保守、
    更适合论文写作的 blind protected residual fusion。

核心思想：
    1) 只读取已经生成好的 val/test 概率缓存；不重新划分数据，不重新训练 neural backbone。
    2) 先在 validation 上对 neural/GAMC 概率做 temperature scaling。
    3) 用 predicted router probability + neural uncertainty 构造 blind low-SNR gate；gate 不接收真实 SNR。
    4) GAMC 不再 hard replace neural，而是作为 residual logit correction：
           logits_fused = logits_neural + alpha * (logits_gamc - logits_neural)
       其中 alpha 由 gate 和 blind reliability evidence 决定。
    5) 在 validation 上训练 OOF meta-selector；selector 特征只包含可推理观测量：
       neural/gamc/candidate probability, router probability, confidence, entropy, margin, agreement 等。
       不把真实 SNR、label、sample index 放入 selector features。
    6) hyperparameter selection 只使用 validation labels/SNRs；test labels/SNRs 只用于最终一次报告。

运行示例：
  python evaluate_greedy_soup_gamc_protected_residual_fusion.py --split_seed 1

说明：
    - 本脚本假设以下 cache 已经存在：
        results/greedy_soup_identity_valtest_probs_for_gamc_fusion.npz
        results/gamc_lite_v2_xgb_split1_valtest_probs_for_fusion.npz
    - 如要测试 roll_phase soup，只需要换 --soup_prob_cache 即可。
    - 本脚本不把 GAMC 分支计入 neural params；它是额外 graph/statistical/XGBoost branch。
"""

import os
import csv
import json
import math
import argparse
import warnings
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")

EPS = 1e-12
NUM_CLASSES = 11
DEFAULT_MOD_CLASSES = [
    "8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK",
    "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"
]

ULTRA_LOW_SNRS = np.array([-20, -18, -16, -14, -12], dtype=np.int32)
TRANSITION_SNRS = np.array([-10, -8, -6, -4, -2], dtype=np.int32)
EDGE_LOW_SNRS = np.array([-18, -16], dtype=np.int32)


def project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def relpath(*parts: str) -> str:
    return os.path.join(project_root(), *parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="High-SNR-protected residual blind fusion for Greedy Soup + GAMC."
    )

    p.add_argument("--split_seed", type=int, default=1)
    p.add_argument("--random_state", type=int, default=2026)

    p.add_argument(
        "--soup_prob_cache",
        type=str,
        default=relpath("results", "greedy_soup_identity_valtest_probs_for_gamc_fusion.npz"),
        help="Greedy Soup val/test probability cache. Keep identity cache by default.",
    )
    p.add_argument(
        "--gamc_cache",
        type=str,
        default=relpath("results", "gamc_lite_v2_xgb_split1_valtest_probs_for_fusion.npz"),
        help="GAMC-V2 XGBoost/graph/statistical branch val/test probability and router cache.",
    )

    # Calibration.
    p.add_argument(
        "--temperature_grid",
        type=float,
        nargs="+",
        default=[0.70, 0.85, 1.00, 1.15, 1.30, 1.50, 1.75, 2.00, 2.50, 3.00, 4.00, 5.00],
        help="Temperature candidates used only on validation labels for calibration.",
    )
    p.add_argument("--disable_temperature_scaling", action="store_true")

    # Blind gate search. These are intentionally moderate to avoid huge grids.
    p.add_argument("--low_bin_counts", type=int, nargs="+", default=[1, 2])
    p.add_argument("--low_thresholds", type=float, nargs="+", default=[0.40, 0.50, 0.60, 0.70])
    p.add_argument("--high_max_thresholds", type=float, nargs="+", default=[0.35, 0.50, 0.65])
    p.add_argument("--low_gap_thresholds", type=float, nargs="+", default=[-0.10, 0.05, 0.20])
    p.add_argument("--neural_conf_thresholds", type=float, nargs="+", default=[0.50, 0.60, 0.70, 0.85])
    p.add_argument("--neural_margin_thresholds", type=float, nargs="+", default=[0.20, 0.35, 0.50, 1.01])
    p.add_argument("--alpha_max_values", type=float, nargs="+", default=[0.25, 0.40, 0.55, 0.70, 0.85])
    p.add_argument(
        "--hard_keep_conf",
        type=float,
        default=0.92,
        help="If neural confidence and margin are both high, keep neural regardless of router evidence.",
    )
    p.add_argument("--hard_keep_margin", type=float, default=0.55)

    # Two-stage search.
    p.add_argument(
        "--top_k_candidates",
        type=int,
        default=12,
        help="Only the best deterministic residual candidates enter OOF selector search.",
    )
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--selector_Cs", type=float, nargs="+", default=[0.01, 0.03, 0.10, 0.30])
    p.add_argument("--selector_neg_weights", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    p.add_argument("--selector_pos_weight", type=float, default=1.0)
    p.add_argument(
        "--selector_thresholds",
        type=float,
        nargs="+",
        default=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
    )
    p.add_argument("--selector_max_iter", type=int, default=1000)
    p.add_argument(
        "--skip_selector",
        action="store_true",
        help="Use deterministic residual fusion only. Useful for debugging.",
    )

    # Validation selection score.
    # Overall accuracy remains dominant; the extra terms encourage low-SNR improvement and high-SNR protection.
    p.add_argument("--score_overall_weight", type=float, default=1.0)
    p.add_argument("--score_negative_gain_weight", type=float, default=0.020)
    p.add_argument("--score_edge_gain_weight", type=float, default=0.020)
    p.add_argument("--score_transition_gain_weight", type=float, default=0.010)
    p.add_argument("--score_high_penalty", type=float, default=3.00)
    p.add_argument("--high_tolerance", type=float, default=0.05)
    p.add_argument("--score_changed_high_penalty", type=float, default=0.015)
    p.add_argument("--score_changed_nonultra_penalty", type=float, default=0.006)

    p.add_argument(
        "--output_suffix",
        type=str,
        default=None,
        help="Default: greedy_soup_gamc_protected_residual_fusion_split{split_seed}",
    )
    p.add_argument("--save_top_records", type=int, default=80)
    p.add_argument("--cm_snrs", type=int, nargs="+", default=[0, -6])

    return p.parse_args()


# ----------------------------- basic probability utilities -----------------------------


def normalize_probs(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float32)
    p = np.clip(p, EPS, 1.0)
    return p / (p.sum(axis=1, keepdims=True) + EPS)


def softmax_np(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float32)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=1, keepdims=True) + EPS)


def prob_to_logits(p: np.ndarray) -> np.ndarray:
    return np.log(normalize_probs(p) + EPS).astype(np.float32)


def temperature_scale_probs(p: np.ndarray, temperature: float) -> np.ndarray:
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = prob_to_logits(p) / temperature
    return normalize_probs(softmax_np(logits))


def nll_from_probs(p: np.ndarray, labels: np.ndarray) -> float:
    p = normalize_probs(p)
    labels = labels.astype(np.int64)
    return float(-np.mean(np.log(p[np.arange(len(labels)), labels] + EPS)))


def fit_temperature(
    val_prob: np.ndarray,
    labels_val: np.ndarray,
    grid: List[float],
    name: str,
) -> Tuple[float, float]:
    best_t = 1.0
    best_nll = nll_from_probs(val_prob, labels_val)
    for t in grid:
        cur = nll_from_probs(temperature_scale_probs(val_prob, t), labels_val)
        if cur < best_nll - 1e-12:
            best_t = float(t)
            best_nll = float(cur)
    print(f"[*] Calibration {name}: best T={best_t:.3f}, val NLL={best_nll:.6f}")
    return best_t, best_nll


def entropy(p: np.ndarray) -> np.ndarray:
    p = normalize_probs(p)
    return -np.sum(p * np.log(p + EPS), axis=1)


def margin(p: np.ndarray) -> np.ndarray:
    p = normalize_probs(p)
    s = np.sort(p, axis=1)
    return (s[:, -1] - s[:, -2]).astype(np.float32)


def one_hot(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.int64)
    out = np.zeros((len(x), n), dtype=np.float32)
    out[np.arange(len(x)), x] = 1.0
    return out


# ----------------------------- loading and sanity checks -----------------------------


def load_npz_required(path: str, required_keys: List[str]) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required cache: {path}")
    z = np.load(path, allow_pickle=True)
    missing = [k for k in required_keys if k not in z]
    if missing:
        raise KeyError(f"{path} is missing keys: {missing}")
    return {k: z[k] for k in z.files}


def load_soup_cache(path: str) -> Dict[str, Any]:
    z = load_npz_required(
        path,
        ["val_prob", "test_prob", "labels_val", "snrs_val", "labels_test", "snrs_test"],
    )
    mod_classes = z.get("mod_classes", np.array(DEFAULT_MOD_CLASSES))
    return {
        "val_prob": normalize_probs(z["val_prob"]),
        "test_prob": normalize_probs(z["test_prob"]),
        "labels_val": z["labels_val"].astype(np.int64),
        "snrs_val": z["snrs_val"].astype(np.int32),
        "labels_test": z["labels_test"].astype(np.int64),
        "snrs_test": z["snrs_test"].astype(np.int32),
        "mod_classes": [str(x) for x in mod_classes.tolist()],
    }


def load_gamc_cache(path: str) -> Dict[str, Any]:
    z = load_npz_required(
        path,
        ["val_prob", "test_prob", "val_router", "test_router"],
    )
    return {
        "val_prob": normalize_probs(z["val_prob"]),
        "test_prob": normalize_probs(z["test_prob"]),
        "val_router": normalize_probs(z["val_router"]),
        "test_router": normalize_probs(z["test_router"]),
        "labels_val": z["labels_val"].astype(np.int64) if "labels_val" in z else None,
        "snrs_val": z["snrs_val"].astype(np.int32) if "snrs_val" in z else None,
        "labels_test": z["labels_test"].astype(np.int64) if "labels_test" in z else None,
        "snrs_test": z["snrs_test"].astype(np.int32) if "snrs_test" in z else None,
    }


def assert_alignment(soup: Dict[str, Any], gamc: Dict[str, Any]) -> None:
    checks = [
        ("labels_val", soup["labels_val"], gamc.get("labels_val")),
        ("labels_test", soup["labels_test"], gamc.get("labels_test")),
        ("snrs_val", soup["snrs_val"], gamc.get("snrs_val")),
        ("snrs_test", soup["snrs_test"], gamc.get("snrs_test")),
    ]
    for name, a, b in checks:
        if b is not None and not np.all(a == b):
            raise RuntimeError(f"Alignment check failed for {name}. Do not fuse misaligned caches.")
    print("[*] Alignment check passed: labels/snrs of neural cache and GAMC cache are identical.")


# ----------------------------- metrics and plots -----------------------------


def metrics_from_probs(probs: np.ndarray, labels: np.ndarray, snrs: np.ndarray) -> Dict[str, Any]:
    probs = normalize_probs(probs)
    labels = labels.astype(np.int64)
    snrs = snrs.astype(np.int32)
    pred = probs.argmax(axis=1).astype(np.int64)

    def acc(mask: np.ndarray) -> float:
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() == 0:
            return 0.0
        return float((pred[mask] == labels[mask]).mean() * 100.0)

    by_snr = {}
    for s in sorted(np.unique(snrs).tolist()):
        by_snr[int(s)] = acc(snrs == s)

    return {
        "overall_acc": float((pred == labels).mean() * 100.0),
        "transition_acc": acc(np.isin(snrs, TRANSITION_SNRS)),
        "edge_low_acc": acc(np.isin(snrs, EDGE_LOW_SNRS)),
        "negative_acc": acc(snrs < 0),
        "high_acc": acc(snrs >= 0),
        "by_snr": by_snr,
        "pred": pred,
    }


def print_metrics_line(prefix: str, m: Dict[str, Any]) -> None:
    print(
        f"{prefix:<34} "
        f"Overall={m['overall_acc']:7.3f}% | "
        f"Trans={m['transition_acc']:7.3f}% | "
        f"Edge={m['edge_low_acc']:7.3f}% | "
        f"Neg={m['negative_acc']:7.3f}% | "
        f"High={m['high_acc']:7.3f}%"
    )


def print_snr_table(by_snr: Dict[int, float]) -> None:
    print("\n各 SNR 准确率：")
    print("------------------------------------------")
    print(f"{'SNR(dB)':<12} | {'Accuracy(%)':>12}")
    print("------------------------------------------")
    for s in sorted(by_snr.keys()):
        print(f"{s:<12} | {by_snr[s]:12.2f}")
    print("------------------------------------------")


def plot_curve(by_snr: Dict[int, float], save_path: str, title: str) -> None:
    xs = sorted(by_snr.keys())
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


def plot_cm_at_snr(labels: np.ndarray, pred: np.ndarray, snrs: np.ndarray,
                   mod_classes: List[str], target_snr: int,
                   save_path: str, title_prefix: str) -> float:
    labels = labels.astype(np.int64)
    pred = pred.astype(np.int64)
    snrs = snrs.astype(np.int32)
    mask = snrs == int(target_snr)
    y_true = labels[mask]
    y_pred = pred[mask]
    acc = float((y_true == y_pred).mean() * 100.0) if len(y_true) else 0.0
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(mod_classes))).astype(np.float32)
    cm = cm / (cm.sum(axis=1, keepdims=True) + EPS)

    plt.figure(figsize=(10, 8))
    im = plt.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f"{title_prefix} at {target_snr} dB (Acc: {acc:.2f}%)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(np.arange(len(mod_classes)), mod_classes, rotation=45, ha="right")
    plt.yticks(np.arange(len(mod_classes)), mod_classes)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            if v >= 0.005:
                plt.text(
                    j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.55 else "black", fontsize=8,
                )
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()
    return acc


# ----------------------------- blind residual fusion -----------------------------


def router_low_high(router: np.ndarray, low_bin_count: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    router = normalize_probs(router)
    rb = router.shape[1]
    low_bin_count = int(max(1, min(low_bin_count, rb)))
    # Assumption inherited from current GAMC cache: earlier router bins correspond to lower SNR regions.
    low_prob = router[:, :low_bin_count].sum(axis=1)
    high_count = min(2, rb)
    high_prob = router[:, rb - high_count:].sum(axis=1)
    low_gap = low_prob - high_prob
    return low_prob.astype(np.float32), high_prob.astype(np.float32), low_gap.astype(np.float32)


def make_gate(neural: np.ndarray, router: np.ndarray, cfg: Dict[str, Any], args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Blind gate. This function intentionally does NOT accept labels or true SNR.
    """
    neural = normalize_probs(neural)
    router = normalize_probs(router)
    n_conf = neural.max(axis=1)
    n_margin = margin(neural)
    low_prob, high_prob, low_gap = router_low_high(router, int(cfg["low_bin_count"]))

    gate = (
        (low_prob >= float(cfg["low_thr"]))
        & (high_prob <= float(cfg["high_max_thr"]))
        & (low_gap >= float(cfg["low_gap_thr"]))
        & (n_conf <= float(cfg["neural_conf_thr"]))
        & (n_margin <= float(cfg["neural_margin_thr"]))
    )

    # High-confidence neural protection. This is still blind because it uses only neural posterior statistics.
    hard_keep = (n_conf >= float(args.hard_keep_conf)) & (n_margin >= float(args.hard_keep_margin))
    gate = gate & (~hard_keep)

    aux = {
        "n_conf": n_conf.astype(np.float32),
        "n_margin": n_margin.astype(np.float32),
        "low_prob": low_prob,
        "high_prob": high_prob,
        "low_gap": low_gap,
        "hard_keep": hard_keep.astype(bool),
    }
    return gate.astype(bool), hard_keep.astype(bool), aux


def make_residual_candidate(
    neural: np.ndarray,
    gamc: np.ndarray,
    router: np.ndarray,
    cfg: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Return candidate probability, blind gate, per-sample alpha, and auxiliary blind evidence.
    No label/SNR is used here.
    """
    neural = normalize_probs(neural)
    gamc = normalize_probs(gamc)
    router = normalize_probs(router)

    gate, hard_keep, aux = make_gate(neural, router, cfg, args)
    n_conf = aux["n_conf"]
    low_prob = aux["low_prob"]

    low_thr = float(cfg["low_thr"])
    nconf_thr = float(cfg["neural_conf_thr"])
    alpha_max = float(cfg["alpha_max"])

    # Continuous blind reliability factor.
    # Stronger low-SNR router evidence and lower neural confidence -> larger residual alpha.
    low_strength = np.clip((low_prob - low_thr) / max(1.0 - low_thr, 1e-6), 0.0, 1.0)
    if nconf_thr <= 1e-6:
        unc_strength = np.zeros_like(n_conf)
    else:
        unc_strength = np.clip((nconf_thr - n_conf) / max(nconf_thr, 1e-6), 0.0, 1.0)

    reliability = (0.50 + 0.50 * low_strength) * (0.50 + 0.50 * unc_strength)
    alpha = alpha_max * reliability * gate.astype(np.float32)
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)

    logits_n = prob_to_logits(neural)
    logits_g = prob_to_logits(gamc)
    logits_c = logits_n + alpha[:, None] * (logits_g - logits_n)
    cand = normalize_probs(softmax_np(logits_c)).astype(np.float32)

    return cand, gate, alpha, aux


def apply_selector(
    neural: np.ndarray,
    cand: np.ndarray,
    switch_prob: np.ndarray,
    threshold: float,
    base_gate: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    neural = normalize_probs(neural)
    cand = normalize_probs(cand)
    switch_prob = np.asarray(switch_prob, dtype=np.float32)
    use_cand = switch_prob >= float(threshold)
    if base_gate is not None:
        use_cand = use_cand & np.asarray(base_gate, dtype=bool)
    out = neural.copy()
    out[use_cand] = cand[use_cand]
    return normalize_probs(out).astype(np.float32), use_cand.astype(bool)


# ----------------------------- diagnostics and validation score -----------------------------


def switch_diagnostics(
    neural: np.ndarray,
    final_prob: np.ndarray,
    gate: np.ndarray,
    use_cand: np.ndarray,
    alpha: np.ndarray,
    snrs: np.ndarray,
) -> Dict[str, float]:
    neural = normalize_probs(neural)
    final_prob = normalize_probs(final_prob)
    snrs = snrs.astype(np.int32)
    gate = np.asarray(gate, dtype=bool)
    use_cand = np.asarray(use_cand, dtype=bool)
    alpha = np.asarray(alpha, dtype=np.float32)

    pred_n = neural.argmax(axis=1)
    pred_f = final_prob.argmax(axis=1)
    changed = pred_n != pred_f

    ultra = np.isin(snrs, ULTRA_LOW_SNRS)
    nonultra = ~ultra
    high = snrs >= 0

    def rate(mask: np.ndarray) -> float:
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() == 0:
            return 0.0
        return float(mask.mean() * 100.0)

    def subrate(flag: np.ndarray, mask: np.ndarray) -> float:
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() == 0:
            return 0.0
        return float(flag[mask].mean() * 100.0)

    effective = use_cand & gate & (alpha > 1e-8)

    return {
        "gate_rate": rate(gate),
        "use_rate": rate(use_cand),
        "effective_use_rate": rate(effective),
        "changed_rate": rate(changed),
        "changed_ultra_rate": subrate(changed, ultra),
        "changed_nonultra_rate": subrate(changed, nonultra),
        "changed_high_rate": subrate(changed, high),
        "effective_ultra_rate": subrate(effective, ultra),
        "effective_nonultra_rate": subrate(effective, nonultra),
        "effective_high_rate": subrate(effective, high),
        "mean_alpha_all": float(alpha.mean()),
        "mean_alpha_gate": float(alpha[gate].mean()) if gate.any() else 0.0,
    }


def selection_score(m: Dict[str, Any], diag: Dict[str, float], base_m: Dict[str, Any], args: argparse.Namespace) -> float:
    high_drop = max(0.0, base_m["high_acc"] - m["high_acc"] - float(args.high_tolerance))
    score = 0.0
    score += float(args.score_overall_weight) * m["overall_acc"]
    score += float(args.score_negative_gain_weight) * (m["negative_acc"] - base_m["negative_acc"])
    score += float(args.score_edge_gain_weight) * (m["edge_low_acc"] - base_m["edge_low_acc"])
    score += float(args.score_transition_gain_weight) * (m["transition_acc"] - base_m["transition_acc"])
    score -= float(args.score_high_penalty) * high_drop
    score -= float(args.score_changed_high_penalty) * diag["changed_high_rate"]
    score -= float(args.score_changed_nonultra_penalty) * diag["changed_nonultra_rate"]
    return float(score)


# ----------------------------- selector features and training -----------------------------


def build_selector_features(
    neural: np.ndarray,
    gamc: np.ndarray,
    cand: np.ndarray,
    router: np.ndarray,
    gate: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """
    Features for selector. This function intentionally does NOT accept labels or true SNR.
    """
    neural = normalize_probs(neural)
    gamc = normalize_probs(gamc)
    cand = normalize_probs(cand)
    router = normalize_probs(router)
    gate = np.asarray(gate, dtype=np.float32)
    alpha = np.asarray(alpha, dtype=np.float32)

    n_top = neural.argmax(axis=1)
    g_top = gamc.argmax(axis=1)
    c_top = cand.argmax(axis=1)
    r_top = router.argmax(axis=1)

    n_conf = neural.max(axis=1)
    g_conf = gamc.max(axis=1)
    c_conf = cand.max(axis=1)
    r_conf = router.max(axis=1)

    n_margin = margin(neural)
    g_margin = margin(gamc)
    c_margin = margin(cand)
    r_margin = margin(router)

    n_ent = entropy(neural)
    g_ent = entropy(gamc)
    c_ent = entropy(cand)
    r_ent = entropy(router)

    ng_agree = (n_top == g_top).astype(np.float32)
    nc_agree = (n_top == c_top).astype(np.float32)
    gc_agree = (g_top == c_top).astype(np.float32)

    l1_ng = np.sum(np.abs(neural - gamc), axis=1)
    l1_nc = np.sum(np.abs(neural - cand), axis=1)
    l1_gc = np.sum(np.abs(gamc - cand), axis=1)
    dot_ng = np.sum(neural * gamc, axis=1)
    dot_nc = np.sum(neural * cand, axis=1)

    kl_ng = np.sum(neural * np.log((neural + EPS) / (gamc + EPS)), axis=1)
    kl_gn = np.sum(gamc * np.log((gamc + EPS) / (neural + EPS)), axis=1)

    idx = np.arange(len(neural))
    gamc_on_neural_top = gamc[idx, n_top]
    neural_on_gamc_top = neural[idx, g_top]
    cand_on_neural_top = cand[idx, n_top]
    neural_on_cand_top = neural[idx, c_top]

    low1, high2, gap1 = router_low_high(router, 1)
    low2, _, gap2 = router_low_high(router, 2)

    scalar = np.stack(
        [
            n_conf, g_conf, c_conf, r_conf,
            n_margin, g_margin, c_margin, r_margin,
            n_ent, g_ent, c_ent, r_ent,
            ng_agree, nc_agree, gc_agree,
            l1_ng, l1_nc, l1_gc, dot_ng, dot_nc, kl_ng, kl_gn,
            low1, low2, high2, gap1, gap2,
            gamc_on_neural_top, neural_on_gamc_top,
            cand_on_neural_top, neural_on_cand_top,
            g_conf - n_conf, c_conf - n_conf,
            g_margin - n_margin, c_margin - n_margin,
            g_ent - n_ent, c_ent - n_ent,
            gate, alpha,
        ],
        axis=1,
    ).astype(np.float32)

    X = np.concatenate(
        [
            scalar,
            neural.astype(np.float32),
            gamc.astype(np.float32),
            cand.astype(np.float32),
            router.astype(np.float32),
            one_hot(n_top, NUM_CLASSES),
            one_hot(g_top, NUM_CLASSES),
            one_hot(c_top, NUM_CLASSES),
            one_hot(r_top, router.shape[1]),
        ],
        axis=1,
    )
    return np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)


def fit_selector_oof(
    X: np.ndarray,
    neural_prob: np.ndarray,
    cand_prob: np.ndarray,
    labels: np.ndarray,
    C: float,
    folds: int,
    seed: int,
    pos_weight: float,
    neg_weight: float,
    max_iter: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    labels = labels.astype(np.int64)
    neural_correct = neural_prob.argmax(axis=1) == labels
    cand_correct = cand_prob.argmax(axis=1) == labels
    discord = neural_correct != cand_correct
    oof = np.zeros(len(labels), dtype=np.float32)

    if int(discord.sum()) < 50:
        return oof, {"valid": False, "discord_count": int(discord.sum()), "valid_folds": 0}

    skf = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    valid_folds = 0
    pos_seen = 0
    neg_seen = 0

    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X, labels), 1):
        tr_discord = tr_idx[discord[tr_idx]]
        if len(tr_discord) < 50:
            continue

        y = cand_correct[tr_discord].astype(np.int64)
        if len(np.unique(y)) < 2:
            continue

        sw = np.where(y == 1, float(pos_weight), float(neg_weight)).astype(np.float32)
        pos_seen += int((y == 1).sum())
        neg_seen += int((y == 0).sum())

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr_discord])
        Xva = scaler.transform(X[va_idx])

        clf = LogisticRegression(
            C=float(C),
            class_weight=None,          # do not force high switch rate
            max_iter=int(max_iter),
            solver="lbfgs",
        )
        clf.fit(Xtr, y, sample_weight=sw)
        oof[va_idx] = clf.predict_proba(Xva)[:, 1].astype(np.float32)
        valid_folds += 1

    return oof, {
        "valid": valid_folds > 0,
        "discord_count": int(discord.sum()),
        "valid_folds": int(valid_folds),
        "pos_seen": int(pos_seen),
        "neg_seen": int(neg_seen),
    }


def fit_final_selector(
    X: np.ndarray,
    neural_prob: np.ndarray,
    cand_prob: np.ndarray,
    labels: np.ndarray,
    C: float,
    pos_weight: float,
    neg_weight: float,
    max_iter: int,
) -> Tuple[Optional[StandardScaler], Optional[LogisticRegression], Dict[str, Any]]:
    labels = labels.astype(np.int64)
    neural_correct = neural_prob.argmax(axis=1) == labels
    cand_correct = cand_prob.argmax(axis=1) == labels
    discord = neural_correct != cand_correct
    idx = np.where(discord)[0]

    if len(idx) < 50:
        return None, None, {"valid": False, "discord_count": int(len(idx))}

    y = cand_correct[idx].astype(np.int64)
    if len(np.unique(y)) < 2:
        return None, None, {"valid": False, "discord_count": int(len(idx))}

    sw = np.where(y == 1, float(pos_weight), float(neg_weight)).astype(np.float32)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[idx])
    clf = LogisticRegression(
        C=float(C),
        class_weight=None,
        max_iter=int(max_iter),
        solver="lbfgs",
    )
    clf.fit(Xtr, y, sample_weight=sw)
    return scaler, clf, {
        "valid": True,
        "discord_count": int(len(idx)),
        "pos_count": int((y == 1).sum()),
        "neg_count": int((y == 0).sum()),
    }


# ----------------------------- search -----------------------------


def candidate_cfgs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    cfgs = []
    for low_bin_count in args.low_bin_counts:
        for low_thr in args.low_thresholds:
            for high_max_thr in args.high_max_thresholds:
                for low_gap_thr in args.low_gap_thresholds:
                    for neural_conf_thr in args.neural_conf_thresholds:
                        for neural_margin_thr in args.neural_margin_thresholds:
                            for alpha_max in args.alpha_max_values:
                                cfgs.append({
                                    "low_bin_count": int(low_bin_count),
                                    "low_thr": float(low_thr),
                                    "high_max_thr": float(high_max_thr),
                                    "low_gap_thr": float(low_gap_thr),
                                    "neural_conf_thr": float(neural_conf_thr),
                                    "neural_margin_thr": float(neural_margin_thr),
                                    "alpha_max": float(alpha_max),
                                })
    return cfgs


def record_from_eval(
    phase: str,
    cfg: Dict[str, Any],
    selector_cfg: Optional[Dict[str, Any]],
    m: Dict[str, Any],
    diag: Dict[str, float],
    score: float,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rec = {
        "phase": phase,
        "score": float(score),
        "overall_acc": float(m["overall_acc"]),
        "transition_acc": float(m["transition_acc"]),
        "edge_low_acc": float(m["edge_low_acc"]),
        "negative_acc": float(m["negative_acc"]),
        "high_acc": float(m["high_acc"]),
        "gate_rate": float(diag["gate_rate"]),
        "use_rate": float(diag["use_rate"]),
        "effective_use_rate": float(diag["effective_use_rate"]),
        "changed_rate": float(diag["changed_rate"]),
        "changed_ultra_rate": float(diag["changed_ultra_rate"]),
        "changed_nonultra_rate": float(diag["changed_nonultra_rate"]),
        "changed_high_rate": float(diag["changed_high_rate"]),
        "effective_ultra_rate": float(diag["effective_ultra_rate"]),
        "effective_nonultra_rate": float(diag["effective_nonultra_rate"]),
        "effective_high_rate": float(diag["effective_high_rate"]),
        "mean_alpha_all": float(diag["mean_alpha_all"]),
        "mean_alpha_gate": float(diag["mean_alpha_gate"]),
        "cfg": dict(cfg),
        "selector_cfg": dict(selector_cfg) if selector_cfg is not None else None,
    }
    if extra:
        rec.update(extra)
    return rec


def search_deterministic_candidates(
    neural_val: np.ndarray,
    gamc_val: np.ndarray,
    router_val: np.ndarray,
    labels_val: np.ndarray,
    snrs_val: np.ndarray,
    base_m: Dict[str, Any],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    cfgs = candidate_cfgs(args)
    print("\n" + "=" * 120)
    print(f"[*] Stage 1: deterministic residual candidate search, configs={len(cfgs)}")
    print("=" * 120)
    records = []

    for i, cfg in enumerate(cfgs, 1):
        cand, gate, alpha, _aux = make_residual_candidate(neural_val, gamc_val, router_val, cfg, args)
        m = metrics_from_probs(cand, labels_val, snrs_val)
        use_cand = gate & (alpha > 1e-8)
        diag = switch_diagnostics(neural_val, cand, gate, use_cand, alpha, snrs_val)
        sc = selection_score(m, diag, base_m, args)
        rec = record_from_eval("deterministic", cfg, None, m, diag, sc)
        records.append(rec)

        if i % 500 == 0 or i == len(cfgs):
            best_so_far = max(records, key=lambda r: r["score"])
            print(
                f"    progress {i:5d}/{len(cfgs)} | "
                f"best Score={best_so_far['score']:.3f}, "
                f"Overall={best_so_far['overall_acc']:.3f}%, "
                f"High={best_so_far['high_acc']:.3f}%, "
                f"ChangedHigh={best_so_far['changed_high_rate']:.2f}%"
            )

    records.sort(key=lambda r: r["score"], reverse=True)
    print("\nStage 1 top deterministic candidates:")
    for k, r in enumerate(records[:min(12, len(records))], 1):
        print(
            f"{k:02d}. Score={r['score']:.3f} | Overall={r['overall_acc']:.3f}% | "
            f"Neg={r['negative_acc']:.3f}% | Edge={r['edge_low_acc']:.3f}% | High={r['high_acc']:.3f}% | "
            f"ChangedHigh={r['changed_high_rate']:.2f}% | Gate={r['gate_rate']:.2f}% | cfg={r['cfg']}"
        )
    return records


def search_selector_candidates(
    neural_val: np.ndarray,
    gamc_val: np.ndarray,
    router_val: np.ndarray,
    labels_val: np.ndarray,
    snrs_val: np.ndarray,
    base_m: Dict[str, Any],
    deterministic_records: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    if args.skip_selector:
        print("[*] Stage 2 skipped by --skip_selector.")
        return []

    top = deterministic_records[:max(1, int(args.top_k_candidates))]
    total = len(top) * len(args.selector_Cs) * len(args.selector_neg_weights)

    print("\n" + "=" * 120)
    print(f"[*] Stage 2: OOF conservative selector search, candidate_configs={len(top)}, selector_fits={total}")
    print("=" * 120)

    selector_records = []
    fit_count = 0

    for ridx, base_rec in enumerate(top, 1):
        cfg = base_rec["cfg"]
        cand, gate, alpha, _aux = make_residual_candidate(neural_val, gamc_val, router_val, cfg, args)
        X = build_selector_features(neural_val, gamc_val, cand, router_val, gate, alpha)

        for C in args.selector_Cs:
            for neg_weight in args.selector_neg_weights:
                fit_count += 1
                oof_prob, info = fit_selector_oof(
                    X=X,
                    neural_prob=neural_val,
                    cand_prob=cand,
                    labels=labels_val,
                    C=float(C),
                    folds=int(args.folds),
                    seed=int(args.random_state),
                    pos_weight=float(args.selector_pos_weight),
                    neg_weight=float(neg_weight),
                    max_iter=int(args.selector_max_iter),
                )

                if fit_count % 10 == 0 or fit_count == total:
                    print(f"    selector progress {fit_count:4d}/{total}")

                if not info.get("valid", False):
                    continue

                for thr in args.selector_thresholds:
                    fused, use_cand = apply_selector(neural_val, cand, oof_prob, thr, base_gate=gate)
                    m = metrics_from_probs(fused, labels_val, snrs_val)
                    diag = switch_diagnostics(neural_val, fused, gate, use_cand, alpha, snrs_val)
                    sc = selection_score(m, diag, base_m, args)
                    selector_cfg = {
                        "selector_C": float(C),
                        "selector_neg_weight": float(neg_weight),
                        "selector_pos_weight": float(args.selector_pos_weight),
                        "selector_thr": float(thr),
                    }
                    rec = record_from_eval(
                        "selector_oof", cfg, selector_cfg, m, diag, sc,
                        extra={
                            "discord_count": int(info.get("discord_count", 0)),
                            "valid_folds": int(info.get("valid_folds", 0)),
                            "pos_seen": int(info.get("pos_seen", 0)),
                            "neg_seen": int(info.get("neg_seen", 0)),
                        },
                    )
                    selector_records.append(rec)

    selector_records.sort(key=lambda r: r["score"], reverse=True)
    print("\nStage 2 top OOF selector candidates:")
    for k, r in enumerate(selector_records[:min(20, len(selector_records))], 1):
        print(
            f"{k:02d}. Score={r['score']:.3f} | Overall={r['overall_acc']:.3f}% | "
            f"Neg={r['negative_acc']:.3f}% | Edge={r['edge_low_acc']:.3f}% | High={r['high_acc']:.3f}% | "
            f"ChangedHigh={r['changed_high_rate']:.2f}% | Use={r['use_rate']:.2f}% | "
            f"cfg={r['cfg']} | selector={r['selector_cfg']}"
        )
    return selector_records


# ----------------------------- final application -----------------------------


def apply_best_to_test(
    neural_val: np.ndarray,
    gamc_val: np.ndarray,
    router_val: np.ndarray,
    labels_val: np.ndarray,
    neural_test: np.ndarray,
    gamc_test: np.ndarray,
    router_test: np.ndarray,
    best: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    cfg = best["cfg"]
    selector_cfg = best.get("selector_cfg", None)

    cand_val, gate_val, alpha_val, _aux_val = make_residual_candidate(neural_val, gamc_val, router_val, cfg, args)
    cand_test, gate_test, alpha_test, _aux_test = make_residual_candidate(neural_test, gamc_test, router_test, cfg, args)

    if selector_cfg is None:
        switch_prob = gate_test.astype(np.float32)
        final_prob = cand_test
        use_cand = gate_test & (alpha_test > 1e-8)
        final_info = {"selector_valid": False, "mode": "deterministic"}
    else:
        X_val = build_selector_features(neural_val, gamc_val, cand_val, router_val, gate_val, alpha_val)
        X_test = build_selector_features(neural_test, gamc_test, cand_test, router_test, gate_test, alpha_test)
        scaler, clf, info = fit_final_selector(
            X=X_val,
            neural_prob=neural_val,
            cand_prob=cand_val,
            labels=labels_val,
            C=float(selector_cfg["selector_C"]),
            pos_weight=float(selector_cfg["selector_pos_weight"]),
            neg_weight=float(selector_cfg["selector_neg_weight"]),
            max_iter=int(args.selector_max_iter),
        )
        if not info.get("valid", False):
            print("[!] Final selector invalid. Falling back to deterministic residual candidate.")
            switch_prob = gate_test.astype(np.float32)
            final_prob = cand_test
            use_cand = gate_test & (alpha_test > 1e-8)
            final_info = {"selector_valid": False, "mode": "fallback_deterministic", **info}
        else:
            switch_prob = clf.predict_proba(scaler.transform(X_test))[:, 1].astype(np.float32)
            final_prob, use_cand = apply_selector(
                neural=neural_test,
                cand=cand_test,
                switch_prob=switch_prob,
                threshold=float(selector_cfg["selector_thr"]),
                base_gate=gate_test,
            )
            final_info = {"selector_valid": True, "mode": "selector", **info}

    return {
        "final_prob": normalize_probs(final_prob),
        "cand_test": normalize_probs(cand_test),
        "gate_test": gate_test.astype(bool),
        "alpha_test": alpha_test.astype(np.float32),
        "switch_prob": switch_prob.astype(np.float32),
        "use_cand": use_cand.astype(bool),
        "final_info": final_info,
    }


# ----------------------------- saving reports -----------------------------


def flatten_record_for_csv(rec: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in rec.items() if k not in ["cfg", "selector_cfg"]}
    cfg = rec.get("cfg", {}) or {}
    sel = rec.get("selector_cfg", {}) or {}
    for k, v in cfg.items():
        out[f"cfg_{k}"] = v
    for k, v in sel.items():
        out[f"selector_{k}"] = v
    out["cfg_json"] = json.dumps(cfg, ensure_ascii=False)
    out["selector_cfg_json"] = json.dumps(sel, ensure_ascii=False)
    return out


def save_records_csv(records: List[Dict[str, Any]], path: str, top_n: int) -> None:
    if not records:
        return
    rows = [flatten_record_for_csv(r) for r in records[:max(1, int(top_n))]]
    keys = sorted(set(k for row in rows for k in row.keys()))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[*] Search records saved: {path}")


def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[*] JSON saved: {path}")


# ----------------------------- main -----------------------------


def main() -> None:
    args = parse_args()
    results_dir = relpath("results")
    os.makedirs(results_dir, exist_ok=True)

    suffix = args.output_suffix or f"greedy_soup_gamc_protected_residual_fusion_split{args.split_seed}"

    print("🚀 Greedy Soup + GAMC protected residual blind fusion")
    print(f"Project root:      {project_root()}")
    print(f"Soup prob cache:   {args.soup_prob_cache}")
    print(f"GAMC cache:        {args.gamc_cache}")
    print(f"Output suffix:     {suffix}")
    print("\nAcademic protocol:")
    print("  - No new train/val/test split is created; this script preserves the existing caches.")
    print("  - True SNR is never passed into gate/selector features.")
    print("  - Validation labels/SNRs are used for calibration, hyperparameter selection, and diagnostics only.")
    print("  - Test labels/SNRs are used only once after selection for final reporting and plots.")

    soup = load_soup_cache(args.soup_prob_cache)
    gamc = load_gamc_cache(args.gamc_cache)
    assert_alignment(soup, gamc)

    labels_val = soup["labels_val"]
    snrs_val = soup["snrs_val"]
    labels_test = soup["labels_test"]
    snrs_test = soup["snrs_test"]
    mod_classes = soup["mod_classes"]

    neural_val_raw = soup["val_prob"]
    neural_test_raw = soup["test_prob"]
    gamc_val_raw = gamc["val_prob"]
    gamc_test_raw = gamc["test_prob"]
    router_val = gamc["val_router"]
    router_test = gamc["test_router"]

    print("\n" + "=" * 120)
    print("Validation baselines before calibration")
    print("=" * 120)
    neural_val_m_raw = metrics_from_probs(neural_val_raw, labels_val, snrs_val)
    gamc_val_m_raw = metrics_from_probs(gamc_val_raw, labels_val, snrs_val)
    print_metrics_line("Neural soup Val raw", neural_val_m_raw)
    print_metrics_line("GAMC Val raw", gamc_val_m_raw)

    if args.disable_temperature_scaling:
        tn, tg = 1.0, 1.0
        neural_val, neural_test = neural_val_raw, neural_test_raw
        gamc_val, gamc_test = gamc_val_raw, gamc_test_raw
        print("[*] Temperature scaling disabled.")
    else:
        tn, _ = fit_temperature(neural_val_raw, labels_val, args.temperature_grid, "neural")
        tg, _ = fit_temperature(gamc_val_raw, labels_val, args.temperature_grid, "GAMC")
        neural_val = temperature_scale_probs(neural_val_raw, tn)
        neural_test = temperature_scale_probs(neural_test_raw, tn)
        gamc_val = temperature_scale_probs(gamc_val_raw, tg)
        gamc_test = temperature_scale_probs(gamc_test_raw, tg)

    print("\n" + "=" * 120)
    print("Validation baselines after calibration")
    print("=" * 120)
    neural_val_m = metrics_from_probs(neural_val, labels_val, snrs_val)
    gamc_val_m = metrics_from_probs(gamc_val, labels_val, snrs_val)
    print_metrics_line("Neural soup Val calibrated", neural_val_m)
    print_metrics_line("GAMC Val calibrated", gamc_val_m)

    deterministic_records = search_deterministic_candidates(
        neural_val=neural_val,
        gamc_val=gamc_val,
        router_val=router_val,
        labels_val=labels_val,
        snrs_val=snrs_val,
        base_m=neural_val_m,
        args=args,
    )

    selector_records = search_selector_candidates(
        neural_val=neural_val,
        gamc_val=gamc_val,
        router_val=router_val,
        labels_val=labels_val,
        snrs_val=snrs_val,
        base_m=neural_val_m,
        deterministic_records=deterministic_records,
        args=args,
    )

    all_records = sorted(deterministic_records + selector_records, key=lambda r: r["score"], reverse=True)
    best = all_records[0]

    print("\n" + "=" * 140)
    print("✅ Selected validation config")
    print("=" * 140)
    print(json.dumps({
        "phase": best["phase"],
        "score": best["score"],
        "metrics": {
            "overall_acc": best["overall_acc"],
            "transition_acc": best["transition_acc"],
            "edge_low_acc": best["edge_low_acc"],
            "negative_acc": best["negative_acc"],
            "high_acc": best["high_acc"],
        },
        "switch_diagnostics": {
            "gate_rate": best["gate_rate"],
            "use_rate": best["use_rate"],
            "effective_use_rate": best["effective_use_rate"],
            "changed_high_rate": best["changed_high_rate"],
            "changed_nonultra_rate": best["changed_nonultra_rate"],
        },
        "cfg": best["cfg"],
        "selector_cfg": best.get("selector_cfg", None),
        "temperature": {"neural_T": tn, "gamc_T": tg},
    }, ensure_ascii=False, indent=2))
    print("=" * 140)

    # Save search artifacts before final test report.
    search_csv = os.path.join(results_dir, f"{suffix}_search_top.csv")
    search_json = os.path.join(results_dir, f"{suffix}_selected_config.json")
    save_records_csv(all_records, search_csv, args.save_top_records)
    save_json({
        "selected": best,
        "temperature": {"neural_T": float(tn), "gamc_T": float(tg)},
        "args": vars(args),
    }, search_json)

    # Final test evaluation: no tuning below this line.
    out = apply_best_to_test(
        neural_val=neural_val,
        gamc_val=gamc_val,
        router_val=router_val,
        labels_val=labels_val,
        neural_test=neural_test,
        gamc_test=gamc_test,
        router_test=router_test,
        best=best,
        args=args,
    )

    final_prob = out["final_prob"]
    final_m = metrics_from_probs(final_prob, labels_test, snrs_test)
    neural_test_m = metrics_from_probs(neural_test, labels_test, snrs_test)
    gamc_test_m = metrics_from_probs(gamc_test, labels_test, snrs_test)
    final_diag = switch_diagnostics(
        neural=neural_test,
        final_prob=final_prob,
        gate=out["gate_test"],
        use_cand=out["use_cand"],
        alpha=out["alpha_test"],
        snrs=snrs_test,
    )

    print("\n" + "=" * 150)
    print("🏆 Final test report: Greedy Soup + GAMC protected residual blind fusion")
    print("=" * 150)
    print_metrics_line("Neural soup Test calibrated", neural_test_m)
    print_metrics_line("GAMC Test calibrated", gamc_test_m)
    print_metrics_line("Protected residual fusion Test", final_m)
    print("-" * 150)
    print(f"Delta vs neural overall:     {final_m['overall_acc'] - neural_test_m['overall_acc']:+.4f} pp")
    print(f"Delta vs neural negative:    {final_m['negative_acc'] - neural_test_m['negative_acc']:+.4f} pp")
    print(f"Delta vs neural edge:        {final_m['edge_low_acc'] - neural_test_m['edge_low_acc']:+.4f} pp")
    print(f"Delta vs neural transition:  {final_m['transition_acc'] - neural_test_m['transition_acc']:+.4f} pp")
    print(f"Delta vs neural high:        {final_m['high_acc'] - neural_test_m['high_acc']:+.4f} pp")
    print("-" * 150)
    print(f"Gate rate total:             {final_diag['gate_rate']:.2f}%")
    print(f"Use-candidate rate total:    {final_diag['use_rate']:.2f}%")
    print(f"Effective use rate total:    {final_diag['effective_use_rate']:.2f}%")
    print(f"Effective use ultra-low:     {final_diag['effective_ultra_rate']:.2f}%")
    print(f"Effective use non-ultra:     {final_diag['effective_nonultra_rate']:.2f}%")
    print(f"Effective use high-SNR:      {final_diag['effective_high_rate']:.2f}%")
    print(f"Changed prediction total:    {final_diag['changed_rate']:.2f}%")
    print(f"Changed prediction high-SNR: {final_diag['changed_high_rate']:.2f}%")
    print(f"Mean alpha on gated samples: {final_diag['mean_alpha_gate']:.4f}")
    print(f"Final selector info:         {out['final_info']}")
    print("=" * 150)

    print_snr_table(final_m["by_snr"])

    pred_path = os.path.join(results_dir, f"{suffix}_predictions.npz")
    selected_payload = {
        "selected": best,
        "temperature": {"neural_T": float(tn), "gamc_T": float(tg)},
        "final_info": out["final_info"],
        "final_test_metrics": {
            "overall_acc": final_m["overall_acc"],
            "transition_acc": final_m["transition_acc"],
            "edge_low_acc": final_m["edge_low_acc"],
            "negative_acc": final_m["negative_acc"],
            "high_acc": final_m["high_acc"],
        },
        "final_switch_diagnostics": final_diag,
    }
    np.savez_compressed(
        pred_path,
        labels=labels_test.astype(np.int64),
        snrs=snrs_test.astype(np.int32),
        pred=final_m["pred"].astype(np.int64),
        final_prob=final_prob.astype(np.float32),
        candidate_prob=out["cand_test"].astype(np.float32),
        switch_prob=out["switch_prob"].astype(np.float32),
        use_cand=out["use_cand"].astype(np.int8),
        gate=out["gate_test"].astype(np.int8),
        alpha=out["alpha_test"].astype(np.float32),
        greedy_soup_prob_raw=neural_test_raw.astype(np.float32),
        gamc_prob_raw=gamc_test_raw.astype(np.float32),
        greedy_soup_prob_calibrated=neural_test.astype(np.float32),
        gamc_prob_calibrated=gamc_test.astype(np.float32),
        router_prob=router_test.astype(np.float32),
        selected_config=np.array([json.dumps(selected_payload, ensure_ascii=False)]),
        mod_classes=np.array(mod_classes),
    )
    print(f"[*] Predictions saved: {pred_path}")

    curve_path = os.path.join(results_dir, f"accuracy_vs_snr_{suffix}.png")
    plot_curve(final_m["by_snr"], curve_path, "Accuracy vs SNR: Protected Residual Blind Fusion")
    print(f"📸 SNR curve saved: {curve_path}")

    for snr_value in args.cm_snrs:
        cm_path = os.path.join(results_dir, f"confusion_matrix_{snr_value}dB_{suffix}.png")
        cm_acc = plot_cm_at_snr(
            labels=labels_test,
            pred=final_m["pred"],
            snrs=snrs_test,
            mod_classes=mod_classes,
            target_snr=int(snr_value),
            save_path=cm_path,
            title_prefix="Protected Residual Blind Fusion Confusion Matrix",
        )
        print(f"📸 {snr_value} dB confusion matrix saved: {cm_path} (Acc={cm_acc:.2f}%)")


if __name__ == "__main__":
    main()
