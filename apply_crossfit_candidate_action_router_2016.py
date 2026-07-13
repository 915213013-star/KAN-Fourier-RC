import argparse
import csv
import json
import os
from collections import Counter

import numpy as np
from sklearn.model_selection import StratifiedKFold

import evaluate_greedy_soup_gamc_protected_residual_fusion as base_metrics


MIDLOW_SNRS = np.array([-14, -12], dtype=np.int32)


def relpath(*parts):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Cross-fit validation action router over fixed CPU/GAMC candidate caches. "
            "For each validation fold, transition actions are selected on the remaining "
            "validation folds and evaluated on the held-out fold before final full-val selection."
        )
    )
    p.add_argument("--base_predictions", type=str, required=True)
    p.add_argument("--candidate_predictions", type=str, nargs="+", required=True)
    p.add_argument("--output_suffix", type=str, required=True)
    p.add_argument("--skip_bad_candidates", action="store_true")
    p.add_argument("--forbid_pairs", type=str, nargs="*", default=[])

    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--min_counts", type=int, nargs="+", default=[10, 15, 20, 30])
    p.add_argument("--min_nets", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--min_precisions", type=float, nargs="+", default=[0.60, 0.65, 0.70])
    p.add_argument("--max_change_rates", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.35])
    p.add_argument("--max_actions", type=int, nargs="+", default=[1, 2, 3, 5])
    p.add_argument("--min_fold_count", type=int, default=1)
    p.add_argument("--min_positive_folds", type=int, nargs="+", default=[2, 3])
    p.add_argument("--max_negative_folds", type=int, nargs="+", default=[0])

    p.add_argument("--min_cv_overall_gain", type=float, default=0.0)
    p.add_argument("--min_cv_overall_acc", type=float, default=0.0)
    p.add_argument("--min_cv_transition_acc", type=float, default=0.0)
    p.add_argument("--min_cv_high_acc", type=float, default=0.0)
    p.add_argument("--min_cv_midlow_acc", type=float, default=0.0)
    p.add_argument("--allow_no_change", action="store_true")
    p.add_argument(
        "--final_report_only",
        action="store_true",
        help="Do not score intermediate base/candidate test predictions; report only the locked final test result.",
    )

    p.add_argument("--score_transition_gain_weight", type=float, default=0.030)
    p.add_argument("--score_midlow_gain_weight", type=float, default=0.015)
    p.add_argument("--score_edge_gain_weight", type=float, default=0.015)
    p.add_argument("--score_high_drop_penalty", type=float, default=5.0)
    p.add_argument("--high_tolerance", type=float, default=0.02)
    p.add_argument("--score_change_penalty", type=float, default=0.030)
    p.add_argument("--score_action_penalty", type=float, default=0.010)
    p.add_argument("--save_top_records", type=int, default=200)
    return p.parse_args()


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def norm(prob):
    p = np.asarray(prob, dtype=np.float64)
    p = np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)
    p = np.maximum(p, 1e-12)
    return (p / p.sum(axis=1, keepdims=True).clip(min=1e-12)).astype(np.float32)


def prediction_parts(z, name):
    if "final_val_prob" in z and "final_prob" in z:
        val_key, test_key = "final_val_prob", "final_prob"
    elif "val_prob" in z and "test_prob" in z:
        val_key, test_key = "val_prob", "test_prob"
    else:
        raise KeyError(f"{name} must contain final_val_prob/final_prob or val_prob/test_prob")

    labels_val = np.asarray(z["labels_val"], dtype=np.int64)
    snrs_val = np.asarray(z["snrs_val"], dtype=np.int32)
    labels = np.asarray(z["labels"] if "labels" in z else z["labels_test"], dtype=np.int64)
    snrs = np.asarray(z["snrs"] if "snrs" in z else z["snrs_test"], dtype=np.int32)
    return norm(z[val_key]), norm(z[test_key]), labels_val, snrs_val, labels, snrs


def assert_aligned(base, other, name):
    key_pairs = [
        ("labels_val", "labels_val"),
        ("snrs_val", "snrs_val"),
        ("labels", "labels"),
        ("snrs", "snrs"),
        ("labels", "labels_test"),
        ("snrs", "snrs_test"),
        ("labels_test", "labels"),
        ("snrs_test", "snrs"),
        ("labels_test", "labels_test"),
        ("snrs_test", "snrs_test"),
    ]
    for akey, bkey in key_pairs:
        if akey in base and bkey in other:
            if not np.array_equal(np.asarray(base[akey]), np.asarray(other[bkey])):
                raise RuntimeError(f"Alignment mismatch for {name}: {akey}")


def parse_pairs(items):
    out = set()
    for item in items or []:
        text = str(item).replace(",", ":").strip()
        if not text:
            continue
        a, b = text.split(":", 1)
        out.add((int(a), int(b)))
    return out


def band_acc_from_pred(pred, labels, snrs, band):
    mask = np.isin(np.asarray(snrs, dtype=np.int32), band)
    if not np.any(mask):
        return 0.0
    return float((np.asarray(pred)[mask] == np.asarray(labels)[mask]).mean() * 100.0)


def metrics_from_pred(pred, labels, snrs, n_classes):
    prob = np.zeros((len(pred), n_classes), dtype=np.float32)
    prob[np.arange(len(pred)), np.asarray(pred, dtype=np.int64)] = 1.0
    m = base_metrics.metrics_from_probs(prob, labels, snrs)
    m["midlow_acc"] = band_acc_from_pred(pred, labels, snrs, MIDLOW_SNRS)
    return m


def print_metrics(name, pred, labels, snrs, n_classes):
    m = metrics_from_pred(pred, labels, snrs, n_classes)
    print(
        f"{name:<36} Overall={m['overall_acc']:7.3f}% | Trans={m['transition_acc']:7.3f}% | "
        f"Edge={m['edge_low_acc']:7.3f}% | Neg={m['negative_acc']:7.3f}% | "
        f"Midlow={m['midlow_acc']:7.3f}% | High={m['high_acc']:7.3f}%"
    )
    return m


def make_folds(labels, snrs, n_splits, seed=2026):
    composite = np.asarray([f"{int(y)}_{int(s)}" for y, s in zip(labels, snrs)])
    skf = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
    return [va for _, va in skf.split(np.zeros(len(labels)), composite)]


def rescue_harm(base_pred, cand_pred, labels, mask):
    y = np.asarray(labels, dtype=np.int64)
    m = np.asarray(mask, dtype=bool)
    rescue = int(((base_pred != y) & (cand_pred == y) & m).sum())
    harm = int(((base_pred == y) & (cand_pred != y) & m).sum())
    neutral = int(m.sum()) - rescue - harm
    denom = rescue + harm
    precision = float(rescue / denom) if denom > 0 else 0.0
    return {
        "count": int(m.sum()),
        "rescue": rescue,
        "harm": harm,
        "neutral": neutral,
        "net_rescue": int(rescue - harm),
        "precision": precision,
    }


def fold_consistency(base_pred, cand_pred, labels, mask, folds, min_fold_count):
    pos = 0
    neg = 0
    fold_net_sum = 0
    for ids in folds:
        fm = np.zeros(len(labels), dtype=bool)
        fm[ids] = True
        row = rescue_harm(base_pred, cand_pred, labels, mask & fm)
        fold_net_sum += row["net_rescue"]
        if row["count"] >= int(min_fold_count):
            if row["net_rescue"] > 0:
                pos += 1
            elif row["net_rescue"] < 0:
                neg += 1
    return pos, neg, fold_net_sum


def build_actions_for_indices(
    base_pred,
    cand_preds,
    labels,
    select_idx,
    candidate_names,
    forbidden,
    min_count,
    min_net,
    min_precision,
    min_positive_folds,
    max_negative_folds,
    min_fold_count,
    consistency_folds,
):
    selected = np.zeros(len(labels), dtype=bool)
    selected[np.asarray(select_idx, dtype=np.int64)] = True
    actions = []
    for ci, cand_pred in enumerate(cand_preds):
        changed = selected & (cand_pred != base_pred)
        for from_to, _count in Counter(zip(base_pred[changed], cand_pred[changed])).items():
            a, b = int(from_to[0]), int(from_to[1])
            if (a, b) in forbidden:
                continue
            mask = changed & (base_pred == a) & (cand_pred == b)
            whole = rescue_harm(base_pred, cand_pred, labels, mask)
            if whole["count"] < int(min_count):
                continue
            if whole["net_rescue"] < int(min_net):
                continue
            if whole["precision"] < float(min_precision):
                continue
            pos, neg, fold_net_sum = fold_consistency(
                base_pred, cand_pred, labels, mask, consistency_folds, min_fold_count
            )
            if pos < int(min_positive_folds) or neg > int(max_negative_folds):
                continue
            action = {
                "candidate": candidate_names[ci],
                "candidate_index": int(ci),
                "from_class": int(a),
                "to_class": int(b),
                **whole,
                "positive_folds": int(pos),
                "negative_folds": int(neg),
                "fold_net_sum": int(fold_net_sum),
            }
            action["reliability_score"] = float(
                2.0 * action["net_rescue"]
                + 0.05 * action["count"]
                + 1.0 * action["positive_folds"]
                - 2.0 * action["negative_folds"]
                + 2.0 * (action["precision"] - 0.5)
            )
            actions.append(action)
    actions.sort(
        key=lambda x: (
            x["reliability_score"],
            x["net_rescue"],
            x["positive_folds"],
            x["precision"],
            x["count"],
        ),
        reverse=True,
    )
    return actions


def apply_actions(base_pred, cand_preds, actions, allowed_idx=None):
    out = np.asarray(base_pred, dtype=np.int64).copy()
    changed = np.zeros(len(out), dtype=bool)
    action_id = np.full(len(out), -1, dtype=np.int32)
    allowed = np.ones(len(out), dtype=bool)
    if allowed_idx is not None:
        allowed[:] = False
        allowed[np.asarray(allowed_idx, dtype=np.int64)] = True
    for idx, action in enumerate(actions):
        cand_pred = cand_preds[action["candidate_index"]]
        mask = (
            allowed
            & (~changed)
            & (base_pred == action["from_class"])
            & (cand_pred == action["to_class"])
        )
        out[mask] = cand_pred[mask]
        changed[mask] = True
        action_id[mask] = idx
    return out, changed, action_id


def truncate_to_budget(base_pred, cand_preds, actions, budget_idx, max_change_rate, max_actions):
    kept = []
    for action in actions[: int(max_actions)]:
        trial = kept + [action]
        _, changed, _ = apply_actions(base_pred, cand_preds, trial, budget_idx)
        if changed[budget_idx].mean() * 100.0 <= float(max_change_rate) + 1e-12:
            kept = trial
    return kept


def score_config(m, base_m, change_rate, num_actions, args):
    high_drop = max(0.0, base_m["high_acc"] - m["high_acc"] - float(args.high_tolerance))
    return (
        m["overall_acc"]
        + float(args.score_transition_gain_weight) * (m["transition_acc"] - base_m["transition_acc"])
        + float(args.score_midlow_gain_weight) * (m["midlow_acc"] - base_m["midlow_acc"])
        + float(args.score_edge_gain_weight) * (m["edge_low_acc"] - base_m["edge_low_acc"])
        - float(args.score_high_drop_penalty) * high_drop
        - float(args.score_change_penalty) * change_rate
        - float(args.score_action_penalty) * num_actions
    )


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def main():
    args = parse_args()
    forbidden = parse_pairs(args.forbid_pairs)

    print("=" * 128)
    print("Cross-fit CPU/GAMC candidate action router")
    print("=" * 128)
    print("Academic protocol:")
    print("  - Base and candidate prediction caches are fixed before this script.")
    print("  - Each validation fold is routed by actions selected without that fold.")
    print("  - Final actions are selected on the full validation split after cross-fit hyperparameter selection.")
    print("  - Test labels are reported once for the final selected configuration.")

    base_z = load_npz(args.base_predictions)
    base_val_prob, base_test_prob, yv, sv, yt, st = prediction_parts(base_z, "base")
    n_classes = base_test_prob.shape[1]
    base_val_pred = base_val_prob.argmax(1).astype(np.int64)
    base_test_pred = base_test_prob.argmax(1).astype(np.int64)
    class_names = [str(x) for x in base_z.get("mod_classes", np.arange(n_classes))]
    if forbidden:
        print("[*] Forbidden transitions:", [f"{class_names[a]}->{class_names[b]}" for a, b in sorted(forbidden)])

    cand_val_probs, cand_test_probs, cand_val_preds, cand_test_preds, cand_names = [], [], [], [], []
    for path in args.candidate_predictions:
        try:
            z = load_npz(path)
            assert_aligned(base_z, z, os.path.basename(path))
            val_prob, test_prob, *_ = prediction_parts(z, os.path.basename(path))
        except Exception as exc:
            if not args.skip_bad_candidates:
                raise
            print(f"[!] Skipping candidate {path}: {exc}")
            continue
        cand_names.append(os.path.basename(path))
        cand_val_probs.append(val_prob)
        cand_test_probs.append(test_prob)
        cand_val_preds.append(val_prob.argmax(1).astype(np.int64))
        cand_test_preds.append(test_prob.argmax(1).astype(np.int64))

    if not cand_names:
        raise RuntimeError("No usable candidate prediction caches.")

    base_val_m = print_metrics("Base Val", base_val_pred, yv, sv, n_classes)
    if args.final_report_only:
        base_test_m = None
        print("[*] Intermediate test diagnostics are deferred until the final configuration is locked.")
    else:
        base_test_m = print_metrics("Base Test", base_test_pred, yt, st, n_classes)
        for name, pred in zip(cand_names, cand_test_preds):
            m = metrics_from_pred(pred, yt, st, n_classes)
            changed = float((pred != base_test_pred).mean() * 100.0)
            print(f"{name[:34]:<36} Test={m['overall_acc']:7.3f}% | delta={m['overall_acc'] - base_test_m['overall_acc']:+7.3f} pp | change={changed:6.3f}%")

    folds = make_folds(yv, sv, args.cv_folds)
    all_idx = np.arange(len(yv))
    records = []
    best = None

    for min_count in args.min_counts:
        for min_net in args.min_nets:
            for min_precision in args.min_precisions:
                for min_pos_folds in args.min_positive_folds:
                    for max_neg_folds in args.max_negative_folds:
                        for max_change in args.max_change_rates:
                            for max_actions in args.max_actions:
                                cv_pred = base_val_pred.copy()
                                cv_changed = np.zeros(len(yv), dtype=bool)
                                fold_action_counts = []
                                for fold_idx in folds:
                                    train_idx = np.setdiff1d(all_idx, fold_idx, assume_unique=False)
                                    train_folds = [
                                        np.intersect1d(train_idx, f, assume_unique=False)
                                        for f in folds
                                    ]
                                    actions = build_actions_for_indices(
                                        base_val_pred,
                                        cand_val_preds,
                                        yv,
                                        train_idx,
                                        cand_names,
                                        forbidden,
                                        min_count,
                                        min_net,
                                        min_precision,
                                        min_pos_folds,
                                        max_neg_folds,
                                        args.min_fold_count,
                                        train_folds,
                                    )
                                    actions = truncate_to_budget(
                                        base_val_pred, cand_val_preds, actions, train_idx, max_change, max_actions
                                    )
                                    pred_fold, changed_fold, _ = apply_actions(
                                        base_val_pred, cand_val_preds, actions, fold_idx
                                    )
                                    cv_pred[fold_idx] = pred_fold[fold_idx]
                                    cv_changed[fold_idx] = changed_fold[fold_idx]
                                    fold_action_counts.append(len(actions))

                                m = metrics_from_pred(cv_pred, yv, sv, n_classes)
                                change_rate = float(cv_changed.mean() * 100.0)
                                gain = float(m["overall_acc"] - base_val_m["overall_acc"])
                                if gain < float(args.min_cv_overall_gain):
                                    continue
                                if args.min_cv_overall_acc and m["overall_acc"] < args.min_cv_overall_acc:
                                    continue
                                if args.min_cv_transition_acc and m["transition_acc"] < args.min_cv_transition_acc:
                                    continue
                                if args.min_cv_high_acc and m["high_acc"] < args.min_cv_high_acc:
                                    continue
                                if args.min_cv_midlow_acc and m["midlow_acc"] < args.min_cv_midlow_acc:
                                    continue
                                rec = {
                                    "score": float(score_config(m, base_val_m, change_rate, max(fold_action_counts), args)),
                                    "cv_overall_acc": float(m["overall_acc"]),
                                    "cv_gain": gain,
                                    "cv_transition_acc": float(m["transition_acc"]),
                                    "cv_edge_low_acc": float(m["edge_low_acc"]),
                                    "cv_negative_acc": float(m["negative_acc"]),
                                    "cv_midlow_acc": float(m["midlow_acc"]),
                                    "cv_high_acc": float(m["high_acc"]),
                                    "cv_change_rate": change_rate,
                                    "min_count": int(min_count),
                                    "min_net": int(min_net),
                                    "min_precision": float(min_precision),
                                    "min_positive_folds": int(min_pos_folds),
                                    "max_negative_folds": int(max_neg_folds),
                                    "max_change_rate": float(max_change),
                                    "max_actions": int(max_actions),
                                    "mean_fold_actions": float(np.mean(fold_action_counts)),
                                    "max_fold_actions": int(max(fold_action_counts) if fold_action_counts else 0),
                                }
                                records.append(rec)
                                if best is None or rec["score"] > best["score"]:
                                    best = rec

    if best is None:
        if not args.allow_no_change:
            raise RuntimeError("No cross-fit action-router config passed floors.")
        best = {
            "score": float(base_val_m["overall_acc"]),
            "cv_overall_acc": float(base_val_m["overall_acc"]),
            "cv_gain": 0.0,
            "cv_transition_acc": float(base_val_m["transition_acc"]),
            "cv_edge_low_acc": float(base_val_m["edge_low_acc"]),
            "cv_negative_acc": float(base_val_m["negative_acc"]),
            "cv_midlow_acc": float(base_val_m["midlow_acc"]),
            "cv_high_acc": float(base_val_m["high_acc"]),
            "cv_change_rate": 0.0,
            "min_count": 10,
            "min_net": 999,
            "min_precision": 1.0,
            "min_positive_folds": 999,
            "max_negative_folds": 0,
            "max_change_rate": 0.0,
            "max_actions": 0,
            "mean_fold_actions": 0.0,
            "max_fold_actions": 0,
        }

    records.sort(key=lambda r: r["score"], reverse=True)
    print("\nTop cross-fit validation configs")
    for i, rec in enumerate(records[:20], 1):
        print(
            f"{i:02d}. score={rec['score']:.3f} | cv={rec['cv_overall_acc']:.3f}% "
            f"gain={rec['cv_gain']:+.4f} | neg={rec['cv_negative_acc']:.3f}% "
            f"edge={rec['cv_edge_low_acc']:.3f}% midlow={rec['cv_midlow_acc']:.3f}% "
            f"trans={rec['cv_transition_acc']:.3f}% high={rec['cv_high_acc']:.3f}% "
            f"chg={rec['cv_change_rate']:.3f}% | cnt={rec['min_count']} net={rec['min_net']} "
            f"prec={rec['min_precision']} pos={rec['min_positive_folds']} bad={rec['max_negative_folds']} "
            f"maxchg={rec['max_change_rate']} maxact={rec['max_actions']}"
        )

    consistency_folds = folds
    final_actions = build_actions_for_indices(
        base_val_pred,
        cand_val_preds,
        yv,
        all_idx,
        cand_names,
        forbidden,
        best["min_count"],
        best["min_net"],
        best["min_precision"],
        best["min_positive_folds"],
        best["max_negative_folds"],
        args.min_fold_count,
        consistency_folds,
    )
    final_actions = truncate_to_budget(
        base_val_pred,
        cand_val_preds,
        final_actions,
        all_idx,
        best["max_change_rate"],
        best["max_actions"],
    )

    final_val_pred, changed_val, action_id_val = apply_actions(base_val_pred, cand_val_preds, final_actions)
    final_test_pred, changed_test, action_id_test = apply_actions(base_test_pred, cand_test_preds, final_actions)
    final_val_prob = base_val_prob.copy()
    final_test_prob = base_test_prob.copy()
    for idx, action in enumerate(final_actions):
        ci = action["candidate_index"]
        final_val_prob[action_id_val == idx] = cand_val_probs[ci][action_id_val == idx]
        final_test_prob[action_id_test == idx] = cand_test_probs[ci][action_id_test == idx]

    print("\nSelected full-validation actions")
    if not final_actions:
        print("  (none)")
    for i, action in enumerate(final_actions, 1):
        print(
            f"{i:02d}. {action['candidate'][:44]:<44} "
            f"{class_names[action['from_class']]}->{class_names[action['to_class']]} "
            f"val_n={action['count']} rescue={action['rescue']} harm={action['harm']} "
            f"net={action['net_rescue']} precision={action['precision']:.3f} "
            f"pos={action['positive_folds']} neg={action['negative_folds']} "
            f"score={action['reliability_score']:.3f}"
        )

    print("\n" + "=" * 128)
    print("Final test report")
    print("=" * 128)
    final_val_m = print_metrics("Full-selected Router Val", final_val_pred, yv, sv, n_classes)
    final_test_m = print_metrics("Full-selected Router Test", final_test_pred, yt, st, n_classes)
    if base_test_m is None:
        base_test_m = metrics_from_pred(base_test_pred, yt, st, n_classes)
    print("-" * 128)
    print(f"Cross-fit Val gain:       {best['cv_gain']:+.4f} pp")
    print(f"Delta vs base overall:    {final_test_m['overall_acc'] - base_test_m['overall_acc']:+.4f} pp")
    print(f"Delta vs base negative:   {final_test_m['negative_acc'] - base_test_m['negative_acc']:+.4f} pp")
    print(f"Delta vs base edge:       {final_test_m['edge_low_acc'] - base_test_m['edge_low_acc']:+.4f} pp")
    print(f"Delta vs base midlow:     {final_test_m['midlow_acc'] - base_test_m['midlow_acc']:+.4f} pp")
    print(f"Delta vs base transition: {final_test_m['transition_acc'] - base_test_m['transition_acc']:+.4f} pp")
    print(f"Delta vs base high:       {final_test_m['high_acc'] - base_test_m['high_acc']:+.4f} pp")
    print(f"Diagnostics: val_change={changed_val.mean() * 100.0:.3f}% test_change={changed_test.mean() * 100.0:.3f}%")

    results_dir = relpath("results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"{args.output_suffix}_search_top.csv")
    json_path = os.path.join(results_dir, f"{args.output_suffix}_selected_config.json")
    pred_path = os.path.join(results_dir, f"{args.output_suffix}_predictions.npz")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if records:
            keys = sorted(records[0].keys())
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for rec in records[: args.save_top_records]:
                w.writerow(rec)

    selected = {
        "best": best,
        "actions": final_actions,
        "base_predictions": args.base_predictions,
        "candidate_predictions": args.candidate_predictions,
        "forbid_pairs": sorted([list(x) for x in forbidden]),
        "protocol": "cross-fit validation-selected CPU/GAMC candidate action router",
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(selected), f, ensure_ascii=False, indent=2)

    np.savez_compressed(
        pred_path,
        labels=yt,
        snrs=st,
        pred=final_test_pred.astype(np.int64),
        final_prob=final_test_prob.astype(np.float32),
        base_prob=base_test_prob.astype(np.float32),
        labels_val=yv,
        snrs_val=sv,
        final_val_prob=final_val_prob.astype(np.float32),
        base_val_prob=base_val_prob.astype(np.float32),
        changed=changed_test,
        changed_val=changed_val,
        action_id=action_id_test,
        action_id_val=action_id_val,
        candidate_names=np.array(cand_names, dtype=object),
        mod_classes=np.array(class_names, dtype=object),
    )
    print(f"[*] CSV saved: {csv_path}")
    print(f"[*] Selected config saved: {json_path}")
    print(f"[*] Predictions saved: {pred_path}")


if __name__ == "__main__":
    main()
