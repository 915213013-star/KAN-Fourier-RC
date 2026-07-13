import argparse
import csv
import json
import os

import numpy as np
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from joblib import Parallel, delayed

import evaluate_greedy_soup_gamc_protected_residual_fusion as base
import evaluate_fourier_oof_gamc_cvtrn_residual_meta_fusion as orig
import train_cv_trn_aux_2016 as common
from model_cache_utils import fit_or_load_estimator


def relpath(*parts):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)


DEFAULT_PAIRS = ["1,10", "7,8", "3,6", "1,2", "4,5"]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Train train-split OOF pairwise confusion specialists. "
            "Each specialist is a binary model for one hard modulation pair; validation is diagnostic only."
        )
    )
    p.add_argument("--split_seed", type=int, default=1)
    p.add_argument("--random_state", type=int, default=2026)
    p.add_argument("--main_oof_cache", type=str, default=relpath("results", "fourier_main_oof_merge_mseed78_79_f3e220_bs128_split1_trainvaltest_probs_for_meta.npz"))
    p.add_argument("--soup_prob_cache", type=str, default=relpath("results", "greedy_soup_identity_valtest_probs_for_gamc_fusion.npz"))
    p.add_argument("--cvtrn_oof_cache", type=str, default=relpath("results", "cv_trn_aux_v2_oof_mseed41_f3e240_split1_trainvaltest_probs_for_meta.npz"))
    p.add_argument("--gamc_oof_cache", type=str, default=relpath("results", "gamc_oof_tree_split1_trainvaltest_probs_for_meta.npz"))
    p.add_argument("--hcs_oof_cache", type=str, default=relpath("results", "hcs_precision_analog_aux_split1_trainvaltest_probs_for_meta.npz"))
    p.add_argument("--hcs_feature_cache", type=str, default=relpath("feature_cache", "hcs_lite_features_v1.npz"))
    p.add_argument("--gamc_feature_cache", type=str, default=relpath("feature_cache", "gamc_lite_features_v3_graph_xgb.npz"))
    p.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    p.add_argument("--selection_mode", choices=["global", "per_pair_greedy"], default="global")
    p.add_argument("--per_pair_min_score_gain", type=float, default=0.002)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--xgb_estimators", type=int, default=360)
    p.add_argument("--xgb_depth", type=int, default=3)
    p.add_argument("--xgb_lr", type=float, default=0.035)
    p.add_argument("--xgb_jobs", type=int, default=4)
    p.add_argument("--xgb_device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--model_cache_dir", type=str, default="")
    p.add_argument("--reuse_models", action="store_true")
    p.add_argument(
        "--stat_feature_sources",
        nargs="+",
        choices=["hcs", "gamc"],
        default=["hcs", "gamc"],
        help="Statistical feature caches to concatenate. Probability features are always included.",
    )
    p.add_argument(
        "--search_jobs",
        type=int,
        default=1,
        help="Threaded parallelism for threshold search. Training still uses --xgb_jobs.",
    )
    p.add_argument("--blend_alphas", type=float, nargs="+", default=[0.35, 0.50, 0.65, 0.80])
    p.add_argument("--pair_conf_thresholds", type=float, nargs="+", default=[0.54, 0.58, 0.62, 0.66, 0.70, 0.75])
    p.add_argument("--pair_margin_thresholds", type=float, nargs="+", default=[0.02, 0.05, 0.08, 0.12, 0.16, 0.22])
    p.add_argument("--main_pair_mass_thresholds", type=float, nargs="+", default=[0.35, 0.45, 0.55, 0.65, 0.75])
    p.add_argument("--main_pair_margin_maxes", type=float, nargs="+", default=[0.04, 0.08, 0.12, 0.18, 0.25, 1.01])
    p.add_argument("--max_change_rates", type=float, nargs="+", default=[0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50])
    p.add_argument("--min_change_rate", type=float, default=0.02)
    p.add_argument("--score_negative_gain_weight", type=float, default=0.020)
    p.add_argument("--score_edge_gain_weight", type=float, default=0.025)
    p.add_argument("--score_transition_gain_weight", type=float, default=0.010)
    p.add_argument("--score_high_penalty", type=float, default=2.5)
    p.add_argument("--high_tolerance", type=float, default=0.04)
    p.add_argument("--data_path", type=str, default=relpath("raw_data", "RML2016.10a_dict.pkl"))
    p.add_argument("--cache_dir", type=str, default=relpath("feature_cache"))
    p.add_argument("--alignment_cache", type=str, default=relpath("results", "greedy_soup_identity_valtest_probs_for_gamc_fusion.npz"))
    p.add_argument("--skip_alignment_check", action="store_true")
    p.add_argument("--output_cache", type=str, default=relpath("results", "pairwise_confusion_aux_split1_trainvaltest_probs_for_meta.npz"))
    p.add_argument("--records_csv", type=str, default="")
    p.add_argument("--save_top_records", type=int, default=160)
    return p.parse_args()


def norm(prob):
    p = np.asarray(prob, dtype=np.float32)
    p = np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)
    p = np.maximum(p, 1e-12)
    return p / p.sum(axis=1, keepdims=True).clip(min=1e-12)


def top_margin(prob):
    part = np.partition(prob, -2, axis=1)
    return part[:, -1] - part[:, -2]


def parse_pairs(items):
    pairs = []
    for item in items:
        a, b = item.replace(":", ",").split(",")[:2]
        pairs.append((int(a), int(b)))
    return pairs


def load_features(args, train_idx, val_idx, test_idx):
    sources = []
    requested = list(dict.fromkeys(args.stat_feature_sources))
    if "hcs" in requested:
        sources.append(np.load(args.hcs_feature_cache, allow_pickle=True)["features"].astype(np.float32))
    if "gamc" in requested:
        sources.append(np.load(args.gamc_feature_cache, allow_pickle=True)["features"].astype(np.float32))
    if not sources:
        raise ValueError("At least one --stat_feature_sources entry is required.")
    full = sources[0] if len(sources) == 1 else np.concatenate(sources, axis=1).astype(np.float32)
    print(f"[*] Pairwise statistical feature sources={requested} shape={full.shape}")
    return full[train_idx], full[val_idx], full[test_idx]


def prob_features(*probs):
    parts = []
    for p in probs:
        q = norm(p)
        pred = q.argmax(1)
        idx = np.arange(len(q))
        parts.extend([q, q.max(1, keepdims=True), top_margin(q)[:, None], pred[:, None].astype(np.float32)])
    return np.concatenate(parts, axis=1).astype(np.float32)


def make_features(stat_x, main_p, cv_p, gamc_p, hcs_p):
    x = np.concatenate([stat_x, prob_features(main_p, cv_p, gamc_p, hcs_p)], axis=1)
    return np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)


def make_model(args, seed):
    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=int(args.xgb_estimators),
        max_depth=int(args.xgb_depth),
        learning_rate=float(args.xgb_lr),
        subsample=0.90,
        colsample_bytree=0.86,
        reg_lambda=4.0,
        reg_alpha=0.08,
        min_child_weight=2.5,
        tree_method="hist",
        device=str(getattr(args, "xgb_device", "cpu")),
        eval_metric="logloss",
        n_jobs=int(args.xgb_jobs),
        random_state=int(seed),
        verbosity=0,
    )


def align_binary_prob(model, x):
    p = model.predict_proba(x)
    classes = getattr(model, "classes_", np.asarray([0, 1]))
    out = np.zeros((len(x), 2), dtype=np.float32)
    for i, cls in enumerate(classes):
        out[:, int(cls)] = p[:, i]
    out = np.maximum(out, 1e-6)
    out /= out.sum(axis=1, keepdims=True)
    return out


def train_pair_oof(args, x_train, y_train, x_val, x_test, pair, pair_i):
    a, b = pair
    pair_train = np.flatnonzero((y_train == a) | (y_train == b))
    local_y = (y_train[pair_train] == b).astype(np.int64)
    train_pair_prob = np.full((len(y_train), 2), 0.5, dtype=np.float32)
    skf = StratifiedKFold(n_splits=int(args.folds), shuffle=True, random_state=int(args.random_state + pair_i * 101))
    for fold, (tr, va) in enumerate(skf.split(x_train[pair_train], local_y), 1):
        tr_ids = pair_train[tr]
        va_ids = pair_train[va]
        model = make_model(args, args.random_state + pair_i * 1000 + fold)
        model, _, _ = fit_or_load_estimator(
            model,
            x_train[tr_ids],
            local_y[tr],
            cache_dir=args.model_cache_dir,
            reuse=args.reuse_models,
            namespace=f"pairwise_{a}_{b}_fold{fold}",
            source_paths=[
                args.main_oof_cache,
                args.cvtrn_oof_cache,
                args.gamc_oof_cache,
                args.hcs_oof_cache,
                *(
                    [args.hcs_feature_cache]
                    if "hcs" in args.stat_feature_sources
                    else []
                ),
                *(
                    [args.gamc_feature_cache]
                    if "gamc" in args.stat_feature_sources
                    else []
                ),
            ],
            context={
                "builder": "pairwise_oof_v1",
                "split_seed": args.split_seed,
                "pair": [a, b],
                "fold": fold,
                "folds": args.folds,
                "stat_feature_sources": args.stat_feature_sources,
            },
        )
        train_pair_prob[va_ids] = align_binary_prob(model, x_train[va_ids])
        print(f"    pair {a}<->{b} fold {fold}/{args.folds} done")
    final = make_model(args, args.random_state + pair_i * 1000 + 99)
    final, _, _ = fit_or_load_estimator(
        final,
        x_train[pair_train],
        local_y,
        cache_dir=args.model_cache_dir,
        reuse=args.reuse_models,
        namespace=f"pairwise_{a}_{b}_final",
        source_paths=[
            args.main_oof_cache,
            args.cvtrn_oof_cache,
            args.gamc_oof_cache,
            args.hcs_oof_cache,
            *([args.hcs_feature_cache] if "hcs" in args.stat_feature_sources else []),
            *([args.gamc_feature_cache] if "gamc" in args.stat_feature_sources else []),
        ],
        context={
            "builder": "pairwise_final_v1",
            "split_seed": args.split_seed,
            "pair": [a, b],
            "stat_feature_sources": args.stat_feature_sources,
        },
    )
    val_prob = align_binary_prob(final, x_val)
    test_prob = align_binary_prob(final, x_test)
    return train_pair_prob, val_prob, test_prob


def limit_mask(mask, priority, max_rate):
    max_n = int(round(len(mask) * float(max_rate) / 100.0))
    if max_n <= 0:
        return np.zeros_like(mask, dtype=bool)
    idx = np.flatnonzero(mask)
    if len(idx) <= max_n:
        return mask
    order = np.argsort(priority[idx])[::-1][:max_n]
    out = np.zeros_like(mask, dtype=bool)
    out[idx[order]] = True
    return out


def apply_pairwise(main_prob, pair_probs, pairs, cfg):
    if "pair_cfgs" in cfg:
        return apply_pairwise_per_pair(main_prob, pair_probs, pairs, cfg["pair_cfgs"])
    main = norm(main_prob)
    out = np.array(main, copy=True)
    gate_any = np.zeros(len(main), dtype=bool)
    main_pred = main.argmax(1)
    main_margin = top_margin(main)
    idx = np.arange(len(main))
    for pair, pp in zip(pairs, pair_probs):
        a, b = pair
        pair_mass = main[:, a] + main[:, b]
        pair_conf = np.maximum(pp[:, 0], pp[:, 1])
        pair_margin = np.abs(pp[:, 1] - pp[:, 0])
        pair_pred = np.where(pp[:, 1] >= pp[:, 0], b, a)
        candidate = (
            np.isin(main_pred, [a, b])
            & (pair_pred != main_pred)
            & (pair_mass >= float(cfg["main_pair_mass_thr"]))
            & (main_margin <= float(cfg["main_pair_margin_max"]))
            & (pair_conf >= float(cfg["pair_conf_thr"]))
            & (pair_margin >= float(cfg["pair_margin_thr"]))
        )
        priority = pair_conf + pair_margin + 0.20 * pair_mass - 0.15 * main_margin
        gate = limit_mask(candidate & (~gate_any), priority, cfg["max_change_rate"])
        if not np.any(gate):
            continue
        alpha = float(cfg["alpha"])
        pair_out = np.array(out[gate], copy=True)
        mass = pair_out[:, a] + pair_out[:, b]
        pair_out[:, a] = (1.0 - alpha) * pair_out[:, a] + alpha * mass * pp[gate, 0]
        pair_out[:, b] = (1.0 - alpha) * pair_out[:, b] + alpha * mass * pp[gate, 1]
        pair_out = norm(pair_out)
        out[gate] = pair_out
        gate_any |= gate
    return norm(out).astype(np.float32), gate_any


def apply_pairwise_per_pair(main_prob, pair_probs, pairs, pair_cfgs):
    out = norm(main_prob)
    gate_any = np.zeros(len(out), dtype=bool)
    idx = np.arange(len(out))
    for pair, pp, cfg in zip(pairs, pair_probs, pair_cfgs):
        if not cfg or float(cfg.get("alpha", 0.0)) <= 0.0:
            continue
        a, b = pair
        current = norm(out)
        current_pred = current.argmax(1)
        current_margin = top_margin(current)
        pair_mass = current[:, a] + current[:, b]
        pair_conf = np.maximum(pp[:, 0], pp[:, 1])
        pair_margin = np.abs(pp[:, 1] - pp[:, 0])
        pair_pred = np.where(pp[:, 1] >= pp[:, 0], b, a)
        candidate = (
            np.isin(current_pred, [a, b])
            & (pair_pred != current_pred)
            & (pair_mass >= float(cfg["main_pair_mass_thr"]))
            & (current_margin <= float(cfg["main_pair_margin_max"]))
            & (pair_conf >= float(cfg["pair_conf_thr"]))
            & (pair_margin >= float(cfg["pair_margin_thr"]))
        )
        priority = pair_conf + pair_margin + 0.20 * pair_mass - 0.15 * current_margin
        gate = limit_mask(candidate & (~gate_any), priority, cfg["max_change_rate"])
        if not np.any(gate):
            continue
        alpha = float(cfg["alpha"])
        pair_out = np.array(current[gate], copy=True)
        mass = pair_out[:, a] + pair_out[:, b]
        pair_out[:, a] = (1.0 - alpha) * pair_out[:, a] + alpha * mass * pp[gate, 0]
        pair_out[:, b] = (1.0 - alpha) * pair_out[:, b] + alpha * mass * pp[gate, 1]
        pair_out = norm(pair_out)
        out[gate] = pair_out
        gate_any |= gate
    return norm(out).astype(np.float32), gate_any


def search_configs(args, main_train, pair_train_probs, pairs, y_train, snrs_train):
    base_m = base.metrics_from_probs(main_train, y_train, snrs_train)
    cfgs = []
    for alpha in args.blend_alphas:
        for pair_conf_thr in args.pair_conf_thresholds:
            for pair_margin_thr in args.pair_margin_thresholds:
                for main_pair_mass_thr in args.main_pair_mass_thresholds:
                    for main_pair_margin_max in args.main_pair_margin_maxes:
                        for max_change_rate in args.max_change_rates:
                            cfgs.append({
                                "alpha": float(alpha),
                                "pair_conf_thr": float(pair_conf_thr),
                                "pair_margin_thr": float(pair_margin_thr),
                                "main_pair_mass_thr": float(main_pair_mass_thr),
                                "main_pair_margin_max": float(main_pair_margin_max),
                                "max_change_rate": float(max_change_rate),
                            })

    def eval_cfg(cfg):
        out, gate = apply_pairwise(main_train, pair_train_probs, pairs, cfg)
        change_rate = float(gate.mean() * 100.0)
        if change_rate < float(args.min_change_rate):
            return None
        m = base.metrics_from_probs(out, y_train, snrs_train)
        high_drop = max(0.0, base_m["high_acc"] - m["high_acc"] - float(args.high_tolerance))
        score = (
            m["overall_acc"]
            + float(args.score_negative_gain_weight) * (m["negative_acc"] - base_m["negative_acc"])
            + float(args.score_edge_gain_weight) * (m["edge_low_acc"] - base_m["edge_low_acc"])
            + float(args.score_transition_gain_weight) * (m["transition_acc"] - base_m["transition_acc"])
            - float(args.score_high_penalty) * high_drop
        )
        return {
            **cfg,
            "score": float(score),
            "overall_acc": float(m["overall_acc"]),
            "negative_acc": float(m["negative_acc"]),
            "edge_low_acc": float(m["edge_low_acc"]),
            "transition_acc": float(m["transition_acc"]),
            "high_acc": float(m["high_acc"]),
            "change_rate": change_rate,
        }

    jobs = int(getattr(args, "search_jobs", 1))
    if jobs > 1 and len(cfgs) > 1:
        rows = Parallel(n_jobs=jobs, prefer="threads", batch_size=16)(
            delayed(eval_cfg)(cfg) for cfg in cfgs
        )
        rows = [r for r in rows if r is not None]
    else:
        rows = []
        for cfg in cfgs:
            row = eval_cfg(cfg)
            if row is not None:
                rows.append(row)
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows, base_m


def disabled_pair_cfg():
    return {
        "alpha": 0.0,
        "pair_conf_thr": 1.01,
        "pair_margin_thr": 1.01,
        "main_pair_mass_thr": 1.01,
        "main_pair_margin_max": 0.0,
        "max_change_rate": 0.0,
        "score": 0.0,
        "overall_acc": 0.0,
        "change_rate": 0.0,
        "enabled": False,
    }


def search_per_pair_greedy(args, main_train, pair_train_probs, pairs, y_train, snrs_train):
    current = norm(main_train)
    selected = []
    all_rows = []
    for i, pair in enumerate(pairs):
        print(f"    [search] pair {i+1}/{len(pairs)}: {pair[0]}<->{pair[1]}", flush=True)
        rows, step_base = search_configs(args, current, [pair_train_probs[i]], [pair], y_train, snrs_train)
        base_score = float(step_base["overall_acc"])
        accepted = False
        if rows:
            best = dict(rows[0])
            gain = float(best["score"]) - base_score
            if gain >= float(args.per_pair_min_score_gain):
                best["enabled"] = True
                best["pair"] = f"{pair[0]},{pair[1]}"
                current, _ = apply_pairwise(current, [pair_train_probs[i]], [pair], best)
                selected.append(best)
                accepted = True
                print(
                    f"      accepted score={best['score']:.3f} gain={gain:+.4f} "
                    f"overall={best['overall_acc']:.3f}% change={best['change_rate']:.3f}%",
                    flush=True,
                )
            for r in rows:
                rr = dict(r)
                rr["pair"] = f"{pair[0]},{pair[1]}"
                rr["accepted"] = bool(accepted and r is rows[0])
                rr["step_base_overall"] = base_score
                all_rows.append(rr)
        if not accepted:
            cfg = disabled_pair_cfg()
            cfg["pair"] = f"{pair[0]},{pair[1]}"
            selected.append(cfg)
            print("      skipped: no config passed gain threshold", flush=True)
    final_m = base.metrics_from_probs(current, y_train, snrs_train)
    best = {
        "pair_cfgs": selected,
        "selection_mode": "per_pair_greedy",
        "score": float(final_m["overall_acc"]),
        "overall_acc": float(final_m["overall_acc"]),
        "negative_acc": float(final_m["negative_acc"]),
        "edge_low_acc": float(final_m["edge_low_acc"]),
        "transition_acc": float(final_m["transition_acc"]),
        "high_acc": float(final_m["high_acc"]),
        "change_rate": float((current.argmax(1) != norm(main_train).argmax(1)).mean() * 100.0),
    }
    all_rows.sort(key=lambda r: r["score"], reverse=True)
    return best, all_rows, base.metrics_from_probs(main_train, y_train, snrs_train)


def save_csv(path, rows, limit):
    if not rows:
        return
    keys = sorted(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows[:limit]:
            w.writerow(row)
    print(f"[*] CSV saved: {path}")


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_cache), exist_ok=True)
    pairs = parse_pairs(args.pairs)
    main_oof = orig.load_npz(args.main_oof_cache)
    soup = orig.load_npz(args.soup_prob_cache)
    cv = orig.load_npz(args.cvtrn_oof_cache)
    gamc = orig.load_npz(args.gamc_oof_cache)
    hcs = orig.load_npz(args.hcs_oof_cache)
    for key in ("labels_train", "snrs_train"):
        for z, name in [(cv, "CVTRN"), (gamc, "GAMC"), (hcs, "HCS")]:
            orig.assert_same(main_oof, z, key, key, f"Main OOF vs {name}")
    for key in ("labels_val", "snrs_val", "labels_test", "snrs_test"):
        for z, name in [(main_oof, "Main OOF"), (cv, "CVTRN"), (gamc, "GAMC"), (hcs, "HCS")]:
            orig.assert_same(soup, z, key, key, f"Soup vs {name}")

    print("=" * 120)
    print("Train pairwise confusion auxiliary experts")
    print("=" * 120)
    print("Academic protocol:")
    print("  - Pairwise specialists are trained only on train split.")
    print("  - Train probabilities are OOF for the pairwise models.")
    print("  - Pairwise gate thresholds are selected from train-split OOF predictions only.")
    print("  - Validation is diagnostic; test labels are not scored in this script.")

    full_dataset = common.build_full_dataset(args)
    train_idx, val_idx, test_idx = common.make_aligned_split(full_dataset, args.split_seed)
    if not args.skip_alignment_check:
        common.check_alignment(full_dataset, val_idx, test_idx, args.alignment_cache)
    stat_train, stat_val, stat_test = load_features(args, train_idx, val_idx, test_idx)

    main_train = main_oof["train_prob"].astype(np.float32)
    main_val = soup["val_prob"].astype(np.float32)
    main_test = soup["test_prob"].astype(np.float32)
    y_train = main_oof["labels_train"].astype(np.int64)
    snrs_train = main_oof["snrs_train"].astype(np.int32)
    y_val = soup["labels_val"].astype(np.int64)
    snrs_val = soup["snrs_val"].astype(np.int32)

    x_train = make_features(stat_train, main_train, cv["train_prob"], gamc["train_prob"], hcs["train_prob"])
    x_val = make_features(stat_val, main_val, cv["val_prob"], gamc["val_prob"], hcs["val_prob"])
    x_test = make_features(stat_test, main_test, cv["test_prob"], gamc["test_prob"], hcs["test_prob"])
    print(f"Feature dim={x_train.shape[1]} | train={len(x_train):,} | val={len(x_val):,} | test={len(x_test):,}")

    train_pair_probs, val_pair_probs, test_pair_probs = [], [], []
    for i, pair in enumerate(pairs, 1):
        a, b = pair
        print(f"\n[*] Pair specialist {i}/{len(pairs)}: {a}<->{b}")
        trp, vp, tep = train_pair_oof(args, x_train, y_train, x_val, x_test, pair, i)
        train_pair_probs.append(trp)
        val_pair_probs.append(vp)
        test_pair_probs.append(tep)

    if args.selection_mode == "per_pair_greedy":
        best, rows, train_base_m = search_per_pair_greedy(args, main_train, train_pair_probs, pairs, y_train, snrs_train)
        print("\nSelected per-pair train-OOF configs")
        for cfg in best["pair_cfgs"]:
            print(
                f"  pair {cfg['pair']:<5} enabled={bool(cfg.get('enabled', False))} "
                f"score={cfg.get('score', 0.0):.3f} overall={cfg.get('overall_acc', 0.0):.3f}% "
                f"change={cfg.get('change_rate', 0.0):.2f}% a={cfg.get('alpha', 0.0)} "
                f"conf={cfg.get('pair_conf_thr', 1.01)} pm={cfg.get('pair_margin_thr', 1.01)} "
                f"mass={cfg.get('main_pair_mass_thr', 1.01)} mm={cfg.get('main_pair_margin_max', 0.0)} "
                f"max={cfg.get('max_change_rate', 0.0)}"
            )
    else:
        rows, train_base_m = search_configs(args, main_train, train_pair_probs, pairs, y_train, snrs_train)
        if rows:
            best = rows[0]
            print("\nTop pairwise train-OOF configs")
            for i, r in enumerate(rows[:20], 1):
                print(
                    f"{i:02d}. score={r['score']:.3f} overall={r['overall_acc']:.3f}% "
                    f"neg={r['negative_acc']:.3f}% edge={r['edge_low_acc']:.3f}% "
                    f"trans={r['transition_acc']:.3f}% high={r['high_acc']:.3f}% "
                    f"change={r['change_rate']:.2f}% a={r['alpha']} conf={r['pair_conf_thr']} "
                    f"pm={r['pair_margin_thr']} mass={r['main_pair_mass_thr']} mm={r['main_pair_margin_max']} max={r['max_change_rate']}"
                )
        else:
            print("[!] No pairwise config found; falling back to main probabilities.")
            best = {
                "alpha": 0.0,
                "pair_conf_thr": 1.01,
                "pair_margin_thr": 1.01,
                "main_pair_mass_thr": 1.01,
                "main_pair_margin_max": 0.0,
                "max_change_rate": 0.0,
                "score": float(train_base_m["overall_acc"]),
                "overall_acc": float(train_base_m["overall_acc"]),
                "change_rate": 0.0,
            }

    train_out, train_gate = apply_pairwise(main_train, train_pair_probs, pairs, best)
    val_out, val_gate = apply_pairwise(main_val, val_pair_probs, pairs, best)
    test_out, test_gate = apply_pairwise(main_test, test_pair_probs, pairs, best)

    records_csv = args.records_csv or os.path.splitext(args.output_cache)[0] + "_search_top.csv"
    save_csv(records_csv, rows if rows else [best], args.save_top_records)

    print("\nDiagnostics")
    base.print_metrics_line("Main Train-OOF", train_base_m)
    base.print_metrics_line("Pairwise Train-OOF", base.metrics_from_probs(train_out, y_train, snrs_train))
    base.print_metrics_line("Main Val", base.metrics_from_probs(main_val, y_val, snrs_val))
    base.print_metrics_line("Pairwise Val", base.metrics_from_probs(val_out, y_val, snrs_val))
    print(f"Gate rates: train={train_gate.mean()*100:.3f}% val={val_gate.mean()*100:.3f}% test={test_gate.mean()*100:.3f}%")

    selected = {
        "pairs": pairs,
        "best": best,
        "protocol": "train-split OOF pairwise confusion auxiliary; validation diagnostics only; test labels not scored",
        "main_oof_cache": args.main_oof_cache,
        "cvtrn_oof_cache": args.cvtrn_oof_cache,
        "gamc_oof_cache": args.gamc_oof_cache,
        "hcs_oof_cache": args.hcs_oof_cache,
    }
    np.savez_compressed(
        args.output_cache,
        train_prob=train_out.astype(np.float32),
        val_prob=val_out.astype(np.float32),
        test_prob=test_out.astype(np.float32),
        pair_classes=np.asarray(pairs, dtype=np.int64),
        train_pair_probs=np.stack(train_pair_probs, axis=0).astype(np.float32),
        val_pair_probs=np.stack(val_pair_probs, axis=0).astype(np.float32),
        test_pair_probs=np.stack(test_pair_probs, axis=0).astype(np.float32),
        train_gate=train_gate.astype(bool),
        val_gate=val_gate.astype(bool),
        test_gate=test_gate.astype(bool),
        labels_train=main_oof["labels_train"],
        snrs_train=main_oof["snrs_train"],
        labels_val=soup["labels_val"],
        snrs_val=soup["snrs_val"],
        labels_test=soup["labels_test"],
        snrs_test=soup["snrs_test"],
        mod_classes=soup.get("mod_classes", np.asarray(common.DEFAULT_MOD_CLASSES)),
        selected_config=np.asarray([json.dumps(selected, ensure_ascii=False)]),
        protocol=np.asarray([selected["protocol"]]),
    )
    print(f"[*] Pairwise auxiliary cache saved: {args.output_cache}")


if __name__ == "__main__":
    main()
