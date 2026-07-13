import argparse
import csv
import json
import os

import numpy as np
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from model_cache_utils import fit_or_load_estimator

import evaluate_fourier_gamc_cvtrn_stacked_residual_fusion as st
import evaluate_fourier_gamc_cvtrn_train_oof_meta_fusion as oof
import evaluate_fourier_gamc_multi_cvtrn_blind_quality_xgb_fusion as bq
import evaluate_greedy_soup_gamc_protected_residual_fusion as base
import evaluate_fourier_oof_gamc_cvtrn_extraaux_residual_meta_fusion as extra_eval
import evaluate_fourier_oof_gamc_cvtrn_residual_meta_fusion as orig
import train_cv_trn_aux_2016 as common


def relpath(*parts):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Train-OOF rescue-risk router for HCS precision auxiliary. "
            "The router learns from train-split OOF rescue/harm labels, then validation labels select only thresholds."
        )
    )
    p.add_argument("--split_seed", type=int, default=1)
    p.add_argument("--random_state", type=int, default=2026)
    p.add_argument("--soup_prob_cache", type=str, default=relpath("results", "greedy_soup_identity_valtest_probs_for_gamc_fusion.npz"))
    p.add_argument("--fourier_oof_cache", type=str, default=relpath("results", "fourier_main_oof_merge_mseed78_79_f3e220_bs128_split1_trainvaltest_probs_for_meta.npz"))
    p.add_argument("--gamc_oof_cache", type=str, default=relpath("results", "gamc_oof_tree_split1_trainvaltest_probs_for_meta.npz"))
    p.add_argument("--cvtrn_oof_cache", type=str, default=relpath("results", "cv_trn_aux_v2_oof_mseed41_f3e240_split1_trainvaltest_probs_for_meta.npz"))
    p.add_argument("--hcs_precision_cache", type=str, default=relpath("results", "hcs_precision_analog_aux_split1_trainvaltest_probs_for_meta.npz"))
    p.add_argument(
        "--cvtrn_valtest_caches",
        type=str,
        nargs="*",
        default=[
            relpath("results", "cv_trn_aux_v2_soup_tta_split1_valtest_probs_for_fusion.npz"),
            relpath("results", "cv_trn_aux_v2_w8d96_soup_tta_split1_valtest_probs_for_fusion.npz"),
        ],
    )
    p.add_argument("--use_oof_cvtrn_only", action="store_true")
    p.add_argument("--use_fourier_oof_infer", action="store_true")
    p.add_argument("--output_suffix", type=str, default="fourier_oof_hcs_rescue_risk_router_split1")

    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--stage_oof_folds", type=int, default=3)
    p.add_argument("--stage_models", nargs="+", default=["xgb_d2_620", "xgb_d3_520", "xgb_d4_400"])
    p.add_argument("--stage_estimator_scale", type=float, default=1.0)
    p.add_argument("--stage_min_val_overall", type=float, default=0.0)
    p.add_argument("--top_stage_configs", type=int, default=12)
    p.add_argument("--blend_alphas", type=float, nargs="+", default=[0.50, 0.65, 0.80])
    p.add_argument("--meta_conf_thresholds", type=float, nargs="+", default=[0.00, 0.25, 0.35, 0.45, 0.55])
    p.add_argument("--advantage_thresholds", type=float, nargs="+", default=[-0.20, -0.10, 0.00, 0.05, 0.10])
    p.add_argument("--max_change_rates", type=float, nargs="+", default=[6.0, 8.0, 10.0, 12.0])
    p.add_argument("--hard_keep_conf", type=float, default=0.93)
    p.add_argument("--hard_keep_margin", type=float, default=0.70)

    p.add_argument("--router_estimators", type=int, default=420)
    p.add_argument("--router_depths", type=int, nargs="+", default=[2, 3])
    p.add_argument("--router_modes", nargs="+", default=["binary", "risk3"])
    p.add_argument("--router_harm_penalty", type=float, default=1.35)
    p.add_argument("--router_learning_rate", type=float, default=0.035)
    p.add_argument("--router_thresholds", type=float, nargs="+", default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70])
    p.add_argument("--router_max_change_rates", type=float, nargs="+", default=[0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0])
    p.add_argument("--router_alphas", type=float, nargs="+", default=[0.50, 0.65, 0.80, 1.00])
    p.add_argument("--router_min_change_rate", type=float, default=0.05)
    p.add_argument("--router_min_stage_overall_gain", type=float, default=0.0)
    p.add_argument("--allow_router_stage_drop", action="store_true")
    p.add_argument("--pair_guard_source", choices=["off", "train", "val", "train_val_agree", "train_val_veto"], default="off")
    p.add_argument("--pair_guard_min_net", type=int, default=1)
    p.add_argument("--pair_guard_min_precision", type=float, default=0.52)
    p.add_argument("--pair_guard_min_count", type=int, default=20)
    p.add_argument("--pair_guard_val_veto_net", type=int, default=-2)
    p.add_argument("--pair_guard_val_veto_precision", type=float, default=0.45)
    p.add_argument("--rescue_weight", type=float, default=28.0)
    p.add_argument("--harm_weight", type=float, default=12.0)
    p.add_argument("--neutral_weight", type=float, default=0.45)
    p.add_argument("--wrong_wrong_weight", type=float, default=1.25)
    p.add_argument("--correct_correct_weight", type=float, default=0.35)

    p.add_argument("--score_overall_weight", type=float, default=1.0)
    p.add_argument("--score_negative_gain_weight", type=float, default=0.020)
    p.add_argument("--score_edge_gain_weight", type=float, default=0.020)
    p.add_argument("--score_transition_gain_weight", type=float, default=0.010)
    p.add_argument("--score_high_penalty", type=float, default=3.00)
    p.add_argument("--high_tolerance", type=float, default=0.05)
    p.add_argument("--score_changed_high_penalty", type=float, default=0.015)
    p.add_argument("--score_changed_nonultra_penalty", type=float, default=0.006)

    p.add_argument("--xgb_jobs", type=int, default=-1)
    p.add_argument("--quality_estimators", type=int, default=280)
    p.add_argument("--quality_max_depth", type=int, default=4)
    p.add_argument("--quality_learning_rate", type=float, default=0.04)
    p.add_argument("--quality_subsample", type=float, default=0.90)
    p.add_argument("--quality_colsample", type=float, default=0.85)
    p.add_argument("--quality_chunk_size", type=int, default=32768)
    p.add_argument("--data_path", type=str, default=relpath("raw_data", "RML2016.10a_dict.pkl"))
    p.add_argument("--cache_dir", type=str, default=relpath("feature_cache"))
    p.add_argument("--alignment_cache", type=str, default=relpath("results", "greedy_soup_identity_valtest_probs_for_gamc_fusion.npz"))
    p.add_argument("--skip_alignment_check", action="store_true")
    p.add_argument("--save_top_records", type=int, default=240)
    p.add_argument("--cm_snrs", type=int, nargs="+", default=[0, -6])
    return p.parse_args()


def xgb_stage_model(args, name, seed):
    if name == "et_depth20":
        return orig.et_model(args, seed)
    if name == "xgb_d2_620":
        depth, est, lr, child = 2, 620, 0.032, 2.0
    elif name == "xgb_d3_520":
        depth, est, lr, child = 3, 520, 0.030, 2.0
    else:
        depth, est, lr, child = 4, 400, 0.028, 3.0
    est = max(2, int(round(est * float(args.stage_estimator_scale))))
    return XGBClassifier(
        objective="multi:softprob",
        num_class=base.NUM_CLASSES,
        n_estimators=est,
        max_depth=depth,
        learning_rate=lr,
        subsample=0.90,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        reg_alpha=0.05,
        min_child_weight=child,
        tree_method="hist",
        device=str(getattr(args, "xgb_device", "cpu")),
        eval_metric="mlogloss",
        n_jobs=int(args.xgb_jobs),
        random_state=int(seed),
        verbosity=0,
    )


def router_model(args, depth, seed):
    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=int(args.router_estimators),
        max_depth=int(depth),
        learning_rate=float(args.router_learning_rate),
        subsample=0.88,
        colsample_bytree=0.90,
        reg_lambda=4.0,
        reg_alpha=0.10,
        min_child_weight=3.0,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=int(args.xgb_jobs),
        random_state=int(seed),
        verbosity=0,
    )


def router_risk3_model(args, depth, seed):
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=int(args.router_estimators),
        max_depth=int(depth),
        learning_rate=float(args.router_learning_rate),
        subsample=0.88,
        colsample_bytree=0.90,
        reg_lambda=4.0,
        reg_alpha=0.10,
        min_child_weight=3.0,
        tree_method="hist",
        eval_metric="mlogloss",
        n_jobs=int(args.xgb_jobs),
        random_state=int(seed),
        verbosity=0,
    )


def top_margin(prob):
    part = np.partition(prob, -2, axis=1)
    return part[:, -1] - part[:, -2]


def top2_values(prob):
    part = np.partition(prob, -2, axis=1)
    return part[:, -1], part[:, -2]


def safe_norm(prob):
    return oof.norm(prob).astype(np.float32)


def build_router_features(stage_prob, hcs_prob, main_prob, cv_prob, gamc_prob, qprob):
    stage = safe_norm(stage_prob)
    hcs = safe_norm(hcs_prob)
    main = safe_norm(main_prob)
    cv = safe_norm(cv_prob)
    gamc = safe_norm(gamc_prob)
    qfeat = bq.quality_probability_features(qprob).astype(np.float32)

    parts = []
    for p in [stage, hcs, main, cv, gamc]:
        parts.extend(oof.expert_blocks(p))
    for a, b in [(stage, hcs), (stage, main), (stage, cv), (stage, gamc), (hcs, main), (hcs, cv), (hcs, gamc)]:
        parts.append(oof.pair_blocks(a, b))

    sp, hp = stage.argmax(1), hcs.argmax(1)
    smax, s2 = top2_values(stage)
    hmax, h2 = top2_values(hcs)
    idx = np.arange(len(stage))
    compact = np.stack(
        [
            (sp == hp).astype(np.float32),
            hmax - smax,
            (hmax - h2) - (smax - s2),
            hcs[idx, sp],
            stage[idx, hp],
            hcs[idx, hp] - stage[idx, hp],
            stage[idx, sp] - hcs[idx, sp],
            np.abs(stage - hcs).sum(1),
            (stage * hcs).sum(1),
        ],
        axis=1,
    ).astype(np.float32)
    parts.append(compact)
    parts.append(qfeat)
    x = np.concatenate(parts, axis=1)
    return np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)


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


def apply_router(stage_prob, hcs_prob, router_score, cfg):
    stage = safe_norm(stage_prob)
    hcs = safe_norm(hcs_prob)
    stage_pred = stage.argmax(1)
    hcs_pred = hcs.argmax(1)
    hconf = hcs.max(1)
    sconf = stage.max(1)
    hmargin = top_margin(hcs)
    smargin = top_margin(stage)
    candidate = hcs_pred != stage_pred
    pair_allow = cfg.get("_pair_allow")
    if pair_allow is not None:
        candidate &= pair_allow[stage_pred, hcs_pred]
    priority = router_score + 0.10 * (hconf - sconf) + 0.05 * (hmargin - smargin)
    raw = candidate & (router_score >= float(cfg["router_threshold"]))
    gate = limit_mask(raw, priority, cfg["router_max_change_rate"])
    out = np.array(stage, copy=True)
    alpha = float(cfg["router_alpha"])
    out[gate] = (1.0 - alpha) * stage[gate] + alpha * hcs[gate]
    out /= out.sum(axis=1, keepdims=True).clip(min=1e-12)
    return out.astype(np.float32), gate


def pair_guard_matrix(stage_prob, hcs_prob, labels, min_net, min_precision, min_count):
    stage_pred = safe_norm(stage_prob).argmax(1)
    hcs_pred = safe_norm(hcs_prob).argmax(1)
    labels = np.asarray(labels, dtype=np.int64)
    allow = np.zeros((base.NUM_CLASSES, base.NUM_CLASSES), dtype=bool)
    stats = []
    for a in range(base.NUM_CLASSES):
        for b in range(base.NUM_CLASSES):
            if a == b:
                continue
            m = (stage_pred == a) & (hcs_pred == b)
            count = int(m.sum())
            if count < int(min_count):
                continue
            rescue = int((m & (labels == b)).sum())
            harm = int((m & (labels == a)).sum())
            net = rescue - harm
            precision = rescue / max(1, rescue + harm)
            ok = (net >= int(min_net)) and (precision >= float(min_precision))
            allow[a, b] = ok
            stats.append(
                {
                    "from": int(a),
                    "to": int(b),
                    "count": count,
                    "rescue": rescue,
                    "harm": harm,
                    "net": net,
                    "precision": float(precision),
                    "allow": bool(ok),
                }
            )
    return allow, stats


def build_pair_guard(args, stage_train, hcs_train, y_train, stage_val, hcs_val, y_val):
    if args.pair_guard_source == "off":
        return None, []
    train_allow, train_stats = pair_guard_matrix(
        stage_train,
        hcs_train,
        y_train,
        args.pair_guard_min_net,
        args.pair_guard_min_precision,
        args.pair_guard_min_count,
    )
    if args.pair_guard_source == "train":
        return train_allow, train_stats
    val_allow, val_stats = pair_guard_matrix(
        stage_val,
        hcs_val,
        y_val,
        args.pair_guard_min_net,
        args.pair_guard_min_precision,
        max(3, min(args.pair_guard_min_count, 8)),
    )
    if args.pair_guard_source == "val":
        return val_allow, val_stats
    merged = train_allow & val_allow
    if args.pair_guard_source == "train_val_veto":
        merged = train_allow.copy()
        val_by_pair = {(s["from"], s["to"]): s for s in val_stats}
        for a in range(base.NUM_CLASSES):
            for b in range(base.NUM_CLASSES):
                if not merged[a, b]:
                    continue
                s = val_by_pair.get((a, b))
                if s is None:
                    continue
                if s["net"] <= int(args.pair_guard_val_veto_net):
                    merged[a, b] = False
                if s["precision"] < float(args.pair_guard_val_veto_precision):
                    merged[a, b] = False
    return merged, train_stats + val_stats


def stage_crossfit(args, x_train, y_train, weights, x_val, x_test):
    branches = []
    skf = StratifiedKFold(n_splits=int(args.stage_oof_folds), shuffle=True, random_state=int(args.random_state + 811))
    for m_i, name in enumerate(args.stage_models):
        print(f"[*] Cross-fitting stage-1 meta model: {name}")
        train_oof = np.zeros((len(x_train), base.NUM_CLASSES), dtype=np.float32)
        for fold, (tr, va) in enumerate(skf.split(x_train, y_train), 1):
            clf = xgb_stage_model(args, name, args.random_state + 1000 * m_i + fold)
            clf, _, _ = fit_or_load_estimator(
                clf,
                x_train[tr],
                y_train[tr],
                sample_weight=weights[tr],
                cache_dir=str(getattr(args, "model_cache_dir", "")),
                reuse=bool(getattr(args, "reuse_models", False)),
                namespace=f"transition_stage_{name}_fold{fold}",
                source_paths=list(getattr(args, "model_cache_source_paths", []) or []),
                context={
                    "builder": "transition_stage_crossfit_v2",
                    "split_seed": int(getattr(args, "split_seed", 1)),
                    "fold": int(fold),
                    "folds": int(args.stage_oof_folds),
                    "stage_aux_sources": list(getattr(args, "stage_aux_sources", []) or []),
                },
            )
            train_oof[va] = orig.aligned_proba(clf, x_train[va]).astype(np.float32)
            print(f"    stage fold {fold}/{args.stage_oof_folds} done")
        final_seed_offsets = {
            "xgb_d2_620": 11,
            "xgb_d3_520": 23,
            "xgb_d4_400": 37,
            "et_depth20": 53,
        }
        final = xgb_stage_model(args, name, args.random_state + final_seed_offsets.get(name, 99 + m_i))
        final, _, _ = fit_or_load_estimator(
            final,
            x_train,
            y_train,
            sample_weight=weights,
            cache_dir=str(getattr(args, "model_cache_dir", "")),
            reuse=bool(getattr(args, "reuse_models", False)),
            namespace=f"transition_stage_{name}_final",
            source_paths=list(getattr(args, "model_cache_source_paths", []) or []),
            context={
                "builder": "transition_stage_crossfit_v2",
                "split_seed": int(getattr(args, "split_seed", 1)),
                "fold": "final",
                "folds": int(args.stage_oof_folds),
                "stage_aux_sources": list(getattr(args, "stage_aux_sources", []) or []),
            },
        )
        val_prob = orig.aligned_proba(final, x_val).astype(np.float32)
        test_prob = orig.aligned_proba(final, x_test).astype(np.float32)
        branches.append({"name": name, "train": train_oof, "val": val_prob, "test": test_prob})
    if len(branches) > 1:
        branches.append(
            {
                "name": "stage1_oof_ensemble",
                "train": orig.log_average([b["train"] for b in branches]).astype(np.float32),
                "val": orig.log_average([b["val"] for b in branches]).astype(np.float32),
                "test": orig.log_average([b["test"] for b in branches]).astype(np.float32),
            }
        )
    return branches


def action_labels(stage_prob, hcs_prob, labels):
    stage_pred = stage_prob.argmax(1)
    hcs_pred = hcs_prob.argmax(1)
    stage_ok = stage_pred == labels
    hcs_ok = hcs_pred == labels
    rescue = (~stage_ok) & hcs_ok & (hcs_pred != stage_pred)
    harm = stage_ok & (~hcs_ok) & (hcs_pred != stage_pred)
    wrong_wrong = (~stage_ok) & (~hcs_ok) & (hcs_pred != stage_pred)
    correct_correct = stage_ok & hcs_ok
    y = rescue.astype(np.int32)
    return y, rescue, harm, wrong_wrong, correct_correct


def risk3_labels(rescue, harm):
    y = np.zeros(len(rescue), dtype=np.int32)
    y[harm] = 1
    y[rescue] = 2
    return y


def router_weights(args, rescue, harm, wrong_wrong, correct_correct, snrs):
    w = np.full(len(rescue), float(args.neutral_weight), dtype=np.float32)
    w[rescue] = float(args.rescue_weight)
    w[harm] = float(args.harm_weight)
    w[wrong_wrong] = float(args.wrong_wrong_weight)
    w[correct_correct] = float(args.correct_correct_weight)
    w[snrs < 0] *= 1.25
    w[np.isin(snrs, [-18, -16])] *= 1.25
    return w


def search_router(stage_val, hcs_val, router_score_val, yv, sv, stage_val_m, args, router_name, pair_allow=None, pair_guard_allowed=0):
    rows = []
    min_overall = float(stage_val_m["overall_acc"]) + float(args.router_min_stage_overall_gain)
    for router_alpha in args.router_alphas:
        for router_threshold in args.router_thresholds:
            for router_max_change_rate in args.router_max_change_rates:
                cfg = {
                    "router_name": router_name,
                    "router_alpha": float(router_alpha),
                    "router_threshold": float(router_threshold),
                    "router_max_change_rate": float(router_max_change_rate),
                }
                if pair_allow is not None:
                    cfg["_pair_allow"] = pair_allow
                out, gate = apply_router(stage_val, hcs_val, router_score_val, cfg)
                change_rate = float(gate.mean() * 100.0)
                if change_rate < float(args.router_min_change_rate):
                    continue
                m = base.metrics_from_probs(out, yv, sv)
                if (not args.allow_router_stage_drop) and m["overall_acc"] < min_overall:
                    continue
                score = (
                    args.score_overall_weight * m["overall_acc"]
                    + args.score_negative_gain_weight * (m["negative_acc"] - stage_val_m["negative_acc"])
                    + args.score_edge_gain_weight * (m["edge_low_acc"] - stage_val_m["edge_low_acc"])
                    + args.score_transition_gain_weight * (m["transition_acc"] - stage_val_m["transition_acc"])
                )
                high_drop = max(0.0, stage_val_m["high_acc"] - m["high_acc"] - args.high_tolerance)
                score -= args.score_high_penalty * high_drop
                score -= args.score_changed_high_penalty * float(np.mean(gate[sv >= 0]) * 100.0)
                score -= args.score_changed_nonultra_penalty * float(np.mean(gate[sv < 0]) * 100.0)
                rows.append(
                    {
                        **cfg,
                        "score": float(score),
                        "overall_acc": float(m["overall_acc"]),
                        "negative_acc": float(m["negative_acc"]),
                        "edge_low_acc": float(m["edge_low_acc"]),
                        "transition_acc": float(m["transition_acc"]),
                        "high_acc": float(m["high_acc"]),
                        "change_rate": change_rate,
                        "pair_guard_source": args.pair_guard_source,
                        "pair_guard_allowed": int(pair_guard_allowed),
                    }
                )
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


def save_csv(path, rows, limit):
    if not rows:
        return
    keys = sorted(set(k for row in rows for k in row.keys() if not k.startswith("_")))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows[:limit]:
            w.writerow({k: row.get(k) for k in keys})
    print(f"[*] CSV saved: {path}")


def print_top(rows, n=20):
    print("\nTop train-OOF rescue-risk router validation configs")
    for i, r in enumerate(rows[:n], 1):
        print(
            f"{i:02d}. score={r['score']:.3f} | overall={r['overall_acc']:.3f}% | "
            f"neg={r['negative_acc']:.3f}% | edge={r['edge_low_acc']:.3f}% | "
            f"trans={r['transition_acc']:.3f}% | high={r['high_acc']:.3f}% | "
            f"change={r['change_rate']:.2f}% | stage={r['stage_branch']} | router={r['router_name']} | "
            f"thr={r['router_threshold']} alpha={r['router_alpha']} max={r['router_max_change_rate']}"
        )


def main():
    args = parse_args()
    os.makedirs(relpath("results"), exist_ok=True)
    suffix = args.output_suffix
    print("=" * 144)
    print("Train-OOF HCS rescue-risk router fusion")
    print("=" * 144)
    print("Academic protocol:")
    print("  - Fourier soup remains the main model.")
    print("  - Stage-1 meta train predictions are cross-fitted on train split only.")
    print("  - HCS rescue-risk router is trained only from train-split OOF rescue/harm labels.")
    print("  - Validation labels select router thresholds/blends only.")
    print("  - Test labels are used only once for the final report.")
    print("  - True SNR is never used as an inference feature.")

    soup = orig.load_npz(args.soup_prob_cache)
    fourier = orig.load_npz(args.fourier_oof_cache)
    gamc = orig.load_npz(args.gamc_oof_cache)
    cv = orig.load_npz(args.cvtrn_oof_cache)
    hcs = orig.load_npz(args.hcs_precision_cache)

    for key in ("labels_train", "snrs_train"):
        orig.assert_same(fourier, gamc, key, key, "Fourier OOF vs GAMC OOF")
        orig.assert_same(fourier, cv, key, key, "Fourier OOF vs CVTRN OOF")
        orig.assert_same(fourier, hcs, key, key, "Fourier OOF vs HCS precision")
    for key in ("labels_val", "snrs_val", "labels_test", "snrs_test"):
        orig.assert_same(soup, fourier, key, key, "Fourier soup vs Fourier OOF")
        orig.assert_same(soup, gamc, key, key, "Fourier soup vs GAMC OOF")
        orig.assert_same(soup, cv, key, key, "Fourier soup vs CVTRN OOF")
        orig.assert_same(soup, hcs, key, key, "Fourier soup vs HCS precision")
    print("[*] Alignment check passed for all OOF caches.")

    y_train = fourier["labels_train"].astype(np.int64)
    snrs_train = fourier["snrs_train"].astype(np.int32)
    yv = soup["labels_val"].astype(np.int64)
    sv = soup["snrs_val"].astype(np.int32)
    yt = soup["labels_test"].astype(np.int64)
    stest = soup["snrs_test"].astype(np.int32)

    cv_train = cv["train_prob"].astype(np.float32)
    cv_val_list = [cv["val_prob"].astype(np.float32)]
    cv_test_list = [cv["test_prob"].astype(np.float32)]
    if args.use_oof_cvtrn_only:
        print("[*] Strict mode: using only CV-TRN OOF fold-soup at inference.")
    else:
        for path in args.cvtrn_valtest_caches:
            if not path or not os.path.exists(path):
                print(f"[!] Optional CV-TRN val/test cache skipped: {path}")
                continue
            z = orig.load_npz(path)
            for key in ("labels_val", "snrs_val", "labels_test", "snrs_test"):
                orig.assert_same(soup, z, key, key, f"Optional CV-TRN {os.path.basename(path)}")
            cv_val_list.append(z["val_prob"].astype(np.float32))
            cv_test_list.append(z["test_prob"].astype(np.float32))
            print(f"[*] Added inference CV-TRN cache: {path}")
    cv_val = orig.log_average(cv_val_list).astype(np.float32)
    cv_test = orig.log_average(cv_test_list).astype(np.float32)

    full_dataset = common.build_full_dataset(args)
    train_idx, val_idx, test_idx = common.make_aligned_split(full_dataset, args.split_seed)
    if not args.skip_alignment_check:
        common.check_alignment(full_dataset, val_idx, test_idx, args.alignment_cache)
    snrs_all = np.asarray(full_dataset.snrs, dtype=np.int32)
    q_train, q_val, q_test, q_acc = oof.build_quality_probs(args, train_idx, val_idx, test_idx, full_dataset, snrs_all)

    main_train = fourier["train_prob"].astype(np.float32)
    main_val = fourier["val_prob"].astype(np.float32) if args.use_fourier_oof_infer else soup["val_prob"].astype(np.float32)
    main_test = fourier["test_prob"].astype(np.float32) if args.use_fourier_oof_infer else soup["test_prob"].astype(np.float32)
    hcs_train = hcs["train_prob"].astype(np.float32)
    hcs_val = hcs["val_prob"].astype(np.float32)
    hcs_test = hcs["test_prob"].astype(np.float32)

    print("\nBaselines")
    base_val_m = base.metrics_from_probs(main_val, yv, sv)
    base.print_metrics_line("Main Fourier Val", base_val_m)
    base.print_metrics_line("CV-TRN Val", base.metrics_from_probs(cv_val, yv, sv))
    base.print_metrics_line("GAMC-tree Val", base.metrics_from_probs(gamc["val_prob"], yv, sv))
    base.print_metrics_line("HCS precision Val", base.metrics_from_probs(hcs_val, yv, sv))

    print("\n[*] Building stage-1 train/val/test meta-features")
    x_train = extra_eval.build_features(main_train, cv_train, gamc["train_prob"], gamc["train_member_probs"], q_train, [hcs_train])
    x_val = extra_eval.build_features(main_val, cv_val, gamc["val_prob"], gamc["val_member_probs"], q_val, [hcs_val])
    x_test = extra_eval.build_features(main_test, cv_test, gamc["test_prob"], gamc["test_member_probs"], q_test, [hcs_test])
    weights = orig.sample_weights(snrs_train, main_train, y_train)
    print(f"Stage feature dim: {x_train.shape[1]} | train={len(x_train):,} | val={len(x_val):,} | test={len(x_test):,}")

    branches = stage_crossfit(args, x_train, y_train, weights, x_val, x_test)
    stage_candidates = []
    for b in branches:
        rows = st.search_configs(main_val, b["val"], yv, sv, base_val_m, args, b["name"])
        rows.sort(key=lambda r: r["score"], reverse=True)
        for r in rows[: max(1, args.top_stage_configs)]:
            stage_candidates.append((r["score"], b, r))
        best = rows[0]
        print(
            f"    {b['name']:<20} best val={best['overall_acc']:.3f}% score={best['score']:.3f} "
            f"alpha={best['alpha']} maxchg={best['max_change_rate']}"
        )
    stage_candidates.sort(key=lambda x: x[0], reverse=True)

    all_rows = []
    router_records = []
    for stage_rank, (_, branch, stage_cfg) in enumerate(stage_candidates[: args.top_stage_configs], 1):
        stage_train, _, _, _ = st.apply_stacked(main_train, branch["train"], stage_cfg, args)
        stage_val, _, _, _ = st.apply_stacked(main_val, branch["val"], stage_cfg, args)
        stage_test, _, _, _ = st.apply_stacked(main_test, branch["test"], stage_cfg, args)
        stage_val_m = base.metrics_from_probs(stage_val, yv, sv)
        if stage_val_m["overall_acc"] < float(args.stage_min_val_overall):
            print(
                f"[*] Stage candidate {stage_rank}: {branch['name']} val={stage_val_m['overall_acc']:.3f}% "
                f"skipped by --stage_min_val_overall {args.stage_min_val_overall:.3f}"
            )
            continue

        router_x_train = build_router_features(stage_train, hcs_train, main_train, cv_train, gamc["train_prob"], q_train)
        router_x_val = build_router_features(stage_val, hcs_val, main_val, cv_val, gamc["val_prob"], q_val)
        router_x_test = build_router_features(stage_test, hcs_test, main_test, cv_test, gamc["test_prob"], q_test)
        y_action, rescue, harm, wrong_wrong, correct_correct = action_labels(stage_train, hcs_train, y_train)
        y_risk3 = risk3_labels(rescue, harm)
        rw = router_weights(args, rescue, harm, wrong_wrong, correct_correct, snrs_train)
        pair_allow, pair_stats = build_pair_guard(args, stage_train, hcs_train, y_train, stage_val, hcs_val, yv)
        pair_guard_allowed = int(pair_allow.sum()) if pair_allow is not None else 0
        pos = int(y_action.sum())
        harm_n = int(harm.sum())
        print(
            f"[*] Stage candidate {stage_rank}: {branch['name']} val={stage_val_m['overall_acc']:.3f}% "
            f"train rescue={pos} harm={harm_n}"
        )
        if pair_allow is not None:
            top_allowed = sorted(
                [s for s in pair_stats if s["allow"]],
                key=lambda s: (s["net"], s["precision"], s["count"]),
                reverse=True,
            )[:6]
            print(
                f"    pair guard source={args.pair_guard_source} allowed={pair_guard_allowed} "
                f"min_net={args.pair_guard_min_net} min_precision={args.pair_guard_min_precision:.2f}"
            )
            for s in top_allowed:
                print(
                    f"      pair {s['from']}->{s['to']} count={s['count']} "
                    f"rescue={s['rescue']} harm={s['harm']} net={s['net']} precision={s['precision']:.3f}"
                )

        for depth in args.router_depths:
            if "binary" in args.router_modes:
                rname = f"binary_xgb_d{depth}"
                clf = router_model(args, depth, args.random_state + 100 * stage_rank + depth)
                clf.fit(router_x_train, y_action, sample_weight=rw)
                val_score = clf.predict_proba(router_x_val)[:, 1].astype(np.float32)
                test_score = clf.predict_proba(router_x_test)[:, 1].astype(np.float32)
                rows = search_router(
                    stage_val,
                    hcs_val,
                    val_score,
                    yv,
                    sv,
                    stage_val_m,
                    args,
                    rname,
                    pair_allow=pair_allow,
                    pair_guard_allowed=pair_guard_allowed,
                )
                for row in rows:
                    row["_branch"] = branch
                    row["_stage_cfg"] = stage_cfg
                    row["_stage_test"] = stage_test
                    row["_test_score"] = test_score
                    row["stage_branch"] = branch["name"]
                    row["stage_val_overall"] = float(stage_val_m["overall_acc"])
                    all_rows.append(row)
                router_records.append((branch["name"], rname, len(rows)))

            if "risk3" in args.router_modes:
                rname = f"risk3_xgb_d{depth}"
                clf = router_risk3_model(args, depth, args.random_state + 200 * stage_rank + depth)
                clf.fit(router_x_train, y_risk3, sample_weight=rw)
                pv = clf.predict_proba(router_x_val).astype(np.float32)
                pt = clf.predict_proba(router_x_test).astype(np.float32)
                val_score = (pv[:, 2] - float(args.router_harm_penalty) * pv[:, 1]).astype(np.float32)
                test_score = (pt[:, 2] - float(args.router_harm_penalty) * pt[:, 1]).astype(np.float32)
                rows = search_router(
                    stage_val,
                    hcs_val,
                    val_score,
                    yv,
                    sv,
                    stage_val_m,
                    args,
                    rname,
                    pair_allow=pair_allow,
                    pair_guard_allowed=pair_guard_allowed,
                )
                for row in rows:
                    row["_branch"] = branch
                    row["_stage_cfg"] = stage_cfg
                    row["_stage_test"] = stage_test
                    row["_test_score"] = test_score
                    row["stage_branch"] = branch["name"]
                    row["stage_val_overall"] = float(stage_val_m["overall_acc"])
                    all_rows.append(row)
                router_records.append((branch["name"], rname, len(rows)))

    if all_rows:
        all_rows.sort(key=lambda r: r["score"], reverse=True)
        print_top(all_rows, 20)
        best = all_rows[0]
        stage_test = best["_stage_test"]
        final_test, router_gate = apply_router(stage_test, hcs_test, best["_test_score"], best)
        used_router = True
    else:
        print("\n[!] No rescue-risk router improved validation overall over its stage candidate; keeping best stage-1 candidate.")
        _, branch, stage_cfg = stage_candidates[0]
        stage_test, _, _, _ = st.apply_stacked(main_test, branch["test"], stage_cfg, args)
        final_test = stage_test
        router_gate = np.zeros(len(stage_test), dtype=bool)
        best = {
            "stage_branch": branch["name"],
            "stage_config": stage_cfg,
            "router_name": "none",
            "router_alpha": 0.0,
            "router_threshold": 1.01,
            "router_max_change_rate": 0.0,
            "score": float(stage_cfg["score"]),
            "overall_acc": float(stage_cfg["overall_acc"]),
            "negative_acc": float(stage_cfg["negative_acc"]),
            "edge_low_acc": float(stage_cfg["edge_low_acc"]),
            "transition_acc": float(stage_cfg["transition_acc"]),
            "high_acc": float(stage_cfg["high_acc"]),
            "change_rate": 0.0,
        }
        used_router = False

    clean_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in all_rows] if all_rows else [best]
    save_csv(relpath("results", f"{suffix}_search_top.csv"), clean_rows, args.save_top_records)

    final_m = base.metrics_from_probs(final_test, yt, stest)
    stage_m = base.metrics_from_probs(stage_test, yt, stest)
    diag = base.switch_diagnostics(stage_test, final_test, router_gate, router_gate, np.full(len(router_gate), best.get("router_alpha", 0.0), dtype=np.float32), stest)

    selected = {
        "best": {k: v for k, v in best.items() if not k.startswith("_")},
        "used_router": used_router,
        "router_records": router_records,
        "blind_cqi_val_bin_acc": q_acc,
        "soup_prob_cache": args.soup_prob_cache,
        "fourier_oof_cache": args.fourier_oof_cache,
        "gamc_oof_cache": args.gamc_oof_cache,
        "cvtrn_oof_cache": args.cvtrn_oof_cache,
        "hcs_precision_cache": args.hcs_precision_cache,
        "protocol": "Train-OOF HCS rescue-risk router; validation-selected thresholds; test used once.",
    }
    with open(relpath("results", f"{suffix}_selected_config.json"), "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 144)
    print("Final test report")
    print("=" * 144)
    base.print_metrics_line("Main Fourier Test", base.metrics_from_probs(main_test, yt, stest))
    base.print_metrics_line("CV-TRN Test", base.metrics_from_probs(cv_test, yt, stest))
    base.print_metrics_line("GAMC-tree Test", base.metrics_from_probs(gamc["test_prob"], yt, stest))
    base.print_metrics_line("HCS precision Test", base.metrics_from_probs(hcs_test, yt, stest))
    base.print_metrics_line("Stage-1 Test", stage_m)
    base.print_metrics_line("Rescue-risk router Test", final_m)
    print("-" * 144)
    main_m = base.metrics_from_probs(main_test, yt, stest)
    print(f"Delta vs main Fourier overall:    {final_m['overall_acc'] - main_m['overall_acc']:+.4f} pp")
    print(f"Delta vs main Fourier negative:   {final_m['negative_acc'] - main_m['negative_acc']:+.4f} pp")
    print(f"Delta vs main Fourier edge:       {final_m['edge_low_acc'] - main_m['edge_low_acc']:+.4f} pp")
    print(f"Delta vs main Fourier transition: {final_m['transition_acc'] - main_m['transition_acc']:+.4f} pp")
    print(f"Delta vs main Fourier high:       {final_m['high_acc'] - main_m['high_acc']:+.4f} pp")
    print(f"Delta vs stage-1 overall:         {final_m['overall_acc'] - stage_m['overall_acc']:+.4f} pp")
    print(f"Router diagnostics: {diag}")
    print("=" * 144)

    pred = final_m["pred"].astype(np.int64)
    np.savez_compressed(
        relpath("results", f"{suffix}_predictions.npz"),
        labels=yt.astype(np.int64),
        snrs=stest.astype(np.int32),
        pred=pred,
        final_prob=final_test.astype(np.float32),
        stage_prob=stage_test.astype(np.float32),
        hcs_prob=hcs_test.astype(np.float32),
        router_gate=router_gate.astype(bool),
        mod_classes=soup.get("mod_classes", np.asarray(common.DEFAULT_MOD_CLASSES)),
    )
    curve_path = relpath("results", f"accuracy_vs_snr_{suffix}.png")
    base.plot_curve(final_m["by_snr"], curve_path, "Accuracy vs SNR: HCS Rescue-Risk Router")
    print(f"[*] SNR curve saved: {curve_path}")
    for snr_value in args.cm_snrs:
        cm_path = relpath("results", f"confusion_matrix_{snr_value}dB_{suffix}.png")
        acc = base.plot_cm_at_snr(
            yt,
            pred,
            stest,
            soup.get("mod_classes", np.asarray(common.DEFAULT_MOD_CLASSES)),
            snr_value,
            cm_path,
            "HCS Rescue-Risk Router",
        )
        print(f"[*] {snr_value} dB confusion matrix saved: {cm_path} (Acc={acc:.2f}%)")
    print(f"[*] CSV saved: {relpath('results', f'{suffix}_search_top.csv')}")
    print(f"[*] Selected config saved: {relpath('results', f'{suffix}_selected_config.json')}")
    print(f"[*] Predictions saved: {relpath('results', f'{suffix}_predictions.npz')}")


if __name__ == "__main__":
    main()
