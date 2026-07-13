import argparse
import copy
import csv
import json
import os

import numpy as np
from xgboost import XGBClassifier

import evaluate_greedy_soup_gamc_protected_residual_fusion as base
import evaluate_fourier_gamc_multi_cvtrn_blind_quality_xgb_fusion as bq
import evaluate_fourier_gamc_cvtrn_train_oof_meta_fusion as oof
import evaluate_fourier_gamc_cvtrn_stacked_residual_fusion as st
import evaluate_fourier_oof_gamc_cvtrn_extraaux_residual_meta_fusion as extra_eval
import evaluate_fourier_oof_gamc_cvtrn_residual_meta_fusion as orig
import evaluate_fourier_oof_hcs_rescue_risk_router_2016 as rr
import train_cv_trn_aux_2016 as common
from model_cache_utils import fit_or_load_estimator


MIDLOW_SNRS = np.array([-14, -12], dtype=np.int32)
WIDE_TRANSITION_SNRS = np.array([-14, -12, -10, -8, -6, -4, -2], dtype=np.int32)


def relpath(*parts):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Train transition-specific rescue routers for a pairwise confusion auxiliary. "
            "Each router is trained only on train-split OOF candidates for one stage->aux transition."
        )
    )
    p.add_argument("--split_seed", type=int, default=1)
    p.add_argument("--random_state", type=int, default=2026)
    p.add_argument("--soup_prob_cache", type=str, default=relpath("results", "greedy_soup_identity_valtest_probs_for_gamc_fusion.npz"))
    p.add_argument("--fourier_oof_cache", type=str, default=relpath("results", "fourier_main_oof_merge_mseed78_79_f3e220_bs128_split1_trainvaltest_probs_for_meta.npz"))
    p.add_argument("--gamc_oof_cache", type=str, default=relpath("results", "gamc_oof_tree_split1_trainvaltest_probs_for_meta.npz"))
    p.add_argument("--cvtrn_oof_cache", type=str, default=relpath("results", "cv_trn_aux_v2_oof_mseed41_f3e240_split1_trainvaltest_probs_for_meta.npz"))
    p.add_argument("--hcs_precision_cache", type=str, default=relpath("results", "hcs_precision_analog_aux_split1_trainvaltest_probs_for_meta.npz"))
    p.add_argument("--pairwise_cache", type=str, default=relpath("results", "pairwise_confusion_v2_aux_split1_trainvaltest_probs_for_meta.npz"))
    p.add_argument("--hcs_feature_cache", type=str, default=relpath("feature_cache", "hcs_lite_features_v1.npz"))
    p.add_argument("--gamc_feature_cache", type=str, default=relpath("feature_cache", "gamc_lite_features_v3_graph_xgb.npz"))
    p.add_argument(
        "--router_stat_features_per_source",
        type=int,
        default=0,
        help="Keep this many highest-variance unlabeled columns per HCS/GAMC cache; 0 keeps all columns.",
    )
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
    p.add_argument("--main_display_name", type=str, default="Main Fourier")
    p.add_argument("--cv_display_name", type=str, default="CV-TRN")
    p.add_argument(
        "--include_cv_mirror_stage",
        action="store_true",
        help=(
            "Add extra stage-1 meta branches where CV-TRN probability slots mirror the "
            "main Fourier probabilities. This costs no extra neural inference and lets "
            "validation choose no-CV behavior in SNR regions where single CV hurts."
        ),
    )
    p.add_argument(
        "--disable_cvtrn",
        action="store_true",
        help=(
            "Ablation mode: do not use any CV-TRN branch information. "
            "The CV-TRN probability slots are filled with the Fourier main probabilities "
            "so the stage/meta/router code path stays identical but receives no independent CV-TRN signal."
        ),
    )
    p.add_argument(
        "--cvtrn_infer_from_valtest_only",
        action="store_true",
        help=(
            "Use CV-TRN OOF probabilities only for train-split meta training, and use only "
            "--cvtrn_valtest_caches for validation/test inference. This gives a cleaner "
            "deployment-complexity estimate for single-seed or small-soup CV-TRN branches."
        ),
    )
    p.add_argument("--use_fourier_oof_infer", action="store_true")
    p.add_argument("--output_suffix", type=str, default="fourier_oof_pairwise_transition_router_split1")
    p.add_argument(
        "--defer_test_report",
        action="store_true",
        help="Write test probabilities without evaluating test labels; a later locked final stage reports them once.",
    )
    p.add_argument(
        "--distill_teacher_cache",
        type=str,
        default="",
        help=(
            "Optional output .npz in GainDistillDataset format.  It stores train-split "
            "OOF teacher targets plus val/test diagnostics for single compressed-student distillation."
        ),
    )
    p.add_argument(
        "--distill_teacher_mode",
        type=str,
        default="label_corrected",
        choices=["label_corrected", "soft_gate", "soft"],
        help=(
            "How to convert the selected router probabilities into train targets. "
            "label_corrected is the original strong label-mix target; soft_gate keeps "
            "teacher dark knowledge and heavily downweights KD where the OOF teacher is wrong; "
            "soft uses the router probabilities directly with confidence weighting."
        ),
    )

    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--stage_oof_folds", type=int, default=3)
    p.add_argument("--stage_models", nargs="+", default=["xgb_d2_620", "xgb_d3_520", "xgb_d4_400", "et_depth20"])
    p.add_argument(
        "--stage_aux_sources",
        nargs="+",
        choices=["hcs", "pairwise"],
        default=["hcs"],
        help="OOF auxiliary probability blocks supplied to the Stage-1 meta model.",
    )
    p.add_argument("--stage_estimator_scale", type=float, default=1.0)
    p.add_argument("--stage_min_val_overall", type=float, default=65.95)
    p.add_argument("--top_stage_configs", type=int, default=6)
    p.add_argument("--blend_alphas", type=float, nargs="+", default=[0.50, 0.65, 0.80])
    p.add_argument("--meta_conf_thresholds", type=float, nargs="+", default=[0.00, 0.25, 0.35, 0.45, 0.55])
    p.add_argument("--advantage_thresholds", type=float, nargs="+", default=[-0.20, -0.10, 0.00, 0.05, 0.10])
    p.add_argument("--max_change_rates", type=float, nargs="+", default=[6.0, 8.0, 10.0, 12.0])
    p.add_argument("--hard_keep_conf", type=float, default=0.93)
    p.add_argument("--hard_keep_margin", type=float, default=0.70)

    p.add_argument("--router_estimators", type=int, default=360)
    p.add_argument("--router_depths", type=int, nargs="+", default=[2, 3])
    p.add_argument("--router_learning_rate", type=float, default=0.035)
    p.add_argument("--router_thresholds", type=float, nargs="+", default=[0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75])
    p.add_argument("--router_max_change_rates", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.35, 0.50, 0.75])
    p.add_argument("--router_alphas", type=float, nargs="+", default=[1.00, 0.80, 0.65])
    p.add_argument("--router_min_change_rate", type=float, default=0.02)
    p.add_argument("--router_min_stage_overall_gain", type=float, default=0.0)
    p.add_argument(
        "--router_min_global_stage_overall_gain",
        type=float,
        default=None,
        help=(
            "Require the selected router validation overall to beat the best eligible Stage-1 validation "
            "overall by this many percentage points; otherwise keep the best Stage-1 branch."
        ),
    )
    p.add_argument("--allow_router_stage_drop", action="store_true")
    p.add_argument("--require_pairwise_gate", action="store_true")
    p.add_argument(
        "--router_aux_sources",
        nargs="+",
        choices=["pairwise", "cvtrn", "hcs", "gamc", "main"],
        default=["pairwise"],
        help="OOF probability sources that may propose a validation-selected correction.",
    )
    p.add_argument(
        "--use_pairwise_raw_router_features",
        action="store_true",
        help="Append raw per-pair specialist OOF scores when the pairwise cache provides them.",
    )
    p.add_argument("--max_transitions_per_aux", type=int, default=12)

    p.add_argument("--min_transition_count", type=int, default=80)
    p.add_argument("--min_transition_pos", type=int, default=12)
    p.add_argument("--min_transition_precision", type=float, default=0.48)
    p.add_argument("--transition_max_train_harm_rate", type=float, default=0.58)
    p.add_argument("--rescue_weight", type=float, default=36.0)
    p.add_argument("--harm_weight", type=float, default=28.0)
    p.add_argument("--other_weight", type=float, default=1.2)

    p.add_argument("--score_overall_weight", type=float, default=1.0)
    p.add_argument("--score_negative_gain_weight", type=float, default=0.020)
    p.add_argument("--score_edge_gain_weight", type=float, default=0.020)
    p.add_argument("--score_transition_gain_weight", type=float, default=0.010)
    p.add_argument(
        "--score_midlow_gain_weight",
        type=float,
        default=0.0,
        help="Reward validation gain on the -14/-12 dB band over the stage model.",
    )
    p.add_argument(
        "--score_wide_transition_gain_weight",
        type=float,
        default=0.0,
        help="Reward validation gain on the wider -14..-2 dB transition band over the stage model.",
    )
    p.add_argument("--score_high_penalty", type=float, default=3.0)
    p.add_argument("--high_tolerance", type=float, default=0.05)
    p.add_argument("--score_changed_high_penalty", type=float, default=0.016)
    p.add_argument("--score_changed_nonultra_penalty", type=float, default=0.006)

    p.add_argument("--xgb_jobs", type=int, default=-1)
    p.add_argument("--xgb_device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--model_cache_dir", type=str, default="")
    p.add_argument("--reuse_models", action="store_true")
    p.add_argument("--quality_estimators", type=int, default=280)
    p.add_argument("--quality_max_depth", type=int, default=4)
    p.add_argument("--quality_learning_rate", type=float, default=0.04)
    p.add_argument("--quality_subsample", type=float, default=0.90)
    p.add_argument("--quality_colsample", type=float, default=0.85)
    p.add_argument("--quality_chunk_size", type=int, default=32768)
    p.add_argument(
        "--quality_extra_feature_caches",
        type=str,
        nargs="*",
        default=[],
        help=(
            "Optional blind feature caches appended to CQI inputs, e.g. HCS/GAMC graph features. "
            "These features must be computed without validation/test labels or true SNR at inference."
        ),
    )
    p.add_argument(
        "--router_quality_extra_feature_caches",
        type=str,
        nargs="*",
        default=[],
        help=(
            "Optional blind feature caches used only for the transition router CQI. "
            "Stage-1 meta features keep the baseline CQI, reducing validation overfit from richer CQI inputs."
        ),
    )
    p.add_argument("--router_quality_estimators", type=int, default=0)
    p.add_argument("--router_quality_max_depth", type=int, default=0)
    p.add_argument("--router_quality_learning_rate", type=float, default=0.0)
    p.add_argument("--router_quality_subsample", type=float, default=0.0)
    p.add_argument("--router_quality_colsample", type=float, default=0.0)
    p.add_argument(
        "--router_quality_expected_mins",
        type=float,
        nargs="*",
        default=[],
        help=(
            "Optional blind CQI guard for transition router search. Values are lower bounds on "
            "expected SNR computed from CQI probabilities with centers [-18,-11,-4,4,14]. "
            "When used with --router_quality_expected_maxes, router gates only samples inside "
            "one searched expected-SNR interval."
        ),
    )
    p.add_argument(
        "--router_quality_expected_maxes",
        type=float,
        nargs="*",
        default=[],
        help="Upper bounds paired by Cartesian search with --router_quality_expected_mins.",
    )
    p.add_argument("--data_path", type=str, default=relpath("raw_data", "RML2016.10a_dict.pkl"))
    p.add_argument("--cache_dir", type=str, default=relpath("feature_cache"))
    p.add_argument("--alignment_cache", type=str, default=relpath("results", "greedy_soup_identity_valtest_probs_for_gamc_fusion.npz"))
    p.add_argument("--skip_alignment_check", action="store_true")
    p.add_argument("--save_top_records", type=int, default=240)
    p.add_argument("--cm_snrs", type=int, nargs="+", default=[0, -6])
    return p.parse_args()


def norm(prob):
    return oof.norm(prob).astype(np.float32)


def top_margin(prob):
    part = np.partition(prob, -2, axis=1)
    return part[:, -1] - part[:, -2]


def entropy(prob):
    p = np.clip(norm(prob), 1e-8, 1.0)
    return -(p * np.log(p)).sum(1)


def binary_proba(model, x):
    p = model.predict_proba(x)
    out = np.zeros(len(x), dtype=np.float32)
    for i, cls in enumerate(getattr(model, "classes_", np.asarray([0, 1]))):
        if int(cls) == 1:
            out = p[:, i].astype(np.float32)
            break
    return out


def transition_model(args, depth, seed):
    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=int(args.router_estimators),
        max_depth=int(depth),
        learning_rate=float(args.router_learning_rate),
        subsample=0.88,
        colsample_bytree=0.90,
        reg_lambda=4.5,
        reg_alpha=0.14,
        min_child_weight=3.0,
        tree_method="hist",
        device=str(getattr(args, "xgb_device", "cpu")),
        eval_metric="logloss",
        n_jobs=int(args.xgb_jobs),
        random_state=int(seed),
        verbosity=0,
    )


def load_stat_features(args, train_idx, val_idx, test_idx):
    train_blocks, val_blocks, test_blocks = [], [], []
    for name, path in (("HCS", args.hcs_feature_cache), ("GAMC", args.gamc_feature_cache)):
        if not path or not os.path.exists(path):
            print(f"[!] {name} stat feature cache not found; transition router will skip this block: {path}")
            continue
        z = np.load(path, allow_pickle=True)
        block = z["features"].astype(np.float32)
        max_idx = int(max(np.max(train_idx), np.max(val_idx), np.max(test_idx)))
        if len(block) <= max_idx:
            raise ValueError(f"{name} stat feature cache is too short: {path} len={len(block)} max_idx={max_idx}")
        keep = int(getattr(args, "router_stat_features_per_source", 0))
        if keep > 0 and block.ndim == 2 and block.shape[1] > keep:
            sample_ids = np.asarray(train_idx, dtype=np.int64)[:: max(1, len(train_idx) // 20000)][:20000]
            variance = np.var(block[sample_ids].astype(np.float64), axis=0)
            cols = np.argsort(variance)[-keep:]
            cols.sort()
            block = block[:, cols]
            print(f"[*] Reduced {name} stat features by unlabeled variance: kept {len(cols)} columns")
        train_blocks.append(block[train_idx])
        val_blocks.append(block[val_idx])
        test_blocks.append(block[test_idx])
        print(f"[*] Loaded {name} stat feature cache: {path} selected_shape={block.shape}")
        del block
    if not train_blocks:
        print("[!] No stat feature caches loaded; transition router will use probability/CQI features only.")
        return (
            np.zeros((len(train_idx), 0), dtype=np.float32),
            np.zeros((len(val_idx), 0), dtype=np.float32),
            np.zeros((len(test_idx), 0), dtype=np.float32),
        )
    return (
        np.concatenate(train_blocks, axis=1).astype(np.float32),
        np.concatenate(val_blocks, axis=1).astype(np.float32),
        np.concatenate(test_blocks, axis=1).astype(np.float32),
    )


def transition_features(stage, aux, main, cv, gamc, hcs, qprob, a, b, stat_x=None, raw_pair_probs=None):
    probs = [norm(stage), norm(aux), norm(main), norm(cv), norm(gamc), norm(hcs)]
    parts = []
    for p in probs:
        pred = p.argmax(1)
        parts.append(p[:, [a, b]])
        parts.append(np.stack([p.max(1), top_margin(p), entropy(p), (pred == a), (pred == b)], axis=1).astype(np.float32))
    s, u = probs[0], probs[1]
    parts.append(
        np.stack(
            [
                u[:, b] - s[:, b],
                u[:, a] - s[:, a],
                (u[:, b] - u[:, a]) - (s[:, b] - s[:, a]),
                s[:, a] + s[:, b],
                u[:, a] + u[:, b],
                np.abs(s - u).sum(1),
                (s * u).sum(1),
            ],
            axis=1,
        ).astype(np.float32)
    )
    parts.append(bq.quality_probability_features(qprob).astype(np.float32))
    if raw_pair_probs is not None:
        raw = np.asarray(raw_pair_probs, dtype=np.float32)
        if raw.ndim != 3 or raw.shape[2] != 2:
            raise ValueError(f"raw pairwise probabilities must have shape [pairs,samples,2], got {raw.shape}")
        p1 = raw[:, :, 1].T
        parts.append(p1.astype(np.float32))
        parts.append((np.abs(p1 - 0.5) * 2.0).astype(np.float32))
    if stat_x is not None:
        parts.append(np.asarray(stat_x, dtype=np.float32))
    x = np.concatenate(parts, axis=1)
    return np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)


def candidate_transitions(stage_train, aux_train, labels, args):
    sp = norm(stage_train).argmax(1)
    ap = norm(aux_train).argmax(1)
    y = np.asarray(labels, dtype=np.int64)
    rows = []
    for a in range(base.NUM_CLASSES):
        for b in range(base.NUM_CLASSES):
            if a == b:
                continue
            m = (sp == a) & (ap == b)
            count = int(m.sum())
            if count < int(args.min_transition_count):
                continue
            pos = int((m & (y == b)).sum())
            harm = int((m & (y == a)).sum())
            pos_harm = pos + harm
            precision = pos / max(1, pos_harm)
            harm_rate = harm / max(1, pos_harm)
            if pos < int(args.min_transition_pos):
                continue
            if precision < float(args.min_transition_precision):
                continue
            if harm_rate > float(args.transition_max_train_harm_rate):
                continue
            rows.append(
                {
                    "from": int(a),
                    "to": int(b),
                    "count": count,
                    "pos": pos,
                    "harm": harm,
                    "precision": float(precision),
                    "harm_rate": float(harm_rate),
                }
            )
    rows.sort(key=lambda r: (r["pos"] - r["harm"], r["precision"], r["count"]), reverse=True)
    return rows


def limit_mask(mask, priority, max_rate):
    return rr.limit_mask(mask, priority, max_rate)


def quality_expected_snr(qprob):
    qprob = norm(qprob)
    centers = np.asarray([-18.0, -11.0, -4.0, 4.0, 14.0], dtype=np.float32)
    return (qprob @ centers).astype(np.float32)


def quality_guard_mask(qprob, cfg):
    if qprob is None:
        return None
    qmin = cfg.get("router_quality_expected_min", None)
    qmax = cfg.get("router_quality_expected_max", None)
    if qmin is None and qmax is None:
        return None
    exp = quality_expected_snr(qprob)
    mask = np.ones(len(exp), dtype=bool)
    if qmin is not None:
        mask &= exp >= float(qmin)
    if qmax is not None:
        mask &= exp <= float(qmax)
    return mask


def router_quality_intervals(args):
    mins = list(getattr(args, "router_quality_expected_mins", []) or [])
    maxes = list(getattr(args, "router_quality_expected_maxes", []) or [])
    if not mins and not maxes:
        return [(None, None)]
    if not mins:
        mins = [None]
    if not maxes:
        maxes = [None]
    out = []
    for qmin in mins:
        for qmax in maxes:
            if qmin is not None and qmax is not None and float(qmin) > float(qmax):
                continue
            out.append((qmin, qmax))
    return out or [(None, None)]


def apply_router(stage_prob, aux_prob, score, cfg, quality_prob=None):
    stage = norm(stage_prob)
    aux = norm(aux_prob)
    sp = stage.argmax(1)
    ap = aux.argmax(1)
    raw = (ap != sp) & (score >= float(cfg["router_threshold"]))
    qmask = quality_guard_mask(quality_prob, cfg)
    if qmask is not None:
        raw &= qmask
    gate = limit_mask(raw, score, cfg["router_max_change_rate"])
    out = np.array(stage, copy=True)
    alpha = float(cfg["router_alpha"])
    if np.any(gate):
        out[gate] = (1.0 - alpha) * stage[gate] + alpha * aux[gate]
    return norm(out).astype(np.float32), gate


def label_corrected_teacher(prob, labels, snrs):
    prob = norm(prob)
    labels = np.asarray(labels, dtype=np.int64)
    snrs = np.asarray(snrs, dtype=np.int32)
    onehot = np.eye(prob.shape[1], dtype=np.float32)[labels]
    correct = prob.argmax(1) == labels
    mix = np.where(correct, 0.04, 0.48).astype(np.float32)[:, None]
    teacher = norm((1.0 - mix) * prob + mix * onehot)

    kd_weight = np.where(correct, 1.00, 0.35).astype(np.float32)
    ce_weight = np.where(correct, 0.90, 1.25).astype(np.float32)
    snr_factor = np.ones(len(labels), dtype=np.float32)
    snr_factor[snrs < 0] *= 1.10
    snr_factor[np.isin(snrs, [-10, -8, -6, -4, -2])] *= 1.08
    snr_factor[snrs <= -14] *= 0.75
    return teacher.astype(np.float32), (kd_weight * snr_factor).astype(np.float32), (ce_weight * snr_factor).astype(np.float32)


def soft_gate_teacher(prob, labels, snrs):
    prob = norm(prob)
    labels = np.asarray(labels, dtype=np.int64)
    snrs = np.asarray(snrs, dtype=np.int32)
    correct = prob.argmax(1) == labels
    onehot = np.eye(prob.shape[1], dtype=np.float32)[labels]

    teacher = np.array(prob, copy=True)
    if np.any(correct):
        teacher[correct] = 0.98 * prob[correct] + 0.02 * onehot[correct]
    teacher = norm(teacher)

    kd_weight = np.where(correct, 1.00, 0.05).astype(np.float32)
    ce_weight = np.where(correct, 0.88, 1.20).astype(np.float32)
    snr_factor = np.ones(len(labels), dtype=np.float32)
    snr_factor[snrs < 0] *= 1.08
    snr_factor[np.isin(snrs, [-10, -8, -6, -4, -2])] *= 1.08
    snr_factor[snrs <= -14] *= 0.70
    return teacher.astype(np.float32), (kd_weight * snr_factor).astype(np.float32), (ce_weight * snr_factor).astype(np.float32)


def soft_teacher(prob, labels, snrs):
    prob = norm(prob)
    labels = np.asarray(labels, dtype=np.int64)
    snrs = np.asarray(snrs, dtype=np.int32)
    conf = prob.max(1).astype(np.float32)
    correct = prob.argmax(1) == labels
    kd_weight = (0.20 + 0.90 * conf).astype(np.float32)
    kd_weight[~correct] *= 0.25
    ce_weight = np.where(correct, 0.92, 1.15).astype(np.float32)
    snr_factor = np.ones(len(labels), dtype=np.float32)
    snr_factor[snrs < 0] *= 1.05
    snr_factor[np.isin(snrs, [-10, -8, -6, -4, -2])] *= 1.05
    snr_factor[snrs <= -14] *= 0.75
    return prob.astype(np.float32), (kd_weight * snr_factor).astype(np.float32), (ce_weight * snr_factor).astype(np.float32)


def make_distill_teacher(prob, labels, snrs, mode):
    if mode == "label_corrected":
        return label_corrected_teacher(prob, labels, snrs)
    if mode == "soft_gate":
        return soft_gate_teacher(prob, labels, snrs)
    if mode == "soft":
        return soft_teacher(prob, labels, snrs)
    raise ValueError(f"Unknown distill teacher mode: {mode}")


def save_distill_teacher_cache(path, train_prob, val_prob, test_prob, labels_train, snrs_train, labels_val, snrs_val, labels_test, snrs_test, selected, mode):
    teacher_prob, kd_weight, ce_weight = make_distill_teacher(train_prob, labels_train, snrs_train, mode)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        teacher_prob=teacher_prob.astype(np.float32),
        kd_weight=kd_weight.astype(np.float32),
        ce_weight=ce_weight.astype(np.float32),
        source=np.full(len(labels_train), "transition_router_teacher", dtype="<U32"),
        labels_train=np.asarray(labels_train, dtype=np.int64),
        snrs_train=np.asarray(snrs_train, dtype=np.int32),
        base_prob=norm(train_prob).astype(np.float32),
        teacher_val=norm(val_prob).astype(np.float32),
        teacher_test=norm(test_prob).astype(np.float32),
        labels_val=np.asarray(labels_val, dtype=np.int64),
        snrs_val=np.asarray(snrs_val, dtype=np.int32),
        labels_test=np.asarray(labels_test, dtype=np.int64),
        snrs_test=np.asarray(snrs_test, dtype=np.int32),
        selected_config=np.asarray([json.dumps(selected, ensure_ascii=False)]),
        diagnostics=np.asarray(
            [
                json.dumps(
                    {
                        "teacher_mode": str(mode),
                        "train_teacher_argmax_acc": float((norm(train_prob).argmax(1) == labels_train).mean() * 100.0),
                        "train_export_argmax_acc": float((teacher_prob.argmax(1) == labels_train).mean() * 100.0),
                        "val_teacher_argmax_acc": float((norm(val_prob).argmax(1) == labels_val).mean() * 100.0),
                        "test_teacher_argmax_acc_diagnostic_only": float((norm(test_prob).argmax(1) == labels_test).mean() * 100.0),
                    },
                    ensure_ascii=False,
                )
            ]
        ),
        protocol=np.asarray(
            [
                f"Transition-router teacher for compressed-student distillation; mode={mode}; train targets are OOF/cross-fitted, validation-selected, test labels diagnostic only."
            ]
        ),
    )
    print(f"[*] Distillation teacher cache saved: {path}")


def snr_band_acc(prob, labels, snrs, band):
    mask = np.isin(np.asarray(snrs, dtype=np.int32), band)
    if not np.any(mask):
        return 0.0
    pred = np.asarray(prob).argmax(1)
    return float((pred[mask] == np.asarray(labels)[mask]).mean() * 100.0)


def search_router(stage_val, aux_val, score_val, yv, sv, stage_val_m, args, router_name, quality_val=None):
    rows = []
    min_overall = float(stage_val_m["overall_acc"]) + float(args.router_min_stage_overall_gain)
    stage_midlow_acc = snr_band_acc(stage_val, yv, sv, MIDLOW_SNRS)
    stage_wide_transition_acc = snr_band_acc(stage_val, yv, sv, WIDE_TRANSITION_SNRS)
    for qmin, qmax in router_quality_intervals(args):
        for alpha in args.router_alphas:
            for thr in args.router_thresholds:
                for max_rate in args.router_max_change_rates:
                    cfg = {
                        "router_name": router_name,
                        "router_alpha": float(alpha),
                        "router_threshold": float(thr),
                        "router_max_change_rate": float(max_rate),
                    }
                    if qmin is not None:
                        cfg["router_quality_expected_min"] = float(qmin)
                    if qmax is not None:
                        cfg["router_quality_expected_max"] = float(qmax)
                    out, gate = apply_router(stage_val, aux_val, score_val, cfg, quality_val)
                    change_rate = float(gate.mean() * 100.0)
                    if change_rate < float(args.router_min_change_rate):
                        continue
                    m = base.metrics_from_probs(out, yv, sv)
                    if (not args.allow_router_stage_drop) and m["overall_acc"] < min_overall:
                        continue
                    midlow_acc = snr_band_acc(out, yv, sv, MIDLOW_SNRS)
                    wide_transition_acc = snr_band_acc(out, yv, sv, WIDE_TRANSITION_SNRS)
                    high_drop = max(0.0, stage_val_m["high_acc"] - m["high_acc"] - float(args.high_tolerance))
                    score = (
                        float(args.score_overall_weight) * m["overall_acc"]
                        + float(args.score_negative_gain_weight) * (m["negative_acc"] - stage_val_m["negative_acc"])
                        + float(args.score_edge_gain_weight) * (m["edge_low_acc"] - stage_val_m["edge_low_acc"])
                        + float(args.score_transition_gain_weight) * (m["transition_acc"] - stage_val_m["transition_acc"])
                        + float(args.score_midlow_gain_weight) * (midlow_acc - stage_midlow_acc)
                        + float(args.score_wide_transition_gain_weight)
                        * (wide_transition_acc - stage_wide_transition_acc)
                        - float(args.score_high_penalty) * high_drop
                    )
                    diag = base.switch_diagnostics(
                        stage_val,
                        out,
                        gate,
                        gate,
                        np.full(len(gate), alpha, dtype=np.float32),
                        sv,
                    )
                    score -= float(args.score_changed_high_penalty) * diag["changed_high_rate"]
                    score -= float(args.score_changed_nonultra_penalty) * diag["changed_nonultra_rate"]
                    rows.append(
                        {
                            **cfg,
                            "score": float(score),
                            "overall_acc": float(m["overall_acc"]),
                            "negative_acc": float(m["negative_acc"]),
                            "edge_low_acc": float(m["edge_low_acc"]),
                            "transition_acc": float(m["transition_acc"]),
                            "midlow_acc": float(midlow_acc),
                            "wide_transition_acc": float(wide_transition_acc),
                            "midlow_gain_from_stage": float(midlow_acc - stage_midlow_acc),
                            "wide_transition_gain_from_stage": float(
                                wide_transition_acc - stage_wide_transition_acc
                            ),
                            "high_acc": float(m["high_acc"]),
                            "change_rate": change_rate,
                        }
                    )
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


def save_csv(path, rows, limit):
    if not rows:
        return
    keys = sorted(set(k for r in rows for k in r.keys() if not k.startswith("_")))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows[:limit]:
            w.writerow({k: r.get(k) for k in keys})
    print(f"[*] CSV saved: {path}")


def print_top(rows, n=20):
    print("\nTop transition-router validation configs")
    for i, r in enumerate(rows[:n], 1):
        qguard = ""
        if "router_quality_expected_min" in r or "router_quality_expected_max" in r:
            qguard = (
                f" qexp=[{r.get('router_quality_expected_min', '-inf')},"
                f"{r.get('router_quality_expected_max', 'inf')}]"
            )
        print(
            f"{i:02d}. score={r['score']:.3f} | overall={r['overall_acc']:.3f}% | "
            f"neg={r['negative_acc']:.3f}% | edge={r['edge_low_acc']:.3f}% | "
            f"trans={r['transition_acc']:.3f}% | high={r['high_acc']:.3f}% | "
            f"change={r['change_rate']:.2f}% | stage={r['stage_branch']} | "
            f"a={r['router_alpha']} thr={r['router_threshold']} max={r['router_max_change_rate']}{qguard} | "
            f"transitions={r['transition_count']}"
        )


def main():
    args = parse_args()
    os.makedirs(relpath("results"), exist_ok=True)
    suffix = args.output_suffix
    print("=" * 144)
    print("Train-OOF pairwise transition-specific rescue router")
    print("=" * 144)
    print("Academic protocol:")
    print(f"  - {args.main_display_name} remains the default/main probability source.")
    print("  - Stage-1 meta train predictions are cross-fitted on train split only.")
    print("  - Transition routers are trained only on train-split OOF transition candidates.")
    print("  - Validation labels select router thresholds/blends only.")
    print("  - Test labels are used only once for the final report.")
    print("  - True SNR is never used as an inference feature.")

    soup = orig.load_npz(args.soup_prob_cache)
    fourier = orig.load_npz(args.fourier_oof_cache)
    gamc = orig.load_npz(args.gamc_oof_cache)
    cv = None if args.disable_cvtrn else orig.load_npz(args.cvtrn_oof_cache)
    hcs = orig.load_npz(args.hcs_precision_cache)
    pairwise = orig.load_npz(args.pairwise_cache)

    for key in ("labels_train", "snrs_train"):
        aligned_train = [(gamc, "GAMC"), (hcs, "HCS"), (pairwise, "Pairwise")]
        if cv is not None:
            aligned_train.append((cv, args.cv_display_name))
        for z, name in aligned_train:
            orig.assert_same(fourier, z, key, key, f"{args.main_display_name} OOF vs {name}")
    for key in ("labels_val", "snrs_val", "labels_test", "snrs_test"):
        aligned_eval = [(fourier, f"{args.main_display_name} OOF"), (gamc, "GAMC"), (hcs, "HCS"), (pairwise, "Pairwise")]
        if cv is not None:
            aligned_eval.append((cv, args.cv_display_name))
        for z, name in aligned_eval:
            orig.assert_same(soup, z, key, key, f"{args.main_display_name} default cache vs {name}")
    print("[*] Alignment check passed for all OOF caches.")

    args.model_cache_source_paths = [
        args.soup_prob_cache,
        args.fourier_oof_cache,
        args.gamc_oof_cache,
        args.hcs_precision_cache,
        args.pairwise_cache,
        *([] if args.disable_cvtrn else [args.cvtrn_oof_cache]),
    ]

    y_train = fourier["labels_train"].astype(np.int64)
    snrs_train = fourier["snrs_train"].astype(np.int32)
    yv = soup["labels_val"].astype(np.int64)
    sv = soup["snrs_val"].astype(np.int32)
    yt = soup["labels_test"].astype(np.int64)
    stest = soup["snrs_test"].astype(np.int32)

    if args.disable_cvtrn:
        print(f"[*] Ablation mode: {args.cv_display_name} disabled; auxiliary probability slots mirror the main branch.")
        cv_train = fourier["train_prob"].astype(np.float32)
        cv_val = soup["val_prob"].astype(np.float32)
        cv_test = soup["test_prob"].astype(np.float32)
    else:
        cv_train = cv["train_prob"].astype(np.float32)
        cv_val_list = []
        cv_test_list = []
        if not args.cvtrn_infer_from_valtest_only:
            cv_val_list.append(cv["val_prob"].astype(np.float32))
            cv_test_list.append(cv["test_prob"].astype(np.float32))
        if args.use_oof_cvtrn_only:
            print(f"[*] Strict mode: using only {args.cv_display_name} OOF fold-soup at inference.")
        elif args.cvtrn_infer_from_valtest_only:
            print(f"[*] {args.cv_display_name} inference mode: using only explicit val/test cache(s); OOF probabilities are train-only.")
            for path in args.cvtrn_valtest_caches:
                if not path or not os.path.exists(path):
                    print(f"[!] Explicit {args.cv_display_name} val/test cache skipped: {path}")
                    continue
                z = orig.load_npz(path)
                for key in ("labels_val", "snrs_val", "labels_test", "snrs_test"):
                    orig.assert_same(soup, z, key, key, f"Explicit {args.cv_display_name} {os.path.basename(path)}")
                cv_val_list.append(z["val_prob"].astype(np.float32))
                cv_test_list.append(z["test_prob"].astype(np.float32))
                print(f"[*] Added explicit inference {args.cv_display_name} cache: {path}")
        else:
            for path in args.cvtrn_valtest_caches:
                if not path or not os.path.exists(path):
                    print(f"[!] Optional {args.cv_display_name} val/test cache skipped: {path}")
                    continue
                z = orig.load_npz(path)
                for key in ("labels_val", "snrs_val", "labels_test", "snrs_test"):
                    orig.assert_same(soup, z, key, key, f"Optional {args.cv_display_name} {os.path.basename(path)}")
                cv_val_list.append(z["val_prob"].astype(np.float32))
                cv_test_list.append(z["test_prob"].astype(np.float32))
                print(f"[*] Added inference {args.cv_display_name} cache: {path}")
        if not cv_val_list or not cv_test_list:
            raise RuntimeError(f"No {args.cv_display_name} validation/test probabilities available for inference.")
        cv_val = orig.log_average(cv_val_list).astype(np.float32)
        cv_test = orig.log_average(cv_test_list).astype(np.float32)

    full_dataset = common.build_full_dataset(args)
    train_idx, val_idx, test_idx = common.make_aligned_split(full_dataset, args.split_seed)
    if not args.skip_alignment_check:
        common.check_alignment(full_dataset, val_idx, test_idx, args.alignment_cache)
    snrs_all = np.asarray(full_dataset.snrs, dtype=np.int32)
    q_train, q_val, q_test, q_acc = oof.build_quality_probs(args, train_idx, val_idx, test_idx, full_dataset, snrs_all)
    q_router_train, q_router_val, q_router_test, q_router_acc = q_train, q_val, q_test, q_acc
    if list(getattr(args, "router_quality_extra_feature_caches", []) or []):
        router_q_args = copy.copy(args)
        router_q_args.quality_extra_feature_caches = list(args.router_quality_extra_feature_caches)
        if int(getattr(args, "router_quality_estimators", 0)) > 0:
            router_q_args.quality_estimators = int(args.router_quality_estimators)
        if int(getattr(args, "router_quality_max_depth", 0)) > 0:
            router_q_args.quality_max_depth = int(args.router_quality_max_depth)
        if float(getattr(args, "router_quality_learning_rate", 0.0)) > 0:
            router_q_args.quality_learning_rate = float(args.router_quality_learning_rate)
        if float(getattr(args, "router_quality_subsample", 0.0)) > 0:
            router_q_args.quality_subsample = float(args.router_quality_subsample)
        if float(getattr(args, "router_quality_colsample", 0.0)) > 0:
            router_q_args.quality_colsample = float(args.router_quality_colsample)
        print("\n[*] Building router-only enhanced CQI; stage-1 keeps baseline CQI features.")
        q_router_train, q_router_val, q_router_test, q_router_acc = oof.build_quality_probs(
            router_q_args, train_idx, val_idx, test_idx, full_dataset, snrs_all
        )
    stat_train, stat_val, stat_test = load_stat_features(args, train_idx, val_idx, test_idx)

    main_train = fourier["train_prob"].astype(np.float32)
    main_val = fourier["val_prob"].astype(np.float32) if args.use_fourier_oof_infer else soup["val_prob"].astype(np.float32)
    main_test = fourier["test_prob"].astype(np.float32) if args.use_fourier_oof_infer else soup["test_prob"].astype(np.float32)
    hcs_train = hcs["train_prob"].astype(np.float32)
    hcs_val = hcs["val_prob"].astype(np.float32)
    hcs_test = hcs["test_prob"].astype(np.float32)
    pair_train = pairwise["train_prob"].astype(np.float32)
    pair_val = pairwise["val_prob"].astype(np.float32)
    pair_test = pairwise["test_prob"].astype(np.float32)
    pair_train_gate = pairwise["train_gate"].astype(bool) if "train_gate" in pairwise else np.ones(len(y_train), dtype=bool)
    pair_val_gate = pairwise["val_gate"].astype(bool) if "val_gate" in pairwise else np.ones(len(yv), dtype=bool)
    pair_test_gate = pairwise["test_gate"].astype(bool) if "test_gate" in pairwise else np.ones(len(yt), dtype=bool)
    raw_pair_train = raw_pair_val = raw_pair_test = None
    if args.use_pairwise_raw_router_features:
        raw_keys = ("train_pair_probs", "val_pair_probs", "test_pair_probs", "pair_classes")
        missing = [key for key in raw_keys if key not in pairwise]
        if missing:
            raise KeyError(
                "Pairwise raw router features requested, but the cache is missing "
                + ", ".join(missing)
                + ". Re-run train_pairwise_confusion_aux_10b.py with the updated writer."
            )
        raw_pair_train = pairwise["train_pair_probs"].astype(np.float32)
        raw_pair_val = pairwise["val_pair_probs"].astype(np.float32)
        raw_pair_test = pairwise["test_pair_probs"].astype(np.float32)
        if raw_pair_train.shape[1] != len(y_train) or raw_pair_val.shape[1] != len(yv) or raw_pair_test.shape[1] != len(yt):
            raise ValueError("Pairwise raw probability arrays are not aligned with train/validation/test splits.")
        print(
            f"[*] Loaded raw pairwise router scores: pairs={raw_pair_train.shape[0]} "
            f"classes={pairwise['pair_classes'].tolist()}"
        )

    print("\nBaselines")
    base_val_m = base.metrics_from_probs(main_val, yv, sv)
    base.print_metrics_line(f"{args.main_display_name} Val", base_val_m)
    if args.disable_cvtrn:
        base.print_metrics_line("CV-slot(main mirror) Val", base.metrics_from_probs(cv_val, yv, sv))
    else:
        base.print_metrics_line(f"{args.cv_display_name} Val", base.metrics_from_probs(cv_val, yv, sv))
    base.print_metrics_line("GAMC-tree Val", base.metrics_from_probs(gamc["val_prob"], yv, sv))
    base.print_metrics_line("HCS precision Val", base.metrics_from_probs(hcs_val, yv, sv))
    base.print_metrics_line("Pairwise Val", base.metrics_from_probs(pair_val, yv, sv))

    print("\n[*] Building stage-1 train/val/test meta-features")
    stage_aux_train = []
    stage_aux_val = []
    stage_aux_test = []
    for source in args.stage_aux_sources:
        if source == "hcs":
            stage_aux_train.append(hcs_train)
            stage_aux_val.append(hcs_val)
            stage_aux_test.append(hcs_test)
        elif source == "pairwise":
            stage_aux_train.append(pair_train)
            stage_aux_val.append(pair_val)
            stage_aux_test.append(pair_test)
    x_train = extra_eval.build_features(
        main_train, cv_train, gamc["train_prob"], gamc["train_member_probs"], q_train, stage_aux_train
    )
    x_val = extra_eval.build_features(
        main_val, cv_val, gamc["val_prob"], gamc["val_member_probs"], q_val, stage_aux_val
    )
    x_test = extra_eval.build_features(
        main_test, cv_test, gamc["test_prob"], gamc["test_member_probs"], q_test, stage_aux_test
    )
    weights = orig.sample_weights(snrs_train, main_train, y_train)
    print(f"Stage feature dim: {x_train.shape[1]} | train={len(x_train):,} | val={len(x_val):,} | test={len(x_test):,}")

    branches = rr.stage_crossfit(args, x_train, y_train, weights, x_val, x_test)
    if args.include_cv_mirror_stage and not args.disable_cvtrn:
        print("[*] Building CV-mirror/no-CV stage branches")
        x_train_mirror = extra_eval.build_features(
            main_train, main_train, gamc["train_prob"], gamc["train_member_probs"], q_train, stage_aux_train
        )
        x_val_mirror = extra_eval.build_features(
            main_val, main_val, gamc["val_prob"], gamc["val_member_probs"], q_val, stage_aux_val
        )
        x_test_mirror = extra_eval.build_features(
            main_test, main_test, gamc["test_prob"], gamc["test_member_probs"], q_test, stage_aux_test
        )
        mirror_branches = rr.stage_crossfit(args, x_train_mirror, y_train, weights, x_val_mirror, x_test_mirror)
        for b in mirror_branches:
            b["name"] = "cv_mirror_" + str(b["name"])
        branches.extend(mirror_branches)
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
    best_stage_state = None
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

        if best_stage_state is None:
            best_stage_state = {
                "branch": branch,
                "stage_cfg": stage_cfg,
                "stage_train": stage_train,
                "stage_val": stage_val,
                "stage_test": stage_test,
                "stage_val_m": stage_val_m,
            }

        aux_sources = {
            "pairwise": (pair_train, pair_val, pair_test),
            "cvtrn": (cv_train, cv_val, cv_test),
            "hcs": (hcs_train, hcs_val, hcs_test),
            "gamc": (gamc["train_prob"], gamc["val_prob"], gamc["test_prob"]),
            "main": (main_train, main_val, main_test),
        }
        score_val = np.full(len(yv), -1e9, dtype=np.float32)
        score_test = np.full(len(yt), -1e9, dtype=np.float32)
        score_train = np.full(len(y_train), -1e9, dtype=np.float32)
        selected_aux_train = np.array(stage_train, copy=True)
        selected_aux_val = np.array(stage_val, copy=True)
        selected_aux_test = np.array(stage_test, copy=True)
        selected_source_train = np.full(len(y_train), -1, dtype=np.int16)
        selected_source_val = np.full(len(yv), -1, dtype=np.int16)
        selected_source_test = np.full(len(yt), -1, dtype=np.int16)
        sp_train = norm(stage_train).argmax(1)
        sp_val = norm(stage_val).argmax(1)
        sp_test = norm(stage_test).argmax(1)
        transition_count = 0

        print(
            f"[*] Stage candidate {stage_rank}: {branch['name']} val={stage_val_m['overall_acc']:.3f}% "
            f"router auxiliaries={args.router_aux_sources}"
        )
        for source_i, source_name in enumerate(args.router_aux_sources):
            aux_train, aux_val, aux_test = aux_sources[source_name]
            transitions = candidate_transitions(stage_train, aux_train, y_train, args)
            transitions = transitions[: max(0, int(args.max_transitions_per_aux))]
            transition_count += len(transitions)
            print(f"    {source_name}: candidate transitions={len(transitions)}")
            for tr in transitions[:6]:
                print(
                    f"      {tr['from']}->{tr['to']} count={tr['count']} pos={tr['pos']} "
                    f"harm={tr['harm']} precision={tr['precision']:.3f}"
                )
            if not transitions:
                continue

            ap_train = norm(aux_train).argmax(1)
            ap_val = norm(aux_val).argmax(1)
            ap_test = norm(aux_test).argmax(1)
            for t_i, tr in enumerate(transitions):
                a, bcls = tr["from"], tr["to"]
                m_train = (sp_train == a) & (ap_train == bcls)
                if args.require_pairwise_gate and source_name == "pairwise":
                    m_train &= pair_train_gate
                local_y = (y_train[m_train] == bcls).astype(np.int64)
                if len(np.unique(local_y)) < 2:
                    continue
                local_w = np.full(m_train.sum(), float(args.other_weight), dtype=np.float32)
                local_w[y_train[m_train] == bcls] = float(args.rescue_weight)
                local_w[y_train[m_train] == a] = float(args.harm_weight)
                local_w[snrs_train[m_train] < 0] *= 1.20

                xtr = transition_features(
                    stage_train[m_train],
                    aux_train[m_train],
                    main_train[m_train],
                    cv_train[m_train],
                    gamc["train_prob"][m_train],
                    hcs_train[m_train],
                    q_router_train[m_train],
                    a,
                    bcls,
                    stat_train[m_train],
                    None if raw_pair_train is None else raw_pair_train[:, m_train, :],
                )
                for depth in args.router_depths:
                    seed = args.random_state + 100000 * stage_rank + 10000 * source_i + 100 * t_i + depth
                    model = transition_model(args, depth, seed)
                    model, _, _ = fit_or_load_estimator(
                        model,
                        xtr,
                        local_y,
                        sample_weight=local_w,
                        cache_dir=args.model_cache_dir,
                        reuse=args.reuse_models,
                        namespace=(
                            f"transition_router_{branch['name']}_{source_name}_{a}_{bcls}_d{depth}_s{stage_rank}"
                        ),
                        source_paths=[
                            *args.model_cache_source_paths,
                            args.hcs_feature_cache,
                            args.gamc_feature_cache,
                        ],
                        context={
                            "builder": "multi_aux_transition_router_v2",
                            "split_seed": int(args.split_seed),
                            "stage_rank": int(stage_rank),
                            "stage_branch": str(branch["name"]),
                            "source": source_name,
                            "transition": [int(a), int(bcls)],
                            "raw_pair_features": bool(args.use_pairwise_raw_router_features),
                        },
                    )
                    s_train = np.full(len(y_train), -1e9, dtype=np.float32)
                    s_val = np.full(len(yv), -1e9, dtype=np.float32)
                    s_test = np.full(len(yt), -1e9, dtype=np.float32)
                    s_train[m_train] = binary_proba(model, xtr)
                    mv = (sp_val == a) & (ap_val == bcls)
                    mt = (sp_test == a) & (ap_test == bcls)
                    if args.require_pairwise_gate and source_name == "pairwise":
                        mv &= pair_val_gate
                        mt &= pair_test_gate
                    if np.any(mv):
                        xv = transition_features(
                            stage_val[mv],
                            aux_val[mv],
                            main_val[mv],
                            cv_val[mv],
                            gamc["val_prob"][mv],
                            hcs_val[mv],
                            q_router_val[mv],
                            a,
                            bcls,
                            stat_val[mv],
                            None if raw_pair_val is None else raw_pair_val[:, mv, :],
                        )
                        s_val[mv] = binary_proba(model, xv)
                    if np.any(mt):
                        xt = transition_features(
                            stage_test[mt],
                            aux_test[mt],
                            main_test[mt],
                            cv_test[mt],
                            gamc["test_prob"][mt],
                            hcs_test[mt],
                            q_router_test[mt],
                            a,
                            bcls,
                            stat_test[mt],
                            None if raw_pair_test is None else raw_pair_test[:, mt, :],
                        )
                        s_test[mt] = binary_proba(model, xt)

                    better_train = s_train > score_train
                    better_val = s_val > score_val
                    better_test = s_test > score_test
                    score_train[better_train] = s_train[better_train]
                    score_val[better_val] = s_val[better_val]
                    score_test[better_test] = s_test[better_test]
                    selected_aux_train[better_train] = aux_train[better_train]
                    selected_aux_val[better_val] = aux_val[better_val]
                    selected_aux_test[better_test] = aux_test[better_test]
                    selected_source_train[better_train] = source_i
                    selected_source_val[better_val] = source_i
                    selected_source_test[better_test] = source_i

        if transition_count == 0:
            continue
        rows = search_router(
            stage_val,
            selected_aux_val,
            score_val,
            yv,
            sv,
            stage_val_m,
            args,
            "multi_aux_transition_xgb",
            q_router_val,
        )
        for row in rows:
            row["_stage_train"] = stage_train
            row["_score_train"] = score_train
            row["_aux_train"] = selected_aux_train
            row["_source_train"] = selected_source_train
            row["_stage_val"] = stage_val
            row["_score_val"] = score_val
            row["_aux_val"] = selected_aux_val
            row["_source_val"] = selected_source_val
            row["_stage_test"] = stage_test
            row["_score_test"] = score_test
            row["_aux_test"] = selected_aux_test
            row["_source_test"] = selected_source_test
            row["_stage_cfg"] = stage_cfg
            row["stage_branch"] = branch["name"]
            row["stage_val_overall"] = float(stage_val_m["overall_acc"])
            row["transition_count"] = int(transition_count)
            all_rows.append(row)

    if all_rows:
        all_rows.sort(key=lambda r: r["score"], reverse=True)
        print_top(all_rows, 20)
        best = all_rows[0]
        use_stage_fallback = False
        if best_stage_state is not None and args.router_min_global_stage_overall_gain is not None:
            required_overall = (
                float(best_stage_state["stage_val_m"]["overall_acc"])
                + float(args.router_min_global_stage_overall_gain)
            )
            if float(best["overall_acc"]) < required_overall:
                use_stage_fallback = True
                print(
                    "\n[!] Best router did not clear global Stage-1 validation guard: "
                    f"router={best['overall_acc']:.3f}% < required={required_overall:.3f}%. "
                    "Keeping best Stage-1 branch."
                )

        if use_stage_fallback:
            branch = best_stage_state["branch"]
            stage_cfg = best_stage_state["stage_cfg"]
            stage_train = best_stage_state["stage_train"]
            final_train = stage_train
            train_router_gate = np.zeros(len(stage_train), dtype=bool)
            selected_aux_train = np.array(stage_train, copy=True)
            selected_source_train = np.full(len(stage_train), -1, dtype=np.int16)
            stage_val = best_stage_state["stage_val"]
            final_val = stage_val
            val_router_gate = np.zeros(len(stage_val), dtype=bool)
            selected_aux_val = np.array(stage_val, copy=True)
            selected_source_val = np.full(len(stage_val), -1, dtype=np.int16)
            stage_test = best_stage_state["stage_test"]
            final_test = stage_test
            router_gate = np.zeros(len(stage_test), dtype=bool)
            selected_aux_test = np.array(stage_test, copy=True)
            selected_source_test = np.full(len(stage_test), -1, dtype=np.int16)
            best = {
                "stage_branch": branch["name"],
                "stage_config": stage_cfg,
                "router_name": "none_global_stage_guard",
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
                "stage_val_overall": float(best_stage_state["stage_val_m"]["overall_acc"]),
                "transition_count": 0,
            }
            used_router = False
        else:
            stage_train = best["_stage_train"]
            selected_aux_train = best["_aux_train"]
            selected_source_train = best["_source_train"]
            final_train, train_router_gate = apply_router(
                stage_train, selected_aux_train, best["_score_train"], best, q_router_train
            )
            stage_val = best["_stage_val"]
            selected_aux_val = best["_aux_val"]
            selected_source_val = best["_source_val"]
            final_val, val_router_gate = apply_router(
                stage_val, selected_aux_val, best["_score_val"], best, q_router_val
            )
            stage_test = best["_stage_test"]
            selected_aux_test = best["_aux_test"]
            selected_source_test = best["_source_test"]
            final_test, router_gate = apply_router(
                stage_test, selected_aux_test, best["_score_test"], best, q_router_test
            )
            used_router = True
    else:
        print("\n[!] No transition router improved validation overall; keeping best stage-1 candidate.")
        if best_stage_state is not None:
            branch = best_stage_state["branch"]
            stage_cfg = best_stage_state["stage_cfg"]
            stage_train = best_stage_state["stage_train"]
            stage_val = best_stage_state["stage_val"]
            stage_test = best_stage_state["stage_test"]
            stage_val_m = best_stage_state["stage_val_m"]
        else:
            _, branch, stage_cfg = stage_candidates[0]
            stage_train, _, _, _ = st.apply_stacked(main_train, branch["train"], stage_cfg, args)
            stage_val, _, _, _ = st.apply_stacked(main_val, branch["val"], stage_cfg, args)
            stage_test, _, _, _ = st.apply_stacked(main_test, branch["test"], stage_cfg, args)
            stage_val_m = base.metrics_from_probs(stage_val, yv, sv)
        final_train = stage_train
        train_router_gate = np.zeros(len(stage_train), dtype=bool)
        selected_aux_train = np.array(stage_train, copy=True)
        selected_source_train = np.full(len(stage_train), -1, dtype=np.int16)
        final_val = stage_val
        val_router_gate = np.zeros(len(stage_val), dtype=bool)
        selected_aux_val = np.array(stage_val, copy=True)
        selected_source_val = np.full(len(stage_val), -1, dtype=np.int16)
        final_test = stage_test
        router_gate = np.zeros(len(stage_test), dtype=bool)
        selected_aux_test = np.array(stage_test, copy=True)
        selected_source_test = np.full(len(stage_test), -1, dtype=np.int16)
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
            "stage_val_overall": float(stage_val_m["overall_acc"]),
            "transition_count": 0,
        }
        used_router = False

    clean_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in all_rows] if all_rows else [best]
    save_csv(relpath("results", f"{suffix}_search_top.csv"), clean_rows, args.save_top_records)

    selected = {
        "best": {k: v for k, v in best.items() if not k.startswith("_")},
        "used_router": used_router,
        "blind_cqi_val_bin_acc": q_acc,
        "router_blind_cqi_val_bin_acc": q_router_acc,
        "router_quality_extra_feature_caches": list(getattr(args, "router_quality_extra_feature_caches", []) or []),
        "pairwise_cache": args.pairwise_cache,
        "stage_aux_sources": list(args.stage_aux_sources),
        "router_aux_sources": list(args.router_aux_sources),
        "use_pairwise_raw_router_features": bool(args.use_pairwise_raw_router_features),
        "test_report_deferred": bool(args.defer_test_report),
        "protocol": "Train-OOF multi-aux transition-specific router; validation-selected thresholds; test report may be deferred.",
    }
    with open(relpath("results", f"{suffix}_selected_config.json"), "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)

    if args.distill_teacher_cache:
        save_distill_teacher_cache(
            args.distill_teacher_cache,
            final_train,
            final_val,
            final_test,
            y_train,
            snrs_train,
            yv,
            sv,
            yt,
            stest,
            selected,
            args.distill_teacher_mode,
        )

    pred = norm(final_test).argmax(1).astype(np.int64)
    if args.defer_test_report:
        print("\n[*] Test-label evaluation deferred to the locked final reporting stage.")
    else:
        final_m = base.metrics_from_probs(final_test, yt, stest)
        stage_m = base.metrics_from_probs(stage_test, yt, stest)
        diag = base.switch_diagnostics(
            stage_test,
            final_test,
            router_gate,
            router_gate,
            np.full(len(router_gate), best.get("router_alpha", 0.0), dtype=np.float32),
            stest,
        )
        print("\n" + "=" * 144)
        print("Final test report")
        print("=" * 144)
        base.print_metrics_line(f"{args.main_display_name} Test", base.metrics_from_probs(main_test, yt, stest))
        if args.disable_cvtrn:
            base.print_metrics_line("CV-slot(main mirror) Test", base.metrics_from_probs(cv_test, yt, stest))
        else:
            base.print_metrics_line(f"{args.cv_display_name} Test", base.metrics_from_probs(cv_test, yt, stest))
        base.print_metrics_line("GAMC-tree Test", base.metrics_from_probs(gamc["test_prob"], yt, stest))
        base.print_metrics_line("HCS precision Test", base.metrics_from_probs(hcs_test, yt, stest))
        base.print_metrics_line("Pairwise Test", base.metrics_from_probs(pair_test, yt, stest))
        base.print_metrics_line("Stage-1 Test", stage_m)
        base.print_metrics_line("Transition router Test", final_m)
        print("-" * 144)
        main_m = base.metrics_from_probs(main_test, yt, stest)
        print(f"Delta vs {args.main_display_name} overall:    {final_m['overall_acc'] - main_m['overall_acc']:+.4f} pp")
        print(f"Delta vs {args.main_display_name} negative:   {final_m['negative_acc'] - main_m['negative_acc']:+.4f} pp")
        print(f"Delta vs {args.main_display_name} edge:       {final_m['edge_low_acc'] - main_m['edge_low_acc']:+.4f} pp")
        print(f"Delta vs {args.main_display_name} transition: {final_m['transition_acc'] - main_m['transition_acc']:+.4f} pp")
        print(f"Delta vs {args.main_display_name} high:       {final_m['high_acc'] - main_m['high_acc']:+.4f} pp")
        print(f"Delta vs stage-1 overall:         {final_m['overall_acc'] - stage_m['overall_acc']:+.4f} pp")
        print(f"Router diagnostics: {diag}")
        if np.any(router_gate):
            source_counts = {
                name: int(np.sum(router_gate & (selected_source_test == i)))
                for i, name in enumerate(args.router_aux_sources)
            }
            print(f"Applied router sources (test diagnostic): {source_counts}")
        print("=" * 144)
    np.savez_compressed(
        relpath("results", f"{suffix}_predictions.npz"),
        labels=yt.astype(np.int64),
        snrs=stest.astype(np.int32),
        pred=pred,
        final_prob=final_test.astype(np.float32),
        final_val_prob=final_val.astype(np.float32),
        stage_prob=stage_test.astype(np.float32),
        stage_val_prob=stage_val.astype(np.float32),
        pairwise_prob=pair_test.astype(np.float32),
        router_aux_prob=selected_aux_test.astype(np.float32),
        router_aux_val_prob=selected_aux_val.astype(np.float32),
        router_source=selected_source_test.astype(np.int16),
        router_source_val=selected_source_val.astype(np.int16),
        router_source_names=np.asarray(args.router_aux_sources),
        router_gate=router_gate.astype(bool),
        router_gate_val=val_router_gate.astype(bool),
        labels_val=yv.astype(np.int64),
        snrs_val=sv.astype(np.int32),
        mod_classes=soup.get("mod_classes", np.asarray(common.DEFAULT_MOD_CLASSES)),
    )
    if not args.defer_test_report:
        curve_path = relpath("results", f"accuracy_vs_snr_{suffix}.png")
        base.plot_curve(final_m["by_snr"], curve_path, "Accuracy vs SNR: Multi-Aux Transition Router")
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
                f"Confusion Matrix at {snr_value} dB: Multi-Aux Transition Router",
            )
            print(f"[*] {snr_value} dB confusion matrix saved: {cm_path} (Acc={acc:.2f}%)")
    print(f"[*] Selected config saved: {relpath('results', f'{suffix}_selected_config.json')}")
    print(f"[*] Predictions saved: {relpath('results', f'{suffix}_predictions.npz')}")


if __name__ == "__main__":
    main()
