import argparse
import json
import os

import numpy as np
from sklearn.model_selection import StratifiedKFold

import evaluate_fourier_gamc_cvtrn_stacked_residual_fusion as st
import evaluate_fourier_gamc_cvtrn_train_oof_meta_fusion as oof
import evaluate_fourier_gamc_multi_cvtrn_blind_quality_xgb_fusion as bq
import evaluate_greedy_soup_gamc_protected_residual_fusion as base
import evaluate_fourier_oof_gamc_cvtrn_residual_meta_fusion as orig
import train_cv_trn_aux_2016 as common
from model_cache_utils import fit_or_load_estimator


MIDLOW_SNRS = np.array([-14, -12], dtype=np.int32)
WIDE_TRANSITION_SNRS = np.array([-14, -12, -10, -8, -6, -4, -2], dtype=np.int32)


def relpath(*parts):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Fourier-OOF residual meta fusion with extra lightweight auxiliary OOF experts. "
            "The extra experts are used as probability experts only."
        )
    )
    p.add_argument("--split_seed", type=int, default=1)
    p.add_argument("--random_state", type=int, default=2026)
    p.add_argument("--soup_prob_cache", type=str, default=relpath("results", "greedy_soup_identity_valtest_probs_for_gamc_fusion.npz"))
    p.add_argument("--fourier_oof_cache", type=str, default=relpath("results", "fourier_main_oof_merge_mseed78_79_f3e220_bs128_split1_trainvaltest_probs_for_meta.npz"))
    p.add_argument("--gamc_oof_cache", type=str, default=relpath("results", "gamc_oof_tree_split1_trainvaltest_probs_for_meta.npz"))
    p.add_argument("--cvtrn_oof_cache", type=str, default=relpath("results", "cv_trn_aux_v2_oof_mseed41_f3e240_split1_trainvaltest_probs_for_meta.npz"))
    p.add_argument(
        "--disable_cvtrn",
        action="store_true",
        help=(
            "OOF source ablation: replace every IQCC/CV-TRN probability block with the "
            "aligned KAN-Fourier probability. The feature schema is preserved, but no "
            "independent IQCC information reaches the utility estimator."
        ),
    )
    p.add_argument("--extra_aux_oof_caches", type=str, nargs="*", default=[])
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
    p.add_argument(
        "--cvtrn_infer_from_valtest_only",
        action="store_true",
        help=(
            "Use CV-TRN OOF probabilities only for train-split meta training, and use only "
            "--cvtrn_valtest_caches for validation/test inference. This matches single deployed "
            "CV-TRN inference while keeping train features OOF."
        ),
    )
    p.add_argument("--use_fourier_oof_infer", action="store_true")
    p.add_argument("--output_suffix", type=str, default="fourier_oof_gamc_cvtrn_extraaux_residual_meta_split1")
    p.add_argument(
        "--select_branches",
        type=str,
        nargs="*",
        default=[],
        help=(
            "Optional branch names to keep during final validation selection, e.g. "
            "xgb_d2_620 xgb_d4_400 extraaux_oof_ensemble. Empty means all branches."
        ),
    )
    p.add_argument(
        "--ensemble_branches",
        type=str,
        nargs="*",
        default=[],
        help=(
            "Optional base branches used to build extraaux_oof_ensemble. Empty preserves the "
            "legacy four-branch ensemble."
        ),
    )

    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--blend_alphas", type=float, nargs="+", default=[0.50, 0.65, 0.80])
    p.add_argument("--meta_conf_thresholds", type=float, nargs="+", default=[0.00, 0.25, 0.35, 0.45, 0.55])
    p.add_argument("--advantage_thresholds", type=float, nargs="+", default=[-0.20, -0.10, 0.00, 0.05, 0.10])
    p.add_argument("--max_change_rates", type=float, nargs="+", default=[6.0, 8.0, 10.0, 12.0])
    p.add_argument("--hard_keep_conf", type=float, default=0.93)
    p.add_argument("--hard_keep_margin", type=float, default=0.70)
    p.add_argument("--min_val_transition_acc", type=float, default=0.0)
    p.add_argument("--min_val_midlow_acc", type=float, default=0.0)
    p.add_argument("--min_val_wide_transition_acc", type=float, default=0.0)
    p.add_argument("--min_val_high_acc", type=float, default=0.0)
    p.add_argument("--meta_stability_folds", type=int, default=0)
    p.add_argument("--min_oof_overall_gain", type=float, default=-999.0)
    p.add_argument("--min_oof_midlow_gain", type=float, default=-999.0)
    p.add_argument("--max_oof_transition_drop", type=float, default=999.0)
    p.add_argument("--max_oof_high_drop", type=float, default=999.0)
    p.add_argument("--max_oof_changed_high_rate", type=float, default=999.0)
    p.add_argument("--min_oof_net_rescue", type=int, default=-1000000)
    p.add_argument("--min_oof_precision", type=float, default=0.0)
    p.add_argument("--oof_score_weight", type=float, default=0.0)

    p.add_argument("--score_overall_weight", type=float, default=1.0)
    p.add_argument("--score_negative_gain_weight", type=float, default=0.020)
    p.add_argument("--score_edge_gain_weight", type=float, default=0.020)
    p.add_argument("--score_transition_gain_weight", type=float, default=0.010)
    p.add_argument(
        "--score_midlow_gain_weight",
        type=float,
        default=0.0,
        help="Reward validation gain on the -14/-12 dB band over the main model.",
    )
    p.add_argument(
        "--score_wide_transition_gain_weight",
        type=float,
        default=0.0,
        help="Reward validation gain on the wider -14..-2 dB transition band over the main model.",
    )
    p.add_argument("--score_high_penalty", type=float, default=3.00)
    p.add_argument("--high_tolerance", type=float, default=0.05)
    p.add_argument("--score_changed_high_penalty", type=float, default=0.015)
    p.add_argument("--score_changed_nonultra_penalty", type=float, default=0.006)
    p.add_argument(
        "--score_amdsb_drop_penalty",
        type=float,
        default=0.0,
        help="Penalty per pp when AM-DSB validation accuracy drops beyond --amdsb_drop_tolerance from main Fourier.",
    )
    p.add_argument("--amdsb_drop_tolerance", type=float, default=5.0)
    p.add_argument(
        "--score_wbfm_gain_weight",
        type=float,
        default=0.0,
        help="Reward per pp WBFM validation gain over main Fourier. Use lightly; WBFM and AM-DSB trade off.",
    )
    p.add_argument(
        "--score_analog_gap_penalty",
        type=float,
        default=0.0,
        help="Penalty per pp for WBFM-AMDSB imbalance growth beyond --analog_gap_tolerance.",
    )
    p.add_argument("--analog_gap_tolerance", type=float, default=2.0)
    p.add_argument("--amdsb_class_idx", type=int, default=1)
    p.add_argument("--wbfm_class_idx", type=int, default=10)
    p.add_argument(
        "--forbid_transition_pairs",
        type=str,
        nargs="*",
        default=[],
        help=(
            "Hard-forbid changed predictions from main->meta for class pairs, e.g. 1:10 forbids "
            "AM-DSB->WBFM. Forbidden samples are reverted to the main Fourier probabilities."
        ),
    )
    p.add_argument(
        "--transition_risk_pairs",
        type=str,
        nargs="*",
        default=[],
        help="Soft-risk class transition pairs to penalize during validation scoring, e.g. 1:10 8:7.",
    )
    p.add_argument(
        "--transition_risk_penalty",
        type=float,
        default=0.0,
        help="Score penalty per percentage point of risky changed predictions in --transition_risk_pairs.",
    )

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
    p.add_argument("--data_path", type=str, default=relpath("raw_data", "RML2016.10a_dict.pkl"))
    p.add_argument("--cache_dir", type=str, default=relpath("feature_cache"))
    p.add_argument("--alignment_cache", type=str, default=relpath("results", "greedy_soup_identity_valtest_probs_for_gamc_fusion.npz"))
    p.add_argument("--skip_alignment_check", action="store_true")
    p.add_argument("--save_top_records", type=int, default=240)
    p.add_argument("--cm_snrs", type=int, nargs="+", default=[0, -6])
    return p.parse_args()


def build_features(main_prob, cv_prob, gamc_prob, member_probs, qprob, extra_probs):
    main = oof.norm(main_prob)
    cv = oof.norm(cv_prob)
    gamc = oof.norm(gamc_prob)
    members = [oof.norm(p) for p in np.asarray(member_probs)]
    extras = [oof.norm(p) for p in extra_probs]
    qfeat = bq.quality_probability_features(qprob)
    parts = []
    parts.extend(oof.expert_blocks(main))
    parts.extend(oof.expert_blocks(cv))
    parts.extend(oof.expert_blocks(gamc))
    parts.append(oof.pair_blocks(main, cv))
    parts.append(oof.pair_blocks(main, gamc))
    parts.append(oof.pair_blocks(cv, gamc))
    for extra in extras:
        parts.extend(oof.expert_blocks(extra))
        parts.append(oof.pair_blocks(main, extra))
        parts.append(oof.pair_blocks(cv, extra))
        parts.append(oof.pair_blocks(gamc, extra))
    parts.append(qfeat.astype(np.float32))
    stack = np.stack([main, cv, gamc] + extras + members, axis=0)
    parts.append(stack.mean(axis=0).astype(np.float32))
    parts.append(stack.std(axis=0).astype(np.float32))
    parts.append(stack.max(axis=0).astype(np.float32))
    parts.append(stack.min(axis=0).astype(np.float32))
    for p in members:
        parts.extend(oof.expert_blocks(p))
        parts.append(oof.pair_blocks(main, p))
        parts.append(oof.pair_blocks(cv, p))
        parts.append(oof.pair_blocks(gamc, p))
        for extra in extras:
            parts.append(oof.pair_blocks(extra, p))
    x = np.concatenate(parts, axis=1)
    return np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)


def load_extra_caches(args, soup, fourier, gamc, cv):
    extras = []
    for path in args.extra_aux_oof_caches:
        if not path:
            continue
        z = orig.load_npz(path)
        for key in ("labels_train", "snrs_train"):
            orig.assert_same(fourier, z, key, key, f"Extra aux OOF {os.path.basename(path)}")
        for key in ("labels_val", "snrs_val", "labels_test", "snrs_test"):
            orig.assert_same(soup, z, key, key, f"Extra aux OOF {os.path.basename(path)}")
            orig.assert_same(gamc, z, key, key, f"Extra aux OOF {os.path.basename(path)}")
            orig.assert_same(cv, z, key, key, f"Extra aux OOF {os.path.basename(path)}")
        extras.append({"path": path, "name": os.path.splitext(os.path.basename(path))[0], "cache": z})
        print(f"[*] Added extra auxiliary OOF cache: {path}")
    return extras


def class_acc(prob, labels, class_idx):
    labels = np.asarray(labels, dtype=np.int64)
    pred = np.asarray(prob).argmax(1)
    mask = labels == int(class_idx)
    if not np.any(mask):
        return 0.0
    return float((pred[mask] == labels[mask]).mean() * 100.0)


def analog_class_score(prob, labels, main_prob, args):
    amdsb_idx = int(args.amdsb_class_idx)
    wbfm_idx = int(args.wbfm_class_idx)
    amdsb = class_acc(prob, labels, amdsb_idx)
    wbfm = class_acc(prob, labels, wbfm_idx)
    main_amdsb = class_acc(main_prob, labels, amdsb_idx)
    main_wbfm = class_acc(main_prob, labels, wbfm_idx)
    amdsb_drop = main_amdsb - amdsb
    wbfm_gain = wbfm - main_wbfm
    main_gap = abs(main_wbfm - main_amdsb)
    gap = abs(wbfm - amdsb)
    score_delta = 0.0
    score_delta -= float(args.score_amdsb_drop_penalty) * max(0.0, amdsb_drop - float(args.amdsb_drop_tolerance))
    score_delta += float(args.score_wbfm_gain_weight) * wbfm_gain
    score_delta -= float(args.score_analog_gap_penalty) * max(0.0, gap - main_gap - float(args.analog_gap_tolerance))
    return {
        "amdsb_acc": float(amdsb),
        "wbfm_acc": float(wbfm),
        "main_amdsb_acc": float(main_amdsb),
        "main_wbfm_acc": float(main_wbfm),
        "amdsb_drop_from_main": float(amdsb_drop),
        "wbfm_gain_from_main": float(wbfm_gain),
        "analog_gap": float(gap),
        "main_analog_gap": float(main_gap),
        "class_score_delta": float(score_delta),
    }


def snr_band_acc(prob, labels, snrs, band):
    mask = np.isin(np.asarray(snrs, dtype=np.int32), band)
    if not np.any(mask):
        return 0.0
    pred = np.asarray(prob).argmax(1)
    return float((pred[mask] == np.asarray(labels)[mask]).mean() * 100.0)


def snr_band_score_info(prob, main_prob, labels, snrs):
    midlow = snr_band_acc(prob, labels, snrs, MIDLOW_SNRS)
    main_midlow = snr_band_acc(main_prob, labels, snrs, MIDLOW_SNRS)
    wide = snr_band_acc(prob, labels, snrs, WIDE_TRANSITION_SNRS)
    main_wide = snr_band_acc(main_prob, labels, snrs, WIDE_TRANSITION_SNRS)
    return {
        "midlow_acc": float(midlow),
        "main_midlow_acc": float(main_midlow),
        "midlow_gain_from_main": float(midlow - main_midlow),
        "wide_transition_acc": float(wide),
        "main_wide_transition_acc": float(main_wide),
        "wide_transition_gain_from_main": float(wide - main_wide),
    }


def parse_transition_pairs(items):
    pairs = []
    for item in items or []:
        if not item:
            continue
        text = str(item).replace(",", ":")
        if ":" not in text:
            raise ValueError(f"Invalid transition pair {item!r}; expected from:to")
        a, b = text.split(":", 1)
        pairs.append((int(a), int(b)))
    return tuple(pairs)


def transition_pair_mask(main_prob, final_prob, pairs):
    if not pairs:
        return np.zeros(len(main_prob), dtype=bool)
    main_pred = oof.norm(main_prob).argmax(1)
    final_pred = oof.norm(final_prob).argmax(1)
    changed = final_pred != main_pred
    mask = np.zeros(len(main_pred), dtype=bool)
    for a, b in pairs:
        mask |= changed & (main_pred == int(a)) & (final_pred == int(b))
    return mask


def apply_transition_policy(main_prob, final, gate, use, alpha_vec, args):
    forbidden = getattr(args, "_forbid_transition_pairs", ())
    if not forbidden:
        return final, gate, use, alpha_vec, {"forbidden_transition_rate": 0.0, "forbidden_transition_count": 0}
    blocked = transition_pair_mask(main_prob, final, forbidden)
    if not np.any(blocked):
        return final, gate, use, alpha_vec, {"forbidden_transition_rate": 0.0, "forbidden_transition_count": 0}
    out = np.asarray(final, dtype=np.float32).copy()
    out[blocked] = oof.norm(main_prob)[blocked]
    out = oof.norm(out)
    main_pred = oof.norm(main_prob).argmax(1)
    new_use = out.argmax(1) != main_pred
    new_gate = np.asarray(gate, dtype=bool).copy()
    new_gate[blocked] = False
    return (
        out,
        new_gate,
        new_use.astype(bool),
        alpha_vec,
        {
            "forbidden_transition_rate": float(blocked.mean() * 100.0),
            "forbidden_transition_count": int(blocked.sum()),
        },
    )


def apply_stacked_with_policy(main_prob, meta_prob, cfg, args):
    final, gate, use, alpha_vec = st.apply_stacked(main_prob, meta_prob, cfg, args)
    return apply_transition_policy(main_prob, final, gate, use, alpha_vec, args)


def transition_risk_info(main_prob, final_prob, args):
    risk_pairs = getattr(args, "_transition_risk_pairs", ())
    mask = transition_pair_mask(main_prob, final_prob, risk_pairs)
    return {"risk_transition_rate": float(mask.mean() * 100.0), "risk_transition_count": int(mask.sum())}


def patch_rescue_stats(base_prob, out_prob, labels, gate):
    base_pred = oof.norm(base_prob).argmax(1)
    out_pred = oof.norm(out_prob).argmax(1)
    y = np.asarray(labels, dtype=np.int64)
    changed = np.asarray(gate, dtype=bool) & (base_pred != out_pred)
    rescue = int(((base_pred != y) & (out_pred == y) & changed).sum())
    harm = int(((base_pred == y) & (out_pred != y) & changed).sum())
    neutral = int(changed.sum()) - rescue - harm
    denom = rescue + harm
    precision = float(rescue / denom) if denom > 0 else 0.0
    return {
        "oof_changed": int(changed.sum()),
        "oof_rescue": rescue,
        "oof_harm": harm,
        "oof_neutral": neutral,
        "oof_net_rescue": int(rescue - harm),
        "oof_precision": precision,
    }


def make_meta_model(args, branch_name, seed):
    if branch_name == "et_depth20":
        return orig.et_model(args, seed)
    return orig.xgb_model(args, branch_name, seed)


def train_meta_oof(args, branch_name, x_train, labels_train, snrs_train, weights):
    n = len(labels_train)
    out = np.zeros((n, base.NUM_CLASSES), dtype=np.float32)
    composite = np.asarray([f"{int(y)}_{int(s)}" for y, s in zip(labels_train, snrs_train)])
    skf = StratifiedKFold(
        n_splits=int(args.meta_stability_folds),
        shuffle=True,
        random_state=int(args.random_state + 3301),
    )
    print(f"    [stability] cross-fitting first-meta OOF for {branch_name} ({args.meta_stability_folds} folds)")
    for fold, (tr, va) in enumerate(skf.split(x_train, composite), 1):
        clf = make_meta_model(args, branch_name, int(args.random_state + 4300 + 100 * fold))
        clf.fit(x_train[tr], labels_train[tr], sample_weight=weights[tr])
        out[va] = orig.aligned_proba(clf, x_train[va])
        print(f"      meta OOF fold {fold}/{args.meta_stability_folds} done")
    return out


def oof_stability_fields(final, gate, use, alpha_vec, main_prob, labels, snrs, base_m, args):
    m = base.metrics_from_probs(final, labels, snrs)
    d = base.switch_diagnostics(main_prob, final, gate, use, alpha_vec, snrs)
    band_info = snr_band_score_info(final, main_prob, labels, snrs)
    class_info = analog_class_score(final, labels, main_prob, args)
    fields = {
        "oof_overall_gain": float(m["overall_acc"] - base_m["overall_acc"]),
        "oof_transition_gain": float(m["transition_acc"] - base_m["transition_acc"]),
        "oof_high_gain": float(m["high_acc"] - base_m["high_acc"]),
        "oof_negative_gain": float(m["negative_acc"] - base_m["negative_acc"]),
        "oof_midlow_gain": float(band_info["midlow_gain_from_main"]),
        "oof_wide_transition_gain": float(band_info["wide_transition_gain_from_main"]),
        "oof_changed_high_rate": float(d["changed_high_rate"]),
        "oof_changed_nonultra_rate": float(d["changed_nonultra_rate"]),
        "oof_amdsb_drop_from_main": float(class_info["amdsb_drop_from_main"]),
        "oof_wbfm_gain_from_main": float(class_info["wbfm_gain_from_main"]),
        **patch_rescue_stats(main_prob, final, labels, gate),
    }
    ok = True
    ok &= fields["oof_overall_gain"] >= float(args.min_oof_overall_gain)
    ok &= fields["oof_midlow_gain"] >= float(args.min_oof_midlow_gain)
    ok &= fields["oof_transition_gain"] >= -float(args.max_oof_transition_drop)
    ok &= fields["oof_high_gain"] >= -float(args.max_oof_high_drop)
    ok &= fields["oof_changed_high_rate"] <= float(args.max_oof_changed_high_rate)
    ok &= fields["oof_net_rescue"] >= int(args.min_oof_net_rescue)
    if int(fields["oof_rescue"] + fields["oof_harm"]) > 0:
        ok &= fields["oof_precision"] >= float(args.min_oof_precision)
    return ok, fields


def search_configs(main_prob, meta_prob, labels, snrs, base_m, args, branch_name,
                   main_oof=None, meta_oof=None, labels_oof=None, snrs_oof=None, base_oof_m=None):
    rows = []
    total = (
        len(args.blend_alphas)
        * len(args.meta_conf_thresholds)
        * len(args.advantage_thresholds)
        * len(args.max_change_rates)
    )
    for alpha in args.blend_alphas:
        for conf in args.meta_conf_thresholds:
            for adv in args.advantage_thresholds:
                for max_rate in args.max_change_rates:
                    cfg = {
                        "branch": branch_name,
                        "alpha": float(alpha),
                        "meta_conf_thr": float(conf),
                        "advantage_thr": float(adv),
                        "max_change_rate": float(max_rate),
                    }
                    final, gate, use, alpha_vec, policy_info = apply_stacked_with_policy(main_prob, meta_prob, cfg, args)
                    m = base.metrics_from_probs(final, labels, snrs)
                    d = base.switch_diagnostics(main_prob, final, gate, use, alpha_vec, snrs)
                    class_info = analog_class_score(final, labels, main_prob, args)
                    band_info = snr_band_score_info(final, main_prob, labels, snrs)
                    risk_info = transition_risk_info(main_prob, final, args)
                    if float(args.min_val_transition_acc) > 0 and m["transition_acc"] < float(args.min_val_transition_acc):
                        continue
                    if float(args.min_val_midlow_acc) > 0 and band_info["midlow_acc"] < float(args.min_val_midlow_acc):
                        continue
                    if (
                        float(args.min_val_wide_transition_acc) > 0
                        and band_info["wide_transition_acc"] < float(args.min_val_wide_transition_acc)
                    ):
                        continue
                    if float(args.min_val_high_acc) > 0 and m["high_acc"] < float(args.min_val_high_acc):
                        continue
                    stable_fields = {}
                    if meta_oof is not None:
                        oof_final, oof_gate, oof_use, oof_alpha, _ = apply_stacked_with_policy(main_oof, meta_oof, cfg, args)
                        stable_ok, stable_fields = oof_stability_fields(
                            oof_final,
                            oof_gate,
                            oof_use,
                            oof_alpha,
                            main_oof,
                            labels_oof,
                            snrs_oof,
                            base_oof_m,
                            args,
                        )
                        if not stable_ok:
                            continue
                    score = base.selection_score(m, d, base_m, args) + class_info["class_score_delta"]
                    score += float(args.score_midlow_gain_weight) * band_info["midlow_gain_from_main"]
                    score += (
                        float(args.score_wide_transition_gain_weight)
                        * band_info["wide_transition_gain_from_main"]
                    )
                    score -= float(args.transition_risk_penalty) * risk_info["risk_transition_rate"]
                    if stable_fields and float(args.oof_score_weight) != 0.0:
                        score += float(args.oof_score_weight) * stable_fields["oof_overall_gain"]
                    rows.append(
                        {
                            "score": float(score),
                            "overall_acc": float(m["overall_acc"]),
                            "transition_acc": float(m["transition_acc"]),
                            "edge_low_acc": float(m["edge_low_acc"]),
                            "negative_acc": float(m["negative_acc"]),
                            "high_acc": float(m["high_acc"]),
                            "gate_rate": float(d["gate_rate"]),
                            "use_rate": float(d["use_rate"]),
                            "changed_high_rate": float(d["changed_high_rate"]),
                            "changed_nonultra_rate": float(d["changed_nonultra_rate"]),
                            **class_info,
                            **band_info,
                            **policy_info,
                            **risk_info,
                            **stable_fields,
                            **cfg,
                        }
                    )
    rows.sort(key=lambda r: r["score"], reverse=True)
    if not rows:
        print(f"    searched {total:4d} configs for {branch_name:<18} | no config passed validation floors")
        return rows
    best = rows[0]
    print(
        f"    searched {total:4d} configs for {branch_name:<18} | "
        f"best val={best['overall_acc']:.3f}% score={best['score']:.3f} "
        f"AMDSB={best['amdsb_acc']:.2f}% WBFM={best['wbfm_acc']:.2f}%"
    )
    return rows


def main():
    args = parse_args()
    args._forbid_transition_pairs = parse_transition_pairs(args.forbid_transition_pairs)
    args._transition_risk_pairs = parse_transition_pairs(args.transition_risk_pairs)
    os.makedirs(relpath("results"), exist_ok=True)
    suffix = args.output_suffix
    print("=" * 144)
    print("Fourier-OOF residual meta fusion + extra lightweight auxiliary experts")
    print("=" * 144)
    print("Academic protocol:")
    print("  - Fourier soup remains the primary classifier.")
    print("  - Meta-classifier is trained with train-split OOF probabilities only.")
    print("  - Extra auxiliary predictors are fixed probability sources.")
    print("  - Validation labels select blend/gate settings only.")
    print("  - Test labels are excluded from every optimization objective.")
    print("  - True SNR is never used as an inference feature.")
    if args._forbid_transition_pairs:
        print(f"  - Forbidden transition pairs are reverted to the main model: {args._forbid_transition_pairs}")
    if args._transition_risk_pairs and float(args.transition_risk_penalty) != 0.0:
        print(
            f"  - Risk transition pairs are penalized during validation selection: "
            f"{args._transition_risk_pairs} penalty={args.transition_risk_penalty}"
        )

    soup = orig.load_npz(args.soup_prob_cache)
    fourier = orig.load_npz(args.fourier_oof_cache)
    gamc = orig.load_npz(args.gamc_oof_cache)
    cv = orig.load_npz(args.cvtrn_oof_cache)

    for key in ("labels_train", "snrs_train"):
        orig.assert_same(fourier, gamc, key, key, "Fourier OOF vs GAMC OOF")
        orig.assert_same(fourier, cv, key, key, "Fourier OOF vs CVTRN OOF")
    for key in ("labels_val", "snrs_val", "labels_test", "snrs_test"):
        orig.assert_same(soup, fourier, key, key, "Fourier soup vs Fourier OOF")
        orig.assert_same(soup, gamc, key, key, "Fourier soup vs GAMC OOF")
        orig.assert_same(soup, cv, key, key, "Fourier soup vs CVTRN OOF")
    extras = load_extra_caches(args, soup, fourier, gamc, cv)
    print("[*] Alignment check passed for all OOF caches.")

    labels_train = fourier["labels_train"].astype(np.int64)
    snrs_train = fourier["snrs_train"].astype(np.int32)
    yv = soup["labels_val"].astype(np.int64)
    sv = soup["snrs_val"].astype(np.int32)
    yt = soup["labels_test"].astype(np.int64)
    stest = soup["snrs_test"].astype(np.int32)

    cv_train = cv["train_prob"].astype(np.float32)
    cv_val_list = []
    cv_test_list = []
    if not args.cvtrn_infer_from_valtest_only:
        cv_val_list.append(cv["val_prob"].astype(np.float32))
        cv_test_list.append(cv["test_prob"].astype(np.float32))
    if args.use_oof_cvtrn_only:
        print("[*] Strict mode: using only CV-TRN OOF fold-soup at inference.")
    elif args.cvtrn_infer_from_valtest_only:
        print("[*] CV-TRN inference mode: OOF CV is train-only; explicit val/test cache(s) are used for inference.")
        for path in args.cvtrn_valtest_caches:
            if not path or not os.path.exists(path):
                print(f"[!] Explicit CV-TRN val/test cache skipped: {path}")
                continue
            z = orig.load_npz(path)
            for key in ("labels_val", "snrs_val", "labels_test", "snrs_test"):
                orig.assert_same(soup, z, key, key, f"Explicit CV-TRN {os.path.basename(path)}")
            cv_val_list.append(z["val_prob"].astype(np.float32))
            cv_test_list.append(z["test_prob"].astype(np.float32))
            print(f"[*] Added explicit inference CV-TRN cache: {path}")
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
    if not cv_val_list or not cv_test_list:
        raise RuntimeError("No CV-TRN validation/test probabilities available for inference.")
    cv_val = orig.log_average(cv_val_list)
    cv_test = orig.log_average(cv_test_list)

    full_dataset = common.build_full_dataset(args)
    train_idx, val_idx, test_idx = common.make_aligned_split(full_dataset, args.split_seed)
    if not args.skip_alignment_check:
        common.check_alignment(full_dataset, val_idx, test_idx, args.alignment_cache)
    snrs_all = np.asarray(full_dataset.snrs, dtype=np.int32)
    q_train, q_val, q_test, q_acc = oof.build_quality_probs(args, train_idx, val_idx, test_idx, full_dataset, snrs_all)

    main_val = fourier["val_prob"].astype(np.float32) if args.use_fourier_oof_infer else soup["val_prob"].astype(np.float32)
    main_test = fourier["test_prob"].astype(np.float32) if args.use_fourier_oof_infer else soup["test_prob"].astype(np.float32)
    if args.disable_cvtrn:
        cv_train = fourier["train_prob"].astype(np.float32)
        cv_val = np.array(main_val, copy=True)
        cv_test = np.array(main_test, copy=True)
        print("[*] IQCC-Former ablation enabled: its probability blocks mirror KAN-Fourier.")
    extra_train = [e["cache"]["train_prob"].astype(np.float32) for e in extras]
    extra_val = [e["cache"]["val_prob"].astype(np.float32) for e in extras]
    extra_test = [e["cache"]["test_prob"].astype(np.float32) for e in extras]

    print("\nBaselines")
    base_val_m = base.metrics_from_probs(main_val, yv, sv)
    base.print_metrics_line("Main Fourier Val", base_val_m)
    base.print_metrics_line("Fourier OOF fold-soup Val", base.metrics_from_probs(fourier["val_prob"], yv, sv))
    cv_label = "IQCC disabled (main mirror)" if args.disable_cvtrn else "IQCC-Former"
    base.print_metrics_line(f"{cv_label} Val", base.metrics_from_probs(cv_val, yv, sv))
    base.print_metrics_line("GAMC-tree Val", base.metrics_from_probs(gamc["val_prob"], yv, sv))
    for e in extras:
        base.print_metrics_line(f"{e['name'][:22]} Val", base.metrics_from_probs(e["cache"]["val_prob"], yv, sv))

    print("\n[*] Building extra-aux Fourier-aware train/val/test meta-features")
    x_train = build_features(fourier["train_prob"], cv_train, gamc["train_prob"], gamc["train_member_probs"], q_train, extra_train)
    x_val = build_features(main_val, cv_val, gamc["val_prob"], gamc["val_member_probs"], q_val, extra_val)
    x_test = build_features(main_test, cv_test, gamc["test_prob"], gamc["test_member_probs"], q_test, extra_test)
    print(f"Feature dim: {x_train.shape[1]} | train={len(x_train):,} | val={len(x_val):,} | test={len(x_test):,}")

    weights = orig.sample_weights(snrs_train, fourier["train_prob"], labels_train)
    stability_enabled = int(args.meta_stability_folds) > 1
    base_train_m = base.metrics_from_probs(fourier["train_prob"], labels_train, snrs_train)
    main_train_oof = fourier["train_prob"].astype(np.float32)
    allowed_branches = set(args.select_branches)
    need_ensemble = (not allowed_branches) or ("extraaux_oof_ensemble" in allowed_branches)
    ensemble_members = set(args.ensemble_branches)
    branches = []
    configs = [
        ("xgb_d2_620", orig.xgb_model(args, "xgb_d2_620", args.random_state + 11)),
        ("xgb_d3_520", orig.xgb_model(args, "xgb_d3_520", args.random_state + 23)),
        ("xgb_d4_400", orig.xgb_model(args, "xgb_d4_400", args.random_state + 37)),
        ("et_depth20", orig.et_model(args, args.random_state + 53)),
    ]
    for name, clf in configs:
        needed_for_ensemble = need_ensemble and (not ensemble_members or name in ensemble_members)
        if allowed_branches and name not in allowed_branches and not needed_for_ensemble:
            print(f"[*] Skipping meta branch {name}: not requested by --select_branches")
            continue
        print(f"[*] Training extra-aux Fourier-aware meta-classifier: {name}")
        clf, _, _ = fit_or_load_estimator(
            clf,
            x_train,
            labels_train,
            sample_weight=weights,
            cache_dir=args.model_cache_dir,
            reuse=args.reuse_models,
            namespace=f"extraaux_meta_{name}",
            source_paths=[
                args.fourier_oof_cache,
                args.cvtrn_oof_cache,
                args.gamc_oof_cache,
                *args.extra_aux_oof_caches,
            ],
            context={
                "builder": "fourier_oof_extraaux_meta_v1",
                "split_seed": args.split_seed,
                "quality_estimators": args.quality_estimators,
                "quality_max_depth": args.quality_max_depth,
                "quality_learning_rate": args.quality_learning_rate,
            },
        )
        pv = orig.aligned_proba(clf, x_val)
        pt = orig.aligned_proba(clf, x_test)
        meta_oof = None
        if stability_enabled:
            meta_oof = train_meta_oof(args, name, x_train, labels_train, snrs_train, weights)
        rows = search_configs(
            main_val,
            pv,
            yv,
            sv,
            base_val_m,
            args,
            name,
            main_oof=main_train_oof,
            meta_oof=meta_oof,
            labels_oof=labels_train,
            snrs_oof=snrs_train,
            base_oof_m=base_train_m,
        )
        branches.append({"name": name, "val": pv, "test": pt, "oof": meta_oof, "rows": rows})

    if need_ensemble and branches:
        ensemble_sources = [b for b in branches if not ensemble_members or b["name"] in ensemble_members]
        if not ensemble_sources:
            raise RuntimeError("No trained branches match --ensemble_branches.")
        ens_val = orig.log_average([b["val"] for b in ensemble_sources])
        ens_test = orig.log_average([b["test"] for b in ensemble_sources])
        ens_oof = None
        if stability_enabled and all(b.get("oof") is not None for b in ensemble_sources):
            ens_oof = orig.log_average([b["oof"] for b in ensemble_sources])
        ens_rows = search_configs(
            main_val,
            ens_val,
            yv,
            sv,
            base_val_m,
            args,
            "extraaux_oof_ensemble",
            main_oof=main_train_oof,
            meta_oof=ens_oof,
            labels_oof=labels_train,
            snrs_oof=snrs_train,
            base_oof_m=base_train_m,
        )
        branches.append({"name": "extraaux_oof_ensemble", "val": ens_val, "test": ens_test, "oof": ens_oof, "rows": ens_rows})

    all_rows = []
    for b in branches:
        if allowed_branches and b["name"] not in allowed_branches:
            continue
        all_rows.extend(b["rows"])
    all_rows.sort(key=lambda r: r["score"], reverse=True)
    if not all_rows:
        raise RuntimeError("No validation config passed the requested floors. Relax the min_val_* thresholds.")
    orig.print_top(all_rows, 20)
    orig.save_csv(relpath("results", f"{suffix}_search_top.csv"), all_rows, args.save_top_records)

    best = all_rows[0]
    branch = next(b for b in branches if b["name"] == best["branch"])
    final_val, val_gate, val_use, val_alpha, val_policy = apply_stacked_with_policy(main_val, branch["val"], best, args)
    final_test, gate, use, alpha, test_policy = apply_stacked_with_policy(main_test, branch["test"], best, args)
    final_m = base.metrics_from_probs(final_test, yt, stest)
    diag = base.switch_diagnostics(main_test, final_test, gate, use, alpha, stest)

    selected = {
        "best": best,
        "branch": branch["name"],
        "blind_cqi_val_bin_acc": q_acc,
        "fourier_oof_cache": args.fourier_oof_cache,
        "gamc_oof_cache": args.gamc_oof_cache,
        "cvtrn_oof_cache": args.cvtrn_oof_cache,
        "disable_cvtrn": bool(args.disable_cvtrn),
        "extra_aux_oof_caches": args.extra_aux_oof_caches,
        "forbid_transition_pairs": [list(x) for x in getattr(args, "_forbid_transition_pairs", ())],
        "transition_risk_pairs": [list(x) for x in getattr(args, "_transition_risk_pairs", ())],
        "transition_risk_penalty": float(args.transition_risk_penalty),
        "val_transition_policy": val_policy,
        "test_transition_policy": test_policy,
        "protocol": "Fourier-aware train-OOF utility estimator with auxiliary probability predictors",
    }
    with open(relpath("results", f"{suffix}_selected_config.json"), "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 144)
    print("Final test report")
    print("=" * 144)
    base.print_metrics_line("Main Fourier Test", base.metrics_from_probs(main_test, yt, stest))
    base.print_metrics_line(f"{cv_label} Test", base.metrics_from_probs(cv_test, yt, stest))
    base.print_metrics_line("GAMC-tree Test", base.metrics_from_probs(gamc["test_prob"], yt, stest))
    for e in extras:
        base.print_metrics_line(f"{e['name'][:22]} Test", base.metrics_from_probs(e["cache"]["test_prob"], yt, stest))
    base.print_metrics_line("Extra-aux meta fusion Test", final_m)
    print("-" * 144)
    base_m_test = base.metrics_from_probs(main_test, yt, stest)
    print(f"Delta vs main Fourier overall:    {final_m['overall_acc'] - base_m_test['overall_acc']:+.4f} pp")
    print(f"Delta vs main Fourier negative:   {final_m['negative_acc'] - base_m_test['negative_acc']:+.4f} pp")
    print(f"Delta vs main Fourier edge:       {final_m['edge_low_acc'] - base_m_test['edge_low_acc']:+.4f} pp")
    print(f"Delta vs main Fourier transition: {final_m['transition_acc'] - base_m_test['transition_acc']:+.4f} pp")
    print(f"Delta vs main Fourier high:       {final_m['high_acc'] - base_m_test['high_acc']:+.4f} pp")
    print(f"Final diagnostics: {diag}")
    print("=" * 144)

    pred = final_m["pred"].astype(np.int64)
    np.savez_compressed(
        relpath("results", f"{suffix}_predictions.npz"),
        labels=yt.astype(np.int64),
        snrs=stest.astype(np.int32),
        pred=pred,
        final_prob=final_test.astype(np.float32),
        final_val_prob=final_val.astype(np.float32),
        meta_prob=branch["test"].astype(np.float32),
        meta_val_prob=branch["val"].astype(np.float32),
        use_candidate=use.astype(bool),
        labels_val=yv.astype(np.int64),
        snrs_val=sv.astype(np.int32),
        use_candidate_val=val_use.astype(bool),
        mod_classes=soup.get("mod_classes", np.asarray(common.DEFAULT_MOD_CLASSES)),
    )
    curve_path = relpath("results", f"accuracy_vs_snr_{suffix}.png")
    base.plot_curve(final_m["by_snr"], curve_path, "Accuracy vs SNR: Extra-Aux OOF Meta Fusion")
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
            "Extra-Aux OOF Meta Fusion",
        )
        print(f"[*] {snr_value} dB confusion matrix saved: {cm_path} (Acc={acc:.2f}%)")


if __name__ == "__main__":
    main()
