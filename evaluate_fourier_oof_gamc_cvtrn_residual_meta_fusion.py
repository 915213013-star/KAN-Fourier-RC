import argparse
import csv
import json
import os

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier

import evaluate_fourier_gamc_cvtrn_stacked_residual_fusion as st
import evaluate_fourier_gamc_cvtrn_train_oof_meta_fusion as oof
import evaluate_fourier_gamc_multi_cvtrn_blind_quality_xgb_fusion as bq
import evaluate_greedy_soup_gamc_protected_residual_fusion as base
import train_cv_trn_aux_2016 as common
from model_cache_utils import fit_or_load_estimator


def project_root():
    return os.path.dirname(os.path.abspath(__file__))


def relpath(*parts):
    return os.path.join(project_root(), *parts)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Fourier-OOF residual meta fusion. This explicitly trains the meta-router with "
            "OOF Fourier main-model probabilities, so it can learn when the Fourier main branch "
            "is likely to be wrong."
        )
    )
    p.add_argument("--split_seed", type=int, default=1)
    p.add_argument("--random_state", type=int, default=2026)
    p.add_argument("--soup_prob_cache", type=str, default=relpath("results", "greedy_soup_identity_valtest_probs_for_gamc_fusion.npz"))
    p.add_argument("--fourier_oof_cache", type=str, default=relpath("results", "fourier_main_oof_mseed77_f3e220_split1_trainvaltest_probs_for_meta.npz"))
    p.add_argument("--gamc_oof_cache", type=str, default=relpath("results", "gamc_oof_tree_split1_trainvaltest_probs_for_meta.npz"))
    p.add_argument("--cvtrn_oof_cache", type=str, default=relpath("results", "cv_trn_aux_v2_oof_mseed41_f3e240_split1_trainvaltest_probs_for_meta.npz"))
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
    p.add_argument("--use_fourier_oof_infer", action="store_true", help="Diagnostic/ablation: use Fourier fold-soup val/test as main instead of greedy soup.")
    p.add_argument("--output_suffix", type=str, default="fourier_oof_gamc_cvtrn_residual_meta_split1")

    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--blend_alphas", type=float, nargs="+", default=[0.50, 0.65, 0.80])
    p.add_argument("--meta_conf_thresholds", type=float, nargs="+", default=[0.00, 0.25, 0.35, 0.45, 0.55])
    p.add_argument("--advantage_thresholds", type=float, nargs="+", default=[-0.20, -0.10, 0.00, 0.05, 0.10])
    p.add_argument("--max_change_rates", type=float, nargs="+", default=[6.0, 8.0, 10.0, 12.0])
    p.add_argument("--hard_keep_conf", type=float, default=0.93)
    p.add_argument("--hard_keep_margin", type=float, default=0.70)

    p.add_argument("--score_overall_weight", type=float, default=1.0)
    p.add_argument("--score_negative_gain_weight", type=float, default=0.020)
    p.add_argument("--score_edge_gain_weight", type=float, default=0.020)
    p.add_argument("--score_transition_gain_weight", type=float, default=0.010)
    p.add_argument("--score_high_penalty", type=float, default=3.00)
    p.add_argument("--high_tolerance", type=float, default=0.05)
    p.add_argument("--score_changed_high_penalty", type=float, default=0.015)
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
    p.add_argument("--data_path", type=str, default=relpath("raw_data", "RML2016.10a_dict.pkl"))
    p.add_argument("--cache_dir", type=str, default=relpath("feature_cache"))
    p.add_argument("--alignment_cache", type=str, default=relpath("results", "greedy_soup_identity_valtest_probs_for_gamc_fusion.npz"))
    p.add_argument("--skip_alignment_check", action="store_true")
    p.add_argument("--save_top_records", type=int, default=240)
    p.add_argument("--cm_snrs", type=int, nargs="+", default=[0, -6])
    return p.parse_args()


def load_npz(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def norm(p):
    return oof.norm(p)


def log_average(prob_list):
    return oof.log_average(prob_list)


def assert_same(a, b, key_a, key_b, name):
    if not np.all(np.asarray(a[key_a]) == np.asarray(b[key_b])):
        raise RuntimeError(f"{name} alignment mismatch: {key_a} vs {key_b}")


def build_features(main_prob, cv_prob, gamc_prob, member_probs, qprob):
    main = norm(main_prob)
    cv = norm(cv_prob)
    gamc = norm(gamc_prob)
    if member_probs is None:
        members = []
    else:
        members = [norm(p) for p in np.asarray(member_probs)]
    qfeat = bq.quality_probability_features(qprob)
    parts = []
    parts.extend(oof.expert_blocks(main))
    parts.extend(oof.expert_blocks(cv))
    parts.extend(oof.expert_blocks(gamc))
    parts.append(oof.pair_blocks(main, cv))
    parts.append(oof.pair_blocks(main, gamc))
    parts.append(oof.pair_blocks(cv, gamc))
    parts.append(qfeat.astype(np.float32))
    stack = np.stack([main, cv, gamc] + members, axis=0)
    parts.append(stack.mean(axis=0).astype(np.float32))
    parts.append(stack.std(axis=0).astype(np.float32))
    parts.append(stack.max(axis=0).astype(np.float32))
    parts.append(stack.min(axis=0).astype(np.float32))
    for p in members:
        parts.extend(oof.expert_blocks(p))
        parts.append(oof.pair_blocks(main, p))
        parts.append(oof.pair_blocks(cv, p))
        parts.append(oof.pair_blocks(gamc, p))
    x = np.concatenate(parts, axis=1)
    return np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)


def get_member_probs(cache, split_name):
    key = f"{split_name}_member_probs"
    if key in cache:
        return cache[key]
    print(f"[*] GAMC cache has no {key}; treating it as a single probability expert.")
    return None


def aligned_proba(clf, x):
    return oof.aligned_proba(clf, x)


def sample_weights(snrs, fourier_prob, labels):
    w = oof.meta_sample_weights(snrs)
    fp = norm(fourier_prob).argmax(axis=1)
    wrong = fp != labels.astype(np.int64)
    w[wrong] *= 1.35
    return w.astype(np.float32)


def xgb_model(args, name, seed):
    if name == "xgb_d2_620":
        depth, est, lr, child = 2, 620, 0.032, 2.0
    elif name == "xgb_d3_520":
        depth, est, lr, child = 3, 520, 0.030, 2.0
    else:
        depth, est, lr, child = 4, 400, 0.028, 3.0
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


def et_model(args, seed):
    return ExtraTreesClassifier(
        n_estimators=520,
        max_depth=20,
        min_samples_leaf=8,
        max_features="sqrt",
        n_jobs=int(args.xgb_jobs),
        random_state=int(seed),
    )


def save_csv(path, rows, n=None):
    rows = rows if n is None else rows[: max(1, int(n))]
    if not rows:
        return
    keys = sorted(set(k for row in rows for k in row.keys()))
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[*] CSV saved: {path}")


def print_top(rows, n=20):
    print("\nTop Fourier-OOF residual meta validation configs")
    for i, r in enumerate(rows[:n], 1):
        print(
            f"{i:02d}. {r['branch']:<18} score={r['score']:.3f} | overall={r['overall_acc']:.3f}% | "
            f"neg={r['negative_acc']:.3f}% | edge={r['edge_low_acc']:.3f}% | high={r['high_acc']:.3f}% | "
            f"alpha={r['alpha']} conf={r['meta_conf_thr']} adv={r['advantage_thr']} maxchg={r['max_change_rate']}"
        )


def main():
    args = parse_args()
    os.makedirs(relpath("results"), exist_ok=True)
    suffix = args.output_suffix
    print("=" * 144)
    print("Fourier-OOF residual meta fusion")
    print("=" * 144)
    print("Academic protocol:")
    print("  - Fourier greedy soup remains the deployed main model.")
    print("  - Meta-classifier is trained with train-split OOF Fourier, CV-TRN, and GAMC probabilities.")
    print("  - Validation labels select blend/gate settings only.")
    print("  - Test labels are used only once for the final report.")
    print("  - True SNR is never used as an inference feature.")

    soup = load_npz(args.soup_prob_cache)
    fourier = load_npz(args.fourier_oof_cache)
    gamc = load_npz(args.gamc_oof_cache)
    cv = load_npz(args.cvtrn_oof_cache)

    for key in ("labels_train", "snrs_train"):
        assert_same(fourier, gamc, key, key, "Fourier OOF vs GAMC OOF")
        assert_same(fourier, cv, key, key, "Fourier OOF vs CVTRN OOF")
    for key in ("labels_val", "snrs_val", "labels_test", "snrs_test"):
        assert_same(soup, fourier, key, key, "Fourier soup vs Fourier OOF")
        assert_same(soup, gamc, key, key, "Fourier soup vs GAMC OOF")
        assert_same(soup, cv, key, key, "Fourier soup vs CVTRN OOF")
    print("[*] Alignment check passed for all OOF caches.")

    labels_train = fourier["labels_train"].astype(np.int64)
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
            z = load_npz(path)
            for key in ("labels_val", "snrs_val", "labels_test", "snrs_test"):
                assert_same(soup, z, key, key, f"Optional CV-TRN {os.path.basename(path)}")
            cv_val_list.append(z["val_prob"].astype(np.float32))
            cv_test_list.append(z["test_prob"].astype(np.float32))
            print(f"[*] Added inference CV-TRN cache: {path}")
    cv_val = log_average(cv_val_list)
    cv_test = log_average(cv_test_list)

    full_dataset = common.build_full_dataset(args)
    train_idx, val_idx, test_idx = common.make_aligned_split(full_dataset, args.split_seed)
    if not args.skip_alignment_check:
        common.check_alignment(full_dataset, val_idx, test_idx, args.alignment_cache)
    snrs_all = np.asarray(full_dataset.snrs, dtype=np.int32)
    q_train, q_val, q_test, q_acc = oof.build_quality_probs(args, train_idx, val_idx, test_idx, full_dataset, snrs_all)

    main_val = fourier["val_prob"].astype(np.float32) if args.use_fourier_oof_infer else soup["val_prob"].astype(np.float32)
    main_test = fourier["test_prob"].astype(np.float32) if args.use_fourier_oof_infer else soup["test_prob"].astype(np.float32)
    if args.use_fourier_oof_infer:
        print("[!] Diagnostic mode: using Fourier fold-soup as main at val/test.")

    print("\nBaselines")
    base_val_m = base.metrics_from_probs(main_val, yv, sv)
    base.print_metrics_line("Main Fourier Val", base_val_m)
    base.print_metrics_line("Fourier OOF fold-soup Val", base.metrics_from_probs(fourier["val_prob"], yv, sv))
    base.print_metrics_line("CV-TRN Val", base.metrics_from_probs(cv_val, yv, sv))
    base.print_metrics_line("GAMC Val", base.metrics_from_probs(gamc["val_prob"], yv, sv))

    print("\n[*] Building Fourier-aware train/val/test meta-features")
    x_train = build_features(fourier["train_prob"], cv_train, gamc["train_prob"], get_member_probs(gamc, "train"), q_train)
    x_val = build_features(main_val, cv_val, gamc["val_prob"], get_member_probs(gamc, "val"), q_val)
    x_test = build_features(main_test, cv_test, gamc["test_prob"], get_member_probs(gamc, "test"), q_test)
    print(f"Feature dim: {x_train.shape[1]} | train={len(x_train):,} | val={len(x_val):,} | test={len(x_test):,}")

    weights = sample_weights(snrs_train, fourier["train_prob"], labels_train)
    branches = []
    configs = [
        ("xgb_d2_620", xgb_model(args, "xgb_d2_620", args.random_state + 11)),
        ("xgb_d3_520", xgb_model(args, "xgb_d3_520", args.random_state + 23)),
        ("xgb_d4_400", xgb_model(args, "xgb_d4_400", args.random_state + 37)),
        ("et_depth20", et_model(args, args.random_state + 53)),
    ]
    for name, clf in configs:
        print(f"[*] Training Fourier-aware meta-classifier: {name}")
        clf, _, _ = fit_or_load_estimator(
            clf,
            x_train,
            labels_train,
            sample_weight=weights,
            cache_dir=args.model_cache_dir,
            reuse=args.reuse_models,
            namespace=f"residual_meta_{name}",
            source_paths=[args.fourier_oof_cache, args.cvtrn_oof_cache, args.gamc_oof_cache],
            context={
                "builder": "fourier_oof_residual_meta_v1",
                "split_seed": args.split_seed,
                "quality_estimators": args.quality_estimators,
                "quality_max_depth": args.quality_max_depth,
                "quality_learning_rate": args.quality_learning_rate,
            },
        )
        pv = aligned_proba(clf, x_val)
        pt = aligned_proba(clf, x_test)
        rows = st.search_configs(main_val, pv, yv, sv, base_val_m, args, name)
        branches.append({"name": name, "val": pv, "test": pt, "rows": rows})

    ens_val = log_average([b["val"] for b in branches])
    ens_test = log_average([b["test"] for b in branches])
    ens_rows = st.search_configs(main_val, ens_val, yv, sv, base_val_m, args, "fourier_oof_ensemble")
    branches.append({"name": "fourier_oof_ensemble", "val": ens_val, "test": ens_test, "rows": ens_rows})

    all_rows = []
    for b in branches:
        all_rows.extend(b["rows"])
    all_rows.sort(key=lambda r: r["score"], reverse=True)
    print_top(all_rows, 20)
    save_csv(relpath("results", f"{suffix}_search_top.csv"), all_rows, args.save_top_records)

    best = all_rows[0]
    branch = next(b for b in branches if b["name"] == best["branch"])
    final_val, val_gate, val_use, val_alpha = st.apply_stacked(main_val, branch["val"], best, args)
    final_test, gate, use, alpha = st.apply_stacked(main_test, branch["test"], best, args)
    final_m = base.metrics_from_probs(final_test, yt, stest)
    diag = base.switch_diagnostics(main_test, final_test, gate, use, alpha, stest)

    selected = {
        "best": best,
        "branch": branch["name"],
        "blind_cqi_val_bin_acc": q_acc,
        "fourier_oof_cache": args.fourier_oof_cache,
        "gamc_oof_cache": args.gamc_oof_cache,
        "cvtrn_oof_cache": args.cvtrn_oof_cache,
        "protocol": "Fourier-aware train-OOF meta-classifier; validation-only blend/gate selection",
    }
    with open(relpath("results", f"{suffix}_selected_config.json"), "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 144)
    print("Final test report")
    print("=" * 144)
    base.print_metrics_line("Main Fourier Test", base.metrics_from_probs(main_test, yt, stest))
    base.print_metrics_line("CV-TRN Test", base.metrics_from_probs(cv_test, yt, stest))
    base.print_metrics_line("GAMC Test", base.metrics_from_probs(gamc["test_prob"], yt, stest))
    base.print_metrics_line("Fourier-aware meta fusion Test", final_m)
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
    base.plot_curve(final_m["by_snr"], curve_path, "Accuracy vs SNR: Fourier-aware OOF Meta Fusion")
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
            "Fourier-aware OOF Meta Fusion",
        )
        print(f"[*] {snr_value} dB confusion matrix saved: {cm_path} (Acc={acc:.2f}%)")


if __name__ == "__main__":
    main()
