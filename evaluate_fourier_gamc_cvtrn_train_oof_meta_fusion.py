import argparse
import csv
import json
import os

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

import evaluate_fourier_gamc_cvtrn_gain_risk_router_fusion as gr
import evaluate_fourier_gamc_cvtrn_stacked_residual_fusion as st
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
            "Train-split OOF meta-router fusion. The meta-classifier is trained on train-OOF "
            "auxiliary probabilities, validation only selects blend/gate settings, and test is "
            "reported once."
        )
    )
    p.add_argument("--split_seed", type=int, default=1)
    p.add_argument("--random_state", type=int, default=2026)
    p.add_argument("--soup_prob_cache", type=str, default=relpath("results", "greedy_soup_identity_valtest_probs_for_gamc_fusion.npz"))
    p.add_argument("--gamc_oof_cache", type=str, default=relpath("results", "gamc_oof_tree_split1_trainvaltest_probs_for_meta.npz"))
    p.add_argument("--cvtrn_oof_cache", type=str, default=relpath("results", "cv_trn_aux_v2_oof_mseed41_split1_trainvaltest_probs_for_meta.npz"))
    p.add_argument(
        "--cvtrn_valtest_caches",
        type=str,
        nargs="*",
        default=[
            relpath("results", "cv_trn_aux_v2_soup_tta_split1_valtest_probs_for_fusion.npz"),
            relpath("results", "cv_trn_aux_v2_w8d96_soup_tta_split1_valtest_probs_for_fusion.npz"),
        ],
        help="Optional full-train CV-TRN val/test caches to log-average with the OOF fold-soup at inference.",
    )
    p.add_argument(
        "--use_oof_cvtrn_only",
        action="store_true",
        help="Use only the CV-TRN OOF fold-soup val/test probabilities at inference.",
    )
    p.add_argument("--output_suffix", type=str, default="fourier_gamc_cvtrn_train_oof_meta_split1")

    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--blend_alphas", type=float, nargs="+", default=[0.35, 0.50, 0.65, 0.80, 0.90, 1.00])
    p.add_argument("--meta_conf_thresholds", type=float, nargs="+", default=[0.00, 0.25, 0.35, 0.45, 0.55, 0.65])
    p.add_argument("--advantage_thresholds", type=float, nargs="+", default=[-0.20, -0.10, 0.00, 0.05, 0.10])
    p.add_argument("--max_change_rates", type=float, nargs="+", default=[6.0, 8.0, 10.0, 12.0, 16.0, 22.0, 100.0])
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
    p.add_argument("--save_top_records", type=int, default=220)
    p.add_argument("--cm_snrs", type=int, nargs="+", default=[0, -6])
    return p.parse_args()


def load_npz(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def norm(p):
    return gr.norm(p)


def log_average(prob_list):
    logs = [np.log(norm(p) + 1e-12) for p in prob_list]
    z = np.mean(np.stack(logs, axis=0), axis=0)
    z -= z.max(axis=1, keepdims=True)
    return norm(np.exp(z))


def assert_same(a, b, key_a, key_b, name):
    if not np.all(np.asarray(a[key_a]) == np.asarray(b[key_b])):
        raise RuntimeError(f"{name} alignment mismatch: {key_a} vs {key_b}")


def margin(p):
    return gr.margin(norm(p))


def one_hot(x, n):
    return np.eye(n, dtype=np.float32)[x.astype(np.int64)]


def expert_blocks(p):
    p = norm(p)
    pred = p.argmax(1)
    return [
        p.astype(np.float32),
        np.log(np.clip(p, 1e-8, 1.0)).astype(np.float32),
        one_hot(pred, base.NUM_CLASSES),
        np.stack([p.max(1), margin(p), gr.entropy(p)], axis=1).astype(np.float32),
    ]


def pair_blocks(a, b):
    a, b = norm(a), norm(b)
    ap, bp = a.argmax(1), b.argmax(1)
    idx = np.arange(len(a))
    return np.stack(
        [
            (ap == bp).astype(np.float32),
            np.abs(a - b).sum(1),
            (a * b).sum(1),
            b[idx, ap],
            a[idx, bp],
            b.max(1) - a.max(1),
            margin(b) - margin(a),
            gr.entropy(b) - gr.entropy(a),
        ],
        axis=1,
    ).astype(np.float32)


def build_meta_features(cv_prob, gamc_prob, member_probs, qprob):
    cv = norm(cv_prob)
    gamc = norm(gamc_prob)
    members = [norm(p) for p in np.asarray(member_probs)]
    qfeat = bq.quality_probability_features(qprob)

    parts = []
    parts.extend(expert_blocks(cv))
    parts.extend(expert_blocks(gamc))
    parts.append(pair_blocks(cv, gamc))
    parts.append(qfeat.astype(np.float32))

    stack = np.stack([cv, gamc] + members, axis=0)
    parts.append(stack.mean(axis=0).astype(np.float32))
    parts.append(stack.std(axis=0).astype(np.float32))
    parts.append(stack.max(axis=0).astype(np.float32))
    parts.append(stack.min(axis=0).astype(np.float32))

    for p in members:
        parts.extend(expert_blocks(p))
        parts.append(pair_blocks(cv, p))
        parts.append(pair_blocks(gamc, p))

    x = np.concatenate(parts, axis=1)
    return np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)


def meta_sample_weights(snrs):
    s = np.asarray(snrs, dtype=np.int32)
    w = np.ones(len(s), dtype=np.float32)
    w[s < 0] = 1.35
    w[np.isin(s, [-18, -16])] = 2.20
    w[np.isin(s, [-14, -12])] = 1.85
    w[np.isin(s, [-10, -8, -6, -4, -2])] = 1.55
    w[s >= 8] = 0.82
    return w


def xgb_model(args, name, seed):
    if name == "xgb_d2_520":
        depth, est, lr, child = 2, 520, 0.035, 2.0
    elif name == "xgb_d3_420":
        depth, est, lr, child = 3, 420, 0.032, 2.0
    else:
        depth, est, lr, child = 4, 340, 0.030, 3.0
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
        eval_metric="mlogloss",
        n_jobs=int(args.xgb_jobs),
        random_state=int(seed),
        verbosity=0,
    )


def et_model(args, seed):
    return ExtraTreesClassifier(
        n_estimators=420,
        max_depth=18,
        min_samples_leaf=10,
        max_features="sqrt",
        n_jobs=int(args.xgb_jobs),
        random_state=int(seed),
    )


def aligned_proba(clf, x):
    p = clf.predict_proba(x)
    out = np.zeros((x.shape[0], base.NUM_CLASSES), dtype=np.float32)
    for j, c in enumerate(clf.classes_):
        out[:, int(c)] = p[:, j]
    return norm(out)


def build_quality_probs(args, train_idx, val_idx, test_idx, full_dataset, snrs):
    print("\n" + "=" * 144)
    print("[*] Cross-fitting blind CQI on train split")
    print("=" * 144)
    x_train = bq.extract_blind_quality_features(full_dataset, train_idx, args.quality_chunk_size)
    x_val = bq.extract_blind_quality_features(full_dataset, val_idx, args.quality_chunk_size)
    x_test = bq.extract_blind_quality_features(full_dataset, test_idx, args.quality_chunk_size)

    extra_paths = list(getattr(args, "quality_extra_feature_caches", []) or [])
    if extra_paths:
        extra_train, extra_val, extra_test = [], [], []
        for path in extra_paths:
            z = np.load(path, allow_pickle=True)
            key = "features" if "features" in z.files else z.files[0]
            feat = np.asarray(z[key], dtype=np.float32)
            feat = np.nan_to_num(feat, nan=0.0, posinf=1e6, neginf=-1e6)
            if feat.shape[0] != len(snrs):
                raise ValueError(f"quality extra feature cache has {feat.shape[0]} rows, expected {len(snrs)}: {path}")
            extra_train.append(feat[train_idx])
            extra_val.append(feat[val_idx])
            extra_test.append(feat[test_idx])
            print(f"[*] Added CQI extra blind features: {path} key={key} dim={feat.shape[1] if feat.ndim > 1 else 1}")
        x_train = np.concatenate([x_train] + extra_train, axis=1).astype(np.float32)
        x_val = np.concatenate([x_val] + extra_val, axis=1).astype(np.float32)
        x_test = np.concatenate([x_test] + extra_test, axis=1).astype(np.float32)
        print(f"[*] Enhanced CQI feature dim: {x_train.shape[1]}")

    yq = bq.snr_to_quality_bin(snrs[train_idx])
    composite = np.asarray([f"{int(q)}_{int(s)}" for q, s in zip(yq, snrs[train_idx])])
    q_train = np.zeros((len(train_idx), 5), dtype=np.float32)
    skf = StratifiedKFold(n_splits=int(args.folds), shuffle=True, random_state=int(args.random_state + 707))
    for fold, (tr, va) in enumerate(skf.split(x_train, composite), 1):
        clf = bq.make_quality_model(args)
        clf.random_state = int(args.random_state + 800 + fold)
        clf, _, _ = fit_or_load_estimator(
            clf,
            x_train[tr],
            yq[tr],
            cache_dir=str(getattr(args, "model_cache_dir", "")),
            reuse=bool(getattr(args, "reuse_models", False)),
            namespace=f"blind_cqi_fold{fold}",
            source_paths=[
                *list(getattr(args, "model_cache_source_paths", []) or []),
                *extra_paths,
            ],
            context={
                "builder": "blind_cqi_crossfit_v2",
                "split_seed": int(getattr(args, "split_seed", 1)),
                "fold": int(fold),
                "folds": int(args.folds),
                "feature_dim": int(x_train.shape[1]),
            },
        )
        q_train[va] = clf.predict_proba(x_train[va]).astype(np.float32)
        print(f"    CQI fold {fold}/{args.folds} done")
    clf = bq.make_quality_model(args)
    clf, _, _ = fit_or_load_estimator(
        clf,
        x_train,
        yq,
        cache_dir=str(getattr(args, "model_cache_dir", "")),
        reuse=bool(getattr(args, "reuse_models", False)),
        namespace="blind_cqi_final",
        source_paths=[
            *list(getattr(args, "model_cache_source_paths", []) or []),
            *extra_paths,
        ],
        context={
            "builder": "blind_cqi_crossfit_v2",
            "split_seed": int(getattr(args, "split_seed", 1)),
            "fold": "final",
            "folds": int(args.folds),
            "feature_dim": int(x_train.shape[1]),
        },
    )
    q_val = clf.predict_proba(x_val).astype(np.float32)
    q_test = clf.predict_proba(x_test).astype(np.float32)
    q_train, q_val, q_test = norm(q_train), norm(q_val), norm(q_test)
    q_acc = float((q_val.argmax(1) == bq.snr_to_quality_bin(snrs[val_idx])).mean() * 100.0)
    print(f"Blind CQI val bin accuracy: {q_acc:.3f}%")
    return q_train, q_val, q_test, q_acc


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
    print("\nTop train-OOF meta validation configs")
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
    print("Train-OOF meta-router fusion: Fourier + GAMC-tree OOF + CV-TRN OOF")
    print("=" * 144)
    print("Academic protocol:")
    print("  - Fourier soup remains the main model.")
    print("  - Auxiliary meta-classifier is trained on train-split OOF probabilities.")
    print("  - Original validation labels select blend/gate settings only.")
    print("  - Test labels are used only once for the final report.")
    print("  - True SNR is never used as an inference feature.")

    soup = load_npz(args.soup_prob_cache)
    gamc = load_npz(args.gamc_oof_cache)
    cv = load_npz(args.cvtrn_oof_cache)

    assert_same(gamc, cv, "labels_train", "labels_train", "GAMC OOF vs CVTRN OOF")
    assert_same(gamc, cv, "snrs_train", "snrs_train", "GAMC OOF vs CVTRN OOF")
    for k in ("labels_val", "snrs_val", "labels_test", "snrs_test"):
        assert_same(soup, gamc, k, k, "Fourier vs GAMC OOF")
        assert_same(soup, cv, k, k, "Fourier vs CVTRN OOF")
    print("[*] Alignment check passed for Fourier val/test and OOF train caches.")

    labels_train = gamc["labels_train"].astype(np.int64)
    snrs_train = gamc["snrs_train"].astype(np.int32)
    yv = soup["labels_val"].astype(np.int64)
    sv = soup["snrs_val"].astype(np.int32)
    yt = soup["labels_test"].astype(np.int64)
    stest = soup["snrs_test"].astype(np.int32)

    cv_train = cv["train_prob"].astype(np.float32)
    cv_val_list = [cv["val_prob"].astype(np.float32)]
    cv_test_list = [cv["test_prob"].astype(np.float32)]
    if args.use_oof_cvtrn_only:
        print("[*] Strict mode: using only CV-TRN OOF fold-soup val/test probabilities.")
    else:
        for path in args.cvtrn_valtest_caches:
            if not path or not os.path.exists(path):
                print(f"[!] Optional CV-TRN val/test cache skipped: {path}")
                continue
            z = load_npz(path)
            for k in ("labels_val", "snrs_val", "labels_test", "snrs_test"):
                assert_same(soup, z, k, k, f"Optional CV-TRN {os.path.basename(path)}")
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
    q_train, q_val, q_test, q_acc = build_quality_probs(args, train_idx, val_idx, test_idx, full_dataset, snrs_all)

    print("\nBaselines")
    nv = norm(soup["val_prob"])
    nt = norm(soup["test_prob"])
    base_val_m = base.metrics_from_probs(nv, yv, sv)
    base.print_metrics_line("Fourier Val", base_val_m)
    base.print_metrics_line("CV-TRN OOF/infer Val", base.metrics_from_probs(cv_val, yv, sv))
    base.print_metrics_line("GAMC-tree OOF Val", base.metrics_from_probs(gamc["val_prob"], yv, sv))

    print("\n[*] Building train/val/test meta-features")
    x_train = build_meta_features(cv_train, gamc["train_prob"], gamc["train_member_probs"], q_train)
    x_val = build_meta_features(cv_val, gamc["val_prob"], gamc["val_member_probs"], q_val)
    x_test = build_meta_features(cv_test, gamc["test_prob"], gamc["test_member_probs"], q_test)
    print(f"Feature dim: {x_train.shape[1]} | train={len(x_train):,} | val={len(x_val):,} | test={len(x_test):,}")

    weights = meta_sample_weights(snrs_train)
    branches = []
    configs = [
        ("xgb_d2_520", xgb_model(args, "xgb_d2_520", args.random_state + 11)),
        ("xgb_d3_420", xgb_model(args, "xgb_d3_420", args.random_state + 23)),
        ("xgb_d4_340", xgb_model(args, "xgb_d4_340", args.random_state + 37)),
        ("et_depth18", et_model(args, args.random_state + 53)),
    ]
    for name, clf in configs:
        print(f"[*] Training train-OOF meta-classifier: {name}")
        if isinstance(clf, ExtraTreesClassifier):
            clf.fit(x_train, labels_train, sample_weight=weights)
        else:
            clf.fit(x_train, labels_train, sample_weight=weights)
        pv = aligned_proba(clf, x_val)
        pt = aligned_proba(clf, x_test)
        rows = st.search_configs(nv, pv, yv, sv, base_val_m, args, name)
        branches.append({"name": name, "val": pv, "test": pt, "rows": rows})

    ens_val = log_average([b["val"] for b in branches])
    ens_test = log_average([b["test"] for b in branches])
    ens_rows = st.search_configs(nv, ens_val, yv, sv, base_val_m, args, "oof_meta_ensemble")
    branches.append({"name": "oof_meta_ensemble", "val": ens_val, "test": ens_test, "rows": ens_rows})

    all_rows = []
    for b in branches:
        all_rows.extend(b["rows"])
    all_rows.sort(key=lambda r: r["score"], reverse=True)
    print_top(all_rows, 20)
    save_csv(relpath("results", f"{suffix}_search_top.csv"), all_rows, args.save_top_records)

    best = all_rows[0]
    branch = next(b for b in branches if b["name"] == best["branch"])
    final_test, gate, use, alpha = st.apply_stacked(nt, branch["test"], best, args)
    final_m = base.metrics_from_probs(final_test, yt, stest)
    diag = base.switch_diagnostics(nt, final_test, gate, use, alpha, stest)

    selected = {
        "best": best,
        "branch": branch["name"],
        "blind_cqi_val_bin_acc": q_acc,
        "gamc_oof_cache": args.gamc_oof_cache,
        "cvtrn_oof_cache": args.cvtrn_oof_cache,
        "cvtrn_valtest_caches": args.cvtrn_valtest_caches,
        "protocol": "train-OOF auxiliary meta-classifier; validation-only blend/gate selection",
    }
    with open(relpath("results", f"{suffix}_selected_config.json"), "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 144)
    print("Final test report")
    print("=" * 144)
    base.print_metrics_line("Fourier soup Test", base.metrics_from_probs(nt, yt, stest))
    base.print_metrics_line("CV-TRN infer Test", base.metrics_from_probs(cv_test, yt, stest))
    base.print_metrics_line("GAMC-tree Test", base.metrics_from_probs(gamc["test_prob"], yt, stest))
    base.print_metrics_line("Train-OOF meta fusion Test", final_m)
    print("-" * 144)
    print(f"Delta vs Fourier overall:    {final_m['overall_acc'] - base.metrics_from_probs(nt, yt, stest)['overall_acc']:+.4f} pp")
    print(f"Delta vs Fourier negative:   {final_m['negative_acc'] - base.metrics_from_probs(nt, yt, stest)['negative_acc']:+.4f} pp")
    print(f"Delta vs Fourier edge:       {final_m['edge_low_acc'] - base.metrics_from_probs(nt, yt, stest)['edge_low_acc']:+.4f} pp")
    print(f"Delta vs Fourier transition: {final_m['transition_acc'] - base.metrics_from_probs(nt, yt, stest)['transition_acc']:+.4f} pp")
    print(f"Delta vs Fourier high:       {final_m['high_acc'] - base.metrics_from_probs(nt, yt, stest)['high_acc']:+.4f} pp")
    print(f"Final diagnostics: {diag}")
    print("=" * 144)

    pred = final_m["pred"].astype(np.int64)
    np.savez_compressed(
        relpath("results", f"{suffix}_predictions.npz"),
        labels=yt.astype(np.int64),
        snrs=stest.astype(np.int32),
        pred=pred,
        final_prob=final_test.astype(np.float32),
        meta_prob=branch["test"].astype(np.float32),
        use_candidate=use.astype(bool),
        mod_classes=soup.get("mod_classes", np.asarray(common.DEFAULT_MOD_CLASSES)),
    )
    curve_path = relpath("results", f"accuracy_vs_snr_{suffix}.png")
    base.plot_curve(final_m["by_snr"], curve_path, "Accuracy vs SNR: Train-OOF Meta Fusion")
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
            "Train-OOF Meta Fusion",
        )
        print(f"[*] {snr_value} dB confusion matrix saved: {cm_path} (Acc={acc:.2f}%)")


if __name__ == "__main__":
    main()
