import argparse
import os

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier

import train_cv_trn_aux_2016 as common


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Train GAMC-feature low-SNR/transition auxiliary experts. "
            "These are tree-based auxiliary experts inspired by GAMC; they do not change the Fourier main model."
        )
    )
    p.add_argument("--split_seed", type=int, default=1)
    p.add_argument("--random_state", type=int, default=2026)
    p.add_argument("--feature_cache", type=str, default=common.relpath("feature_cache", "gamc_lite_features_v3_graph_xgb.npz"))
    p.add_argument("--output_cache", type=str, default=common.relpath("results", "gamc_low_snr_aux_xgb_split1_valtest_probs_for_fusion.npz"))
    p.add_argument("--data_path", type=str, default=common.relpath("raw_data", "RML2016.10a_dict.pkl"))
    p.add_argument("--cache_dir", type=str, default=common.relpath("feature_cache"))
    p.add_argument("--alignment_cache", type=str, default=common.relpath("results", "greedy_soup_identity_valtest_probs_for_gamc_fusion.npz"))
    p.add_argument("--skip_alignment_check", action="store_true")
    p.add_argument("--xgb_jobs", type=int, default=-1)
    p.add_argument("--xgb_estimators", type=int, default=520)
    p.add_argument("--xgb_depth", type=int, default=4)
    p.add_argument("--xgb_lr", type=float, default=0.035)
    p.add_argument("--et_estimators", type=int, default=520)
    p.add_argument("--et_min_samples_leaf", type=int, default=3)
    return p.parse_args()


def norm(p):
    p = np.asarray(p, dtype=np.float32)
    p = np.clip(p, 1e-12, 1.0)
    return p / (p.sum(axis=1, keepdims=True) + 1e-12)


def align_proba(clf, p, n_classes=11):
    out = np.zeros((p.shape[0], n_classes), dtype=np.float32)
    for j, c in enumerate(clf.classes_):
        out[:, int(c)] = p[:, j]
    return norm(out)


def log_average(prob_list):
    logs = [np.log(norm(p) + 1e-12) for p in prob_list]
    z = np.mean(np.stack(logs, axis=0), axis=0)
    z -= z.max(axis=1, keepdims=True)
    return norm(np.exp(z))


def metrics_from_probs(probs, labels, snrs):
    probs = norm(probs)
    pred = probs.argmax(axis=1)
    labels = labels.astype(np.int64)
    snrs = snrs.astype(np.int32)

    def acc(mask):
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() == 0:
            return 0.0
        return float((pred[mask] == labels[mask]).mean() * 100.0)

    return {
        "overall": float((pred == labels).mean() * 100.0),
        "negative": acc(snrs < 0),
        "transition": acc(np.isin(snrs, [-10, -8, -6, -4, -2])),
        "edge": acc(np.isin(snrs, [-18, -16])),
        "high": acc(snrs >= 0),
    }


def print_metrics(name, m):
    print(
        f"{name:<28} overall={m['overall']:.3f}% | neg={m['negative']:.3f}% | "
        f"trans={m['transition']:.3f}% | edge={m['edge']:.3f}% | high={m['high']:.3f}%"
    )


def xgb_model(args, seed, depth=None, estimators=None, lr=None):
    return XGBClassifier(
        objective="multi:softprob",
        num_class=11,
        n_estimators=int(estimators or args.xgb_estimators),
        max_depth=int(depth or args.xgb_depth),
        learning_rate=float(lr or args.xgb_lr),
        subsample=0.90,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        reg_alpha=0.05,
        min_child_weight=1.0,
        tree_method="hist",
        eval_metric="mlogloss",
        n_jobs=int(args.xgb_jobs),
        random_state=int(seed),
        verbosity=0,
    )


def et_model(args, seed):
    return ExtraTreesClassifier(
        n_estimators=int(args.et_estimators),
        max_depth=None,
        min_samples_leaf=int(args.et_min_samples_leaf),
        max_features="sqrt",
        class_weight=None,
        n_jobs=int(args.xgb_jobs),
        random_state=int(seed),
    )


def weights_for(snrs, mode):
    s = np.asarray(snrs, dtype=np.int32)
    w = np.ones(len(s), dtype=np.float32)
    if mode == "balanced_low":
        w[s <= -16] = 2.3
        w[(s >= -14) & (s <= -8)] = 2.0
        w[(s >= -6) & (s <= -2)] = 1.45
        w[s >= 8] = 0.75
    elif mode == "ultra_edge":
        w[s <= -16] = 3.4
        w[(s >= -14) & (s <= -12)] = 2.2
        w[(s >= -10) & (s <= -6)] = 1.45
        w[s >= 4] = 0.65
    elif mode == "transition":
        w[(s >= -10) & (s <= 0)] = 2.6
        w[s <= -12] = 1.40
        w[s >= 6] = 0.70
    elif mode == "negative":
        w[s < 0] = 2.1
        w[s <= -16] = 2.7
        w[s >= 4] = 0.70
    elif mode == "soft_global":
        w[s < 0] = 1.45
        w[np.isin(s, [-18, -16])] = 1.90
    else:
        raise ValueError(mode)
    return w.astype(np.float32)


def fit_member(name, model, x_train, y_train, w_train, x_val, x_test):
    print(f"[*] Training member: {name}")
    model.fit(x_train, y_train, sample_weight=w_train)
    pv = align_proba(model, model.predict_proba(x_val))
    pt = align_proba(model, model.predict_proba(x_test))
    return pv, pt


def main():
    args = parse_args()
    os.makedirs(common.relpath("results"), exist_ok=True)
    print("=" * 120)
    print("Train GAMC low-SNR/transition auxiliary tree experts")
    print("=" * 120)
    print("Academic protocol:")
    print("  - Fit tree experts on train split only.")
    print("  - Use SNR metadata only for train-split sample weighting/subset construction.")
    print("  - Print validation metrics for sanity; do not score test here.")
    print("  - Export test probabilities for final fusion only.")

    full_dataset = common.build_full_dataset(args)
    train_idx, val_idx, test_idx = common.make_aligned_split(full_dataset, args.split_seed)
    if not args.skip_alignment_check:
        common.check_alignment(full_dataset, val_idx, test_idx, args.alignment_cache)
    labels = np.asarray(full_dataset.labels, dtype=np.int64)
    snrs = np.asarray(full_dataset.snrs, dtype=np.int32)

    if not os.path.exists(args.feature_cache):
        raise FileNotFoundError(args.feature_cache)
    z = np.load(args.feature_cache, allow_pickle=True)
    key = "features" if "features" in z.files else z.files[0]
    feats = np.nan_to_num(z[key].astype(np.float32), nan=0.0, posinf=1e5, neginf=-1e5)
    print(f"Feature cache: {args.feature_cache}")
    print(f"Feature key={key}, shape={feats.shape}")

    x_train = feats[train_idx]
    y_train = labels[train_idx]
    s_train = snrs[train_idx]
    x_val = feats[val_idx]
    y_val = labels[val_idx]
    s_val = snrs[val_idx]
    x_test = feats[test_idx]
    y_test = labels[test_idx]
    s_test = snrs[test_idx]

    members = []
    member_names = []

    configs = [
        ("xgb_soft_global", xgb_model(args, args.random_state + 11, depth=4, estimators=args.xgb_estimators), "soft_global", None),
        ("xgb_balanced_low", xgb_model(args, args.random_state + 23, depth=4, estimators=args.xgb_estimators), "balanced_low", None),
        ("xgb_ultra_edge", xgb_model(args, args.random_state + 37, depth=3, estimators=args.xgb_estimators + 80, lr=0.030), "ultra_edge", None),
        ("xgb_transition", xgb_model(args, args.random_state + 53, depth=4, estimators=args.xgb_estimators, lr=0.032), "transition", None),
        ("xgb_negative", xgb_model(args, args.random_state + 71, depth=4, estimators=args.xgb_estimators), "negative", None),
        ("et_balanced_low", et_model(args, args.random_state + 89), "balanced_low", None),
    ]

    for name, model, mode, _ in configs:
        pv, pt = fit_member(name, model, x_train, y_train, weights_for(s_train, mode), x_val, x_test)
        print_metrics(name + " Val", metrics_from_probs(pv, y_val, s_val))
        members.append((pv, pt))
        member_names.append(name)

    low_mask = s_train <= -6
    if low_mask.sum() > 0 and len(np.unique(y_train[low_mask])) == 11:
        name = "xgb_low_subset"
        print(f"[*] Training member: {name} on train SNR <= -6 only, n={int(low_mask.sum())}")
        model = xgb_model(args, args.random_state + 107, depth=3, estimators=args.xgb_estimators + 120, lr=0.028)
        model.fit(x_train[low_mask], y_train[low_mask], sample_weight=weights_for(s_train[low_mask], "balanced_low"))
        pv = align_proba(model, model.predict_proba(x_val))
        pt = align_proba(model, model.predict_proba(x_test))
        print_metrics(name + " Val", metrics_from_probs(pv, y_val, s_val))
        members.append((pv, pt))
        member_names.append(name)

    val_member_probs = np.stack([m[0] for m in members], axis=0).astype(np.float32)
    test_member_probs = np.stack([m[1] for m in members], axis=0).astype(np.float32)
    val_prob = log_average([m[0] for m in members])
    test_prob = log_average([m[1] for m in members])
    print_metrics("GAMC-low ensemble Val", metrics_from_probs(val_prob, y_val, s_val))

    np.savez_compressed(
        args.output_cache,
        val_prob=val_prob.astype(np.float32),
        test_prob=test_prob.astype(np.float32),
        val_member_probs=val_member_probs,
        test_member_probs=test_member_probs,
        member_names=np.asarray(member_names),
        labels_val=y_val.astype(np.int64),
        snrs_val=s_val.astype(np.int32),
        labels_test=y_test.astype(np.int64),
        snrs_test=s_test.astype(np.int32),
        mod_classes=np.asarray(getattr(full_dataset, "mod_classes", common.DEFAULT_MOD_CLASSES)),
        protocol=np.asarray(["train-only GAMC low-SNR auxiliary experts; test probabilities exported without scoring"]),
    )
    print(f"[*] GAMC-low auxiliary cache saved: {args.output_cache}")
    print("[*] Test probabilities were exported for final fusion; no test accuracy was reported here.")


if __name__ == "__main__":
    main()
