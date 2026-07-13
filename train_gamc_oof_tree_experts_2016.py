import argparse
import os

import numpy as np
from sklearn.model_selection import StratifiedKFold

import train_cv_trn_aux_2016 as common
import train_gamc_low_snr_aux_experts_2016 as gl


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Cross-fit GAMC/GAMC-low tree auxiliary experts on the train split. "
            "This exports train-OOF probabilities for academically clean meta-router training."
        )
    )
    p.add_argument("--split_seed", type=int, default=1)
    p.add_argument("--random_state", type=int, default=2026)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--feature_cache", type=str, default=common.relpath("feature_cache", "gamc_lite_features_v3_graph_xgb.npz"))
    p.add_argument("--output_cache", type=str, default=common.relpath("results", "gamc_oof_tree_split1_trainvaltest_probs_for_meta.npz"))
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


def member_specs():
    return [
        ("xgb_soft_global", "xgb", "soft_global", dict(depth=4, est=0, lr=0.035), False),
        ("xgb_balanced_low", "xgb", "balanced_low", dict(depth=4, est=0, lr=0.035), False),
        ("xgb_ultra_edge", "xgb", "ultra_edge", dict(depth=3, est=80, lr=0.030), False),
        ("xgb_transition", "xgb", "transition", dict(depth=4, est=0, lr=0.032), False),
        ("xgb_negative", "xgb", "negative", dict(depth=4, est=0, lr=0.035), False),
        ("et_balanced_low", "et", "balanced_low", {}, False),
        ("xgb_low_subset", "xgb", "balanced_low", dict(depth=3, est=120, lr=0.028), True),
    ]


def make_model(args, spec, seed):
    _name, kind, _mode, hp, _subset = spec
    if kind == "et":
        return gl.et_model(args, seed)
    return gl.xgb_model(
        args,
        seed,
        depth=hp.get("depth", args.xgb_depth),
        estimators=args.xgb_estimators + int(hp.get("est", 0)),
        lr=hp.get("lr", args.xgb_lr),
    )


def fit_predict(model, x_tr, y_tr, w_tr, x_pred):
    model.fit(x_tr, y_tr, sample_weight=w_tr)
    return gl.align_proba(model, model.predict_proba(x_pred))


def main():
    args = parse_args()
    os.makedirs(common.relpath("results"), exist_ok=True)
    print("=" * 120)
    print("Cross-fit GAMC/GAMC-low tree auxiliary experts")
    print("=" * 120)
    print("Academic protocol:")
    print("  - Outer train/val/test split is unchanged.")
    print("  - Train probabilities are out-of-fold predictions from train split only.")
    print("  - SNR metadata is used only inside the train split for weighting/subset construction.")
    print("  - Validation metrics are sanity checks; test labels are not scored here.")

    full_dataset = common.build_full_dataset(args)
    train_idx, val_idx, test_idx = common.make_aligned_split(full_dataset, args.split_seed)
    if not args.skip_alignment_check:
        common.check_alignment(full_dataset, val_idx, test_idx, args.alignment_cache)
    labels = np.asarray(full_dataset.labels, dtype=np.int64)
    snrs = np.asarray(full_dataset.snrs, dtype=np.int32)

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

    composite = np.asarray([f"{int(y)}_{int(s)}" for y, s in zip(y_train, s_train)])
    skf = StratifiedKFold(n_splits=int(args.folds), shuffle=True, random_state=int(args.random_state))

    train_members, val_members, test_members, names = [], [], [], []
    for m, spec in enumerate(member_specs(), 1):
        name, _kind, mode, _hp, low_subset = spec
        print("\n" + "-" * 120)
        print(f"[*] Member {m}/{len(member_specs())}: {name}")
        train_oof = np.zeros((len(train_idx), common.NUM_CLASSES), dtype=np.float32)

        for fold, (tr, va) in enumerate(skf.split(x_train, composite), 1):
            fit_ids = tr
            if low_subset:
                fit_ids = tr[s_train[tr] <= -6]
            if len(fit_ids) == 0 or len(np.unique(y_train[fit_ids])) < common.NUM_CLASSES:
                raise RuntimeError(f"{name} fold {fold} lacks class coverage after subset filtering.")
            model = make_model(args, spec, args.random_state + 1000 * m + fold)
            w = gl.weights_for(s_train[fit_ids], mode)
            train_oof[va] = fit_predict(model, x_train[fit_ids], y_train[fit_ids], w, x_train[va])
            print(f"    fold {fold}/{args.folds} done | train={len(fit_ids):,} | holdout={len(va):,}")

        fit_all = np.arange(len(train_idx))
        if low_subset:
            fit_all = fit_all[s_train <= -6]
        final_model = make_model(args, spec, args.random_state + 9000 + m)
        final_w = gl.weights_for(s_train[fit_all], mode)
        val_prob = fit_predict(final_model, x_train[fit_all], y_train[fit_all], final_w, x_val)
        test_prob = gl.align_proba(final_model, final_model.predict_proba(x_test))

        gl.print_metrics(name + " Train-OOF", gl.metrics_from_probs(train_oof, y_train, s_train))
        gl.print_metrics(name + " Val", gl.metrics_from_probs(val_prob, y_val, s_val))
        train_members.append(train_oof)
        val_members.append(val_prob)
        test_members.append(test_prob)
        names.append(name)

    train_member_probs = np.stack(train_members, axis=0).astype(np.float32)
    val_member_probs = np.stack(val_members, axis=0).astype(np.float32)
    test_member_probs = np.stack(test_members, axis=0).astype(np.float32)
    train_prob = gl.log_average(train_members)
    val_prob = gl.log_average(val_members)
    test_prob = gl.log_average(test_members)

    gl.print_metrics("GAMC-tree Train-OOF ensemble", gl.metrics_from_probs(train_prob, y_train, s_train))
    gl.print_metrics("GAMC-tree Val ensemble", gl.metrics_from_probs(val_prob, y_val, s_val))

    np.savez_compressed(
        args.output_cache,
        train_prob=train_prob.astype(np.float32),
        val_prob=val_prob.astype(np.float32),
        test_prob=test_prob.astype(np.float32),
        train_member_probs=train_member_probs,
        val_member_probs=val_member_probs,
        test_member_probs=test_member_probs,
        member_names=np.asarray(names),
        train_indices=train_idx.astype(np.int64),
        val_indices=val_idx.astype(np.int64),
        test_indices=test_idx.astype(np.int64),
        labels_train=y_train.astype(np.int64),
        snrs_train=s_train.astype(np.int32),
        labels_val=y_val.astype(np.int64),
        snrs_val=s_val.astype(np.int32),
        labels_test=labels[test_idx].astype(np.int64),
        snrs_test=snrs[test_idx].astype(np.int32),
        mod_classes=np.asarray(getattr(full_dataset, "mod_classes", common.DEFAULT_MOD_CLASSES)),
        protocol=np.asarray(["train-split OOF GAMC/GAMC-low tree experts; test probabilities exported without scoring"]),
    )
    print(f"[*] OOF tree cache saved: {args.output_cache}")


if __name__ == "__main__":
    main()
