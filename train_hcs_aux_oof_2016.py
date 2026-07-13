import argparse
import os
import sys

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

import evaluate_greedy_soup_gamc_protected_residual_fusion as base
import evaluate_fourier_oof_gamc_cvtrn_residual_meta_fusion as fourier_meta
import train_cv_trn_aux_2016 as common
from model_cache_utils import fit_or_load_estimator


try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


GROUPS = {
    "analog": {
        "classes": [1, 2, 10],  # AM-DSB, AM-SSB, WBFM
        "boost": 2.0,
        "context_boost": 1.25,
        "neg_boost": 1.00,
        "trans_boost": 1.15,
        "high_boost": 1.15,
        "depth": 3,
        "estimators_delta": 0,
    },
    "constellation": {
        "classes": [0, 3, 7, 8, 9],  # 8PSK, BPSK, QAM16, QAM64, QPSK
        "boost": 1.85,
        "context_boost": 1.30,
        "neg_boost": 1.15,
        "trans_boost": 1.18,
        "high_boost": 1.00,
        "depth": 4,
        "estimators_delta": 80,
    },
    "fskpam": {
        "classes": [3, 4, 5, 6],  # BPSK, CPFSK, GFSK, PAM4
        "boost": 1.75,
        "context_boost": 1.25,
        "neg_boost": 1.20,
        "trans_boost": 1.10,
        "high_boost": 1.00,
        "depth": 4,
        "estimators_delta": 60,
    },
}


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Train confusion-aware lightweight HCS auxiliary specialists for RML2016.10A. "
            "Each specialist predicts a modulation family plus other, then maps back to 11-class probabilities."
        )
    )
    p.add_argument("--split_seed", type=int, default=1)
    p.add_argument("--random_state", type=int, default=2037)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--groups", type=str, nargs="+", default=["analog", "constellation", "fskpam"], choices=sorted(GROUPS))
    p.add_argument("--model_type", type=str, default="xgb", choices=["xgb", "et"])
    p.add_argument("--xgb_jobs", type=int, default=-1)
    p.add_argument("--xgb_device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--model_cache_dir", type=str, default="")
    p.add_argument("--reuse_models", action="store_true")
    p.add_argument("--xgb_estimators", type=int, default=420)
    p.add_argument("--xgb_lr", type=float, default=0.035)
    p.add_argument("--xgb_subsample", type=float, default=0.92)
    p.add_argument("--xgb_colsample", type=float, default=0.88)
    p.add_argument("--et_estimators", type=int, default=520)
    p.add_argument("--et_min_leaf", type=int, default=4)
    p.add_argument("--specialist_blend", type=float, default=0.78)
    p.add_argument("--gate_conf", type=float, default=0.985)
    p.add_argument("--gate_margin", type=float, default=0.92)
    p.add_argument("--feature_chunk", type=int, default=8192)
    p.add_argument("--feature_cache", type=str, default=common.relpath("feature_cache", "hcs_lite_features_v1.npz"))
    p.add_argument("--force_rebuild_features", action="store_true")
    p.add_argument("--data_path", type=str, default=common.relpath("raw_data", "RML2016.10a_dict.pkl"))
    p.add_argument("--cache_dir", type=str, default=common.relpath("feature_cache"))
    p.add_argument("--alignment_cache", type=str, default=common.relpath("results", "greedy_soup_identity_valtest_probs_for_gamc_fusion.npz"))
    p.add_argument("--fourier_oof_cache", type=str, default=common.relpath("results", "fourier_main_oof_merge_mseed78_79_f3e220_bs128_split1_trainvaltest_probs_for_meta.npz"))
    p.add_argument("--soup_prob_cache", type=str, default=common.relpath("results", "greedy_soup_identity_valtest_probs_for_gamc_fusion.npz"))
    p.add_argument("--skip_alignment_check", action="store_true")
    p.add_argument("--output_cache", type=str, default=common.relpath("results", "hcs_aux_oof_split1_trainvaltest_probs_for_meta.npz"))
    return p.parse_args()


def normalize(p):
    p = np.asarray(p, dtype=np.float32)
    p = np.clip(p, 1e-12, 1.0)
    return p / (p.sum(axis=1, keepdims=True) + 1e-12)


def log_average(prob_list):
    logs = [np.log(normalize(p) + 1e-12) for p in prob_list]
    z = np.mean(np.stack(logs, axis=0), axis=0)
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / (e.sum(axis=1, keepdims=True) + 1e-12)).astype(np.float32)


def hist_features(values, bins):
    idx = np.searchsorted(bins, values, side="right") - 1
    idx = np.clip(idx, 0, len(bins) - 2)
    out = np.zeros((values.shape[0], len(bins) - 1), dtype=np.float32)
    rows = np.repeat(np.arange(values.shape[0]), values.shape[1])
    np.add.at(out, (rows, idx.reshape(-1)), 1.0)
    return out / float(values.shape[1])


def stats_features(x):
    x = x.astype(np.float32, copy=False)
    mean = x.mean(axis=1)
    std = x.std(axis=1) + 1e-6
    centered = x - mean[:, None]
    skew = (centered**3).mean(axis=1) / (std**3)
    kurt = (centered**4).mean(axis=1) / (std**4)
    q25, q50, q75 = np.percentile(x, [25, 50, 75], axis=1)
    return np.stack(
        [
            mean,
            std,
            x.min(axis=1),
            x.max(axis=1),
            q25,
            q50,
            q75,
            skew,
            kurt,
            np.mean(np.abs(x), axis=1),
            np.mean(x * x, axis=1),
        ],
        axis=1,
    ).astype(np.float32)


def autocorr_features(x, lags=(1, 2, 4, 8, 16, 32)):
    outs = []
    denom = np.mean(x * x, axis=1) + 1e-6
    for lag in lags:
        c = np.mean(x[:, lag:] * x[:, :-lag], axis=1) / denom
        outs.append(c)
    return np.stack(outs, axis=1).astype(np.float32)


def build_hcs_features(full_dataset, args):
    if os.path.exists(args.feature_cache) and not args.force_rebuild_features:
        z = np.load(args.feature_cache, allow_pickle=True)
        print(f"[*] Loading HCS feature cache: {args.feature_cache}")
        return z["features"].astype(np.float32)
    print("[*] Building HCS lightweight polar/spectral/statistical features")
    data = np.asarray(full_dataset.data, dtype=np.float32)
    hos = np.asarray(full_dataset.hos_data, dtype=np.float32)
    feats = []
    amp_bins = np.asarray([0.0, 0.45, 0.65, 0.85, 1.05, 1.25, 1.50, 1.85, 2.30, 3.0, 10.0], dtype=np.float32)
    phase_bins = np.linspace(-np.pi, np.pi, 17, dtype=np.float32)
    dphi_bins = np.linspace(-np.pi, np.pi, 17, dtype=np.float32)
    for start in range(0, len(data), int(args.feature_chunk)):
        end = min(len(data), start + int(args.feature_chunk))
        x = data[start:end]
        if x.shape[1] >= 3:
            iq = x[:, [1, 2], :]
        else:
            iq = x[:, :2, :]
        i = iq[:, 0, :]
        q = iq[:, 1, :]
        amp = np.sqrt(i * i + q * q + 1e-8)
        phase = np.arctan2(q, i + 1e-8)
        dot = i[:, 1:] * i[:, :-1] + q[:, 1:] * q[:, :-1]
        cross = q[:, 1:] * i[:, :-1] - i[:, 1:] * q[:, :-1]
        dphi = np.concatenate([np.zeros((len(i), 1), dtype=np.float32), np.arctan2(cross, dot + 1e-8)], axis=1)
        zc = i.astype(np.complex64) + 1j * q.astype(np.complex64)
        mag = np.fft.fftshift(np.abs(np.fft.fft(zc, axis=1)), axes=1).astype(np.float32)
        logmag = np.log1p(mag)
        logmag = (logmag - logmag.mean(axis=1, keepdims=True)) / (logmag.std(axis=1, keepdims=True) + 1e-5)
        amp_norm = amp / (amp.mean(axis=1, keepdims=True) + 1e-6)
        bands = logmag.reshape(len(i), 16, 8)
        chunk_parts = [
            hos[start:end].astype(np.float32),
            stats_features(i),
            stats_features(q),
            stats_features(amp),
            stats_features(dphi),
            stats_features(logmag),
            autocorr_features(i),
            autocorr_features(q),
            autocorr_features(amp),
            autocorr_features(dphi),
            bands.mean(axis=2).astype(np.float32),
            bands.std(axis=2).astype(np.float32),
            bands.max(axis=2).astype(np.float32),
            hist_features(amp_norm, amp_bins),
            hist_features(phase, phase_bins),
            hist_features(dphi, dphi_bins),
        ]
        feats.append(np.concatenate(chunk_parts, axis=1).astype(np.float32))
        if end % (int(args.feature_chunk) * 8) == 0 or end == len(data):
            print(f"    features {end:,}/{len(data):,}")
    features = np.concatenate(feats, axis=0)
    features = np.nan_to_num(features, nan=0.0, posinf=1e5, neginf=-1e5).astype(np.float32)
    os.makedirs(os.path.dirname(args.feature_cache), exist_ok=True)
    np.savez_compressed(args.feature_cache, features=features)
    print(f"[*] HCS feature cache saved: {args.feature_cache} shape={features.shape}")
    return features


def local_labels(y, group_classes):
    group_classes = list(map(int, group_classes))
    out = np.full(len(y), len(group_classes), dtype=np.int64)
    for i, cls in enumerate(group_classes):
        out[y == int(cls)] = i
    return out


def align_local_proba(model, p, n_classes):
    out = np.zeros((p.shape[0], n_classes), dtype=np.float32)
    classes = getattr(model, "classes_", np.arange(p.shape[1]))
    for i, cls in enumerate(classes):
        out[:, int(cls)] = p[:, i]
    return normalize(out)


def make_model(args, cfg, local_num_classes, seed):
    if args.model_type == "et":
        return ExtraTreesClassifier(
            n_estimators=int(args.et_estimators),
            max_depth=None,
            min_samples_leaf=int(args.et_min_leaf),
            max_features="sqrt",
            n_jobs=int(args.xgb_jobs),
            random_state=int(seed),
        )
    return XGBClassifier(
        objective="multi:softprob",
        num_class=int(local_num_classes),
        n_estimators=int(args.xgb_estimators) + int(cfg.get("estimators_delta", 0)),
        max_depth=int(cfg.get("depth", 4)),
        learning_rate=float(args.xgb_lr),
        subsample=float(args.xgb_subsample),
        colsample_bytree=float(args.xgb_colsample),
        reg_lambda=3.0,
        reg_alpha=0.04,
        min_child_weight=2.0,
        tree_method="hist",
        device=str(getattr(args, "xgb_device", "cpu")),
        eval_metric="mlogloss",
        n_jobs=int(args.xgb_jobs),
        random_state=int(seed),
        verbosity=0,
    )


def group_weights(y, snr, main_prob, cfg):
    y = y.astype(np.int64)
    snr = snr.astype(np.int32)
    group = np.asarray(cfg["classes"], dtype=np.int64)
    w = np.ones(len(y), dtype=np.float32)
    in_group = np.isin(y, group)
    w[in_group] *= float(cfg["boost"])
    top2 = np.argsort(main_prob, axis=1)[:, -2:]
    context = np.isin(top2, group).any(axis=1)
    w[context] *= float(cfg["context_boost"])
    w[snr < 0] *= float(cfg["neg_boost"])
    w[np.isin(snr, [-10, -8, -6, -4, -2])] *= float(cfg["trans_boost"])
    w[snr >= 0] *= float(cfg["high_boost"])
    return (w / (w.mean() + 1e-8)).astype(np.float32)


def map_local_to_full(local_prob, main_prob, group_classes):
    group_classes = list(map(int, group_classes))
    k = len(group_classes)
    out = np.zeros((local_prob.shape[0], common.NUM_CLASSES), dtype=np.float32)
    for i, cls in enumerate(group_classes):
        out[:, cls] = local_prob[:, i]
    outside = np.ones(common.NUM_CLASSES, dtype=bool)
    outside[group_classes] = False
    base_out = main_prob[:, outside]
    denom = base_out.sum(axis=1, keepdims=True) + 1e-12
    out[:, outside] = local_prob[:, k:k + 1] * base_out / denom
    return normalize(out)


def combine_groups(mapped_by_group, main_prob, args):
    main_prob = normalize(main_prob)
    top2 = np.argsort(main_prob, axis=1)[:, -2:]
    top = top2[:, 1]
    second = top2[:, 0]
    conf = main_prob[np.arange(len(main_prob)), top]
    margin = conf - main_prob[np.arange(len(main_prob)), second]
    outputs = []
    active_any = np.zeros(len(main_prob), dtype=bool)
    for name, mapped in mapped_by_group.items():
        group = np.asarray(GROUPS[name]["classes"], dtype=np.int64)
        active = np.isin(top2, group).any(axis=1)
        active &= (conf < float(args.gate_conf)) | (margin < float(args.gate_margin))
        active_any |= active
        cur = main_prob.copy()
        cur[active] = mapped[active]
        outputs.append(cur)
    if not outputs:
        return main_prob.astype(np.float32)
    avg = log_average(outputs)
    hcs = main_prob.copy()
    blend = float(args.specialist_blend)
    hcs[active_any] = (1.0 - blend) * main_prob[active_any] + blend * avg[active_any]
    return normalize(hcs)


def print_rescue(name, main_prob, aux_prob, labels, snrs):
    main_pred = normalize(main_prob).argmax(axis=1)
    aux_pred = normalize(aux_prob).argmax(axis=1)
    ok = main_pred == labels
    rescue = (~ok) & (aux_pred == labels)
    harm = ok & (aux_pred != labels)
    print(f"{name:<22} rescue={int(rescue.sum())} harm={int(harm.sum())} net={int(rescue.sum() - harm.sum())}")
    for label, mask in [
        ("neg", snrs < 0),
        ("edge", np.isin(snrs, [-18, -16])),
        ("trans", np.isin(snrs, [-10, -8, -6, -4, -2])),
        ("high", snrs >= 0),
    ]:
        print(f"    {label:<5} rescue={int(rescue[mask].sum()):4d} harm={int(harm[mask].sum()):4d} net={int(rescue[mask].sum() - harm[mask].sum()):4d}")


def main():
    args = parse_args()
    os.makedirs(common.relpath("results"), exist_ok=True)
    print("=" * 120)
    print("Train confusion-aware HCS lightweight auxiliary specialists")
    print("=" * 120)
    print("Academic protocol:")
    print("  - Specialists are trained only on train split with OOF predictions for train samples.")
    print("  - Validation labels are not used for training; test labels are not scored during training.")
    print("  - Inference gating uses main-model top-2 probabilities only, not true SNR.")
    print(f"groups={args.groups} | model_type={args.model_type} | folds={args.folds}")

    full_dataset = common.build_full_dataset(args)
    train_idx, val_idx, test_idx = common.make_aligned_split(full_dataset, args.split_seed)
    if not args.skip_alignment_check:
        common.check_alignment(full_dataset, val_idx, test_idx, args.alignment_cache)
    labels = np.asarray(full_dataset.labels, dtype=np.int64)
    snrs = np.asarray(full_dataset.snrs, dtype=np.int32)
    features = build_hcs_features(full_dataset, args)

    fourier = fourier_meta.load_npz(args.fourier_oof_cache)
    soup = fourier_meta.load_npz(args.soup_prob_cache)
    for key in ("labels_train", "snrs_train"):
        if not np.all(fourier[key] == labels[train_idx] if key.startswith("labels") else fourier[key] == snrs[train_idx]):
            raise RuntimeError(f"Fourier OOF alignment mismatch: {key}")
    for key, ref in [
        ("labels_val", labels[val_idx]),
        ("snrs_val", snrs[val_idx]),
        ("labels_test", labels[test_idx]),
        ("snrs_test", snrs[test_idx]),
    ]:
        if not np.all(soup[key] == ref):
            raise RuntimeError(f"Soup alignment mismatch: {key}")

    x_train = features[train_idx]
    y_train = labels[train_idx]
    s_train = snrs[train_idx]
    x_val = features[val_idx]
    y_val = labels[val_idx]
    s_val = snrs[val_idx]
    x_test = features[test_idx]
    y_test = labels[test_idx]
    s_test = snrs[test_idx]
    main_train = normalize(fourier["train_prob"])
    main_val = normalize(soup["val_prob"])
    main_test = normalize(soup["test_prob"])

    composite = np.asarray([f"{int(y)}_{int(s)}" for y, s in zip(y_train, s_train)])
    skf = StratifiedKFold(n_splits=int(args.folds), shuffle=True, random_state=int(args.random_state))

    mapped_train_by_group, mapped_val_by_group, mapped_test_by_group = {}, {}, {}
    member_train, member_val, member_test, member_names = [], [], [], []
    for gi, name in enumerate(args.groups, 1):
        cfg = GROUPS[name]
        local_num = len(cfg["classes"]) + 1
        y_local = local_labels(y_train, cfg["classes"])
        train_local = np.zeros((len(train_idx), local_num), dtype=np.float32)
        val_locals, test_locals = [], []
        print("\n" + "-" * 120)
        print(f"[*] HCS specialist {gi}/{len(args.groups)}: {name} classes={cfg['classes']} + other")
        for fold, (tr, va) in enumerate(skf.split(x_train, composite), 1):
            model = make_model(args, cfg, local_num, args.random_state + gi * 1000 + fold)
            w = group_weights(y_train[tr], s_train[tr], main_train[tr], cfg)
            model, _, _ = fit_or_load_estimator(
                model,
                x_train[tr],
                y_local[tr],
                sample_weight=w,
                cache_dir=args.model_cache_dir,
                reuse=args.reuse_models,
                namespace=f"hcs_{name}_fold{fold}",
                source_paths=[args.feature_cache, args.fourier_oof_cache],
                context={
                    "builder": "hcs_oof_v1",
                    "split_seed": args.split_seed,
                    "group": name,
                    "group_classes": cfg["classes"],
                    "fold": fold,
                    "folds": args.folds,
                },
            )
            train_local[va] = align_local_proba(model, model.predict_proba(x_train[va]), local_num)
            val_locals.append(align_local_proba(model, model.predict_proba(x_val), local_num))
            test_locals.append(align_local_proba(model, model.predict_proba(x_test), local_num))
            print(f"    fold {fold}/{args.folds} done")
        val_local = log_average(val_locals)
        test_local = log_average(test_locals)
        train_full = map_local_to_full(train_local, main_train, cfg["classes"])
        val_full = map_local_to_full(val_local, main_val, cfg["classes"])
        test_full = map_local_to_full(test_local, main_test, cfg["classes"])
        mapped_train_by_group[name] = train_full
        mapped_val_by_group[name] = val_full
        mapped_test_by_group[name] = test_full
        member_train.append(train_full)
        member_val.append(val_full)
        member_test.append(test_full)
        member_names.append(name)
        base.print_metrics_line(f"HCS-{name} Val", base.metrics_from_probs(val_full, y_val, s_val))
        print_rescue(f"HCS-{name} Val", main_val, val_full, y_val, s_val)

    train_prob = combine_groups(mapped_train_by_group, main_train, args)
    val_prob = combine_groups(mapped_val_by_group, main_val, args)
    test_prob = combine_groups(mapped_test_by_group, main_test, args)
    print("\nHCS combined diagnostics")
    base.print_metrics_line("HCS combined Val", base.metrics_from_probs(val_prob, y_val, s_val))
    print_rescue("HCS combined Val", main_val, val_prob, y_val, s_val)
    print("[*] Test probabilities exported for final fusion; test labels are not reported here.")

    np.savez_compressed(
        args.output_cache,
        train_prob=train_prob.astype(np.float32),
        val_prob=val_prob.astype(np.float32),
        test_prob=test_prob.astype(np.float32),
        train_member_probs=np.stack(member_train, axis=0).astype(np.float32),
        val_member_probs=np.stack(member_val, axis=0).astype(np.float32),
        test_member_probs=np.stack(member_test, axis=0).astype(np.float32),
        member_names=np.asarray(member_names),
        train_indices=train_idx.astype(np.int64),
        val_indices=val_idx.astype(np.int64),
        test_indices=test_idx.astype(np.int64),
        labels_train=y_train.astype(np.int64),
        snrs_train=s_train.astype(np.int32),
        labels_val=y_val.astype(np.int64),
        snrs_val=s_val.astype(np.int32),
        labels_test=y_test.astype(np.int64),
        snrs_test=s_test.astype(np.int32),
        mod_classes=np.asarray(getattr(full_dataset, "mod_classes", common.DEFAULT_MOD_CLASSES)),
        groups=np.asarray(args.groups),
        protocol=np.asarray(["train-split OOF confusion-aware HCS specialists; inference gate uses main top-2 probabilities only"]),
    )
    print(f"[*] HCS auxiliary OOF cache saved: {args.output_cache}")


if __name__ == "__main__":
    main()
