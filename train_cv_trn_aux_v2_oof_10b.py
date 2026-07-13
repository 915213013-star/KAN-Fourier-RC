"""10B wrapper for CV-TRN-v2 train-split OOF auxiliary probabilities."""

from __future__ import annotations

import os

import train_cv_trn_aux_10b_common as common10b
import train_cv_trn_aux_v2_2016 as v2
import train_cv_trn_aux_v2_oof_2016 as impl


impl.common = common10b
impl.v2.common = common10b
v2.common = common10b


def checkpoint_paths_10b(args, fold):
    suffix = f"cv_trn_aux_v2_oof_10b_mseed{args.model_seed}_fold{fold}_split{args.split_seed}"
    return (
        common10b.relpath("checkpoints", f"best_{suffix}.pth"),
        common10b.relpath("checkpoints", f"latest_{suffix}.pth"),
    )


impl.checkpoint_paths = checkpoint_paths_10b


def main():
    args = impl.parse_args()
    if args.data_path == common10b.relpath("raw_data", "RML2016.10a_dict.pkl"):
        args.data_path = common10b.DEFAULT_10B_DATA_PATH
    if args.cache_dir == common10b.relpath("feature_cache"):
        args.cache_dir = common10b.DEFAULT_10B_CACHE_DIR
    if args.alignment_cache == common10b.relpath("results", "greedy_soup_identity_valtest_probs_for_gamc_fusion.npz"):
        args.alignment_cache = common10b.DEFAULT_10B_ALIGNMENT_CACHE
    if not args.output_cache:
        args.output_cache = common10b.relpath(
            "results",
            f"cv_trn_aux_v2_oof_10b_mseed{args.model_seed}_f{args.folds}e{args.epochs}_split{args.split_seed}_trainvaltest_probs_for_meta.npz",
        )
    os.makedirs(common10b.relpath("checkpoints"), exist_ok=True)
    os.makedirs(common10b.relpath("results"), exist_ok=True)
    print("[10B wrapper] data_path=", args.data_path)
    print("[10B wrapper] cache_dir=", args.cache_dir)
    print("[10B wrapper] alignment_cache=", args.alignment_cache)
    return impl.main_with_args(args) if hasattr(impl, "main_with_args") else _main_impl(args)


def _main_impl(args):
    device = impl.torch.device("cuda" if impl.torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.disable_amp)
    print("=" * 120)
    print("Train RML2016.10B CV-TRN-v2 OOF auxiliary experts")
    print("=" * 120)
    print(f"device={device} | amp={use_amp} | split_seed={args.split_seed} | model_seed={args.model_seed}")
    print("Academic protocol:")
    print("  - CV-TRN is an auxiliary expert, not the Fourier main model.")
    print("  - Original validation/test splits are never used for CV-TRN fold training.")
    print("  - Test probabilities are exported for final fixed fusion; test labels are not scored here.")

    full_dataset = common10b.build_full_dataset(args)
    train_idx, val_idx, test_idx = common10b.make_aligned_split(full_dataset, args.split_seed)
    if not args.skip_alignment_check:
        common10b.check_alignment(full_dataset, val_idx, test_idx, args.alignment_cache)
    labels_all = impl.np.asarray(full_dataset.labels, dtype=impl.np.int64)
    snrs_all = impl.np.asarray(full_dataset.snrs, dtype=impl.np.int32)
    class_names = getattr(full_dataset, "mod_classes", common10b.DEFAULT_MOD_CLASSES)
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Split: train={len(train_idx):,} | val={len(val_idx):,} | test={len(test_idx):,}")

    val_loader = impl.make_loader(full_dataset, val_idx, args.eval_batch_size, False, args.num_workers, device)
    test_loader = impl.make_loader(full_dataset, test_idx, args.eval_batch_size, False, args.num_workers, device)

    train_oof = impl.np.zeros((len(train_idx), common10b.NUM_CLASSES), dtype=impl.np.float32)
    y_train = labels_all[train_idx]
    s_train = snrs_all[train_idx]
    val_probs, test_probs = [], []

    composite = impl.np.asarray([f"{int(y)}_{int(s)}" for y, s in zip(y_train, s_train)])
    skf = impl.StratifiedKFold(n_splits=int(args.folds), shuffle=True, random_state=int(args.model_seed))
    for fold, (tr, va) in enumerate(skf.split(train_idx, composite), 1):
        fold_train_idx = train_idx[tr]
        fold_holdout_idx = train_idx[va]
        print("\n" + "-" * 120)
        print(f"OOF CV-TRN-v2 10B fold {fold}/{args.folds} | train={len(fold_train_idx):,} | holdout={len(fold_holdout_idx):,}")
        train_loader = impl.make_loader(full_dataset, fold_train_idx, args.batch_size, True, args.num_workers, device, drop_last=True)
        holdout_loader = impl.make_loader(full_dataset, fold_holdout_idx, args.eval_batch_size, False, args.num_workers, device)
        model, best_score = impl.train_fold(args, fold, train_loader, holdout_loader, device, use_amp)
        ph, yh, sh = common10b.collect_probs(model, holdout_loader, device)
        train_oof[va] = ph
        pv, yv, sv = common10b.collect_probs(model, val_loader, device)
        pt, yt, st = common10b.collect_probs(model, test_loader, device)
        val_probs.append(pv)
        test_probs.append(pt)
        common10b.print_metrics(f"Fold {fold} Holdout", common10b.metrics_from_probs(ph, yh, sh), best_score)
        common10b.print_metrics(f"Fold {fold} Val", common10b.metrics_from_probs(pv, yv, sv), common10b.score_metrics(common10b.metrics_from_probs(pv, yv, sv)))

    val_prob = impl.log_average(val_probs)
    test_prob = impl.log_average(test_probs)
    train_m = common10b.metrics_from_probs(train_oof, y_train, s_train)
    val_m = common10b.metrics_from_probs(val_prob, labels_all[val_idx], snrs_all[val_idx])
    common10b.print_metrics("CVTRN-v2 10B OOF Train", train_m, common10b.score_metrics(train_m))
    common10b.print_metrics("CVTRN-v2 10B fold-soup Val", val_m, common10b.score_metrics(val_m))

    impl.np.savez_compressed(
        args.output_cache,
        train_prob=train_oof.astype(impl.np.float32),
        val_prob=val_prob.astype(impl.np.float32),
        test_prob=test_prob.astype(impl.np.float32),
        train_indices=train_idx.astype(impl.np.int64),
        val_indices=val_idx.astype(impl.np.int64),
        test_indices=test_idx.astype(impl.np.int64),
        labels_train=y_train.astype(impl.np.int64),
        snrs_train=s_train.astype(impl.np.int32),
        labels_val=labels_all[val_idx].astype(impl.np.int64),
        snrs_val=snrs_all[val_idx].astype(impl.np.int32),
        labels_test=labels_all[test_idx].astype(impl.np.int64),
        snrs_test=snrs_all[test_idx].astype(impl.np.int32),
        mod_classes=impl.np.asarray(class_names),
        protocol=impl.np.asarray(["RML2016.10B train-split OOF CV-TRN-v2 experts; test probabilities exported without scoring"]),
    )
    print(f"[*] CV-TRN 10B OOF cache saved: {args.output_cache}")


if __name__ == "__main__":
    main()
