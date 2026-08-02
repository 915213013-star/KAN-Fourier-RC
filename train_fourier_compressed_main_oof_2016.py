"""Leakage-controlled fold-held-out OOF training for compact KAN-Fourier.

For each fold, the held-out rows are excluded from gradient updates and provide
the fold-level checkpoint score.  The selected checkpoint exports those same
OOF rows.  Policy selection remains confined to the independent validation
partition, and the official test partition is evaluated only after freezing.
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

import oof_protocol as oofp
import train_cv_trn_aux_2016 as common
import train_fourier_compressed_main_seeds_2016 as seed_train
import train_fourier_main_oof_2016 as fm
from model_moe_attention_compressed import COMPRESSED_VARIANTS, build_compressed_model

try:
    import model_stability as stable_base
except Exception:
    stable_base = None


NUM_CLASSES = 11
HOS_DIM = 20


def parse_args():
    p = argparse.ArgumentParser(
        description="Train compact KAN-Fourier with the paper's fold-held-out OOF protocol."
    )
    p.add_argument("--variant", type=str, default="small", choices=sorted(COMPRESSED_VARIANTS))
    p.add_argument("--split_seed", type=int, default=1)
    p.add_argument("--model_seed", type=int, default=181)
    p.add_argument(
        "--fold_model_seeds",
        type=int,
        nargs="*",
        default=None,
        help="A single seed list fixed before all outer folds; every member follows the same audited OOF protocol.",
    )
    p.add_argument(
        "--fold_soup_mode",
        type=str,
        default="weight",
        choices=["weight", "prob"],
        help="Combine fixed per-fold members by checkpoint averaging or probability log averaging.",
    )
    p.add_argument("--fold_soup_weights", type=float, nargs="*", default=None)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=220)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--eval_batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--eta_min", type=float, default=4e-6)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--alpha_supcon", type=float, default=0.30)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--soft_rank_weight", type=float, default=0.0)
    p.add_argument("--soft_rank_keep_ratio", type=float, default=0.82)
    p.add_argument("--soft_rank_every", type=int, default=12)
    p.add_argument("--negative_snr_weight", type=float, default=1.0)
    p.add_argument("--high_snr_weight", type=float, default=1.0)
    p.add_argument("--edge_snr_weight", type=float, default=1.0)
    p.add_argument("--transition_snr_weight", type=float, default=1.0)
    p.add_argument("--roll_prob", type=float, default=0.0)
    p.add_argument("--roll_max", type=int, default=6)
    p.add_argument("--iq_noise_prob", type=float, default=0.0)
    p.add_argument("--iq_noise_std", type=float, default=0.010)
    p.add_argument("--augment_warmup_epochs", type=int, default=0)
    p.add_argument("--snr_weight_warmup_epochs", type=int, default=0)
    p.add_argument("--patience", type=int, default=42)
    p.add_argument("--force_restart", action="store_true")
    p.add_argument("--run_tag", type=str, default="")

    # Accepted only to fail loudly when an incompatible legacy command is reused.
    p.add_argument("--fold_seed_map", type=str, nargs="*", default=[], help=argparse.SUPPRESS)
    p.add_argument("--init_checkpoint", type=str, default="", help=argparse.SUPPRESS)
    p.add_argument("--init_checkpoint_template", type=str, default="", help=argparse.SUPPRESS)
    p.add_argument("--resume_best_reset", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--freeze_good_folds", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--freeze_min_overall", type=float, default=62.5, help=argparse.SUPPRESS)
    p.add_argument("--freeze_min_high", type=float, default=91.0, help=argparse.SUPPRESS)

    p.add_argument("--data_path", type=str, default=common.relpath("raw_data", "RML2016.10a_dict.pkl"))
    p.add_argument("--cache_dir", type=str, default=common.relpath("feature_cache"))
    p.add_argument(
        "--alignment_cache",
        type=str,
        default=common.relpath("results", "greedy_soup_identity_valtest_probs_for_gamc_fusion.npz"),
    )
    p.add_argument("--skip_alignment_check", action="store_true")
    p.add_argument("--output_cache", type=str, default="")
    oofp.add_protocol_args(p)
    return p.parse_args()


def _reject_legacy_controls(args):
    rejected = []
    if list(getattr(args, "fold_seed_map", []) or []):
        rejected.append("--fold_seed_map")
    if str(getattr(args, "init_checkpoint", "") or "").strip():
        rejected.append("--init_checkpoint")
    if str(getattr(args, "init_checkpoint_template", "") or "").strip():
        rejected.append("--init_checkpoint_template")
    if bool(getattr(args, "resume_best_reset", False)):
        rejected.append("--resume_best_reset")
    if bool(getattr(args, "freeze_good_folds", False)):
        rejected.append("--freeze_good_folds")
    if rejected:
        raise ValueError(
            "The release protocol rejects non-predeclared legacy controls: "
            + ", ".join(rejected)
            + ". Use one predeclared --fold_model_seeds list for every outer fold."
        )


def build_model(args, device):
    model = build_compressed_model(args.variant, num_classes=NUM_CLASSES, hos_dim=HOS_DIM).to(device)
    if stable_base is not None:
        try:
            stable_base.patch_lieqkan_stability(model, name=f"fourier_compressed_oof_{args.variant}")
        except Exception as exc:
            print(f"[!] Stability patch skipped: {exc}")
    return model


def _clean_tag(value):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value or "").strip())


def checkpoint_paths(args, fold):
    tag = _clean_tag(getattr(args, "run_tag", ""))
    tag_part = f"_{tag}" if tag else ""
    suffix = (
        f"fourier_compressed_oof_{oofp.PROTOCOL_TAG}_{args.variant}{tag_part}_"
        f"mseed{args.model_seed}_fold{fold}_split{args.split_seed}"
    )
    return (
        common.relpath("checkpoints", f"best_{suffix}.pth"),
        common.relpath("checkpoints", f"latest_{suffix}.pth"),
    )


def normalize_weights(raw_weights, count):
    if count < 1:
        raise ValueError("At least one fixed model seed is required.")
    if raw_weights is None or len(raw_weights) == 0:
        weights = np.ones(count, dtype=np.float64)
    else:
        if len(raw_weights) != count:
            raise ValueError(f"--fold_soup_weights must have {count} values, got {len(raw_weights)}")
        weights = np.asarray(raw_weights, dtype=np.float64)
    if np.any(weights < 0) or not np.all(np.isfinite(weights)) or float(weights.sum()) <= 0:
        raise ValueError("Fold-soup weights must be finite, nonnegative, and have a positive sum.")
    return (weights / weights.sum()).astype(np.float64)


def average_checkpoint_states(paths, weights, device):
    averaged = None
    for path, weight in zip(paths, weights):
        checkpoint = common.safe_torch_load(path, map_location=device)
        state = checkpoint["model_state_dict"]
        if averaged is None:
            averaged = {key: value.detach().clone().float() * float(weight) for key, value in state.items()}
        else:
            if set(averaged) != set(state):
                raise RuntimeError(f"Checkpoint keys differ for fixed weight soup: {path}")
            for key, value in state.items():
                averaged[key].add_(value.detach().float(), alpha=float(weight))
    return averaged


def _train_member(
    args,
    fold,
    full_dataset,
    outer_train_idx,
    outer_holdout_idx,
    device,
    policy_validation_idx,
    official_test_idx,
):
    original_build_model = fm.build_model
    original_checkpoint_paths = fm.checkpoint_paths
    original_train_one_epoch = fm.train_one_epoch
    try:
        fm.build_model = lambda dev: build_model(args, dev)
        fm.checkpoint_paths = lambda _args, member_fold: checkpoint_paths(args, member_fold)
        fm.train_one_epoch = (
            lambda model, loader, optimizer, scheduler, ce, supcon, dev, member_args: seed_train.train_one_epoch_elastic(
                model,
                loader,
                optimizer,
                scheduler,
                ce,
                supcon,
                dev,
                member_args,
                epoch=getattr(member_args, "_current_epoch", 1),
            )
        )
        return fm.train_fold(
            args,
            fold,
            full_dataset,
            outer_train_idx,
            outer_holdout_idx,
            device,
            policy_validation_idx=policy_validation_idx,
            official_test_idx=official_test_idx,
        )
    finally:
        fm.build_model = original_build_model
        fm.checkpoint_paths = original_checkpoint_paths
        fm.train_one_epoch = original_train_one_epoch


def _default_output(args):
    return common.relpath(
        "results",
        (
            f"fourier_compressed_oof_{oofp.PROTOCOL_TAG}_{args.variant}_mseed{args.model_seed}_"
            f"f{args.folds}e{args.epochs}_split{args.split_seed}_trainvaltest_probs_for_meta.npz"
        ),
    )


def main_with_args(args, title="Train compact KAN-Fourier OOF cache"):
    _reject_legacy_controls(args)
    base_model_seed = int(args.model_seed)
    fixed_member_seeds = [int(value) for value in (args.fold_model_seeds or [base_model_seed])]
    if len(set(fixed_member_seeds)) != len(fixed_member_seeds):
        raise ValueError("--fold_model_seeds contains duplicate values.")
    fixed_weights = normalize_weights(args.fold_soup_weights, len(fixed_member_seeds))
    if not args.output_cache:
        args.output_cache = _default_output(args)

    os.makedirs(common.relpath("checkpoints"), exist_ok=True)
    os.makedirs(common.relpath("results"), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 120)
    print(title)
    print("=" * 120)
    print(f"OOF protocol: {oofp.PROTOCOL_ID}")
    print("  - Each fold holdout is excluded from gradient updates.")
    print("  - The fold holdout supplies that fold's checkpoint score and OOF rows.")
    print("  - Member seeds and soup weights are fixed before fold training.")
    print("  - The same predeclared seed list and soup weights are used for every outer fold.")
    print("  - Independent validation selects the correction policy; official test is frozen evaluation only.")
    print("  - This is leakage-controlled fold training, not a nested-CV estimate.")
    print(
        f"device={device} | variant={args.variant} | split_seed={args.split_seed} | "
        f"fixed_member_seeds={fixed_member_seeds} | folds={args.folds} | soup={args.fold_soup_mode}"
    )

    full_dataset = common.build_full_dataset(args)
    train_idx, val_idx, test_idx = common.make_aligned_split(full_dataset, args.split_seed)
    if not args.skip_alignment_check:
        common.check_alignment(full_dataset, val_idx, test_idx, args.alignment_cache)
    labels_all = np.asarray(full_dataset.labels, dtype=np.int64)
    snrs_all = np.asarray(full_dataset.snrs, dtype=np.int32)
    labels_train = labels_all[train_idx]
    snrs_train = snrs_all[train_idx]
    composite = np.asarray([f"{int(label)}_{int(snr)}" for label, snr in zip(labels_train, snrs_train)])

    val_loader = fm.make_loader(full_dataset, val_idx, args.eval_batch_size, False, args.num_workers, device)
    test_loader = fm.make_loader(full_dataset, test_idx, args.eval_batch_size, False, args.num_workers, device)
    splitter = StratifiedKFold(n_splits=int(args.folds), shuffle=True, random_state=base_model_seed)
    train_oof = np.zeros((len(train_idx), NUM_CLASSES), dtype=np.float32)
    fold_val_probs, fold_test_probs = [], []
    fold_records = []
    labels_val = snrs_val = labels_test = snrs_test = None

    probe = build_model(args, device)
    print(f"Trainable params: {sum(p.numel() for p in probe.parameters() if p.requires_grad) / 1e6:.3f}M")
    del probe
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for fold, (outer_train_pos, outer_holdout_pos) in enumerate(splitter.split(train_idx, composite), 1):
        outer_train_idx = train_idx[outer_train_pos]
        outer_holdout_idx = train_idx[outer_holdout_pos]
        holdout_loader = fm.make_loader(
            full_dataset, outer_holdout_idx, args.eval_batch_size, False, args.num_workers, device
        )
        member_holdout, member_val, member_test = [], [], []
        member_paths, member_records = [], []

        print("\n" + "=" * 120)
        print(
            f"Compact {args.variant} OOF fold {fold}/{args.folds} | "
            f"outer-train={len(outer_train_idx):,} | outer-holdout={len(outer_holdout_idx):,}"
        )
        print("=" * 120)
        for member_seed in fixed_member_seeds:
            args.model_seed = int(member_seed)
            args._model_type = f"fourier_compressed_oof_{args.variant}"
            print(f"[*] Fold {fold}, fixed member seed {member_seed}")
            model, selection_score, selected_epoch, checkpoint_meta = _train_member(
                args,
                fold,
                full_dataset,
                outer_train_idx,
                outer_holdout_idx,
                device,
                val_idx,
                test_idx,
            )
            member_paths.append(checkpoint_paths(args, fold)[0])
            member_records.append(
                {
                    "model_seed": int(member_seed),
                    "selected_epoch": int(selected_epoch),
                    "fold_selection_score": float(selection_score),
                    "checkpoint": checkpoint_meta,
                }
            )
            if args.fold_soup_mode == "prob" or len(fixed_member_seeds) == 1:
                ph, yh, sh = fm.collect_probs(model, holdout_loader, device)
                pv, yv, sv = fm.collect_probs(model, val_loader, device)
                pt, yt, st = fm.collect_probs(model, test_loader, device)
                member_holdout.append(ph)
                member_val.append(pv)
                member_test.append(pt)
                labels_val, snrs_val, labels_test, snrs_test = yv, sv, yt, st
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if args.fold_soup_mode == "weight" and len(fixed_member_seeds) > 1:
            args.model_seed = int(fixed_member_seeds[0])
            soup_model = build_model(args, device)
            soup_model.load_state_dict(average_checkpoint_states(member_paths, fixed_weights, device), strict=True)
            ph, yh, sh = fm.collect_probs(soup_model, holdout_loader, device)
            pv, yv, sv = fm.collect_probs(soup_model, val_loader, device)
            pt, yt, st = fm.collect_probs(soup_model, test_loader, device)
            labels_val, snrs_val, labels_test, snrs_test = yv, sv, yt, st
            del soup_model
        else:
            ph = fm.log_average(member_holdout)
            pv = fm.log_average(member_val)
            pt = fm.log_average(member_test)

        train_oof[outer_holdout_pos] = ph
        fold_val_probs.append(pv)
        fold_test_probs.append(pt)
        holdout_metrics = common.metrics_from_probs(ph, yh, sh)
        common.print_metrics(
            f"Fold {fold} selected-checkpoint OOF holdout",
            holdout_metrics,
            common.score_metrics(holdout_metrics),
        )
        fold_records.append(
            {
                "fold": int(fold),
                "fixed_member_seeds": fixed_member_seeds,
                "fixed_weights": fixed_weights.tolist(),
                "members": member_records,
            }
        )

    args.model_seed = base_model_seed
    if hasattr(args, "_model_type"):
        delattr(args, "_model_type")
    val_prob = fm.log_average(fold_val_probs)
    test_prob = fm.log_average(fold_test_probs)
    train_metrics = common.metrics_from_probs(train_oof, labels_train, snrs_train)
    val_metrics = common.metrics_from_probs(val_prob, labels_val, snrs_val)
    common.print_metrics("Compact KAN-Fourier OOF train", train_metrics, common.score_metrics(train_metrics))
    common.print_metrics("Compact KAN-Fourier validation export", val_metrics, common.score_metrics(val_metrics))
    print("[*] Test probabilities exported; test labels were not used for fitting or selection.")

    protocol_summary = oofp.cache_protocol_summary(args, fold_records)
    protocol_summary.update(
        {
            "variant": str(args.variant),
            "fixed_member_seeds": fixed_member_seeds,
            "fold_soup_mode": str(args.fold_soup_mode),
            "fixed_fold_soup_weights": fixed_weights.tolist(),
            "non_predeclared_legacy_controls": "rejected",
        }
    )
    np.savez_compressed(
        args.output_cache,
        train_prob=train_oof.astype(np.float32),
        val_prob=val_prob.astype(np.float32),
        test_prob=test_prob.astype(np.float32),
        fold_val_probs=np.stack(fold_val_probs, axis=0).astype(np.float32),
        fold_test_probs=np.stack(fold_test_probs, axis=0).astype(np.float32),
        train_indices=train_idx.astype(np.int64),
        val_indices=val_idx.astype(np.int64),
        test_indices=test_idx.astype(np.int64),
        labels_train=labels_train.astype(np.int64),
        snrs_train=snrs_train.astype(np.int32),
        labels_val=labels_val.astype(np.int64),
        snrs_val=snrs_val.astype(np.int32),
        labels_test=labels_test.astype(np.int64),
        snrs_test=snrs_test.astype(np.int32),
        mod_classes=np.asarray(getattr(full_dataset, "mod_classes", common.DEFAULT_MOD_CLASSES)),
        variant=np.asarray([args.variant]),
        fixed_member_seeds=np.asarray(fixed_member_seeds, dtype=np.int64),
        fold_soup_mode=np.asarray([args.fold_soup_mode]),
        fold_soup_weights=fixed_weights.astype(np.float32),
        protocol_id=np.asarray([oofp.PROTOCOL_ID]),
        config_fingerprint=np.asarray([oofp.config_fingerprint(args)]),
        protocol_metadata_json=np.asarray([oofp.metadata_json(protocol_summary)]),
        fold_records_json=np.asarray([json.dumps(fold_records, sort_keys=True, separators=(",", ":"))]),
        protocol=np.asarray(
            [
                "leakage-controlled train-split OOF; fold holdout excluded from gradients and used "
                "for fold checkpoint selection plus OOF export; independent validation selects policy; "
                "official test used only after freezing"
            ]
        ),
    )
    print(f"[*] Compact KAN-Fourier OOF cache saved: {args.output_cache}")
    return args.output_cache


def main():
    return main_with_args(parse_args())


if __name__ == "__main__":
    main()
