import argparse
import os
import time

import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

import train_cv_trn_aux_2016 as common
import train_cv_trn_aux_v2_2016 as v2


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Train CV-TRN-v2 cross-fitted auxiliary experts. "
            "Exports train-OOF probabilities for clean stacking/meta-router training."
        )
    )
    p.add_argument("--split_seed", type=int, default=1)
    p.add_argument("--model_seed", type=int, default=41)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=180)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--eval_batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--eta_min", type=float, default=1.5e-5)
    p.add_argument("--grad_clip", type=float, default=2.0)
    p.add_argument("--patience", type=int, default=36)
    p.add_argument("--label_smoothing", type=float, default=0.02)
    p.add_argument("--iq_head_weight", type=float, default=0.25)
    p.add_argument("--consistency_weight", type=float, default=0.05)
    p.add_argument("--disable_amp", action="store_true")
    p.add_argument("--force_restart", action="store_true")

    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--frame_len", type=int, default=16)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--input_denoise", action="store_true")
    p.add_argument("--denoise_hidden", type=int, default=8)
    p.add_argument("--denoise_partial_ratio", type=float, default=0.5)
    p.add_argument("--denoise_kernel", type=int, default=5)
    p.add_argument("--denoise_cap", type=float, default=0.20)
    p.add_argument("--denoise_gate_bias", type=float, default=-2.0)

    p.add_argument("--rpo_prob", type=float, default=0.90)
    p.add_argument("--roll_prob", type=float, default=0.60)
    p.add_argument("--roll_max", type=int, default=10)
    p.add_argument("--noise_prob", type=float, default=0.12)
    p.add_argument("--noise_std", type=float, default=0.015)
    p.add_argument("--no_augment", action="store_true")

    p.add_argument("--data_path", type=str, default=common.relpath("raw_data", "RML2016.10a_dict.pkl"))
    p.add_argument("--cache_dir", type=str, default=common.relpath("feature_cache"))
    p.add_argument("--alignment_cache", type=str, default=common.relpath("results", "greedy_soup_identity_valtest_probs_for_gamc_fusion.npz"))
    p.add_argument("--skip_alignment_check", action="store_true")
    p.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help=(
            "Optional explicit suffix for fold checkpoints and default cache. "
            "Use this when training smaller dim/depth variants with the same seed."
        ),
    )
    p.add_argument("--output_cache", type=str, default="")
    return p.parse_args()


def run_suffix(args):
    suffix = str(getattr(args, "output_suffix", "") or "").strip()
    if suffix:
        return suffix
    return f"cv_trn_aux_v2_oof_mseed{args.model_seed}_split{args.split_seed}"


def make_loader(full_dataset, indices, batch_size, shuffle, workers, device, drop_last=False):
    return DataLoader(
        common.IQSubset(full_dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        drop_last=drop_last,
    )


def log_average(prob_list):
    logs = [np.log(common.normalize_probs(p) + 1e-12) for p in prob_list]
    z = np.mean(np.stack(logs, axis=0), axis=0)
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / (e.sum(axis=1, keepdims=True) + 1e-12)).astype(np.float32)


def checkpoint_paths(args, fold):
    base = run_suffix(args)
    suffix = f"{base}_fold{fold}"
    return (
        common.relpath("checkpoints", f"best_{suffix}.pth"),
        common.relpath("checkpoints", f"latest_{suffix}.pth"),
    )


def save_fold_checkpoint(path, model, optimizer, scheduler, epoch, best_score, best_metrics, args, fold):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": int(epoch),
            "best_score": float(best_score),
            "best_metrics": best_metrics,
            "model_seed": int(args.model_seed),
            "split_seed": int(args.split_seed),
            "fold": int(fold),
            "model_type": "cv_trn_aux_v2_oof_2016",
            "args": vars(args),
        },
        path,
    )


def train_fold(args, fold, train_loader, holdout_loader, device, use_amp):
    common.set_seed(int(args.model_seed) + 1000 * int(fold))
    model = v2.build_model(args, device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
    scaler = common.make_grad_scaler(use_amp)
    best_path, latest_path = checkpoint_paths(args, fold)

    start_epoch = 1
    best_score = -1e9
    best_metrics = None
    stale = 0

    if os.path.exists(latest_path) and not args.force_restart:
        ckpt = common.safe_torch_load(latest_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_score = float(ckpt.get("best_score", best_score))
        best_metrics = ckpt.get("best_metrics", None)
        print(f"[*] Resume fold {fold}: start_epoch={start_epoch}, best_score={best_score:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_m = v2.train_one_epoch(model, train_loader, optimizer, scaler, device, args, use_amp)
        scheduler.step()
        prob, y, s = common.collect_probs(model, holdout_loader, device)
        metrics = common.metrics_from_probs(prob, y, s)
        score = common.score_metrics(metrics)
        print(
            f"\nOOF fold {fold} | Epoch {epoch:03d}/{args.epochs} | {time.time() - t0:.1f}s | "
            f"loss={train_m['loss']:.4f} fused={train_m['fused']:.4f} iq={train_m['iq']:.4f} "
            f"cons={train_m['cons']:.4f} acc={train_m['acc']:.2f}% skipped={train_m['skipped']}"
        )
        common.print_metrics("Holdout", metrics, score)

        save_fold_checkpoint(latest_path, model, optimizer, scheduler, epoch, best_score, best_metrics, args, fold)
        if score > best_score:
            best_score = score
            best_metrics = metrics
            stale = 0
            save_fold_checkpoint(best_path, model, optimizer, scheduler, epoch, best_score, best_metrics, args, fold)
            print(f"[*] New fold best saved: {best_path}")
        else:
            stale += 1
            if stale >= int(args.patience):
                print(f"[*] Early stop fold {fold}: stale={stale}")
                break

    if os.path.exists(best_path):
        ckpt = common.safe_torch_load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    return model, best_score


def main():
    args = parse_args()
    if not args.output_cache:
        args.output_cache = common.relpath(
            "results",
            f"{run_suffix(args)}_trainvaltest_probs_for_meta.npz",
        )
    os.makedirs(common.relpath("checkpoints"), exist_ok=True)
    os.makedirs(common.relpath("results"), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.disable_amp)

    print("=" * 120)
    print("Train CV-TRN-v2 OOF auxiliary experts")
    print("=" * 120)
    print("Academic protocol:")
    print("  - OOF train probabilities are generated by models that never saw that fold.")
    print("  - Original validation/test splits are never used for CV-TRN fold training.")
    print("  - Test probabilities are exported for final fusion only; test accuracy is not printed here.")
    print(f"device={device} | amp={use_amp} | split_seed={args.split_seed} | model_seed={args.model_seed}")

    full_dataset = common.build_full_dataset(args)
    train_idx, val_idx, test_idx = common.make_aligned_split(full_dataset, args.split_seed)
    if not args.skip_alignment_check:
        common.check_alignment(full_dataset, val_idx, test_idx, args.alignment_cache)
    labels = np.asarray(full_dataset.labels, dtype=np.int64)
    snrs = np.asarray(full_dataset.snrs, dtype=np.int32)
    y_train = labels[train_idx]
    s_train = snrs[train_idx]
    composite = np.asarray([f"{int(y)}_{int(s)}" for y, s in zip(y_train, s_train)])

    val_loader = make_loader(full_dataset, val_idx, args.eval_batch_size, False, args.num_workers, device)
    test_loader = make_loader(full_dataset, test_idx, args.eval_batch_size, False, args.num_workers, device)
    skf = StratifiedKFold(n_splits=int(args.folds), shuffle=True, random_state=int(args.model_seed))

    train_oof = np.zeros((len(train_idx), common.NUM_CLASSES), dtype=np.float32)
    val_probs, test_probs = [], []
    labels_val = snrs_val = labels_test = snrs_test = None

    probe = v2.build_model(args, device)
    print(f"Trainable params: {sum(p.numel() for p in probe.parameters() if p.requires_grad) / 1e6:.3f}M")
    del probe
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for fold, (tr, va) in enumerate(skf.split(train_idx, composite), 1):
        print("\n" + "=" * 120)
        print(f"OOF CV-TRN-v2 fold {fold}/{args.folds}")
        print("=" * 120)
        fold_train_idx = train_idx[tr]
        fold_holdout_idx = train_idx[va]
        train_loader = make_loader(full_dataset, fold_train_idx, args.batch_size, True, args.num_workers, device, drop_last=True)
        holdout_loader = make_loader(full_dataset, fold_holdout_idx, args.eval_batch_size, False, args.num_workers, device)

        model, best_score = train_fold(args, fold, train_loader, holdout_loader, device, use_amp)
        ph, yh, sh = common.collect_probs(model, holdout_loader, device)
        train_oof[va] = ph.astype(np.float32)
        pv, yv, sv = common.collect_probs(model, val_loader, device)
        pt, yt, st = common.collect_probs(model, test_loader, device)
        common.print_metrics(f"Fold {fold} Holdout", common.metrics_from_probs(ph, yh, sh), best_score)
        common.print_metrics(f"Fold {fold} Val", common.metrics_from_probs(pv, yv, sv), common.score_metrics(common.metrics_from_probs(pv, yv, sv)))
        val_probs.append(pv)
        test_probs.append(pt)
        labels_val, snrs_val, labels_test, snrs_test = yv, sv, yt, st
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    val_prob = log_average(val_probs)
    test_prob = log_average(test_probs)
    train_m = common.metrics_from_probs(train_oof, y_train, s_train)
    val_m = common.metrics_from_probs(val_prob, labels_val, snrs_val)
    common.print_metrics("CVTRN-v2 OOF Train", train_m, common.score_metrics(train_m))
    common.print_metrics("CVTRN-v2 fold-soup Val", val_m, common.score_metrics(val_m))

    np.savez_compressed(
        args.output_cache,
        train_prob=train_oof.astype(np.float32),
        val_prob=val_prob.astype(np.float32),
        test_prob=test_prob.astype(np.float32),
        fold_val_probs=np.stack(val_probs, axis=0).astype(np.float32),
        fold_test_probs=np.stack(test_probs, axis=0).astype(np.float32),
        train_indices=train_idx.astype(np.int64),
        val_indices=val_idx.astype(np.int64),
        test_indices=test_idx.astype(np.int64),
        labels_train=y_train.astype(np.int64),
        snrs_train=s_train.astype(np.int32),
        labels_val=labels_val.astype(np.int64),
        snrs_val=snrs_val.astype(np.int32),
        labels_test=labels_test.astype(np.int64),
        snrs_test=snrs_test.astype(np.int32),
        mod_classes=np.asarray(getattr(full_dataset, "mod_classes", common.DEFAULT_MOD_CLASSES)),
        protocol=np.asarray(["train-split OOF CV-TRN-v2 experts; test probabilities exported without scoring"]),
    )
    print(f"[*] CV-TRN OOF cache saved: {args.output_cache}")


if __name__ == "__main__":
    main()
