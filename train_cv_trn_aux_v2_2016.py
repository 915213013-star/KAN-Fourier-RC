import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import train_cv_trn_aux_2016 as common
from model_cv_trn_aux_v2_2016 import build_cv_trn_aux_v2_model


def parse_args():
    p = argparse.ArgumentParser(
        description="Train CV-TRN-v2 auxiliary experts with I/Q/fused supervision for RML2016.10A."
    )
    p.add_argument("--split_seed", type=int, default=1)
    p.add_argument("--model_seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--epochs", type=int, default=220)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--eta_min", type=float, default=1.5e-5)
    p.add_argument("--grad_clip", type=float, default=2.0)
    p.add_argument("--patience", type=int, default=240)
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
            "Optional explicit suffix for checkpoints/results. "
            "Useful when training smaller dim/depth variants with the same seed."
        ),
    )
    return p.parse_args()


def run_suffix(args, model_seed):
    suffix = str(getattr(args, "output_suffix", "") or "").strip()
    if suffix:
        return suffix
    return f"cv_trn_aux_v2_mseed{model_seed}_split{args.split_seed}"


def build_model(args, device):
    return build_cv_trn_aux_v2_model(
        device=device,
        num_classes=common.NUM_CLASSES,
        dim=args.dim,
        depth=args.depth,
        num_heads=args.heads,
        frame_len=args.frame_len,
        stride=args.stride,
        dropout=args.dropout,
        input_denoise=getattr(args, "input_denoise", False),
        denoise_hidden=getattr(args, "denoise_hidden", 8),
        denoise_partial_ratio=getattr(args, "denoise_partial_ratio", 0.5),
        denoise_kernel=getattr(args, "denoise_kernel", 5),
        denoise_cap=getattr(args, "denoise_cap", 0.20),
        denoise_gate_bias=getattr(args, "denoise_gate_bias", -2.0),
    )


def kl_to_mean(logits_a, logits_b, logits_f):
    log_pa = F.log_softmax(logits_a.float(), dim=1)
    log_pb = F.log_softmax(logits_b.float(), dim=1)
    log_pf = F.log_softmax(logits_f.float(), dim=1)
    target = (log_pa.exp() + log_pb.exp() + log_pf.exp()) / 3.0
    loss = 0.0
    for log_p in (log_pa, log_pb, log_pf):
        loss = loss + F.kl_div(log_p, target, reduction="batchmean")
    return loss / 3.0


def train_one_epoch(model, loader, optimizer, scaler, device, args, use_amp):
    model.train()
    total_loss, total_fused, total_iq, total_cons = 0.0, 0.0, 0.0, 0.0
    total, correct, skipped = 0, 0, 0

    for step, (x, y, snr) in enumerate(loader, 1):
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True)
        snr = snr.to(device, non_blocking=True)
        x = common.random_rpo_roll_aug(x, args)

        optimizer.zero_grad(set_to_none=True)
        with common.autocast_context(device.type, use_amp):
            out = model(x, return_aux=True)
            logits = out["logits"]
            logits_i = out["logits_i"]
            logits_q = out["logits_q"]
            fused = common.weighted_ce(logits.float(), y, snr, args.label_smoothing)
            loss_i = common.weighted_ce(logits_i.float(), y, snr, args.label_smoothing)
            loss_q = common.weighted_ce(logits_q.float(), y, snr, args.label_smoothing)
            iq_loss = 0.5 * (loss_i + loss_q)
            cons = kl_to_mean(logits_i, logits_q, logits)
            loss = fused + args.iq_head_weight * iq_loss + args.consistency_weight * cons

        if not torch.isfinite(loss):
            skipped += 1
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not torch.isfinite(grad):
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            continue
        scaler.step(optimizer)
        scaler.update()

        bsz = y.size(0)
        total_loss += float(loss.detach().cpu()) * bsz
        total_fused += float(fused.detach().cpu()) * bsz
        total_iq += float(iq_loss.detach().cpu()) * bsz
        total_cons += float(cons.detach().cpu()) * bsz
        correct += int((logits.argmax(dim=1) == y).sum().item())
        total += bsz

        if step % 120 == 0:
            print(f"    batch {step}/{len(loader)} loss={loss.item():.4f} acc={100.0 * correct / max(1, total):.2f}%")

    total = max(1, total)
    return {
        "loss": total_loss / total,
        "fused": total_fused / total,
        "iq": total_iq / total,
        "cons": total_cons / total,
        "acc": 100.0 * correct / total,
        "skipped": skipped,
    }


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_score, best_metrics, args, model_seed):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": int(epoch),
            "best_score": float(best_score),
            "best_metrics": best_metrics,
            "model_seed": int(model_seed),
            "split_seed": int(args.split_seed),
            "model_type": "cv_trn_aux_v2_2016",
            "args": vars(args),
        },
        path,
    )


def export_seed_cache(model, val_loader, test_loader, class_names, args, model_seed, device):
    val_prob, labels_val, snrs_val = common.collect_probs(model, val_loader, device)
    test_prob, labels_test, snrs_test = common.collect_probs(model, test_loader, device)
    val_m = common.metrics_from_probs(val_prob, labels_val, snrs_val)
    test_m = common.metrics_from_probs(test_prob, labels_test, snrs_test)
    common.print_metrics(f"CVTRN-v2 s{model_seed} Val", val_m, common.score_metrics(val_m))
    common.print_metrics(f"CVTRN-v2 s{model_seed} Test", test_m)

    suffix = run_suffix(args, model_seed)
    cache_path = common.relpath("results", f"{suffix}_valtest_probs_for_fusion.npz")
    np.savez_compressed(
        cache_path,
        val_prob=val_prob.astype(np.float32),
        test_prob=test_prob.astype(np.float32),
        labels_val=labels_val.astype(np.int64),
        snrs_val=snrs_val.astype(np.int32),
        labels_test=labels_test.astype(np.int64),
        snrs_test=snrs_test.astype(np.int32),
        mod_classes=np.asarray(class_names),
    )
    print(f"[*] Seed fusion cache saved: {cache_path}")


def train_one_seed(args, model_seed, train_loader, val_loader, test_loader, class_names, device, use_amp):
    common.set_seed(model_seed)
    model = build_model(args, device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
    scaler = common.make_grad_scaler(use_amp)

    suffix = run_suffix(args, model_seed)
    best_path = common.relpath("checkpoints", f"best_{suffix}.pth")
    latest_path = common.relpath("checkpoints", f"latest_{suffix}.pth")

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
        print(f"[*] Resume seed {model_seed}: start_epoch={start_epoch}, best_score={best_score:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, optimizer, scaler, device, args, use_amp)
        scheduler.step()
        val_prob, labels_val, snrs_val = common.collect_probs(model, val_loader, device)
        val_m = common.metrics_from_probs(val_prob, labels_val, snrs_val)
        val_score = common.score_metrics(val_m)

        print(
            f"\nSeed {model_seed} | Epoch {epoch:03d}/{args.epochs} | {time.time() - t0:.1f}s | "
            f"loss={train_m['loss']:.4f} fused={train_m['fused']:.4f} iq={train_m['iq']:.4f} "
            f"cons={train_m['cons']:.4f} acc={train_m['acc']:.2f}% skipped={train_m['skipped']}"
        )
        common.print_metrics("Val", val_m, val_score)

        save_checkpoint(latest_path, model, optimizer, scheduler, epoch, best_score, best_metrics, args, model_seed)
        if val_score > best_score:
            best_score = val_score
            best_metrics = val_m
            stale = 0
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_score, best_metrics, args, model_seed)
            print(f"[*] New best saved: {best_path}")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"[*] Early stop seed {model_seed}: stale={stale}")
                break

    if os.path.exists(best_path):
        ckpt = common.safe_torch_load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    export_seed_cache(model, val_loader, test_loader, class_names, args, model_seed, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    os.makedirs(common.relpath("checkpoints"), exist_ok=True)
    os.makedirs(common.relpath("results"), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.disable_amp)

    print("=" * 120)
    print("Train CV-TRN-v2 auxiliary experts")
    print("=" * 120)
    print(f"device={device} | amp={use_amp} | split_seed={args.split_seed} | seeds={args.model_seeds}")

    full_dataset = common.build_full_dataset(args)
    train_idx, val_idx, test_idx = common.make_aligned_split(full_dataset, args.split_seed)
    if not args.skip_alignment_check:
        common.check_alignment(full_dataset, val_idx, test_idx, args.alignment_cache)
    class_names = getattr(full_dataset, "mod_classes", common.DEFAULT_MOD_CLASSES)

    train_loader = DataLoader(
        common.IQSubset(full_dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        common.IQSubset(full_dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        common.IQSubset(full_dataset, test_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    probe = build_model(args, device)
    print(f"Trainable params: {sum(p.numel() for p in probe.parameters() if p.requires_grad) / 1e6:.3f}M")
    del probe
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for model_seed in args.model_seeds:
        print("\n" + "=" * 120)
        print(f"Training CV-TRN-v2 seed {model_seed}")
        print("=" * 120)
        train_one_seed(args, model_seed, train_loader, val_loader, test_loader, class_names, device, use_amp)


if __name__ == "__main__":
    main()
