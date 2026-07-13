import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import train_cv_trn_aux_2016 as common
import train_fourier_main_oof_2016 as fm
from model_moe_attention_compressed import COMPRESSED_VARIANTS, build_compressed_model

try:
    import train_oracle_privileged_distill as stable_base
except Exception:
    stable_base = None


NUM_CLASSES = 11
HOS_DIM = 20


def parse_args():
    p = argparse.ArgumentParser(
        description="Train compressed Fourier-KAN main-model seeds for parameter/FLOP ablation."
    )
    p.add_argument("--variant", type=str, default="small", choices=sorted(COMPRESSED_VARIANTS))
    p.add_argument("--split_seed", type=int, default=1)
    p.add_argument("--model_seeds", type=int, nargs="+", default=[181, 182])
    p.add_argument("--epochs", type=int, default=260)
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
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--force_restart", action="store_true")
    p.add_argument("--data_path", type=str, default=common.relpath("raw_data", "RML2016.10a_dict.pkl"))
    p.add_argument("--cache_dir", type=str, default=common.relpath("feature_cache"))
    p.add_argument("--alignment_cache", type=str, default=common.relpath("results", "greedy_soup_identity_valtest_probs_for_gamc_fusion.npz"))
    p.add_argument("--skip_alignment_check", action="store_true")
    p.add_argument("--run_tag", type=str, default="")
    p.add_argument("--init_checkpoint", type=str, default="")
    return p.parse_args()


def build_model(args, device):
    model = build_compressed_model(args.variant, num_classes=NUM_CLASSES, hos_dim=HOS_DIM).to(device)
    if stable_base is not None:
        try:
            stable_base.patch_lieqkan_stability(model, name=f"fourier_compressed_{args.variant}")
        except Exception as e:
            print(f"[!] Stability patch skipped: {e}")
    return model


def soft_rank_penalty(model, keep_ratio=0.82):
    """Encourage compressible low-tail spectra without hard rank truncation."""
    keep_ratio = float(np.clip(keep_ratio, 0.05, 0.98))
    target_tokens = ("lie_encoder", "experts", "attention_fusion", "router", "hos_mlp")
    penalties = []
    for name, param in model.named_parameters():
        if not param.requires_grad or not name.endswith("weight"):
            continue
        if not any(tok in name for tok in target_tokens):
            continue
        if param.ndim < 2 or min(param.shape) < 8:
            continue
        matrix = param.float().reshape(param.shape[0], -1)
        if min(matrix.shape) < 8:
            continue
        svals = torch.linalg.svdvals(matrix)
        keep = int(max(1, min(svals.numel() - 1, round(svals.numel() * keep_ratio))))
        tail = svals[keep:]
        if tail.numel() > 0:
            penalties.append(tail.mean() / (svals.mean().detach() + 1e-6))
    if not penalties:
        return None
    return torch.stack(penalties).mean()


def rebuild_quat_from_iq(x, iq):
    i_part = iq[:, 0, :]
    q_part = iq[:, 1, :]
    amp = torch.sqrt(i_part.square() + q_part.square() + 1e-6)
    i_next = torch.roll(i_part, shifts=-1, dims=-1)
    q_next = torch.roll(q_part, shifts=-1, dims=-1)
    cross = i_part * q_next - q_part * i_next
    dot = i_part * i_next + q_part * q_next
    dphi = torch.atan2(cross, dot + 1e-6)
    dphi = torch.cat([dphi[:, :-1], dphi[:, -2:-1]], dim=-1)
    return torch.stack([amp, i_part, q_part, dphi], dim=1).to(dtype=x.dtype)


def augment_quat_input(x, args, scale=1.0):
    y = x
    scale = float(np.clip(scale, 0.0, 1.0))
    roll_prob = float(getattr(args, "roll_prob", 0.0)) * scale
    if roll_prob > 0.0 and torch.rand((), device=x.device) < roll_prob:
        roll_max = max(1, int(getattr(args, "roll_max", 6)))
        shift = int(torch.randint(-roll_max, roll_max + 1, (), device=x.device).item())
        if shift:
            y = torch.roll(y, shifts=shift, dims=-1)
    noise_prob = float(getattr(args, "iq_noise_prob", 0.0)) * scale
    if noise_prob > 0.0 and torch.rand((), device=x.device) < noise_prob:
        iq = y[:, 1:3, :] + torch.randn_like(y[:, 1:3, :]) * float(getattr(args, "iq_noise_std", 0.010)) * scale
        y = rebuild_quat_from_iq(y, iq)
    return y


def weighted_ce_loss(logits, target, snr, args, ce, scale=1.0):
    scale = float(np.clip(scale, 0.0, 1.0))
    neg_weight = 1.0 + scale * (float(getattr(args, "negative_snr_weight", 1.0)) - 1.0)
    high_weight = 1.0 + scale * (float(getattr(args, "high_snr_weight", 1.0)) - 1.0)
    edge_weight = 1.0 + scale * (float(getattr(args, "edge_snr_weight", 1.0)) - 1.0)
    transition_weight = 1.0 + scale * (float(getattr(args, "transition_snr_weight", 1.0)) - 1.0)
    if (
        abs(neg_weight - 1.0) < 1e-8
        and abs(high_weight - 1.0) < 1e-8
        and abs(edge_weight - 1.0) < 1e-8
        and abs(transition_weight - 1.0) < 1e-8
    ):
        return ce(logits, target)
    loss = F.cross_entropy(logits, target, reduction="none")
    weights = torch.ones_like(loss)
    weights = weights * torch.where(snr < 0, torch.full_like(weights, neg_weight), torch.ones_like(weights))
    weights = weights * torch.where(snr >= 0, torch.full_like(weights, high_weight), torch.ones_like(weights))
    weights = weights * torch.where(snr <= -16, torch.full_like(weights, edge_weight), torch.ones_like(weights))
    transition = (snr >= -10) & (snr <= -2)
    weights = weights * torch.where(transition, torch.full_like(weights, transition_weight), torch.ones_like(weights))
    return (loss * weights).sum() / (weights.sum() + 1e-8)


def train_one_epoch_elastic(model, loader, optimizer, scheduler, ce, supcon, device, args, epoch=1):
    model.train()
    total, correct, loss_sum, skipped = 0, 0, 0.0, 0
    rank_sum, rank_count = 0.0, 0
    rank_weight = float(getattr(args, "soft_rank_weight", 0.0))
    rank_every = max(1, int(getattr(args, "soft_rank_every", 12)))
    aug_warmup = max(0, int(getattr(args, "augment_warmup_epochs", 0)))
    snr_warmup = max(0, int(getattr(args, "snr_weight_warmup_epochs", 0)))
    aug_scale = 1.0 if aug_warmup <= 0 else min(1.0, float(epoch) / float(aug_warmup))
    snr_scale = 1.0 if snr_warmup <= 0 else min(1.0, float(epoch) / float(snr_warmup))
    for step, (data, hos, target, snr) in enumerate(loader, 1):
        data = data.to(device, non_blocking=True).float()
        hos = hos.to(device, non_blocking=True).float()
        target = target.to(device, non_blocking=True)
        snr = snr.to(device, non_blocking=True)
        data = augment_quat_input(data, args, scale=aug_scale)
        optimizer.zero_grad(set_to_none=True)
        feat, logits = model(data, hos)
        loss_ce = weighted_ce_loss(logits, target, snr, args, ce, scale=snr_scale)
        loss_con = supcon(feat, target, logits)
        loss = float(args.alpha_supcon) * loss_con + (1.0 - float(args.alpha_supcon)) * loss_ce
        rank_loss = None
        if rank_weight > 0.0 and (step % rank_every == 0):
            rank_loss = soft_rank_penalty(model, getattr(args, "soft_rank_keep_ratio", 0.82))
            if rank_loss is not None:
                loss = loss + rank_weight * rank_loss
                rank_sum += float(rank_loss.detach().cpu())
                rank_count += 1
        if not torch.isfinite(loss) or float(loss.detach().cpu()) > 5.0:
            skipped += 1
            continue
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        if not torch.isfinite(grad) or float(grad.detach().cpu()) > 100.0:
            optimizer.zero_grad(set_to_none=True)
            skipped += 1
            continue
        optimizer.step()
        scheduler.step()
        bsz = target.size(0)
        loss_sum += float(loss.detach().cpu()) * bsz
        correct += int((logits.argmax(dim=1) == target).sum().item())
        total += bsz
        if step % 200 == 0:
            lr = optimizer.param_groups[0]["lr"]
            rank_msg = f" rank={rank_sum / max(1, rank_count):.4f}" if rank_count else ""
            print(
                f"    batch {step}/{len(loader)} loss={loss.item():.4f} "
                f"lr={lr:.3e} acc={100.0 * correct / max(1, total):.2f}%{rank_msg}"
            )
    return {
        "loss": loss_sum / max(1, total),
        "acc": 100.0 * correct / max(1, total),
        "skipped": skipped,
        "rank": rank_sum / max(1, rank_count),
    }


def checkpoint_paths(args, seed):
    tag = str(getattr(args, "run_tag", "") or "").strip()
    tag = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in tag)
    tag_part = f"_{tag}" if tag else ""
    suffix = f"fourier_compressed_{args.variant}{tag_part}_mseed{seed}_split{args.split_seed}"
    return (
        common.relpath("checkpoints", f"best_{suffix}.pth"),
        common.relpath("checkpoints", f"latest_{suffix}.pth"),
        common.relpath("results", f"{suffix}_valtest_probs_for_soup.npz"),
    )


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_score, best_metrics, args, seed):
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_score": float(best_score),
            "best_metrics": best_metrics,
            "split_seed": int(args.split_seed),
            "model_seed": int(seed),
            "variant": str(args.variant),
            "model_type": "fourier_compressed_main_seed_2016",
            "args": vars(args),
        },
        path,
    )


def train_seed(args, seed, full_dataset, train_idx, val_idx, test_idx, class_names, device):
    common.set_seed(int(seed))
    train_loader = fm.make_loader(full_dataset, train_idx, args.batch_size, True, args.num_workers, device, drop_last=True)
    val_loader = fm.make_loader(full_dataset, val_idx, args.eval_batch_size, False, args.num_workers, device)
    test_loader = fm.make_loader(full_dataset, test_idx, args.eval_batch_size, False, args.num_workers, device)

    model = build_model(args, device)
    if str(getattr(args, "init_checkpoint", "") or "").strip():
        init_path = str(args.init_checkpoint)
        if not os.path.exists(init_path):
            raise FileNotFoundError(init_path)
        init_ckpt = common.safe_torch_load(init_path, map_location=device)
        state = init_ckpt["model_state_dict"] if isinstance(init_ckpt, dict) and "model_state_dict" in init_ckpt else init_ckpt
        model.load_state_dict(state, strict=True)
        print(f"[*] Initialized model from: {init_path}")
    ce = nn.CrossEntropyLoss().to(device)
    supcon = fm.EntropyAwareSupConLoss(args.temperature).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(args.epochs) * len(train_loader)),
        eta_min=float(args.eta_min),
    )
    best_path, latest_path, cache_path = checkpoint_paths(args, seed)
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
        print(f"[*] Resume compressed {args.variant} seed {seed}: start_epoch={start_epoch}, best_score={best_score:.4f}")

    for epoch in range(start_epoch, int(args.epochs) + 1):
        t0 = time.time()
        train_m = train_one_epoch_elastic(model, train_loader, optimizer, scheduler, ce, supcon, device, args, epoch=epoch)
        pv, yv, sv = fm.collect_probs(model, val_loader, device)
        val_m = fm.metrics_from_probs(pv, yv, sv)
        score = common.score_metrics(val_m)
        print(
            f"\nCompressed {args.variant} seed {seed} | Epoch {epoch:03d}/{args.epochs} | {time.time() - t0:.1f}s | "
            f"loss={train_m['loss']:.4f} acc={train_m['acc']:.2f}% rank={train_m['rank']:.4f} skipped={train_m['skipped']}"
        )
        common.print_metrics("Val", val_m, score)

        save_checkpoint(latest_path, model, optimizer, scheduler, epoch, best_score, best_metrics, args, seed)
        if score > best_score:
            best_score = score
            best_metrics = val_m
            stale = 0
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_score, best_metrics, args, seed)
            print(f"[*] New best saved: {best_path}")
        else:
            stale += 1
            if stale >= int(args.patience):
                print(f"[*] Early stop seed {seed}: stale={stale}")
                break

    if os.path.exists(best_path):
        ckpt = common.safe_torch_load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    val_prob, labels_val, snrs_val = fm.collect_probs(model, val_loader, device)
    test_prob, labels_test, snrs_test = fm.collect_probs(model, test_loader, device)
    val_m = fm.metrics_from_probs(val_prob, labels_val, snrs_val)
    common.print_metrics(f"Compressed {args.variant} seed {seed} Val", val_m, common.score_metrics(val_m))
    print("[*] Test probabilities exported for fixed downstream fusion; test labels are not scored here.")
    np.savez_compressed(
        cache_path,
        val_prob=val_prob.astype(np.float32),
        test_prob=test_prob.astype(np.float32),
        labels_val=labels_val.astype(np.int64),
        snrs_val=snrs_val.astype(np.int32),
        labels_test=labels_test.astype(np.int64),
        snrs_test=snrs_test.astype(np.int32),
        mod_classes=np.asarray(class_names),
        checkpoint=np.asarray([best_path]),
        variant=np.asarray([args.variant]),
    )
    print(f"[*] Compressed seed cache saved: {cache_path}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    os.makedirs(common.relpath("checkpoints"), exist_ok=True)
    os.makedirs(common.relpath("results"), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 120)
    print("Train compressed Fourier-KAN main-model seeds")
    print("=" * 120)
    print("Academic protocol:")
    print("  - Only original train split is used for training.")
    print("  - Validation selects checkpoints and compressed soup weights.")
    print("  - Test probabilities are exported for one-shot final reporting.")
    print(f"device={device} | variant={args.variant} | split_seed={args.split_seed} | seeds={args.model_seeds}")

    full_dataset = common.build_full_dataset(args)
    train_idx, val_idx, test_idx = common.make_aligned_split(full_dataset, args.split_seed)
    if not args.skip_alignment_check:
        common.check_alignment(full_dataset, val_idx, test_idx, args.alignment_cache)
    class_names = getattr(full_dataset, "mod_classes", common.DEFAULT_MOD_CLASSES)

    probe = build_model(args, device)
    print(f"Trainable params: {sum(p.numel() for p in probe.parameters() if p.requires_grad) / 1e6:.3f}M")
    del probe
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for seed in args.model_seeds:
        print("\n" + "=" * 120)
        print(f"Training compressed {args.variant} seed {seed}")
        print("=" * 120)
        train_seed(args, int(seed), full_dataset, train_idx, val_idx, test_idx, class_names, device)


if __name__ == "__main__":
    main()
