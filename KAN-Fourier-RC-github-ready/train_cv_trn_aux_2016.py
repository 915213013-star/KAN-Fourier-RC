import argparse
import math
import os
import random
import time
from contextlib import nullcontext

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataprocessnew4 import RML2016Dataset
from model_cv_trn_aux_2016 import build_cv_trn_aux_model


NUM_CLASSES = 11
DEFAULT_MOD_CLASSES = [
    "8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK",
    "PAM4", "QAM16", "QAM64", "QPSK", "WBFM",
]
TRANSITION_SNRS = np.array([-10, -8, -6, -4, -2], dtype=np.int32)
EDGE_LOW_SNRS = np.array([-18, -16], dtype=np.int32)


def project_root():
    return os.path.dirname(os.path.abspath(__file__))


def relpath(*parts):
    return os.path.join(project_root(), *parts)


def parse_args():
    p = argparse.ArgumentParser(
        description="Train a CV-TRN auxiliary branch for RML2016.10A. This branch is used only as post-training fusion expert."
    )
    p.add_argument("--split_seed", type=int, default=1)
    p.add_argument("--model_seed", type=int, default=1)
    p.add_argument("--epochs", type=int, default=180)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--eta_min", type=float, default=2e-5)
    p.add_argument("--grad_clip", type=float, default=2.0)
    p.add_argument("--patience", type=int, default=28)
    p.add_argument("--label_smoothing", type=float, default=0.02)
    p.add_argument("--disable_amp", action="store_true")
    p.add_argument("--force_restart", action="store_true")

    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--frame_len", type=int, default=16)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.10)

    p.add_argument("--rpo_prob", type=float, default=0.85)
    p.add_argument("--roll_prob", type=float, default=0.50)
    p.add_argument("--roll_max", type=int, default=8)
    p.add_argument("--noise_prob", type=float, default=0.10)
    p.add_argument("--noise_std", type=float, default=0.015)
    p.add_argument("--no_augment", action="store_true")

    p.add_argument("--data_path", type=str, default=relpath("raw_data", "RML2016.10a_dict.pkl"))
    p.add_argument("--cache_dir", type=str, default=relpath("feature_cache"))
    p.add_argument("--alignment_cache", type=str, default=relpath("results", "greedy_soup_identity_valtest_probs_for_gamc_fusion.npz"))
    p.add_argument("--skip_alignment_check", action="store_true")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def make_grad_scaler(use_amp):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def autocast_context(device_type, enabled):
    if device_type == "cuda" and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled)
    if device_type == "cuda":
        return torch.cuda.amp.autocast(enabled=enabled)
    return nullcontext()


def build_full_dataset(args):
    return RML2016Dataset(
        data_path=args.data_path,
        transform=True,
        return_snr=True,
        use_cache=True,
        cache_dir=args.cache_dir,
    )


def make_aligned_split(full_dataset, split_seed):
    labels = np.asarray(full_dataset.labels, dtype=np.int64)
    snrs = np.asarray(full_dataset.snrs, dtype=np.int32)
    composite = np.array([f"{int(y)}_{int(s)}" for y, s in zip(labels, snrs)])
    indices = np.arange(len(labels))
    train_idx, temp_idx, _, temp_targets = train_test_split(
        indices,
        composite,
        test_size=0.2,
        random_state=split_seed,
        stratify=composite,
    )
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx,
        temp_targets,
        test_size=0.5,
        random_state=split_seed,
        stratify=temp_targets,
    )

    candidate_paths = [
        relpath(f"test_indices_model_oracle_distill_split{split_seed}.npy"),
        relpath(f"test_indices_model_multitf_specialist_split{split_seed}.npy"),
        relpath(f"test_indices_model_snr_estimator_split{split_seed}.npy"),
        relpath(f"test_indices_model_transition_specialist_split{split_seed}.npy"),
        relpath(f"test_indices_model_complex_tcn_distill_split{split_seed}.npy"),
        relpath(f"test_indices_model_4stream_moe_joint_strat_rml2016_seed{split_seed}.npy"),
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            saved = np.load(path)
            if len(saved) == len(test_idx) and np.array_equal(np.sort(saved), np.sort(test_idx)):
                test_idx = saved.astype(np.int64)
                print(f"[*] Reusing aligned test indices: {path}")
                break
    return train_idx.astype(np.int64), val_idx.astype(np.int64), test_idx.astype(np.int64)


def check_alignment(full_dataset, val_idx, test_idx, cache_path):
    if not os.path.exists(cache_path):
        print(f"[!] Alignment cache not found, skipped: {cache_path}")
        return
    z = np.load(cache_path, allow_pickle=True)
    labels = np.asarray(full_dataset.labels, dtype=np.int64)
    snrs = np.asarray(full_dataset.snrs, dtype=np.int32)
    pairs = [
        ("labels_val", labels[val_idx], z["labels_val"].astype(np.int64)),
        ("snrs_val", snrs[val_idx], z["snrs_val"].astype(np.int32)),
        ("labels_test", labels[test_idx], z["labels_test"].astype(np.int64)),
        ("snrs_test", snrs[test_idx], z["snrs_test"].astype(np.int32)),
    ]
    for name, cur, ref in pairs:
        if len(cur) != len(ref) or not np.all(cur == ref):
            raise RuntimeError(f"CV-TRN split is not aligned with Fourier soup cache: {name}")
    print("[*] Alignment check passed against Fourier soup val/test cache.")


class IQSubset(Dataset):
    def __init__(self, full_dataset, indices):
        self.data = np.asarray(full_dataset.data, dtype=np.float32)
        self.labels = np.asarray(full_dataset.labels, dtype=np.int64)
        self.snrs = np.asarray(full_dataset.snrs, dtype=np.int32)
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        x = self.data[real_idx]
        if x.shape[0] >= 3:
            iq = x[[1, 2], :]
        else:
            iq = x[:2, :]
        return (
            torch.from_numpy(iq.astype(np.float32, copy=False)),
            torch.tensor(int(self.labels[real_idx]), dtype=torch.long),
            torch.tensor(int(self.snrs[real_idx]), dtype=torch.long),
        )


def random_rpo_roll_aug(x, args):
    if args.no_augment:
        return x
    y = x
    if torch.rand((), device=x.device) < args.roll_prob:
        shift = int(torch.randint(-args.roll_max, args.roll_max + 1, (), device=x.device).item())
        if shift:
            y = torch.roll(y, shifts=shift, dims=-1)
    if torch.rand((), device=x.device) < args.rpo_prob:
        y = y.clone()
        theta = torch.empty(y.size(0), device=y.device).uniform_(-math.pi, math.pi)
        c = torch.cos(theta).view(-1, 1)
        s = torch.sin(theta).view(-1, 1)
        i_part = y[:, 0, :].clone()
        q_part = y[:, 1, :].clone()
        y[:, 0, :] = i_part * c - q_part * s
        y[:, 1, :] = i_part * s + q_part * c
    if torch.rand((), device=x.device) < args.noise_prob:
        y = y + torch.randn_like(y) * float(args.noise_std)
    return y


def sample_weights(snrs):
    w = torch.ones_like(snrs, dtype=torch.float32)
    w = torch.where(snrs <= -16, torch.full_like(w, 1.30), w)
    w = torch.where((snrs >= -14) & (snrs <= -8), torch.full_like(w, 1.18), w)
    return w


def weighted_ce(logits, y, snrs, label_smoothing):
    loss = F.cross_entropy(logits, y, reduction="none", label_smoothing=label_smoothing)
    w = sample_weights(snrs).to(logits.device)
    return (loss * w).sum() / (w.sum() + 1e-8)


def normalize_probs(p):
    p = np.asarray(p, dtype=np.float32)
    p = np.clip(p, 1e-12, 1.0)
    return p / (p.sum(axis=1, keepdims=True) + 1e-12)


def metrics_from_probs(probs, labels, snrs):
    probs = normalize_probs(probs)
    labels = labels.astype(np.int64)
    snrs = snrs.astype(np.int32)
    pred = probs.argmax(axis=1).astype(np.int64)

    def acc(mask):
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() == 0:
            return 0.0
        return float((pred[mask] == labels[mask]).mean() * 100.0)

    by_snr = {}
    for s in sorted(np.unique(snrs).tolist()):
        by_snr[int(s)] = acc(snrs == s)
    return {
        "overall_acc": float((pred == labels).mean() * 100.0),
        "transition_acc": acc(np.isin(snrs, TRANSITION_SNRS)),
        "edge_low_acc": acc(np.isin(snrs, EDGE_LOW_SNRS)),
        "negative_acc": acc(snrs < 0),
        "high_acc": acc(snrs >= 0),
        "by_snr": by_snr,
        "pred": pred,
    }


def score_metrics(m):
    return 0.92 * m["overall_acc"] + 0.06 * m["transition_acc"] + 0.02 * m["edge_low_acc"]


def print_metrics(prefix, m, score=None):
    suffix = "" if score is None else f" | score={score:.4f}"
    print(
        f"{prefix:<26} overall={m['overall_acc']:.3f}% | trans={m['transition_acc']:.3f}% | "
        f"edge={m['edge_low_acc']:.3f}% | neg={m['negative_acc']:.3f}% | high={m['high_acc']:.3f}%{suffix}"
    )


def plot_accuracy_vs_snr(by_snr, save_path, title):
    xs = sorted(by_snr.keys())
    ys = [by_snr[x] for x in xs]
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.xticks(xs, rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


@torch.no_grad()
def collect_probs(model, loader, device):
    model.eval()
    probs, labels, snrs = [], [], []
    for x, y, s in loader:
        x = x.to(device, non_blocking=True).float()
        logits = model(x)
        probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32))
        labels.append(y.numpy().astype(np.int64))
        snrs.append(s.numpy().astype(np.int32))
    return normalize_probs(np.concatenate(probs)), np.concatenate(labels), np.concatenate(snrs)


def train_one_epoch(model, loader, optimizer, scaler, device, args, use_amp):
    model.train()
    total_loss, total, correct = 0.0, 0, 0
    skipped = 0
    for step, (x, y, snr) in enumerate(loader, 1):
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True)
        snr = snr.to(device, non_blocking=True)
        x = random_rpo_roll_aug(x, args)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device.type, use_amp):
            logits = model(x)
            loss = weighted_ce(logits.float(), y, snr, args.label_smoothing)

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
        correct += int((logits.argmax(dim=1) == y).sum().item())
        total += bsz
        if step % 120 == 0:
            print(f"    batch {step}/{len(loader)} loss={loss.item():.4f} acc={100.0 * correct / max(1, total):.2f}%")

    total = max(1, total)
    return {"loss": total_loss / total, "acc": 100.0 * correct / total, "skipped": skipped}


def build_model_from_args(args, device):
    return build_cv_trn_aux_model(
        device=device,
        num_classes=NUM_CLASSES,
        dim=args.dim,
        depth=args.depth,
        num_heads=args.heads,
        frame_len=args.frame_len,
        stride=args.stride,
        dropout=args.dropout,
    )


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_score, best_metrics, args):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": int(epoch),
            "best_score": float(best_score),
            "best_metrics": best_metrics,
            "model_type": "cv_trn_aux_2016",
            "args": vars(args),
        },
        path,
    )


def export_valtest_cache(model, val_loader, test_loader, class_names, args, device):
    val_prob, labels_val, snrs_val = collect_probs(model, val_loader, device)
    test_prob, labels_test, snrs_test = collect_probs(model, test_loader, device)
    val_m = metrics_from_probs(val_prob, labels_val, snrs_val)
    test_m = metrics_from_probs(test_prob, labels_test, snrs_test)
    print_metrics("CV-TRN Aux Val", val_m, score_metrics(val_m))
    print_metrics("CV-TRN Aux Test", test_m)

    os.makedirs(relpath("results"), exist_ok=True)
    suffix = f"cv_trn_aux_split{args.split_seed}"
    cache_path = relpath("results", f"{suffix}_valtest_probs_for_fusion.npz")
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
    print(f"[*] CV-TRN fusion cache saved: {cache_path}")

    pred_path = relpath("results", f"{suffix}_predictions.npz")
    np.savez_compressed(
        pred_path,
        labels=labels_test.astype(np.int64),
        snrs=snrs_test.astype(np.int32),
        pred=test_m["pred"].astype(np.int64),
        final_prob=test_prob.astype(np.float32),
        mod_classes=np.asarray(class_names),
    )
    print(f"[*] CV-TRN predictions saved: {pred_path}")

    curve_path = relpath("results", f"accuracy_vs_snr_{suffix}.png")
    plot_accuracy_vs_snr(test_m["by_snr"], curve_path, "CV-TRN auxiliary branch accuracy vs SNR")
    print(f"[*] SNR curve saved: {curve_path}")


def main():
    args = parse_args()
    set_seed(args.model_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.disable_amp)
    os.makedirs(relpath("checkpoints"), exist_ok=True)
    os.makedirs(relpath("results"), exist_ok=True)

    print("=" * 120)
    print("Train CV-TRN auxiliary branch for RML2016.10A")
    print("=" * 120)
    print(f"device={device} | amp={use_amp} | split_seed={args.split_seed} | model_seed={args.model_seed}")
    print("[*] This branch is independent. It will only export probabilities for later fusion.")

    full_dataset = build_full_dataset(args)
    train_idx, val_idx, test_idx = make_aligned_split(full_dataset, args.split_seed)
    if not args.skip_alignment_check:
        check_alignment(full_dataset, val_idx, test_idx, args.alignment_cache)

    class_names = getattr(full_dataset, "mod_classes", DEFAULT_MOD_CLASSES)
    print(f"Split: train={len(train_idx):,} | val={len(val_idx):,} | test={len(test_idx):,}")

    train_loader = DataLoader(
        IQSubset(full_dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        IQSubset(full_dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        IQSubset(full_dataset, test_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model_from_args(args, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params / 1e6:.3f}M")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
    scaler = make_grad_scaler(use_amp)

    suffix = f"cv_trn_aux_mseed{args.model_seed}_split{args.split_seed}"
    best_path = relpath("checkpoints", f"best_{suffix}.pth")
    latest_path = relpath("checkpoints", f"latest_{suffix}.pth")
    start_epoch = 1
    best_score = -1e9
    best_metrics = None
    stale = 0

    if os.path.exists(latest_path) and not args.force_restart:
        ckpt = safe_torch_load(latest_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_score = float(ckpt.get("best_score", best_score))
        best_metrics = ckpt.get("best_metrics", None)
        print(f"[*] Resuming from {latest_path}: start_epoch={start_epoch}, best_score={best_score:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, optimizer, scaler, device, args, use_amp)
        scheduler.step()
        val_prob, labels_val, snrs_val = collect_probs(model, val_loader, device)
        val_m = metrics_from_probs(val_prob, labels_val, snrs_val)
        val_score = score_metrics(val_m)

        print(
            f"\nEpoch {epoch:03d}/{args.epochs} done in {time.time() - t0:.1f}s | "
            f"train_loss={train_m['loss']:.4f} | train_acc={train_m['acc']:.2f}% | skipped={train_m['skipped']}"
        )
        print_metrics("Val", val_m, val_score)

        save_checkpoint(latest_path, model, optimizer, scheduler, epoch, best_score, best_metrics, args)
        if val_score > best_score:
            best_score = val_score
            best_metrics = val_m
            stale = 0
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_score, best_metrics, args)
            print(f"[*] New best saved: {best_path}")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"[*] Early stopping: stale={stale}")
                break

    if os.path.exists(best_path):
        ckpt = safe_torch_load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[*] Loaded best checkpoint for export: {best_path}")

    export_valtest_cache(model, val_loader, test_loader, class_names, args, device)


if __name__ == "__main__":
    main()
