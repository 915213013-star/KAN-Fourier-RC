import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

import train_cv_trn_aux_2016 as common
import oof_protocol as oofp
from model_moe_attention import LieQKAN, MoEFusedClassifier

try:
    import model_stability as stable_base
except Exception:
    stable_base = None


NUM_CLASSES = 11
HOS_DIM = 20


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Train original Fourier/KAN MoE main model in train-split OOF form. "
            "Exports OOF train probabilities so downstream routers can learn Fourier error patterns cleanly."
        )
    )
    p.add_argument("--split_seed", type=int, default=1)
    p.add_argument("--model_seed", type=int, default=77)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=220)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--eval_batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--eta_min", type=float, default=4e-6)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--alpha_supcon", type=float, default=0.30)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--patience", type=int, default=42)
    p.add_argument("--force_restart", action="store_true")
    p.add_argument("--data_path", type=str, default=common.relpath("raw_data", "RML2016.10a_dict.pkl"))
    p.add_argument("--cache_dir", type=str, default=common.relpath("feature_cache"))
    p.add_argument("--alignment_cache", type=str, default=common.relpath("results", "greedy_soup_identity_valtest_probs_for_gamc_fusion.npz"))
    p.add_argument("--skip_alignment_check", action="store_true")
    p.add_argument("--output_cache", type=str, default="")
    oofp.add_protocol_args(p)
    return p.parse_args()


class MoESubset(Dataset):
    def __init__(self, full_dataset, indices):
        self.full_dataset = full_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(full_dataset.labels, dtype=np.int64)
        self.snrs = np.asarray(full_dataset.snrs, dtype=np.int32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        real_idx = int(self.indices[i])
        item = self.full_dataset[real_idx]
        if isinstance(item, (tuple, list)) and len(item) == 4:
            data, hos, label, snr = item
        elif isinstance(item, (tuple, list)) and len(item) == 3:
            data, hos, label = item
            snr = self.snrs[real_idx]
        else:
            raise ValueError("Unexpected RML dataset item format.")
        return data, hos, torch.tensor(int(label), dtype=torch.long), torch.tensor(int(snr), dtype=torch.long)


class EntropyAwareSupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, features, labels, logits):
        device = features.device
        batch_size = features.shape[0]
        probs = torch.softmax(logits.detach(), dim=1)
        confidence, _ = probs.max(dim=1)
        sim = torch.matmul(features, features.T) / self.temperature
        sim = sim - sim.max(dim=1, keepdim=True)[0].detach()
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.ones_like(mask)
        logits_mask.scatter_(1, torch.arange(batch_size, device=device).view(-1, 1), 0)
        mask = mask * logits_mask
        exp_logits = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-6)
        denom = mask.sum(dim=1)
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / denom
        return (-(mean_log_prob_pos * confidence)).mean()


def set_seed(seed):
    common.set_seed(int(seed))


def build_model(device):
    lie_model = LieQKAN(out_dim=128)
    model = MoEFusedClassifier(
        lie_model=lie_model,
        num_classes=NUM_CLASSES,
        hos_dim=HOS_DIM,
        num_experts=3,
    ).to(device)
    if stable_base is not None:
        try:
            stable_base.patch_lieqkan_stability(model, name="fourier_oof")
        except Exception as e:
            print(f"[!] Stability patch skipped: {e}")
    return model


def make_loader(full_dataset, indices, batch_size, shuffle, workers, device, drop_last=False):
    return DataLoader(
        MoESubset(full_dataset, indices),
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(workers),
        pin_memory=(device.type == "cuda"),
        drop_last=drop_last,
    )


def checkpoint_paths(args, fold):
    suffix = f"fourier_main_oof_{oofp.PROTOCOL_TAG}_mseed{args.model_seed}_fold{fold}_split{args.split_seed}"
    return (
        common.relpath("checkpoints", f"best_{suffix}.pth"),
        common.relpath("checkpoints", f"latest_{suffix}.pth"),
    )


def selection_checkpoint_paths(args, fold):
    best_path, latest_path = checkpoint_paths(args, fold)
    return (
        os.path.join(os.path.dirname(best_path), "select_" + os.path.basename(best_path)),
        os.path.join(os.path.dirname(latest_path), "select_" + os.path.basename(latest_path)),
    )


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_score,
    best_metrics,
    args,
    fold,
    protocol_metadata,
    **extra,
):
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "best_score": float(best_score),
        "best_metrics": best_metrics,
        "split_seed": int(args.split_seed),
        "model_seed": int(args.model_seed),
        "fold": int(fold),
        "model_type": str(getattr(args, "_model_type", "fourier_main_oof_2016")),
        "args": vars(args),
        "protocol_metadata": protocol_metadata,
    }
    payload.update(extra)
    torch.save(payload, path)


def metrics_from_probs(probs, labels, snrs):
    return common.metrics_from_probs(probs, labels.astype(np.int64), snrs.astype(np.int32))


@torch.no_grad()
def collect_probs(model, loader, device):
    model.eval()
    probs, labels, snrs = [], [], []
    for data, hos, target, snr in loader:
        data = data.to(device, non_blocking=True).float()
        hos = hos.to(device, non_blocking=True).float()
        _feat, logits = model(data, hos)
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy().astype(np.float32))
        labels.append(target.numpy().astype(np.int64))
        snrs.append(snr.numpy().astype(np.int32))
    p = np.concatenate(probs, axis=0)
    p = p / (p.sum(axis=1, keepdims=True) + 1e-12)
    return p.astype(np.float32), np.concatenate(labels), np.concatenate(snrs)


def train_one_epoch(model, loader, optimizer, scheduler, ce, supcon, device, args):
    model.train()
    total, correct, loss_sum, skipped = 0, 0, 0.0, 0
    for step, (data, hos, target, _snr) in enumerate(loader, 1):
        data = data.to(device, non_blocking=True).float()
        hos = hos.to(device, non_blocking=True).float()
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        feat, logits = model(data, hos)
        loss_ce = ce(logits, target)
        loss_con = supcon(feat, target, logits)
        loss = float(args.alpha_supcon) * loss_con + (1.0 - float(args.alpha_supcon)) * loss_ce
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
            print(f"    batch {step}/{len(loader)} loss={loss.item():.4f} lr={lr:.3e} acc={100.0 * correct / max(1, total):.2f}%")
    return {"loss": loss_sum / max(1, total), "acc": 100.0 * correct / max(1, total), "skipped": skipped}


def log_average(prob_list):
    logs = [np.log(np.clip(p, 1e-12, 1.0)) for p in prob_list]
    z = np.mean(np.stack(logs, axis=0), axis=0)
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / (e.sum(axis=1, keepdims=True) + 1e-12)).astype(np.float32)


def _new_training_state(args, device, train_loader, epochs):
    model = build_model(device)
    optimizer = optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(epochs) * len(train_loader)),
        eta_min=float(args.eta_min),
    )
    return model, optimizer, scheduler


def _load_matching_checkpoint(path, expected_metadata, device):
    if not os.path.exists(path):
        return None
    checkpoint = common.safe_torch_load(path, map_location=device)
    if not oofp.checkpoint_matches(checkpoint, expected_metadata):
        print(f"[!] Ignoring incompatible or legacy checkpoint: {path}")
        return None
    return checkpoint


def train_fold(
    args,
    fold,
    full_dataset,
    outer_train_idx,
    outer_holdout_idx,
    device,
    policy_validation_idx=None,
    official_test_idx=None,
):
    """Fit on the outer-train rows and select the fold checkpoint on its holdout."""
    oofp.assert_fold_partition(
        outer_train_idx,
        outer_holdout_idx,
        policy_validation_idx,
        official_test_idx,
    )
    print(
        f"[*] Fold {fold}: gradient-train={len(outer_train_idx):,}, "
        f"checkpoint/OOF-holdout={len(outer_holdout_idx):,}"
    )
    train_loader = make_loader(
        full_dataset,
        outer_train_idx,
        args.batch_size,
        True,
        args.num_workers,
        device,
        drop_last=True,
    )
    holdout_loader = make_loader(
        full_dataset,
        outer_holdout_idx,
        args.eval_batch_size,
        False,
        args.num_workers,
        device,
    )
    fold_meta = oofp.protocol_metadata(
        args,
        fold,
        "fold_training",
        outer_train_idx,
        outer_holdout_idx,
        target_epochs=int(args.epochs),
        policy_validation_indices=policy_validation_idx,
        official_test_indices=official_test_idx,
    )
    best_path, latest_path = checkpoint_paths(args, fold)
    set_seed(int(args.model_seed) + 1000 * int(fold))
    model, optimizer, scheduler = _new_training_state(args, device, train_loader, args.epochs)
    ce = nn.CrossEntropyLoss().to(device)
    supcon = EntropyAwareSupConLoss(args.temperature).to(device)
    start_epoch, best_score, best_metrics, stale = 1, -1e9, None, 0

    if not args.force_restart:
        latest = _load_matching_checkpoint(latest_path, fold_meta, device)
        if latest is not None:
            best_score = float(latest.get("best_score", best_score))
            best_metrics = latest.get("best_metrics")
            stale = int(latest.get("stale", 0))
            if bool(latest.get("training_complete", False)):
                selected = _load_matching_checkpoint(best_path, fold_meta, device)
                if selected is None:
                    raise RuntimeError(f"Completed fold {fold} has no matching selected checkpoint.")
                model.load_state_dict(selected["model_state_dict"])
                selected_epoch = int(selected["epoch"])
                print(f"[*] Reusing selected fold checkpoint {fold}: epoch={selected_epoch}")
                return model, best_score, selected_epoch, fold_meta
            model.load_state_dict(latest["model_state_dict"])
            optimizer.load_state_dict(latest["optimizer_state_dict"])
            scheduler.load_state_dict(latest["scheduler_state_dict"])
            start_epoch = int(latest.get("epoch", 0)) + 1
            print(f"[*] Resume fold {fold}: start_epoch={start_epoch}")

    for epoch in range(start_epoch, int(args.epochs) + 1):
        t0 = time.time()
        setattr(args, "_current_epoch", int(epoch))
        train_m = train_one_epoch(model, train_loader, optimizer, scheduler, ce, supcon, device, args)
        prob, labels, snrs = collect_probs(model, holdout_loader, device)
        metrics = metrics_from_probs(prob, labels, snrs)
        score = common.score_metrics(metrics)
        print(
            f"\nFourier fold {fold} | Epoch {epoch:03d}/{args.epochs} | "
            f"{time.time() - t0:.1f}s | loss={train_m['loss']:.4f} "
            f"acc={train_m['acc']:.2f}% skipped={train_m['skipped']}"
        )
        common.print_metrics("Fold checkpoint score", metrics, score)
        if score > best_score:
            best_score, best_metrics, stale = score, metrics, 0
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_score,
                best_metrics,
                args,
                fold,
                fold_meta,
                training_complete=False,
            )
            print(f"[*] New fold checkpoint saved: {best_path}")
        else:
            stale += 1
        save_checkpoint(
            latest_path,
            model,
            optimizer,
            scheduler,
            epoch,
            best_score,
            best_metrics,
            args,
            fold,
            fold_meta,
            stale=stale,
            training_complete=False,
        )
        if stale >= int(args.patience):
            print(f"[*] Fold early stop {fold}: stale={stale}")
            break

    selected = _load_matching_checkpoint(best_path, fold_meta, device)
    if selected is None:
        raise RuntimeError(f"No matching selected checkpoint was produced for fold {fold}.")
    model.load_state_dict(selected["model_state_dict"])
    selected_epoch = int(selected["epoch"])
    best_score = float(selected["best_score"])
    best_metrics = selected.get("best_metrics")
    save_checkpoint(
        latest_path,
        model,
        optimizer,
        scheduler,
        selected_epoch,
        best_score,
        best_metrics,
        args,
        fold,
        fold_meta,
        stale=stale,
        training_complete=True,
        selected_epoch=selected_epoch,
    )
    return model, best_score, selected_epoch, fold_meta


def main():
    args = parse_args()
    if not args.output_cache:
        args.output_cache = common.relpath(
            "results",
            f"fourier_main_oof_{oofp.PROTOCOL_TAG}_mseed{args.model_seed}_f{args.folds}e{args.epochs}_split{args.split_seed}_trainvaltest_probs_for_meta.npz",
        )
    os.makedirs(common.relpath("checkpoints"), exist_ok=True)
    os.makedirs(common.relpath("results"), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 120)
    print("Train Fourier main model OOF cache")
    print("=" * 120)
    print(f"OOF protocol: {oofp.PROTOCOL_ID}")
    print("  - Only the original train split is cross-fitted.")
    print("  - Each fold holdout is excluded from gradient updates.")
    print("  - The fold holdout supplies that fold's checkpoint/early-stopping score and OOF rows.")
    print("  - The independent validation split selects correction actions and thresholds only.")
    print("  - Official test labels are excluded from model fitting and policy selection.")
    print("  - This is leakage-controlled fold training; it is not a nested-CV claim.")
    print(f"device={device} | split_seed={args.split_seed} | model_seed={args.model_seed} | folds={args.folds}")

    full_dataset = common.build_full_dataset(args)
    train_idx, val_idx, test_idx = common.make_aligned_split(full_dataset, args.split_seed)
    if not args.skip_alignment_check:
        common.check_alignment(full_dataset, val_idx, test_idx, args.alignment_cache)
    labels_all = np.asarray(full_dataset.labels, dtype=np.int64)
    snrs_all = np.asarray(full_dataset.snrs, dtype=np.int32)
    labels_train = labels_all[train_idx]
    snrs_train = snrs_all[train_idx]
    composite = np.asarray([f"{int(y)}_{int(s)}" for y, s in zip(labels_train, snrs_train)])

    val_loader = make_loader(full_dataset, val_idx, args.eval_batch_size, False, args.num_workers, device)
    test_loader = make_loader(full_dataset, test_idx, args.eval_batch_size, False, args.num_workers, device)
    skf = StratifiedKFold(n_splits=int(args.folds), shuffle=True, random_state=int(args.model_seed))
    train_oof = np.zeros((len(train_idx), NUM_CLASSES), dtype=np.float32)
    val_probs, test_probs = [], []
    fold_protocol_records, selected_epochs = [], []
    labels_val = snrs_val = labels_test = snrs_test = None

    probe = build_model(device)
    print(f"Trainable params: {sum(p.numel() for p in probe.parameters() if p.requires_grad) / 1e6:.3f}M")
    del probe
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for fold, (tr, va) in enumerate(skf.split(train_idx, composite), 1):
        print("\n" + "=" * 120)
        print(f"Fourier main OOF fold {fold}/{args.folds}")
        print("=" * 120)
        fold_train_idx = train_idx[tr]
        holdout_idx = train_idx[va]
        model, selection_score, selected_epoch, fold_meta = train_fold(
            args,
            fold,
            full_dataset,
            fold_train_idx,
            holdout_idx,
            device,
            policy_validation_idx=val_idx,
            official_test_idx=test_idx,
        )
        holdout_loader = make_loader(
            full_dataset, holdout_idx, args.eval_batch_size, False, args.num_workers, device
        )
        ph, yh, sh = collect_probs(model, holdout_loader, device)
        train_oof[va] = ph
        pv, yv, sv = collect_probs(model, val_loader, device)
        pt, yt, st = collect_probs(model, test_loader, device)
        holdout_metrics = metrics_from_probs(ph, yh, sh)
        val_metrics = metrics_from_probs(pv, yv, sv)
        common.print_metrics(
            f"Fold {fold} selected-checkpoint OOF holdout",
            holdout_metrics,
            common.score_metrics(holdout_metrics),
        )
        common.print_metrics(f"Fold {fold} validation export", val_metrics, common.score_metrics(val_metrics))
        val_probs.append(pv)
        test_probs.append(pt)
        selected_epochs.append(int(selected_epoch))
        fold_protocol_records.append(
            {
                "fold": int(fold),
                "selected_epoch": int(selected_epoch),
                "fold_selection_score": float(selection_score),
                "checkpoint": fold_meta,
            }
        )
        labels_val, snrs_val, labels_test, snrs_test = yv, sv, yt, st
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    val_prob = log_average(val_probs)
    test_prob = log_average(test_probs)
    train_m = metrics_from_probs(train_oof, labels_train, snrs_train)
    val_m = metrics_from_probs(val_prob, labels_val, snrs_val)
    common.print_metrics("Fourier OOF Train", train_m, common.score_metrics(train_m))
    common.print_metrics("Fourier fold-soup Val", val_m, common.score_metrics(val_m))

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
        labels_train=labels_train.astype(np.int64),
        snrs_train=snrs_train.astype(np.int32),
        labels_val=labels_val.astype(np.int64),
        snrs_val=snrs_val.astype(np.int32),
        labels_test=labels_test.astype(np.int64),
        snrs_test=snrs_test.astype(np.int32),
        mod_classes=np.asarray(getattr(full_dataset, "mod_classes", common.DEFAULT_MOD_CLASSES)),
        protocol_id=np.asarray([oofp.PROTOCOL_ID]),
        config_fingerprint=np.asarray([oofp.config_fingerprint(args)]),
        selected_epochs=np.asarray(selected_epochs, dtype=np.int32),
        protocol_metadata_json=np.asarray(
            [oofp.metadata_json(oofp.cache_protocol_summary(args, fold_protocol_records))]
        ),
        protocol=np.asarray(
            [
                "leakage-controlled train-split OOF; fold holdout excluded from gradients and used "
                "for fold checkpoint selection plus OOF export; independent validation selects policy; "
                "official test used only after freezing"
            ]
        ),
    )
    print(f"[*] Fourier OOF cache saved: {args.output_cache}")


if __name__ == "__main__":
    main()
