import argparse
import os

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

import train_cv_trn_aux_2016 as common
import train_fourier_main_oof_2016 as fm
import train_fourier_compressed_main_seeds_2016 as seed_train
from model_moe_attention_compressed import COMPRESSED_VARIANTS, build_compressed_model

try:
    import train_oracle_privileged_distill as stable_base
except Exception:
    stable_base = None


NUM_CLASSES = 11
HOS_DIM = 20


def parse_args():
    p = argparse.ArgumentParser(description="Train compressed Fourier-KAN main model in train-split OOF form.")
    p.add_argument("--variant", type=str, default="small", choices=sorted(COMPRESSED_VARIANTS))
    p.add_argument("--split_seed", type=int, default=1)
    p.add_argument("--model_seed", type=int, default=181)
    p.add_argument(
        "--fold_model_seeds",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional per-fold seed list. Each seed trains on the same fold-training subset; "
            "their predictions are ensembled for clean OOF holdout probabilities."
        ),
    )
    p.add_argument(
        "--fold_seed_map",
        type=str,
        nargs="*",
        default=[],
        help=(
            "Optional per-fold seed override, e.g. 1:261 2:269,271 3:261. "
            "Useful when one fold is stuck in a bad basin while other folds already have good checkpoints."
        ),
    )
    p.add_argument(
        "--fold_soup_mode",
        type=str,
        default="weight",
        choices=["weight", "prob"],
        help="How to combine multiple per-fold seeds: deploy-like weight soup or probability log-average.",
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
    p.add_argument(
        "--init_checkpoint",
        type=str,
        default="",
        help=(
            "Optional checkpoint used to initialize every fold before training. "
            "For strict OOF meta training, prefer --init_checkpoint_template with fold-specific checkpoints."
        ),
    )
    p.add_argument(
        "--init_checkpoint_template",
        type=str,
        default="",
        help=(
            "Optional fold-specific init checkpoint template, e.g. "
            "<checkpoint_dir>/best_fourier_compressed_oof_denoise_full_v2_bal_mseed231_fold{fold}_split1.pth. "
            "Available fields: {fold}, {seed}, {split_seed}, {variant}, {run_tag}."
        ),
    )
    p.add_argument(
        "--resume_best_reset",
        action="store_true",
        help="Load an existing best checkpoint but restart optimizer/scheduler from the requested LR.",
    )
    p.add_argument("--freeze_good_folds", action="store_true")
    p.add_argument("--freeze_min_overall", type=float, default=62.5)
    p.add_argument("--freeze_min_high", type=float, default=91.0)
    p.add_argument("--data_path", type=str, default=common.relpath("raw_data", "RML2016.10a_dict.pkl"))
    p.add_argument("--cache_dir", type=str, default=common.relpath("feature_cache"))
    p.add_argument("--alignment_cache", type=str, default=common.relpath("results", "greedy_soup_identity_valtest_probs_for_gamc_fusion.npz"))
    p.add_argument("--skip_alignment_check", action="store_true")
    p.add_argument("--output_cache", type=str, default="")
    return p.parse_args()


def build_model(args, device):
    model = build_compressed_model(args.variant, num_classes=NUM_CLASSES, hos_dim=HOS_DIM).to(device)
    if stable_base is not None:
        try:
            stable_base.patch_lieqkan_stability(model, name=f"fourier_compressed_oof_{args.variant}")
        except Exception as e:
            print(f"[!] Stability patch skipped: {e}")
    return model


def checkpoint_paths(args, fold):
    tag = str(getattr(args, "run_tag", "") or "").strip()
    tag = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in tag)
    tag_part = f"_{tag}" if tag else ""
    suffix = f"fourier_compressed_oof_{args.variant}{tag_part}_mseed{args.model_seed}_fold{fold}_split{args.split_seed}"
    return (
        common.relpath("checkpoints", f"best_{suffix}.pth"),
        common.relpath("checkpoints", f"latest_{suffix}.pth"),
    )


def resolve_init_checkpoint(args, fold):
    template = str(getattr(args, "init_checkpoint_template", "") or "").strip()
    if template:
        return template.format(
            fold=int(fold),
            seed=int(args.model_seed),
            split_seed=int(args.split_seed),
            variant=str(args.variant),
            run_tag=str(getattr(args, "run_tag", "") or ""),
        )
    return str(getattr(args, "init_checkpoint", "") or "").strip()


def build_initialized_model(args, fold, device):
    model = build_model(args, device)
    init_path = resolve_init_checkpoint(args, fold)
    if init_path:
        if not os.path.exists(init_path):
            raise FileNotFoundError(f"Init checkpoint not found for fold {fold}: {init_path}")
        ckpt = common.safe_torch_load(init_path, map_location=device)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state, strict=True)
        print(f"[*] Initialized compressed OOF fold {fold} from: {init_path}")
    return model


def normalize_weights(raw_weights, n):
    if raw_weights is None or len(raw_weights) == 0:
        weights = np.ones(n, dtype=np.float64)
    else:
        if len(raw_weights) != n:
            raise ValueError(f"--fold_soup_weights must have {n} values, got {len(raw_weights)}")
        weights = np.asarray(raw_weights, dtype=np.float64)
    if not np.all(np.isfinite(weights)) or float(weights.sum()) <= 0:
        raise ValueError("Fold soup weights must be finite and sum to a positive value.")
    return (weights / weights.sum()).astype(np.float64)


def parse_fold_seed_map(items):
    mapping = {}
    for item in items or []:
        text = str(item).strip()
        if not text:
            continue
        if ":" not in text:
            raise ValueError(f"Invalid --fold_seed_map item {text!r}; expected FOLD:SEED[,SEED...]")
        fold_text, seed_text = text.split(":", 1)
        fold = int(fold_text.strip())
        seeds = [int(x.strip()) for x in seed_text.split(",") if x.strip()]
        if fold <= 0 or not seeds:
            raise ValueError(f"Invalid --fold_seed_map item {text!r}")
        mapping[fold] = seeds
    return mapping


def checkpoint_meets_freeze(path, args):
    if not os.path.exists(path):
        return False
    try:
        ckpt = common.safe_torch_load(path, map_location="cpu")
    except Exception as exc:
        print(f"    [!] Could not inspect checkpoint for freeze: {path} ({exc})")
        return False
    metrics = ckpt.get("best_metrics", {}) if isinstance(ckpt, dict) else {}
    overall = float(metrics.get("overall_acc", -1.0))
    high = float(metrics.get("high_acc", -1.0))
    return overall >= float(args.freeze_min_overall) and high >= float(args.freeze_min_high)


def load_checkpoint_model(args, fold, device):
    best_path, _latest_path = checkpoint_paths(args, fold)
    ckpt = common.safe_torch_load(best_path, map_location=device)
    model = build_model(args, device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    best_score = float(ckpt.get("best_score", -1e9)) if isinstance(ckpt, dict) else -1e9
    return model, best_score


def average_checkpoint_states(paths, weights, device):
    avg = None
    for path, weight in zip(paths, weights):
        ckpt = common.safe_torch_load(path, map_location=device)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        if avg is None:
            avg = {k: v.detach().clone().float() * float(weight) for k, v in state.items()}
        else:
            if set(avg) != set(state):
                raise RuntimeError(f"Checkpoint keys differ for fold soup: {path}")
            for k, v in state.items():
                avg[k].add_(v.detach().float(), alpha=float(weight))
    return avg


def train_fold(args, fold, train_loader, holdout_loader, device):
    original_build_model = fm.build_model
    original_checkpoint_paths = fm.checkpoint_paths
    original_train_one_epoch = fm.train_one_epoch
    try:
        fm.build_model = lambda dev: build_initialized_model(args, fold, dev)
        fm.checkpoint_paths = lambda a, f: checkpoint_paths(args, f)
        fm.train_one_epoch = lambda model, loader, optimizer, scheduler, ce, supcon, dev, a: seed_train.train_one_epoch_elastic(
            model, loader, optimizer, scheduler, ce, supcon, dev, a, epoch=getattr(a, "_current_epoch", 1)
        )
        model, best_score = fm.train_fold(args, fold, train_loader, holdout_loader, device)
    finally:
        fm.build_model = original_build_model
        fm.checkpoint_paths = original_checkpoint_paths
        fm.train_one_epoch = original_train_one_epoch
    return model, best_score


def main():
    args = parse_args()
    if not args.output_cache:
        args.output_cache = common.relpath(
            "results",
            f"fourier_compressed_oof_{args.variant}_mseed{args.model_seed}_f{args.folds}e{args.epochs}_split{args.split_seed}_trainvaltest_probs_for_meta.npz",
        )
    os.makedirs(common.relpath("checkpoints"), exist_ok=True)
    os.makedirs(common.relpath("results"), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common.set_seed(args.model_seed)
    print("=" * 120)
    print("Train compressed Fourier-KAN OOF cache")
    print("=" * 120)
    print("Academic protocol:")
    print("  - OOF train probabilities are generated by models that never saw that fold.")
    print("  - Validation selects fold checkpoints; test probabilities are exported without scoring.")
    base_model_seed = int(args.model_seed)
    default_fold_model_seeds = list(args.fold_model_seeds) if args.fold_model_seeds else [base_model_seed]
    fold_seed_map = parse_fold_seed_map(args.fold_seed_map)
    default_fold_soup_weights = normalize_weights(args.fold_soup_weights, len(default_fold_model_seeds))
    print(
        f"device={device} | variant={args.variant} | split_seed={args.split_seed} | "
        f"model_seed={base_model_seed} | folds={args.folds}"
    )
    if len(default_fold_model_seeds) > 1:
        print(
            f"default per-fold seeds={default_fold_model_seeds} | fold_soup_mode={args.fold_soup_mode} | "
            f"weights={default_fold_soup_weights.tolist()}"
        )
    if fold_seed_map:
        print(f"fold seed map={fold_seed_map}")
    if args.freeze_good_folds:
        print(
            f"freeze_good_folds=True | min_overall={args.freeze_min_overall:.3f}% "
            f"min_high={args.freeze_min_high:.3f}%"
        )

    full_dataset = common.build_full_dataset(args)
    train_idx, val_idx, test_idx = common.make_aligned_split(full_dataset, args.split_seed)
    if not args.skip_alignment_check:
        common.check_alignment(full_dataset, val_idx, test_idx, args.alignment_cache)
    class_names = getattr(full_dataset, "mod_classes", common.DEFAULT_MOD_CLASSES)
    print(f"Split: train={len(train_idx):,} | val={len(val_idx):,} | test={len(test_idx):,}")

    val_loader = fm.make_loader(full_dataset, val_idx, args.eval_batch_size, False, args.num_workers, device)
    test_loader = fm.make_loader(full_dataset, test_idx, args.eval_batch_size, False, args.num_workers, device)
    labels_all = np.asarray(full_dataset.labels, dtype=np.int64)
    snrs_all = np.asarray(full_dataset.snrs, dtype=np.int32)

    train_oof = np.zeros((len(train_idx), NUM_CLASSES), dtype=np.float32)
    y_train = labels_all[train_idx].astype(np.int64)
    s_train = snrs_all[train_idx].astype(np.int32)
    val_probs, test_probs = [], []

    probe = build_model(args, device)
    print(f"Trainable params: {sum(p.numel() for p in probe.parameters() if p.requires_grad) / 1e6:.3f}M")
    del probe
    if device.type == "cuda":
        torch.cuda.empty_cache()

    skf = StratifiedKFold(n_splits=int(args.folds), shuffle=True, random_state=base_model_seed)
    composite = np.asarray([f"{int(y)}_{int(s)}" for y, s in zip(y_train, s_train)])
    for fold, (tr, va) in enumerate(skf.split(train_idx, composite), 1):
        fold_train_idx = train_idx[tr]
        holdout_idx = train_idx[va]
        print("\n" + "-" * 120)
        print(f"Compressed {args.variant} OOF fold {fold}/{args.folds} | train={len(fold_train_idx):,} | holdout={len(holdout_idx):,}")
        train_loader = fm.make_loader(full_dataset, fold_train_idx, args.batch_size, True, args.num_workers, device, drop_last=True)
        holdout_loader = fm.make_loader(full_dataset, holdout_idx, args.eval_batch_size, False, args.num_workers, device)
        seed_holdout_probs, seed_val_probs, seed_test_probs = [], [], []
        yh = sh = None
        best_scores = []
        best_paths = []
        fold_model_seeds = list(fold_seed_map.get(fold, default_fold_model_seeds))
        fold_soup_weights = normalize_weights(
            args.fold_soup_weights if (args.fold_soup_weights and len(args.fold_soup_weights) == len(fold_model_seeds)) else None,
            len(fold_model_seeds),
        )
        if fold_seed_map:
            print(
                f"    fold seeds={fold_model_seeds} | fold_soup_mode={args.fold_soup_mode} | "
                f"weights={fold_soup_weights.tolist()}"
            )
        for seed in fold_model_seeds:
            args.model_seed = int(seed)
            print(f"    [*] Fold {fold} seed {seed}")
            best_path, _latest_path = checkpoint_paths(args, fold)
            if args.freeze_good_folds and not args.force_restart and checkpoint_meets_freeze(best_path, args):
                print(f"    [*] Reusing good checkpoint without training: {best_path}")
                model, best_score = load_checkpoint_model(args, fold, device)
            else:
                model, best_score = train_fold(args, fold, train_loader, holdout_loader, device)
            best_scores.append(float(best_score))
            best_path, _latest_path = checkpoint_paths(args, fold)
            best_paths.append(best_path)
            if args.fold_soup_mode == "prob" or len(fold_model_seeds) == 1:
                ph_i, yh_i, sh_i = fm.collect_probs(model, holdout_loader, device)
                pv_i, yv_i, sv_i = fm.collect_probs(model, val_loader, device)
                pt_i, yt_i, st_i = fm.collect_probs(model, test_loader, device)
                seed_holdout_probs.append(ph_i)
                seed_val_probs.append(pv_i)
                seed_test_probs.append(pt_i)
                yh, sh = yh_i, sh_i
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if args.fold_soup_mode == "weight" and len(fold_model_seeds) > 1:
            args.model_seed = int(fold_model_seeds[0])
            soup_state = average_checkpoint_states(best_paths, fold_soup_weights, device)
            model = build_model(args, device)
            model.load_state_dict(soup_state, strict=True)
            ph, yh, sh = fm.collect_probs(model, holdout_loader, device)
            pv, yv, sv = fm.collect_probs(model, val_loader, device)
            pt, yt, st = fm.collect_probs(model, test_loader, device)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            ph = fm.log_average(seed_holdout_probs)
            pv = fm.log_average(seed_val_probs)
            pt = fm.log_average(seed_test_probs)
            yv = labels_all[val_idx].astype(np.int64)
            sv = snrs_all[val_idx].astype(np.int32)
            yt = labels_all[test_idx].astype(np.int64)
            st = snrs_all[test_idx].astype(np.int32)

        args.model_seed = base_model_seed
        train_oof[va] = ph
        val_probs.append(pv)
        test_probs.append(pt)
        common.print_metrics(f"Fold {fold} Holdout", common.metrics_from_probs(ph, yh, sh), max(best_scores))
        common.print_metrics(f"Fold {fold} Val", common.metrics_from_probs(pv, yv, sv), common.score_metrics(common.metrics_from_probs(pv, yv, sv)))

    val_prob = fm.log_average(val_probs)
    test_prob = fm.log_average(test_probs)
    train_m = common.metrics_from_probs(train_oof, y_train, s_train)
    val_m = common.metrics_from_probs(val_prob, labels_all[val_idx], snrs_all[val_idx])
    common.print_metrics("Compressed Fourier OOF Train", train_m, common.score_metrics(train_m))
    common.print_metrics("Compressed Fourier fold-soup Val", val_m, common.score_metrics(val_m))
    print("[*] Test probabilities exported for fixed downstream fusion; test labels are not scored here.")

    np.savez_compressed(
        args.output_cache,
        train_prob=train_oof.astype(np.float32),
        val_prob=val_prob.astype(np.float32),
        test_prob=test_prob.astype(np.float32),
        labels_train=y_train.astype(np.int64),
        snrs_train=s_train.astype(np.int32),
        labels_val=labels_all[val_idx].astype(np.int64),
        snrs_val=snrs_all[val_idx].astype(np.int32),
        labels_test=labels_all[test_idx].astype(np.int64),
        snrs_test=snrs_all[test_idx].astype(np.int32),
        mod_classes=np.asarray(class_names),
        variant=np.asarray([args.variant]),
        fold_model_seeds=np.asarray(fold_model_seeds, dtype=np.int64),
        fold_soup_mode=np.asarray([args.fold_soup_mode]),
        fold_soup_weights=fold_soup_weights.astype(np.float32),
        protocol=np.asarray(["train-split OOF compressed Fourier-KAN main model; test probabilities exported without scoring"]),
    )
    print(f"[*] Compressed Fourier OOF cache saved: {args.output_cache}")


if __name__ == "__main__":
    main()
