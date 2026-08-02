"""10B wrapper for compressed Fourier/KAN main-model seeds.

This keeps the RML2016.10A compressed training implementation intact, but
switches the dataset adapter, class count, default paths, and output names to
RML2016.10B.
"""

from __future__ import annotations

import os

import train_cv_trn_aux_10b_common as common10b
import train_fourier_compressed_main_seeds_2016 as impl


impl.common = common10b
impl.fm.common = common10b
impl.NUM_CLASSES = common10b.NUM_CLASSES
impl.HOS_DIM = 20
impl.fm.NUM_CLASSES = common10b.NUM_CLASSES
impl.fm.HOS_DIM = 20


def checkpoint_paths_10b(args, seed):
    tag = str(getattr(args, "run_tag", "") or "").strip()
    tag = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in tag)
    tag_part = f"_{tag}" if tag else ""
    suffix = f"fourier_compressed_10b_{args.variant}{tag_part}_mseed{seed}_split{args.split_seed}"
    return (
        common10b.relpath("checkpoints", f"best_{suffix}.pth"),
        common10b.relpath("checkpoints", f"latest_{suffix}.pth"),
        common10b.relpath("results", f"{suffix}_valtest_probs_for_soup.npz"),
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

    os.makedirs(common10b.relpath("checkpoints"), exist_ok=True)
    os.makedirs(common10b.relpath("results"), exist_ok=True)

    device = impl.torch.device("cuda" if impl.torch.cuda.is_available() else "cpu")
    print("=" * 120)
    print("Train compressed RML2016.10B Fourier-KAN main-model seeds")
    print("=" * 120)
    print("Academic protocol:")
    print("  - Only original train split is used for training.")
    print("  - Validation selects checkpoints and compressed soup weights.")
    print("  - Test probabilities are exported for one-shot final reporting.")
    print(f"device={device} | variant={args.variant} | split_seed={args.split_seed} | seeds={args.model_seeds}")
    print("[10B wrapper] data_path=", args.data_path)
    print("[10B wrapper] cache_dir=", args.cache_dir)
    print("[10B wrapper] alignment_cache=", args.alignment_cache)

    full_dataset = common10b.build_full_dataset(args)
    train_idx, val_idx, test_idx = common10b.make_aligned_split(full_dataset, args.split_seed)
    if not args.skip_alignment_check:
        common10b.check_alignment(full_dataset, val_idx, test_idx, args.alignment_cache)
    class_names = getattr(full_dataset, "mod_classes", common10b.DEFAULT_MOD_CLASSES)
    print(f"Split: train={len(train_idx):,} | val={len(val_idx):,} | test={len(test_idx):,}")
    print(f"Classes ({len(class_names)}): {class_names}")

    probe = impl.build_model(args, device)
    print(f"Trainable params: {sum(p.numel() for p in probe.parameters() if p.requires_grad) / 1e6:.3f}M")
    del probe
    if device.type == "cuda":
        impl.torch.cuda.empty_cache()

    for seed in args.model_seeds:
        print("\n" + "=" * 120)
        print(f"Training compressed 10B {args.variant} seed {seed}")
        print("=" * 120)
        impl.train_seed(args, int(seed), full_dataset, train_idx, val_idx, test_idx, class_names, device)


if __name__ == "__main__":
    main()
