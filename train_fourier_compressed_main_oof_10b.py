"""10B wrapper for train-split OOF compressed Fourier/KAN probabilities."""

from __future__ import annotations

import os

import train_cv_trn_aux_10b_common as common10b
import train_fourier_compressed_main_oof_2016 as impl


impl.common = common10b
impl.fm.common = common10b
impl.seed_train.common = common10b
impl.seed_train.fm.common = common10b
impl.NUM_CLASSES = common10b.NUM_CLASSES
impl.HOS_DIM = 20
impl.fm.NUM_CLASSES = common10b.NUM_CLASSES
impl.fm.HOS_DIM = 20
impl.seed_train.NUM_CLASSES = common10b.NUM_CLASSES
impl.seed_train.HOS_DIM = 20


def checkpoint_paths_10b(args, fold):
    tag = str(getattr(args, "run_tag", "") or "").strip()
    tag = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in tag)
    tag_part = f"_{tag}" if tag else ""
    suffix = f"fourier_compressed_oof_10b_{args.variant}{tag_part}_mseed{args.model_seed}_fold{fold}_split{args.split_seed}"
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
            (
                f"fourier_compressed_oof_10b_{args.variant}_mseed{args.model_seed}_"
                f"f{args.folds}e{args.epochs}_split{args.split_seed}_trainvaltest_probs_for_meta.npz"
            ),
        )

    os.makedirs(common10b.relpath("checkpoints"), exist_ok=True)
    os.makedirs(common10b.relpath("results"), exist_ok=True)
    print("[10B wrapper] data_path=", args.data_path)
    print("[10B wrapper] cache_dir=", args.cache_dir)
    print("[10B wrapper] alignment_cache=", args.alignment_cache)
    print("[10B wrapper] output_cache=", args.output_cache)
    impl.parse_args = lambda: args
    return impl.main()


if __name__ == "__main__":
    main()
