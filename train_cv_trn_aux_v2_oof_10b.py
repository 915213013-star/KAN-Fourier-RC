"""10B wrapper for CV-TRN-v2 train-split OOF auxiliary probabilities."""

from __future__ import annotations

import os

import oof_protocol as oofp
import train_cv_trn_aux_10b_common as common10b
import train_cv_trn_aux_v2_2016 as v2
import train_cv_trn_aux_v2_oof_2016 as impl


impl.common = common10b
impl.v2.common = common10b
v2.common = common10b


def checkpoint_paths_10b(args, fold):
    suffix = (
        f"cv_trn_aux_v2_oof_10b_{oofp.PROTOCOL_TAG}_"
        f"mseed{args.model_seed}_fold{fold}_split{args.split_seed}"
    )
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
                f"cv_trn_aux_v2_oof_10b_{oofp.PROTOCOL_TAG}_mseed{args.model_seed}_"
                f"f{args.folds}e{args.epochs}_split{args.split_seed}_trainvaltest_probs_for_meta.npz"
            ),
        )
    os.makedirs(common10b.relpath("checkpoints"), exist_ok=True)
    os.makedirs(common10b.relpath("results"), exist_ok=True)
    print("[10B wrapper] data_path=", args.data_path)
    print("[10B wrapper] cache_dir=", args.cache_dir)
    print("[10B wrapper] alignment_cache=", args.alignment_cache)
    return impl.main_with_args(
        args,
        title="Train RML2016.10B CV-TRN-v2 OOF auxiliary experts",
    )


if __name__ == "__main__":
    main()
