"""10B wrapper for cross-fitted GAMC/GAMC-low tree auxiliary experts."""

from __future__ import annotations

import os

import train_cv_trn_aux_10b_common as common10b
import train_gamc_low_snr_aux_experts_2016 as gl
import train_gamc_oof_tree_experts_2016 as impl


impl.common = common10b
impl.gl.common = common10b
gl.common = common10b


_orig_align_proba = gl.align_proba


def align_proba_10b(clf, p, n_classes=10):
    return _orig_align_proba(clf, p, n_classes=10)


gl.align_proba = align_proba_10b
impl.gl.align_proba = align_proba_10b


def main():
    args = impl.parse_args()
    if args.data_path == common10b.relpath("raw_data", "RML2016.10a_dict.pkl"):
        args.data_path = common10b.DEFAULT_10B_DATA_PATH
    if args.cache_dir == common10b.relpath("feature_cache"):
        args.cache_dir = common10b.DEFAULT_10B_CACHE_DIR
    if args.alignment_cache == common10b.relpath("results", "greedy_soup_identity_valtest_probs_for_gamc_fusion.npz"):
        args.alignment_cache = common10b.DEFAULT_10B_ALIGNMENT_CACHE
    if args.feature_cache == common10b.relpath("feature_cache", "gamc_lite_features_v3_graph_xgb.npz"):
        args.feature_cache = str(common10b.TENB_ROOT / "feature_cache" / "gamc_lite_10b_features_v3_graph_xgb.npz")
    if args.output_cache == common10b.relpath("results", "gamc_oof_tree_split1_trainvaltest_probs_for_meta.npz"):
        args.output_cache = common10b.relpath("results", f"gamc_oof_tree_10b_split{args.split_seed}_trainvaltest_probs_for_meta.npz")
    os.makedirs(common10b.relpath("results"), exist_ok=True)
    print("[10B wrapper] data_path=", args.data_path)
    print("[10B wrapper] cache_dir=", args.cache_dir)
    print("[10B wrapper] alignment_cache=", args.alignment_cache)
    print("[10B wrapper] feature_cache=", args.feature_cache)
    impl.parse_args = lambda: args
    return impl.main()


if __name__ == "__main__":
    main()
