"""RML2016.10B adapter for HCS/Pairwise extra-aux OOF meta fusion."""

from __future__ import annotations

import os

import evaluate_fourier_oof_gamc_cvtrn_extraaux_residual_meta_fusion as impl
import evaluate_greedy_soup_gamc_protected_residual_fusion as base
import train_cv_trn_aux_10b_common as common10b


def _patch_modules():
    base.NUM_CLASSES = common10b.NUM_CLASSES
    base.DEFAULT_MOD_CLASSES = common10b.DEFAULT_MOD_CLASSES
    impl.common = common10b
    impl.base = base
    for mod in (impl.orig, impl.oof, impl.st, impl.bq):
        if hasattr(mod, "common"):
            mod.common = common10b
        if hasattr(mod, "base"):
            mod.base = base
    if hasattr(impl.oof, "bq"):
        impl.oof.bq.common = common10b
        impl.oof.bq.base = base
    if hasattr(impl.oof, "st"):
        impl.oof.st.base = base
    if hasattr(impl.st, "gr"):
        impl.st.gr.base = base
    if hasattr(impl.bq, "gr"):
        impl.bq.gr.base = base
    if hasattr(impl.bq, "st"):
        impl.bq.st.base = base


def main():
    _patch_modules()
    args = impl.parse_args()

    default_soup = impl.relpath("results", "greedy_soup_identity_valtest_probs_for_gamc_fusion.npz")
    default_fourier = impl.relpath(
        "results", "fourier_main_oof_merge_mseed78_79_f3e220_bs128_split1_trainvaltest_probs_for_meta.npz"
    )
    default_gamc = impl.relpath("results", "gamc_oof_tree_split1_trainvaltest_probs_for_meta.npz")
    default_cv = impl.relpath("results", "cv_trn_aux_v2_oof_mseed41_f3e240_split1_trainvaltest_probs_for_meta.npz")

    if args.soup_prob_cache == default_soup:
        args.soup_prob_cache = common10b.DEFAULT_10B_ALIGNMENT_CACHE
    if args.fourier_oof_cache == default_fourier:
        args.fourier_oof_cache = impl.relpath(
            "results", "fourier_compressed_oof_10b_full_geo_2expert_mseed369_f3e220_single361_split1_trainvaltest_probs_for_meta.npz"
        )
    if args.gamc_oof_cache == default_gamc:
        args.gamc_oof_cache = impl.relpath("results", "gamc_oof_tree_10b_split1_trainvaltest_probs_for_meta.npz")
    if args.cvtrn_oof_cache == default_cv:
        args.cvtrn_oof_cache = impl.relpath(
            "results", "cv_trn_aux_v2_oof_10b_mseed141_f3e220_split1_trainvaltest_probs_for_meta.npz"
        )
    if args.data_path == impl.relpath("raw_data", "RML2016.10a_dict.pkl"):
        args.data_path = common10b.DEFAULT_10B_DATA_PATH
    if args.cache_dir == impl.relpath("feature_cache"):
        args.cache_dir = common10b.DEFAULT_10B_CACHE_DIR
    if args.alignment_cache == default_soup:
        args.alignment_cache = common10b.DEFAULT_10B_ALIGNMENT_CACHE
    if args.output_suffix == "fourier_oof_gamc_cvtrn_extraaux_residual_meta_split1":
        args.output_suffix = "fourier_oof_gamc_cvtrn_hcs_pairwise_extraaux_10b_split1"
    args.amdsb_class_idx = 1
    args.wbfm_class_idx = 9

    os.makedirs(impl.relpath("results"), exist_ok=True)
    impl.parse_args = lambda: args
    return impl.main()


if __name__ == "__main__":
    main()
