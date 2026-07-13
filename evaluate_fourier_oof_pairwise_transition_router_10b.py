"""RML2016.10B adapter for the formal multi-aux OOF transition router."""

from __future__ import annotations

import os

import evaluate_fourier_oof_pairwise_transition_router_2016 as impl
import evaluate_greedy_soup_gamc_protected_residual_fusion as base
import train_cv_trn_aux_10b_common as common10b


def _patch_modules():
    base.NUM_CLASSES = common10b.NUM_CLASSES
    base.DEFAULT_MOD_CLASSES = common10b.DEFAULT_MOD_CLASSES
    impl.common = common10b
    impl.base = base
    modules = (impl.orig, impl.oof, impl.st, impl.bq, impl.extra_eval, impl.rr)
    for mod in modules:
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
    if hasattr(impl.rr, "orig"):
        impl.rr.orig.base = base
        impl.rr.orig.common = common10b


def main():
    _patch_modules()
    args = impl.parse_args()

    default_soup = impl.relpath("results", "greedy_soup_identity_valtest_probs_for_gamc_fusion.npz")
    default_fourier = impl.relpath(
        "results", "fourier_main_oof_merge_mseed78_79_f3e220_bs128_split1_trainvaltest_probs_for_meta.npz"
    )
    default_gamc = impl.relpath("results", "gamc_oof_tree_split1_trainvaltest_probs_for_meta.npz")
    default_cv = impl.relpath("results", "cv_trn_aux_v2_oof_mseed41_f3e240_split1_trainvaltest_probs_for_meta.npz")
    default_hcs = impl.relpath("results", "hcs_precision_analog_aux_split1_trainvaltest_probs_for_meta.npz")
    default_pair = impl.relpath("results", "pairwise_confusion_v2_aux_split1_trainvaltest_probs_for_meta.npz")

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
    if args.hcs_precision_cache == default_hcs:
        args.hcs_precision_cache = impl.relpath(
            "results", "hcs_analog_aux_oof_10b_mseed2037_f3e320_split1_trainvaltest_probs_for_meta.npz"
        )
    if args.pairwise_cache == default_pair:
        args.pairwise_cache = impl.relpath(
            "results", "pairwise_confusion_selected6_raw_oof_10b_f3e280_split1_trainvaltest_probs_for_meta.npz"
        )
    if args.data_path == impl.relpath("raw_data", "RML2016.10a_dict.pkl"):
        args.data_path = common10b.DEFAULT_10B_DATA_PATH
    if args.cache_dir == impl.relpath("feature_cache"):
        args.cache_dir = common10b.DEFAULT_10B_CACHE_DIR
    if args.alignment_cache == default_soup:
        args.alignment_cache = common10b.DEFAULT_10B_ALIGNMENT_CACHE
    if args.hcs_feature_cache == impl.relpath("feature_cache", "hcs_lite_features_v1.npz"):
        args.hcs_feature_cache = os.path.join(common10b.DEFAULT_10B_CACHE_DIR, "hcs_lite_10b_features_v1.npz")
    if args.gamc_feature_cache == impl.relpath("feature_cache", "gamc_lite_features_v3_graph_xgb.npz"):
        args.gamc_feature_cache = os.path.join(
            common10b.DEFAULT_10B_CACHE_DIR, "gamc_lite_10b_features_v3_graph_xgb.npz"
        )
    if args.output_suffix == "fourier_oof_pairwise_transition_router_split1":
        args.output_suffix = "fourier_compressed_10b_formal_multiaux_transition_router_split1"

    os.makedirs(impl.relpath("results"), exist_ok=True)
    impl.parse_args = lambda: args
    return impl.main()


if __name__ == "__main__":
    main()
