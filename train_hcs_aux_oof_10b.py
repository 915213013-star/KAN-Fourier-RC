"""RML2016.10B adapter for the confusion-aware HCS OOF specialists."""

from __future__ import annotations

import evaluate_greedy_soup_gamc_protected_residual_fusion as base
import train_cv_trn_aux_10b_common as common10b
import train_hcs_aux_oof_2016 as impl


GROUPS_10B = {
    "analog": {
        "classes": [1, 9],  # AM-DSB, WBFM
        "boost": 2.0,
        "context_boost": 1.25,
        "neg_boost": 1.00,
        "trans_boost": 1.15,
        "high_boost": 1.15,
        "depth": 3,
        "estimators_delta": 0,
    },
    "constellation": {
        "classes": [0, 2, 6, 7, 8],  # 8PSK, BPSK, QAM16, QAM64, QPSK
        "boost": 1.85,
        "context_boost": 1.30,
        "neg_boost": 1.15,
        "trans_boost": 1.18,
        "high_boost": 1.00,
        "depth": 4,
        "estimators_delta": 80,
    },
    "fskpam": {
        "classes": [2, 3, 4, 5],  # BPSK, CPFSK, GFSK, PAM4
        "boost": 1.75,
        "context_boost": 1.25,
        "neg_boost": 1.20,
        "trans_boost": 1.10,
        "high_boost": 1.00,
        "depth": 4,
        "estimators_delta": 60,
    },
}


def _patch_modules():
    base.NUM_CLASSES = common10b.NUM_CLASSES
    base.DEFAULT_MOD_CLASSES = common10b.DEFAULT_MOD_CLASSES
    impl.common = common10b
    impl.base = base
    impl.GROUPS = GROUPS_10B
    impl.fourier_meta.common = common10b
    impl.fourier_meta.base = base


def main():
    _patch_modules()
    return impl.main()


if __name__ == "__main__":
    main()
