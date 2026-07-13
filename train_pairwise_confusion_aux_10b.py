"""RML2016.10B adapter for train-split OOF pairwise specialists."""

from __future__ import annotations

import evaluate_greedy_soup_gamc_protected_residual_fusion as base
import train_cv_trn_aux_10b_common as common10b
import train_pairwise_confusion_aux_2016 as impl


DEFAULT_PAIRS_10B = ["1,9", "6,7", "3,4", "0,8", "0,2", "2,5"]


def _patch_modules():
    base.NUM_CLASSES = common10b.NUM_CLASSES
    base.DEFAULT_MOD_CLASSES = common10b.DEFAULT_MOD_CLASSES
    impl.common = common10b
    impl.base = base
    impl.DEFAULT_PAIRS = DEFAULT_PAIRS_10B
    impl.orig.common = common10b
    impl.orig.base = base


def main():
    _patch_modules()
    return impl.main()


if __name__ == "__main__":
    main()
