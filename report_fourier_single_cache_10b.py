"""RML2016.10B adapter for reporting one KAN-Fourier probability cache."""

from __future__ import annotations

import evaluate_greedy_soup_gamc_protected_residual_fusion as base
import report_fourier_single_cache_2016 as impl
import train_cv_trn_aux_10b_common as common10b


def main():
    base.NUM_CLASSES = common10b.NUM_CLASSES
    base.DEFAULT_MOD_CLASSES = common10b.DEFAULT_MOD_CLASSES
    impl.base = base
    return impl.main()


if __name__ == "__main__":
    main()
