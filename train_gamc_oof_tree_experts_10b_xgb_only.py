"""10B GAMC OOF tree experts without the memory-heavy ExtraTrees member.

RML2016.10B has 960k train samples under split_seed=1.  The original
ExtraTrees member can require too much RAM during its full-train final fit, so
this wrapper keeps the XGBoost/GAMC-low members and skips only ExtraTrees.
"""

from __future__ import annotations

import train_gamc_oof_tree_experts_10b as wrapper
import train_gamc_oof_tree_experts_2016 as impl


_ORIGINAL_MEMBER_SPECS = impl.member_specs


def member_specs_xgb_only():
    return [spec for spec in _ORIGINAL_MEMBER_SPECS() if spec[1] != "et"]


impl.member_specs = member_specs_xgb_only
wrapper.impl.member_specs = member_specs_xgb_only


def main():
    print("[10B xgb-only] Skipping ExtraTrees member to avoid 10B RAM overflow.")
    return wrapper.main()


if __name__ == "__main__":
    main()
