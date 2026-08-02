"""Verify the model identities and the paper's deployment-complexity ledger.

The FLOP values in the paper use one multiply-accumulate (MAC) = two FLOPs.
Custom quaternion, SPD, and complex-correlation operators are accounted for by
the frozen ledger rather than delegated to a backend with incomplete custom-op
coverage. This script instantiates the released RadioML neural architectures,
checks their exact trainable-parameter counts, and validates the ledger.
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from model_cv_trn_aux_v2_2016 import build_cv_trn_aux_v2_model
from model_moe_attention_compressed import build_compressed_model


ROOT = Path(__file__).resolve().parent
LEDGER = ROOT / "audit_artifacts" / "complexity_table_vi.csv"


def trainable_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def read_ledger() -> list[dict[str, str]]:
    with LEDGER.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    primary = build_compressed_model("full_geo_2expert", num_classes=11, hos_dim=20)
    iqcc = build_cv_trn_aux_v2_model(torch.device("cpu"), num_classes=11)
    primary_params = trainable_parameters(primary)
    iqcc_params = trainable_parameters(iqcc)
    subtotal_params = primary_params + iqcc_params

    if primary_params != 197_290:
        raise RuntimeError(f"Unexpected RadioML primary parameter count: {primary_params}")
    if iqcc_params != 172_117:
        raise RuntimeError(f"Unexpected IQCC-Former parameter count: {iqcc_params}")
    if subtotal_params != 369_407:
        raise RuntimeError(f"Unexpected RadioML neural subtotal: {subtotal_params}")

    rows = read_ledger()
    if len(rows) != 4:
        raise RuntimeError("The Table VI complexity ledger must contain four rows.")
    for row in rows:
        macs = int(row["macs"])
        flops = int(row["flops"])
        if flops != 2 * macs:
            raise RuntimeError(f"{row['dataset']}/{row['component']}: FLOPs != 2 x MACs")

    radio_primary = next(
        row for row in rows
        if row["dataset"] == "RadioML2016.10A/10B" and row["component"] == "KAN-Fourier primary"
    )
    radio_subtotal = next(
        row for row in rows
        if row["dataset"] == "RadioML2016.10A/10B" and row["component"] == "Deployed neural subtotal"
    )
    if int(radio_primary["trainable_parameters"]) != primary_params:
        raise RuntimeError("RadioML primary ledger row does not match the released architecture.")
    if int(radio_subtotal["trainable_parameters"]) != subtotal_params:
        raise RuntimeError("RadioML neural-subtotal ledger row does not match the released architectures.")

    print(f"RadioML KAN-Fourier parameters : {primary_params:,}")
    print(f"RadioML IQCC-Former parameters : {iqcc_params:,}")
    print(f"RadioML neural subtotal        : {subtotal_params:,}")
    print("Complexity ledger verified (1 MAC = 2 FLOPs).")


if __name__ == "__main__":
    main()
