"""Reference HisarMod2019.1 impairment transforms and aggregate checks.

This script makes the post-hoc stress conditions unambiguous without shipping
trained models or sample-level predictions.  It can verify the released
aggregate ledger and exposes deterministic NumPy transforms for authorized
replay with the controlled frozen artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import zlib
from pathlib import Path

import numpy as np


def restore_rms(reference: np.ndarray, impaired: np.ndarray) -> np.ndarray:
    reference_rms = np.sqrt(np.mean(np.square(reference, dtype=np.float64), axis=(1, 2)) + 1e-12)
    impaired_rms = np.sqrt(np.mean(np.square(impaired, dtype=np.float64), axis=(1, 2)) + 1e-12)
    scale = reference_rms / np.maximum(impaired_rms, 1e-8)
    return np.asarray(impaired * scale[:, None, None], dtype=np.float32)


def complex_to_iq(values: np.ndarray) -> np.ndarray:
    return np.stack([values.real, values.imag], axis=1).astype(np.float32, copy=False)


def apply_multipath(complex_iq: np.ndarray, taps: np.ndarray) -> np.ndarray:
    output = np.zeros_like(complex_iq, dtype=np.complex64)
    for lag in range(taps.shape[1]):
        if lag == 0:
            output += taps[:, lag, None] * complex_iq
        else:
            output[:, lag:] += taps[:, lag, None] * complex_iq[:, :-lag]
    return output


def condition_seed(base_seed: int, condition: str, chunk_start: int) -> int:
    return int((int(base_seed) + zlib.crc32(condition.encode("utf-8")) + int(chunk_start)) % (2**32 - 1))


def apply_impairment(iq: np.ndarray, condition: str, seed: int) -> np.ndarray:
    reference = np.asarray(iq, dtype=np.float32)
    if reference.ndim != 3 or reference.shape[1] != 2:
        raise ValueError("Expected I/Q input with shape (N, 2, L).")
    if condition == "clean":
        return np.array(reference, copy=True)
    complex_iq = reference[:, 0].astype(np.complex64) + 1j * reference[:, 1].astype(np.complex64)
    if condition.startswith("cfo_"):
        epsilon = {"cfo_m001": -0.001, "cfo_p001": 0.001, "cfo_m003": -0.003, "cfo_p003": 0.003}[condition]
        phase = np.exp(1j * 2.0 * np.pi * epsilon * np.arange(complex_iq.shape[1], dtype=np.float32)).astype(np.complex64)
        return restore_rms(reference, complex_to_iq(complex_iq * phase[None]))
    if condition.startswith("iq_"):
        gain_db, phase_deg = {
            "iq_mild_neg": (-0.5, -3.0), "iq_mild_pos": (0.5, 3.0),
            "iq_severe_neg": (-1.0, -5.0), "iq_severe_pos": (1.0, 5.0),
        }[condition]
        gain = float(10.0 ** (gain_db / 40.0))
        phase = math.radians(phase_deg)
        i_part = gain * reference[:, 0]
        q_part = (math.cos(phase) * reference[:, 1] + math.sin(phase) * reference[:, 0]) / gain
        return restore_rms(reference, np.stack([i_part, q_part], axis=1).astype(np.float32))
    if condition not in {"rayleigh_3tap", "rician_k6_3tap"}:
        raise ValueError(f"Unknown impairment condition: {condition}")
    rng = np.random.default_rng(int(seed))
    power = np.power(10.0, np.asarray([0.0, -3.0, -6.0]) / 10.0)
    power /= power.sum()
    scatter = (rng.standard_normal((len(reference), 3)) + 1j * rng.standard_normal((len(reference), 3))) / np.sqrt(2.0)
    scatter = scatter.astype(np.complex64)
    if condition == "rayleigh_3tap":
        taps = scatter * np.sqrt(power[None]).astype(np.float32)
    else:
        k_linear = float(10.0 ** (6.0 / 10.0))
        taps = np.sqrt(1.0 / (k_linear + 1.0)) * scatter * np.sqrt(power[None]).astype(np.float32)
        taps[:, 0] += np.sqrt(k_linear / (k_linear + 1.0))
    taps = np.asarray(taps / np.sqrt(np.sum(np.square(np.abs(taps)), axis=1, keepdims=True) + 1e-12), dtype=np.complex64)
    return restore_rms(reference, complex_to_iq(apply_multipath(complex_iq, taps)))


def verify_aggregate(config_path: Path, results_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    conditions = config["conditions"]
    observed = [row["condition"] for row in rows]
    if set(observed) != set(conditions) or len(rows) != len(conditions):
        raise RuntimeError("Stress ledger does not contain exactly the configured conditions.")
    if config["analysis_role"] != "post_hoc_frozen_policy_sensitivity" or config["policy_refit"]:
        raise RuntimeError("Stress analysis must remain post hoc with no policy refit.")
    for row in rows:
        if int(row["rows"]) != int(config["sample_count"]):
            raise RuntimeError(f"Unexpected row count for {row['condition']}")
        gain = float(row["gain_pp"])
        if not float(row["gain_ci_low_pp"]) <= gain <= float(row["gain_ci_high_pp"]):
            raise RuntimeError(f"Gain lies outside its interval for {row['condition']}")
        expected = 100.0 * (float(row["final_accuracy"]) - float(row["primary_accuracy"]))
        if not math.isclose(gain, expected, abs_tol=1e-9):
            raise RuntimeError(f"Accuracy/gain mismatch for {row['condition']}")
    print(f"Verified {len(rows)} frozen-policy stress conditions: {results_path}")
    print("Synthetic fading conditions are stress transforms, not recovered physical-channel labels.")


def self_test(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rng = np.random.default_rng(7)
    iq = rng.standard_normal((4, 2, 1024), dtype=np.float32)
    for condition in config["conditions"]:
        seed = condition_seed(config["impairment_seed"], condition, 0)
        first = apply_impairment(iq, condition, seed)
        second = apply_impairment(iq, condition, seed)
        if first.shape != iq.shape or first.dtype != np.float32 or not np.array_equal(first, second):
            raise RuntimeError(f"Determinism/shape check failed for {condition}")
    print("Reference impairment transforms passed deterministic self-test.")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=root / "audit_artifacts" / "impairment_config.json")
    parser.add_argument("--results", type=Path, default=root / "audit_artifacts" / "impairment_robustness.csv")
    parser.add_argument("--verify-aggregate", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.verify_aggregate and not args.self_test:
        args.verify_aggregate = args.self_test = True
    if args.self_test:
        self_test(args.config)
    if args.verify_aggregate:
        verify_aggregate(args.config, args.results)


if __name__ == "__main__":
    main()
