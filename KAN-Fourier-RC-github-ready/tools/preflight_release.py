"""Fail-closed checks for the selective public research release."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


DATASET_SIZES = {
    "RML2016.10A": 22_000,
    "RML2016.10B": 120_000,
    "HisarMod2019.1": 260_000,
}
EXPECTED_COMPARISONS = {
    (dataset, comparison)
    for dataset in DATASET_SIZES
    for comparison in {
        "Full_ERU_RC_vs_Primary",
        "Full_ERU_RC_vs_OOF_Linear_Stacking",
        "Full_ERU_RC_vs_OOF_XGBoost_Stacking",
        "Full_ERU_RC_vs_OOF_Candidate_Competence",
        "Full_ERU_RC_vs_Isolated_OOF_ERU",
    }
}
REQUIRED_RELEASE_FILES = {
    "LICENSE",
    "README.md",
    "MANIFEST.md",
    "oof_protocol.py",
    "apply_crossfit_candidate_action_router_2016.py",
    "audit_artifacts/decision_level_benchmarks.csv",
    "audit_artifacts/paired_significance.csv",
    "audit_artifacts/action_attribution.csv",
    "audit_artifacts/incremental_action_attribution.csv",
    "audit_artifacts/partition_stability.csv",
    "docs/OOF_PROTOCOL.md",
    "docs/RELEASE_BOUNDARY.md",
    "tests/test_oof_protocol.py",
}
FORBIDDEN_RELEASE_NAMES = {
    "LICENSE_PENDING.md",
    "model_complex_tcn_fusion.py",
    "train_oracle_privileged_distill.py",
    "run_rc_10b_from_caches.ps1",
}
FORBIDDEN_RELEASE_PREFIXES = (
    "evaluate_",
    "generate_",
    "train_gamc_",
    "train_hcs_",
    "train_pairwise_",
)
FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".dat",
    ".h5",
    ".joblib",
    ".npy",
    ".npz",
    ".pkl",
    ".pt",
    ".pth",
}
TEXT_SUFFIXES = {".bib", ".cff", ".csv", ".md", ".ps1", ".py", ".txt"}
LOCAL_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:[\\/]"),
    re.compile(r"/(?:root|home)/autodl(?:-tmp)?/", re.IGNORECASE),
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float:
    value = row.get(key, "").strip()
    if not value:
        return math.nan
    return float(value)


def probability(value: str) -> tuple[float, bool]:
    value = value.strip()
    censored = value.startswith("<")
    if censored:
        value = value[1:]
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        fail(f"Invalid probability: {value}")
    return parsed, censored


def check_decision_table(root: Path) -> None:
    path = root / "audit_artifacts" / "decision_level_benchmarks.csv"
    rows = read_csv(path)
    primary = {}
    for row in rows:
        dataset = row["dataset"]
        if dataset not in DATASET_SIZES:
            fail(f"Unknown dataset in {path.name}: {dataset}")
        if row["method"] == "Primary":
            primary[dataset] = number(row, "overall_percent")
    if set(primary) != set(DATASET_SIZES):
        fail("Decision table must contain one Primary row for every dataset.")

    for row in rows:
        dataset = row["dataset"]
        size = DATASET_SIZES[dataset]
        changed = int(row["changed_count"])
        rescue = int(row["rescue_count"])
        harm = int(row["harm_count"])
        net = int(row["net_gain_count"])
        if net != rescue - harm:
            fail(f"{dataset}/{row['method']}: net_gain_count != rescue_count - harm_count")
        expected_changed = 100.0 * changed / size
        if abs(number(row, "changed_percent") - expected_changed) > 1e-6:
            fail(f"{dataset}/{row['method']}: changed percentage disagrees with count")
        expected_gain = 100.0 * net / size
        if abs(number(row, "net_gain_pp") - expected_gain) > 1e-6:
            fail(f"{dataset}/{row['method']}: net gain disagrees with sample counts")
        expected_overall = primary[dataset] + expected_gain
        if abs(number(row, "overall_percent") - expected_overall) > 1e-6:
            fail(f"{dataset}/{row['method']}: accuracy does not reconcile to Primary + net gain")
        utility = number(row, "conditional_utility_percent")
        if changed:
            expected_utility = 100.0 * net / changed
            if abs(utility - expected_utility) > 1e-6:
                fail(f"{dataset}/{row['method']}: conditional utility disagrees with counts")
        elif not math.isnan(utility):
            fail(f"{dataset}/{row['method']}: zero-change row must leave utility blank")


def check_reported_metrics(root: Path) -> None:
    path = root / "audit_artifacts" / "reported_metrics.csv"
    rows = read_csv(path)
    by_dataset = defaultdict(dict)
    for row in rows:
        by_dataset[row["dataset"]][row["system"]] = row
        size = int(row["split_size"])
        count = int(row["correct_count"])
        expected = 100.0 * count / size
        if abs(number(row, "accuracy_percent") - expected) > 1e-6:
            fail(f"{row['dataset']}/{row['system']}: accuracy disagrees with correct_count")
    for dataset, systems in by_dataset.items():
        primary = systems.get("Primary")
        full = systems.get("Full_ERU_RC")
        if not primary or not full:
            fail(f"{dataset}: reported metrics require Primary and Full_ERU_RC")
        expected = number(full, "accuracy_percent") - number(primary, "accuracy_percent")
        if abs(number(full, "gain_over_primary_pp") - expected) > 2e-6:
            fail(f"{dataset}: displayed Full gain is inconsistent")


def check_significance(root: Path) -> None:
    path = root / "audit_artifacts" / "paired_significance.csv"
    rows = read_csv(path)
    observed = [(row["dataset"], row["comparison"]) for row in rows]
    if len(observed) != len(set(observed)):
        fail("Paired-significance table contains duplicate comparisons.")
    if set(observed) != EXPECTED_COMPARISONS:
        missing = sorted(EXPECTED_COMPARISONS - set(observed))
        extra = sorted(set(observed) - EXPECTED_COMPARISONS)
        fail(f"Paired-significance family must be the fixed 15 comparisons; missing={missing}, extra={extra}")
    for row in rows:
        dataset = row["dataset"]
        size = DATASET_SIZES[dataset]
        expected = 100.0 * (int(row["n10"]) - int(row["n01"])) / size
        if abs(number(row, "difference_pp") - expected) > 1e-6:
            fail(f"{dataset}/{row['comparison']}: paired counts disagree with accuracy difference")
        if number(row, "ci95_low_pp") > number(row, "ci95_high_pp"):
            fail(f"{dataset}/{row['comparison']}: inverted confidence interval")
        if not number(row, "ci95_low_pp") <= number(row, "difference_pp") <= number(row, "ci95_high_pp"):
            fail(f"{dataset}/{row['comparison']}: point estimate lies outside its confidence interval")
        if int(row["bootstrap_repetitions"]) != 10_000:
            fail(f"{dataset}/{row['comparison']}: unexpected bootstrap repetition count")
        expected_stratum = "class_x_snr_x_storage_block" if dataset == "HisarMod2019.1" else "class_x_snr"
        if row["stratification"] != expected_stratum:
            fail(f"{dataset}/{row['comparison']}: unexpected bootstrap stratification")
        raw, raw_censored = probability(row["mcnemar_p_raw"])
        adjusted, adjusted_censored = probability(row["holm_adjusted_p"])
        if not raw_censored and not adjusted_censored and adjusted + 1e-15 < raw:
            fail(f"{dataset}/{row['comparison']}: Holm-adjusted p is below the raw p")


def check_partition_stability(root: Path) -> None:
    path = root / "audit_artifacts" / "partition_stability.csv"
    rows = read_csv(path)
    expected_partitions = {f"Storage_block_{index}" for index in range(1, 6)} | {"All_storage_blocks"}
    observed = {row["partition"] for row in rows}
    if len(rows) != 6 or observed != expected_partitions:
        fail("Partition stability must contain five storage blocks and one aggregate row.")
    for row in rows:
        if row["dataset"] != "HisarMod2019.1":
            fail("Partition stability may only describe HisarMod2019.1.")
        if row["analysis_role"] != "post_hoc_sensitivity":
            fail(f"{row['partition']}: storage analysis must remain explicitly post hoc")
        if row["physical_channel_claim"] != "not_claimed":
            fail(f"{row['partition']}: storage blocks must not be relabeled as physical channels")
        gain = number(row, "gain_pp")
        low = number(row, "ci95_low_pp")
        high = number(row, "ci95_high_pp")
        if not low <= gain <= high:
            fail(f"{row['partition']}: gain lies outside its confidence interval")
        expected_size = 260_000 if row["partition"] == "All_storage_blocks" else 52_000
        if int(row["partition_size"]) != expected_size:
            fail(f"{row['partition']}: unexpected partition size")

    aggregate = next(row for row in rows if row["partition"] == "All_storage_blocks")
    primary = number(aggregate, "primary_percent")
    full = number(aggregate, "frozen_policy_percent")
    gain = number(aggregate, "gain_pp")
    if abs((full - primary) - gain) > 2e-6:
        fail("Hisar aggregate partition gain does not reconcile to the displayed accuracies.")
    if abs((number(aggregate, "rescue_percent") - number(aggregate, "harm_percent")) - gain) > 2e-6:
        fail("Hisar aggregate partition gain does not reconcile to rescue minus harm.")
    if number(aggregate, "changed_percent") + 1e-12 < (
        number(aggregate, "rescue_percent") + number(aggregate, "harm_percent")
    ):
        fail("Hisar aggregate changed rate is below rescue plus harm.")


def check_action_tables(root: Path) -> None:
    final_rows = read_csv(root / "audit_artifacts" / "action_attribution.csv")
    grouped = defaultdict(list)
    for row in final_rows:
        grouped[row["dataset"]].append(row)
        if int(row["net_gain_count"]) != int(row["rescue_count"]) - int(row["harm_count"]):
            fail(f"{row['dataset']}/{row['action_family']}: invalid action net count")
    for dataset, rows in grouped.items():
        size = DATASET_SIZES[dataset]
        if sum(int(row["activation_count"]) for row in rows) != size:
            fail(f"{dataset}: mutually exclusive action counts do not sum to split size")
        active = [row for row in rows if row["action_family"] not in {"Blocked_reverted_candidate_action", "Retain_primary"}]
        net = sum(int(row["net_gain_count"]) for row in active)
        full = next(row for row in read_csv(root / "audit_artifacts" / "decision_level_benchmarks.csv")
                    if row["dataset"] == dataset and row["method"] == "Full_ERU_RC")
        if net != int(full["net_gain_count"]):
            fail(f"{dataset}: final action-family gain does not reconcile to Full ERU-RC")
        if sum(int(row["activation_count"]) for row in active) != int(full["changed_count"]):
            fail(f"{dataset}: active action counts do not reconcile to Full changed_count")

    incremental = read_csv(root / "audit_artifacts" / "incremental_action_attribution.csv")
    grouped_incremental = defaultdict(list)
    for row in incremental:
        grouped_incremental[row["dataset"]].append(row)
        if int(row["net_gain_count"]) != int(row["rescue_count"]) - int(row["harm_count"]):
            fail(f"{row['dataset']}/{row['incremental_action']}: invalid incremental net count")
    decision = read_csv(root / "audit_artifacts" / "decision_level_benchmarks.csv")
    for dataset, rows in grouped_incremental.items():
        full = next(row for row in decision if row["dataset"] == dataset and row["method"] == "Full_ERU_RC")
        isolated = next(row for row in decision if row["dataset"] == dataset and row["method"] == "Isolated_OOF_ERU")
        expected = int(full["net_gain_count"]) - int(isolated["net_gain_count"])
        actual = sum(int(row["net_gain_count"]) for row in rows)
        if actual != expected:
            fail(f"{dataset}: incremental attribution does not reconcile to Full minus isolated ERU")


def check_release_boundary(root: Path) -> None:
    missing = sorted(relative for relative in REQUIRED_RELEASE_FILES if not (root / relative).is_file())
    if missing:
        fail(f"Required public-release files are missing: {missing}")
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in {".git", "__pycache__", "build"} for part in relative.parts):
            continue
        if path.name in FORBIDDEN_RELEASE_NAMES or path.name.startswith(FORBIDDEN_RELEASE_PREFIXES):
            fail(f"Private or exploratory implementation crossed the release boundary: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            fail(f"Private/binary artifact must not be included: {relative}")
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"LICENSE", "MANIFEST.md"}:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            for pattern in LOCAL_PATH_PATTERNS:
                if pattern.search(text):
                    fail(f"Absolute local path found in public text file: {relative}")

    router = (root / "apply_crossfit_candidate_action_router_2016.py").read_text(encoding="utf-8")
    forbidden_router_fragments = ('print_metrics("Base Test"', 'print_metrics("Candidate')
    if any(fragment in router for fragment in forbidden_router_fragments):
        fail("Reference router exposes held-out diagnostics before the final frozen-policy report.")


def check_markdown_links(root: Path) -> None:
    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for path in root.rglob("*.md"):
        if any(part in {".git", "__pycache__", "build"} for part in path.relative_to(root).parts):
            continue
        for target in link_pattern.findall(path.read_text(encoding="utf-8-sig")):
            target = target.strip().strip("<>").split("#", 1)[0]
            if not target or re.match(r"(?:https?|mailto):", target, flags=re.IGNORECASE):
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                fail(f"Broken local Markdown link in {path.relative_to(root)}: {target}")


def compile_python(root: Path) -> None:
    sources = [
        path
        for path in root.rglob("*.py")
        if not any(part in {"__pycache__", "build"} for part in path.relative_to(root).parts)
    ]
    for path in sources:
        source = path.read_text(encoding="utf-8-sig")
        try:
            compile(source, str(path), "exec")
        except SyntaxError as exc:
            fail(f"Python compilation check failed for {path.relative_to(root)}: {exc}")


def run_protocol_tests(root: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=root,
        env=environment,
        check=False,
    )
    if result.returncode:
        fail("OOF protocol unit tests failed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    compile_python(root)
    run_protocol_tests(root)
    check_decision_table(root)
    check_reported_metrics(root)
    check_significance(root)
    check_action_tables(root)
    check_partition_stability(root)
    check_release_boundary(root)
    check_markdown_links(root)
    print("Release preflight checks passed.")


if __name__ == "__main__":
    main()
