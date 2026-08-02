# Aggregate audit artifacts

This directory contains aggregate, non-identifying evidence for the results
reported with KAN-Fourier-RC. The tables preserve sample counts, fixed
comparison families, protocol metadata, and cryptographic anchors needed to
audit the numerical claims without redistributing licensed datasets.

Files:

- `reported_metrics.csv`: primary and final-system headline results, including
  exact correct-sample counts when a frozen prediction archive is available.
- `decision_level_benchmarks.csv`: matched decision-level baselines and
  rescue/harm accounting.
- `paired_significance.csv`: prespecified paired comparisons, stratified paired
  bootstrap intervals, McNemar counts, and Holm-adjusted p-values.
- `action_attribution.csv`: mutually exclusive final action families relative
  to the primary prediction.
- `incremental_action_attribution.csv`: additional actions of the full system
  relative to isolated OOF ERU. Its reference baseline differs from
  `action_attribution.csv` and the two tables must not be added together.
- `artifact_checksums.csv`: SHA-256 anchors for the frozen source evidence from
  which the aggregate tables were audited.
- `complexity_table_vi.csv`: the deployment-complexity ledger and the stated
  one-MAC-equals-two-FLOPs convention.
- `impairment_robustness.csv` and `impairment_config.json`: the aggregate
  post-hoc HisarMod2019.1 impairment analysis and its frozen generation
  settings.
- `partition_stability.csv`: post-hoc leave-one-storage-block sensitivity for
  HisarMod2019.1. Storage blocks are not interpreted as physical channels.

Percentages are percentage points of the corresponding held-out set. Displayed
accuracies are rounded independently; gains should be reconstructed from sample
counts where counts are provided. The formal RadioML prediction archives are
anchored by byte size and SHA-256 digest in `artifact_checksums.csv`.

Claim-specific verification paths, including aggregate recomputation and
artifact hash checks, are described in
[`docs/ARTIFACT_AVAILABILITY.md`](../docs/ARTIFACT_AVAILABILITY.md).
