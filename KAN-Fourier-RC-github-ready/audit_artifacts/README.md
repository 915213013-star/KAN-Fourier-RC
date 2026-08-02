# Aggregate audit artifacts

This directory contains aggregate, non-identifying evidence for the results
reported with KAN-Fourier-RC. It intentionally does **not** contain model
weights, feature caches, per-sample probability arrays, test labels, or the
complete policy-search grid.

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
- `artifact_checksums.csv`: SHA-256 anchors for the private frozen evidence from
  which the aggregate tables were audited.
- `partition_stability.csv`: post-hoc leave-one-storage-block sensitivity for
  HisarMod2019.1. Storage blocks are not interpreted as physical channels.

Percentages are percentage points of the corresponding held-out set. Displayed
accuracies are rounded independently; gains should be reconstructed from sample
counts where counts are provided. The formal 10B result is anchored by the
aggregate benchmark record because no row-level archive is distributed or
identified as the formal `66.168%` result in this public release.

The row-level artifacts are available for confidential academic verification
subject to data-license, privacy, and intellectual-property constraints. See
[`docs/ARTIFACT_AVAILABILITY.md`](../docs/ARTIFACT_AVAILABILITY.md).
