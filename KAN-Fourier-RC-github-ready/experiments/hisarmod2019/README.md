# HisarMod2019.1 Experiment Scope

The paper's HisarMod2019.1 configuration uses length-1024 I/Q records, the
official 520,000/260,000 storage split, a KAN-Fourier primary, IQCC-Former,
non-neural candidates, and a long-spectral lightweight candidate. The frozen
full system reports 79.866923% versus 77.769231% for the primary.

This selective GitHub release does not include the dataset, its derived arrays,
the full Hisar training pipeline, or sample-level predictions. It includes:

- aggregate decision baselines in `audit_artifacts/decision_level_benchmarks.csv`;
- paired tests in `audit_artifacts/paired_significance.csv`;
- action attribution in `audit_artifacts/action_attribution.csv`;
- hashes of the frozen local prediction archive;
- documentation of the official split and storage-block interpretation.

The five robustness partitions are equal-sized storage blocks. They are not
advertised as physical channel families. Scoped frozen-artifact verification is
described in `docs/ARTIFACT_AVAILABILITY.md`.

