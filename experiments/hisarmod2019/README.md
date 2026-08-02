# HisarMod2019.1 Experiment Scope

The paper's HisarMod2019.1 configuration uses length-1024 I/Q records, the
official 520,000/260,000 storage split, a KAN-Fourier primary, IQCC-Former,
non-neural candidates, and a long-spectral lightweight candidate. The frozen
full system reports 79.866923% versus 77.769231% for the primary.

The academic release records the configuration and the aggregate evidence used
to audit this experiment:

- aggregate decision baselines in `audit_artifacts/decision_level_benchmarks.csv`;
- paired tests in `audit_artifacts/paired_significance.csv`;
- action attribution in `audit_artifacts/action_attribution.csv`;
- the frozen-archive SHA-256 anchor in `audit_artifacts/artifact_checksums.csv`;
- the exact synthetic-impairment settings and aggregate paired-bootstrap
  results in `audit_artifacts/impairment_config.json` and
  `audit_artifacts/impairment_robustness.csv`;
- documentation of the official split and storage-block interpretation.

The five robustness partitions are equal-sized storage blocks. They are not
advertised as physical channel families. The impairment experiment is a
post-hoc frozen-policy diagnostic: neither its conditions nor its labels are
used to refit the policy. Controlled frozen-artifact verification is described
in `docs/ARTIFACT_AVAILABILITY.md`.
