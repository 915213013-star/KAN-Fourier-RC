# Reproducibility and Provenance

## Public verification paths

With legitimately obtained datasets, users can run the public RadioML model and
OOF entry points, fit the reference HCS, pairwise, GAMC-inspired, and
long-spectral-lite candidates, compare matched OOF decision objectives, inspect
the retain-or-correct cache contract, execute protocol tests, and recompute the
aggregate consistency checks. The machine-readable records in
`audit_artifacts/` cover:

- primary, decision-level baseline, and full-system accuracy;
- changed, rescue, harm, net gain, and conditional utility;
- paired bootstrap intervals, McNemar counts, and the fixed Holm family;
- final and incremental action-family attribution;
- HisarMod2019.1 storage-partition sensitivity and impairment diagnostics;
- reportable complexity and frozen-artifact provenance anchors.

## Protocol version

The neural OOF entry points use
`fold_holdout_checkpoint_selection_v1` (`fhcsv1` in cache names):

1. gradient fitting on the complementary `K-1` folds;
2. checkpoint selection and early stopping on the held-out fold;
3. OOF inference on that same held-out fold;
4. restoration of predictions to original model-learning indices;
5. independent validation-only policy selection;
6. frozen official-test evaluation.

The protocol metadata states this behavior directly and does not describe the
cache as nested-CV performance estimation. See `OOF_PROTOCOL.md` for the exact
scope of the leakage-control claim.

## Numerical provenance

Aggregate accuracies and gains are reconciled to correct-sample counts whenever
counts are available. The 15 prespecified paired comparisons record bootstrap
strata, repetitions, McNemar discordant counts, and Holm-adjusted probabilities.
`artifact_checksums.csv` provides SHA-256 identifiers for retained frozen source
archives. A hash identifies the evidence used for an audit without changing the
dataset provider's redistribution terms.

Fresh training need not be bitwise identical across CUDA, cuDNN, PyTorch, and
XGBoost versions. For each new run, retain:

- software and accelerator versions;
- dataset and split-index hashes;
- full command line and protocol fingerprint;
- selected checkpoint epoch for each fold;
- output archive SHA-256 hashes.

Run `scripts/preflight.ps1` before publishing or comparing a release copy.
