# Executable Reference Methods

This directory exposes compact, runnable definitions of the auxiliary and
decision-level method families used by KAN-Fourier-RC. The interfaces are
dataset-agnostic and share the fail-closed cache schema in `CACHE_SCHEMA.md`.

## Auxiliary evidence

- `fit_hcs_reference.py`: high-order and distributional I/Q descriptors with a
  cross-fitted tree probability expert.
- `fit_gamc_reference.py`: constellation-geometry descriptors with a
  cross-fitted tree probability expert.
- `fit_pairwise_reference.py`: confusion pairs derived inside each fitting
  fold, followed by pair-specific logistic probability specialists.
- `fit_long_spectral_lite_reference.py`: a small depthwise temporal path plus a
  compact spectral path for long I/Q records. Fold checkpoints are resumable.

The non-neural scripts export train OOF probabilities, validation
probabilities, optional held-out probabilities, and a run configuration. Pair
discovery is repeated inside each OOF fitting fold; a held fold does not define
its own pair list.

## Decision-level comparisons

`run_oof_decision_baselines.py` evaluates, from the same candidate pool and
observable probability features:

1. OOF linear stacking;
2. OOF XGBoost stacking;
3. OOF candidate competence;
4. isolated OOF ERU.

`run_validation_frozen_eru_reference.py` fits candidate-specific utility
estimators from train-OOF evidence. Validation labels select the action-score
threshold and maximum change rate. The selected policy is serialized before an
optional held-out cache is evaluated.

These entry points generate fresh run-specific results. The paper's displayed
values are reconciled separately by the immutable aggregate records and
provenance identifiers in `audit_artifacts/`.

## Typical order

1. Produce primary and IQCC fold-OOF, validation, and held-out probability
   caches with stable sample IDs.
2. Fit any desired HCS, pairwise, GAMC-inspired, or long-spectral-lite
   candidates.
3. Merge all candidates independently for train OOF, validation, and held-out
   partitions.
4. Run the matched decision baselines.
5. Fit and freeze the ERU retain-or-correct policy on train OOF plus validation.
6. Evaluate the already-frozen policy on the held-out candidate cache.

The official held-out labels are not an input to fitting or policy-selection
commands. When a held-out cache is supplied, labels are used only to summarize
the frozen predictions.
