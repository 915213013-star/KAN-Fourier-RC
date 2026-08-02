# Leakage-Controlled OOF Protocol

## Threat model

An OOF cache is intended to represent predictions on samples not used to fit the
corresponding fold model. If the outer-holdout labels choose an epoch,
checkpoint, seed, or hyperparameter, the cache is no longer independent of those
labels even though the samples were absent from gradient updates.

The public protocol prevents that pathway for neural OOF trainers.

## Fold construction

For every outer fold:

```text
model-training partition
  |-- outer-train
  |     |-- inner-train
  |     `-- inner-selection
  `-- outer-holdout
```

The inner split is deterministic for a fixed configuration and stratified by
class-by-SNR when every stratum supports it. A documented class-only fallback is
used otherwise.

## Selection and refit

1. Train a selection model on inner-train.
2. Use only inner-selection labels for epoch selection and early stopping.
3. Record the selected epoch count; do not retain the selection weights for OOF
   inference.
4. Reinitialize the same architecture from the declared seed.
5. Refit on all outer-train samples for exactly the selected epoch count.
6. Infer outer-holdout once and place predictions at their original indices.

For a declared multi-seed primary, each seed follows the same procedure.
Ensemble weights and member seeds must be fixed globally before outer-fold
training; fold-specific seed maps, initialization checkpoints, and manually
frozen folds are rejected by the public trainer.

## Cache contract

Every compatible cache records:

- `protocol_id = inner_select_outer_refit_v1`;
- a canonical configuration fingerprint;
- split seed and model seed(s);
- SHA-256 digests of outer-train, inner-train, inner-selection, and
  outer-holdout indices;
- selected epoch count and refit horizon;
- class/SNR stratification mode.

The invariant checker verifies that inner partitions are disjoint, both lie
inside outer-train, and outer-train is disjoint from outer-holdout. Missing or
mismatched protocol metadata causes cache reuse to fail.

## Scope of the claim

`Leakage-controlled` describes the explicitly controlled data-flow paths above.
It is not a claim that every possible source of benchmark adaptation, software
nondeterminism, or human experiment selection has been mathematically excluded.
Test labels remain outside all fitting and validation-selection objectives.

