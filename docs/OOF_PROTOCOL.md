# Leakage-Controlled OOF Residual-Evidence Protocol

## Purpose

The OOF cache is a model-learning artifact used to train residual and
candidate-utility models. It is not used as an unbiased estimate of final model
accuracy. The protocol controls gradient-fitting leakage and separates neural
model learning, policy selection, and official test evaluation.

## Fold construction and checkpoint selection

For each of `K` deterministic folds of the model-learning partition:

```text
model-learning partition
  |-- fold-train: K-1 folds, used for gradient updates
  `-- fold-holdout: excluded from gradient updates
```

The fold model is trained on `fold-train`. Accuracy on `fold-holdout` is
monitored across epochs to choose the checkpoint and implement early stopping.
That selected checkpoint then predicts `fold-holdout`, and its probabilities
are written to the OOF cache at the original sample indices. Every model-
learning sample is predicted exactly once by a model whose gradient updates did
not include that sample.

Because fold-holdout labels participate in checkpoint selection, this procedure
is not nested cross-validation and the resulting OOF accuracy is not claimed as
an unbiased generalization estimate. It is a cross-fitted feature-construction
protocol for downstream residual reasoning.

## Independent policy and test partitions

The model-learning OOF cache is consumed by residual-evidence learners. A
separate policy-validation partition selects thresholds, candidate masks,
approved transitions, action strengths, and other frozen policy choices. The
official test partition is evaluated only after the predictors and policy have
been fixed. Official test labels are excluded from model fitting, OOF
checkpoint selection, and policy selection.

## Cache contract

Compatible caches record:

- `protocol_id = fold_holdout_checkpoint_selection_v1`;
- a canonical configuration fingerprint;
- split seed and model seed(s);
- SHA-256 digests of fold-train and fold-holdout indices;
- selected checkpoint epoch for every fold;
- class/SNR stratification mode and output-index digest.

The invariant checker verifies fold coverage, disjoint gradient and holdout
indices, stable index restoration, and compatible protocol metadata. Missing or
mismatched metadata causes cache reuse to fail closed.

## Interpretation of "leakage-controlled"

`Leakage-controlled` refers to the explicitly separated data-flow paths:

- no sample contributes gradients to the fold model producing its OOF vector;
- policy-validation labels do not fit the neural predictors;
- official test labels enter no fitting or policy-selection objective.

It does not mean nested cross-validation, nor does it claim that fold-holdout
labels are absent from checkpoint selection. It also does not exclude ordinary
hardware nondeterminism or broader researcher choices made before the frozen
evaluation protocol.
