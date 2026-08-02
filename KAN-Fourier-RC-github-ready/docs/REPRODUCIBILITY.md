# Reproducibility and Provenance

## What can be reproduced from the public package

With legitimately obtained RadioML data, users can inspect and execute the
public neural model definitions and corrected OOF protocol. A generic
cache-based decision interface, reporting utilities, and complexity utility are
also included. The paper-specific non-neural experts and exact frozen policy
orchestration are intentionally outside this selective release.

The compact files in `audit_artifacts/` reproduce numerical accounting checks
for the reported aggregate claims:

- primary and full-system accuracy;
- decision-level baselines;
- changed, rescue, harm, net gain, and conditional utility;
- paired bootstrap intervals, McNemar counts, and Holm-adjusted comparisons;
- final and incremental action-family attribution.
- post-hoc HisarMod2019.1 storage-partition sensitivity, without assigning
  physical-channel identities to those partitions.

## What is not claimed

This repository does not claim that a fresh run on arbitrary hardware will be
bitwise identical to the frozen paper artifacts. Checkpoints, sample-level
probability archives, feature caches, exhaustive search traces, and third-party
dataset derivatives are not public. CUDA kernels, library versions, and
floating-point reduction order can also affect fresh training.

The paper metrics are therefore accompanied by hashes of the authors' frozen
source artifacts. Hashes establish provenance but do not disclose the artifacts.
Specific assets can be checked under controlled academic verification as
described in `ARTIFACT_AVAILABILITY.md`.

## Public OOF protocol version

The public neural OOF entry points use protocol
`inner_select_outer_refit_v1` (`isorfv1` in filenames):

1. fold-local inner selection on outer-train only;
2. fresh initialization after epoch selection;
3. fixed-horizon refit on all outer-train samples;
4. one inference pass over outer-holdout;
5. cache metadata with protocol ID, configuration fingerprint, and index hashes.

This protocol hardens the public implementation against selecting a checkpoint
on the same labels later represented in the OOF cache. Older caches without a
matching protocol fingerprint are intentionally not reused.

## Frozen paper records and current public code

Aggregate paper records and the current public training implementation have
separate provenance. The audit tables identify the frozen paper results. The
current source exposes a conservative, reviewable OOF implementation and does
not assert byte-for-byte regeneration of withheld paper archives. This
distinction avoids silently treating a new protocol run as an old frozen result.

## Recommended environment record

For a new run, retain:

- Python, PyTorch, CUDA, cuDNN, NumPy, scikit-learn, and XGBoost versions;
- GPU model and deterministic-algorithm settings;
- dataset hash and split-index hashes;
- full command line and protocol fingerprint;
- selected epoch per inner split and refit horizon per outer fold;
- output SHA-256 hashes.

Run `scripts/preflight.ps1` before publishing or comparing results.
