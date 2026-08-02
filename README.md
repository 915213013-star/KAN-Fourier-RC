# KAN-Fourier-RC

Academic reference implementation and claim-audit package for
**primary-preserving residual decision correction** in automatic modulation
classification (AMC).

KAN-Fourier-RC treats inference as a retain-or-correct decision. A structured
KAN-Fourier classifier supplies the default probability vector, heterogeneous
predictors provide candidate evidence, and an out-of-fold (OOF) residual model
estimates candidate-specific rescue and harm evidence. A policy selected on a
separate validation partition freezes the permitted actions before held-out
evaluation.

## Scientific scope

The repository provides the RadioML model definitions and neural OOF training
paths, executable reference implementations of the HCS, pairwise,
GAMC-inspired, and long-spectral-lite auxiliary families, four matched OOF
decision baselines, a validation-frozen ERU policy interface, protocol tests,
reporting tools, and machine-readable audit records supporting the paper's
principal numerical claims. The audit bundle covers all three evaluated
datasets and includes decision-level baselines, paired significance, mutually
exclusive action accounting, storage-partition sensitivity, complexity
records, robustness diagnostics, and provenance hashes.

Third-party datasets are obtained from their original providers. Runtime
products such as checkpoints and caches are generated or supplied locally and
are ignored by Git. Frozen numerical claims are linked to their configurations,
aggregate records, and provenance hashes in `audit_artifacts/`.

## Reference results

The following values identify the frozen configurations reported in the paper.
Displayed accuracies are rounded independently; percentage-point gains are
computed from the underlying correct-sample counts.

| Dataset | Primary | Full KAN-Fourier-RC | Gain |
|---|---:|---:|---:|
| RML2016.10A | 63.632% | 66.332% | +2.700 pp |
| RML2016.10B | 65.161% | 66.168% | +1.008 pp |
| HisarMod2019.1 | 77.769% | 79.867% | +2.098 pp |

The corresponding counts and matched decision-level comparisons are in
[`audit_artifacts/`](audit_artifacts/). Aggregate records are provided for
claim auditing and do not replace the dataset-specific training protocol.

## Leakage-controlled OOF residual evidence

For each fold of the model-learning partition, the public neural OOF trainers:

1. fit model parameters using only the complementary `K-1` folds;
2. monitor the held-out fold to select that fold model's checkpoint and early
   stopping point;
3. use the selected checkpoint to write probabilities for that held-out fold;
4. restore all fold predictions to their original model-learning indices.

Thus, a sample written to the OOF cache is excluded from the gradient updates
of the model that predicts it. The fold holdout is nevertheless used for
checkpoint selection, so this protocol is not presented as nested cross-
validation or as an unbiased performance estimator. Its purpose is to construct
cross-fitted residual evidence for a downstream policy. Policy selection uses a
separate validation partition, and the official test labels are excluded from
all fitting and policy-selection objectives. See
[`docs/OOF_PROTOCOL.md`](docs/OOF_PROTOCOL.md).

## Installation

Python 3.10 is recommended. Install the PyTorch build appropriate for the local
CUDA driver first, then install the remaining dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\preflight.ps1 -SkipDataCheck
```

## Dataset layout

```text
KAN-Fourier-RC/
  raw_data/RML2016.10a_dict.pkl
  data/RML2016.10b.dat
  feature_cache/     # generated locally; ignored by Git
  checkpoints/      # generated locally; ignored by Git
  results/          # generated locally; ignored by Git
```

Dataset acquisition and split notes are in
[`docs/DATASETS.md`](docs/DATASETS.md). The neural launchers cover
RML2016.10A and RML2016.10B. The dataset-agnostic reference interfaces operate
on aligned I/Q and probability caches, and the long-spectral-lite candidate
accepts arbitrary I/Q sequence lengths, including length 1024. The
HisarMod2019.1 audit and stress-test records use the official storage split
described in the paper.

## Training entry points

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_primary_10a.ps1
powershell -ExecutionPolicy Bypass -File scripts\train_primary_10b.ps1
```

Inspect the OOF interfaces before launching a long run:

```powershell
python -u train_fourier_compressed_main_oof_2016.py --help
python -u train_fourier_compressed_main_oof_10b.py --help
python -u train_cv_trn_aux_v2_oof_2016.py --help
python -u train_cv_trn_aux_v2_oof_10b.py --help
```

Once aligned caches exist, the generic 10A retain-or-correct interface can be
inspected through:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_rc_10a_from_caches.ps1 -Help
```

Legacy filenames containing `compressed` are retained for experiment
traceability. The released `full_geo_2expert` architecture is called
**KAN-Fourier** in the paper.

## Reference auxiliary and decision methods

The following entry points expose the public method families under one checked
cache contract:

```powershell
python fit_hcs_reference.py --help
python fit_pairwise_reference.py --help
python fit_gamc_reference.py --help
python fit_long_spectral_lite_reference.py --help
python merge_candidate_caches.py --help
python run_oof_decision_baselines.py --help
python run_validation_frozen_eru_reference.py --help
```

The three non-neural expert scripts perform fold-level fitting and export OOF,
validation, and optional test probabilities. The long-spectral-lite script
provides resumable fold training and a compact temporal/spectral candidate for
long I/Q records. `run_oof_decision_baselines.py` evaluates OOF linear stacking,
OOF XGBoost stacking, OOF candidate competence, and isolated OOF ERU from the
same candidate pool. `run_validation_frozen_eru_reference.py` fits
candidate-specific rescue-minus-harm utilities on OOF data and selects the
retain-or-correct threshold and change budget on validation data before any
optional held-out inference.

All probability blocks are aligned by explicit `sample_ids`; label or class
dimension mismatches fail closed. See
[`docs/CACHE_SCHEMA.md`](docs/CACHE_SCHEMA.md) and
[`docs/REFERENCE_METHODS.md`](docs/REFERENCE_METHODS.md) for the executable
workflow. Fresh runs write run-specific configurations and predictions, while
the rounded paper values remain tied to the frozen audit records in
`audit_artifacts/`.

## Audit and verification

```powershell
python measure_reportable_model_complexity.py
python scripts\recompute_paired_significance.py --verify-aggregate
python scripts\recompute_impairment_reference.py --verify-aggregate
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\preflight.ps1 -SkipDataCheck
```

The preflight compiles every public Python source, runs protocol unit tests,
reconciles count-derived metrics, verifies the fixed 15-comparison Holm family,
checks action accounting and robustness metadata, and rejects machine-specific
paths or runtime binaries in a release candidate. Fresh training can vary with
hardware and numerical-library versions; each new run should retain its own
configuration fingerprint and output hashes.

## Citation and license

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). Source code is
released under the
[`KAN-Fourier-RC Academic Evaluation Source License 1.0`](LICENSE), which permits
non-commercial academic evaluation and private research modification under its
stated terms. Dataset licenses remain independent of this source license.
