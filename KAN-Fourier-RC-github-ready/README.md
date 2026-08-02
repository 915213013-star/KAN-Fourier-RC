# KAN-Fourier-RC

Selective academic reference implementation for **primary-preserving residual
decision correction** in automatic modulation classification (AMC).

KAN-Fourier-RC treats inference as a retain-or-correct decision. A structured
KAN-Fourier classifier supplies the default probability vector, heterogeneous
predictors supply candidate evidence, and an out-of-fold (OOF) residual model
estimates whether a candidate action is likely to rescue or harm the primary
decision. A policy selected on a separate validation partition then freezes the
allowed actions before held-out evaluation.

## Public release boundary

This repository is a **selective source-available research release**, not a
complete dump of the authors' experimental workspace.

Publicly included:

- the KAN-Fourier and IQCC-Former model definitions used by the RadioML paths;
- leakage-controlled OOF trainers with inner epoch selection and fresh
  outer-fold refitting;
- a generic aligned-candidate action interface and a conservative reference
  router for inspecting the retain-or-correct protocol;
- reporting and complexity utilities needed to inspect the public paths;
- aggregate three-dataset metrics, paired statistics, action audits, and
  storage-partition sensitivity records with cryptographic provenance anchors;
- release tests and a fail-closed preflight checker.

Not publicly redistributed:

- third-party datasets;
- trained checkpoints, sample-level probability archives, and feature caches;
- the paper-specific HCS, pairwise-confusion, GAMC, and late-route engineering
  implementations and their exhaustive policy grids;
- private experiment logs and transfer bundles;
- exhaustive exploratory grids and unpublished engineering probes.

The omitted files contain third-party data derivatives, large binary artifacts,
or implementation know-how beyond the reference release. Editors and reviewers
may request controlled academic verification of specific frozen artifacts. See
[`docs/ARTIFACT_AVAILABILITY.md`](docs/ARTIFACT_AVAILABILITY.md) and
[`docs/RELEASE_BOUNDARY.md`](docs/RELEASE_BOUNDARY.md).

## Reference results

The following are the frozen paper configurations. Accuracies are display-rounded;
percentage-point gains are computed from the underlying sample counts.

| Dataset | Primary | Full KAN-Fourier-RC | Gain |
|---|---:|---:|---:|
| RML2016.10A | 63.632% | 66.332% | +2.700 pp |
| RML2016.10B | 65.161% | 66.168% | +1.008 pp |
| HisarMod2019.1 | 77.769% | 79.867% | +2.098 pp |

The machine-readable decision baselines, rescue/harm counts, confidence
intervals, McNemar statistics, and action-family accounting are in
[`audit_artifacts/`](audit_artifacts/). These compact tables are intended for
claim verification; they are not substitutes for the withheld sample-level
archives.

## OOF protocol in this release

Each public neural OOF trainer now uses the following fold-local protocol:

1. split the model-training partition into outer-train and outer-holdout;
2. split outer-train into inner-train and inner-selection using class-by-SNR
   stratification when feasible;
3. select the training horizon and early stopping point on inner-selection only;
4. discard the selection weights and initialize a fresh model;
5. refit for the selected number of epochs on all outer-train samples;
6. infer the outer-holdout once and write those predictions to the OOF cache.

Outer-holdout labels do not select epochs, checkpoints, seeds, or
hyperparameters. Caches include a protocol identifier, configuration
fingerprint, and index digests; incompatible legacy caches fail closed. This is
the operational meaning of leakage-controlled OOF in the public implementation.
Detailed invariants are documented in [`docs/OOF_PROTOCOL.md`](docs/OOF_PROTOCOL.md).

## Installation

Python 3.10 is recommended. Install the PyTorch build suitable for the local
CUDA driver first, then install the remaining dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\preflight.ps1 -SkipDataCheck
```

## Dataset layout

Dataset files must be obtained from their respective providers and are never
covered by this repository's license.

```text
KAN-Fourier-RC/
  raw_data/RML2016.10a_dict.pkl
  data/RML2016.10b.dat
  feature_cache/     # generated locally; ignored by Git
  checkpoints/      # generated locally; ignored by Git
  results/          # generated locally; ignored by Git
```

The public executable training paths cover RML2016.10A and RML2016.10B. The
HisarMod2019.1 public material is limited to dataset/protocol documentation and
aggregate audit records because the original dataset and its derived artifacts
cannot be redistributed in this package. See [`docs/DATASETS.md`](docs/DATASETS.md).

## Training entry points

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_primary_10a.ps1
powershell -ExecutionPolicy Bypass -File scripts\train_primary_10b.ps1
```

Inspect the corrected OOF interfaces before launching a long run:

```powershell
python -u train_fourier_compressed_main_oof_2016.py --help
python -u train_fourier_compressed_main_oof_10b.py --help
python -u train_cv_trn_aux_v2_oof_2016.py --help
python -u train_cv_trn_aux_v2_oof_10b.py --help
```

Once aligned caches exist, the generic 10A reference router can be inspected
through:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_rc_10a_from_caches.ps1 -Help
```

This launcher demonstrates the frozen-cache interface; it is not the private
paper orchestration or an exhaustive reproduction grid.

Legacy filenames containing `compressed` are retained for traceability. The
released `full_geo_2expert` architecture is called **KAN-Fourier** in the paper.

## Reproducibility and provenance

Run the release checker before using or publishing a copy:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\preflight.ps1 -SkipDataCheck
```

The checker compiles the public OOF entry points, runs protocol unit tests,
validates aggregate accounting, rejects private binary artifacts and absolute
local paths, and checks documentation links. Exact byte-for-byte regeneration
of the paper archives is not claimed without the controlled frozen assets.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). Please cite the
associated paper when using the code or audit records.

## License

The source is available under the
[`KAN-Fourier-RC Academic Evaluation Source License 1.0`](LICENSE). It permits
non-commercial academic evaluation and private research modification, but it is
not an OSI-approved open-source license and does not permit redistribution of
the source or derived public repositories without written permission. Viewing
and forking through GitHub's built-in functionality remain subject to GitHub's
Terms of Service; those platform rights do not expand the permitted research
uses stated in `LICENSE`.
