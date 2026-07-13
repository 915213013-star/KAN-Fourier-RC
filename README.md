# KAN-Fourier-RC

Research code for automatic modulation classification (AMC) with a structured
KAN-Fourier primary classifier and out-of-fold (OOF) expected residual utility
(ERU) decision correction.

The release covers the core implementation used on RadioML2016.10A and
RadioML2016.10B: KAN-Fourier training, OOF probability generation, the
IQCC-Former auxiliary predictor, HCS/Pairwise/GAMC-inspired non-neural
predictors, ERU routing, validation-only policy selection, complexity reporting,
and paper-figure generation.

## Reference results

| Dataset | Configuration | Overall accuracy |
|---|---|---:|
| RML2016.10A | KAN-Fourier-RC, split seed 1 | 66.332% |
| RML2016.10B | KAN-Fourier-RC, split seed 1 | 66.168% |

These values identify the frozen paper configurations. Dataset files,
checkpoints, feature caches, and prediction archives are not redistributed.
Exact regeneration therefore requires the original datasets and the same cache
alignment described in `docs/REPRODUCIBILITY.md`.

## Scope of this repository

Included:

- KAN-Fourier model and component-ablation variants;
- full-train and train-split OOF training code;
- IQCC-Former and classical auxiliary predictors;
- ERU meta-features, residual utility estimators, risk guards, and stable action selection;
- result comparison, complexity, visualization, and figure-generation utilities;
- portable PowerShell entry points in `scripts/`.

Excluded:

- RadioML datasets and derived feature caches;
- trained checkpoints and prediction files;
- exploratory probes, private transfer bundles, logs, and manuscript drafts.

Legacy Python filenames retain the word `compressed` to preserve traceability
to the experiment history. In the paper and this README, the released
`full_geo_2expert` architecture is simply called **KAN-Fourier**.

## Installation

Python 3.10 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File scripts\preflight.ps1 -SkipDataCheck
```

For CUDA, install the PyTorch build appropriate for the local driver before
installing the remaining requirements.

## Data layout

Supply the datasets yourself:

```text
KAN-Fourier-RC/
  raw_data/RML2016.10a_dict.pkl
  data/RML2016.10b.dat
  data/RML2016.10BGAMC/          # optional extracted 10B helper layout
  feature_cache/                 # generated locally
  checkpoints/                  # generated locally
  results/                      # generated locally
```

See `docs/DATASETS.md` for the expected formats and licensing note.
See `docs/PUBLISHING.md` for the final GitHub checklist.

## Quick start

Train the released KAN-Fourier primary classifier. Rerunning either command
resumes from its `latest_*.pth` checkpoint unless `-ForceRestart` is supplied.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_primary_10a.ps1
powershell -ExecutionPolicy Bypass -File scripts\train_primary_10b.ps1
```

The OOF and auxiliary scripts are exposed directly, for example:

```powershell
python -u train_fourier_compressed_main_oof_2016.py --help
python -u train_cv_trn_aux_v2_oof_2016.py --help
python -u train_hcs_aux_oof_2016.py --help
python -u train_pairwise_confusion_aux_2016.py --help
python -u train_gamc_oof_tree_experts_2016.py --help
```

After the aligned probability caches are available, use the cache-based RC
entry points:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_rc_10a_from_caches.ps1 -Help
powershell -ExecutionPolicy Bypass -File scripts\run_rc_10b_from_caches.ps1 -PreflightOnly
```

The 10B launcher implements the frozen three-stage route used for the formal
experiment and expects the cache filenames documented inside the script. Run it
with `-SkipExisting` to reuse completed stages after an interruption. The 10A
launcher is intentionally parameterized so only explicitly supplied, aligned
candidate caches participate in the final stability selection.

## Evaluation protocol

All trainable models, thresholds, blending coefficients, and routing actions are
optimized using only train-split OOF predictions and validation data. Test labels
are excluded from every optimization objective and are used only to compute the
frozen final metrics.

## Authors

- Linzhuo Han, School of Mathematical Sciences, University of Electronic Science and Technology of China
- Zongyong Cui, School of Information and Communication Engineering, University of Electronic Science and Technology of China
- Houbiao Li, School of Mathematical Sciences, University of Electronic Science and Technology of China

## License

No open-source license is asserted by this preparation bundle. Before making the
repository public, choose a license and replace `LICENSE_PENDING.md`. Dataset
licenses remain separate from the code license.
