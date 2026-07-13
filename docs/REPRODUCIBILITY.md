# Reproducibility notes

## Frozen split protocol

The paper uses split seed 1. Primary and auxiliary predictors produce aligned
train, validation, and test probability arrays. Train-split predictions used by
the ERU estimator are out-of-fold: each training sample is predicted by a model
that did not train on that sample.

The validation split selects thresholds, blending coefficients, transition
sets, and routing actions. Test labels are excluded from all optimization
objectives.

## Reference configurations

- RML2016.10A primary seed: 261; architecture: `full_geo_2expert`.
- RML2016.10B primary seed: 361; architecture: `full_geo_2expert`.
- RML2016.10A frozen KAN-Fourier-RC result: 66.332%.
- RML2016.10B frozen KAN-Fourier-RC result: 66.168%.

The 10A training launcher exposes an optional warm-start checkpoint because the
paper run was developed through staged training. Pretrained weights are not
bundled. A from-scratch run is scientifically valid but is not guaranteed to
reproduce the exact reported number bit for bit.

## Cache alignment

Do not combine independently shuffled probability files. Each cache must share
the same split seed, label order, modulation order, validation indices, and test
indices. Keep alignment checks enabled whenever an alignment cache is available.

## Resume behavior

Neural training writes both `best_*.pth` and `latest_*.pth`. Re-running the same
command resumes optimizer and scheduler state from `latest_*.pth`; use
`-ForceRestart` only when intentionally starting a new run.

Classical estimators are fingerprint-cached under `results/model_cache/`.
Cache-based pipelines can be rerun with `-SkipExisting` after interruption.

## Naming

The paper name is KAN-Fourier. Historical source filenames containing
`compressed` refer to the same released `full_geo_2expert` architecture and are
retained to preserve experiment traceability.

