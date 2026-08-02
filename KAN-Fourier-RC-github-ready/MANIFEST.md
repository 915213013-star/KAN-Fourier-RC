# Release Manifest

This manifest describes the intentionally selective GitHub package.

## Included

- selected `model_*.py`: public KAN-Fourier and IQCC-Former definitions.
- selected `train_*_oof_*.py`: public neural OOF training entry points.
- `oof_protocol.py`: fold-local selection/refit invariants and metadata.
- `apply_crossfit_candidate_action_router_2016.py`: generic aligned-cache
  reference router; it is not the paper's complete policy implementation.
- `metrics_2016.py`, reporting scripts, and the complexity utility.
- `scripts/`: portable PowerShell launchers and release preflight.
- `tools/preflight_release.py`: static and numerical release checks.
- `tests/test_oof_protocol.py`: protocol isolation tests.
- `audit_artifacts/`: aggregate metrics, paired statistics, action accounting,
  and hashes; no sample-level labels or predictions.
- `docs/`: dataset, protocol, provenance, and release-boundary documentation.
- `experiments/hisarmod2019/README.md`: HisarMod scope and controlled-verification
  statement.
- `GITHUB_UPLOAD_CHECKLIST.md`: whole-repository replacement and preflight
  instructions.

## Intentionally excluded

- datasets and extracted records;
- `.pth`, `.pt`, `.ckpt`, `.joblib`, `.npz`, `.npy`, `.pkl`, and `.dat` files;
- checkpoints, feature caches, probability archives, and sample indices;
- paper-specific non-neural candidate implementations, late-stage policy code,
  and exact orchestration grids;
- exhaustive search logs, exploratory probes, and unpublished model variants;
- manuscript drafts and private transfer bundles.

Empty runtime directories contain only `.gitkeep` placeholders. The preflight
checker fails if restricted binary artifacts or machine-specific absolute paths
are found in a release candidate.
