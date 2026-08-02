# Release Manifest

This package is the complete public academic reference distribution for the
KAN-Fourier-RC repository.

## Source and execution interfaces

- KAN-Fourier and IQCC-Former RadioML model definitions.
- RML2016.10A and RML2016.10B neural OOF training entry points.
- `oof_protocol.py` protocol metadata and partition invariants.
- Generic aligned-candidate retain-or-correct interface.
- Executable HCS, pairwise-confusion, and GAMC-inspired non-neural reference
  experts with fold-level OOF export.
- Resumable long-spectral-lite OOF candidate for long I/Q records.
- Matched OOF linear stacking, XGBoost stacking, candidate competence, and
  isolated ERU baselines.
- Validation-frozen candidate-specific ERU policy selection.
- Reporting and Table-VI complexity verification utilities.
- Portable PowerShell launchers, unit tests, and release preflight.

## Claim-audit records

`audit_artifacts/` contains compact machine-readable records for headline
metrics, matched decision baselines, paired significance, final and incremental
action attribution, HisarMod partition sensitivity, impairment robustness,
complexity, and provenance hashes.

## Documentation

`docs/` records the dataset layout, implemented OOF protocol, cache schema,
reference-method workflow, reproducibility contract, artifact provenance,
publishing steps, and release scope. Empty runtime directories contain
`.gitkeep` placeholders and are populated locally when the corresponding data
and experiments are run.

The preflight rejects runtime model/data binaries and machine-specific absolute
paths so that the package remains portable and consistent with the documented
academic release.
