# Artifact Availability

The repository combines executable reference source with compact audit records.

| Material | Location or access path |
|---|---|
| RadioML model and OOF source | Repository root and `scripts/` |
| HCS, pairwise, GAMC-inspired, and long-spectral reference source | `reference_methods/` and root entry points |
| OOF decision baselines and validation-frozen ERU | Root entry points and `reference_methods/` |
| Protocol and release tests | `oof_protocol.py`, `tests/`, `tools/` |
| Aggregate claim-audit tables | `audit_artifacts/` |
| Complexity and robustness specifications | `audit_artifacts/` and `scripts/` |
| RadioML and HisarMod datasets | Original dataset providers |
| Runtime checkpoints and caches | Generated or supplied in local ignored directories |

`audit_artifacts/artifact_checksums.csv` lists cryptographic identifiers for
the frozen source archives used in the numerical audit. These identifiers allow
claim-specific provenance checks while respecting the distribution terms of
third-party datasets and derived runtime artifacts.

Editors or reviewers needing a claim-specific verification may contact the
authors with the dataset, comparison, and artifact identifier. Depending on the
applicable data and institutional terms, verification can use a hash check,
aggregate recomputation, or scoped inspection.
