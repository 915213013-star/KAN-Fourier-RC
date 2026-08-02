# Artifact Availability

The project uses a tiered release to support verification without redistributing
third-party data or publishing the authors' complete experimental know-how.

| Tier | Material | Availability |
|---|---|---|
| Public source | Selected RadioML neural models, corrected OOF protocol, generic candidate-cache interface, tests | This repository |
| Public audit | Aggregate metrics, paired statistics, action and partition accounting, SHA-256 provenance anchors | `audit_artifacts/` |
| User supplied | RadioML and HisarMod datasets | Obtain from dataset providers |
| Controlled verification | Claim-specific frozen checkpoints, aligned sample-level predictions, or supervised recomputation | Editors/reviewers on a scoped academic request, subject to dataset, institutional, and intellectual-property constraints |
| Not distributed | Exhaustive exploratory grids, abandoned probes, unrelated workspace files | Withheld |

## Request principles

A controlled request should identify the paper claim and artifact needed for
verification. The authors may provide a hash check, supervised screen-share,
aggregate recomputation, or a time-limited private package instead of unrestricted
download. Dataset licenses and participant institutions may further limit what
can be transferred.

No public document states that checkpoints or prediction archives are included.
Available provenance anchors are listed in
`audit_artifacts/artifact_checksums.csv`; a hash authenticates a retained asset
but does not make that asset publicly downloadable.
