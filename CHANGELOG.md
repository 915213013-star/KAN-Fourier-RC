# Changelog

## 2026-08-02 - Complete academic reference package

- Added explicit OOF protocol identifiers, configuration fingerprints, index
  digests, and fail-closed cache compatibility checks.
- Documented the implemented fold protocol accurately: gradient fitting uses
  complementary folds, while the corresponding fold holdout selects the
  checkpoint and receives the OOF predictions.
- Kept validation-only policy selection and official held-out evaluation as
  disjoint stages, with official test labels excluded from every fitting and
  policy-selection objective.
- Added aggregate three-dataset metrics, the fixed 15-comparison statistical
  family, mutually exclusive action accounting, partition stability,
  impairment diagnostics, complexity records, and frozen-artifact hashes.
- Added protocol unit tests and a fail-closed release preflight checker.
- Added executable public reference implementations for HCS, pairwise
  confusion, GAMC-inspired geometry, and the long-spectral-lite candidate.
- Added matched OOF linear/XGBoost stacking, candidate-competence, isolated-ERU,
  and validation-frozen ERU interfaces under a shared sample-aligned cache
  contract.
- Added cache-alignment, feature-finiteness, action-budget, and long-sequence
  forward tests to the release preflight.
- Added the KAN-Fourier-RC Academic Evaluation Source License 1.0.

The preflight validates count-derived accuracy, rescue/harm accounting,
bootstrap settings, action-family reconciliation, storage-block terminology,
robustness parameters, complexity conventions, documentation links, protocol
and reference-method tests, and release portability.
