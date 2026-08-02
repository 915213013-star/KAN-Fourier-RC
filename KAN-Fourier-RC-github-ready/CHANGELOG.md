# Changelog

## 2026-08-02 - Public protocol hardening

- Replaced outer-holdout checkpoint selection in public neural OOF trainers with
  inner selection followed by fresh outer-train refitting.
- Added protocol IDs, configuration fingerprints, index digests, and fail-closed
  cache compatibility checks.
- Removed executable fold-specific seed maps, external initialization
  checkpoints, resume-reset selection, and manual good-fold freezing from the
  compressed OOF public interface.
- Added a source-available academic evaluation license and removed the pending
  license marker.
- Reframed the repository as a selective reference release and documented the
  public, controlled, and withheld artifact tiers.
- Added aggregate three-dataset metrics, paired statistics, action accounting,
  and provenance hashes without publishing sample-level predictions.
- Added the pre-specified 15-comparison Holm family and the five-block
  HisarMod2019.1 post-hoc partition-stability record to the machine-readable
  audit bundle.
- Removed paper-specific non-neural expert training, late-route evaluation,
  private plotting, and exploratory orchestration files from the public
  package; the retained router is a generic aligned-cache reference path.
- Added protocol unit tests and a release preflight checker.

The preflight checker now verifies the exact comparison family, count-derived
accuracy differences, bootstrap settings, action accounting, partition labels,
documentation links, restricted artifacts, and protocol isolation tests.

The frozen paper metrics remain provenance records. Newly generated caches use a
new protocol tag and are not represented as byte-identical paper archives.
