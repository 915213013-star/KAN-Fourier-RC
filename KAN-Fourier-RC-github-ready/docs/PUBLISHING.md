# Publishing Checklist

Run from the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\preflight.ps1 -SkipDataCheck
```

The command must pass before a GitHub upload. It checks:

- Python imports and compilation of every public neural OOF entry point;
- OOF protocol unit tests and fail-closed cache fingerprints;
- internal conservation of aggregate audit tables;
- README/document links;
- absence of `LICENSE_PENDING` markers;
- absence of private binary artifacts and machine-specific absolute paths.

Before publication, also verify manually:

1. `LICENSE` and `CITATION.cff` are present.
2. Empty runtime directories contain only `.gitkeep`.
3. No dataset, checkpoint, sample-level prediction, index, feature cache, log,
   manuscript draft, or transfer archive is staged.
4. Git history does not contain a previously staged private artifact. If one
   exists, remove it from history transparently before publication and rotate
   any exposed credentials; never rely on a later deletion commit.
5. Aggregate metrics match `audit_artifacts/reported_metrics.csv`.
6. Any controlled-verification promise can actually be fulfilled by the authors.

This package should be described as a selective academic reference release, not
as an unrestricted open-source or complete experiment dump.

