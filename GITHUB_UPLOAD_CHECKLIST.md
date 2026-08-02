# GitHub Repository Replacement Checklist

This archive is the complete replacement tree for the GitHub academic reference
release.

## Before uploading

1. Preserve the current repository state with a Git tag or backup branch.
2. Remove obsolete tracked files from the repository working tree, but keep the
   local `.git` directory.
3. Extract this archive locally.
4. Copy the contents of `KAN-Fourier-RC-github-ready/` into the repository root.
5. Confirm that `.gitignore`, `.gitkeep`, `LICENSE`, `README.md`, and
   `CITATION.cff` are present.

## Required checks

Run the release preflight before committing:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\preflight.ps1 -SkipDataCheck
```

The check must report all discovered protocol and reference-method tests as
passing and no release-boundary violations.

## Commit scope

The replacement commit should contain deletions of obsolete public files as
well as additions and modifications from this archive. A suitable commit title
is:

```text
Add executable reference methods and numerical audit artifacts
```

Review the staged tree once more and confirm that the release preflight passes
on the exact files to be committed.
