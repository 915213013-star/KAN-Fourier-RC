# GitHub Repository Replacement Checklist

This archive is a complete replacement tree for the selective public research
release. It is not an archive of the private experimental workspace.

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

The check must report five passing OOF protocol tests and no release-boundary
violations.

## Commit scope

The replacement commit should contain deletions of obsolete public files as
well as additions and modifications from this archive. A suitable commit title
is:

```text
Harden public OOF protocol and clarify release boundary
```

Do not upload datasets, checkpoints, feature caches, sample-level prediction
archives, local logs, manuscript drafts, or private orchestration files.

