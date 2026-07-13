# Publishing on GitHub

1. Review `LICENSE_PENDING.md` and add the license selected by the authors or institution.
2. Create an empty GitHub repository named `KAN-Fourier-RC` without a generated README.
3. Extract the release ZIP locally and upload the contents of the extracted folder, not the ZIP itself.
4. Confirm that no dataset, checkpoint, cache, or prediction archive appears in the commit.
5. Replace any placeholder paper or repository URL after the manuscript and repository locations are final.
6. Create a tagged release such as `v1.0.0` only after the public repository passes `scripts/preflight.ps1`.

The prepared package is intentionally small enough for either GitHub's web
uploader or a normal Git client. Large research artifacts should be distributed
through an institutional archive or an explicitly licensed data service, not
committed to the source repository.

