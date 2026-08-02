"""Numerical-stability helpers shared by the public KAN-Fourier trainers."""

import types

import torch


def stable_safe_log_euclidean_map(self, spd_matrix):
    """Compute a stable batched matrix logarithm for SPD-like inputs.

    The retry ladder handles repeated or poorly conditioned eigenvalues on GPU.
    It preserves the original computation when the first eigendecomposition is
    well conditioned and uses progressively conservative fallbacks otherwise.
    """
    if spd_matrix.dim() != 3:
        raise ValueError(f"spd_matrix must have shape [B, C, C], got {spd_matrix.shape}")

    batch_size, channels, _ = spd_matrix.shape
    device = spd_matrix.device
    dtype = spd_matrix.dtype
    spd = 0.5 * (spd_matrix + spd_matrix.transpose(1, 2))
    spd = torch.nan_to_num(spd, nan=0.0, posinf=1e4, neginf=-1e4)

    eye = torch.eye(channels, device=device, dtype=dtype).unsqueeze(0)
    ramp = torch.linspace(1.0, 2.0, channels, device=device, dtype=dtype)
    ramp_diag = torch.diag(ramp).unsqueeze(0)
    min_eig = 1e-5
    jitter_levels = (
        0.0,
        1e-7,
        3e-7,
        1e-6,
        3e-6,
        1e-5,
        3e-5,
        1e-4,
        3e-4,
        1e-3,
        3e-3,
        1e-2,
    )

    for epsilon in jitter_levels:
        try:
            matrix = spd if epsilon == 0.0 else spd + epsilon * (ramp_diag + eye)
            eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
            log_values = torch.log(eigenvalues.clamp(min=min_eig))
            log_cov = eigenvectors @ torch.diag_embed(log_values) @ eigenvectors.transpose(1, 2)
            if torch.isfinite(log_cov).all():
                return log_cov
        except (RuntimeError, torch.linalg.LinAlgError):
            continue

    outputs = []
    sample_eye = torch.eye(channels, device=device, dtype=dtype)
    sample_ramp = torch.diag(ramp)
    for index in range(batch_size):
        sample = spd[index]
        output = None
        for epsilon in jitter_levels:
            try:
                matrix = sample if epsilon == 0.0 else sample + epsilon * (sample_ramp + sample_eye)
                matrix = 0.5 * (matrix + matrix.transpose(0, 1))
                eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
                output = (
                    eigenvectors
                    @ torch.diag(torch.log(eigenvalues.clamp(min=min_eig)))
                    @ eigenvectors.transpose(0, 1)
                )
                if torch.isfinite(output).all():
                    break
                output = None
            except (RuntimeError, torch.linalg.LinAlgError):
                output = None

        if output is None:
            try:
                matrix = sample + 1e-3 * (sample_ramp + sample_eye)
                matrix = 0.5 * (matrix + matrix.transpose(0, 1))
                u, singular_values, _ = torch.linalg.svd(matrix)
                output = (
                    u
                    @ torch.diag(torch.log(singular_values.clamp(min=min_eig)))
                    @ u.transpose(0, 1)
                )
                if not torch.isfinite(output).all():
                    output = None
            except (RuntimeError, torch.linalg.LinAlgError):
                output = None

        if output is None:
            diagonal = torch.diagonal(sample, dim1=0, dim2=1).clamp(min=min_eig)
            output = torch.diag(torch.log(diagonal))
        outputs.append(output)

    return torch.stack(outputs, dim=0)


def patch_lieqkan_stability(model, name="model"):
    """Install the stable SPD map when a model exposes a Lie encoder."""
    lie_encoder = getattr(model, "lie_encoder", None)
    if lie_encoder is None or not hasattr(lie_encoder, "safe_log_euclidean_map"):
        return
    lie_encoder.safe_log_euclidean_map = types.MethodType(
        stable_safe_log_euclidean_map,
        lie_encoder,
    )
    print(f"[*] Installed stable SPD log map for {name}.")
