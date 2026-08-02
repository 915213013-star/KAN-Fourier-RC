import torch
import torch.nn as nn
import torch.nn.functional as F

from model_cv_trn_aux_2016 import CVTRNBlock


class IQDenoiseStem(nn.Module):
    """Small identity-initialized residual denoiser for I/Q sequences."""

    def __init__(
        self,
        hidden=8,
        partial_ratio=0.5,
        kernel_size=5,
        cap=0.20,
        gate_bias=-2.0,
    ):
        super().__init__()
        hidden = max(4, int(hidden))
        if hidden % 2:
            hidden += 1
        channels = 2 * hidden
        partial = max(2, int(round(channels * float(partial_ratio))))
        partial = min(channels, partial)
        self.partial = int(partial)
        self.cap = float(cap)
        self.in_proj = nn.Conv1d(2, channels, kernel_size=1)
        self.partial_dw = nn.Conv1d(
            self.partial,
            self.partial,
            kernel_size=int(kernel_size),
            padding=int(kernel_size) // 2,
            groups=self.partial,
        )
        self.mix = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=2, num_channels=channels)
        self.out_proj = nn.Conv1d(channels, 2, kernel_size=1)
        self.gate = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.constant_(self.gate[-2].bias, float(gate_bias))

    def forward(self, x):
        feat = self.in_proj(x)
        proc = self.partial_dw(feat[:, : self.partial, :])
        feat = torch.cat([proc, feat[:, self.partial :, :]], dim=1)
        feat = self.norm(F.silu(self.mix(F.silu(feat))))
        delta = torch.tanh(self.out_proj(feat))
        stats = torch.cat([x.mean(dim=-1), x.std(dim=-1, unbiased=False)], dim=1)
        gate = self.gate(stats).view(-1, 1, 1)
        return x + self.cap * gate * delta


class CVTRNAuxV2_2016(nn.Module):
    """
    CV-TRN auxiliary expert v2 for RML2016.10A.

    Compared with v1, v2 keeps the Fourier main model untouched but makes the
    auxiliary branch closer to the CV-TRN paper:
      - I/Q streams share frame embedding.
      - I/Q streams interact through complex-valued attention blocks.
      - Training can supervise I-head, Q-head, and fused-head simultaneously.
    """

    def __init__(
        self,
        num_classes=11,
        dim=64,
        depth=3,
        num_heads=4,
        frame_len=16,
        stride=8,
        seq_len=128,
        dropout=0.10,
        input_denoise=False,
        denoise_hidden=8,
        denoise_partial_ratio=0.5,
        denoise_kernel=5,
        denoise_cap=0.20,
        denoise_gate_bias=-2.0,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.dim = int(dim)
        self.frame_len = int(frame_len)
        self.stride = int(stride)
        self.input_denoise = (
            IQDenoiseStem(
                hidden=denoise_hidden,
                partial_ratio=denoise_partial_ratio,
                kernel_size=denoise_kernel,
                cap=denoise_cap,
                gate_bias=denoise_gate_bias,
            )
            if input_denoise
            else None
        )

        n_frames = (int(seq_len) - self.frame_len) // self.stride + 1
        max_tokens = n_frames + 1

        self.frame_embed = nn.Conv1d(1, dim, kernel_size=self.frame_len, stride=self.stride, bias=True)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.input_norm = nn.LayerNorm(dim)

        self.blocks = nn.ModuleList(
            [CVTRNBlock(dim=dim, num_heads=num_heads, dropout=dropout, max_tokens=max_tokens) for _ in range(depth)]
        )
        self.norm_i = nn.LayerNorm(dim)
        self.norm_q = nn.LayerNorm(dim)

        self.head_i = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )
        self.head_q = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )
        self.head_fused = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, num_classes),
        )
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x, return_aux=False, return_embedding=False):
        if x.dim() != 3 or x.size(1) != 2:
            raise ValueError(f"CVTRNAuxV2_2016 expects [B,2,T], got {tuple(x.shape)}")
        if self.input_denoise is not None:
            x = self.input_denoise(x)

        xi = self.frame_embed(x[:, 0:1, :]).transpose(1, 2)
        xq = self.frame_embed(x[:, 1:2, :]).transpose(1, 2)

        cls = self.cls.expand(x.size(0), -1, -1)
        xi = self.input_norm(torch.cat([cls, xi], dim=1))
        xq = self.input_norm(torch.cat([cls, xq], dim=1))

        for block in self.blocks:
            xi, xq = block(xi, xq)

        ci = self.norm_i(xi[:, 0])
        cq = self.norm_q(xq[:, 0])
        mag = torch.sqrt(ci * ci + cq * cq + 1e-8)
        phase_like = ci * cq
        emb = torch.cat([ci, cq, mag, phase_like], dim=1)

        logits_i = self.head_i(ci)
        logits_q = self.head_q(cq)
        logits_fused = self.head_fused(emb)

        if return_aux:
            return {
                "logits": logits_fused,
                "logits_i": logits_i,
                "logits_q": logits_q,
                "embedding": emb,
            }
        if return_embedding:
            return emb, logits_fused
        return logits_fused


def build_cv_trn_aux_v2_model(
    device,
    num_classes=11,
    dim=64,
    depth=3,
    num_heads=4,
    frame_len=16,
    stride=8,
    dropout=0.10,
    input_denoise=False,
    denoise_hidden=8,
    denoise_partial_ratio=0.5,
    denoise_kernel=5,
    denoise_cap=0.20,
    denoise_gate_bias=-2.0,
):
    model = CVTRNAuxV2_2016(
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        num_heads=num_heads,
        frame_len=frame_len,
        stride=stride,
        dropout=dropout,
        input_denoise=input_denoise,
        denoise_hidden=denoise_hidden,
        denoise_partial_ratio=denoise_partial_ratio,
        denoise_kernel=denoise_kernel,
        denoise_cap=denoise_cap,
        denoise_gate_bias=denoise_gate_bias,
    )
    return model.to(device)
