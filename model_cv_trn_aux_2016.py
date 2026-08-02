import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativePositionBias1D(nn.Module):
    def __init__(self, num_heads, max_tokens=32):
        super().__init__()
        self.num_heads = int(num_heads)
        self.max_tokens = int(max_tokens)
        self.table = nn.Parameter(torch.zeros(self.num_heads, 2 * self.max_tokens - 1))
        nn.init.trunc_normal_(self.table, std=0.02)

    def forward(self, n_tokens):
        n_tokens = int(n_tokens)
        if n_tokens > self.max_tokens:
            raise ValueError(f"n_tokens={n_tokens} exceeds max_tokens={self.max_tokens}")
        pos = torch.arange(n_tokens, device=self.table.device)
        rel = pos[:, None] - pos[None, :] + self.max_tokens - 1
        return self.table[:, rel].unsqueeze(0)


class DBGLU(nn.Module):
    def __init__(self, dim, hidden_mult=2.0, dropout=0.10):
        super().__init__()
        hidden = int(round(dim * hidden_mult))
        self.fc_a = nn.Linear(dim, hidden)
        self.fc_b = nn.Linear(dim, hidden)
        self.fc_out = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = F.gelu(self.fc_a(x)) * torch.sigmoid(self.fc_b(x))
        y = self.drop(y)
        return self.fc_out(y)


class ComplexMHSA(nn.Module):
    """
    Lightweight CV-TRN-inspired attention.

    I and Q streams share Q/K/V projections. The attention matrix uses complex
    correlation:
        real = Qi Ki^T + Qq Kq^T
        imag = Qq Ki^T - Qi Kq^T
    Outputs use a complex multiplication style mixing with Vi/Vq.
    """

    def __init__(self, dim=64, num_heads=4, dropout=0.10, max_tokens=32):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.rpe = RelativePositionBias1D(num_heads, max_tokens=max_tokens)
        self.out_proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def _heads(self, x):
        b, n, d = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x):
        b, h, n, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, n, h * d)

    def forward(self, xi, xq):
        qi = self._heads(self.q_proj(xi))
        qq = self._heads(self.q_proj(xq))
        ki = self._heads(self.k_proj(xi))
        kq = self._heads(self.k_proj(xq))
        vi = self._heads(self.v_proj(xi))
        vq = self._heads(self.v_proj(xq))

        attn_real = (torch.matmul(qi, ki.transpose(-2, -1)) + torch.matmul(qq, kq.transpose(-2, -1))) * self.scale
        attn_imag = (torch.matmul(qq, ki.transpose(-2, -1)) - torch.matmul(qi, kq.transpose(-2, -1))) * self.scale
        bias = self.rpe(xi.size(1))

        w_real = torch.softmax(attn_real + bias, dim=-1)
        w_imag = torch.softmax(attn_imag + bias, dim=-1)
        w_real = self.drop(w_real)
        w_imag = self.drop(w_imag)

        out_i = torch.matmul(w_real, vi) - torch.matmul(w_imag, vq)
        out_q = torch.matmul(w_real, vq) + torch.matmul(w_imag, vi)
        out_i = self.out_proj(self._merge(out_i))
        out_q = self.out_proj(self._merge(out_q))
        return out_i, out_q


class CVTRNBlock(nn.Module):
    def __init__(self, dim=64, num_heads=4, dropout=0.10, max_tokens=32):
        super().__init__()
        self.norm_i1 = nn.LayerNorm(dim)
        self.norm_q1 = nn.LayerNorm(dim)
        self.attn = ComplexMHSA(dim=dim, num_heads=num_heads, dropout=dropout, max_tokens=max_tokens)
        self.norm_i2 = nn.LayerNorm(dim)
        self.norm_q2 = nn.LayerNorm(dim)
        self.ffn = DBGLU(dim=dim, hidden_mult=2.0, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, xi, xq):
        ai, aq = self.attn(self.norm_i1(xi), self.norm_q1(xq))
        xi = xi + self.drop(ai)
        xq = xq + self.drop(aq)
        xi = xi + self.drop(self.ffn(self.norm_i2(xi)))
        xq = xq + self.drop(self.ffn(self.norm_q2(xq)))
        return xi, xq


class CVTRNAux2016(nn.Module):
    def __init__(
        self,
        num_classes=11,
        dim=64,
        depth=2,
        num_heads=4,
        frame_len=16,
        stride=8,
        seq_len=128,
        dropout=0.10,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.dim = int(dim)
        self.frame_len = int(frame_len)
        self.stride = int(stride)
        n_frames = (int(seq_len) - self.frame_len) // self.stride + 1
        max_tokens = n_frames + 1

        self.frame_embed = nn.Conv1d(1, dim, kernel_size=self.frame_len, stride=self.stride, bias=True)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.input_norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList(
            [CVTRNBlock(dim=dim, num_heads=num_heads, dropout=dropout, max_tokens=max_tokens) for _ in range(depth)]
        )
        self.final_norm_i = nn.LayerNorm(dim)
        self.final_norm_q = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, num_classes),
        )
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x, return_embedding=False):
        if x.dim() != 3 or x.size(1) != 2:
            raise ValueError(f"CVTRNAux2016 expects [B,2,T], got {tuple(x.shape)}")
        xi = self.frame_embed(x[:, 0:1, :]).transpose(1, 2)
        xq = self.frame_embed(x[:, 1:2, :]).transpose(1, 2)
        b = x.size(0)
        cls = self.cls.expand(b, -1, -1)
        xi = torch.cat([cls, xi], dim=1)
        xq = torch.cat([cls, xq], dim=1)
        xi = self.input_norm(xi)
        xq = self.input_norm(xq)
        for block in self.blocks:
            xi, xq = block(xi, xq)
        ci = self.final_norm_i(xi[:, 0])
        cq = self.final_norm_q(xq[:, 0])
        mag = torch.sqrt(ci * ci + cq * cq + 1e-8)
        phase_like = ci * cq
        emb = torch.cat([ci, cq, mag, phase_like], dim=1)
        logits = self.head(emb)
        if return_embedding:
            return emb, logits
        return logits


def build_cv_trn_aux_model(device, num_classes=11, dim=64, depth=2, num_heads=4, frame_len=16, stride=8, dropout=0.10):
    model = CVTRNAux2016(
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        num_heads=num_heads,
        frame_len=frame_len,
        stride=stride,
        dropout=dropout,
    )
    return model.to(device)
