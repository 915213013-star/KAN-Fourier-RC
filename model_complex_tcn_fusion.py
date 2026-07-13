import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexConv1d(nn.Module):
    """
    简单复数卷积。

    输入:
        x: [B, 2 * in_complex_channels, T]
           前一半为 real，后一半为 imag。

    输出:
        y: [B, 2 * out_complex_channels, T]
    """

    def __init__(self, in_complex_channels, out_complex_channels, kernel_size, padding=0, dilation=1):
        super().__init__()

        self.real_conv = nn.Conv1d(
            in_complex_channels,
            out_complex_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=True,
        )

        self.imag_conv = nn.Conv1d(
            in_complex_channels,
            out_complex_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=True,
        )

    def forward(self, x):
        xr, xi = torch.chunk(x, chunks=2, dim=1)

        yr = self.real_conv(xr) - self.imag_conv(xi)
        yi = self.real_conv(xi) + self.imag_conv(xr)

        return torch.cat([yr, yi], dim=1)


class TCNBlock(nn.Module):
    def __init__(self, channels, dilation=1, dropout=0.15):
        super().__init__()

        padding = dilation

        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation,
                groups=1,
            ),
            nn.BatchNorm1d(channels),
            nn.GELU(),

            nn.Conv1d(
                channels,
                channels,
                kernel_size=1,
            ),
            nn.BatchNorm1d(channels),
            nn.GELU(),

            nn.Dropout(dropout),
        )

        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        return x + torch.tanh(self.alpha) * self.net(x)


class AttentionPool1d(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.score = nn.Sequential(
            nn.Conv1d(channels, channels // 2, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(channels // 2, 1, kernel_size=1),
        )

    def forward(self, x):
        """
        x: [B, C, T]
        """
        attn = self.score(x)
        attn = torch.softmax(attn, dim=-1)

        pooled = torch.sum(x * attn, dim=-1)

        return pooled


class ComplexTCNBranch(nn.Module):
    """
    I/Q 复数时序分支。
    """

    def __init__(self, out_dim=128):
        super().__init__()

        self.cconv1 = ComplexConv1d(
            in_complex_channels=1,
            out_complex_channels=32,
            kernel_size=5,
            padding=2,
        )

        self.bn1 = nn.BatchNorm1d(64)

        self.blocks = nn.Sequential(
            TCNBlock(64, dilation=1, dropout=0.12),
            TCNBlock(64, dilation=2, dropout=0.12),
            TCNBlock(64, dilation=4, dropout=0.12),
            TCNBlock(64, dilation=8, dropout=0.12),
        )

        self.pool_attn = AttentionPool1d(64)

        self.fc = nn.Sequential(
            nn.Linear(64 + 64, 160),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(160, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, raw_iq):
        """
        raw_iq: [B, 2, T]，channel 0 = I, channel 1 = Q
        """
        x = self.cconv1(raw_iq)
        x = F.gelu(self.bn1(x))

        x = self.blocks(x)

        avg_pool = x.mean(dim=-1)
        attn_pool = self.pool_attn(x)

        feat = torch.cat([avg_pool, attn_pool], dim=1)

        return self.fc(feat)


class APBranch(nn.Module):
    """
    幅度 + 差分相位分支。
    """

    def __init__(self, out_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),

            TCNBlock(32, dilation=1, dropout=0.10),
            TCNBlock(32, dilation=2, dropout=0.10),
            TCNBlock(32, dilation=4, dropout=0.10),

            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),

            nn.Linear(32 * 16, 96),
            nn.GELU(),
            nn.Dropout(0.20),

            nn.Linear(96, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, raw_iq):
        I = raw_iq[:, 0, :]
        Q = raw_iq[:, 1, :]

        amp = torch.sqrt(I ** 2 + Q ** 2 + 1e-8)

        I_prev = torch.roll(I, shifts=1, dims=1)
        Q_prev = torch.roll(Q, shifts=1, dims=1)

        cross = I_prev * Q - Q_prev * I
        dot = I_prev * I + Q_prev * Q

        dphi = torch.atan2(cross, dot + 1e-8)
        dphi[:, 0] = 0.0

        x = torch.stack([amp, dphi], dim=1)

        return self.net(x)


class FFTBranch(nn.Module):
    """
    频域分支。
    """

    def __init__(self, out_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),

            nn.Conv1d(32, 48, kernel_size=5, padding=2),
            nn.BatchNorm1d(48),
            nn.GELU(),

            nn.Conv1d(48, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),

            nn.Linear(64 * 16, 128),
            nn.GELU(),
            nn.Dropout(0.25),

            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, raw_iq):
        I = raw_iq[:, 0, :]
        Q = raw_iq[:, 1, :]

        complex_sig = torch.complex(I, Q)

        fft = torch.fft.fft(complex_sig, dim=-1)
        fft = torch.fft.fftshift(fft, dim=-1)

        mag = torch.log1p(torch.abs(fft))
        real = fft.real
        imag = fft.imag

        mag = (mag - mag.mean(dim=1, keepdim=True)) / (
            mag.std(dim=1, keepdim=True) + 1e-6
        )

        power = torch.sqrt(
            (real ** 2 + imag ** 2).mean(dim=1, keepdim=True) + 1e-6
        )

        real = real / power
        imag = imag / power

        x = torch.stack([mag, real, imag], dim=1)

        return self.net(x)


class HOSBranch(nn.Module):
    def __init__(self, hos_dim=20, out_dim=48):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hos_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.20),

            nn.Linear(64, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x_hos):
        return self.net(x_hos)


class FusionGate(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()

        hidden = max(dim // reduction, 32)

        self.gate = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class ComplexTCNFusionClassifier(nn.Module):
    """
    与 model_moe_attention 明显不同的 complementary backbone。

    输入:
        x_iq:  [B, 4, 128]，四元数格式 [r, I, Q, phase_gradient]
        x_hos: [B, 20]

    输出:
        feat:   [B, fusion_dim]
        logits: [B, num_classes]
    """

    def __init__(self, num_classes=11, hos_dim=20):
        super().__init__()

        self.complex_branch = ComplexTCNBranch(out_dim=128)
        self.ap_branch = APBranch(out_dim=64)
        self.fft_branch = FFTBranch(out_dim=64)
        self.hos_branch = HOSBranch(hos_dim=hos_dim, out_dim=48)

        self.fusion_dim = 128 + 64 + 64 + 48

        self.gate = FusionGate(self.fusion_dim)

        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 192),
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Dropout(0.35),

            nn.Linear(192, 96),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.Dropout(0.25),

            nn.Linear(96, num_classes),
        )

    def forward(self, x_iq, x_hos):
        raw_iq = x_iq[:, 1:3, :]

        feat_complex = self.complex_branch(raw_iq)
        feat_ap = self.ap_branch(raw_iq)
        feat_fft = self.fft_branch(raw_iq)
        feat_hos = self.hos_branch(x_hos)

        feat = torch.cat(
            [feat_complex, feat_ap, feat_fft, feat_hos],
            dim=1,
        )

        feat = self.gate(feat)

        logits = self.classifier(feat)

        return feat, logits