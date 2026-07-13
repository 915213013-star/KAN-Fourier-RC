import os
import types
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_moe_attention import (
    LieQKAN,
    MoEFusedClassifier,
    BottleneckSignalLSTM,
    DynamicAttentionFusion,
)

CHECKPOINT_DIR = "checkpoints"
HOS_DIM = 20
NUM_CLASSES = 11


def safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def patch_lieqkan_stability(model):
    lie_encoder = getattr(model, "lie_encoder", None)
    if lie_encoder is None:
        return model

    def stable_safe_log_euclidean_map(self, spd_matrix):
        spd = 0.5 * (spd_matrix + spd_matrix.transpose(1, 2))
        b, c, _ = spd.shape
        eye = torch.eye(c, device=spd.device, dtype=spd.dtype).unsqueeze(0).expand(b, c, c)

        for jitter in [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]:
            try:
                mat = spd + jitter * eye
                eigvals, eigvecs = torch.linalg.eigh(mat)
                eigvals = eigvals.clamp(min=1e-5, max=1e5)
                log_cov = torch.bmm(eigvecs, torch.diag_embed(torch.log(eigvals)))
                log_cov = torch.bmm(log_cov, eigvecs.transpose(1, 2))
                return log_cov
            except Exception:
                continue

        mat = spd + 1e-2 * eye
        u, s, _ = torch.linalg.svd(mat)
        s = s.clamp(min=1e-5, max=1e5)
        log_cov = torch.bmm(u, torch.diag_embed(torch.log(s)))
        log_cov = torch.bmm(log_cov, u.transpose(1, 2))
        return log_cov

    lie_encoder.safe_log_euclidean_map = types.MethodType(
        stable_safe_log_euclidean_map,
        lie_encoder,
    )
    return model


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        padding = dilation
        self.block = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class IQResidualBranch(nn.Module):
    def __init__(self, out_dim=96):
        super().__init__()
        self.stem = ConvBNAct(2, 32, kernel_size=5, padding=2)
        self.blocks = nn.Sequential(
            ResidualConvBlock(32, dilation=1),
            ResidualConvBlock(32, dilation=2),
            ResidualConvBlock(32, dilation=4),
            ConvBNAct(32, 64, kernel_size=3, padding=1),
            ResidualConvBlock(64, dilation=1),
            ResidualConvBlock(64, dilation=2),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 2, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.SiLU(),
            nn.Dropout(0.15),
        )

    def forward(self, raw_iq):
        x = self.stem(raw_iq)
        x = self.blocks(x)
        avg = torch.mean(x, dim=-1)
        mx = torch.amax(x, dim=-1)
        return self.head(torch.cat([avg, mx], dim=1))


class FFTFeatureBranch(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNAct(4, 32, kernel_size=5, padding=2),
            ResidualConvBlock(32, dilation=1),
            ResidualConvBlock(32, dilation=2),
            ConvBNAct(32, 48, kernel_size=3, padding=1),
            ResidualConvBlock(48, dilation=1),
        )
        self.head = nn.Sequential(
            nn.Linear(48 * 2, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.SiLU(),
            nn.Dropout(0.15),
        )

    def forward(self, raw_iq):
        i = raw_iq[:, 0, :]
        q = raw_iq[:, 1, :]
        z = torch.complex(i.float(), q.float())
        spec = torch.fft.fft(z, dim=-1)

        real = spec.real
        imag = spec.imag
        mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        log_mag = torch.log1p(mag)
        phase = torch.atan2(imag, real)
        phase_diff = torch.diff(phase, dim=-1, prepend=phase[:, :1])

        feats = torch.stack([real, imag, log_mag, phase_diff], dim=1)
        feats = (feats - feats.mean(dim=-1, keepdim=True)) / (feats.std(dim=-1, keepdim=True) + 1e-5)

        x = self.conv(feats)
        avg = torch.mean(x, dim=-1)
        mx = torch.amax(x, dim=-1)
        return self.head(torch.cat([avg, mx], dim=1))


class WideSparseTeacherStudent(nn.Module):
    def __init__(self, num_classes=11, hos_dim=20, num_experts=4):
        super().__init__()

        self.lie_encoder = LieQKAN(out_dim=128)

        self.hos_mlp = nn.Sequential(
            nn.Linear(hos_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )

        self.time_encoder = BottleneckSignalLSTM(target_len=128)
        self.iq_branch = IQResidualBranch(out_dim=96)
        self.fft_branch = FFTFeatureBranch(out_dim=64)

        self.fusion_dim = 128 + 64 + 48 + 96 + 64
        self.attention_fusion = DynamicAttentionFusion(fusion_dim=self.fusion_dim, reduction=4)

        self.router = nn.Sequential(
            nn.Linear(self.fusion_dim, 96),
            nn.ReLU(),
            nn.Linear(96, num_experts),
        )

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.fusion_dim, 192),
                nn.BatchNorm1d(192),
                nn.SiLU(),
                nn.Dropout(0.25),
                nn.Linear(192, num_classes),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x_iq, x_hos):
        raw_iq = x_iq[:, 1:3, :]

        geo_feat = self.lie_encoder(x_iq)
        stat_feat = self.hos_mlp(x_hos)
        time_feat = self.time_encoder(raw_iq)
        iq_feat = self.iq_branch(raw_iq)
        fft_feat = self.fft_branch(raw_iq)

        combined = torch.cat([geo_feat, stat_feat, time_feat, iq_feat, fft_feat], dim=1)
        combined = self.attention_fusion(combined)

        routing_weights = F.softmax(self.router(combined), dim=1)

        final_logits = 0.0
        for i, expert in enumerate(self.experts):
            final_logits = final_logits + routing_weights[:, i:i + 1] * expert(combined)

        return geo_feat, final_logits


def build_original_model():
    return MoEFusedClassifier(
        lie_model=LieQKAN(out_dim=128),
        num_classes=NUM_CLASSES,
        hos_dim=HOS_DIM,
        num_experts=3,
    )


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def try_load_checkpoint(model, path, device):
    if path is None or path == "":
        return model

    if not os.path.exists(path):
        print(f"[!] checkpoint 不存在，跳过加载: {path}")
        return model

    ckpt = safe_torch_load(path, map_location=device)

    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    try:
        model.load_state_dict(state, strict=True)
        print(f"[*] checkpoint 加载成功: {path}")
    except Exception as e:
        print(f"[!] checkpoint 加载失败，只统计结构复杂度: {e}")

    return model


def measure_flops_thop(model, device):
    try:
        from thop import profile
    except Exception:
        return None

    model.eval()

    x_iq = torch.randn(1, 4, 128).to(device)
    x_hos = torch.randn(1, 20).to(device)

    with torch.no_grad():
        flops, params = profile(
            model,
            inputs=(x_iq, x_hos),
            verbose=False,
        )

    return flops


def measure_flops_ptflops(model, device):
    try:
        from ptflops import get_model_complexity_info
    except Exception:
        return None

    class Wrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            x_iq = x[:, :4, :]
            hos = x[:, 4:, 0]
            _, logits = self.inner(x_iq, hos)
            return logits

    wrapped = Wrapper(model).to(device).eval()

    macs, params = get_model_complexity_info(
        wrapped,
        (24, 128),
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
    )

    return macs * 2


def report_model(name, model, checkpoint_path, device):
    print("\n" + "=" * 90)
    print(f"模型: {name}")
    print("=" * 90)

    model = model.to(device)
    patch_lieqkan_stability(model)

    model = try_load_checkpoint(model, checkpoint_path, device)

    total, trainable = count_params(model)

    print(f"Total Params:     {total:,} ({total / 1e6:.4f} M)")
    print(f"Trainable Params: {trainable:,} ({trainable / 1e6:.4f} M)")

    flops = measure_flops_thop(model, device)

    if flops is None:
        flops = measure_flops_ptflops(model, device)

    if flops is None:
        print("FLOPs: 无法自动测量。请先安装 thop 或 ptflops：")
        print("       pip install thop")
    else:
        print(f"FLOPs:            {flops:,.0f} ({flops / 1e6:.4f} MFLOPs)")

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--original_ckpt",
        type=str,
        default=os.path.join(CHECKPOINT_DIR, "best_model_moe_attention_joint_strat_rml2016_seed1.pth"),
    )

    parser.add_argument(
        "--student_s2_ckpt",
        type=str,
        default=os.path.join(CHECKPOINT_DIR, "best_model_consistency_distill_mseed2_split1.pth"),
    )

    parser.add_argument(
        "--wide_ckpt",
        type=str,
        default=os.path.join(CHECKPOINT_DIR, "best_model_wide_sparse_teacher_student_seed11_split1.pth"),
    )

    parser.add_argument(
        "--soup_ckpt",
        type=str,
        default="checkpoints/greedy_model_soup_identity_split1.pth",
        help="checkpoints/greedy_model_soup_identity_split1.pth",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using Device: {device}")

    report_model(
        name="Original model_moe_attention",
        model=build_original_model(),
        checkpoint_path=args.original_ckpt,
        device=device,
    )

    report_model(
        name="Consistency Distill student_s2",
        model=build_original_model(),
        checkpoint_path=args.student_s2_ckpt,
        device=device,
    )

    if args.soup_ckpt:
        report_model(
            name="Greedy Model Soup, same original architecture",
            model=build_original_model(),
            checkpoint_path=args.soup_ckpt,
            device=device,
        )

    report_model(
        name="Wide Sparse-Teacher Student",
        model=WideSparseTeacherStudent(
            num_classes=NUM_CLASSES,
            hos_dim=HOS_DIM,
            num_experts=4,
        ),
        checkpoint_path=args.wide_ckpt,
        device=device,
    )


if __name__ == "__main__":
    main()