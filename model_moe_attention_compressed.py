import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuaternionConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        assert in_channels % 4 == 0
        assert out_channels % 4 == 0
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        split_in = in_channels // 4
        split_out = out_channels // 4
        self.r_weight = nn.Parameter(torch.empty(split_out, split_in, kernel_size))
        self.i_weight = nn.Parameter(torch.empty(split_out, split_in, kernel_size))
        self.j_weight = nn.Parameter(torch.empty(split_out, split_in, kernel_size))
        self.k_weight = nn.Parameter(torch.empty(split_out, split_in, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.r_weight)
        nn.init.xavier_normal_(self.i_weight)
        nn.init.xavier_normal_(self.j_weight)
        nn.init.xavier_normal_(self.k_weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        cat_r = torch.cat([self.r_weight, -self.i_weight, -self.j_weight, -self.k_weight], dim=1)
        cat_i = torch.cat([self.i_weight, self.r_weight, -self.k_weight, self.j_weight], dim=1)
        cat_j = torch.cat([self.j_weight, self.k_weight, self.r_weight, -self.i_weight], dim=1)
        cat_k = torch.cat([self.k_weight, -self.j_weight, self.i_weight, self.r_weight], dim=1)
        kernel = torch.cat([cat_r, cat_i, cat_j, cat_k], dim=0)
        return F.conv1d(x, kernel, self.bias, self.stride, self.padding)


class MultiScaleQuaternionConv1d(nn.Module):
    def __init__(self, in_channels, out_channels_per_branch=12):
        super().__init__()
        self.branch_k3 = QuaternionConv1d(in_channels, out_channels_per_branch, kernel_size=3, padding=1)
        self.branch_k7 = QuaternionConv1d(in_channels, out_channels_per_branch, kernel_size=7, padding=3)
        self.branch_k11 = QuaternionConv1d(in_channels, out_channels_per_branch, kernel_size=11, padding=5)

    def forward(self, x):
        return torch.cat([self.branch_k3(x), self.branch_k7(x), self.branch_k11(x)], dim=1)


class MultiScaleRealConv1d(nn.Module):
    """Real-valued replacement used by the quaternion component ablation."""

    def __init__(self, in_channels, out_channels_per_branch=12):
        super().__init__()
        self.branch_k3 = nn.Conv1d(in_channels, out_channels_per_branch, kernel_size=3, padding=1)
        self.branch_k7 = nn.Conv1d(in_channels, out_channels_per_branch, kernel_size=7, padding=3)
        self.branch_k11 = nn.Conv1d(in_channels, out_channels_per_branch, kernel_size=11, padding=5)
        for layer in (self.branch_k3, self.branch_k7, self.branch_k11):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        return torch.cat([self.branch_k3(x), self.branch_k7(x), self.branch_k11(x)], dim=1)


class FourierKANLinear(nn.Module):
    def __init__(self, in_features, out_features, fourier_order=3, scale_base=1.0, scale_fourier=0.1):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.fourier_order = int(fourier_order)
        self.scale_base = float(scale_base)
        self.scale_fourier = float(scale_fourier)
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.fourier_weight = nn.Parameter(torch.empty(out_features, in_features, 2 * self.fourier_order))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            self.fourier_weight.uniform_(-self.scale_fourier, self.scale_fourier)

    def compute_fourier_bases(self, x):
        x = x * math.pi
        x = x.unsqueeze(-1)
        freqs = torch.arange(1, self.fourier_order + 1, dtype=x.dtype, device=x.device)
        return torch.cat([torch.sin(x * freqs), torch.cos(x * freqs)], dim=-1)

    def forward(self, x):
        base_output = F.linear(F.silu(x), self.base_weight)
        bases = self.compute_fourier_bases(x)
        fourier_output = F.linear(bases.reshape(x.size(0), -1), self.fourier_weight.reshape(self.out_features, -1))
        return base_output + fourier_output


class PiFourierKANLinear(nn.Module):
    def __init__(self, in_features, out_features, num_branches=2, fourier_order=3):
        super().__init__()
        self.num_branches = int(num_branches)
        self.out_features = int(out_features)
        self.synapse_kan = FourierKANLinear(in_features, out_features * num_branches, fourier_order=fourier_order)
        self.membrane_weight = nn.Parameter(torch.ones(out_features))

    def forward(self, x):
        synaptic = self.synapse_kan(x)
        dendritic = synaptic.view(x.shape[0], self.out_features, self.num_branches)
        dendritic = torch.tanh(dendritic)
        return torch.prod(dendritic, dim=2) * self.membrane_weight


class CompressedLieQKAN(nn.Module):
    def __init__(
        self,
        out_dim=96,
        branch_channels=12,
        reduce_channels=12,
        kan_hidden=48,
        fourier_order=3,
        pi_branches=2,
        use_quaternion=True,
        use_spd=True,
        use_fourier_kan=True,
        mlp_hidden=560,
    ):
        super().__init__()
        self.use_quaternion = bool(use_quaternion)
        self.use_spd = bool(use_spd)
        self.use_fourier_kan = bool(use_fourier_kan)
        if self.use_quaternion:
            assert branch_channels % 4 == 0
            self.q_downsample = QuaternionConv1d(in_channels=4, out_channels=4, kernel_size=3, padding=1)
            self.ms_q_conv = MultiScaleQuaternionConv1d(in_channels=4, out_channels_per_branch=branch_channels)
        else:
            self.q_downsample = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, padding=1)
            self.ms_q_conv = MultiScaleRealConv1d(in_channels=4, out_channels_per_branch=branch_channels)
        self.dim_reduce = nn.Conv1d(3 * branch_channels, reduce_channels, kernel_size=1)
        self.instance_norm = nn.InstanceNorm1d(reduce_channels, affine=False)
        self.tangent_dim = reduce_channels * (reduce_channels + 1) // 2
        self.tangent_bn = nn.BatchNorm1d(self.tangent_dim)
        self.non_spd_project = (
            None if self.use_spd else nn.Linear(2 * reduce_channels, self.tangent_dim, bias=False)
        )
        if self.use_fourier_kan:
            self.kan1 = FourierKANLinear(self.tangent_dim, kan_hidden, fourier_order=fourier_order)
            self.kan2 = PiFourierKANLinear(
                kan_hidden,
                out_dim,
                num_branches=pi_branches,
                fourier_order=fourier_order,
            )
        else:
            # For full_geo_2expert, 560 hidden units match the two Fourier-KAN
            # layers within roughly 0.1% trainable parameters.
            self.kan1 = nn.Linear(self.tangent_dim, int(mlp_hidden))
            self.kan2 = nn.Linear(int(mlp_hidden), out_dim)
        self.epsilon = 1e-4
        self.out_dim = int(out_dim)

    def compute_spd_matrix(self, x):
        batch_size, channels, time = x.size()
        x_mean = x - x.mean(dim=2, keepdim=True)
        cov = torch.bmm(x_mean, x_mean.transpose(1, 2)) / (time - 1)
        eye = torch.eye(channels, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        return cov + self.epsilon * eye

    def safe_log_euclidean_map(self, spd_matrix):
        eigvals, eigvecs = torch.linalg.eigh(spd_matrix.float())
        eigvals = eigvals.clamp(min=1e-5)
        log_cov = torch.bmm(eigvecs, torch.diag_embed(torch.log(eigvals)))
        log_cov = torch.bmm(log_cov, eigvecs.transpose(1, 2))
        return log_cov.to(dtype=spd_matrix.dtype)

    def upper_triangular_flatten(self, matrix):
        dim = matrix.size(1)
        triu_indices = torch.triu_indices(dim, dim, device=matrix.device)
        return matrix[:, triu_indices[0], triu_indices[1]]

    def forward(self, x):
        x = self.q_downsample(x)
        feat = self.instance_norm(self.dim_reduce(self.ms_q_conv(x)))
        if self.use_spd:
            spd = self.compute_spd_matrix(feat)
            log_spd = self.safe_log_euclidean_map(spd)
            tangent_vec = self.upper_triangular_flatten(log_spd)
        else:
            first_order = torch.cat(
                [feat.mean(dim=2), feat.std(dim=2, unbiased=False)],
                dim=1,
            )
            tangent_vec = self.non_spd_project(first_order)
        tangent_vec = torch.tanh(self.tangent_bn(tangent_vec))
        k1 = F.silu(self.kan1(tangent_vec))
        embedding = self.kan2(k1)
        return F.normalize(embedding, p=2, dim=1)


class CompressedSignalLSTM(nn.Module):
    def __init__(self, target_len=128, hidden1=24, hidden2=16, dropout=0.25):
        super().__init__()
        self.adaptive_pool = nn.AdaptiveAvgPool1d(target_len)
        self.lstm1 = nn.LSTM(input_size=2, hidden_size=hidden1, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(input_size=2 * hidden1, hidden_size=hidden2, num_layers=1, batch_first=True, bidirectional=True)
        self.ln = nn.LayerNorm(2 * hidden2)
        self.out_dim = 2 * hidden2

    def forward(self, x):
        if x.size(-1) > 128:
            x = self.adaptive_pool(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        out = torch.mean(out, dim=1)
        return self.ln(out)


class PartialComplexDenoiseStem(nn.Module):
    """Tiny residual I/Q denoise stem inspired by CPPC-style partial channel mixing."""

    def __init__(
        self,
        hidden=8,
        partial_ratio=0.5,
        kernel_size=5,
        cap=0.30,
        norm_type="batch",
        identity_init=False,
        gate_bias=0.0,
    ):
        super().__init__()
        hidden = max(4, int(hidden))
        if hidden % 2:
            hidden += 1
        partial_channels = max(2, int(round((2 * hidden) * float(partial_ratio))))
        partial_channels = min(2 * hidden, partial_channels)
        self.partial_channels = int(partial_channels)
        self.cap = float(cap)
        self.in_proj = nn.Conv1d(2, 2 * hidden, kernel_size=1)
        self.partial_dw = nn.Conv1d(
            self.partial_channels,
            self.partial_channels,
            kernel_size=int(kernel_size),
            padding=int(kernel_size) // 2,
            groups=self.partial_channels,
        )
        self.mix = nn.Conv1d(2 * hidden, 2 * hidden, kernel_size=1)
        self.out_proj = nn.Conv1d(2 * hidden, 2, kernel_size=1)
        if str(norm_type).lower() == "group":
            self.norm = nn.GroupNorm(num_groups=2, num_channels=2 * hidden)
        else:
            self.norm = nn.BatchNorm1d(2 * hidden)
        self.gate = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
        if identity_init:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
        nn.init.constant_(self.gate[-2].bias, float(gate_bias))

    @staticmethod
    def rebuild_quat(x, iq):
        i_part = iq[:, 0, :]
        q_part = iq[:, 1, :]
        amp = torch.sqrt(i_part.square() + q_part.square() + 1e-6)
        i_next = torch.roll(i_part, shifts=-1, dims=-1)
        q_next = torch.roll(q_part, shifts=-1, dims=-1)
        cross = i_part * q_next - q_part * i_next
        dot = i_part * i_next + q_part * q_next
        dphi = torch.atan2(cross, dot + 1e-6)
        dphi = torch.cat([dphi[:, :-1], dphi[:, -2:-1]], dim=-1)
        return torch.stack([amp, i_part, q_part, dphi], dim=1).to(dtype=x.dtype)

    def forward(self, x):
        iq = x[:, 1:3, :]
        feat = self.in_proj(iq)
        proc = feat[:, : self.partial_channels, :]
        bypass = feat[:, self.partial_channels :, :]
        proc = self.partial_dw(proc)
        feat = torch.cat([proc, bypass], dim=1)
        feat = self.norm(F.silu(self.mix(F.silu(feat))))
        delta = torch.tanh(self.out_proj(feat))
        mean = iq.mean(dim=-1)
        std = iq.std(dim=-1, unbiased=False)
        gate = self.gate(torch.cat([mean, std], dim=1)).view(-1, 1, 1)
        clean_iq = iq + self.cap * gate * delta
        return self.rebuild_quat(x, clean_iq)


class DynamicAttentionFusion(nn.Module):
    def __init__(self, fusion_dim, reduction=4):
        super().__init__()
        hidden = max(8, fusion_dim // int(reduction))
        self.attention = nn.Sequential(
            nn.Linear(fusion_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, fusion_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attention(x)


class CompressedMoEFusedClassifier(nn.Module):
    def __init__(
        self,
        lie_model,
        num_classes=11,
        hos_dim=20,
        num_experts=3,
        stat_dim=48,
        stat_hidden=48,
        time_hidden1=24,
        time_hidden2=16,
        router_hidden=48,
        expert_hidden=96,
        attention_reduction=4,
        dropout=0.30,
        input_denoise=False,
        denoise_hidden=8,
        denoise_partial_ratio=0.5,
        denoise_kernel=5,
        denoise_cap=0.30,
        denoise_norm="batch",
        denoise_identity_init=False,
        denoise_gate_bias=0.0,
    ):
        super().__init__()
        self.lie_encoder = lie_model
        self.num_experts = int(num_experts)
        self.input_denoise = (
            PartialComplexDenoiseStem(
                hidden=denoise_hidden,
                partial_ratio=denoise_partial_ratio,
                kernel_size=denoise_kernel,
                cap=denoise_cap,
                norm_type=denoise_norm,
                identity_init=denoise_identity_init,
                gate_bias=denoise_gate_bias,
            )
            if input_denoise
            else None
        )
        self.hos_mlp = nn.Sequential(
            nn.Linear(hos_dim, stat_hidden),
            nn.BatchNorm1d(stat_hidden),
            nn.ReLU(),
            nn.Linear(stat_hidden, stat_dim),
            nn.Tanh(),
        )
        self.time_encoder = CompressedSignalLSTM(hidden1=time_hidden1, hidden2=time_hidden2, dropout=dropout)
        self.fusion_dim = int(lie_model.out_dim) + int(stat_dim) + int(self.time_encoder.out_dim)
        self.attention_fusion = DynamicAttentionFusion(self.fusion_dim, reduction=attention_reduction)
        self.router = nn.Sequential(
            nn.Linear(self.fusion_dim, router_hidden),
            nn.ReLU(),
            nn.Linear(router_hidden, num_experts),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.fusion_dim, expert_hidden),
                    nn.BatchNorm1d(expert_hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_hidden, num_classes),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x_iq, x_hos):
        if self.input_denoise is not None:
            x_iq = self.input_denoise(x_iq)
        raw_iq = x_iq[:, 1:3, :]
        geo_feat = self.lie_encoder(x_iq)
        stat_feat = self.hos_mlp(x_hos)
        time_feat = self.time_encoder(raw_iq)
        combined = torch.cat([geo_feat, stat_feat, time_feat], dim=1)
        combined = self.attention_fusion(combined)
        routing_weights = F.softmax(self.router(combined), dim=1)
        final_logits = torch.zeros(combined.size(0), self.experts[0][-1].out_features, device=combined.device)
        for i, expert in enumerate(self.experts):
            final_logits += routing_weights[:, i].unsqueeze(1) * expert(combined)
        return geo_feat, final_logits


COMPRESSED_VARIANTS = {
    "small": dict(
        lie_out=96,
        branch_channels=12,
        reduce_channels=12,
        kan_hidden=48,
        fourier_order=3,
        stat_dim=48,
        stat_hidden=48,
        time_hidden1=24,
        time_hidden2=16,
        router_hidden=48,
        expert_hidden=96,
        num_experts=3,
    ),
    "small_plus": dict(
        lie_out=96,
        branch_channels=12,
        reduce_channels=12,
        kan_hidden=48,
        fourier_order=3,
        stat_dim=48,
        stat_hidden=48,
        time_hidden1=28,
        time_hidden2=18,
        router_hidden=48,
        expert_hidden=104,
        num_experts=3,
    ),
    "small_f4": dict(
        lie_out=88,
        branch_channels=12,
        reduce_channels=12,
        kan_hidden=44,
        fourier_order=4,
        stat_dim=48,
        stat_hidden=48,
        time_hidden1=24,
        time_hidden2=16,
        router_hidden=48,
        expert_hidden=88,
        num_experts=3,
        dropout=0.18,
    ),
    "small_geo_f4": dict(
        lie_out=80,
        branch_channels=16,
        reduce_channels=12,
        kan_hidden=40,
        fourier_order=4,
        stat_dim=48,
        stat_hidden=48,
        time_hidden1=22,
        time_hidden2=15,
        router_hidden=48,
        expert_hidden=80,
        num_experts=3,
        dropout=0.18,
    ),
    "full_geo_2expert": dict(
        lie_out=104,
        branch_channels=16,
        reduce_channels=16,
        kan_hidden=56,
        fourier_order=3,
        stat_dim=40,
        stat_hidden=40,
        time_hidden1=18,
        time_hidden2=12,
        router_hidden=40,
        expert_hidden=80,
        num_experts=2,
        dropout=0.18,
    ),
    "full_geo_2expert_keep_time": dict(
        lie_out=104,
        branch_channels=16,
        reduce_channels=16,
        kan_hidden=56,
        fourier_order=3,
        stat_dim=40,
        stat_hidden=40,
        time_hidden1=24,
        time_hidden2=16,
        router_hidden=40,
        expert_hidden=80,
        num_experts=2,
        dropout=0.18,
    ),
    "full_geo_keep_time": dict(
        lie_out=88,
        branch_channels=16,
        reduce_channels=16,
        kan_hidden=48,
        fourier_order=3,
        stat_dim=40,
        stat_hidden=40,
        time_hidden1=24,
        time_hidden2=16,
        router_hidden=40,
        expert_hidden=72,
        num_experts=3,
        dropout=0.18,
    ),
    "full_geo_3expert_lite": dict(
        lie_out=96,
        branch_channels=16,
        reduce_channels=16,
        kan_hidden=52,
        fourier_order=3,
        stat_dim=40,
        stat_hidden=40,
        time_hidden1=24,
        time_hidden2=16,
        router_hidden=44,
        expert_hidden=80,
        num_experts=3,
        dropout=0.18,
    ),
    "elastic_mid": dict(
        lie_out=112,
        branch_channels=16,
        reduce_channels=16,
        kan_hidden=56,
        fourier_order=3,
        stat_dim=56,
        stat_hidden=56,
        time_hidden1=28,
        time_hidden2=20,
        router_hidden=56,
        expert_hidden=112,
        num_experts=3,
        dropout=0.20,
    ),
    "denoise_elastic_mid": dict(
        lie_out=112,
        branch_channels=16,
        reduce_channels=16,
        kan_hidden=56,
        fourier_order=3,
        stat_dim=56,
        stat_hidden=56,
        time_hidden1=28,
        time_hidden2=20,
        router_hidden=56,
        expert_hidden=112,
        num_experts=3,
        dropout=0.20,
        input_denoise=True,
        denoise_hidden=8,
        denoise_partial_ratio=0.5,
        denoise_kernel=5,
        denoise_cap=0.30,
    ),
    "denoise_full": dict(
        lie_out=128,
        branch_channels=16,
        reduce_channels=16,
        kan_hidden=64,
        fourier_order=3,
        stat_dim=64,
        stat_hidden=64,
        time_hidden1=32,
        time_hidden2=24,
        router_hidden=64,
        expert_hidden=128,
        num_experts=3,
        dropout=0.30,
        input_denoise=True,
        denoise_hidden=8,
        denoise_partial_ratio=0.5,
        denoise_kernel=5,
        denoise_cap=0.30,
    ),
    "denoise_full_v2": dict(
        lie_out=128,
        branch_channels=16,
        reduce_channels=16,
        kan_hidden=64,
        fourier_order=3,
        stat_dim=64,
        stat_hidden=64,
        time_hidden1=32,
        time_hidden2=24,
        router_hidden=64,
        expert_hidden=128,
        num_experts=3,
        dropout=0.30,
        input_denoise=True,
        denoise_hidden=8,
        denoise_partial_ratio=0.5,
        denoise_kernel=5,
        denoise_cap=0.25,
        denoise_norm="group",
        denoise_identity_init=True,
        denoise_gate_bias=-2.0,
    ),
    "denoise_elastic_mid_v2": dict(
        lie_out=112,
        branch_channels=16,
        reduce_channels=16,
        kan_hidden=56,
        fourier_order=3,
        stat_dim=56,
        stat_hidden=56,
        time_hidden1=28,
        time_hidden2=20,
        router_hidden=56,
        expert_hidden=112,
        num_experts=3,
        dropout=0.20,
        input_denoise=True,
        denoise_hidden=8,
        denoise_partial_ratio=0.5,
        denoise_kernel=5,
        denoise_cap=0.25,
        denoise_norm="group",
        denoise_identity_init=True,
        denoise_gate_bias=-2.0,
    ),
    "tiny": dict(
        lie_out=80,
        branch_channels=8,
        reduce_channels=12,
        kan_hidden=40,
        fourier_order=3,
        stat_dim=40,
        stat_hidden=40,
        time_hidden1=20,
        time_hidden2=14,
        router_hidden=40,
        expert_hidden=80,
        num_experts=3,
    ),
    "nano": dict(
        lie_out=64,
        branch_channels=8,
        reduce_channels=8,
        kan_hidden=32,
        fourier_order=2,
        stat_dim=32,
        stat_hidden=32,
        time_hidden1=16,
        time_hidden2=12,
        router_hidden=32,
        expert_hidden=64,
        num_experts=2,
    ),
}

# Matched full_geo_2expert component ablations. Only the named component is
# replaced; the temporal/statistical branches, routed heads, optimizer, and
# train/validation/test protocol remain unchanged.
COMPRESSED_VARIANTS["full_geo_2expert_no_quaternion"] = dict(
    COMPRESSED_VARIANTS["full_geo_2expert"],
    use_quaternion=False,
)
COMPRESSED_VARIANTS["full_geo_2expert_no_spd"] = dict(
    COMPRESSED_VARIANTS["full_geo_2expert"],
    use_spd=False,
)
COMPRESSED_VARIANTS["full_geo_2expert_no_fourier_kan"] = dict(
    COMPRESSED_VARIANTS["full_geo_2expert"],
    use_fourier_kan=False,
    mlp_hidden=560,
)


def build_compressed_model(variant="small", num_classes=11, hos_dim=20):
    if variant not in COMPRESSED_VARIANTS:
        raise ValueError(f"Unknown variant {variant}. Choose from {sorted(COMPRESSED_VARIANTS)}")
    cfg = COMPRESSED_VARIANTS[variant]
    lie = CompressedLieQKAN(
        out_dim=cfg["lie_out"],
        branch_channels=cfg["branch_channels"],
        reduce_channels=cfg["reduce_channels"],
        kan_hidden=cfg["kan_hidden"],
        fourier_order=cfg["fourier_order"],
        use_quaternion=cfg.get("use_quaternion", True),
        use_spd=cfg.get("use_spd", True),
        use_fourier_kan=cfg.get("use_fourier_kan", True),
        mlp_hidden=cfg.get("mlp_hidden", 560),
    )
    return CompressedMoEFusedClassifier(
        lie_model=lie,
        num_classes=num_classes,
        hos_dim=hos_dim,
        num_experts=cfg["num_experts"],
        stat_dim=cfg["stat_dim"],
        stat_hidden=cfg["stat_hidden"],
        time_hidden1=cfg["time_hidden1"],
        time_hidden2=cfg["time_hidden2"],
        router_hidden=cfg["router_hidden"],
        expert_hidden=cfg["expert_hidden"],
        dropout=cfg.get("dropout", 0.30),
        input_denoise=cfg.get("input_denoise", False),
        denoise_hidden=cfg.get("denoise_hidden", 8),
        denoise_partial_ratio=cfg.get("denoise_partial_ratio", 0.5),
        denoise_kernel=cfg.get("denoise_kernel", 5),
        denoise_cap=cfg.get("denoise_cap", 0.30),
        denoise_norm=cfg.get("denoise_norm", "batch"),
        denoise_identity_init=cfg.get("denoise_identity_init", False),
        denoise_gate_bias=cfg.get("denoise_gate_bias", 0.0),
    )
