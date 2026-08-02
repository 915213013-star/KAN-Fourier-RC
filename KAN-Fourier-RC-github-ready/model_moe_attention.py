import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 1. 四元数卷积组件 (完全不变) ---
class QuaternionConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(QuaternionConv1d, self).__init__()
        assert in_channels % 4 == 0
        assert out_channels % 4 == 0
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        split_in = in_channels // 4
        split_out = out_channels // 4
        self.r_weight = nn.Parameter(torch.Tensor(split_out, split_in, kernel_size))
        self.i_weight = nn.Parameter(torch.Tensor(split_out, split_in, kernel_size))
        self.j_weight = nn.Parameter(torch.Tensor(split_out, split_in, kernel_size))
        self.k_weight = nn.Parameter(torch.Tensor(split_out, split_in, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.r_weight)
        nn.init.xavier_normal_(self.i_weight)
        nn.init.xavier_normal_(self.j_weight)
        nn.init.xavier_normal_(self.k_weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        cat_kernels_4_r = torch.cat([self.r_weight, -self.i_weight, -self.j_weight, -self.k_weight], dim=1)
        cat_kernels_4_i = torch.cat([self.i_weight, self.r_weight, -self.k_weight, self.j_weight], dim=1)
        cat_kernels_4_j = torch.cat([self.j_weight, self.k_weight, self.r_weight, -self.i_weight], dim=1)
        cat_kernels_4_k = torch.cat([self.k_weight, -self.j_weight, self.i_weight, self.r_weight], dim=1)
        hamilton_kernel = torch.cat([cat_kernels_4_r, cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k], dim=0)
        return F.conv1d(x, hamilton_kernel, self.bias, self.stride, self.padding)

class MultiScaleQuaternionConv1d(nn.Module):
    def __init__(self, in_channels, out_channels_per_branch=16):
        super().__init__()
        self.branch_k3 = QuaternionConv1d(in_channels, out_channels_per_branch, kernel_size=3, padding=1)
        self.branch_k7 = QuaternionConv1d(in_channels, out_channels_per_branch, kernel_size=7, padding=3)
        self.branch_k11 = QuaternionConv1d(in_channels, out_channels_per_branch, kernel_size=11, padding=5)

    def forward(self, x):
        return torch.cat([self.branch_k3(x), self.branch_k7(x), self.branch_k11(x)], dim=1)

# --- 2. Fourier KAN 组件 (完全不变) ---
class FourierKANLinear(nn.Module):
    def __init__(self, in_features, out_features, fourier_order=3, scale_base=1.0, scale_fourier=0.1):
        super(FourierKANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fourier_order = fourier_order
        self.scale_base = scale_base
        self.scale_fourier = scale_fourier
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.fourier_weight = nn.Parameter(torch.Tensor(out_features, in_features, 2 * fourier_order))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            self.fourier_weight.data.uniform_(-self.scale_fourier, self.scale_fourier)

    def compute_fourier_bases(self, x):
        x = x * math.pi
        x = x.unsqueeze(-1)
        freqs = torch.arange(1, self.fourier_order + 1, dtype=x.dtype, device=x.device)
        x_times_freqs = x * freqs
        bases = torch.cat([torch.sin(x_times_freqs), torch.cos(x_times_freqs)], dim=-1)
        return bases

    def forward(self, x):
        base_output = F.linear(F.silu(x), self.base_weight)
        bases = self.compute_fourier_bases(x)
        fourier_output = F.linear(bases.view(x.size(0), -1), self.fourier_weight.view(self.out_features, -1))
        return base_output + fourier_output

class PiFourierKANLinear(nn.Module):
    def __init__(self, in_features, out_features, num_branches=2, fourier_order=3):
        super(PiFourierKANLinear, self).__init__()
        self.num_branches = num_branches
        self.out_features = out_features
        self.synapse_kan = FourierKANLinear(in_features, out_features * num_branches, fourier_order=fourier_order)
        self.membrane_weight = nn.Parameter(torch.ones(out_features))

    def forward(self, x):
        synaptic = self.synapse_kan(x)
        dendritic = synaptic.view(x.shape[0], self.out_features, self.num_branches)
        dendritic = torch.tanh(dendritic)
        prod_output = torch.prod(dendritic, dim=2)
        return prod_output * self.membrane_weight

# --- 3. 几何流主模型 (完全不变) ---
class LieQKAN(nn.Module):
    def __init__(self, out_dim=128):
        super(LieQKAN, self).__init__()
        self.q_downsample = QuaternionConv1d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.ms_q_conv = MultiScaleQuaternionConv1d(in_channels=4, out_channels_per_branch=16)
        self.dim_reduce = nn.Conv1d(48, 16, kernel_size=1)
        self.instance_norm = nn.InstanceNorm1d(16, affine=False)
        self.tangent_dim = 16 * 17 // 2
        self.tangent_bn = nn.BatchNorm1d(self.tangent_dim)
        self.kan1 = FourierKANLinear(self.tangent_dim, 64, fourier_order=3)
        self.kan2 = PiFourierKANLinear(64, out_dim, num_branches=2, fourier_order=3)
        self.epsilon = 1e-4

    def compute_spd_matrix(self, x):
        batch_size, channels, time = x.size()
        x_mean = x - x.mean(dim=2, keepdim=True)
        cov = torch.bmm(x_mean, x_mean.transpose(1, 2)) / (time - 1)
        trace_term = self.epsilon * torch.eye(channels, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        return cov + trace_term

    def safe_log_euclidean_map(self, spd_matrix):
        eigvals, eigvecs = torch.linalg.eigh(spd_matrix)
        eigvals_clipped = eigvals.clamp(min=1e-5)
        log_eigvals = torch.log(eigvals_clipped)
        log_cov = torch.bmm(eigvecs, torch.diag_embed(log_eigvals))
        log_cov = torch.bmm(log_cov, eigvecs.transpose(1, 2))
        return log_cov

    def upper_triangular_flatten(self, matrix):
        batch_size, dim, _ = matrix.size()
        triu_indices = torch.triu_indices(dim, dim, device=matrix.device)
        return matrix[:, triu_indices[0], triu_indices[1]]

    def forward(self, x):
        x = self.q_downsample(x)
        feat = self.instance_norm(self.dim_reduce(self.ms_q_conv(x)))
        spd = self.compute_spd_matrix(feat)
        log_spd = self.safe_log_euclidean_map(spd)
        tangent_vec = self.upper_triangular_flatten(log_spd)
        tangent_vec = torch.tanh(self.tangent_bn(tangent_vec))
        k1 = F.silu(self.kan1(tangent_vec))
        embedding = self.kan2(k1)
        hyperspherical_emb = F.normalize(embedding, p=2, dim=1)
        return hyperspherical_emb

# --- 4. 微缩漏斗版双层 Bi-LSTM (时序流) ---
class BottleneckSignalLSTM(nn.Module):
    def __init__(self, target_len=128):
        super(BottleneckSignalLSTM, self).__init__()
        self.adaptive_pool = nn.AdaptiveAvgPool1d(target_len)
        self.lstm1 = nn.LSTM(input_size=2, hidden_size=32, num_layers=1,
                             batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=24, num_layers=1,
                             batch_first=True, bidirectional=True)
        self.ln = nn.LayerNorm(48)

    def forward(self, x):
        if x.size(-1) > 128:
            x = self.adaptive_pool(x)
        x = x.permute(0, 2, 1)

        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, _ = self.lstm2(out)

        out = torch.mean(out, dim=1)
        out = self.ln(out)
        return out


# --- 5. 【新增】：动态交叉注意力特征融合模块 ---
class DynamicAttentionFusion(nn.Module):
    """
    Squeeze-and-Excitation 风格的通道注意力机制
    用于动态评估拼接后的特征中，哪些特征在当前 SNR 下最可靠并予以放大。
    """
    def __init__(self, fusion_dim, reduction=4):
        super().__init__()
        # 利用两层 MLP 学习各个特征通道之间的关联性，生成权重掩码
        self.attention = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // reduction),
            nn.ReLU(),
            nn.Linear(fusion_dim // reduction, fusion_dim),
            nn.Sigmoid()  # 将权重压缩到 0~1 之间
        )

    def forward(self, x):
        # x shape: [Batch, fusion_dim]
        attn_weights = self.attention(x)
        # 将原始特征与其对应的注意力掩码逐元素相乘 (动态加权)
        return x * attn_weights


# --- 6. 终极融合模型：注意力融合 + MoE分类头 ---
class MoEFusedClassifier(nn.Module):
    # 默认 hos_dim = 20，兼容方案四的 (12维HOS + 8维图谱)
    def __init__(self, lie_model, num_classes=11, hos_dim=20, num_experts=3):
        super().__init__()
        self.lie_encoder = lie_model
        self.num_experts = num_experts

        # 流 2: 统计图谱流 (HOS + Graph)
        self.hos_mlp = nn.Sequential(
            nn.Linear(hos_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Tanh()
        )

        # 流 3: 时序流 (Bi-LSTM)
        self.time_encoder = BottleneckSignalLSTM(target_len=128)

        # 融合维度：删去了 FFT，回归为 128(Lie) + 64(HOS/Graph) + 48(LSTM) = 240
        self.fusion_dim = 128 + 64 + 48

        # 挂载动态交叉注意力融合模块
        self.attention_fusion = DynamicAttentionFusion(fusion_dim=self.fusion_dim)

        # 路由网络 (CQI Router): 判断当前信号属于哪个 SNR 区间
        self.router = nn.Sequential(
            nn.Linear(self.fusion_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts)
        )

        # 专家网络 (Experts): 独立的分类头
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.fusion_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            ) for _ in range(num_experts)
        ])

    def forward(self, x_iq, x_hos):
        # 提取原始 IQ 用于时域流
        raw_iq = x_iq[:, 1:3, :]

        # 1. 三大特征流提取
        geo_feat = self.lie_encoder(x_iq)        # 流1: 几何特征 (128)
        stat_feat = self.hos_mlp(x_hos)          # 流2: 统计图谱 (64)
        time_feat = self.time_encoder(raw_iq)    # 流3: 时序特征 (48)

        # 2. 初始级联 (Shape: [Batch, 240])
        combined_raw = torch.cat([geo_feat, stat_feat, time_feat], dim=1)

        # 3. 动态注意力交叉融合 (Shape: [Batch, 240])
        # 模型在此处自主决定增强抗噪特征，抑制无效特征
        combined_attended = self.attention_fusion(combined_raw)

        # 4. 计算路由权重 (基于加权后的特征)
        router_logits = self.router(combined_attended)
        routing_weights = F.softmax(router_logits, dim=1)

        # 5. 融合专家输出
        batch_size = combined_attended.size(0)
        num_classes = self.experts[0][-1].out_features
        final_logits = torch.zeros(batch_size, num_classes, device=combined_attended.device)

        for i, expert in enumerate(self.experts):
            expert_output = expert(combined_attended)
            weight = routing_weights[:, i].unsqueeze(1)
            final_logits += weight * expert_output

        return geo_feat, final_logits