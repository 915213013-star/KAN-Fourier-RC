import numpy as np
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform


def extract_hos_features(iq_signal):
    """
    提取高阶统计量(HOS)和瞬时特征 (12 维)
    """
    iq_signal = np.array(iq_signal, dtype=np.float32)
    complex_sig = iq_signal[0] + 1j * iq_signal[1]

    std_val = np.std(complex_sig)
    if std_val == 0:
        std_val = 1e-6
    complex_sig = complex_sig / std_val

    mag = np.abs(complex_sig)
    phase = np.angle(complex_sig)
    freq = np.diff(np.unwrap(phase), prepend=0)

    mean_mag2 = np.mean(mag ** 2) + 1e-6
    gamma_max = np.max(mag ** 2) / mean_mag2
    sigma_a = np.std(mag)
    skew_a = stats.skew(mag)
    kurt_a = stats.kurtosis(mag)

    sigma_f = np.std(freq)
    skew_f = stats.skew(freq)
    kurt_f = stats.kurtosis(freq)

    m20 = np.mean(complex_sig ** 2)
    m21 = np.mean(np.abs(complex_sig) ** 2)
    m40 = np.mean(complex_sig ** 4)
    m42 = np.mean((complex_sig ** 2) * (np.conj(complex_sig) ** 2))
    m63 = np.mean((complex_sig ** 3) * (np.conj(complex_sig) ** 3))

    c20 = m20
    c21 = m21
    c40 = m40 - 3 * m20 ** 2
    c42 = m42 - np.abs(m20) ** 2 - 2 * m21 ** 2
    c63 = m63 - 9 * c42 * c21 - 6 * c21 ** 3

    hos_feats = np.array([
        np.abs(c20), np.abs(c21),
        np.abs(c42), np.abs(c63),
        np.abs(m40)
    ])

    features = np.concatenate([
        [gamma_max, sigma_a, skew_a, kurt_a],
        [sigma_f, skew_f, kurt_f],
        hos_feats
    ])

    return np.nan_to_num(features).astype(np.float32)


def extract_graph_features(iq_signal, k=8, lambda_t=1.0):
    """
    提取星座图时空图谱特征 (8 维)
    复刻 GAMC 论文的 Graph Laplacian 特征提取
    """
    # 信号形状转换: (2, 128) -> (128, 2) 用于计算距离
    X = np.array(iq_signal, dtype=np.float32).T
    N = X.shape[0]

    # --- 1. 构建空间 K-NN 图 (Spatial Graph) ---
    dist_sq = squareform(pdist(X, 'sqeuclidean'))

    # 找到每个点最近的 k 个邻居的索引
    idx = np.argpartition(dist_sq, k + 1, axis=1)[:, :k + 1]
    knn_mask = np.zeros_like(dist_sq, dtype=bool)

    # 【修复核心】：直接传入 idx，不需要加前面的 N 偏移量
    np.put_along_axis(knn_mask, idx, True, axis=1)

    # 对称化并去除自环
    knn_mask = knn_mask | knn_mask.T
    np.fill_diagonal(knn_mask, False)

    # 计算高斯 RBF 核作为权重
    sigma = np.mean(dist_sq[knn_mask]) if np.any(knn_mask) else 1.0
    if sigma == 0: sigma = 1e-6
    A_s = np.zeros_like(dist_sq)
    A_s[knn_mask] = np.exp(-dist_sq[knn_mask] / (2 * sigma))

    # --- 2. 构建时序图 (Temporal Graph) ---
    A_t = np.zeros_like(dist_sq)
    np.fill_diagonal(A_t[1:], 1)
    np.fill_diagonal(A_t[:, 1:], 1)

    # --- 3. 融合为时空图 (Spatio-Temporal Graph) ---
    A_st = A_s + lambda_t * A_t

    # --- 4. 计算对称归一化拉普拉斯矩阵 ---
    d = np.sum(A_st, axis=1)
    d_inv_sqrt = np.power(d, -0.5, where=d > 0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_st = np.eye(N) - D_inv_sqrt @ A_st @ D_inv_sqrt

    # --- 5. 求解特征值并提取 8 维图谱统计量 ---
    eigvals = np.linalg.eigvalsh(L_st)  # 升序排列
    eigvals = np.clip(eigvals, 0, 2)  # 规范化截断防越界

    # 1) 代数连通度等前几个关键特征值
    lambda_1 = eigvals[1] if N > 1 else 0
    lambda_2 = eigvals[2] if N > 2 else 0
    # 2) 特征值比率与间隙
    ratio = lambda_2 / (lambda_1 + 1e-6)
    max_gap = np.max(np.diff(eigvals))
    # 3) 谱熵 (Spectral Entropy)
    p = eigvals / (np.sum(eigvals) + 1e-6)
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    # 4) 分布统计
    mean_val = np.mean(eigvals)
    var_val = np.var(eigvals)
    skew_val = stats.skew(eigvals)

    graph_features = np.array([
        lambda_1, lambda_2, ratio, max_gap,
        entropy, mean_val, var_val, skew_val
    ], dtype=np.float32)

    return np.nan_to_num(graph_features)


def extract_combined_features(iq_signal):
    """
    联合提取 HOS 与 图谱特征
    输出: 12 + 8 = 20 维特征向量
    """
    hos_feats = extract_hos_features(iq_signal)
    graph_feats = extract_graph_features(iq_signal)
    return np.concatenate([hos_feats, graph_feats])