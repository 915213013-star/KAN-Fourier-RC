import os
import math
import time
import random
import argparse
import types
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from dataprocessnew4 import RML2016Dataset
from model_moe_attention import LieQKAN, MoEFusedClassifier
from model_complex_tcn_fusion import ComplexTCNFusionClassifier


# ============================================================
# 0. 配置
# ============================================================
DATA_PATH = r"raw_data/RML2016.10a_dict.pkl"
CACHE_DIR = "feature_cache"

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

NUM_CLASSES = 11
HOS_DIM = 20

SNR_VALUES = list(range(-20, 20, 2))
TRANSITION_SNRS = [-10, -8, -6, -4, -2]
EDGE_LOW_SNRS = [-18, -16]

TTA_TRANSFORMS = [
    {"name": "identity", "shift": 0, "phase": 0.0},
    {"name": "roll+8", "shift": 8, "phase": 0.0},
    {"name": "roll-8", "shift": -8, "phase": 0.0},
    {"name": "phase+90", "shift": 0, "phase": math.pi / 2},
    {"name": "phase-90", "shift": 0, "phase": -math.pi / 2},
]

TTA_INDICES = [0, 1, 2, 3, 4]


# ============================================================
# 1. 参数
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--split_seed", type=int, default=1)
    parser.add_argument("--model_seed", type=int, default=1)

    parser.add_argument("--epochs", type=int, default=55)
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--lr", type=float, default=7e-5)
    parser.add_argument("--weight_decay", type=float, default=2e-5)

    parser.add_argument("--weight_step", type=float, default=0.2)

    parser.add_argument("--init_from", type=str, default="student_s2")

    parser.add_argument("--teacher_temp", type=float, default=1.0)
    parser.add_argument("--student_temp", type=float, default=2.0)

    parser.add_argument("--loss_ce_weight", type=float, default=0.45)
    parser.add_argument("--loss_kd_weight", type=float, default=0.50)
    parser.add_argument("--loss_cons_weight", type=float, default=0.05)

    parser.add_argument("--label_smoothing", type=float, default=0.01)

    parser.add_argument("--force_restart", action="store_true")
    parser.add_argument("--rebuild_teacher_cache", action="store_true")

    return parser.parse_args()


# ============================================================
# 2. 通用工具
# ============================================================
def load_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ============================================================
# 3. 关键修复：稳定版 Log-Euclidean SPD 映射
# ============================================================
def stable_safe_log_euclidean_map(self, spd_matrix):
    """
    替换 LieQKAN.safe_log_euclidean_map 的稳定版本。

    处理 torch.linalg.eigh 在 CUDA 上偶发的 ill-conditioned / repeated eigenvalue
    不收敛问题。策略：
        1. 对称化；
        2. nan_to_num；
        3. batch-level 多级微扰重试；
        4. 失败后逐样本重试；
        5. 仍失败则 SVD fallback；
        6. 最后退化为 diagonal-log fallback。
    """
    if spd_matrix.dim() != 3:
        raise ValueError(f"spd_matrix should be [B, C, C], got {spd_matrix.shape}")

    batch_size, channels, _ = spd_matrix.shape
    device = spd_matrix.device
    dtype = spd_matrix.dtype

    spd = 0.5 * (spd_matrix + spd_matrix.transpose(1, 2))
    spd = torch.nan_to_num(spd, nan=0.0, posinf=1e4, neginf=-1e4)

    eye = torch.eye(channels, device=device, dtype=dtype).unsqueeze(0)

    # ramp jitter 比单纯 eps * I 更能打破重复特征值导致的数值问题
    ramp = torch.linspace(1.0, 2.0, channels, device=device, dtype=dtype)
    ramp_diag = torch.diag(ramp).unsqueeze(0)

    min_eig = 1e-5

    jitter_list = [
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
    ]

    # ---------- batch-level retry ----------
    for eps in jitter_list:
        try:
            if eps == 0.0:
                mat = spd
            else:
                mat = spd + eps * ramp_diag + eps * eye

            eigvals, eigvecs = torch.linalg.eigh(mat)

            eigvals = eigvals.clamp(min=min_eig)
            log_eigvals = torch.log(eigvals)

            log_cov = torch.bmm(eigvecs, torch.diag_embed(log_eigvals))
            log_cov = torch.bmm(log_cov, eigvecs.transpose(1, 2))

            if torch.isfinite(log_cov).all():
                return log_cov

        except Exception:
            continue

    # ---------- per-sample fallback ----------
    outputs = []

    for b in range(batch_size):
        mat_b = spd[b]
        out_b = None

        for eps in jitter_list:
            try:
                if eps == 0.0:
                    mat = mat_b
                else:
                    mat = mat_b + eps * torch.diag(ramp) + eps * torch.eye(
                        channels,
                        device=device,
                        dtype=dtype,
                    )

                mat = 0.5 * (mat + mat.transpose(0, 1))
                eigvals, eigvecs = torch.linalg.eigh(mat)

                eigvals = eigvals.clamp(min=min_eig)
                log_eigvals = torch.log(eigvals)

                out_b = eigvecs @ torch.diag(log_eigvals) @ eigvecs.transpose(0, 1)

                if torch.isfinite(out_b).all():
                    break

                out_b = None

            except Exception:
                out_b = None
                continue

        # ---------- SVD fallback ----------
        if out_b is None:
            try:
                mat = mat_b + 1e-3 * torch.diag(ramp) + 1e-3 * torch.eye(
                    channels,
                    device=device,
                    dtype=dtype,
                )
                mat = 0.5 * (mat + mat.transpose(0, 1))

                u, s, vh = torch.linalg.svd(mat)

                s = s.clamp(min=min_eig)
                out_b = u @ torch.diag(torch.log(s)) @ u.transpose(0, 1)

                if not torch.isfinite(out_b).all():
                    out_b = None

            except Exception:
                out_b = None

        # ---------- diagonal-log fallback ----------
        if out_b is None:
            diag = torch.diagonal(mat_b, dim1=0, dim2=1).clamp(min=min_eig)
            out_b = torch.diag(torch.log(diag))

        outputs.append(out_b)

    return torch.stack(outputs, dim=0)


def patch_lieqkan_stability(model, name="model"):
    """
    对包含 lie_encoder 的 MoEFusedClassifier 注入稳定版 SPD log map。
    ComplexTCN 等无 lie_encoder 的模型会自动跳过。
    """
    if hasattr(model, "lie_encoder") and hasattr(model.lie_encoder, "safe_log_euclidean_map"):
        model.lie_encoder.safe_log_euclidean_map = types.MethodType(
            stable_safe_log_euclidean_map,
            model.lie_encoder,
        )
        print(f"[*] 已为 {name} 注入稳定版 safe_log_euclidean_map。")


# ============================================================
# 4. Dataset
# ============================================================
def build_dataset():
    try:
        dataset = RML2016Dataset(
            data_path=DATA_PATH,
            transform=True,
            return_snr=True,
            use_cache=True,
            cache_dir=CACHE_DIR,
            force_rebuild_cache=False,
        )
    except TypeError:
        print("[!] 当前 dataprocessnew.py 似乎是旧版，将使用旧方式加载。")
        dataset = RML2016Dataset(
            data_path=DATA_PATH,
            transform=True,
        )

    return dataset


class SNRSubset(Dataset):
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)

        if not hasattr(base_dataset, "snrs"):
            raise AttributeError("base_dataset 没有 snrs 属性。")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, local_idx):
        real_idx = int(self.indices[local_idx])
        item = self.base_dataset[real_idx]

        if isinstance(item, (tuple, list)) and len(item) == 4:
            return item

        if isinstance(item, (tuple, list)) and len(item) == 3:
            data, hos, label = item
            snr = self.base_dataset.snrs[real_idx]
            return data, hos, label, snr

        raise ValueError("Dataset 返回格式异常。")


class DistillDataset(Dataset):
    def __init__(self, base_dataset, indices, teacher_probs):
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.teacher_probs = np.asarray(teacher_probs, dtype=np.float32)

        assert len(self.indices) == len(self.teacher_probs)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, local_idx):
        real_idx = int(self.indices[local_idx])
        item = self.base_dataset[real_idx]

        if isinstance(item, (tuple, list)) and len(item) == 4:
            data, hos, label, snr = item
        elif isinstance(item, (tuple, list)) and len(item) == 3:
            data, hos, label = item
            snr = self.base_dataset.snrs[real_idx]
        else:
            raise ValueError("Dataset 返回格式异常。")

        teacher_prob = self.teacher_probs[local_idx]

        return data, hos, label, snr, teacher_prob


def make_joint_stratified_split(full_dataset, split_seed):
    labels = full_dataset.labels
    snrs = full_dataset.snrs

    composite_targets = np.array(
        [f"{int(lbl)}_{int(snr)}" for lbl, snr in zip(labels, snrs)]
    )

    indices = np.arange(len(full_dataset))

    train_idx, temp_idx, _, temp_targets = train_test_split(
        indices,
        composite_targets,
        test_size=0.2,
        random_state=split_seed,
        stratify=composite_targets,
    )

    val_idx, test_idx, _, _ = train_test_split(
        temp_idx,
        temp_targets,
        test_size=0.5,
        random_state=split_seed,
        stratify=temp_targets,
    )

    candidate_paths = [
        f"test_indices_model_oracle_distill_split{split_seed}.npy",
        f"test_indices_model_multitf_specialist_split{split_seed}.npy",
        f"test_indices_model_snr_estimator_split{split_seed}.npy",
        f"test_indices_model_transition_specialist_split{split_seed}.npy",
        f"test_indices_model_complex_tcn_distill_split{split_seed}.npy",
        f"test_indices_model_4stream_moe_joint_strat_rml2016_seed{split_seed}.npy",
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            saved_test_idx = np.load(path)

            if len(saved_test_idx) == len(test_idx) and np.array_equal(
                np.sort(saved_test_idx),
                np.sort(test_idx),
            ):
                test_idx = saved_test_idx
                print(f"[*] 已加载测试集索引: {path}")
                break

    test_index_path = f"test_indices_model_oracle_distill_split{split_seed}.npy"

    if not os.path.exists(test_index_path):
        np.save(test_index_path, test_idx)
        print(f"[*] 已保存 oracle-distill 测试集索引: {test_index_path}")

    return train_idx, val_idx, test_idx


# ============================================================
# 5. 模型加载
# ============================================================
def build_moe_model(device):
    base_model = LieQKAN(out_dim=128)

    model = MoEFusedClassifier(
        lie_model=base_model,
        num_classes=NUM_CLASSES,
        hos_dim=HOS_DIM,
        num_experts=3,
    ).to(device)

    return model


def load_moe_model(path, name, device):
    model = build_moe_model(device)

    if not os.path.exists(path):
        raise FileNotFoundError(f"没有找到 {name}: {path}")

    ckpt = load_checkpoint(path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[*] 加载 {name}: {path}")
        print(f"    -> epoch: {ckpt.get('epoch', 'Unknown')}")
        print(f"    -> best_val_acc: {ckpt.get('best_val_acc', 'Unknown')}")
        if "best_score" in ckpt:
            print(f"    -> best_score: {ckpt.get('best_score', 'Unknown')}")
    else:
        model.load_state_dict(ckpt)
        print(f"[*] 加载 {name}: 纯 state_dict")

    patch_lieqkan_stability(model, name=name)

    model.eval()
    return model


def load_complex_model(path, name, device):
    model = ComplexTCNFusionClassifier(
        num_classes=NUM_CLASSES,
        hos_dim=HOS_DIM,
    ).to(device)

    if not os.path.exists(path):
        raise FileNotFoundError(f"没有找到 {name}: {path}")

    ckpt = load_checkpoint(path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[*] 加载 {name}: {path}")
        print(f"    -> epoch: {ckpt.get('epoch', 'Unknown')}")
        print(f"    -> best_val_acc: {ckpt.get('best_val_acc', 'Unknown')}")
        if "best_score" in ckpt:
            print(f"    -> best_score: {ckpt.get('best_score', 'Unknown')}")
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model


def checkpoint_path(name, split_seed):
    if name == "original":
        return os.path.join(
            CHECKPOINT_DIR,
            f"best_model_moe_attention_joint_strat_rml2016_seed{split_seed}.pth",
        )

    if name == "student_s1":
        return os.path.join(
            CHECKPOINT_DIR,
            f"best_model_consistency_distill_seed{split_seed}.pth",
        )

    if name == "student_s2":
        return os.path.join(
            CHECKPOINT_DIR,
            f"best_model_consistency_distill_mseed2_split{split_seed}.pth",
        )

    if name == "student_s3":
        return os.path.join(
            CHECKPOINT_DIR,
            f"best_model_consistency_distill_mseed3_split{split_seed}.pth",
        )

    if name == "complex_tcn_s1":
        return os.path.join(
            CHECKPOINT_DIR,
            f"best_model_complex_tcn_distill_mseed1_split{split_seed}.pth",
        )

    if name == "transition_s1":
        return os.path.join(
            CHECKPOINT_DIR,
            f"best_model_transition_specialist_mseed1_split{split_seed}.pth",
        )

    raise ValueError(f"未知模型名: {name}")


def load_pool_models(split_seed, device):
    model_names = [
        "original",
        "student_s1",
        "student_s2",
        "student_s3",
        "complex_tcn_s1",
        "transition_s1",
    ]

    models = []
    valid_names = []

    for name in model_names:
        path = checkpoint_path(name, split_seed)

        if not os.path.exists(path):
            print(f"[!] 跳过 {name}，没有找到: {path}")
            continue

        if name.startswith("complex_tcn"):
            model = load_complex_model(path, name, device)
        else:
            model = load_moe_model(path, name, device)

        for p in model.parameters():
            p.requires_grad = False

        model.eval()

        models.append(model)
        valid_names.append(name)

    if len(models) == 0:
        raise RuntimeError("没有成功加载任何 teacher pool model。")

    return models, valid_names


def init_student_model(init_from, split_seed, device):
    model = build_moe_model(device)

    path = checkpoint_path(init_from, split_seed)

    if not os.path.exists(path):
        print(f"[!] 初始化 checkpoint 不存在: {path}，改用 original。")
        path = checkpoint_path("original", split_seed)

    ckpt = load_checkpoint(path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[*] Student 初始化自: {path}")
    else:
        model.load_state_dict(ckpt)

    patch_lieqkan_stability(model, name="student")

    return model


# ============================================================
# 6. TTA / teacher oracle
# ============================================================
def apply_tta_transform(x, cfg):
    y = x

    shift = int(cfg.get("shift", 0))
    phase = float(cfg.get("phase", 0.0))

    if shift != 0:
        y = torch.roll(y, shifts=shift, dims=-1)

    if abs(phase) > 1e-12:
        y = y.clone()

        c = math.cos(phase)
        s = math.sin(phase)

        I = y[:, 1, :].clone()
        Q = y[:, 2, :].clone()

        y[:, 1, :] = I * c - Q * s
        y[:, 2, :] = I * s + Q * c

    return y


def collect_pool_tta_probs(pool_models, loader, device, name="set"):
    """
    返回:
        labels: [N]
        snrs:   [N]
        stacked_probs: [N, M, C]
    """
    for m in pool_models:
        m.eval()

    labels_all = []
    snrs_all = []

    num_models = len(pool_models)
    probs_by_model = [[] for _ in range(num_models)]

    total_batches = len(loader)

    with torch.no_grad():
        for batch_idx, (data, hos, target, snr) in enumerate(loader):
            data = data.to(device)
            hos = hos.to(device)

            labels_all.append(target.numpy())
            snrs_all.append(snr.numpy())

            for model_idx, model in enumerate(pool_models):
                prob_sum = None

                for tta_idx in TTA_INDICES:
                    x_tta = apply_tta_transform(data, TTA_TRANSFORMS[tta_idx])
                    _, logits = model(x_tta, hos)

                    prob = F.softmax(logits, dim=1)

                    prob_sum = prob if prob_sum is None else prob_sum + prob

                prob_mean = prob_sum / len(TTA_INDICES)
                probs_by_model[model_idx].append(prob_mean.cpu().numpy().astype(np.float32))

            if batch_idx % 20 == 0:
                print(f"\r    -> {name} teacher TTA 进度: {batch_idx}/{total_batches}", end="")

    print()

    labels = np.concatenate(labels_all, axis=0)
    snrs = np.concatenate(snrs_all, axis=0)

    stacked = np.stack(
        [np.concatenate(probs_by_model[i], axis=0) for i in range(num_models)],
        axis=1,
    )

    stacked = stacked / (stacked.sum(axis=2, keepdims=True) + 1e-12)

    return labels, snrs, stacked.astype(np.float32)


def generate_simplex_weights(num_models, step):
    units = int(round(1.0 / step))

    weights = []
    current = []

    def rec(depth, remaining):
        if depth == num_models - 1:
            current.append(remaining)
            weights.append(np.asarray(current, dtype=np.float32) / units)
            current.pop()
            return

        for v in range(remaining + 1):
            current.append(v)
            rec(depth + 1, remaining - v)
            current.pop()

    rec(0, units)

    return weights


def weighted_ensemble_probs(stacked_probs, weights):
    w = np.asarray(weights, dtype=np.float32).reshape(1, -1, 1)
    prob = np.sum(stacked_probs * w, axis=1)
    prob = prob / (prob.sum(axis=1, keepdims=True) + 1e-12)
    return prob.astype(np.float32)


def search_oracle_weights_per_snr(labels_val, snrs_val, stacked_val, model_names, weight_step):
    weight_candidates = generate_simplex_weights(
        num_models=len(model_names),
        step=weight_step,
    )

    weights_by_snr = {}

    final_val = np.zeros((len(labels_val), NUM_CLASSES), dtype=np.float32)

    print("\n[*] 正在验证集上搜索 True-SNR Oracle teacher 权重...")

    for snr in sorted(np.unique(snrs_val)):
        mask = snrs_val == snr

        best_acc = -1.0
        best_w = None
        best_prob = None

        for w in weight_candidates:
            prob = weighted_ensemble_probs(stacked_val[mask], w)
            pred = prob.argmax(axis=1)
            acc = 100.0 * np.mean(pred == labels_val[mask])

            if acc > best_acc:
                best_acc = acc
                best_w = w
                best_prob = prob

        weights_by_snr[int(snr)] = best_w
        final_val[mask] = best_prob

        weight_str = ", ".join([f"{n}:{v:.2f}" for n, v in zip(model_names, best_w)])
        print(f"    SNR {int(snr):>3} dB | ValAcc={best_acc:6.2f}% | [{weight_str}]")

    overall = 100.0 * np.mean(final_val.argmax(axis=1) == labels_val)

    transition_mask = np.zeros_like(snrs_val, dtype=bool)
    for s in TRANSITION_SNRS:
        transition_mask |= snrs_val == s

    transition = 100.0 * np.mean(
        final_val[transition_mask].argmax(axis=1) == labels_val[transition_mask]
    )

    print(f"[*] Oracle teacher Val Overall={overall:.2f}% | Transition={transition:.2f}%")

    return weights_by_snr


def apply_oracle_teacher(stacked_probs, snrs, weights_by_snr):
    teacher = np.zeros((len(snrs), NUM_CLASSES), dtype=np.float32)

    for snr, w in weights_by_snr.items():
        mask = snrs == int(snr)

        if not np.any(mask):
            continue

        teacher[mask] = weighted_ensemble_probs(stacked_probs[mask], w)

    teacher = teacher / (teacher.sum(axis=1, keepdims=True) + 1e-12)

    return teacher.astype(np.float32)


def teacher_cache_path(args, pool_names):
    pool_tag = "_".join(pool_names)

    fname = (
        f"oracle_teacher_probs_split{args.split_seed}"
        f"_mseed{args.model_seed}"
        f"_wstep{args.weight_step}"
        f"_{pool_tag}.npz"
    )

    return os.path.join(CACHE_DIR, fname)


# ============================================================
# 7. 训练 loss
# ============================================================
def random_aug(x):
    y = x

    if random.random() < 0.85:
        shift = random.randint(-16, 16)

        if shift != 0:
            y = torch.roll(y, shifts=shift, dims=-1)

    if random.random() < 0.85:
        y = y.clone()

        batch_size = y.size(0)
        theta = torch.empty(batch_size, device=y.device).uniform_(-math.pi, math.pi)

        c = torch.cos(theta).view(-1, 1)
        s = torch.sin(theta).view(-1, 1)

        I = y[:, 1, :].clone()
        Q = y[:, 2, :].clone()

        y[:, 1, :] = I * c - Q * s
        y[:, 2, :] = I * s + Q * c

    # 原来的 zero erase 可能放大 SPD 奇异问题，这里改成轻微幅度缩放扰动
    if random.random() < 0.20:
        y = y.clone()
        scale = torch.empty(y.size(0), 1, 1, device=y.device).uniform_(0.85, 1.15)
        y = y * scale

    return y


def snr_weight(snr):
    snr = snr.to(dtype=torch.int32)

    w = torch.ones_like(snr, dtype=torch.float32, device=snr.device)

    w = torch.where(snr == -20, torch.full_like(w, 0.55), w)
    w = torch.where(snr == -18, torch.full_like(w, 1.25), w)
    w = torch.where(snr == -16, torch.full_like(w, 1.25), w)
    w = torch.where((snr >= -14) & (snr <= -12), torch.full_like(w, 1.10), w)
    w = torch.where((snr >= -10) & (snr <= -2), torch.full_like(w, 1.45), w)
    w = torch.where(snr >= 0, torch.full_like(w, 0.85), w)

    return w


def weighted_mean(loss_per_sample, weight):
    return (loss_per_sample * weight).sum() / (weight.sum() + 1e-6)


def soft_ce_from_prob(logits, teacher_prob, temperature):
    log_prob = F.log_softmax(logits / temperature, dim=1)
    return -torch.sum(teacher_prob * log_prob, dim=1) * (temperature ** 2)


def compute_distill_loss(model, data, hos, target, snr, teacher_prob, args):
    data_aug = random_aug(data)

    _, logits_clean = model(data, hos)
    _, logits_aug = model(data_aug, hos)

    w = snr_weight(snr)

    ce_clean = F.cross_entropy(
        logits_clean,
        target,
        reduction="none",
        label_smoothing=args.label_smoothing,
    )

    ce_aug = F.cross_entropy(
        logits_aug,
        target,
        reduction="none",
        label_smoothing=args.label_smoothing,
    )

    loss_ce = 0.5 * weighted_mean(ce_clean, w) + 0.5 * weighted_mean(ce_aug, w)

    kd_clean = soft_ce_from_prob(
        logits_clean,
        teacher_prob,
        temperature=args.student_temp,
    )

    kd_aug = soft_ce_from_prob(
        logits_aug,
        teacher_prob,
        temperature=args.student_temp,
    )

    loss_kd = 0.5 * weighted_mean(kd_clean, w) + 0.5 * weighted_mean(kd_aug, w)

    with torch.no_grad():
        clean_prob_detached = F.softmax(logits_clean.detach() / args.student_temp, dim=1)

    cons_aug = soft_ce_from_prob(
        logits_aug,
        clean_prob_detached,
        temperature=args.student_temp,
    )

    loss_cons = weighted_mean(cons_aug, w)

    loss = (
        args.loss_ce_weight * loss_ce
        + args.loss_kd_weight * loss_kd
        + args.loss_cons_weight * loss_cons
    )

    return loss, loss_ce, loss_kd, loss_cons, logits_clean


# ============================================================
# 8. 评估
# ============================================================
def evaluate_tta(model, loader, device):
    model.eval()

    total = 0
    correct = 0

    snr_correct = defaultdict(int)
    snr_total = defaultdict(int)

    with torch.no_grad():
        for data, hos, target, snr in loader:
            data = data.to(device)
            hos = hos.to(device)
            target = target.to(device)
            snr = snr.to(device)

            prob_sum = None

            for idx in TTA_INDICES:
                x_tta = apply_tta_transform(data, TTA_TRANSFORMS[idx])
                _, logits = model(x_tta, hos)
                prob = F.softmax(logits, dim=1)

                prob_sum = prob if prob_sum is None else prob_sum + prob

            prob_mean = prob_sum / len(TTA_INDICES)
            pred = prob_mean.argmax(dim=1)

            ok = pred.eq(target)

            total += target.size(0)
            correct += ok.sum().item()

            for s in torch.unique(snr):
                mask = snr == s
                s_int = int(s.item())

                snr_total[s_int] += mask.sum().item()
                snr_correct[s_int] += ok[mask].sum().item()

    overall_acc = 100.0 * correct / (total + 1e-6)

    def group_acc(snr_list):
        c = sum(snr_correct.get(int(s), 0) for s in snr_list)
        t = sum(snr_total.get(int(s), 0) for s in snr_list)
        return 100.0 * c / (t + 1e-6)

    transition_acc = group_acc(TRANSITION_SNRS)
    edge_low_acc = group_acc(EDGE_LOW_SNRS)

    negative_snrs = [s for s in snr_total.keys() if s < 0]
    high_snrs = [s for s in snr_total.keys() if s >= 0]

    negative_acc = group_acc(negative_snrs)
    high_acc = group_acc(high_snrs)

    by_snr = {}

    for s in sorted(snr_total.keys()):
        by_snr[s] = 100.0 * snr_correct[s] / (snr_total[s] + 1e-6)

    score = 0.90 * overall_acc + 0.08 * transition_acc + 0.02 * edge_low_acc

    return {
        "overall_acc": overall_acc,
        "transition_acc": transition_acc,
        "edge_low_acc": edge_low_acc,
        "negative_acc": negative_acc,
        "high_acc": high_acc,
        "score": score,
        "by_snr": by_snr,
    }


def print_snr_table(by_snr):
    print("\n各 SNR 准确率：")
    print("-" * 42)
    print(f"{'SNR(dB)':<12} | {'Accuracy(%)':>12}")
    print("-" * 42)

    for s, acc in by_snr.items():
        print(f"{s:<12} | {acc:>12.2f}")

    print("-" * 42)


# ============================================================
# 9. main
# ============================================================
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🚀 启动 Privileged True-SNR Oracle Distillation 训练")
    print(f"Using Device: {device}")

    set_seed(args.model_seed)

    best_path = os.path.join(
        CHECKPOINT_DIR,
        f"best_model_oracle_privileged_distill_mseed{args.model_seed}_split{args.split_seed}.pth",
    )

    latest_path = os.path.join(
        CHECKPOINT_DIR,
        f"latest_model_oracle_privileged_distill_mseed{args.model_seed}_split{args.split_seed}.pth",
    )

    full_dataset = build_dataset()

    train_idx, val_idx, test_idx = make_joint_stratified_split(
        full_dataset,
        split_seed=args.split_seed,
    )

    val_dataset = SNRSubset(full_dataset, val_idx)
    test_dataset = SNRSubset(full_dataset, test_idx)

    val_loader_plain = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    pool_models, pool_names = load_pool_models(args.split_seed, device)

    print("\n[*] Teacher pool:")
    for n in pool_names:
        print(f"    - {n}")

    # --------------------------------------------------------
    # 1. 用验证集搜索 True-SNR oracle 权重
    # --------------------------------------------------------
    val_labels, val_snrs, val_stacked = collect_pool_tta_probs(
        pool_models,
        val_loader_plain,
        device,
        name="val",
    )

    weights_by_snr = search_oracle_weights_per_snr(
        labels_val=val_labels,
        snrs_val=val_snrs,
        stacked_val=val_stacked,
        model_names=pool_names,
        weight_step=args.weight_step,
    )

    # --------------------------------------------------------
    # 2. 给训练集预计算 / 加载 oracle teacher prob
    # --------------------------------------------------------
    cache_path = teacher_cache_path(args, pool_names)
    teacher_probs_train = None

    if os.path.exists(cache_path) and not args.rebuild_teacher_cache:
        try:
            cache = np.load(cache_path, allow_pickle=False)

            cached_train_idx = cache["train_idx"]
            cached_probs = cache["teacher_probs"]

            if len(cached_train_idx) == len(train_idx) and np.array_equal(cached_train_idx, train_idx):
                teacher_probs_train = cached_probs.astype(np.float32)
                print(f"[*] 已加载 teacher soft target 缓存: {cache_path}")
                print(f"    -> teacher_probs_train shape: {teacher_probs_train.shape}")
            else:
                print("[!] teacher 缓存 train_idx 不匹配，将重新计算。")

        except Exception as e:
            print(f"[!] teacher 缓存读取失败，将重新计算: {e}")

    if teacher_probs_train is None:
        train_dataset_plain = SNRSubset(full_dataset, train_idx)

        train_loader_plain = DataLoader(
            train_dataset_plain,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        print("\n[*] 正在给训练集预计算 True-SNR oracle teacher soft targets...")

        train_labels, train_snrs, train_stacked = collect_pool_tta_probs(
            pool_models,
            train_loader_plain,
            device,
            name="train",
        )

        teacher_probs_train = apply_oracle_teacher(
            train_stacked,
            train_snrs,
            weights_by_snr,
        )

        print(f"[*] teacher_probs_train shape: {teacher_probs_train.shape}")

        np.savez_compressed(
            cache_path,
            train_idx=train_idx.astype(np.int64),
            teacher_probs=teacher_probs_train.astype(np.float32),
        )

        print(f"[*] 已保存 teacher soft target 缓存: {cache_path}")

    # 释放 teacher pool 显存
    for m in pool_models:
        del m

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # 3. Student 训练
    # --------------------------------------------------------
    train_dataset = DistillDataset(
        full_dataset,
        train_idx,
        teacher_probs_train,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    student = init_student_model(
        init_from=args.init_from,
        split_seed=args.split_seed,
        device=device,
    )

    optimizer = optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * len(train_loader),
        eta_min=1e-6,
    )

    start_epoch = 0
    best_score = 0.0
    best_metrics = None

    if os.path.exists(latest_path) and not args.force_restart:
        print(f"[*] 检测到 latest，恢复训练: {latest_path}")

        ckpt = load_checkpoint(latest_path, map_location=device)

        student.load_state_dict(ckpt["model_state_dict"])
        patch_lieqkan_stability(student, name="student-resumed")

        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        start_epoch = ckpt["epoch"] + 1
        best_score = float(ckpt.get("best_score", 0.0))
        best_metrics = ckpt.get("best_metrics", None)

        print(f"[*] 恢复成功，从 Epoch {start_epoch + 1} 继续。")
    else:
        print("[*] 从初始化模型开始 privileged oracle distillation。")

    print("\n" + "=" * 80)
    print("训练配置")
    print("=" * 80)
    print(f"Best path:   {best_path}")
    print(f"Latest path: {latest_path}")
    print(f"Init from:   {args.init_from}")
    print(f"Weight step: {args.weight_step}")
    print(f"LR:          {args.lr}")
    print(
        f"Loss = {args.loss_ce_weight}*CE + "
        f"{args.loss_kd_weight}*OracleKD + "
        f"{args.loss_cons_weight}*Consistency"
    )
    print("Val Score = 0.90*Overall + 0.08*Transition + 0.02*(-18/-16)")
    print("=" * 80 + "\n")

    for epoch in range(start_epoch, args.epochs):
        student.train()

        start_time = time.time()

        total = 0
        correct = 0

        loss_total = 0.0
        ce_total = 0.0
        kd_total = 0.0
        cons_total = 0.0
        skipped = 0
        skipped_forward = 0

        for batch_idx, (data, hos, target, snr, teacher_prob) in enumerate(train_loader):
            data = data.to(device)
            hos = hos.to(device)
            target = target.to(device)
            snr = snr.to(device)
            teacher_prob = teacher_prob.to(device)

            optimizer.zero_grad(set_to_none=True)

            try:
                loss, loss_ce, loss_kd, loss_cons, logits_clean = compute_distill_loss(
                    student,
                    data,
                    hos,
                    target,
                    snr,
                    teacher_prob,
                    args,
                )

            except Exception as e:
                msg = str(e)
                name = type(e).__name__

                if ("linalg.eigh" in msg) or ("LinAlg" in name) or ("converge" in msg):
                    skipped_forward += 1
                    optimizer.zero_grad(set_to_none=True)

                    if batch_idx % 100 == 0:
                        print(f"\n[!] 跳过一个 forward 数值异常 batch: {name}: {msg[:160]}")

                    continue

                raise

            if (not torch.isfinite(loss)) or loss.item() > 20.0:
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()

            total_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)

            if (not torch.isfinite(total_norm)) or total_norm.item() > 100.0:
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.step()
            scheduler.step()

            loss_total += loss.item()
            ce_total += loss_ce.item()
            kd_total += loss_kd.item()
            cons_total += loss_cons.item()

            with torch.no_grad():
                pred = logits_clean.argmax(dim=1)
                total += target.size(0)
                correct += pred.eq(target).sum().item()

            if batch_idx % 100 == 0:
                train_acc = 100.0 * correct / (total + 1e-6)
                lr_now = optimizer.param_groups[0]["lr"]

                print(
                    f"\rEpoch {epoch + 1:03d}/{args.epochs} "
                    f"[{batch_idx:04d}/{len(train_loader)}] "
                    f"Loss={loss.item():.4f} "
                    f"CE={loss_ce.item():.4f} "
                    f"KD={loss_kd.item():.4f} "
                    f"Cons={loss_cons.item():.4f} "
                    f"TrainAcc={train_acc:.2f}% "
                    f"LR={lr_now:.2e}",
                    end="",
                )

        val_metrics = evaluate_tta(student, val_loader_plain, device)

        is_best = val_metrics["score"] > best_score

        if is_best:
            best_score = val_metrics["score"]
            best_metrics = val_metrics

        train_acc = 100.0 * correct / (total + 1e-6)
        epoch_time = time.time() - start_time

        ckpt = {
            "epoch": epoch,
            "model_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_score": best_score,
            "best_metrics": best_metrics,
            "current_metrics": val_metrics,
            "weights_by_snr": weights_by_snr,
            "pool_names": pool_names,
            "config": vars(args),
        }

        torch.save(ckpt, latest_path)

        skip_msg = ""
        if skipped > 0 or skipped_forward > 0:
            skip_msg = f" | SkippedLoss={skipped} | SkippedForward={skipped_forward}"

        denom = max(len(train_loader) - skipped - skipped_forward, 1)

        print()
        print(
            f"Epoch {epoch + 1:03d} | "
            f"Time={epoch_time:.1f}s | "
            f"TrainAcc={train_acc:.2f}% | "
            f"Val={val_metrics['overall_acc']:.2f}% | "
            f"ValTrans={val_metrics['transition_acc']:.2f}% | "
            f"ValEdge(-18/-16)={val_metrics['edge_low_acc']:.2f}% | "
            f"ValNeg={val_metrics['negative_acc']:.2f}% | "
            f"ValHigh={val_metrics['high_acc']:.2f}% | "
            f"Score={val_metrics['score']:.2f} | "
            f"Loss={loss_total / denom:.4f}"
            f"{skip_msg}"
        )

        if is_best:
            torch.save(ckpt, best_path)
            print(
                f"★ New Best Oracle-Distill Student | "
                f"Score={best_score:.2f} | "
                f"Val={val_metrics['overall_acc']:.2f}% | "
                f"Trans={val_metrics['transition_acc']:.2f}%"
            )

        print("-" * 80)

    print("\n训练结束，加载 best 做测试。")

    best_ckpt = load_checkpoint(best_path, map_location=device)
    student.load_state_dict(best_ckpt["model_state_dict"])
    patch_lieqkan_stability(student, name="student-best")

    test_metrics = evaluate_tta(student, test_loader, device)

    print("\n" + "=" * 80)
    print("🏆 Oracle-Privileged Distill Student 测试结果，单模型 + TTA")
    print("=" * 80)
    print(f"Overall Test Acc:     {test_metrics['overall_acc']:.2f}%")
    print(f"Transition Test Acc:  {test_metrics['transition_acc']:.2f}%")
    print(f"Edge -18/-16 Acc:     {test_metrics['edge_low_acc']:.2f}%")
    print(f"Negative Test Acc:    {test_metrics['negative_acc']:.2f}%")
    print(f"High-SNR Test Acc:    {test_metrics['high_acc']:.2f}%")
    print("=" * 80)

    print_snr_table(test_metrics["by_snr"])


if __name__ == "__main__":
    main()