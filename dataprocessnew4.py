import os
import pickle
import numpy as np
from torch.utils.data import Dataset

from featuresnew import extract_combined_features


DATA_PATH = r"raw_data/RML2016.10a_dict.pkl"

CACHE_VERSION = "v1_combined20_quat"


class RML2016Dataset(Dataset):
    """
    RML2016.10a 数据集封装。

    支持：
        1. 自动提取 HOS + Graph 20维特征
        2. 自动保存缓存，下次直接加载
        3. 可选择是否返回 SNR

    参数：
        data_path: 原始 pkl 数据路径
        transform: 是否执行 IQ -> 四元数变换
        return_snr: 是否在 __getitem__ 中返回 snr
        use_cache: 是否使用缓存
        cache_dir: 缓存目录
        cache_path: 手动指定缓存路径
        force_rebuild_cache: 是否强制重建缓存
    """

    def __init__(
        self,
        data_path=DATA_PATH,
        transform=None,
        return_snr=False,
        use_cache=True,
        cache_dir="feature_cache",
        cache_path=None,
        force_rebuild_cache=False,
    ):
        self.data_path = data_path
        self.transform = transform
        self.return_snr = return_snr
        self.use_cache = use_cache
        self.cache_dir = cache_dir

        if cache_path is None:
            cache_path = self._default_cache_path(
                data_path=data_path,
                transform=transform,
                cache_dir=cache_dir,
            )

        self.cache_path = cache_path

        if use_cache and (not force_rebuild_cache) and os.path.exists(cache_path):
            loaded = self._load_cache(cache_path)

            if loaded:
                return

            print("[!] 缓存加载失败，将重新从原始数据提取特征。")

        self._build_from_raw(data_path=data_path, transform=transform)

        if use_cache:
            self._save_cache(cache_path)

    @staticmethod
    def _default_cache_path(data_path, transform, cache_dir):
        base_name = os.path.splitext(os.path.basename(data_path))[0]
        mode = "quat" if transform else "iq"

        os.makedirs(cache_dir, exist_ok=True)

        return os.path.join(
            cache_dir,
            f"{base_name}_{mode}_{CACHE_VERSION}.npz",
        )

    def _load_cache(self, cache_path):
        try:
            print(f"[*] 检测到特征缓存，正在加载: {cache_path}")

            with np.load(cache_path, allow_pickle=False) as cache:
                required_keys = {
                    "data",
                    "hos_data",
                    "labels",
                    "snrs",
                    "mod_classes",
                    "transform_flag",
                    "cache_version",
                }

                missing = required_keys - set(cache.files)

                if missing:
                    print(f"[!] 缓存文件缺少字段: {missing}")
                    return False

                cache_version = str(cache["cache_version"][0])
                if cache_version != CACHE_VERSION:
                    print(
                        f"[!] 缓存版本不匹配: 当前={CACHE_VERSION}, "
                        f"缓存={cache_version}"
                    )
                    return False

                transform_flag = bool(cache["transform_flag"][0])
                current_transform_flag = bool(self.transform)

                if transform_flag != current_transform_flag:
                    print(
                        f"[!] 缓存 transform 标记不匹配: 当前={current_transform_flag}, "
                        f"缓存={transform_flag}"
                    )
                    return False

                self.data = cache["data"].astype(np.float32, copy=False)
                self.hos_data = cache["hos_data"].astype(np.float32, copy=False)
                self.labels = cache["labels"].astype(np.longlong, copy=False)
                self.snrs = cache["snrs"].astype(np.int32, copy=False)

                self.mod_classes = [str(x) for x in cache["mod_classes"].tolist()]
                self.mod_to_idx = {mod: i for i, mod in enumerate(self.mod_classes)}

            print(
                f"✅ 缓存加载完成. Samples: {len(self.data)}, "
                f"统计特征维度: {self.hos_data.shape[1]}"
            )
            print(f"    -> data shape: {self.data.shape}")
            print(f"    -> hos_data shape: {self.hos_data.shape}")

            return True

        except Exception as e:
            print(f"[!] 读取缓存时出错: {repr(e)}")
            return False

    def _save_cache(self, cache_path):
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)

            tmp_path = cache_path + ".tmp"

            print(f"[*] 正在保存特征缓存，下次将直接加载: {cache_path}")

            with open(tmp_path, "wb") as f:
                np.savez_compressed(
                    f,
                    data=self.data.astype(np.float32),
                    hos_data=self.hos_data.astype(np.float32),
                    labels=self.labels.astype(np.longlong),
                    snrs=self.snrs.astype(np.int32),
                    mod_classes=np.array(self.mod_classes),
                    transform_flag=np.array([bool(self.transform)]),
                    cache_version=np.array([CACHE_VERSION]),
                )

            os.replace(tmp_path, cache_path)

            print("✅ 特征缓存保存完成。")

        except Exception as e:
            print(f"[!] 保存缓存失败: {repr(e)}")

            tmp_path = cache_path + ".tmp"
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _build_from_raw(self, data_path, transform):
        print(f"正在加载数据: {data_path} ...")

        with open(data_path, "rb") as f:
            raw_data = pickle.load(f, encoding="latin1")

        self.data = []
        self.hos_data = []
        self.labels = []
        self.snrs = []

        self.mod_classes = sorted(list(set([k[0] for k in raw_data.keys()])))
        self.mod_to_idx = {mod: i for i, mod in enumerate(self.mod_classes)}

        print("正在进行特征提取 (四元数 + HOS + 星座图谱)...")
        print("注意: 求解拉普拉斯矩阵特征值计算量较大，全量提取可能需要 5-15 分钟，请耐心等待 ☕")

        total_samples = sum([v.shape[0] for v in raw_data.values()])
        processed_count = 0

        for key in raw_data.keys():
            mod_type, snr = key
            samples = raw_data[key]

            for i in range(samples.shape[0]):
                sig = samples[i]  # shape: (2, 128)

                combined_feat = extract_combined_features(sig)
                self.hos_data.append(combined_feat)

                if transform:
                    q_sig = self.iq_to_quaternion(sig)
                    self.data.append(q_sig)
                else:
                    self.data.append(sig)

                self.labels.append(self.mod_to_idx[mod_type])
                self.snrs.append(snr)

                processed_count += 1

                if processed_count % 10000 == 0:
                    print(
                        f"    -> 特征提取进度: {processed_count} / {total_samples} "
                        f"({(processed_count / total_samples) * 100:.1f}%)"
                    )

        self.data = np.array(self.data, dtype=np.float32)
        self.hos_data = np.array(self.hos_data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.longlong)
        self.snrs = np.array(self.snrs, dtype=np.int32)

        print(
            f"✅ 加载并提取完成. Samples: {len(self.data)}, "
            f"统计特征维度: {self.hos_data.shape[1]}"
        )

    def iq_to_quaternion(self, sig):
        I = sig[0, :]
        Q = sig[1, :]

        r = np.sqrt(I ** 2 + Q ** 2)
        x = I
        y = Q

        phase = np.unwrap(np.arctan2(Q, I))
        z = np.gradient(phase)

        return np.stack([r, x, y, z], axis=0).astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.return_snr:
            return (
                self.data[idx],
                self.hos_data[idx],
                self.labels[idx],
                self.snrs[idx],
            )

        return (
            self.data[idx],
            self.hos_data[idx],
            self.labels[idx],
        )