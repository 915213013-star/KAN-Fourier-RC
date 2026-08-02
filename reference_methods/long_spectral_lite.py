"""Lightweight long-sequence neural expert with resumable OOF export.

The model combines a depthwise-separable temporal path with a compact spectrum
path. It is intended as a portable reference expert for long I/Q sequences,
not as a bitwise reconstruction of a retained experiment artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .cache_io import SignalCache, load_signal_cache, save_prediction_cache


class DepthwiseSeparable1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.block(values)


class LongSpectralLite(nn.Module):
    """Small temporal/spectral classifier for I/Q sequences of arbitrary length."""

    def __init__(self, class_count: int, width: int = 24, pooled_bins: int = 8, dropout: float = 0.15):
        super().__init__()
        if class_count < 2 or width < 8 or pooled_bins < 2:
            raise ValueError("class_count, width, or pooled_bins is too small")
        spectral_width = max(8, width // 2)
        self.temporal = nn.Sequential(
            nn.Conv1d(2, width, 9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
            DepthwiseSeparable1d(width, width, 9, stride=2),
            DepthwiseSeparable1d(width, 2 * width, 7, stride=2),
            nn.AdaptiveAvgPool1d(pooled_bins),
        )
        self.spectral = nn.Sequential(
            nn.Conv1d(2, spectral_width, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(spectral_width),
            nn.GELU(),
            DepthwiseSeparable1d(spectral_width, width, 5, stride=2),
            nn.AdaptiveAvgPool1d(pooled_bins),
        )
        feature_count = 3 * width * pooled_bins
        self.classifier = nn.Sequential(
            nn.Linear(feature_count, 2 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, class_count),
        )

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        temporal = self.temporal(iq)
        # CUDA FFT does not support every reduced precision/length combination.
        with torch.amp.autocast("cuda", enabled=False):
            spectrum = torch.log1p(torch.abs(torch.fft.rfft(iq.float(), dim=-1, norm="ortho")))
        spectral = self.spectral(spectrum)
        features = torch.cat((temporal.flatten(1), spectral.flatten(1)), dim=1)
        return self.classifier(features)


@dataclass(frozen=True)
class TrainingConfig:
    folds: int = 3
    epochs: int = 100
    patience: int = 15
    batch_size: int = 256
    eval_batch_size: int = 1024
    workers: int = 0
    width: int = 24
    pooled_bins: int = 8
    dropout: float = 0.15
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    seed: int = 1
    device: str = "auto"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def _loader(iq: np.ndarray, labels: np.ndarray, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(np.asarray(iq, dtype=np.float32)), torch.from_numpy(labels.astype(np.int64)))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


@torch.no_grad()
def _predict(model: nn.Module, cache: SignalCache, batch_size: int, workers: int, device: torch.device) -> np.ndarray:
    model.eval()
    loader = _loader(cache.iq, cache.labels, batch_size, workers, False)
    blocks = []
    for iq, _ in loader:
        logits = model(iq.to(device, non_blocking=True))
        blocks.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(blocks, axis=0).astype(np.float32, copy=False)


@torch.no_grad()
def _accuracy(model: nn.Module, cache: SignalCache, batch_size: int, workers: int, device: torch.device) -> float:
    probabilities = _predict(model, cache, batch_size, workers, device)
    return float((probabilities.argmax(axis=1) == cache.labels).mean())


def _build_model(class_count: int, config: TrainingConfig, device: torch.device) -> LongSpectralLite:
    return LongSpectralLite(
        class_count,
        width=config.width,
        pooled_bins=config.pooled_bins,
        dropout=config.dropout,
    ).to(device)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> None:
    model.train()
    for iq, labels in loader:
        iq = iq.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            loss = nn.functional.cross_entropy(model(iq), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def _fit_with_monitor(
    fit: SignalCache,
    monitor: SignalCache,
    class_count: int,
    config: TrainingConfig,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int, float]:
    _seed_everything(seed)
    model = _build_model(class_count, config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.epochs, 1), eta_min=config.learning_rate * 0.05
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loader = _loader(fit.iq, fit.labels, config.batch_size, config.workers, True)
    best_state = None
    best_epoch = 0
    best_accuracy = -1.0
    stale = 0
    for epoch in range(1, config.epochs + 1):
        _train_epoch(model, loader, optimizer, scaler, device)
        scheduler.step()
        accuracy = _accuracy(model, monitor, config.eval_batch_size, config.workers, device)
        print(f"epoch {epoch:03d}/{config.epochs} monitor_accuracy={100.0 * accuracy:.3f}%", flush=True)
        if accuracy > best_accuracy + 1e-8:
            best_accuracy = accuracy
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    return best_state, best_epoch, best_accuracy


def _fit_fixed_epochs(
    train: SignalCache,
    class_count: int,
    config: TrainingConfig,
    epochs: int,
    device: torch.device,
) -> nn.Module:
    _seed_everything(config.seed + 10000)
    model = _build_model(class_count, config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=config.learning_rate * 0.05
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loader = _loader(train.iq, train.labels, config.batch_size, config.workers, True)
    for epoch in range(1, epochs + 1):
        _train_epoch(model, loader, optimizer, scaler, device)
        scheduler.step()
        print(f"full fit epoch {epoch:03d}/{epochs}", flush=True)
    return model


def _subset(cache: SignalCache, indices: np.ndarray) -> SignalCache:
    return SignalCache(
        iq=cache.iq[indices],
        labels=cache.labels[indices],
        sample_ids=cache.sample_ids[indices],
        base_probabilities=None if cache.base_probabilities is None else cache.base_probabilities[indices],
    )


def _fingerprint(train: SignalCache, config: TrainingConfig) -> str:
    payload = json.dumps(
        {
            "config": asdict(config),
            "shape": list(train.iq.shape),
            "labels_sha256": hashlib.sha256(np.ascontiguousarray(train.labels).tobytes()).hexdigest(),
            "sample_ids_sha256": hashlib.sha256(np.ascontiguousarray(train.sample_ids).tobytes()).hexdigest(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_long_spectral_lite(
    train_path: str | Path,
    validation_path: str | Path,
    test_path: str | Path | None,
    output_dir: str | Path,
    config: TrainingConfig,
    *,
    skip_existing: bool = False,
) -> dict:
    train = load_signal_cache(train_path)
    validation = load_signal_cache(validation_path)
    test = load_signal_cache(test_path) if test_path else None
    class_count = int(train.labels.max()) + 1
    for name, cache in (("validation", validation), ("test", test)):
        if cache is None:
            continue
        if cache.iq.shape[1:] != train.iq.shape[1:]:
            raise ValueError(f"{name} signal shape differs from training")
        if cache.labels.max(initial=0) >= class_count:
            raise ValueError(f"{name} contains a class absent from training")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _fingerprint(train, config)
    config_path = output_dir / "run_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") != fingerprint:
            raise RuntimeError("output directory contains a different long-spectral-lite configuration")
    else:
        config_path.write_text(
            json.dumps({"fingerprint": fingerprint, "config": asdict(config)}, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    expected = [output_dir / "train_oof_probs.npz", output_dir / "validation_probs.npz", output_dir / "summary.json"]
    if test is not None:
        expected.append(output_dir / "test_probs.npz")
    if skip_existing and all(path.exists() for path in expected):
        return json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    device = _device(config.device)
    splitter = StratifiedKFold(n_splits=config.folds, shuffle=True, random_state=config.seed)
    oof = np.zeros((train.labels.shape[0], class_count), dtype=np.float32)
    selected_epochs = []
    fold_scores = []
    folds_dir = output_dir / "folds"
    folds_dir.mkdir(exist_ok=True)
    for fold, (fit_indices, holdout_indices) in enumerate(splitter.split(train.iq, train.labels)):
        fold_path = folds_dir / f"fold_{fold}.npz"
        if skip_existing and fold_path.exists():
            with np.load(fold_path, allow_pickle=False) as archive:
                if not np.array_equal(archive["sample_ids"], train.sample_ids[holdout_indices]):
                    raise RuntimeError(f"fold {fold} cache does not match current split")
                probabilities = np.asarray(archive["probabilities"], dtype=np.float32)
                selected_epoch = int(archive["selected_epoch"])
                score = float(archive["monitor_accuracy"])
        else:
            print(f"fold {fold + 1}/{config.folds}: gradient fit excludes its OOF holdout", flush=True)
            state, selected_epoch, score = _fit_with_monitor(
                _subset(train, fit_indices),
                _subset(train, holdout_indices),
                class_count,
                config,
                config.seed + fold,
                device,
            )
            model = _build_model(class_count, config, device)
            model.load_state_dict(state)
            probabilities = _predict(
                model, _subset(train, holdout_indices), config.eval_batch_size, config.workers, device
            )
            np.savez_compressed(
                fold_path,
                probabilities=probabilities,
                sample_ids=train.sample_ids[holdout_indices],
                selected_epoch=np.asarray(selected_epoch),
                monitor_accuracy=np.asarray(score),
            )
        oof[holdout_indices] = probabilities
        selected_epochs.append(selected_epoch)
        fold_scores.append(score)

    save_prediction_cache(
        output_dir / "train_oof_probs.npz", oof, train.labels, train.sample_ids, "long_spectral_lite"
    )
    final_epochs = max(1, int(np.median(selected_epochs)))
    final_path = output_dir / "final_model.pt"
    if skip_existing and final_path.exists():
        model = _build_model(class_count, config, device)
        model.load_state_dict(torch.load(final_path, map_location=device))
    else:
        model = _fit_fixed_epochs(train, class_count, config, final_epochs, device)
        torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, final_path)
    validation_probabilities = _predict(model, validation, config.eval_batch_size, config.workers, device)
    save_prediction_cache(
        output_dir / "validation_probs.npz",
        validation_probabilities,
        validation.labels,
        validation.sample_ids,
        "long_spectral_lite",
    )
    test_accuracy = None
    if test is not None:
        test_probabilities = _predict(model, test, config.eval_batch_size, config.workers, device)
        save_prediction_cache(
            output_dir / "test_probs.npz",
            test_probabilities,
            test.labels,
            test.sample_ids,
            "long_spectral_lite",
        )
        test_accuracy = float((test_probabilities.argmax(axis=1) == test.labels).mean())
    report = {
        "method": "long_spectral_lite",
        "reference_implementation": True,
        "protocol": {
            "gradient_fit": "complementary folds only",
            "fold_checkpoint_selection": "gradient-excluded outer holdout",
            "oof_export": "selected checkpoint on the same outer holdout",
            "final_epoch_selection": "median selected OOF epoch",
            "policy_validation_usage": "inference only; not used for neural fitting",
            "test_usage": "optional frozen reporting only",
        },
        "parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "selected_fold_epochs": selected_epochs,
        "fold_monitor_accuracy": fold_scores,
        "final_fit_epochs": final_epochs,
        "oof_accuracy": float((oof.argmax(axis=1) == train.labels).mean()),
        "validation_accuracy": float((validation_probabilities.argmax(axis=1) == validation.labels).mean()),
        "test_accuracy": test_accuracy,
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--test-cache")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--pooled-bins", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    config = TrainingConfig(
        folds=args.folds,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        workers=args.workers,
        width=args.width,
        pooled_bins=args.pooled_bins,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
    )
    report = run_long_spectral_lite(
        args.train_cache,
        args.validation_cache,
        args.test_cache,
        args.output_dir,
        config,
        skip_existing=args.skip_existing,
    )
    print(
        f"long_spectral_lite OOF={100.0 * report['oof_accuracy']:.3f}% "
        f"validation={100.0 * report['validation_accuracy']:.3f}% "
        f"parameters={report['parameters']:,}"
    )


if __name__ == "__main__":
    main()
