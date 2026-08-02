"""Observable probability and signal features used by reference methods."""

from __future__ import annotations

import numpy as np


def probability_meta_features(probabilities: np.ndarray, extra: np.ndarray | None = None) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.ndim != 3:
        raise ValueError("probabilities must have shape [rows, candidates, classes]")
    rows, candidates, classes = probs.shape
    sorted_probs = np.sort(probs, axis=2)
    confidence = sorted_probs[:, :, -1]
    margin = sorted_probs[:, :, -1] - sorted_probs[:, :, -2]
    entropy = -(probs * np.log(np.clip(probs, 1e-7, 1.0))).sum(axis=2) / np.log(max(classes, 2))
    top = probs.argmax(axis=2)
    disagreement = (top != top[:, :1]).astype(np.float32)
    gap_from_primary = confidence - confidence[:, :1]
    blocks = [
        probs.reshape(rows, -1),
        confidence,
        margin,
        entropy,
        disagreement,
        gap_from_primary,
    ]
    if extra is not None:
        extra = np.asarray(extra, dtype=np.float32)
        if extra.ndim != 2 or extra.shape[0] != rows:
            raise ValueError("extra meta features must have shape [rows, features]")
        blocks.append(extra)
    return np.concatenate(blocks, axis=1).astype(np.float32, copy=False)


def _moments(values: np.ndarray) -> list[np.ndarray]:
    mean = values.mean(axis=1)
    centered = values - mean[:, None]
    variance = (centered**2).mean(axis=1)
    scale = np.sqrt(np.maximum(variance, 1e-8))
    skew = (centered**3).mean(axis=1) / np.maximum(scale**3, 1e-8)
    kurtosis = (centered**4).mean(axis=1) / np.maximum(variance**2, 1e-8)
    return [mean, variance, skew, kurtosis]


def hcs_features(iq: np.ndarray) -> np.ndarray:
    """Low-cost higher-order and temporal descriptors from raw I/Q."""
    iq = np.asarray(iq, dtype=np.float32)
    if iq.ndim != 3 or iq.shape[1] != 2:
        raise ValueError("iq must have shape [rows, 2, length]")
    i = iq[:, 0]
    q = iq[:, 1]
    z = i.astype(np.complex64) + 1j * q.astype(np.complex64)
    amplitude = np.abs(z).astype(np.float32)
    phase_step = np.angle(z[:, 1:] * np.conj(z[:, :-1])).astype(np.float32)
    power = amplitude**2
    blocks = _moments(i) + _moments(q) + _moments(amplitude) + _moments(phase_step)
    c20 = np.mean(z**2, axis=1)
    c21 = np.mean(np.abs(z) ** 2, axis=1)
    c40 = np.mean(z**4, axis=1) - 3.0 * np.mean(z**2, axis=1) ** 2
    c42 = np.mean(np.abs(z) ** 4, axis=1) - np.abs(np.mean(z**2, axis=1)) ** 2 - 2.0 * c21**2
    blocks.extend(
        [
            np.real(c20),
            np.imag(c20),
            c21,
            np.real(c40),
            np.imag(c40),
            np.real(c42),
            np.mean(power[:, 1:] * power[:, :-1], axis=1),
            np.mean(np.real(z[:, 1:] * np.conj(z[:, :-1])), axis=1),
        ]
    )
    features = np.stack(blocks, axis=1).astype(np.float32)
    return np.nan_to_num(features, copy=False)


def gamc_geometry_features(iq: np.ndarray, radial_bins: int = 6, angular_bins: int = 12) -> np.ndarray:
    """Compact constellation-geometry descriptors inspired by graph AMC evidence."""
    iq = np.asarray(iq, dtype=np.float32)
    z = iq[:, 0].astype(np.complex64) + 1j * iq[:, 1].astype(np.complex64)
    radius = np.abs(z)
    radius = radius / np.maximum(np.sqrt(np.mean(radius**2, axis=1, keepdims=True)), 1e-6)
    angle = (np.angle(z) + np.pi) / (2.0 * np.pi)
    features = []
    for row in range(z.shape[0]):
        radial, _ = np.histogram(radius[row], bins=radial_bins, range=(0.0, 3.0), density=False)
        angular, _ = np.histogram(angle[row], bins=angular_bins, range=(0.0, 1.0), density=False)
        points = np.stack((np.real(z[row]), np.imag(z[row])), axis=1)
        points = points[:: max(1, points.shape[0] // 64)][:64]
        distances = np.sqrt(np.maximum(((points[:, None] - points[None, :]) ** 2).sum(axis=2), 0.0))
        distances += np.eye(distances.shape[0], dtype=np.float32) * 1e6
        nearest = np.partition(distances, kth=min(3, distances.shape[1] - 1), axis=1)[:, :3]
        covariance = np.cov(points.T) if points.shape[0] > 1 else np.eye(2)
        eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
        row_features = np.concatenate(
            (
                radial / max(radial.sum(), 1),
                angular / max(angular.sum(), 1),
                np.asarray(
                    [
                        nearest.mean(),
                        nearest.std(),
                        np.quantile(nearest, 0.25),
                        np.quantile(nearest, 0.75),
                        eigenvalues[0],
                        eigenvalues[-1],
                    ]
                ),
            )
        )
        features.append(row_features)
    return np.asarray(features, dtype=np.float32)
