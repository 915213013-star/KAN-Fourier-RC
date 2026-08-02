import unittest

import numpy as np
import torch

from reference_methods.cache_io import CandidateCache, merge_prediction_caches
from reference_methods.features import gamc_geometry_features, hcs_features, probability_meta_features
from reference_methods.long_spectral_lite import LongSpectralLite
from reference_methods.policy import apply_candidate_actions


class ReferenceMethodTests(unittest.TestCase):
    def test_merge_prediction_caches_aligns_sample_ids(self):
        primary = CandidateCache(
            probabilities=np.asarray(
                [[[0.8, 0.2]], [[0.1, 0.9]], [[0.7, 0.3]]], dtype=np.float32
            ),
            labels=np.asarray([0, 1, 0]),
            candidate_names=("primary",),
            sample_ids=np.asarray([10, 20, 30]),
        )
        auxiliary = CandidateCache(
            probabilities=np.asarray(
                [[[0.4, 0.6]], [[0.6, 0.4]], [[0.2, 0.8]]], dtype=np.float32
            ),
            labels=np.asarray([0, 0, 1]),
            candidate_names=("auxiliary",),
            sample_ids=np.asarray([30, 10, 20]),
        )

        merged = merge_prediction_caches(primary, {"auxiliary": auxiliary})

        self.assertEqual(merged.candidate_names, ("primary", "auxiliary"))
        np.testing.assert_array_equal(merged.sample_ids, primary.sample_ids)
        np.testing.assert_allclose(
            merged.probabilities[:, 1],
            np.asarray([[0.6, 0.4], [0.2, 0.8], [0.4, 0.6]], dtype=np.float32),
        )

    def test_merge_prediction_caches_rejects_label_mismatch(self):
        primary = CandidateCache(
            probabilities=np.asarray([[[0.8, 0.2]], [[0.1, 0.9]]], dtype=np.float32),
            labels=np.asarray([0, 1]),
            candidate_names=("primary",),
            sample_ids=np.asarray([10, 20]),
        )
        auxiliary = CandidateCache(
            probabilities=np.asarray([[[0.7, 0.3]], [[0.3, 0.7]]], dtype=np.float32),
            labels=np.asarray([1, 1]),
            candidate_names=("auxiliary",),
            sample_ids=np.asarray([10, 20]),
        )

        with self.assertRaisesRegex(ValueError, "label"):
            merge_prediction_caches(primary, {"auxiliary": auxiliary})

    def test_reference_features_are_finite(self):
        rng = np.random.default_rng(7)
        iq = rng.normal(size=(8, 2, 128)).astype(np.float32)
        probabilities = rng.uniform(size=(8, 4, 6)).astype(np.float32)
        probabilities /= probabilities.sum(axis=2, keepdims=True)

        for features in (
            hcs_features(iq),
            gamc_geometry_features(iq),
            probability_meta_features(probabilities),
        ):
            self.assertEqual(features.shape[0], iq.shape[0])
            self.assertTrue(np.isfinite(features).all())

    def test_action_policy_respects_change_budget(self):
        probabilities = np.asarray(
            [
                [[0.9, 0.1], [0.1, 0.9]],
                [[0.8, 0.2], [0.2, 0.8]],
                [[0.7, 0.3], [0.3, 0.7]],
                [[0.6, 0.4], [0.4, 0.6]],
                [[0.9, 0.1], [0.1, 0.9]],
            ],
            dtype=np.float32,
        )
        output = apply_candidate_actions(
            probabilities,
            candidate_indices=np.ones(5, dtype=np.int64),
            action_scores=np.asarray([5, 4, 3, 2, 1], dtype=np.float32),
            threshold=0.0,
            max_change_rate=0.4,
        )
        changed = np.any(np.abs(output - probabilities[:, 0]) > 1e-7, axis=1)
        self.assertEqual(int(changed.sum()), 2)

    def test_long_spectral_lite_forward_shape(self):
        model = LongSpectralLite(class_count=5, width=8, pooled_bins=4, dropout=0.0)
        logits = model(torch.randn(3, 2, 128))

        self.assertEqual(tuple(logits.shape), (3, 5))
        self.assertTrue(torch.isfinite(logits).all().item())


if __name__ == "__main__":
    unittest.main()
