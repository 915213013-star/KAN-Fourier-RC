import unittest
from argparse import Namespace

import numpy as np

from oof_protocol import (
    PROTOCOL_ID,
    assert_partition_invariants,
    checkpoint_matches,
    config_fingerprint,
    index_digest,
    make_inner_selection_split,
    protocol_metadata,
)


class OOFProtocolTests(unittest.TestCase):
    def setUp(self):
        # Six samples per class-by-SNR stratum are enough for a deterministic
        # stratified split while keeping this test tiny.
        labels = []
        snrs = []
        for label in range(3):
            for snr in (-2, 0):
                labels.extend([label] * 6)
                snrs.extend([snr] * 6)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.snrs = np.asarray(snrs, dtype=np.int32)
        self.outer_train = np.arange(len(self.labels), dtype=np.int64)

    def test_inner_split_is_disjoint_complete_and_deterministic(self):
        train_a, select_a = make_inner_selection_split(
            self.outer_train, self.labels, self.snrs, seed=41, fraction=1.0 / 3.0
        )
        train_b, select_b = make_inner_selection_split(
            self.outer_train, self.labels, self.snrs, seed=41, fraction=1.0 / 3.0
        )
        assert_partition_invariants(self.outer_train, train_a, select_a)
        self.assertTrue(np.array_equal(train_a, train_b))
        self.assertTrue(np.array_equal(select_a, select_b))
        self.assertEqual(np.intersect1d(train_a, select_a).size, 0)
        for label in range(3):
            for snr in (-2, 0):
                mask = (self.labels[select_a] == label) & (self.snrs[select_a] == snr)
                self.assertEqual(int(mask.sum()), 2)

    def test_outer_holdout_overlap_fails_closed(self):
        inner_train = self.outer_train[:-6]
        inner_select = self.outer_train[-6:]
        with self.assertRaises(RuntimeError):
            assert_partition_invariants(
                self.outer_train,
                inner_train,
                inner_select,
                outer_holdout=np.asarray([0, 100], dtype=np.int64),
            )

    def test_metadata_binds_protocol_indices_and_configuration(self):
        args = Namespace(seed=17, epochs=20, output_cache="ignored-a.npz")
        outer_holdout = np.arange(100, 106, dtype=np.int64)
        inner_train = self.outer_train[:-6]
        inner_select = self.outer_train[-6:]
        meta = protocol_metadata(
            args,
            fold=2,
            phase="outer_refit",
            outer_train_indices=self.outer_train,
            outer_holdout_indices=outer_holdout,
            inner_train_indices=inner_train,
            inner_selection_indices=inner_select,
            selected_epoch=7,
            target_epochs=7,
        )
        self.assertEqual(meta["protocol_id"], PROTOCOL_ID)
        self.assertEqual(meta["outer_train_sha256"], index_digest(self.outer_train))
        self.assertTrue(checkpoint_matches({"protocol_metadata": meta}, meta))
        tampered = dict(meta)
        tampered["selected_epoch"] = 8
        self.assertFalse(checkpoint_matches({"protocol_metadata": meta}, tampered))

    def test_runtime_paths_do_not_change_configuration_fingerprint(self):
        a = Namespace(seed=3, epochs=9, output_cache="one.npz", cache_dir="a")
        b = Namespace(seed=3, epochs=9, output_cache="two.npz", cache_dir="b")
        c = Namespace(seed=4, epochs=9, output_cache="one.npz", cache_dir="a")
        self.assertEqual(config_fingerprint(a), config_fingerprint(b))
        self.assertNotEqual(config_fingerprint(a), config_fingerprint(c))

    def test_invalid_inner_fraction_is_rejected(self):
        for fraction in (0.0, 0.5, 1.0):
            with self.subTest(fraction=fraction):
                with self.assertRaises(ValueError):
                    make_inner_selection_split(
                        self.outer_train, self.labels, self.snrs, seed=1, fraction=fraction
                    )


if __name__ == "__main__":
    unittest.main()
