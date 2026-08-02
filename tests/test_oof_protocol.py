import unittest
from argparse import Namespace

import numpy as np

from oof_protocol import (
    PROTOCOL_ID,
    assert_fold_partition,
    checkpoint_matches,
    config_fingerprint,
    index_digest,
    protocol_metadata,
)


class OOFProtocolTests(unittest.TestCase):
    def setUp(self):
        self.train = np.arange(0, 24, dtype=np.int64)
        self.holdout = np.arange(24, 30, dtype=np.int64)
        self.validation = np.arange(30, 36, dtype=np.int64)
        self.test = np.arange(36, 42, dtype=np.int64)

    def test_all_protocol_partitions_are_disjoint(self):
        assert_fold_partition(self.train, self.holdout, self.validation, self.test)
        with self.assertRaises(RuntimeError):
            assert_fold_partition(self.train, np.asarray([2, 25]), self.validation, self.test)
        with self.assertRaises(RuntimeError):
            assert_fold_partition(self.train, self.holdout, np.asarray([31, 37]), self.test)

    def test_metadata_records_the_reported_fold_selection_protocol(self):
        args = Namespace(seed=17, epochs=20, output_cache="ignored-a.npz")
        meta = protocol_metadata(
            args,
            fold=2,
            phase="fold_training",
            outer_train_indices=self.train,
            outer_holdout_indices=self.holdout,
            target_epochs=20,
            policy_validation_indices=self.validation,
            official_test_indices=self.test,
        )
        self.assertEqual(meta["protocol_id"], PROTOCOL_ID)
        self.assertEqual(meta["outer_train_sha256"], index_digest(self.train))
        self.assertEqual(meta["checkpoint_selection"], "outer_holdout_fold")
        self.assertEqual(meta["policy_selection"], "independent_validation_only")
        self.assertEqual(meta["official_test_usage"], "frozen_evaluation_only")
        self.assertFalse(meta["nested_cv_claim"])
        self.assertTrue(checkpoint_matches({"protocol_metadata": meta}, meta))
        tampered = dict(meta)
        tampered["outer_holdout_sha256"] = "0" * 64
        self.assertFalse(checkpoint_matches({"protocol_metadata": meta}, tampered))

    def test_runtime_paths_do_not_change_configuration_fingerprint(self):
        a = Namespace(seed=3, epochs=9, output_cache="one.npz", cache_dir="a")
        b = Namespace(seed=3, epochs=9, output_cache="two.npz", cache_dir="b")
        c = Namespace(seed=4, epochs=9, output_cache="one.npz", cache_dir="a")
        self.assertEqual(config_fingerprint(a), config_fingerprint(b))
        self.assertNotEqual(config_fingerprint(a), config_fingerprint(c))

    def test_duplicate_indices_fail_closed(self):
        with self.assertRaises(RuntimeError):
            assert_fold_partition(np.asarray([0, 0, 1]), self.holdout)


if __name__ == "__main__":
    unittest.main()
