"""Contract tests for the read-only personal SSVEP filterbank experiment."""
from __future__ import annotations

import unittest

import numpy as np

from scripts import experiment_personal_ssvep_filterbank_20261014 as experiment


class PersonalSsvepFilterbankExperimentTests(unittest.TestCase):
    def test_contract_is_exploratory_and_fail_closed(self) -> None:
        config = experiment.validate_experiment_config()
        self.assertTrue(config["exploratory_only"])
        self.assertFalse(config["runtime_enabled"])
        self.assertFalse(config["production_validated"])
        self.assertFalse(config["default_deployment_authorized"])
        self.assertFalse(config["presented_matches_are_gaze_accuracy"])
        self.assertFalse(config["forward_uses_recorded_labels"])
        self.assertTrue(config["evaluation_is_strict_cross_session_holdout"])

    def test_preregistered_channel_subsets_are_fixed(self) -> None:
        self.assertEqual(experiment.PREREGISTERED_CHANNEL_SUBSETS["posterior_ch5_ch8"], (4, 5, 6, 7))
        with self.assertRaises(KeyError):
            experiment.PREREGISTERED_CHANNEL_SUBSETS["data_driven_best"]

    def test_cca_score_is_finite_and_deterministic(self) -> None:
        rng = np.random.default_rng(20261014)
        signal = rng.normal(size=(8, 1000))
        first = experiment.cca_score(signal, 10.0, 2, (4, 5, 6, 7))
        second = experiment.cca_score(signal, 10.0, 2, (4, 5, 6, 7))
        self.assertTrue(np.isfinite(first))
        self.assertGreaterEqual(first, 0.0)
        self.assertLessEqual(first, 1.0)
        self.assertEqual(first, second)

    def test_invalid_channel_subset_is_rejected(self) -> None:
        rng = np.random.default_rng(7)
        signal = rng.normal(size=(8, 1000))
        with self.assertRaises(ValueError):
            experiment.cca_score(signal, 10.0, 1, (0, 0))


if __name__ == "__main__":
    unittest.main()
