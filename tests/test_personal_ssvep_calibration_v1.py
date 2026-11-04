"""Contract tests for personal calibration v1."""
from __future__ import annotations

import importlib.abc
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from eeg_tools import personal_ssvep_calibration_v1 as cal


class PersonalCalibrationV1Tests(unittest.TestCase):
    TARGETS = ["a", "b", "c", "d"]

    def test_formal_config_and_fail_closed_flags(self):
        config = cal.get_config()
        self.assertEqual(config["status"], "formal")
        self.assertEqual(config["minimum_repetitions_per_target"], 8)
        self.assertEqual(config["selected_channel_indices"], [4, 5, 6, 7])
        self.assertFalse(config["runtime_enabled"])
        self.assertFalse(config["default_deployment_authorized"])

    def test_insufficient_repetitions_are_ineligible(self):
        scores = np.tile(np.asarray([[0.9, 0.1, 0.1, 0.1]], dtype=float), (12, 1))
        labels = [self.TARGETS[index % 4] for index in range(12)]
        model = cal.fit_calibrator(scores, labels, self.TARGETS)
        self.assertFalse(model["eligible"])
        self.assertFalse(model["runtime_enabled"])
        self.assertEqual(model["reason"], "minimum_repetitions_per_target_not_met")
        result = cal.predict(model, scores[0])
        self.assertFalse(result["accepted"])
        self.assertIsNone(result["target_id"])

    def test_synthetic_calibrator_supports_unknown_and_acceptance(self):
        rng = np.random.default_rng(20261014)
        rows, labels = [], []
        for index, target in enumerate(self.TARGETS):
            for _ in range(8):
                row = rng.normal(0.05, 0.01, 4)
                row[index] = 0.8 + rng.normal(0, 0.01)
                rows.append(row)
                labels.append(target)
        model = cal.fit_calibrator(np.asarray(rows), labels, self.TARGETS)
        model["holdout_validation_passed"] = True
        result = cal.predict(model, np.asarray([0.85, 0.04, 0.03, 0.02]))
        self.assertTrue(result["available"])
        self.assertTrue(result["accepted"])
        self.assertEqual(result["target_id"], "a")
        unknown = cal.predict(model, np.asarray([0.1, 0.1, 0.1, 0.1]))
        self.assertFalse(unknown["accepted"])
        self.assertIsNone(unknown["target_id"])
        self.assertFalse(result["confidence_is_probability"])

    def test_labels_are_only_calibration_inputs(self):
        scores = np.arange(128, dtype=float).reshape(32, 4) / 100.0
        labels = [self.TARGETS[index % 4] for index in range(32)]
        model_a = cal.fit_calibrator(scores, labels, self.TARGETS, minimum_repetitions=2, allow_ineligible=True)
        changed = list(labels)
        changed[0], changed[1] = changed[1], changed[0]
        model_b = cal.fit_calibrator(scores, changed, self.TARGETS, minimum_repetitions=2, allow_ineligible=True)
        self.assertNotEqual(model_a["parameters"]["a"]["positive_mean"], model_b["parameters"]["a"]["positive_mean"])

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            cal.fit_calibrator(np.zeros((4, 3)), ["a"] * 4, self.TARGETS)
        result = cal.predict({"target_ids": self.TARGETS}, np.zeros(4))
        self.assertFalse(result["available"])
        self.assertFalse(result["accepted"])

    def test_help_is_lazy(self):
        code = """
import importlib.abc, runpy, sys
blocked = {'numpy', 'scipy'}
assert not blocked.intersection(sys.modules)
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('scientific import during help')
sys.meta_path.insert(0, Block())
sys.argv = ['evaluate_personal_ssvep_calibration_v1.py', '--help']
try:
    runpy.run_module('scripts.evaluate_personal_ssvep_calibration_v1', run_name='__main__')
except SystemExit as exc:
    assert exc.code == 0
"""
        result = subprocess.run([sys.executable, "-S", "-B", "-c", code], cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()




