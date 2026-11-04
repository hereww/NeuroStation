"""Contract tests for the formal evidence-gated v3 personal decoder."""
from __future__ import annotations

import importlib.abc
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from eeg_tools import personal_ssvep_decoder_v3 as decoder


class PersonalDecoderV3Tests(unittest.TestCase):
    TARGETS = [("a", 10.0), ("b", 12.0), ("c", 15.0), ("d", 20.0)]

    def test_formal_config_is_evidence_gated_and_fail_closed(self):
        config = decoder.get_config()
        self.assertEqual(config["status"], "formal")
        self.assertEqual(config["target_selection"], "evidence_gated_unknown_allowed")
        self.assertFalse(config["detection_validated"])
        self.assertFalse(config["runtime_enabled"])
        self.assertFalse(config["default_deployment_authorized"])
        self.assertEqual(config["selected_channel_indices"], [4, 5, 6, 7])

    def test_synthetic_target_is_accepted_with_rest_evidence(self):
        rng = np.random.default_rng(20261014)
        fs, samples = 250.0, 750
        t = np.arange(samples, dtype=np.float64) / fs
        stimulus = rng.normal(0.0, 0.05, (8, samples))
        rest = rng.normal(0.0, 0.05, (8, samples))
        for channel in range(8):
            phase = 0.2 * channel
            stimulus[channel] += np.sin(2 * np.pi * 10.0 * t + phase)
            rest[channel] += 0.2 * np.sin(2 * np.pi * 12.0 * t + phase)
        result = decoder.decode_window(stimulus, rest, fs, self.TARGETS)
        self.assertTrue(result["available"])
        self.assertTrue(result["accepted"])
        self.assertEqual(result["target_id"], "a")
        self.assertEqual(result["rejection_reason"], None)
        self.assertTrue(all(item["passed"] for item in result["gates"].values()))
        self.assertFalse(result["confidence_is_probability"])

    def test_noise_or_rest_dominant_signal_returns_unknown(self):
        rng = np.random.default_rng(7)
        stimulus = rng.normal(0.0, 0.05, (8, 750))
        rest = rng.normal(0.0, 0.05, (8, 750))
        result = decoder.decode_window(stimulus, rest, 250.0, self.TARGETS)
        self.assertTrue(result["available"])
        self.assertFalse(result["accepted"])
        self.assertIsNone(result["target_id"])
        self.assertIn(result["rejection_reason"], {"insufficient_physiological_evidence", "ambiguous_evidence"})

    def test_forward_does_not_accept_recorded_labels(self):
        rng = np.random.default_rng(8)
        fs, samples = 250.0, 750
        t = np.arange(samples, dtype=np.float64) / fs
        stimulus = rng.normal(0.0, 0.05, (8, samples))
        rest = rng.normal(0.0, 0.05, (8, samples))
        for channel in range(8):
            phase = 0.2 * channel
            stimulus[channel] += np.sin(2 * np.pi * 10.0 * t + phase)
            rest[channel] += 0.2 * np.sin(2 * np.pi * 12.0 * t + phase)
        first = decoder.decode_window(stimulus, rest, fs, self.TARGETS)
        second = decoder.decode_window(stimulus, rest, fs, [("changed_label", 10.0), ("b", 12.0), ("c", 15.0), ("d", 20.0)])
        self.assertEqual(first["target_id"], "a")
        self.assertEqual(second["target_id"], "changed_label")
        self.assertAlmostEqual(first["canonical_score"], second["canonical_score"], places=10)
        self.assertFalse(first["gaze_accuracy_claim"])

    def test_bad_shape_rate_and_missing_rest_are_rejected(self):
        with self.assertRaises(ValueError):
            decoder.decode_window(np.zeros((4, 750)), np.zeros((8, 750)), 250.0, self.TARGETS)
        with self.assertRaises(ValueError):
            decoder.decode_window(np.zeros((8, 750)), np.zeros((8, 750)), 249.0, self.TARGETS)
        with self.assertRaises(ValueError):
            decoder.decode_window(np.zeros((8, 750)), np.zeros((8, 200)), 250.0, self.TARGETS)

    def test_help_import_is_lazy(self):
        code = """
import importlib.abc, runpy, sys
blocked = {'numpy', 'scipy'}
assert not blocked.intersection(sys.modules)
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('scientific import during evaluator help')
sys.meta_path.insert(0, Block())
sys.argv = ['evaluate_personal_ssvep_decoder_v3.py', '--help']
try:
    runpy.run_module('scripts.evaluate_personal_ssvep_decoder_v3', run_name='__main__')
except SystemExit as exc:
    assert exc.code == 0
"""
        result = subprocess.run([sys.executable, "-S", "-B", "-c", code], cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()

