"""Contract tests for the offline-only formal personal SSVEP v4 candidate."""
from __future__ import annotations

import copy
import importlib.abc
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from eeg_tools import personal_ssvep_decoder_v4 as decoder


class PersonalDecoderV4Tests(unittest.TestCase):
    TARGETS = [("a", 10.0), ("b", 12.0), ("c", 15.0), ("d", 20.0)]
    FS = 250.0
    SAMPLES = 750

    @classmethod
    def signal_pair(cls, frequency: float, seed: int, amplitude: float = 1.0):
        rng = np.random.default_rng(seed)
        t = np.arange(cls.SAMPLES, dtype=np.float64) / cls.FS
        stimulus = rng.normal(0.0, 0.025, (8, cls.SAMPLES))
        rest = rng.normal(0.0, 0.025, (8, cls.SAMPLES))
        for channel in range(8):
            phase = 0.17 * channel
            stimulus[channel] += amplitude * np.sin(2 * np.pi * frequency * t + phase)
        return stimulus, rest

    @classmethod
    def calibration(cls):
        sessions = []
        for target_index, (target_id, frequency) in enumerate(cls.TARGETS):
            for repetition in range(3):
                signal, rest = cls.signal_pair(frequency, 1000 + target_index * 10 + repetition)
                sessions.append({"target_id": target_id, "signal": signal, "rest_signal": rest})
        return decoder.fit_calibration(sessions, cls.FS, cls.TARGETS)

    def test_formal_config_is_locked_and_candidate_only(self):
        config = decoder.get_config()
        self.assertEqual(config["status"], "formal")
        self.assertEqual(config["model_version"], "personal_ssvep_cca_v4_adaptive_candidate")
        self.assertEqual(config["selected_channel_indices"], [4, 5, 6, 7])
        self.assertEqual(config["selected_channel_names"], ["CH5", "CH6", "CH7", "CH8"])
        self.assertFalse(config["runtime_enabled"])
        self.assertFalse(config["production_validated"])
        self.assertFalse(config["default_deployment_authorized"])
        self.assertFalse(config["source_domain_validated"])
        self.assertFalse(config["gaze_accuracy_claim"])
        self.assertFalse(config["presented_matches_are_gaze_accuracy"])
        self.assertTrue(config["recorded_presentation_labels_are_proxy"])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            changed = copy.deepcopy(config)
            changed["runtime_enabled"] = True
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(ValueError):
                decoder.validate_config(path)

    def test_calibration_has_target_centroids_and_robust_normalization(self):
        model = self.calibration()
        self.assertTrue(model["eligible"])
        self.assertEqual(model["selected_channel_indices"], [4, 5, 6, 7])
        self.assertEqual(len(model["target_centroids"]), len(self.TARGETS))
        self.assertEqual(model["feature_dimension"], 2 * len(self.TARGETS))
        self.assertEqual(model["label_source"], "recorded_presentation_labels_proxy_only_not_independent_gaze_truth")
        self.assertFalse(model["source_domain_validated"])
        decoder.validate_calibration_model(model)

    def test_synthetic_target_is_accepted_and_deterministic(self):
        model = self.calibration()
        signal, rest = self.signal_pair(10.0, 9001)
        first = decoder.decode_window(signal, rest, self.FS, self.TARGETS, model)
        second = decoder.decode_window(signal, rest, self.FS, self.TARGETS, model)
        self.assertEqual(first, second)
        self.assertTrue(first["available"])
        self.assertTrue(first["accepted"])
        self.assertEqual(first["target_id"], "a")
        self.assertIsNone(first["rejection_reason"])
        self.assertTrue(all(item["passed"] for item in first["gates"].values()))
        self.assertFalse(first["confidence_is_probability"])
        self.assertFalse(first["source_domain_validated"])

    def test_unknown_gates_are_fail_closed(self):
        model = self.calibration()
        rng = np.random.default_rng(77)
        stimulus = rng.normal(0.0, 0.05, (8, self.SAMPLES))
        rest = rng.normal(0.0, 0.05, (8, self.SAMPLES))
        result = decoder.decode_window(stimulus, rest, self.FS, self.TARGETS, model)
        self.assertTrue(result["available"])
        self.assertFalse(result["accepted"])
        self.assertIsNone(result["target_id"])
        self.assertIn(result["rejection_reason"], {"insufficient_physiological_evidence", "ambiguous_calibrated_evidence"})

    def test_input_validation_is_strict(self):
        model = self.calibration()
        with self.assertRaises(ValueError):
            decoder.decode_window(np.zeros((4, self.SAMPLES)), np.zeros((8, self.SAMPLES)), self.FS, self.TARGETS, model)
        with self.assertRaises(ValueError):
            decoder.decode_window(np.zeros((8, self.SAMPLES)), np.zeros((8, self.SAMPLES)), 249.0, self.TARGETS, model)
        with self.assertRaises(ValueError):
            decoder.decode_window(np.zeros((8, self.SAMPLES)), np.zeros((8, 200)), self.FS, self.TARGETS, model)
        with self.assertRaises(ValueError):
            decoder.fit_calibration([], self.FS, self.TARGETS)
        with self.assertRaises(ValueError):
            decoder.fit_calibration([{"target_id": "a", "signal": np.zeros((8, self.SAMPLES)), "rest_signal": np.zeros((8, self.SAMPLES))}], self.FS, self.TARGETS)

    def test_decode_does_not_consume_recorded_presentation_labels(self):
        model = self.calibration()
        signal, rest = self.signal_pair(10.0, 9003)
        first = decoder.decode_window(signal, rest, self.FS, self.TARGETS, model)
        relabeled_targets = [("presentation_label_changed", 10.0), ("other_b", 12.0), ("other_c", 15.0), ("other_d", 20.0)]
        second = decoder.decode_window(signal, rest, self.FS, relabeled_targets, model)
        self.assertEqual(first["target_id"], "a")
        self.assertEqual(second["target_id"], "presentation_label_changed")
        self.assertEqual(list(first["stimulus_scores"].values()), list(second["stimulus_scores"].values()))
        self.assertEqual(list(first["centroid_distances"].values()), list(second["centroid_distances"].values()))
        self.assertFalse(second["presented_matches_are_gaze_accuracy"])

    def test_explicit_channel_selection_must_match_calibration_and_default_is_prior(self):
        model = self.calibration()
        signal, rest = self.signal_pair(10.0, 9003)
        result = decoder.decode_window(signal, rest, self.FS, self.TARGETS, model, selected_channel_indices=[4, 5, 6, 7])
        self.assertEqual(result["selected_channel_indices"], [4, 5, 6, 7])
        with self.assertRaises(ValueError):
            decoder.decode_window(signal, rest, self.FS, self.TARGETS, model, selected_channel_indices=[3, 4, 5, 6])

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
sys.argv = ['evaluate_personal_ssvep_decoder_v4.py', '--help']
try:
    runpy.run_module('scripts.evaluate_personal_ssvep_decoder_v4', run_name='__main__')
except SystemExit as exc:
    assert exc.code == 0
"""
        result = subprocess.run([sys.executable, "-S", "-B", "-c", code], cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()

