"""Contract tests for the analysis-only MultiFrequency_Mu2024 benchmark."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "test_multifrequency_mu2024_ssvep_decode.py"
spec = importlib.util.spec_from_file_location("mf_decode_analysis", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MultiFrequencyDecodeAnalysisTests(unittest.TestCase):
    def test_cca_score_prefers_matching_frequency(self) -> None:
        fs = 512.0
        n = 512 * 3
        t = np.arange(n) / fs
        trial = np.stack(
            [
                np.sin(2 * np.pi * 19 * t),
                0.7 * np.cos(2 * np.pi * 19 * t),
                np.sin(2 * np.pi * 19 * t + 0.2),
                np.cos(2 * np.pi * 19 * t + 0.3),
                np.sin(2 * np.pi * 19 * t + 0.4),
                np.cos(2 * np.pi * 19 * t + 0.5),
            ]
        )
        prediction, scores = module._decode_trial(trial, fs, 3.0, 1)
        self.assertEqual(prediction, 19.0)
        self.assertGreater(scores[19.0], scores[23.0])

    def test_select_accepts_converter_padding_and_truncated_string(self) -> None:
        data = {
            "frequency_hz": np.asarray([[19, -1, -1], [23, -1, -1], [17, -1, -1]], dtype=np.int16),
            "session": np.asarray([1, 1, 1]),
            "test_method": np.asarray(["T1", "T1", "T1"]),
            "stimulation_type": np.asarray(["singl", "singl", "singl"]),
        }
        np.testing.assert_array_equal(module._select_single_frequency_trials(data), np.asarray([0, 1]))

    def test_run_marks_analysis_only_and_writes_report(self) -> None:
        rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            npz = tmp_path / "sample.npz"
            out = tmp_path / "report.json"
            n_trials, n_samples = 4, 512 * 5
            eeg = rng.normal(size=(n_trials, 6, n_samples)).astype(np.float32)
            np.savez(
                npz,
                eeg=eeg,
                frequency_hz=np.asarray([[19, -1, -1], [19, -1, -1], [23, -1, -1], [23, -1, -1]], dtype=np.int16),
                participant=np.asarray([1, 1, 2, 2], dtype=np.int16),
                session=np.ones(n_trials, dtype=np.int8),
                test_method=np.asarray(["T1"] * n_trials),
                stimulation_type=np.asarray(["singl"] * n_trials),
                source_file=np.asarray(["P01_Ses1.mat"] * n_trials),
            )
            report = module.run(npz, out)
            self.assertTrue(out.is_file())
            self.assertTrue(report["analysis_only"])
            self.assertFalse(report["runtime_enabled"])
            self.assertFalse(report["formal_configuration_modified"])
            self.assertEqual(report["source"]["n_selected_trials"], 4)
            self.assertIn("command_output", report)
            self.assertFalse(report["command_output"]["executed"])
            self.assertEqual(len(report["command_output"]["events"]), 4)
            self.assertTrue(all(event["executed"] is False for event in report["command_output"]["events"]))


if __name__ == "__main__":
    unittest.main()
