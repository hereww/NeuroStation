import json
from pathlib import Path
import sys
import unittest
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_open_ssvep_commands import (  # noqa: E402
    benchmark,
    cca_score,
    load_single_trials,
    parse_command_map,
    reference_matrix,
)


class BenchmarkOpenSsvepCommandsTests(unittest.TestCase):
    def test_reference_and_cca_detect_synthetic_frequency(self):
        fs = 512
        n = fs * 4
        t = np.arange(n) / fs
        rng = np.random.default_rng(42)
        clean = np.stack(
            [
                np.sin(2 * np.pi * 19 * t) + 0.1 * np.sin(2 * np.pi * 38 * t),
                0.7 * np.cos(2 * np.pi * 19 * t) + 0.1 * rng.normal(size=n),
            ],
            axis=1,
        )
        wrong = reference_matrix(n, fs, 23, 3)
        right = reference_matrix(n, fs, 19, 3)
        self.assertGreater(cca_score(clean, right), cca_score(clean, wrong) + 0.2)

    def test_load_rejects_runtime_enabled_source(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "source.npz"
            np.savez(path, eeg=np.zeros((1, 1, 128)), participant=np.array([1]), session=np.array([1]), stimulation_type=np.array(["single"]), frequency_hz=np.array([[19, -1, -1]]), sampling_rate_hz=np.array(512), exploratory_only=np.array(True), runtime_enabled=np.array(True))
            with self.assertRaises(ValueError):
                load_single_trials(path)

    def test_benchmark_emits_only_accepted_textual_commands(self):
        fs = 512
        n = fs * 4
        t = np.arange(n) / fs
        rng = np.random.default_rng(7)
        rows = []
        for frequency in (19, 23):
            for repetition in range(4):
                eeg = np.stack(
                    [
                        np.sin(2 * np.pi * frequency * t) + 0.03 * rng.normal(size=n),
                        np.cos(2 * np.pi * frequency * t) + 0.03 * rng.normal(size=n),
                    ],
                    axis=0,
                )
                rows.append(eeg)
        eeg = np.stack(rows)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.npz"
            output = root / "report.json"
            np.savez(
                source,
                eeg=eeg,
                participant=np.ones(8, dtype=np.int16),
                session=np.ones(8, dtype=np.int8),
                stimulation_type=np.array(["single"] * 8),
                frequency_hz=np.array([[19, -1, -1]] * 4 + [[23, -1, -1]] * 4),
                sampling_rate_hz=np.array(fs),
                exploratory_only=np.array(True),
                runtime_enabled=np.array(False),
            )
            report = benchmark(
                source,
                output=output,
                frequencies=[19, 23],
                calibration_repetitions=2,
                evaluation_repetitions=2,
                emit_commands=True,
                score_threshold=0.05,
                margin_threshold=0.01,
            )
            self.assertFalse(report["runtime_enabled"])
            self.assertFalse(report["command_output"]["executed"])
            self.assertGreater(report["summary"]["accepted_trial_count"], 0)
            self.assertTrue(all(row["executed"] is False for row in report["command_output"]["rows"]))
            self.assertTrue(all(row["command"] in {"SELECT", "BACK"} for row in report["command_output"]["rows"]))
            loaded = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(loaded["command_output"]["mapping"]["19"], "SELECT")

    def test_command_map_requires_all_targets(self):
        with self.assertRaises(ValueError):
            parse_command_map("19=SELECT", [19, 23])


if __name__ == "__main__":
    unittest.main()
