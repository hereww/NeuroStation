from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from eeg_tools.offline_analysis import analyze_session


class OfflineAnalysisTests(unittest.TestCase):
    def test_fft_baseline_scores_injected_target_and_writes_audit_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session_dir = Path(directory) / "session"
            session_dir.mkdir()
            sample_rate = 250
            samples = 5 * sample_rate
            time_axis = np.arange(samples) / sample_rate
            eeg = 2.0 * np.sin(2 * np.pi * 12 * time_axis) + 0.4 * np.sin(2 * np.pi * 24 * time_axis)
            raw_path = session_dir / "raw_brainflow.tsv"
            with raw_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(["sample_index", "eeg_ch1", "timestamp_s", "marker"])
                for index, value in enumerate(eeg):
                    marker = 110 if index == 0 else 210 if index == samples - 1 else 0
                    writer.writerow([index, value, index / sample_rate, marker])
            with (session_dir / "events.tsv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["event_name", "trial_index", "target_id", "frequency_hz", "marker_code", "sample_index"],
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerow({"event_name": "stimulus_onset", "trial_index": 0, "target_id": "top_right", "frequency_hz": 12, "marker_code": 110, "sample_index": 0})
                writer.writerow({"event_name": "stimulus_offset", "trial_index": 0, "target_id": "top_right", "frequency_hz": 12, "marker_code": 210, "sample_index": samples - 1})
            (session_dir / "session.json").write_text(
                json.dumps({"session_id": "test", "sampling_rate_hz": sample_rate}), encoding="utf-8"
            )
            (session_dir / "protocol.json").write_text(
                json.dumps({"targets": [["top_left", 10], ["top_right", 12]]}), encoding="utf-8"
            )

            summary = analyze_session(session_dir)
            self.assertEqual(1, summary["trial_count"])
            self.assertEqual(1.0, summary["presented_target_accuracy"])
            self.assertTrue((session_dir / "analysis.json").is_file())
            self.assertTrue((session_dir / "trial_features.tsv").is_file())
            with (session_dir / "trial_features.tsv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual("top_right", rows[0]["eeg_predicted_target_id"])


if __name__ == "__main__":
    unittest.main()
