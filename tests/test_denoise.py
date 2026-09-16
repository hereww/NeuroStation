from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from eeg_tools.denoise import run_denoise_pipeline


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "configs" / "denoise_pipeline_v1.json"


class DenoisePipelineTests(unittest.TestCase):
    def _make_session(self, root: Path, *, add_long_gap: bool = False) -> tuple[Path, bytes]:
        session = root / "session_20260916_000000"
        session.mkdir()
        sample_rate = 250
        samples = 10 * sample_rate
        time_axis = np.arange(samples) / sample_rate
        eeg = np.vstack([
            2.0 * np.sin(2 * np.pi * 12 * time_axis)
            + 0.5 * np.sin(2 * np.pi * 24 * time_axis)
            + 0.8 * np.sin(2 * np.pi * 50 * time_axis)
            + 0.7 * np.sin(2 * np.pi * 60 * time_axis)
            for _ in range(8)
        ])
        eeg[0, 30:35] = np.nan
        if add_long_gap:
            eeg[1, 700:800] = np.nan
        timestamps = np.arange(samples) / sample_rate
        raw_path = session / "raw_brainflow.tsv"
        with raw_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["sample_index", *[f"eeg_ch{i}" for i in range(1, 9)], "timestamp_s", "marker"])
            for index in range(samples):
                marker = 110 if index == 125 else 210 if index == 1125 else 0
                writer.writerow([index, *eeg[:, index], timestamps[index], marker])
        with (session / "raw_columns.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["column_name", "brainflow_row", "semantic", "unit", "role", "sampling_rate_hz"])
            for index, name in enumerate(["sample_index", *[f"eeg_ch{i}" for i in range(1, 9)], "timestamp_s", "marker"]):
                writer.writerow([name, index, name, "unit", "raw", sample_rate])
        with (session / "events.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["event_name", "trial_index", "target_id", "frequency_hz", "marker_code", "sample_index"], delimiter="\t")
            writer.writeheader()
            writer.writerow({"event_name": "stimulus_onset", "trial_index": 0, "target_id": "top_right", "frequency_hz": 12, "marker_code": 110, "sample_index": 125})
            writer.writerow({"event_name": "stimulus_offset", "trial_index": 0, "target_id": "top_right", "frequency_hz": 12, "marker_code": 210, "sample_index": 1125})
        (session / "session.json").write_text(json.dumps({"session_id": session.name, "sampling_rate_hz": sample_rate}), encoding="utf-8")
        (session / "protocol.json").write_text(json.dumps({"targets": [["top_left", 10], ["top_right", 12]], "trial": {"stimulus_s": 4}}), encoding="utf-8")
        raw_before = raw_path.read_bytes()
        return session, raw_before

    def test_generates_two_tracks_and_preserves_raw(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session, raw_before = self._make_session(Path(directory))
            result = run_denoise_pipeline(session, PIPELINE)
            output = session / "derived" / "denoise_v1"
            self.assertEqual("completed", result["status"])
            self.assertTrue((output / "denoised_50hz.npz").is_file())
            self.assertTrue((output / "denoised_60hz.npz").is_file())
            self.assertTrue((output / "artifact_segments.tsv").is_file())
            self.assertEqual(raw_before, (session / "raw_brainflow.tsv").read_bytes())
            metrics = json.loads((output / "denoising_metrics.json").read_text(encoding="utf-8"))
            self.assertIn("annotated", metrics["tracks"])
            self.assertIn("50hz", metrics["tracks"])
            self.assertIn("60hz", metrics["tracks"])
            self.assertEqual(1, metrics["analysis"]["50hz"]["trial_count"])

    def test_target_is_retained_and_line_noise_is_reduced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session, _ = self._make_session(Path(directory))
            run_denoise_pipeline(session, PIPELINE)
            metrics = json.loads((session / "derived" / "denoise_v1" / "denoising_metrics.json").read_text(encoding="utf-8"))
            before = metrics["tracks"]["annotated"]
            after = metrics["tracks"]["50hz"]
            self.assertGreater(after["target_snr_db"]["top_right"], 0.0)
            self.assertLess(after["line_noise_ratio"]["50.0"], before["line_noise_ratio"]["50.0"] / 50.0)

    def test_long_gap_remains_invalid_and_short_gap_is_repaired(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session, _ = self._make_session(Path(directory), add_long_gap=True)
            run_denoise_pipeline(session, PIPELINE)
            with np.load(session / "derived" / "denoise_v1" / "denoised_50hz.npz") as values:
                self.assertTrue(np.isfinite(values["eeg"][0, 32]))
                self.assertTrue(np.isnan(values["eeg"][1, 750]))
            artifact_text = (session / "derived" / "denoise_v1" / "artifact_segments.tsv").read_text(encoding="utf-8")
            self.assertIn("nonfinite", artifact_text)

    def test_same_inputs_are_cached_and_manifest_hashes_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session, _ = self._make_session(Path(directory))
            first = run_denoise_pipeline(session, PIPELINE)
            second = run_denoise_pipeline(session, PIPELINE)
            self.assertEqual("up_to_date", second["status"])
            self.assertEqual(first["artifact_segment_count"], second["artifact_segment_count"])
            with (session / "derived" / "denoise_v1" / "derived_manifest.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                if row["role"] != "derived":
                    continue
                path = session / "derived" / "denoise_v1" / row["file_name"]
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(digest, row["sha256"])

    def test_pipeline_hash_change_recomputes_and_manifest_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session, _ = self._make_session(root)
            custom_pipeline = root / "pipeline.json"
            custom_pipeline.write_text(PIPELINE.read_text(encoding="utf-8"), encoding="utf-8")
            first = run_denoise_pipeline(session, custom_pipeline)
            custom_value = json.loads(custom_pipeline.read_text(encoding="utf-8"))
            custom_value["notch_quality_factor"] = 25.0
            custom_pipeline.write_text(json.dumps(custom_value), encoding="utf-8")
            second = run_denoise_pipeline(session, custom_pipeline)
            self.assertEqual("completed", first["status"])
            self.assertEqual("completed", second["status"])
            manifest = session / "manifest.csv"
            manifest.write_text("file_name,sha256\nraw_brainflow.tsv,deadbeef\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash does not match"):
                run_denoise_pipeline(session, custom_pipeline)

    def test_cli_returns_structured_error_for_missing_session(self) -> None:
        from preprocess_eeg_session import main

        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            self.assertEqual(2, main([str(missing), "--pipeline", str(PIPELINE)]))


if __name__ == "__main__":
    unittest.main()
