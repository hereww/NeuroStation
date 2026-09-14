from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from eeg_tools.session_files import (
    brainflow_column_definitions,
    sha256,
    write_brainflow_tsv,
    write_events,
    write_json,
    write_manifest,
)


class SessionFileTests(unittest.TestCase):
    def test_writers_create_a_verifiable_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            event_path = root / "events.tsv"
            session_path = root / "session.json"
            manifest_path = root / "session_manifest.csv"
            write_events(event_path, [])
            write_json(session_path, {"session_id": "test"})
            write_manifest(manifest_path, "test", [event_path, session_path])

            with manifest_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(["events.tsv", "session.json"], [row["file_name"] for row in rows])
            self.assertEqual(sha256(event_path), rows[0]["sha256"])

    def test_brainflow_writer_emits_explicit_header_and_column_definitions(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.tsv"
            definitions = write_brainflow_tsv(
                path, np.array([[7.0, 8.0], [1.0, 2.0]]), None, sampling_rate_hz=250
            )
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual("sample_index\tboard_row_0\tboard_row_1", lines[0])
            self.assertEqual("0\t7\t1", lines[1])
            definitions_for_cyton = brainflow_column_definitions(6, 32)
            self.assertIn("timestamp_s", {item["column_name"] for item in definitions_for_cyton})
            self.assertIn("marker", {item["column_name"] for item in definitions_for_cyton})
            self.assertEqual(3, len(definitions))


if __name__ == "__main__":
    unittest.main()
