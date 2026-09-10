from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from eeg_tools.session_files import sha256, write_events, write_json, write_manifest


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


if __name__ == "__main__":
    unittest.main()
