from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from neurostation_diagnostics import DiagnosticStore


class DiagnosticStoreTests(unittest.TestCase):
    def test_events_are_rolling_jsonl_and_json_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = DiagnosticStore(root=root, max_events=20)
            event = store.info(
                "test",
                "recorded event",
                context={"path": root / "sample", "values": {1, 2}},
            )
            for index in range(25):
                store.info("test", f"rolling event {index}")

            self.assertEqual("INFO", event["level"])
            self.assertEqual(20, len(store.events()))
            self.assertEqual("rolling event 24", store.events(1)[0]["message"])
            self.assertEqual((), store.events(0))
            lines = store.log_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(26, len(lines))
            self.assertEqual(
                str(root / "sample"),
                json.loads(lines[0])["context"]["path"],
            )

            restored = DiagnosticStore(root=root, max_events=20)
            self.assertEqual(store.events(), restored.events())

    def test_report_and_export_include_checks_and_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = DiagnosticStore(root=Path(directory), max_events=20)
            store.error("test", "exported event")
            report = store.build_report()

            self.assertIn("application", report)
            self.assertIn("platform", report)
            self.assertIn("checks", report)
            self.assertTrue(report["checks"])
            self.assertIn("runtime", report)
            self.assertEqual(1, report["runtime"]["event_count"])

            output = store.export_report(Path(directory) / "report.json")
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(output, Path(directory) / "report.json")
            self.assertEqual("exported event", payload["events"][0]["message"])
            self.assertTrue(payload["checks"])


if __name__ == "__main__":
    unittest.main()
