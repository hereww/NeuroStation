"""Contract tests for post-hoc temporal-confirmation diagnostics."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest

from scripts import analyze_ssvep_temporal_confirmation as analysis


def row(trial_id: str, offset: int, frequency: int, action: str, accepted: bool = True):
    return {"trial_id": trial_id, "subject": 25, "block": 0,
            "frequency_hz": frequency, "window_start_offset_samples": offset,
            "ood_passed": True,
            "raw": {"accepted": accepted, "command": action}}


def report(rows):
    return {"scope": "Wang2016 cross-subject 10/12/15 Hz; exact 1000-sample live-game window",
            "split": "validation", "subjects": [25, 26, 27, 28, 29],
            "window_samples": 1000, "window_offsets_samples": [0, 250],
            "personal_validation": False, "final_test_evaluated": False,
            "rows": rows}


class TemporalConfirmationTests(unittest.TestCase):
    def test_requires_two_accepted_matching_moves_and_counts_per_trial(self):
        rows = [row("agree", 0, 10, "MOVE_LEFT"), row("agree", 250, 10, "MOVE_LEFT"),
                row("disagree", 0, 12, "MOVE_RIGHT"), row("disagree", 250, 12, "MOVE_LEFT"),
                row("reject", 0, 10, "MOVE_LEFT", accepted=False),
                row("reject", 250, 10, "MOVE_LEFT")]
        result = analysis.analyze(report(rows))
        metrics = result["methods"]["two_consecutive_same_direction"]
        self.assertEqual(metrics["source_trial_count"], 3)
        self.assertEqual(metrics["accepted_move_count"], 1)
        self.assertEqual(metrics["false_move_count"], 0)
        self.assertFalse(result["selection_use"])
        self.assertFalse(result["runtime_enabled"])

    def test_missing_or_duplicate_window_is_rejected(self):
        rows = [row("trial", 0, 10, "MOVE_LEFT"), row("trial", 0, 10, "MOVE_LEFT")]
        with self.assertRaises(ValueError):
            analysis.analyze(report(rows))

    def test_final_test_split_is_rejected(self):
        invalid = copy.deepcopy(report([row("trial", 0, 10, "MOVE_LEFT"),
                                        row("trial", 250, 10, "MOVE_LEFT")]))
        invalid["split"] = "final_test"
        with self.assertRaises(ValueError):
            analysis.analyze(invalid)


class SourceBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="temporal-contract-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / "custom-validation.json"
        self.output = self.root / "new-diagnostic.json"
        self.data = report([row("trial", 0, 10, "MOVE_LEFT"), row("trial", 250, 10, "MOVE_LEFT")])
        self.raw = (json.dumps(self.data, indent=3) + "\n\n").encode()
        self.input.write_bytes(self.raw)

    def test_in_memory_analysis_has_no_invented_file_identity(self):
        with patch.object(Path, "read_bytes", side_effect=AssertionError("pure computation")):
            value = analysis.analyze(self.data)
        self.assertIsNone(value["source_report"])
        self.assertIsNone(value["source_report_sha256"])
        self.assertEqual(value["source_binding_status"], "unbound_in_memory")

    def test_custom_file_binds_exact_consumed_bytes_without_default(self):
        with patch.object(analysis, "DEFAULT_INPUT", self.root / "absent.json"):
            result = analysis.run(self.input, self.output)
        self.assertEqual(result["source_report_sha256"], hashlib.sha256(self.raw).hexdigest())
        self.assertEqual(result["source_report"], str(self.input.resolve()))
        self.assertEqual(result["source_binding_status"], "bound_consumed_file_bytes")
        self.assertEqual(json.loads(self.output.read_text("utf-8")), result)
        self.assertEqual(self.input.read_bytes(), self.raw)
        self.assertFalse(result["commands_executed"])

    def test_changed_source_refuses_publication(self):
        original = analysis.analyze
        def changed(report):
            value = original(report)
            self.input.write_bytes(self.raw + b" ")
            return value
        with patch.object(analysis, "analyze", side_effect=changed), self.assertRaisesRegex(ValueError, "changed"):
            analysis.run(self.input, self.output)
        self.assertFalse(self.output.exists())

    def test_existing_output_is_preserved_without_analysis(self):
        self.output.write_bytes(b"keep existing result")
        with patch.object(analysis, "analyze", side_effect=AssertionError), self.assertRaisesRegex(ValueError, "overwrite"):
            analysis.run(self.input, self.output)
        self.assertEqual(self.output.read_bytes(), b"keep existing result")

    def test_unsafe_alias_rejected_before_resolving_away_the_link(self):
        with patch.object(Path, "is_symlink", lambda p: p == self.input), self.assertRaisesRegex(ValueError, "unsafe"):
            analysis.run(self.input, self.output)
        self.assertFalse(self.output.exists())

    def test_invalid_report_does_not_create_output(self):
        self.data["split"] = "final_test"
        self.input.write_text(json.dumps(self.data))
        with self.assertRaises(ValueError):
            analysis.run(self.input, self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
