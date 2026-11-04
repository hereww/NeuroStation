"""Contract tests for post-hoc temporal-confirmation diagnostics."""
from __future__ import annotations

import copy
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


if __name__ == "__main__":
    unittest.main()
