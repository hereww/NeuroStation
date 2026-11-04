from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import evaluate_personal_ssvep_decoder_v4_sessions as evaluator


class PersonalDecoderV4SessionSplitTests(unittest.TestCase):
    TARGETS = [("left", 10.0)]

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.calibration = self.root / "calibration"
        self.evaluation = self.root / "evaluation"
        self.calibration.mkdir()
        self.evaluation.mkdir()

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def session_result(source: Path, *, session_id: str, raw_hash: str):
        info = {
            "source_session": str(source.resolve()),
            "session_id": session_id,
            "source_sha256": {"raw_brainflow.tsv": raw_hash},
            "header": ["eeg_ch1"],
        }
        return [], PersonalDecoderV4SessionSplitTests.TARGETS, info

    def run_with_results(self, calibration_result, evaluation_result):
        output = self.root / "report.json"
        with (
            patch.object(evaluator, "_pairs_for_session", side_effect=[calibration_result, evaluation_result]),
            patch("eeg_tools.personal_ssvep_decoder_v4.fit_calibration", return_value={"eligible": False}) as fit,
        ):
            report = evaluator.evaluate([self.calibration], [self.evaluation], output)
        return report, fit

    def test_same_resolved_path_is_rejected_before_reading_sessions(self):
        with patch.object(evaluator, "_pairs_for_session") as load:
            with self.assertRaisesRegex(ValueError, "paths must be disjoint"):
                evaluator.evaluate([self.calibration], [self.calibration], self.root / "report.json")
        load.assert_not_called()

    def test_duplicate_session_id_is_rejected_before_calibration_fit(self):
        calibration = self.session_result(self.calibration, session_id="same-session", raw_hash="a" * 64)
        evaluation = self.session_result(self.evaluation, session_id="same-session", raw_hash="b" * 64)
        with patch.object(evaluator, "_pairs_for_session", side_effect=[calibration, evaluation]):
            with patch("eeg_tools.personal_ssvep_decoder_v4.fit_calibration") as fit:
                with self.assertRaisesRegex(ValueError, "overlaps calibration session by session_id"):
                    evaluator.evaluate([self.calibration], [self.evaluation], self.root / "report.json")
        fit.assert_not_called()

    def test_identical_raw_recording_is_rejected_even_with_different_session_ids(self):
        raw_hash = "c" * 64
        calibration = self.session_result(self.calibration, session_id="cal-session", raw_hash=raw_hash)
        evaluation = self.session_result(self.evaluation, session_id="eval-session", raw_hash=raw_hash)
        with patch.object(evaluator, "_pairs_for_session", side_effect=[calibration, evaluation]):
            with patch("eeg_tools.personal_ssvep_decoder_v4.fit_calibration") as fit:
                with self.assertRaisesRegex(ValueError, "overlaps calibration session by raw_sha256"):
                    evaluator.evaluate([self.calibration], [self.evaluation], self.root / "report.json")
        fit.assert_not_called()

    def test_disjoint_sessions_report_identity_check_without_enabling_deployment(self):
        calibration = self.session_result(self.calibration, session_id="cal-session", raw_hash="d" * 64)
        evaluation = self.session_result(self.evaluation, session_id="eval-session", raw_hash="e" * 64)
        with patch.object(evaluator, "_pairs_for_session", side_effect=[calibration, evaluation]):
            with patch("eeg_tools.personal_ssvep_decoder_v4.fit_calibration", return_value={"eligible": False}):
                with patch("eeg_tools.personal_ssvep_decoder_v4.config_summary", return_value={}):
                    report = evaluator.evaluate([self.calibration], [self.evaluation], self.root / "report.json")

        self.assertTrue(report["source_session_identity_disjoint"])
        self.assertFalse(report["independent_personal_validation"])
        self.assertFalse(report["runtime_enabled"])
        self.assertFalse(report["default_deployment_authorized"])


if __name__ == "__main__":
    unittest.main()
