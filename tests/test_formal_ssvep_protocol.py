"""Formal SSVEP calibration protocol must collect enough repeated trials."""
from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from neurostation_contract import CaptureConfig
from eeg_tools.workstation.ssvep import SSVEPProtocol

ROOT = Path(__file__).resolve().parents[1]


class FormalSSVEPProtocolTests(unittest.TestCase):
    def test_formal_protocol_has_calibration_repetitions_and_rest(self):
        path = ROOT / "configs" / "protocols" / "ssvep_four_target_v2.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(value["status"], "formal")
        self.assertEqual(value["trial"]["repetitions"], 10)
        self.assertEqual(value["trial"]["stimulus_s"], 5)
        self.assertEqual(value["trial"]["post_trial_rest_s"], 5)
        # The worker runs countdown, stimulus and inter-trial rest only.
        # Do not count baseline/cue phases that it does not execute.
        self.assertEqual(value["trial"]["pre_session_rest_s"], 0)
        self.assertEqual(value["trial"]["pre_trial_rest_s"], 0)
        self.assertEqual([target["frequency_hz"] for target in value["targets"]], [10, 12, 15, 20])


    def test_ui_protocol_and_legacy_resource_share_default_timing(self):
        config = CaptureConfig()
        protocol = SSVEPProtocol.load(ROOT / "configs/protocols/ssvep_four_target_v2.json")
        legacy = json.loads((ROOT / "configs/ssvep_config_v1.json").read_text(encoding="utf-8"))
        trial = legacy["trial_protocol"]
        self.assertEqual(config.stimulus_seconds, protocol.stimulus_s)
        self.assertEqual(config.rest_seconds, protocol.post_trial_rest_s)
        self.assertEqual(config.repetitions, protocol.repetitions)
        self.assertEqual(config.trials, protocol.trial_count)
        self.assertEqual(config.recording_seconds, protocol.recording_duration_s)
        self.assertEqual(config.total_seconds, protocol.total_duration_s)
        self.assertEqual(config.expected_events, protocol.expected_event_count)
        self.assertEqual(trial["stimulus_s"], protocol.stimulus_s)
        self.assertEqual(trial["post_trial_rest_s"], protocol.post_trial_rest_s)
        self.assertEqual(trial["repetitions_per_frequency"], protocol.repetitions)
        self.assertEqual(trial["pre_session_rest_s"], protocol.pre_session_rest_s)
        self.assertEqual(trial["pre_trial_rest_s"], protocol.pre_trial_rest_s)
        self.assertEqual(tuple(legacy["frequencies_hz"]), protocol.frequencies_hz)

    def test_cli_validate_only_reports_default_timing_without_starting_worker(self):
        from run_ssvep_session import main

        output = io.StringIO()
        with patch("run_ssvep_session.worker_main") as worker, redirect_stdout(output):
            result = main(["--validate-only"])
        worker.assert_not_called()
        self.assertEqual(result, 0)
        report = json.loads(output.getvalue())
        self.assertTrue(report["valid"])
        self.assertEqual(report["trial_count"], 40)
        self.assertEqual(report["recording_duration_s"], 395)

    def test_explicit_parameters_still_override_new_defaults(self):
        config = CaptureConfig(
            participant="U0001", eye_side="left", acknowledge_flicker_risk=True,
            stimulus_seconds=5, rest_seconds=3, repetitions=3,
        )
        config.validate()
        protocol = SSVEPProtocol.load(ROOT / "configs/protocols/ssvep_four_target_v2.json")
        runtime = protocol.with_runtime_parameters(
            stimulus_s=config.stimulus_seconds, rest_s=config.rest_seconds,
            repetitions=config.repetitions,
        )
        self.assertEqual(runtime.trial_count, 12)
        self.assertEqual(runtime.recording_duration_s, 93)
        self.assertEqual(runtime.total_duration_s, 98)
        self.assertEqual(protocol.repetitions, 10)
        self.assertEqual(protocol.post_trial_rest_s, 5)


if __name__ == "__main__":
    unittest.main()
