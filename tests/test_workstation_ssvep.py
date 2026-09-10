from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eeg_tools.workstation import (
    DatasetRepository,
    SSVEPProtocol,
    SSVEPProtocolError,
    SSVEPTask,
    TaskPhase,
    WorkstationGateway,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "protocols" / "ssvep_four_target_v2.json"


class SSVEPProtocolTests(unittest.TestCase):
    def test_default_protocol_calculates_the_reviewed_ui_values(self) -> None:
        protocol = SSVEPProtocol.load(PROTOCOL_PATH)
        self.assertEqual((10, 12, 15, 20), protocol.frequencies_hz)
        self.assertEqual(12, protocol.trial_count)
        self.assertEqual(93.0, protocol.recording_duration_s)
        self.assertEqual(98.0, protocol.total_duration_s)
        self.assertEqual(23_250, protocol.expected_samples_per_channel)
        self.assertEqual(26, protocol.expected_event_count)
        self.assertEqual(12, len(protocol.build_trials()))
        self.assertEqual(0, protocol.pre_session_rest_s)
        self.assertEqual(0, protocol.pre_trial_rest_s)
        self.assertEqual(3, protocol.post_trial_rest_s)

    def test_runtime_rest_override_omits_rest_after_the_last_trial(self) -> None:
        protocol = SSVEPProtocol.load(PROTOCOL_PATH).with_runtime_parameters(
            repetitions=1, stimulus_s=5, rest_s=2
        )
        self.assertEqual(4, protocol.trial_count)
        self.assertEqual(26, protocol.recording_duration_s)

    def test_frequency_must_divide_display_refresh_rate(self) -> None:
        value = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        value["targets"][0]["frequency_hz"] = 11
        with self.assertRaisesRegex(SSVEPProtocolError, "does not divide"):
            SSVEPProtocol.from_dict(value)


class SSVEPTaskTests(unittest.TestCase):
    def test_countdown_run_completion_and_dataset_navigation(self) -> None:
        protocol = SSVEPProtocol.load(PROTOCOL_PATH)
        with tempfile.TemporaryDirectory() as directory:
            repository = DatasetRepository(Path(directory))
            task = SSVEPTask(
                protocol,
                repository,
                participant_id="sub-001",
                session_name="SSVEP acceptance",
                simulation_speed=8.0,
            )
            self.assertEqual(TaskPhase.COUNTDOWN, task.start(100.0).phase)
            self.assertGreater(task.snapshot(104.999).countdown_remaining_s, 0)
            running = task.snapshot(105.0)
            self.assertEqual(TaskPhase.RUNNING, running.phase)
            self.assertEqual(0.0, running.recording_elapsed_s)

            completed = task.snapshot(105.0 + protocol.recording_duration_s / 8.0)
            self.assertEqual(TaskPhase.COMPLETED, completed.phase)
            self.assertIsNotNone(completed.result)
            self.assertTrue(completed.result.output_dir.is_dir())
            self.assertEqual(1, len(repository.list_records()))
            session = json.loads(
                (completed.result.output_dir / "session.json").read_text(encoding="utf-8")
            )
            self.assertTrue(session["simulated"])
            self.assertEqual(0, session["recorded_samples_per_channel"])
            self.assertIn("no hardware EEG", session["notice"])

    def test_active_task_can_be_aborted_and_saved(self) -> None:
        protocol = SSVEPProtocol.load(PROTOCOL_PATH)
        with tempfile.TemporaryDirectory() as directory:
            task = SSVEPTask(
                protocol,
                DatasetRepository(Path(directory)),
                participant_id="sub-002",
                session_name="Abort acceptance",
            )
            task.start(0.0)
            aborted = task.abort(2.0)
            self.assertEqual(TaskPhase.ABORTED, aborted.phase)
            self.assertEqual("aborted", aborted.result.status)


class GatewayTests(unittest.TestCase):
    def test_ui_gateway_exposes_apps_parameters_and_completed_dataset(self) -> None:
        now = [10.0]
        with tempfile.TemporaryDirectory() as directory:
            gateway = WorkstationGateway(
                protocol_path=PROTOCOL_PATH,
                dataset_root=Path(directory),
                clock=lambda: now[0],
            )
            apps = gateway.list_acquisition_apps()
            self.assertTrue(apps[0].available)
            self.assertEqual("ssvep", apps[0].app_id)
            self.assertFalse(apps[1].available)
            details = gateway.ssvep_details(repetitions=2, stimulus_s=4)
            self.assertEqual(8, details["trial_count"])
            self.assertEqual((10, 12, 15, 20), details["frequencies_hz"])

            gateway.start_ssvep(
                participant_id="sub-003",
                session_name="Gateway acceptance",
                repetitions=1,
                simulation_speed=8,
            )
            now[0] += 6
            self.assertEqual(TaskPhase.RUNNING, gateway.poll_task().phase)
            now[0] += 10
            self.assertEqual(TaskPhase.COMPLETED, gateway.poll_task().phase)
            self.assertEqual(1, len(gateway.list_datasets()))


if __name__ == "__main__":
    unittest.main()
