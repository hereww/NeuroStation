from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from eeg_tools.workstation import DatasetRepository, SSVEPProtocol, SSVEPProtocolError
from eeg_tools.workstation.gateway import WorkstationGateway
from eeg_tools.workstation.task import SSVEPTask


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
        self.assertEqual(27, protocol.expected_event_count)

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


class RealHardwareBoundaryTests(unittest.TestCase):
    def test_legacy_task_constructor_is_disabled(self) -> None:
        protocol = SSVEPProtocol.load(PROTOCOL_PATH)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "validation.real_hardware_only"):
                SSVEPTask(protocol, DatasetRepository(Path(directory)), participant_id="P001", session_name="test")

    def test_legacy_gateway_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            gateway = WorkstationGateway(
                protocol_path=PROTOCOL_PATH,
                dataset_root=Path(directory),
            )
            with self.assertRaisesRegex(ValueError, "validation.real_hardware_only"):
                gateway.start_ssvep(participant_id="P001", session_name="test")


if __name__ == "__main__":
    unittest.main()
