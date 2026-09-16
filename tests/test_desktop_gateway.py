from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from apps.workstation_ui.gateway import CaptureConfig, CaptureMode
from eeg_tools.workstation.desktop_gateway import DesktopGateway, MetadataGateway
from eeg_tools.workstation.process_gateway import AcquisitionProcessGateway
from eeg_tools.config import validate_channel_config


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "protocols" / "ssvep_four_target_v2.json"
CHANNELS = ROOT / "configs" / "channel_config_v1_auto.json"


def calibration_payload() -> dict:
    return {
        "serial_port": "COM5",
        "channels": [
            {
                "name": f"CH{index}",
                "gui_index": index - 1,
                "board_input": f"N{index}P",
                "electrode_position": position,
            }
            for index, position in enumerate(
                ("Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2"),
                start=1,
            )
        ],
        "reference": {
            "label": "SRB",
            "position": "left_earlobe",
            "hardware_connection": "SRB ear clip",
        },
        "bias": {
            "label": "BIAS",
            "position": "right_earlobe",
            "hardware_connection": "BIAS ear clip",
        },
        "ground": {
            "label": "AGND",
            "position": "mastoid",
            "hardware_connection": "AGND on acquisition board",
        },
        "tests": [
            {
                "status": "passed",
                "channel": index,
                "metrics": {
                    "finite_fraction": 1.0,
                    "flat_fraction": 0.0,
                    "saturation_fraction": 0.0,
                },
            }
            for index in range(1, 9)
        ],
        "protocol_reviewed": True,
    }


class DesktopGatewayTests(unittest.TestCase):
    def test_metadata_gateway_cannot_start_capture(self) -> None:
        gateway = MetadataGateway(protocol_path=PROTOCOL)
        with self.assertRaisesRegex(ValueError, "validation.real_hardware_only"):
            gateway.start_ssvep(CaptureConfig())

    def test_desktop_gateway_rejects_legacy_capture_modes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            gateway = DesktopGateway(
                protocol_path=PROTOCOL,
                channel_config_path=CHANNELS,
                dataset_root=Path(directory),
            )
            config = CaptureConfig(
                participant="P001",
                name="real hardware boundary",
                mode=CaptureMode.SYNTHETIC,
                acknowledge_flicker_risk=True,
            )
            with self.assertRaisesRegex(ValueError, "validation.real_hardware_only"):
                gateway.start_ssvep(config, 1)

    def test_desktop_gateway_requires_one_times_real_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            gateway = DesktopGateway(
                protocol_path=PROTOCOL,
                channel_config_path=CHANNELS,
                dataset_root=Path(directory),
            )
            config = CaptureConfig(
                participant="P001",
                name="real hardware boundary",
                mode=CaptureMode.CYTON,
                acknowledge_flicker_risk=True,
            )
            with self.assertRaisesRegex(ValueError, "validation.user_not_found"):
                gateway.start_ssvep(config, 1)

    def test_channel_calibration_requires_eight_passed_channel_tests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory) / "home"
            gateway = DesktopGateway(
                protocol_path=PROTOCOL,
                channel_config_path=CHANNELS,
                dataset_root=Path(directory) / "Datasets",
            )
            invalid = calibration_payload()
            invalid["tests"] = copy.deepcopy(invalid["tests"])
            invalid["tests"][3]["status"] = "warning"
            with patch("neurostation_contract.Path.home", return_value=home):
                with self.assertRaisesRegex(ValueError, "validation.calibration_test_required"):
                    gateway.save_channel_calibration(invalid)

                invalid_position = calibration_payload()
                invalid_position["channels"][0]["electrode_position"] = "unspecified"
                with self.assertRaisesRegex(ValueError, "validation.calibration_missing"):
                    gateway.save_channel_calibration(invalid_position)

    def test_channel_calibration_saves_valid_pair_to_user_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory) / "home"
            gateway = DesktopGateway(
                protocol_path=PROTOCOL,
                channel_config_path=CHANNELS,
                dataset_root=Path(directory) / "Datasets",
            )
            with patch("neurostation_contract.Path.home", return_value=home):
                result = gateway.save_channel_calibration(calibration_payload())
                channel_path = Path(result["channel_config_path"])
                protocol_path = Path(result["protocol_path"])
                self.assertTrue(channel_path.is_file())
                self.assertTrue(protocol_path.is_file())
                channel = json.loads(channel_path.read_text(encoding="utf-8"))
                protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
                self.assertEqual("CHANNEL-V1-FORMAL", channel["config_version"])
                self.assertEqual([], validate_channel_config(channel, channel_path))
                self.assertEqual("formal_candidate", protocol["status"])
                self.assertEqual(protocol_path, gateway.protocol_path)
                self.assertEqual(channel_path, gateway.channel_config_path)
                self.assertEqual(channel, gateway.load_channel_calibration())

    def test_channel_calibration_test_extracts_selected_channel_metrics(self) -> None:
        gateway = AcquisitionProcessGateway(
            protocol_path=PROTOCOL,
            channel_config_path=CHANNELS,
        )
        gateway.preflight_cyton = lambda seconds=3.0, port="AUTO": {
            "status": "passed",
            "selected_port": "COM5",
            "checks": [
                {
                    "name": "channels",
                    "status": "passed",
                    "metrics": {
                        "channels": [
                            {
                                "channel": 1,
                                "finite_fraction": 1.0,
                                "flat_fraction": 0.0,
                                "saturation_fraction": 0.0,
                            },
                            {
                                "channel": 2,
                                "finite_fraction": 1.0,
                                "flat_fraction": 0.02,
                                "saturation_fraction": 0.01,
                            },
                        ]
                    },
                }
            ],
        }
        result = gateway.test_cyton_channel(2, port="COM5")
        self.assertEqual("passed", result["status"])
        self.assertEqual(2, result["channel"])
        self.assertEqual("COM5", result["selected_port"])
        self.assertEqual(0.02, result["metrics"]["flat_fraction"])


if __name__ == "__main__":
    unittest.main()
