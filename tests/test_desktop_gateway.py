from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from apps.workstation_ui.gateway import CaptureConfig, CaptureMode
from eeg_tools.workstation.desktop_gateway import DesktopGateway, MetadataGateway


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "protocols" / "ssvep_four_target_v2.json"
CHANNELS = ROOT / "configs" / "channel_config_v1_auto.json"


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


if __name__ == "__main__":
    unittest.main()
