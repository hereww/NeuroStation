from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from eeg_tools.config import validate_channel_config
from eeg_tools.workstation.acquisition_worker import main as worker_main
from eeg_tools.workstation.desktop_gateway import DesktopGateway
from eeg_tools.workstation.ssvep import SSVEPProtocol
from neurostation_contract import CaptureConfig, Phase, TaskSnapshot, UserProfile
from neurostation_diagnostics import DiagnosticStore


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "protocols" / "ssvep_four_target_v2.json"
CHANNELS = ROOT / "configs" / "channel_config_v1_auto.json"
TEMPLATE = ROOT / "configs" / "channel_config_v1_template.json"


class SessionBoundaryReached(Exception):
    """Stop the worker after config checks, before any hardware or stimulus."""


class FormalAcquisitionDefaultsTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.home_patch = patch(
            "neurostation_contract.Path.home", return_value=self.root / "home"
        )
        self.home_patch.start()
        self.addCleanup(self.home_patch.stop)

    def draft_protocol(self) -> Path:
        value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        value["status"] = "draft_for_technical_validation"
        path = self.root / "draft_protocol.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def gateway_and_config(self, protocol_path: Path = PROTOCOL):
        gateway = DesktopGateway(
            protocol_path=protocol_path,
            channel_config_path=CHANNELS,
            dataset_root=self.root / "Datasets",
        )
        profile = gateway.add_user(UserProfile(user_id="", name="Regression participant", age=28))
        config = CaptureConfig(
            participant=profile.user_id,
            user_id=profile.user_id,
            save_directory=self.root / "Datasets",
            eye_side="left",
            acknowledge_flicker_risk=True,
            allow_draft_hardware_config=False,
        )
        return gateway, config

    def worker_arguments(self, protocol: Path = PROTOCOL, channels: Path = CHANNELS):
        return [
            "--protocol", str(protocol),
            "--channel-config", str(channels),
            "--output-root", str(self.root / "worker-output"),
            "--participant", "P001",
            "--session-name", "Regression test",
            "--eye-side", "left",
            "--acknowledge-flicker-risk",
        ]

    def assert_worker_accepts_config(self, arguments: list[str]) -> None:
        with patch(
            "eeg_tools.workstation.acquisition_worker._session_directory",
            side_effect=SessionBoundaryReached,
        ) as session, patch(
            "eeg_tools.workstation.acquisition_worker._prepare_cyton_board"
        ) as board:
            with self.assertRaises(SessionBoundaryReached):
                worker_main(arguments)
            session.assert_called_once()
            board.assert_not_called()
        self.assertFalse((self.root / "worker-output").exists())

    def assert_worker_rejects_config(self, arguments: list[str], message: str) -> None:
        error = io.StringIO()
        with patch(
            "eeg_tools.workstation.acquisition_worker._session_directory"
        ) as session, contextlib.redirect_stderr(error):
            with self.assertRaises(SystemExit) as raised:
                worker_main(arguments)
            self.assertEqual(2, raised.exception.code)
            session.assert_not_called()
        self.assertIn(message, error.getvalue())

    def test_bundled_protocol_and_channel_map_are_formal(self) -> None:
        value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        self.assertEqual("formal", value["status"])
        protocol = SSVEPProtocol.load(PROTOCOL)
        self.assertEqual(8, protocol.channel_count)
        self.assertEqual(250, protocol.sampling_rate_hz)
        self.assertEqual(60, protocol.refresh_rate_hz)
        self.assertEqual(
            [], validate_channel_config(json.loads(CHANNELS.read_text(encoding="utf-8")), CHANNELS)
        )

    def test_desktop_starts_defaults_without_draft_override(self) -> None:
        gateway, config = self.gateway_and_config()
        snapshot = TaskSnapshot(phase=Phase.COUNTDOWN)
        with patch.object(gateway._acquisition, "start_ssvep", return_value=snapshot) as start:
            self.assertIs(snapshot, gateway.start_ssvep(config))
        forwarded, speed = start.call_args.args
        self.assertFalse(forwarded.allow_draft_hardware_config)
        self.assertEqual(config.user_id, forwarded.participant)
        self.assertEqual(1, speed)
        gateway._acquisition._status_file = self.root / "status.json"
        gateway._acquisition._cancel_file = self.root / "cancel.request"
        command = gateway._acquisition._command(forwarded)
        self.assertNotIn("--allow-draft-protocol", command)
        self.assertNotIn("--allow-draft-channel-config", command)

    def test_desktop_still_blocks_custom_draft_protocol(self) -> None:
        gateway, config = self.gateway_and_config(self.draft_protocol())
        with patch.object(gateway._acquisition, "start_ssvep") as start:
            with self.assertRaisesRegex(ValueError, "validation.hardware_draft"):
                gateway.start_ssvep(config)
            start.assert_not_called()

    def test_desktop_still_blocks_incomplete_custom_channel_map(self) -> None:
        gateway, config = self.gateway_and_config()
        with patch.object(gateway._acquisition, "start_ssvep") as start:
            with self.assertRaisesRegex(ValueError, "validation.hardware_draft"):
                gateway.start_ssvep(replace(config, channel_config=TEMPLATE))
            start.assert_not_called()

    def test_desktop_explicit_technical_override_is_preserved(self) -> None:
        gateway, config = self.gateway_and_config(self.draft_protocol())
        with patch.object(gateway._acquisition, "start_ssvep") as start:
            gateway.start_ssvep(replace(
                config, channel_config=TEMPLATE, allow_draft_hardware_config=True
            ))
        self.assertTrue(start.call_args.args[0].allow_draft_hardware_config)

    def test_desktop_still_requires_flicker_risk_acknowledgment(self) -> None:
        gateway, config = self.gateway_and_config()
        for override in (False, True):
            with self.subTest(override=override), patch.object(
                gateway._acquisition, "start_ssvep"
            ) as start:
                with self.assertRaisesRegex(ValueError, "validation.flicker_ack"):
                    gateway.start_ssvep(replace(
                        config, acknowledge_flicker_risk=False,
                        allow_draft_hardware_config=override,
                    ))
                start.assert_not_called()

    def test_worker_accepts_defaults_without_draft_flags(self) -> None:
        self.assert_worker_accepts_config(self.worker_arguments())

    def test_worker_still_blocks_custom_draft_protocol(self) -> None:
        self.assert_worker_rejects_config(
            self.worker_arguments(protocol=self.draft_protocol()), "refuses a draft protocol"
        )

    def test_worker_still_blocks_incomplete_custom_channel_map(self) -> None:
        self.assert_worker_rejects_config(
            self.worker_arguments(channels=TEMPLATE), "refuses the draft/incomplete channel map"
        )

    def test_worker_explicit_technical_override_still_works(self) -> None:
        self.assert_worker_accepts_config(
            self.worker_arguments(protocol=self.draft_protocol(), channels=TEMPLATE)
            + ["--allow-draft-protocol", "--allow-draft-channel-config"]
        )

    def test_default_diagnostics_no_longer_reports_a_draft(self) -> None:
        report = DiagnosticStore(root=self.root / "Diagnostics", persist=False).build_report()
        self.assertEqual("ready", report["configuration"]["status"])
        self.assertEqual("formal", report["configuration"]["protocol_status"])
        self.assertEqual([], report["configuration"]["warnings"])


if __name__ == "__main__":
    unittest.main()
