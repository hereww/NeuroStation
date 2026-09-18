from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from apps.workstation_ui.gateway import CaptureConfig, CaptureMode
from eeg_tools.workstation.acquisition_worker import (
    _prepare_cyton_board,
    build_parser,
)
from eeg_tools.workstation.process_gateway import AcquisitionProcessGateway


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "protocols" / "ssvep_four_target_v2.json"
CHANNELS = ROOT / "configs" / "channel_config_v1_auto.json"


class AcquisitionWorkerTests(unittest.TestCase):
    def test_worker_parser_only_exposes_cyton(self) -> None:
        action = build_parser()._option_string_actions["--board"]
        self.assertEqual(("cyton",), tuple(action.choices))

    def test_cyton_prepare_scans_candidates_until_one_is_ready(self) -> None:
        class FakeBoard:
            def __init__(self, _board_id, params):
                self.port = params.serial_port

            def prepare_session(self):
                if self.port == "COM1":
                    raise RuntimeError("BOARD_NOT_READY_ERROR:7")

            def release_session(self):
                return None

        class Params:
            serial_port = "AUTO"

        with patch(
            "eeg_tools.workstation.acquisition_worker.candidate_serial_ports",
            return_value=(type("Port", (), {"device": "COM1"})(), type("Port", (), {"device": "COM2"})()),
        ), patch("brainflow.board_shim.BoardShim", FakeBoard):
            board, selected, failures = _prepare_cyton_board(6, Params(), "AUTO")
        self.assertEqual("COM2", selected)
        self.assertEqual("COM2", board.port)
        self.assertEqual("COM1", failures[0]["port"])

    def test_cyton_prepare_explains_board_not_ready_as_radio_handshake(self) -> None:
        class FakeBoard:
            def __init__(self, _board_id, _params):
                pass

            def prepare_session(self):
                raise RuntimeError("BrainFlowError: BOARD_NOT_READY_ERROR:7 unable to prepare streaming session")

            def release_session(self):
                return None

        class Params:
            serial_port = "AUTO"

        with patch(
            "eeg_tools.workstation.acquisition_worker.candidate_serial_ports",
            return_value=(type("Port", (), {"device": "COM5"})(),),
        ), patch("brainflow.board_shim.BoardShim", FakeBoard):
            with self.assertRaisesRegex(RuntimeError, "没有收到 Cyton 欢迎字符"):
                _prepare_cyton_board(6, Params(), "AUTO")

    def test_process_gateway_rejects_legacy_capture_modes(self) -> None:
        gateway = AcquisitionProcessGateway(
            protocol_path=PROTOCOL,
            channel_config_path=CHANNELS,
        )
        with self.assertRaisesRegex(ValueError, "validation.real_hardware_only"):
            gateway._command(CaptureConfig(mode=CaptureMode.SYNTHETIC))
        with self.assertRaisesRegex(ValueError, "validation.real_hardware_only"):
            gateway._command(CaptureConfig(mode=CaptureMode.VISUAL_PREVIEW))

    def test_process_gateway_command_always_selects_cyton(self) -> None:
        gateway = AcquisitionProcessGateway(
            protocol_path=PROTOCOL,
            channel_config_path=CHANNELS,
        )
        gateway._runtime_dir = ROOT / ".tmp-test-worker"
        gateway._status_file = gateway._runtime_dir / "status.json"
        gateway._cancel_file = gateway._runtime_dir / "cancel.request"
        command = gateway._command(CaptureConfig(mode=CaptureMode.CYTON, eye_side="left"))
        board_index = command.index("--board")
        self.assertEqual("cyton", command[board_index + 1])
        eye_index = command.index("--eye-side")
        self.assertEqual("left", command[eye_index + 1])


if __name__ == "__main__":
    unittest.main()
