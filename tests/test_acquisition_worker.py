from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time
import unittest

from apps.workstation_ui.gateway import CaptureConfig, CaptureMode, Phase
from eeg_tools.workstation.acquisition_worker import main as worker_main
from eeg_tools.workstation.acquisition_worker import _prepare_cyton_board
from eeg_tools.workstation.process_gateway import AcquisitionProcessGateway


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "protocols" / "ssvep_four_target_v2.json"
CHANNELS = ROOT / "configs" / "channel_config_v1_template.json"


class AcquisitionWorkerTests(unittest.TestCase):
    def test_cyton_prepare_scans_candidates_until_one_is_ready(self) -> None:
        from unittest.mock import patch

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

        with patch("eeg_tools.workstation.acquisition_worker.candidate_serial_ports", return_value=(
            type("Port", (), {"device": "COM1"})(),
            type("Port", (), {"device": "COM2"})(),
        )), patch("brainflow.board_shim.BoardShim", FakeBoard):
            board, selected, failures = _prepare_cyton_board(6, Params(), "AUTO")
        self.assertEqual("COM2", selected)
        self.assertEqual("COM2", board.port)
        self.assertEqual("COM1", failures[0]["port"])

    def test_demo_headless_persists_visual_preview_metadata_without_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "视觉预览"
            exit_code = worker_main(
                [
                    "--protocol", str(PROTOCOL),
                    "--channel-config", str(ROOT / "configs" / "channel_config_v1_auto.json"),
                    "--output-root", str(output),
                    "--participant", "P-PREVIEW",
                    "--session-name", "全屏预览验收",
                    "--board", "demo",
                    "--headless",
                    "--repetitions", "1",
                    "--stimulus-seconds", "0.01",
                    "--rest-seconds", "0",
                    "--countdown-seconds", "0.01",
                ]
            )
            self.assertEqual(0, exit_code)
            session_dir = next(output.glob("session_*"))
            session = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
            self.assertEqual("demo", session["board"])
            self.assertTrue(session["simulated"])
            self.assertEqual(0, session["channel_count"])
            self.assertEqual(0, session["recorded_samples_per_channel"])
            self.assertEqual(10, session["event_count"])
            self.assertTrue((session_dir / "quality.json").is_file())
            self.assertTrue((session_dir / "manifest.csv").is_file())

    def test_synthetic_headless_persists_raw_data_in_chinese_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "中文数据集"
            exit_code = worker_main(
                [
                    "--protocol", str(PROTOCOL),
                    "--output-root", str(output),
                    "--participant", "验收P001",
                    "--session-name", "合成板验收",
                    "--board", "synthetic",
                    "--headless",
                    "--repetitions", "1",
                    "--stimulus-seconds", "0.05",
                    "--rest-seconds", "0.01",
                    "--countdown-seconds", "0.01",
                ]
            )
            self.assertEqual(0, exit_code)
            sessions = list(output.glob("session_*"))
            self.assertEqual(1, len(sessions))
            session_dir = sessions[0]
            session = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
            self.assertEqual("completed", session["status"])
            self.assertTrue(session["simulated"])
            self.assertEqual(4, session["completed_trials"])
            self.assertEqual(10, session["event_count"])
            self.assertGreater(session["recorded_samples_per_channel"], 0)
            self.assertGreater((session_dir / "raw_brainflow.tsv").stat().st_size, 0)
            self.assertTrue((session_dir / "manifest.csv").is_file())
            quality = json.loads((session_dir / "quality.json").read_text(encoding="utf-8"))
            self.assertEqual("ok", quality["status"])
            self.assertEqual(session["channel_count"], quality["channel_count"])
            self.assertEqual(session["recorded_samples_per_channel"], quality["samples_per_channel"])
            self.assertTrue((session_dir / "ssvep_config.json").is_file())

    def test_process_gateway_cancel_preserves_aborted_session(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "中止数据"
            gateway = AcquisitionProcessGateway(
                protocol_path=PROTOCOL,
                channel_config_path=CHANNELS,
            )
            config = CaptureConfig(
                participant="P-CANCEL",
                name="cancel-test",
                repetitions=1,
                stimulus_seconds=1,
                rest_seconds=0,
                save_directory=root,
                mode=CaptureMode.SYNTHETIC,
                acknowledge_flicker_risk=True,
            )
            self.assertEqual(Phase.COUNTDOWN, gateway.start_ssvep(config, 1).phase)
            deadline = time.monotonic() + 5
            while gateway.snapshot.phase == Phase.COUNTDOWN and time.monotonic() < deadline:
                gateway.tick()
                if gateway._status_file and gateway._status_file.exists():
                    break
                time.sleep(0.02)
            cancelled = gateway.cancel()
            self.assertEqual(Phase.CANCELLED, cancelled.phase)
            self.assertIsNotNone(cancelled.result)
            self.assertEqual("aborted", cancelled.result.status)
            self.assertTrue(cancelled.result.path.is_dir())
            self.assertTrue((cancelled.result.path / "session.json").is_file())

    def test_preview_mode_maps_to_demo_worker(self) -> None:
        gateway = AcquisitionProcessGateway(
            protocol_path=PROTOCOL,
            channel_config_path=ROOT / "configs" / "channel_config_v1_auto.json",
        )
        gateway._runtime_dir = Path(tempfile.gettempdir()) / "neurostation-command-test"
        gateway._status_file = gateway._runtime_dir / "status.json"
        gateway._cancel_file = gateway._runtime_dir / "cancel.request"
        command = gateway._command(
            CaptureConfig(
                mode=CaptureMode.VISUAL_PREVIEW,
                acknowledge_flicker_risk=True,
            )
        )
        board_index = command.index("--board")
        self.assertEqual("--board", command[board_index])
        self.assertEqual("demo", command[board_index + 1])


if __name__ == "__main__":
    unittest.main()
