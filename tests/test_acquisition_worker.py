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

    def test_cyton_prepare_explains_board_not_ready_as_radio_handshake(self) -> None:
        from unittest.mock import patch

        class FakeBoard:
            def __init__(self, _board_id, _params):
                return None

            def prepare_session(self):
                raise RuntimeError(
                    "BrainFlowError: BOARD_NOT_READY_ERROR:7 "
                    "unable to prepare streaming session"
                )

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
            self.assertIn("display", session)
            self.assertEqual({}, session["display"])
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
            self.assertEqual(11, session["event_count"])
            self.assertGreater(session["recorded_samples_per_channel"], 0)
            self.assertGreater((session_dir / "raw_brainflow.tsv").stat().st_size, 0)
            raw_header = (session_dir / "raw_brainflow.tsv").read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("sample_index", raw_header)
            self.assertIn("timestamp_s", raw_header)
            self.assertTrue((session_dir / "raw_columns.tsv").is_file())
            event_header = (session_dir / "events.tsv").read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("sample_index", event_header)
            self.assertIn("presented_target_id", event_header)
            self.assertTrue((session_dir / "manifest.csv").is_file())
            quality = json.loads((session_dir / "quality.json").read_text(encoding="utf-8"))
            self.assertEqual("ok", quality["status"])
            self.assertEqual(session["channel_count"], quality["channel_count"])
            self.assertEqual(session["recorded_samples_per_channel"], quality["samples_per_channel"])
            self.assertTrue((session_dir / "ssvep_config.json").is_file())
            self.assertEqual("technical_validation", session["validation_mode"])
            self.assertEqual(session["protocol_provenance"]["sha256"], session["protocol_provenance"]["sha256"].lower())
            self.assertEqual(64, len(session["protocol_provenance"]["sha256"]))


    def test_cyton_preflight_reports_handshake_samples_and_channel_quality(self) -> None:
        import numpy as np
        from eeg_tools.workstation.preflight import run_cyton_preflight

        class FakeIds:
            CYTON_BOARD = 6

        class FakeBoardShim:
            @staticmethod
            def get_eeg_channels(_board_id):
                return [0, 1]

            @staticmethod
            def get_timestamp_channel(_board_id):
                return 2

            @staticmethod
            def get_sampling_rate(_board_id):
                return 250

        class FakeBoard:
            def __init__(self):
                self.started = False

            def start_stream(self):
                self.started = True

            def stop_stream(self):
                self.started = False

            def release_session(self):
                return None

            def get_board_data(self):
                return np.array([[1, 2, 3, 4], [4, 4, 4, 4], [0.0, 0.004, 0.008, 0.012]])

        board = FakeBoard()
        attempts = []
        def prepare(_board_id, _params, _requested):
            attempts.append(True)
            if len(attempts) == 1:
                raise RuntimeError("transient handshake miss")
            return board, "COM5", []

        report = run_cyton_preflight(prepare_board=prepare, board_shim=FakeBoardShim, board_ids=FakeIds, sleep_fn=lambda _seconds: None)
        self.assertEqual("warning", report.status)
        self.assertEqual("COM5", report.selected_port)
        self.assertEqual("passed", report.checks[0].status)
        self.assertEqual("warning", report.checks[-1].status)
        self.assertEqual([2], report.checks[-1].metrics["warning_channels"])
        self.assertEqual(2, len(attempts))
        self.assertEqual("1", report.checks[0].metrics["attempt_failures"][0]["attempt"])

    def test_cyton_preflight_separates_timestamp_jitter_from_packet_loss(self) -> None:
        import numpy as np
        from eeg_tools.workstation.preflight import run_cyton_preflight

        class FakeIds:
            CYTON_BOARD = 6

        class FakeBoardShim:
            @staticmethod
            def get_eeg_channels(_board_id):
                return [0, 1]

            @staticmethod
            def get_timestamp_channel(_board_id):
                return 2

            @staticmethod
            def get_package_num_channel(_board_id):
                return 3

            @staticmethod
            def get_sampling_rate(_board_id):
                return 250

        class FakeBoard:
            def start_stream(self):
                return None

            def stop_stream(self):
                return None

            def release_session(self):
                return None

            def get_board_data(self):
                return np.array([
                    [1, 2, 3, 4],
                    [4, 5, 6, 7],
                    [0.0, 0.004, 0.012, 0.016],
                    [10, 11, 12, 13],
                ])

        report = run_cyton_preflight(
            prepare_board=lambda *_args: (FakeBoard(), "COM5", []),
            board_shim=FakeBoardShim,
            board_ids=FakeIds,
            sleep_fn=lambda _seconds: None,
        )
        timestamps = next(item for item in report.checks if item.name == "timestamps").metrics
        packet_sequence = next(item for item in report.checks if item.name == "packet_sequence").metrics
        self.assertEqual("warning", report.status)
        self.assertTrue(packet_sequence["packet_sequence_available"])
        self.assertEqual(0, packet_sequence["packet_loss_count"])
        self.assertEqual(0, packet_sequence["packet_sequence_mismatch_count"])
        self.assertEqual(1, timestamps["timestamp_gap_count"])

    def test_cyton_preflight_classifies_large_timestamp_gaps(self) -> None:
        import numpy as np
        from eeg_tools.workstation.preflight import run_cyton_preflight
        class Ids: CYTON_BOARD = 6
        class Shim:
            get_eeg_channels = staticmethod(lambda _: [0])
            get_timestamp_channel = staticmethod(lambda _: 1)
            get_sampling_rate = staticmethod(lambda _: 250)
        class Board:
            start_stream = lambda self: None
            stop_stream = lambda self: None
            release_session = lambda self: None
            get_board_data = lambda self: np.array([[1, 2, 3], [0.0, 0.6, 1.2]])
        report = run_cyton_preflight(prepare_board=lambda *_: (Board(), "COM5", []), board_shim=Shim, board_ids=Ids, sleep_fn=lambda _: None)
        self.assertEqual("degraded", next(c for c in report.checks if c.name == "timestamps").status)

    def test_cyton_preflight_fails_on_timestamp_reversal(self) -> None:
        import numpy as np
        from eeg_tools.workstation.preflight import run_cyton_preflight
        class Ids: CYTON_BOARD = 6
        class Shim:
            get_eeg_channels = staticmethod(lambda _: [0])
            get_timestamp_channel = staticmethod(lambda _: 1)
            get_sampling_rate = staticmethod(lambda _: 250)
        class Board:
            start_stream = lambda self: None
            stop_stream = lambda self: None
            release_session = lambda self: None
            get_board_data = lambda self: np.array([[1, 2, 3], [0.0, 0.004, 0.002]])
        report = run_cyton_preflight(prepare_board=lambda *_: (Board(), "COM5", []), board_shim=Shim, board_ids=Ids, sleep_fn=lambda _: None)
        self.assertEqual("failed", report.status)
        self.assertEqual("failed", next(c for c in report.checks if c.name == "timestamps").status)

    def test_worker_persists_preflight_report_when_supplied(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "preflight-session"
            preflight = Path(directory) / "preflight.json"
            preflight.write_text(json.dumps({"status": "passed", "selected_port": "COM5", "checks": []}), encoding="utf-8")
            exit_code = worker_main([
                "--protocol", str(PROTOCOL),
                "--output-root", str(output),
                "--participant", "PREFLIGHT",
                "--session-name", "preflight metadata",
                "--board", "synthetic", "--headless",
                "--repetitions", "1", "--stimulus-seconds", "0.01",
                "--rest-seconds", "0", "--countdown-seconds", "0.01",
                "--preflight-file", str(preflight),
            ])
            self.assertEqual(0, exit_code)
            session_dir = next(output.glob("session_*"))
            session = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
            self.assertEqual("passed", session["preflight"]["status"])
            self.assertTrue((session_dir / "preflight.json").is_file())
            self.assertIn("preflight.json", (session_dir / "manifest.csv").read_text(encoding="utf-8"))

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
            self.assertTrue((cancelled.result.path / "status.final.json").is_file())
            self.assertTrue((cancelled.result.path / "worker.log").is_file())
            manifest = (cancelled.result.path / "manifest.csv").read_text(encoding="utf-8")
            self.assertIn("status.final.json", manifest)

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
