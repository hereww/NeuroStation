"""Injected Cyton adapters only: these tests never enumerate or open serial ports."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from eeg_tools import realtime_capture as capture
from eeg_tools.session_files import sha256
from eeg_tools.workstation.preflight import run_cyton_preflight

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "channel_config_v1_auto.json"


def samples(count, start=0):
    data = np.zeros((24, count), dtype=float)
    indices = np.arange(start, start + count)
    data[0] = indices % 256
    for row in range(1, 9):
        data[row] = row + indices / 997.0
    data[22] = 1700000000.0 + indices / 250.0
    data[23] = 123.45678901234567
    return data


class FakeShim:
    @staticmethod
    def get_eeg_channels(_):
        return list(range(1, 9))

    @staticmethod
    def get_sampling_rate(_):
        return 250

    @staticmethod
    def get_num_rows(_):
        return 24

    @staticmethod
    def get_package_num_channel(_):
        return 0

    @staticmethod
    def get_timestamp_channel(_):
        return 22


class FakeBoard:
    def __init__(self, chunks=(), *, stop=None, failure=""):
        self.chunks = list(chunks)
        self.stop = stop
        self.failure = failure
        self.calls = []
        self.starts = self.stops = self.releases = 0

    def start_stream(self):
        self.calls.append("start")
        self.starts += 1
        if self.failure == "start":
            raise RuntimeError("start failure")

    def get_board_data(self):
        self.calls.append("drain")
        if self.failure == "read":
            raise RuntimeError("read failure")
        if self.chunks:
            result = self.chunks.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        if self.stop is not None:
            self.stop.request()
        return samples(0)

    def stop_stream(self):
        self.calls.append("stop")
        self.stops += 1
        if self.failure == "stop":
            raise RuntimeError("stop failure")

    def release_session(self):
        self.calls.append("release")
        self.releases += 1
        if self.failure == "release":
            raise RuntimeError("release failure")


class Clock:
    def __init__(self):
        self.value = 0.0

    def now(self):
        return self.value

    def wait(self, seconds):
        self.value += seconds


class RealtimeCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.config = self.path / "channels.json"
        self.config.write_bytes(CONFIG.read_bytes())
        self.session = self.path / "session"
        self.events = []
        self.stop = capture.StopController()
        self.clock = Clock()
        self.prepare_calls = []

    def hardware(self, board, api=FakeShim, selected="COM5", callback=None):
        def prepare(board_id, params, port):
            self.prepare_calls.append((board_id, port))
            if callback:
                callback()
            return board, selected, []
        return capture.CaptureHardware(api, 0, object(), prepare)

    def preflight(self, status="passed", *, runner=None, stop=None):
        def default_runner(port, seconds, **kwargs):
            self.assertEqual(seconds, 3.0)
            self.assertIsNotNone(kwargs["sample_callback"])
            return {"status": status, "requested_port": port, "selected_port": "COM5", "checks": [], "error": ""}
        return capture.run_preflight("AUTO", self.config, self.session, stop=stop,
            emit=self.events.append, preflight_runner=runner or default_runner, wall_time=lambda: 1000.0)

    def run_board(self, board, **kwargs):
        options = dict(stop=self.stop, emit=self.events.append,
                       hardware=self.hardware(board), wall_time=lambda: 1001.0,
                       monotonic=self.clock.now, wait=self.clock.wait)
        options.update(kwargs)
        return capture.run_capture("AUTO", self.config, self.session, **options)

    def read_json(self, name):
        return json.loads((self.session / name).read_text(encoding="utf-8"))

    def test_complete_config_and_parser_have_no_simulation_switch(self):
        config, digest = capture.load_channel_config(self.config)
        self.assertEqual(len(config["channels"]), 8)
        self.assertEqual(digest, sha256(self.config))
        parser = capture.build_parser()
        self.assertEqual(parser._option_string_actions["--mode"].choices, ("preflight", "capture"))
        self.assertNotIn("--board", parser._option_string_actions)
        self.assertNotIn("--simulate", parser._option_string_actions)

    def test_reject_draft_template_incomplete_reordered_and_wrong_hardware(self):
        original = json.loads(self.config.read_text(encoding="utf-8"))
        mutations = [lambda c: c.update(status="draft"), lambda c: c.update(status="formal_candidate"),
            lambda c: c.update(config_version="CHANNEL-TEMPLATE"),
            lambda c: c["channels"][0].update(electrode_position=""),
            lambda c: c["reference"].update(position=""),
            lambda c: c["channels"][0].update(electrode_position="Fp2"),
            lambda c: c["channels"][0].update(board_input="N2P"),
            lambda c: c.update(sampling_rate_hz=125), lambda c: c.update(board="synthetic")]
        for mutate in mutations:
            value = json.loads(json.dumps(original))
            mutate(value)
            self.config.write_text(json.dumps(value), encoding="utf-8")
            with self.subTest(value=value), self.assertRaises(ValueError):
                capture.load_channel_config(self.config)

    def test_optional_complete_map_validator_is_called(self):
        with patch.object(capture.config_tools, "_validate_complete_channel_map", create=True) as validator:
            capture.load_channel_config(self.config)
        validator.assert_called_once()

    def test_preflight_envelope_and_failed_degraded_cancellation(self):
        for status in ("passed", "warning", "failed", "degraded", "cancelled"):
            with self.subTest(status=status):
                code = self.preflight(status)
                value = self.read_json("preflight.json")
                self.assertEqual(value, self.events[-1]["report"])
                self.assertEqual(value["channel_config_sha256"], sha256(self.config))
                self.assertEqual(value["configured_positions"], list(capture.CHANNEL_NAMES))
                self.assertEqual(value["position_verification"], "configuration_only")
                self.assertEqual(value["report"]["status"], status)
                self.assertEqual(code, 0 if status in {"passed", "warning"} else 2)
                self.assertEqual(value["status"], status if code == 0 else "failed")

    def test_shared_preflight_releases_in_success_and_failure(self):
        for failure in ("", "read", "start"):
            board = FakeBoard([samples(750)], failure=failure)
            def runner(port, seconds, **kwargs):
                # Exercise the existing preflight's finally with injected hardware,
                # no sleep and no live serial discovery.
                return run_cyton_preflight(port, seconds, board_shim=FakeShim,
                    board_ids=type("Ids", (), {"CYTON_BOARD": 0}), params=object(),
                    prepare_board=lambda *_: (board, "COM5", []), sleep_fn=lambda _: None)
            with self.subTest(failure=failure):
                code = self.preflight(runner=runner)
                self.assertEqual(board.releases, 1)
                self.assertEqual(code, 0 if not failure else 2)

    def test_invalid_config_preflight_is_saved_with_actual_hash_without_hardware(self):
        value = json.loads(self.config.read_text(encoding="utf-8"))
        value["channels"][0]["electrode_position"] = ""
        self.config.write_text(json.dumps(value), encoding="utf-8")
        def forbidden(*args, **kwargs):
            self.fail("Invalid configuration must not open hardware")
        self.assertEqual(self.preflight(runner=forbidden), 2)
        saved = self.read_json("preflight.json")
        self.assertEqual(saved["channel_config_sha256"], sha256(self.config))
        self.assertEqual(saved["status"], "failed")
        self.assertEqual(saved["configured_positions"], [])

    def test_production_reconnect_recovers_on_preflight_port_and_records_failed_attempt(self):
        self.preflight()
        first = Mock()
        first.prepare_session.side_effect = RuntimeError("BOARD_NOT_READY_ERROR:7 unable to prepare streaming session")
        recovered = Mock(wraps=FakeBoard([samples(3)], stop=self.stop))
        recovered.prepare_session = Mock()
        with patch("brainflow.board_shim.BoardShim", side_effect=[first, recovered]) as native, \
                patch("eeg_tools.workstation.acquisition_worker.time.sleep") as delay, \
                patch("eeg_tools.workstation.device_discovery.discover_serial_ports") as discover:
            real = capture.CaptureHardware.real()
            adapter = capture.CaptureHardware(FakeShim, real.board_id, real.params, real.prepare_board)
            self.assertEqual(self.run_board(recovered, hardware=adapter), 0)
        self.assertEqual(native.call_count, 2)
        discover.assert_not_called()
        delay.assert_called_once_with(0.5)
        first.release_session.assert_called_once()
        recovered.start_stream.assert_called_once()
        recovered.release_session.assert_called_once()
        self.assertEqual(real.params.serial_port, "COM5")
        session = self.read_json("session.json")
        self.assertEqual(session["sample_count"], 3)
        self.assertEqual(session["handshake_attempt_failures"][0]["port"], "COM5")
        self.assertEqual(session["handshake_attempt_failures"][0]["attempt"], "1")
        self.assertEqual(sum(event["type"] == "capture_ready" for event in self.events), 1)

    def test_production_reconnect_stops_after_three_failed_handshakes(self):
        self.preflight()
        boards = [Mock() for _ in range(3)]
        for board in boards:
            board.prepare_session.side_effect = RuntimeError("BOARD_NOT_READY_ERROR:7 unable to prepare streaming session")
        with patch("brainflow.board_shim.BoardShim", side_effect=boards) as native, \
                patch("eeg_tools.workstation.acquisition_worker.time.sleep") as delay, \
                patch("eeg_tools.workstation.device_discovery.discover_serial_ports") as discover:
            real = capture.CaptureHardware.real()
            adapter = capture.CaptureHardware(FakeShim, real.board_id, real.params, real.prepare_board)
            self.assertEqual(self.run_board(FakeBoard(), hardware=adapter), 2)
        self.assertEqual(native.call_count, 3)
        self.assertEqual(delay.call_count, 2)
        discover.assert_not_called()
        for board in boards:
            board.release_session.assert_called_once()
            board.start_stream.assert_not_called()
        self.assertFalse(any(event["type"] == "capture_ready" for event in self.events))
        error = next(event for event in self.events if event["type"] == "error")
        self.assertEqual((error["stage"], error["port"]), ("device_handshake", "COM5"))
        self.assertIn("BOARD_NOT_READY_ERROR:7", error["message"])
        self.assertEqual(self.read_json("session.json")["error_stage"], "device_handshake")

    def test_production_reconnect_does_not_retry_nontransient_errors(self):
        board = Mock()
        board.prepare_session.side_effect = RuntimeError("INVALID_ARGUMENTS_ERROR:13")
        with patch("brainflow.board_shim.BoardShim", return_value=board) as native, \
                patch("eeg_tools.workstation.acquisition_worker.time.sleep") as delay:
            real = capture.CaptureHardware.real()
            with self.assertRaisesRegex(RuntimeError, "INVALID_ARGUMENTS_ERROR"):
                real.prepare_board(real.board_id, real.params, "COM5")
        native.assert_called_once()
        delay.assert_not_called()
        board.release_session.assert_called_once()
        board.start_stream.assert_not_called()

    def test_hardware_prints_cannot_pollute_ndjson_stdout(self):
        self.preflight()
        class NoisyBoard(FakeBoard):
            def start_stream(inner):
                print("native wrapper starting")
                super().start_stream()
            def get_board_data(inner):
                print("native wrapper reading")
                return super().get_board_data()
            def stop_stream(inner):
                print("native wrapper stopping")
                super().stop_stream()
            def release_session(inner):
                print("native wrapper releasing")
                super().release_session()
        board = NoisyBoard([samples(3)], stop=self.stop)
        stream, logs = io.StringIO(), io.StringIO()
        with patch("sys.stderr", logs):
            code = self.run_board(board, emit=capture.NDJSONEmitter(stream, self.stop))
        self.assertEqual(code, 0)
        self.assertIn("native wrapper reading", logs.getvalue())
        messages = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual([message["type"] for message in messages], ["capture_ready", "samples", "capture_stopped"])

    def test_cancelled_before_hardware_does_not_prepare(self):
        self.preflight()
        self.stop.request("stdin_eof")
        board = FakeBoard()
        self.assertEqual(self.run_board(board), 0)
        self.assertEqual(self.prepare_calls, [])
        self.assertEqual(board.calls, [])
        self.assertEqual(self.read_json("session.json")["recording_status"], "cancelled")

    def test_handshake_failure_saves_failure_and_empty_raw(self):
        self.preflight()
        def fail(*_):
            raise RuntimeError("handshake failed")
        adapter = capture.CaptureHardware(FakeShim, 0, object(), fail)
        self.assertEqual(self.run_board(FakeBoard(), hardware=adapter), 2)
        session = self.read_json("session.json")
        self.assertEqual(session["recording_status"], "failed")
        self.assertEqual(session["sample_count"], 0)
        self.assertEqual(len((self.session / "raw_brainflow.tsv").read_text().splitlines()), 1)
        self.assertTrue((self.session / "manifest.csv").is_file())

    def test_preflight_cancellation_reuses_polling_path_and_releases(self):
        board = FakeBoard()
        def runner(port, seconds, **kwargs):
            return run_cyton_preflight(port, seconds, board_shim=FakeShim,
                board_ids=type("Ids", (), {"CYTON_BOARD": 0}), params=object(),
                prepare_board=lambda *_: (board, "COM5", []),
                sleep_fn=lambda _: self.stop.request("stdin_eof"),
                sample_callback=kwargs["sample_callback"], cancel_event=kwargs["cancel_event"])
        self.assertEqual(self.preflight(runner=runner, stop=self.stop), 2)
        self.assertTrue(self.read_json("preflight.json")["cancelled"])
        self.assertEqual((board.stops, board.releases), (1, 1))

    def test_drains_split_batches_preserve_all_rows_precision_and_duplicate_timestamps(self):
        self.preflight()
        first, second = samples(601), samples(3, 601)
        first[22, 20:23] = first[22, 19]
        first[1, 0] = 1.2345678901234567
        board = FakeBoard([first, second], stop=self.stop)
        self.assertEqual(self.run_board(board), 0)
        events = [item for item in self.events if item["type"] == "samples"]
        self.assertEqual([len(item["sample_indices"]) for item in events], [250, 250, 101, 3])
        self.assertEqual([index for item in events for index in item["sample_indices"]], list(range(604)))
        self.assertEqual([number for item in events for number in item["package_numbers"]], (np.arange(604) % 256).tolist())
        np.testing.assert_array_equal(np.concatenate([np.asarray(item["samples"]) for item in events], axis=1), np.concatenate([first[1:9], second[1:9]], axis=1))
        ready = next(item for item in self.events if item["type"] == "capture_ready")
        self.assertLess(self.events.index(ready), self.events.index(events[0]))
        self.assertEqual(ready["port"], "COM5")
        self.assertEqual(self.prepare_calls, [(0, "COM5")])
        self.assertEqual((board.starts, board.stops, board.releases), (1, 1, 1))
        self.assertNotIn("get_current_board_data", board.calls)
        with (self.session / "raw_brainflow.tsv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream, delimiter="\t"))
        self.assertEqual((len(rows), len(rows[0])), (605, 25))
        self.assertEqual(rows[1][2], "1.2345678901234567")
        np.testing.assert_array_equal(np.asarray([row[1:] for row in rows[1:]], dtype=float).T, np.concatenate([first, second], axis=1))
        quality = self.read_json("quality.json")
        self.assertEqual(quality["sample_count"], 604)
        self.assertEqual(quality["packet_loss_count"], 0)
        self.assertEqual(quality["timestamp_nonincreasing_count"], 3)
        metadata = self.read_json("session.json")
        self.assertEqual(metadata["status"], "formal")
        self.assertEqual(metadata["source"], "standalone_realtime_commands")
        self.assertEqual(metadata["recording_status"], "completed")
        self.assertTrue(metadata["completed"])
        self.assertFalse(metadata["cancelled"])
        self.assertEqual(metadata["raw_brainflow_sha256"], sha256(self.session / "raw_brainflow.tsv"))
        with (self.session / "manifest.csv").open(newline="", encoding="utf-8") as stream:
            manifest = list(csv.DictReader(stream))
        self.assertEqual({row["file_name"] for row in manifest}, {"raw_brainflow.tsv", "raw_columns.tsv", "quality.json", "session.json", "preflight.json"})
        for row in manifest:
            self.assertEqual(row["sha256"], sha256(Path(row["file_path"])))

    def test_packet_quality_wrap_duplicates_gaps_and_invalid_across_batches(self):
        self.preflight()
        first, second = samples(2), samples(6, 2)
        first[0] = [254, 255]
        second[0] = [0, 0, 3, np.nan, 7.5, 8]
        first[1] = 187000
        second[1] = [187000, 187000, np.nan, np.inf, -187000, -187000]
        second[22, 3] = np.nan
        board = FakeBoard([first, second], stop=self.stop)
        self.assertEqual(self.run_board(board), 0)
        quality = self.read_json("quality.json")
        self.assertEqual(quality["packet_loss_count"], 2)
        self.assertEqual(quality["packet_duplicate_count"], 1)
        self.assertEqual(quality["packet_sequence_mismatch_count"], 2)
        self.assertEqual(quality["invalid_package_count"], 2)
        self.assertEqual(quality["nonfinite_count"], 4)
        self.assertEqual(quality["timestamp_nonfinite_count"], 1)
        self.assertEqual(quality["channels"]["Fp1"]["saturation_count"], 6)
        self.assertEqual(quality["channels"]["Fp1"]["flat_pair_count"], 4)
        self.assertEqual(quality["channels"]["Fp1"]["finite_pair_count"], 4)
        second_event = [item for item in self.events if item["type"] == "samples"][1]
        self.assertEqual(second_event["samples"][0][2:4], [None, None])
        self.assertIsNone(second_event["package_numbers"][3])
        self.assertIsNone(second_event["timestamps_s"][3])
        json.dumps(second_event, allow_nan=False)
        self.assertIn("nan", (self.session / "raw_brainflow.tsv").read_text())
        self.assertIn("inf", (self.session / "raw_brainflow.tsv").read_text())

    def test_gate_rejects_before_opening_hardware_and_saves_failure(self):
        cases = ("missing", "failed", "degraded", "inner_degraded", "failed_check", "hash", "map", "port", "inner_port", "stale", "future", "nonfinite_age", "cancelled")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                self.session = Path(directory)
                if case != "missing":
                    self.preflight()
                    value = self.read_json("preflight.json")
                    if case in {"failed", "degraded"}:
                        value["status"] = case
                    elif case == "inner_degraded":
                        value["report"]["status"] = "degraded"
                    elif case == "failed_check":
                        value["report"]["checks"] = [{"status": "failed"}]
                    elif case == "hash":
                        value["channel_config_sha256"] = "bad"
                    elif case == "map":
                        value["configured_positions"] = []
                    elif case == "port":
                        value["port"] = "COM9"
                    elif case == "inner_port":
                        value["report"]["selected_port"] = "COM9"
                    elif case == "stale":
                        value["completed_timestamp_s"] = 700.0
                    elif case == "future":
                        value["completed_timestamp_s"] = 1002.0
                    elif case == "nonfinite_age":
                        value["completed_timestamp_s"] = float("nan")
                    elif case == "cancelled":
                        value["cancelled"] = True
                    (self.session / "preflight.json").write_text(json.dumps(value), encoding="utf-8")
                board = FakeBoard(stop=self.stop)
                self.assertEqual(self.run_board(board), 2)
                self.assertEqual(board.calls, [])
                self.assertEqual(self.read_json("session.json")["recording_status"], "failed")
                self.assertEqual(self.read_json("quality.json")["status"], "failed")
                self.assertTrue((self.session / "manifest.csv").is_file())

    def test_explicit_port_mismatch_and_age_boundary(self):
        self.preflight()
        _, digest = capture.load_channel_config(self.config)
        self.assertEqual(capture.validate_preflight(self.session, digest, "\\\\.\\com5", 1300.0)["port"], "COM5")
        with self.assertRaisesRegex(ValueError, "port mismatch"):
            capture.validate_preflight(self.session, digest, "COM6", 1001.0)
        with self.assertRaisesRegex(ValueError, "stale"):
            capture.validate_preflight(self.session, digest, "AUTO", 1300.001)

    def test_wrong_rate_or_eeg_count_never_prepares(self):
        for api in (type("BadRate", (FakeShim,), {"get_sampling_rate": staticmethod(lambda _: 125)}),
                    type("BadChannels", (FakeShim,), {"get_eeg_channels": staticmethod(lambda _: list(range(16)))})):
            with self.subTest(api=api), tempfile.TemporaryDirectory() as directory:
                self.session = Path(directory)
                self.preflight()
                board = FakeBoard(stop=self.stop)
                self.assertEqual(self.run_board(board, hardware=self.hardware(board, api)), 2)
                self.assertEqual(board.calls, [])

    def test_errors_preserve_samples_and_release(self):
        for failure in ("start", "read", "stop", "release", "shape", "after_samples"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                self.session = Path(directory)
                self.stop = capture.StopController()
                self.preflight()
                chunks = [samples(4)]
                if failure == "shape":
                    chunks = [np.zeros((3, 5))]
                elif failure == "after_samples":
                    chunks.append(RuntimeError("connection lost"))
                board = FakeBoard(chunks, stop=self.stop, failure=failure)
                self.assertEqual(self.run_board(board), 2)
                self.assertEqual(board.stops, 1)
                self.assertEqual(board.releases, 1)
                session = self.read_json("session.json")
                self.assertEqual(session["recording_status"], "failed")
                self.assertEqual(session["status"], "formal")
                self.assertEqual(self.events[-1]["type"], "capture_stopped")
                self.assertEqual(self.read_json("quality.json")["status"], "failed")
                if failure == "after_samples":
                    self.assertEqual(session["sample_count"], 4)

    def test_final_drain_keeps_tail_on_stop(self):
        self.preflight()
        board = FakeBoard([samples(2), samples(3, 2)])
        def emit(value):
            self.events.append(value)
            if value["type"] == "samples":
                self.stop.request()
        self.assertEqual(self.run_board(board, emit=emit), 0)
        self.assertEqual(self.read_json("session.json")["sample_count"], 5)
        self.assertEqual(board.calls, ["start", "drain", "stop", "drain", "release"])

    def test_stop_during_prepare_does_not_start_and_still_releases(self):
        self.preflight()
        board = FakeBoard()
        self.assertEqual(self.run_board(board, hardware=self.hardware(board, callback=lambda: self.stop.request("stdin_eof"))), 0)
        self.assertEqual((board.starts, board.stops, board.releases), (0, 0, 1))
        self.assertEqual(self.read_json("session.json")["recording_status"], "cancelled")

    def test_handshake_port_mismatch_releases_without_start(self):
        self.preflight()
        board = FakeBoard()
        self.assertEqual(self.run_board(board, hardware=self.hardware(board, selected="COM8")), 2)
        self.assertEqual((board.starts, board.releases), (0, 1))

    def test_empty_stream_warns_then_fails_without_waiting_real_seconds(self):
        self.preflight()
        board = FakeBoard()
        self.assertEqual(self.run_board(board), 2)
        warnings = [item for item in self.events if item["type"] == "warning"]
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]["code"], "no_data")
        self.assertGreaterEqual(warnings[0]["empty_duration_s"], 1.5)
        self.assertGreater(self.clock.value, 5)
        self.assertEqual((board.stops, board.releases), (1, 1))

    def test_stdin_daemon_stop_eof_malformed_and_reader_error(self):
        for text, reason in (("{\"type\":\"stop\"}\n", "stop_command"), ("", "stdin_eof"),
                             ("garbage\n[]\n{\"type\":\"stop\"}\n", "stop_command")):
            stop = capture.StopController()
            reader = capture.start_stdin_reader(io.StringIO(text), stop)
            self.assertTrue(reader.daemon)
            self.assertTrue(stop.event.wait(1))
            reader.join(1)
            self.assertEqual(stop.reason, reason)
        class FailedReader:
            def readline(self):
                raise OSError("closed")
        stop = capture.StopController()
        capture.start_stdin_reader(FailedReader(), stop).join(1)
        self.assertEqual(stop.reason, "stdin_error")

    def test_stdin_eof_while_streaming_stops_and_releases(self):
        self.preflight()
        board = FakeBoard([samples(3)])
        def emit(value):
            self.events.append(value)
            if value["type"] == "samples":
                capture.start_stdin_reader(io.StringIO(""), self.stop).join(1)
        self.assertEqual(self.run_board(board, emit=emit), 0)
        self.assertEqual((board.stops, board.releases), (1, 1))
        self.assertEqual(self.read_json("session.json")["recording_status"], "cancelled")
        self.assertEqual(self.events[-1]["stop_reason"], "stdin_eof")

    def test_closed_stdout_requests_stop_and_releases(self):
        self.preflight()
        class BrokenOutput:
            def write(self, _):
                raise BrokenPipeError()
        board = FakeBoard()
        emitter = capture.NDJSONEmitter(BrokenOutput(), self.stop)
        self.assertEqual(self.run_board(board, emit=emitter), 0)
        self.assertEqual((board.stops, board.releases), (1, 1))
        self.assertEqual(self.read_json("session.json")["stop_reason"], "stdout_closed")
        self.assertEqual(self.read_json("quality.json")["status"], "warning")

    def test_existing_raw_is_never_overwritten(self):
        self.preflight()
        path = self.session / "raw_brainflow.tsv"
        path.write_text("precious raw\n")
        board = FakeBoard()
        self.assertEqual(self.run_board(board), 2)
        self.assertEqual(path.read_text(), "precious raw\n")
        self.assertFalse((self.session / "session.json").exists())
        self.assertEqual(board.calls, [])

    def test_preflight_cannot_rewrite_historical_recording_evidence(self):
        self.preflight()
        before = (self.session / "preflight.json").read_bytes()
        (self.session / "raw_brainflow.tsv").write_text("original raw")
        self.assertEqual(self.preflight("failed"), 2)
        self.assertEqual((self.session / "preflight.json").read_bytes(), before)
        self.assertEqual((self.session / "raw_brainflow.tsv").read_text(), "original raw")
        self.assertEqual(self.events[-1]["code"], "session_exists")

    def test_json_emitter_uses_strict_ndjson_and_null_for_nonfinite(self):
        stream = io.StringIO()
        capture.NDJSONEmitter(stream, self.stop)({"type": "test", "values": [np.nan, np.inf, -np.inf, 1.5]})
        self.assertEqual(json.loads(stream.getvalue())["values"], [None, None, None, 1.5])
        self.assertEqual(len(stream.getvalue().splitlines()), 1)

    def test_cli_bad_args_is_ndjson_and_stdin_eof_never_opens_hardware(self):
        script = ROOT / "scripts" / "run_realtime_capture.py"
        result = subprocess.run([sys.executable, str(script), "--mode", "fake"], input="", text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["type"], "error")
        self.preflight()
        result = subprocess.run([sys.executable, str(script), "--mode", "capture", "--port", "AUTO",
                                 "--channel-config", str(self.config), "--session-dir", str(self.session)],
                                 input="", text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 2)  # Fixture is intentionally stale at real wall time.
        self.assertEqual(json.loads(result.stdout.splitlines()[-1])["type"], "capture_stopped")
        self.assertFalse(self.read_json("session.json")["capture_started"])


if __name__ == "__main__":
    unittest.main()
