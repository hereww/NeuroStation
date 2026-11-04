"""Standalone controller tests with fake devices and no physical display."""
from __future__ import annotations
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication
from apps import realtime_commands as ui
from eeg_tools.realtime_commands import CHANNEL_NAMES


class FakeWorker(QObject):
    message = Signal(object)
    failed = Signal(str)
    finished = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.starts = []
        self.requests = []
        self.stop_calls = 0

    def start(self, *args):
        self.starts.append(args)
        self.running = True

    def send(self, message):
        self.requests.append(deepcopy(message))
        return self.running

    def stop(self):
        self.stop_calls += 1

    def complete(self, code=0):
        self.running = False
        self.finished.emit(code)


class FakeDisplay(QObject):
    calibration_finished = Signal(object)
    timing_failed = Signal(str)
    aborted = Signal()
    frame_record = Signal(object)
    start_requested = Signal()

    def __init__(self, screen):
        super().__init__()
        self.mode = "idle"
        self.calibration_report = None
        self.calibration_valid = False
        self.flicker_starts = 0
        self.status_text = ""

    def start_calibration(self):
        self.mode = "calibration"
        return True

    def finish_calibration(self, passed=True):
        self.mode = "idle"
        self.calibration_valid = passed
        self.calibration_report = {"passed": passed, "status": "passed" if passed else "failed",
                                   "completed_monotonic_s": time.perf_counter(),
                                   "reason": "test_fixture_only", "identity": {"fixture_only": True}}
        self.calibration_finished.emit(self.calibration_report)

    def start_flicker(self):
        if not self.calibration_valid:
            return False
        self.flicker_starts += 1
        self.mode = "flicker"
        return True

    def stop_flicker(self):
        self.mode = "idle"

    def set_status_text(self, text):
        self.status_text = text

    def close(self):
        self.mode = "idle"


def ready():
    return {**ui.FLAGS, "type": "ready", "model_loaded": True,
            "channel_names": list(CHANNEL_NAMES), "sample_rate_hz": 250,
            "sample_count": 1250, "training_sample_count": 1250, "input_features": 72,
            "device": "cpu", "command_gate": {"min_probability": .65, "min_probability_margin": .15},
            "ood_gate": {"maxabs_limit": 8, "p95abs_limit": 4, "heuristic": True, "validated": False}}


def batch(start, *, packet_offset=0):
    return {"type": "samples", "sample_rate_hz": 250, "channel_names": list(CHANNEL_NAMES),
            "samples": [[float((i + start) % 101 + ch) for i in range(250)] for ch in range(8)],
            "sample_indices": list(range(start, start + 250)),
            "package_numbers": [(i + start + packet_offset) % 256 for i in range(250)],
            "timestamps_s": [100.] * 250, "received_monotonic_s": time.monotonic(),
            "quality": {"status": "passed", "fixture_only": True},
            # Synthetic control-flow authority only; never a hardware-validation record.
            "quality_v2": {"schema_version": 2, "fixture_only": True,
                           "processing_id": "quality-fixture", "stream_id": "fixture-stream",
                           "start": start, "stop": start + 250, "updated_monotonic_s": time.monotonic(),
                           "dimensions": {k: "pass" for k in ("integrity", "signal_quality", "coverage", "stimulus_timing")}}}


def prediction(request, index=2):
    probabilities = [.05] * 3
    probabilities[index] = .9
    frequency = (10, 12, 15)[index]
    command = ui.COMMANDS[frequency]
    return {**ui.FLAGS, "type": "prediction", "request_id": request["request_id"],
            "generation": request["generation"], "reason": "accepted", "model_loaded": True,
            "sample_count": 1250, "training_sample_count": 1250, "window_matches_training": True,
            "probabilities": probabilities, "prediction": {"command": command, "accepted": True, "executed": False},
            "raw_prediction": {"command": command, "accepted": True, "executed": False,
                               "predicted_frequency_hz": frequency, "probabilities": probabilities, "margin": .85},
            "ood": {"passed": True, "heuristic": True, "validated": False, "maxabs_limit": 8,
                    "p95abs_limit": 4, "maxabs": 1, "p95abs": .5}}


class ControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.stdout = io.StringIO()
        self.patches = [patch.object(ui, "JsonWorker", FakeWorker),
                        patch.object(ui, "CommandStimulusWindow", FakeDisplay), patch("sys.stdout", self.stdout)]
        for item in self.patches:
            item.start()
        self.window = ui.RealtimeCommandsWindow()
        self.window.timer.stop()
        self.window.data_root_edit.setText(self.temp.name)
        self.window.model_python_edit.setText(str(Path(ui.sys.executable)))

    def tearDown(self):
        self.window.stop("test_cleanup")
        self.window.capture_worker.complete()
        self.window.model_worker.complete()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def prepare(self):
        self.window.electrode_check.setChecked(True)
        self.window.risk_check.setChecked(True)
        self.window._start_preflight()
        report = {"schema_version": 1, "type": "cyton_realtime_preflight", "status": "passed",
                  "completed_timestamp_s": time.time(), "port": "COM9", "requested_port": "AUTO",
                  "channel_config_sha256": self.window._settings["config_hash"],
                  "configured_positions": list(CHANNEL_NAMES), "cancelled": False,
                  "report": {"status": "passed", "selected_port": "COM9", "checks": []}}
        self.window.model_worker.message.emit(ready())
        self.window.capture_worker.message.emit({"type": "preflight", "report": report})
        self.window.capture_worker.complete()
        self.assertEqual(self.window.state, "prepared")

    def run_capture(self):
        self.prepare()
        self.window._start_calibration()
        self.window.display.finish_calibration()
        self.window.display.start_requested.emit()
        self.assertEqual(self.window.state, "capture_starting")
        self.assertEqual(self.window.display.flicker_starts, 0)
        self.window.capture_worker.message.emit({"type": "capture_ready", "status": "formal",
            "recording_status": "recording", "sample_rate_hz": 250, "channel_names": list(CHANNEL_NAMES),
            "channel_config_sha256": self.window._settings["config_hash"], "port": "COM9",
            "session_dir": str(self.window.session_dir)})
        self.assertEqual(self.window.state, "running")
        self.assertEqual(self.window.display.flicker_starts, 1)

    def fill_window(self):
        for start in range(0, 1500, 250):
            self.window.capture_worker.message.emit(batch(start))
        self.assertEqual(len(self.window.model_worker.requests), 1)
        return self.window.model_worker.requests[-1]

    def test_missing_quality_cannot_send_or_accept(self):
        self.run_capture()
        for start in range(0, 1500, 250):
            message = batch(start)
            message.pop("quality_v2")
            self.window.capture_worker.message.emit(message)
        self.assertEqual(self.window.model_worker.requests, [])
        self.assertFalse(any(r.get("accepted") for r in self.outputs()))

    def outputs(self):
        return [json.loads(line) for line in self.stdout.getvalue().splitlines()]
    def test_opening_starts_no_process_or_stimulus(self):
        self.assertEqual(self.window.state, "idle")
        self.assertFalse(self.window.electrode_check.isChecked())
        self.assertFalse(self.window.risk_check.isChecked())
        self.assertFalse(self.window.udp_check.isChecked())
        self.window._start_preflight()
        self.assertEqual(self.window.capture_worker.starts, [])
        self.assertEqual(self.window.model_worker.starts, [])
        self.assertIsNone(self.window.display)

    def test_capture_does_not_start_without_valid_calibration(self):
        self.prepare()
        self.window._request_start()
        self.assertEqual(len(self.window.capture_worker.starts), 1)
        self.assertEqual(self.window.state, "stopping")
        self.assertEqual(self.outputs()[-1]["command"], "UNKNOWN")

    def test_declined_display_calibration_stops_and_saves(self):
        self.prepare()
        self.window._start_calibration()
        self.window.display.finish_calibration(False)
        self.assertEqual(self.window.state, "stopping")
        self.assertEqual(len(self.window.capture_worker.starts), 1)

    def test_full_window_and_no_target_label_leaks(self):
        self.run_capture()
        first = batch(0)
        self.window.capture_worker.message.emit(first)
        self.assertEqual(self.window.buffer.sample_count, 0)
        self.assertEqual(self.window.model_worker.requests, [])
        for start in range(250, 1250, 250):
            value = batch(start)
            value["target_label"] = "SHOULD_NEVER_REACH_MODEL"
            self.window.capture_worker.message.emit(value)
        self.assertEqual(self.window.model_worker.requests, [])
        self.window.capture_worker.message.emit(batch(1250))
        request = self.window.model_worker.requests[-1]
        self.assertEqual(set(request), {"type", "request_id", "generation", "sample_rate_hz", "channel_names", "samples"})
        self.assertEqual((len(request["samples"]), len(request["samples"][0])), (8, 1250))
        self.assertEqual(request["samples"][0][0], batch(250)["samples"][0][0])
        self.assertEqual(self.window._inflight["window_metadata"]["sample_start"], 250)
        self.assertEqual(self.window._inflight["window_metadata"]["sample_end"], 1499)

    def test_model_select_reaches_command_log_and_stdout(self):
        self.run_capture()
        request = self.fill_window()
        self.window.model_worker.message.emit(prediction(request))
        record = self.outputs()[-1]
        self.assertEqual(record["command"], "SELECT")
        self.assertTrue(record["accepted"])
        self.assertFalse(record["executed"])
        self.assertFalse(record["personal_validated"])
        self.assertEqual((record["sample_start"], record["sample_end"]), (250, 1499))
        self.assertGreaterEqual(record["event_id"], 0)
        self.assertEqual(record["session_id"], self.window.session_dir.name)
        self.assertEqual(record["valid_for_ms"], 1500)
        log = [json.loads(line) for line in (self.window.session_dir / "commands.ndjson").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(log[-1], record)

    def test_gap_invalidates_pending_response_and_requires_new_window(self):
        self.run_capture()
        request = self.fill_window()
        self.window.capture_worker.message.emit(batch(1500, packet_offset=1))
        self.assertLess(self.window.buffer.sample_count, 1250)
        self.assertEqual(self.outputs()[-1]["command"], "UNKNOWN")
        self.window.model_worker.message.emit(prediction(request, 0))
        self.assertFalse(any(row["accepted"] for row in self.outputs()))
        self.assertIsNone(self.window._inflight)

    def test_stale_prediction_cannot_move_even_before_watchdog_tick(self):
        self.run_capture()
        request = self.fill_window()
        self.window._last_samples = time.monotonic() - 1.1
        self.window.model_worker.message.emit(prediction(request, 1))
        self.assertFalse(any(row["accepted"] for row in self.outputs()))
        self.assertEqual(self.outputs()[-1]["command"], "UNKNOWN")
        self.assertEqual(self.window.buffer.sample_count, 0)

    def test_two_second_timeout_keeps_single_physical_request(self):
        self.run_capture()
        request = self.fill_window()
        self.window.gate._pending["deadline"] = time.monotonic() - .01
        self.window._tick()
        self.assertEqual(self.outputs()[-1]["reason"], "prediction_timeout")
        self.window.capture_worker.message.emit(batch(1500))
        self.assertEqual(len(self.window.model_worker.requests), 1)
        self.window.model_worker.message.emit(prediction(request))
        self.assertFalse(any(row["accepted"] for row in self.outputs()))
        self.window.capture_worker.message.emit(batch(1750))
        self.assertEqual(len(self.window.model_worker.requests), 2)

    def test_stop_immediate_unknown_then_wait_both_workers(self):
        self.run_capture()
        request = self.fill_window()
        self.window.model_worker.message.emit(prediction(request, 0))
        session = self.window.session_dir
        self.window.stop("test_operator_stop")
        self.assertEqual(self.outputs()[-1]["command"], "UNKNOWN")
        self.assertEqual(self.outputs()[-1]["valid_for_ms"], 0)
        self.assertEqual(self.window.state, "stopping")
        self.assertFalse((session / "command_manifest.json").exists())
        self.window.capture_worker.complete()
        self.assertEqual(self.window.state, "stopping")
        self.window.model_worker.complete()
        self.assertEqual(self.window.state, "idle")
        self.assertTrue((session / "command_manifest.json").is_file())

    def test_display_timing_failure_forces_unknown_and_stop(self):
        self.run_capture()
        self.window.display.timing_failed.emit("synthetic missed vsync")
        self.assertEqual(self.window.state, "stopping")
        self.assertEqual(self.outputs()[-1]["command"], "UNKNOWN")

    def test_handshake_failure_after_calibration_preserves_detail_without_flickering(self):
        self.prepare()
        self.window._start_calibration()
        self.window.display.finish_calibration()
        self.window.display.start_requested.emit()
        display = self.window.display
        session = self.window.session_dir
        detail = "BOARD_NOT_READY_ERROR:7 unable to prepare streaming session"
        self.window.capture_worker.message.emit({"type": "error", "code": "capture_failed",
            "stage": "device_handshake", "port": "COM9", "message": detail})
        self.assertEqual(self.window.state, "stopping")
        self.assertEqual(display.flicker_starts, 0)
        command = self.outputs()[-1]
        self.assertEqual(command["command"], "UNKNOWN")
        self.assertIn(detail, command["reason"])
        self.assertIn("COM9", command["reason"])
        self.assertIn("device_handshake", command["reason"])
        self.window.capture_worker.complete(2)
        self.window.model_worker.complete()
        self.assertEqual(self.window.state, "idle")
        self.assertIn("已中止", self.window.status_label.text())
        self.assertIn(detail, self.window.status_label.text())
        self.assertIn(str(session / "capture.stderr.log"), self.window.status_label.text())
        self.assertIn(detail, self.window.reason_label.text())
        self.assertTrue((session / "command_manifest.json").is_file())
        self.prepare()
        self.assertIsNone(self.window._failure_reason)
        self.assertNotIn(detail, self.window.status_label.text())

    def test_user_stop_remains_a_normal_completed_shutdown(self):
        self.run_capture()
        self.window.stop("user_stop")
        self.window.capture_worker.complete()
        self.window.model_worker.complete()
        self.assertEqual(self.window.status_label.text(), "已停止；会话与独立命令清单已保存")
        self.assertIsNone(self.window._failure_reason)

    def test_duplicate_model_response_cannot_repeat_command(self):
        self.run_capture()
        request = self.fill_window()
        response = prediction(request, 1)
        self.window.model_worker.message.emit(response)
        count = len(self.outputs())
        self.window.model_worker.message.emit(response)
        self.assertEqual(len(self.outputs()), count)

    def test_preflight_config_change_prevents_start(self):
        self.prepare()
        self.window._settings["config_hash"] = "wrong-hash"
        self.window._start_calibration()
        self.assertIsNone(self.window.display)
        self.assertEqual(self.window.state, "stopping")

    def test_revoking_consent_aborts_capture(self):
        self.run_capture()
        self.window.risk_check.setChecked(False)
        self.assertEqual(self.window.state, "stopping")
        self.assertEqual(self.outputs()[-1]["reason"], "consent_withdrawn")


    def test_recorded_formal_protocol_matches_actual_display_targets(self):
        from apps.realtime_command_display import TARGETS
        self.prepare()
        protocol = ui.REALTIME_PROTOCOL
        self.assertEqual(protocol['status'], 'formal')
        self.assertEqual([(target['command'], target['frequency_hz']) for target in protocol['targets']],
                         [(target.command, target.frequency_hz) for target in TARGETS])
        events = [json.loads(line) for line in (self.window.session_dir / 'command_events.ndjson').read_text(encoding='utf-8').splitlines()]
        recorded = next(event for event in events if event['type'] == 'realtime_protocol')
        self.assertEqual(recorded['targets'], protocol['targets'])
        self.assertEqual((recorded['window_samples'], recorded['stride_samples']), (1250, 250))

    def test_live_quality_reports_invalid_channel_without_accepting_a_command(self):
        self.run_capture()
        value = batch(0)
        value['quality'].update(status='warning', packet_loss_count=7, packet_duplicate_count=2,
                                invalid_package_count=0, nonfinite_count=1)
        value['samples'][2][20] = None
        self.window.capture_worker.message.emit(value)
        text = self.window.quality_label.text()
        self.assertIn('warning', text)
        self.assertIn('7', text)
        self.assertIn('C3 无效', text)
        self.assertFalse(any(row['accepted'] for row in self.outputs()))

if __name__ == "__main__":
    unittest.main()