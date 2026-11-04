"""Real Qt QProcess transport tests using temporary, hardware-free child scripts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QProcess, QTimer
from PySide6.QtWidgets import QApplication

from apps.realtime_command_transport import JsonWorker


CHILD = r'''
import json
import os
import sys
import time

mode = sys.argv[1]
out = sys.stdout.buffer

def write(data):
    out.write(data)
    out.flush()

def emit(value):
    write(json.dumps(value, ensure_ascii=False).encode("utf-8") + b"\n")

if mode == "echo":
    emit({"type": "ready"})
    for line in sys.stdin:
        value = json.loads(line)
        if value.get("type") == "stop":
            # stop must send a command AND close stdin, so this read completes.
            eof = sys.stdin.read()
            emit({"type": "stopped", "eof": eof == ""})
            break
        emit({"type": "echo", "value": value})
    else:
        emit({"type": "eof"})
elif mode == "partial":
    write(b'\n  \r\n{"type":"part')
    time.sleep(0.12)
    write(b'ial","value":"\xe8')
    time.sleep(0.08)
    write(b'\x84\x91\xe7\x94\xb5"}\r\n{"type":"tail"}\n')
elif mode == "invalid":
    emit({"type": "before"})
    write(sys.argv[2].encode("utf-8") + b"\n")
    emit({"type": "after_same_read"})
    time.sleep(0.15)
    emit({"type": "after_later_read"})
elif mode == "invalid_utf8":
    write(b'{"type":"\xff"}\n')
elif mode == "incomplete":
    write(b'{"type":"unfinished"}')
elif mode == "incomplete_on_stop":
    emit({"type": "ready"})
    sys.stdin.read()
    write(b'{"type":"unfinished"}')
elif mode == "oversized":
    write(b'{"type":"' + b'x' * int(sys.argv[2]))
    if sys.argv[3] == "newline":
        write(b'"}\n')
    else:
        time.sleep(0.12)
    emit({"type": "after_oversized"})
elif mode == "exact_size":
    value = b'{"type":"' + b'x' * (int(sys.argv[2]) - 11) + b'"}'
    assert len(value) == int(sys.argv[2])
    if len(sys.argv) > 3 and sys.argv[3] == "split_crlf":
        write(value + b'\r')
        time.sleep(0.1)
        write(b'\n')
    else:
        write(value + b'\n')
elif mode == "stderr":
    sys.stderr.buffer.write("日志：正常\n".encode("utf-8") + b'invalid:\xff\n')
    sys.stderr.buffer.flush()
    emit({"type": "ok"})
elif mode == "nonzero":
    emit({"type": "last", "code": 7})
    sys.stderr.write("child failure detail\n")
    sys.exit(7)
elif mode == "crash":
    emit({"type": "last"})
    os._exit(37)
elif mode == "ignore_stop":
    emit({"type": "ready"})
    while True:
        time.sleep(0.1)
elif mode == "stop_during_partial":
    write(b'{"type":"re')
    time.sleep(0.1)
    write(b'ady"}\n')
    sys.stdin.read()
    emit({"type": "eof"})
elif mode == "many":
    for index in range(150):
        emit({"type": "item", "index": index})
elif mode == "deep":
    write(b'{"type":' + b'[' * 20000 + b'0' + b']' * 20000 + b'}\n')
elif mode == "silent":
    pass
else:
    raise RuntimeError("unexpected test mode")
'''


class TransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="qt-ndjson-")
        self.directory = Path(self.temporary.name)
        self.script = self.directory / "child script 脑电.py"
        self.script.write_text(CHILD, encoding="utf-8")
        self.workers = []
        self.heartbeat_count = 0
        self.heartbeat = QTimer()
        self.heartbeat.setInterval(5)
        self.heartbeat.timeout.connect(self._tick)
        self.heartbeat.start()

    def _tick(self):
        self.heartbeat_count += 1

    def wait_until(self, predicate, timeout_ms=3500):
        if predicate():
            return True
        loop = QEventLoop()
        poll = QTimer()
        poll.setInterval(5)
        poll.timeout.connect(lambda: loop.quit() if predicate() else None)
        timeout = QTimer()
        timeout.setSingleShot(True)
        timeout.timeout.connect(loop.quit)
        poll.start()
        timeout.start(timeout_ms)
        loop.exec()
        poll.stop()
        timeout.stop()
        return bool(predicate())

    def pump(self, milliseconds=80):
        loop = QEventLoop()
        QTimer.singleShot(milliseconds, loop.quit)
        loop.exec()

    def tearDown(self):
        for worker, record in self.workers:
            if worker.running:
                worker.process.kill()
                self.wait_until(lambda: not worker.running, 1500)
            self.pump(10)
            if worker._log is not None:
                worker._log.close()
                worker._log = None
            worker.deleteLater()
        self.heartbeat.stop()
        self.app.processEvents()
        self.temporary.cleanup()

    def worker(self, *, limit=None):
        worker = JsonWorker()
        if limit is not None:
            worker.MAX_LINE = limit
        worker.STOP_TIMEOUT_MS = 180
        record = {"messages": [], "failures": [], "finished": [], "order": []}
        worker.message.connect(lambda value: (record["messages"].append(value), record["order"].append(("message", value))))
        worker.failed.connect(lambda value: (record["failures"].append(value), record["order"].append(("failed", value))))
        worker.finished.connect(lambda code: (record["finished"].append(code), record["order"].append(("finished", code))))
        self.workers.append((worker, record))
        return worker, record

    def start(self, worker, mode, *args, python=None):
        worker.start(Path(python or sys.executable), self.script, [mode, *map(str, args)],
                     self.directory, self.directory / "stderr.log")

    def complete(self, worker, record, timeout_ms=3500):
        self.assertTrue(self.wait_until(lambda: bool(record["finished"]), timeout_ms),
                        f"worker did not finish: {record}")
        self.pump(40)
        self.assertFalse(worker.running)
        self.assertFalse(worker._kill_timer.isActive())
        self.assertIsNone(worker._log)
        self.assertEqual(len(record["finished"]), 1, record)

    def ready(self, worker, record, mode="echo"):
        self.start(worker, mode)
        self.assertTrue(self.wait_until(lambda: any(message.get("type") == "ready" for message in record["messages"])), record)

    def test_echo_stop_and_eof_are_real_qprocess_and_nonblocking(self):
        worker, record = self.worker()
        self.assertIsInstance(worker.process, QProcess)
        self.ready(worker, record)
        self.assertTrue(worker.send({"type": "payload", "text": "脑电", "number": 1.25}))
        self.assertTrue(self.wait_until(lambda: len(record["messages"]) == 2))
        self.assertEqual(record["messages"][1], {"type": "echo", "value": {"type": "payload", "text": "脑电", "number": 1.25}})
        worker.stop()
        worker.stop()
        self.assertFalse(worker.send({"type": "too_late"}))
        self.complete(worker, record)
        self.assertEqual(record["messages"][-1], {"type": "stopped", "eof": True})
        self.assertEqual(record["failures"], [])
        self.assertEqual(record["finished"], [0])
        self.assertGreater(self.heartbeat_count, 2)

    def test_partial_stdout_utf8_crlf_and_blank_lines(self):
        worker, record = self.worker()
        self.start(worker, "partial")
        self.complete(worker, record)
        self.assertEqual(record["messages"], [{"type": "partial", "value": "脑电"}, {"type": "tail"}])
        self.assertEqual(record["failures"], [])
        self.assertGreater(self.heartbeat_count, 10)

    def test_many_messages_in_order_and_final_output_drained(self):
        worker, record = self.worker()
        self.start(worker, "many")
        self.complete(worker, record)
        self.assertEqual([message["index"] for message in record["messages"]], list(range(150)))
        self.assertEqual(record["order"][-1], ("finished", 0))
        self.assertEqual(record["failures"], [])

    def test_stderr_is_separate_utf8_log_with_invalid_bytes_replaced(self):
        worker, record = self.worker()
        self.start(worker, "stderr")
        self.complete(worker, record)
        self.assertEqual(record["messages"], [{"type": "ok"}])
        self.assertEqual(record["failures"], [])
        self.assertEqual((self.directory / "stderr.log").read_text(encoding="utf-8"), "日志：正常\ninvalid:�\n")

    def test_nonzero_exit_reports_failure_and_one_finished_after_last_message(self):
        worker, record = self.worker()
        self.start(worker, "nonzero")
        self.complete(worker, record)
        self.assertEqual(record["messages"], [{"type": "last", "code": 7}])
        self.assertEqual(record["failures"], ["worker_exit_code: 7"])
        self.assertEqual(record["finished"], [7])
        self.assertEqual([item[0] for item in record["order"]], ["message", "failed", "finished"])
        self.assertIn("child failure detail", (self.directory / "stderr.log").read_text())

    def test_abrupt_process_exit_reports_failure(self):
        worker, record = self.worker()
        self.start(worker, "crash")
        self.complete(worker, record)
        self.assertEqual(record["messages"], [{"type": "last"}])
        self.assertEqual(len(record["failures"]), 1)
        self.assertNotEqual(record["finished"], [0])

    def test_unexpected_process_kill_reports_crash_and_one_finished(self):
        worker, record = self.worker()
        self.ready(worker, record)
        worker.process.kill()
        self.complete(worker, record)
        self.assertEqual(len(record["failures"]), 1)
        self.assertTrue(record["failures"][0].startswith("worker_process_error:"), record)
        self.assertNotEqual(record["finished"], [0])

    def test_failed_to_start_reports_once_and_can_restart(self):
        worker, record = self.worker()
        self.start(worker, "silent", python=self.directory / "missing-python.exe")
        self.complete(worker, record)
        self.assertEqual(len(record["failures"]), 1)
        self.assertTrue(record["failures"][0].startswith("worker_failed_to_start:"))
        self.assertEqual(record["finished"], [-1])
        for value in record.values():
            value.clear()
        self.start(worker, "silent")
        self.complete(worker, record)
        self.assertEqual(record["failures"], [])
        self.assertEqual(record["finished"], [0])

    def test_invalid_json_stops_and_quarantines_later_stdout(self):
        worker, record = self.worker()
        self.start(worker, "invalid", "broken json")
        self.complete(worker, record)
        self.assertEqual(record["failures"], ["worker_invalid_json"])
        self.assertEqual(record["messages"], [{"type": "before"}])

    def test_json_nonobjects_and_nonfinite_constants_are_rejected(self):
        for value in ('[]', 'null', '3', '"text"', '{"n":NaN}', '{"n":Infinity}', '{"n":-Infinity}', '{"n":1e9999}', '{"n":-1e9999}'):
            with self.subTest(value=value):
                worker, record = self.worker()
                self.start(worker, "invalid", value)
                self.complete(worker, record)
                self.assertEqual(record["failures"], ["worker_invalid_json"])
                self.assertEqual(record["messages"], [{"type": "before"}])

    def test_invalid_utf8_and_deep_json_fail_cleanly(self):
        for mode in ("invalid_utf8", "deep"):
            with self.subTest(mode=mode):
                worker, record = self.worker()
                self.start(worker, mode)
                self.complete(worker, record)
                self.assertEqual(record["failures"], ["worker_invalid_json"])
                self.assertEqual(record["messages"], [])

    def test_incomplete_final_line_is_failure(self):
        worker, record = self.worker()
        self.start(worker, "incomplete")
        self.complete(worker, record)
        self.assertEqual(record["failures"], ["worker_incomplete_json"])
        self.assertEqual(record["messages"], [])

    def test_incomplete_final_line_on_stop_is_still_failure(self):
        worker, record = self.worker()
        self.ready(worker, record, "incomplete_on_stop")
        worker.stop()
        self.complete(worker, record)
        self.assertEqual(record["failures"], ["worker_incomplete_json"])

    def test_oversized_line_with_and_without_newline_is_quarantined(self):
        for ending in ("newline", "partial"):
            with self.subTest(ending=ending):
                worker, record = self.worker(limit=128)
                self.start(worker, "oversized", 129, ending)
                self.complete(worker, record)
                self.assertEqual(record["failures"], ["worker_line_too_large"])
                self.assertEqual(record["messages"], [])
                self.assertLessEqual(len(worker._buffer), worker.MAX_LINE)

    def test_exact_line_limit_is_accepted(self):
        worker, record = self.worker(limit=128)
        self.start(worker, "exact_size", 128)
        self.complete(worker, record)
        self.assertEqual(record["messages"], [{"type": "x" * 117}])
        self.assertEqual(record["failures"], [])

    def test_exact_line_limit_accepts_crlf_split_between_reads(self):
        worker, record = self.worker(limit=128)
        self.start(worker, "exact_size", 128, "split_crlf")
        self.complete(worker, record)
        self.assertEqual(record["messages"], [{"type": "x" * 117}])
        self.assertEqual(record["failures"], [])

    def test_default_line_limit_rejects_actual_oversized_output(self):
        worker, record = self.worker()
        self.assertEqual(worker.MAX_LINE, 512 * 1024)
        self.start(worker, "oversized", worker.MAX_LINE, "newline")
        self.complete(worker, record)
        self.assertEqual(record["failures"], ["worker_line_too_large"])
        self.assertEqual(record["messages"], [])
        self.assertEqual(worker._buffer, bytearray())

    def test_exact_outgoing_line_limit_is_accepted(self):
        worker, record = self.worker(limit=128)
        self.ready(worker, record)
        value = {"type": "x" * 117}
        self.assertEqual(len(json.dumps(value, separators=(",", ":")).encode()), 128)
        self.assertTrue(worker.send(value))
        # The echo wraps the maximum-size request and would exceed our small
        # inbound limit. Restore it before pumping Qt to inspect the reply.
        worker.MAX_LINE = 512 * 1024
        self.assertTrue(self.wait_until(lambda: len(record["messages"]) == 2))
        self.assertEqual(record["messages"][1], {"type": "echo", "value": value})
        worker.stop()
        self.complete(worker, record)
        self.assertEqual(record["failures"], [])

    def test_python_with_missing_script_reports_exit_code_and_stderr(self):
        worker, record = self.worker()
        worker.start(Path(sys.executable), self.directory / "missing-child.py", [],
                     self.directory, self.directory / "stderr.log")
        self.complete(worker, record)
        self.assertEqual(record["finished"], [2])
        self.assertEqual(record["failures"], ["worker_exit_code: 2"])
        self.assertIn("missing-child.py", (self.directory / "stderr.log").read_text())

    def test_error_run_can_restart_without_old_data_or_failure_flags(self):
        worker, record = self.worker()
        self.start(worker, "invalid", "bad")
        self.complete(worker, record)
        self.assertEqual(record["failures"], ["worker_invalid_json"])
        for value in record.values():
            value.clear()
        self.ready(worker, record)
        worker.stop()
        self.complete(worker, record)
        self.assertEqual(record["messages"], [{"type": "ready"}, {"type": "stopped", "eof": True}])
        self.assertEqual(record["failures"], [])

    def test_outgoing_nonobject_is_refused(self):
        worker, record = self.worker()
        self.ready(worker, record)
        self.assertFalse(worker.send([1, 2]))
        self.complete(worker, record)
        self.assertEqual(record["failures"], ["invalid_outgoing_json"])
        self.assertFalse(any(message.get("type") == "echo" for message in record["messages"]))

    def test_outgoing_nonfinite_and_unserializable_fail_without_recursion(self):
        for value in ({"n": float("nan")}, {"value": object()}):
            with self.subTest(value=value):
                worker, record = self.worker()
                self.ready(worker, record)
                self.assertFalse(worker.send(value))
                self.complete(worker, record)
                self.assertEqual(record["failures"], ["invalid_outgoing_json"])

    def test_outgoing_oversized_line_stops_without_recursive_stop_failure(self):
        worker, record = self.worker(limit=128)
        self.ready(worker, record)
        self.assertFalse(worker.send({"type": "payload", "value": "x" * 129}))
        self.complete(worker, record)
        self.assertEqual(record["failures"], ["outgoing_line_too_large"])

    def test_outgoing_deep_json_is_refused(self):
        worker, record = self.worker()
        self.ready(worker, record)
        value = []
        for _ in range(20000):
            value = [value]
        self.assertFalse(worker.send({"value": value}))
        self.complete(worker, record)
        self.assertEqual(record["failures"], ["invalid_outgoing_json"])

    def test_backpressure_accounts_for_next_write_and_closes_without_recursion(self):
        worker, record = self.worker(limit=256)
        self.ready(worker, record, "ignore_stop")
        payload = {"type": "payload", "value": "x" * 110}
        self.assertTrue(worker.send(payload))
        pending = worker.process.bytesToWrite()
        self.assertGreater(pending, 0)
        # No event-loop yield: both writes target the actual Qt pending buffer.
        self.assertFalse(worker.send(payload))
        self.assertLessEqual(worker.process.bytesToWrite(), worker.MAX_LINE + 1)
        self.complete(worker, record, 1500)
        self.assertEqual(record["failures"], ["worker_input_backpressure"])

    def test_uncooperative_child_is_killed_on_short_timer_without_blocking_qt(self):
        worker, record = self.worker()
        self.ready(worker, record, "ignore_stop")
        ticks = self.heartbeat_count
        worker.stop()
        self.complete(worker, record, 1500)
        self.assertGreater(self.heartbeat_count - ticks, 10)
        self.assertEqual(record["failures"], [])

    def test_parent_stdin_close_delivers_child_eof(self):
        worker, record = self.worker()
        self.ready(worker, record)
        worker.process.closeWriteChannel()
        self.complete(worker, record)
        self.assertEqual(record["messages"][-1], {"type": "eof"})
        self.assertEqual(record["failures"], [])

    def test_stop_while_starting_delivers_eof(self):
        worker, record = self.worker()
        self.start(worker, "echo")
        worker.stop()
        self.complete(worker, record)
        self.assertEqual(record["messages"][-1], {"type": "stopped", "eof": True})
        self.assertEqual(record["failures"], [])

    def test_stop_does_not_discard_valid_partial_message(self):
        worker, record = self.worker()
        self.start(worker, "stop_during_partial")
        self.assertTrue(self.wait_until(lambda: bool(worker._buffer)))
        worker.stop()
        self.complete(worker, record)
        self.assertEqual(record["messages"], [{"type": "ready"}, {"type": "eof"}])
        self.assertEqual(record["failures"], [])

    def test_stop_from_message_slot_completes_without_reentrancy_failure(self):
        worker, record = self.worker()
        worker.message.connect(lambda value: worker.stop() if value["type"] == "ready" else None)
        self.start(worker, "echo")
        self.complete(worker, record)
        self.assertEqual(record["messages"][-1], {"type": "stopped", "eof": True})
        self.assertEqual(record["failures"], [])

    def test_send_when_idle_or_finished_returns_false(self):
        worker, record = self.worker()
        self.assertFalse(worker.send({"type": "idle"}))
        worker.stop()
        self.assertEqual(record["finished"], [])
        self.start(worker, "silent")
        self.complete(worker, record)
        self.assertFalse(worker.send({"type": "late"}))
        worker.stop()
        self.pump()
        self.assertEqual(record["finished"], [0])

    def test_cannot_start_twice_while_running(self):
        worker, record = self.worker()
        self.ready(worker, record)
        with self.assertRaisesRegex(RuntimeError, "worker_already_running"):
            self.start(worker, "silent")
        worker.stop()
        self.complete(worker, record)

    def test_restart_is_refused_between_notrunning_and_finished(self):
        worker, record = self.worker()
        attempts = []
        def state_changed(state):
            if state == QProcess.ProcessState.NotRunning:
                try:
                    self.start(worker, "silent")
                except RuntimeError as error:
                    attempts.append(str(error))
                else:
                    attempts.append("incorrectly restarted")
        worker.process.stateChanged.connect(state_changed)
        self.start(worker, "silent")
        self.complete(worker, record)
        self.assertEqual(attempts, ["worker_already_running"])

    def test_stop_with_existing_full_qprocess_queue_closes_without_recursion(self):
        worker, record = self.worker(limit=128)
        self.ready(worker, record, "ignore_stop")
        # Fill the actual Qt queue without yielding. This reproduces the prior
        # stop -> send -> fail -> stop recursion when shutdown itself hits pressure.
        self.assertEqual(worker.process.write(b"x" * 256), 256)
        self.assertGreater(worker.process.bytesToWrite(), worker.MAX_LINE)
        worker.stop()
        self.complete(worker, record, 1500)
        self.assertEqual(record["failures"], ["worker_input_backpressure"])
        self.assertFalse(worker.send({"type": "after_stop"}))

    def test_repeated_finished_callback_is_ignored(self):
        worker, record = self.worker()
        self.start(worker, "silent")
        self.complete(worker, record)
        worker._finished(0, QProcess.ExitStatus.NormalExit)
        self.assertEqual(record["finished"], [0])

    def test_restart_inside_finished_slot_retains_new_run_state(self):
        worker, record = self.worker()
        def restart(_code):
            if len(record["finished"]) == 1:
                self.start(worker, "stderr")
        worker.finished.connect(restart)
        self.start(worker, "silent")
        self.assertTrue(self.wait_until(lambda: len(record["finished"]) == 2), record)
        self.pump()
        self.assertFalse(worker.running)
        self.assertIsNone(worker._log)
        self.assertEqual(record["finished"], [0, 0])
        self.assertEqual(record["messages"], [{"type": "ok"}])
        self.assertEqual(record["failures"], [])
        self.assertIn("日志", (self.directory / "stderr.log").read_text(encoding="utf-8"))

    def test_failed_signal_observes_closed_transport_and_can_request_stop_again(self):
        worker, record = self.worker()
        attempts = []
        worker.failed.connect(lambda _reason: (attempts.append(worker.send({"type": "after_failure"})), worker.stop()))
        self.start(worker, "invalid", "bad")
        self.complete(worker, record)
        self.assertEqual(attempts, [False])
        self.assertEqual(record["failures"], ["worker_invalid_json"])

    def test_log_open_failure_does_not_launch_child(self):
        worker, record = self.worker()
        with self.assertRaises(OSError):
            worker.start(Path(sys.executable), self.script, ["silent"], self.directory,
                         self.directory / "missing" / "stderr.log")
        self.assertFalse(worker.running)
        self.assertEqual(record["finished"], [])
        self.assertIsNone(worker._log)

    def test_closed_stderr_log_fails_and_stops_cleanly(self):
        worker, record = self.worker()
        self.ready(worker, record)
        worker._log.close()
        # The real child rejects malformed input and writes a traceback to stderr.
        worker.process.write(b'not json\n')
        self.complete(worker, record)
        self.assertEqual(len(record["failures"]), 1)
        self.assertTrue(record["failures"][0].startswith("worker_stderr_log_failed:"), record)


if __name__ == "__main__":
    unittest.main()
