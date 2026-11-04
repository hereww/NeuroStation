"""Nonblocking NDJSON subprocess bridge for the standalone realtime application."""
from __future__ import annotations

import json
import math
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QTimer, Signal


def _finite_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("nonfinite_json_number")
    return value


class JsonWorker(QObject):
    message = Signal(object)
    failed = Signal(str)
    finished = Signal(int)
    MAX_LINE = 512 * 1024
    STOP_TIMEOUT_MS = 6000

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.readyReadStandardError.connect(self._read_errors)
        self.process.errorOccurred.connect(self._process_error)
        self.process.finished.connect(self._finished)
        self._buffer = bytearray()
        self._log = None
        self._closing = False
        self._failed = False
        self._active = False
        self._finishing = False
        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self.process.kill)

    @property
    def running(self):
        return self.process.state() != QProcess.ProcessState.NotRunning

    def start(self, python: Path, script: Path, args: list[str], cwd: Path, log_path: Path):
        # QProcess may already be NotRunning while its final output and signals
        # still need dispatch. Do not replace that run's log or buffer yet.
        if self._active or self.running:
            raise RuntimeError("worker_already_running")
        log = log_path.open("a", encoding="utf-8")
        self._kill_timer.stop()
        self._buffer.clear()
        self._closing = self._failed = self._finishing = False
        self._log = log
        self._active = True
        self.process.setWorkingDirectory(str(cwd))
        environment = self.process.processEnvironment()
        if environment.isEmpty():
            from PySide6.QtCore import QProcessEnvironment
            environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        environment.insert("PYTHONUTF8", "1")
        self.process.setProcessEnvironment(environment)
        self.process.start(str(python), ["-u", str(script), *[str(a) for a in args]])

    def send(self, value: dict) -> bool:
        if not self.running or self._closing or self._failed:
            return False
        try:
            if not isinstance(value, dict):
                raise ValueError("message_must_be_object")
            data = (json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n").encode()
        except (TypeError, ValueError, RecursionError):
            self._fail("invalid_outgoing_json")
            return False
        # MAX_LINE bounds JSON bytes, excluding the terminating newline.
        if len(data) - 1 > self.MAX_LINE:
            self._fail("outgoing_line_too_large")
            return False
        if self.process.bytesToWrite() + len(data) > self.MAX_LINE + 1:
            self._fail("worker_input_backpressure")
            return False
        if self.process.write(data) != len(data):
            self._fail("worker_input_write_failed")
            return False
        return True

    def _close_input(self):
        if self._closing:
            return
        self._closing = True
        if self.running:
            self.process.closeWriteChannel()
            self._kill_timer.start(self.STOP_TIMEOUT_MS)

    def stop(self):
        if not self._active or self._closing:
            return
        if self.running and not self._failed:
            self.send({"type": "stop"})
        self._close_input()

    def _fail(self, reason: str):
        if self._failed or not self._active:
            return
        self._failed = True
        # Closing stdin is a worker shutdown signal in its own right. Avoid
        # stop -> send -> fail -> stop recursion when the input queue is full.
        # Close before emitting so even a reentrant UI slot cannot send again.
        self._close_input()
        self.failed.emit(reason)

    def _read_output(self):
        data = bytes(self.process.readAllStandardOutput())
        if self._failed or not self._active:
            self._buffer.clear()
            return
        self._buffer.extend(data)
        while b"\n" in self._buffer:
            end = self._buffer.index(b"\n")
            line = bytes(self._buffer[:end])
            del self._buffer[:end + 1]
            if line.endswith(b"\r"):
                line = line[:-1]
            if len(line) > self.MAX_LINE:
                self._buffer.clear()
                self._fail("worker_line_too_large")
                return
            if not line.strip():
                continue
            try:
                value = json.loads(line.decode("utf-8"), parse_float=_finite_float, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
                if not isinstance(value, dict):
                    raise ValueError("message_must_be_object")
            except (ValueError, UnicodeError, RecursionError):
                self._buffer.clear()
                self._fail("worker_invalid_json")
                return
            self.message.emit(value)
            if self._failed:
                self._buffer.clear()
                return
        crlf_pending = len(self._buffer) == self.MAX_LINE + 1 and self._buffer.endswith(b"\r")
        if len(self._buffer) > self.MAX_LINE and not crlf_pending:
            self._buffer.clear()
            self._fail("worker_line_too_large")

    def _read_errors(self):
        data = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
        if self._log and data:
            try:
                self._log.write(data)
                self._log.flush()
            except (OSError, ValueError) as error:
                self._fail("worker_stderr_log_failed: " + str(error))

    def _process_error(self, error):
        if not self._active:
            return
        if error == QProcess.ProcessError.FailedToStart:
            self._fail("worker_failed_to_start: " + self.process.errorString())
            self._finished(-1)
        elif not self._closing:
            self._fail("worker_process_error: " + self.process.errorString())

    def _finished(self, code, exit_status=None):
        if not self._active or self._finishing:
            return
        self._finishing = True
        self._kill_timer.stop()
        self._read_output()
        self._read_errors()
        if self._buffer.strip() and not self._failed:
            self._fail("worker_incomplete_json")
        self._buffer.clear()
        # A normal nonzero exit is an error even if no QProcess error occurred.
        # A crash caused by the explicit stop deadline remains an intentional stop.
        if exit_status == QProcess.ExitStatus.CrashExit:
            if not self._closing:
                self._fail("worker_process_error: " + self.process.errorString())
        elif int(code) != 0 and not self._failed:
            self._fail("worker_exit_code: " + str(int(code)))
        if self._log:
            try:
                self._log.close()
            except (OSError, ValueError) as error:
                self._fail("worker_stderr_log_failed: " + str(error))
            self._log = None
        self._active = self._finishing = False
        # Settle all state before emitting: a finished slot may start a new run.
        self.finished.emit(int(code))
