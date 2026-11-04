"""Window-owned offline jobs; never load torch in the GUI process."""
from __future__ import annotations
import importlib
import json
import math
from collections import Counter
from pathlib import Path
from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Qt, Signal, QUrl, QRectF
from PySide6.QtGui import QColor, QDesktopServices, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView)
from .components import action, label


class V9OfflineController(QObject):
    busy_changed = Signal(bool)
    updated = Signal(str)
    finished = Signal(str, object)
    OUTPUT_LIMIT = 8192

    def __init__(self, owner, *, backend=None, process_factory=None):
        super().__init__(owner)
        self.owner, self._backend = owner, backend
        self._process_factory = process_factory or QProcess
        self.process = None
        self.busy = self.capture_busy = False
        self.python_override = self.session = self.output_dir = None
        self.states = {}
        self._stdout = self._stderr = b""
        self._discard_line = self._cancelled = self._closed = False
        self._process_error = ""
        self._worker_error = ""
        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self._kill)

    @staticmethod
    def session_key(session_dir):
        return str(Path(session_dir).expanduser().resolve())

    @property
    def backend(self):
        if self._backend is None:
            # Stdlib-only contract. Preserve the UI/core import boundary.
            self._backend = importlib.import_module("eeg_tools.workstation.v9_offline")
        return self._backend

    def eligible(self, session_dir):
        try:
            return self.backend.eligible_native_session(Path(self.session_key(session_dir)))
        except Exception as error:
            return False, str(error)

    def ml_python(self):
        if self.python_override is not None:
            return self.python_override
        try:
            return self.backend.default_ml_python()
        except Exception:
            return None

    def set_capture_busy(self, busy):
        if self.capture_busy != bool(busy):
            self.capture_busy = bool(busy)
            self.updated.emit("")

    def state_for(self, session_dir):
        return self.states.get(self.session_key(session_dir), {"phase": "idle"})

    def restore(self, session_dir):
        key = self.session_key(session_dir)
        cached = self.states.get(key)
        if cached and cached.get("phase") != "complete":
            return
        try:
            output = (Path(cached["output_dir"]) if cached
                      else self.backend.find_latest_result(Path(key)))
            if output is None:
                return
            result = self._read_verified(output, key)
            self.states[key] = {"phase": "complete", "result": result, "output_dir": str(output)}
        except Exception as error:
            self.states[key] = {"phase": "failed", "error": str(error)}
        self.updated.emit(key)

    def _read_verified(self, output, key):
        result = self.backend.read_result(Path(output), session_dir=Path(key))
        if (result.get("status") != "formal" or result.get("completion_status") != "complete"
                or self.session_key(result.get("source_session", "")) != key):
            raise ValueError("v9.invalid_result")
        return result

    def _publish_error(self, key, error):
        self.states[key] = {"phase": "failed", "error": str(error)}
        self.updated.emit(key)
        return False

    def start(self, session_dir, device="cuda:0", python=None):
        key = self.session_key(session_dir)
        query = getattr(self.owner, "_offline_capture_busy", None)
        if self.busy or self._closed:
            return False
        if self.capture_busy or (callable(query) and query()):
            return self._publish_error(key, "v9.capture_busy")
        if device not in {"cuda:0", "cpu"}:
            return self._publish_error(key, "v9.invalid_device")
        try:
            eligible, reason = self.eligible(key)
            if not eligible:
                raise ValueError(reason or "v9.unsupported")
            interpreter = Path(python) if python is not None else self.ml_python()
            if interpreter is None or not interpreter.is_file():
                raise FileNotFoundError("v9.python_missing")
            output = self.backend.new_output_dir(Path(key))
            command, cwd = self.backend.build_command(Path(key), output, device=device, python=interpreter)
            if not command:
                raise ValueError("v9.invalid_command")
            process = self._process_factory(self)
            environment = QProcessEnvironment.systemEnvironment()
            environment.insert("PYTHONIOENCODING", "utf-8")
            environment.insert("PYTHONUTF8", "1")
            process.setProcessEnvironment(environment)
            process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
            process.setWorkingDirectory(str(cwd))
            process.setProgram(str(command[0]))
            process.setArguments([str(arg) for arg in command[1:]])
        except Exception as error:
            return self._publish_error(key, error)
        self.process, self.session, self.output_dir = process, key, Path(output)
        self._stdout = self._stderr = b""
        self._discard_line = self._cancelled = False
        self._process_error = ""
        self._worker_error = ""
        self.states[key] = {"phase": "running", "device": device, "output_dir": str(output)}
        process.readyReadStandardOutput.connect(lambda: self._read_stdout(process))
        process.readyReadStandardError.connect(lambda: self._read_stderr(process))
        process.errorOccurred.connect(lambda error: self._error(process, error))
        process.finished.connect(lambda code, status: self._exited(process, code, status))
        self.busy = True
        self.busy_changed.emit(True)
        self.updated.emit(key)
        try:
            process.start()
        except Exception as error:
            self._finish(process, "failed", str(error))
        return self.busy

    def _read_stdout(self, process):
        if process is not self.process:
            return
        # Oversized JSON lines are discarded through their newline, including
        # trailing fragments. Retain at most one bounded line between signals.
        for fragment in bytes(process.readAllStandardOutput()).splitlines(keepends=True):
            complete = fragment.endswith(b"\n")
            if not self._discard_line:
                if len(self._stdout) + len(fragment) > self.OUTPUT_LIMIT:
                    self._stdout, self._discard_line = b"", True
                else:
                    self._stdout += fragment
            if complete:
                if not self._discard_line:
                    self._progress_line(self._stdout)
                self._stdout, self._discard_line = b"", False

    def _progress_line(self, data):
        try:
            progress = json.loads(data)
        except (ValueError, UnicodeDecodeError):
            return
        if isinstance(progress, dict) and self.session is not None:
            if progress.get("type") == "error":
                message = progress.get("message")
                if isinstance(message, str) and message.strip():
                    self._worker_error = message.strip()[:self.OUTPUT_LIMIT]
            self.states[self.session]["progress"] = progress
            self.updated.emit(self.session)

    def _read_stderr(self, process):
        if process is self.process:
            self._stderr = (self._stderr + bytes(process.readAllStandardError()))[-self.OUTPUT_LIMIT:]

    def _error(self, process, error):
        if process is not self.process:
            return
        self._process_error = process.errorString()
        if error == QProcess.ProcessError.FailedToStart:
            self._finish(process, "cancelled" if self._cancelled else "failed", self._process_error)

    def _exited(self, process, code, status):
        if process is not self.process:
            return
        self._read_stdout(process)
        self._read_stderr(process)
        if self._stdout and not self._discard_line:
            self._progress_line(self._stdout)
        if self._cancelled:
            self._finish(process, "cancelled")
        elif code != 0 or status != QProcess.ExitStatus.NormalExit:
            detail = self._stderr.decode("utf-8", errors="replace").strip()
            if self._worker_error:
                detail = (self._worker_error + ("\n" + detail if detail else ""))[:self.OUTPUT_LIMIT]
            self._finish(process, "failed", (detail or self._process_error or f"Process exit: {code}")[:self.OUTPUT_LIMIT])
        else:
            try:
                result = self._read_verified(self.output_dir, self.session)
            except Exception as error:
                self._finish(process, "failed", str(error))
            else:
                self._finish(process, "complete", result=result)

    def _finish(self, process, phase, error="", result=None):
        if process is not self.process:
            return
        self._kill_timer.stop()
        key, state = self.session, self.states[self.session]
        state["phase"] = phase
        if error:
            state["error"] = error
        if result is not None:
            state["result"] = result
        self.process, self.busy = None, False
        process.deleteLater()
        self.busy_changed.emit(False)
        self.updated.emit(key)
        self.finished.emit(key, dict(state))

    def cancel(self):
        if not self.busy or self.process is None:
            return
        self._cancelled = True
        self.states[self.session]["phase"] = "cancelling"
        self.updated.emit(self.session)
        process = self.process
        process.terminate()
        if process is self.process:
            self._kill_timer.start(700)

    def _kill(self):
        if self.process is not None:
            self.process.kill()

    def close(self):
        self._closed = True
        self._kill_timer.stop()
        process = self.process
        if process is None:
            return True
        self._cancelled = True
        process.kill()
        process.waitForFinished(1500)
        if process.state() != QProcess.ProcessState.NotRunning:
            self._closed = False
            self.states[self.session]["error"] = "v9.close_failed"
            self.updated.emit(self.session)
            return False
        if process is self.process:
            self._finish(process, "cancelled")
        return True


class V9ComparisonWaveform(QWidget):
    """Local means per track; one shared vertical range, no filtering."""
    TRACKS = ("raw", "traditional", "v9")
    COLORS = {"raw": "#64748b", "traditional": "#d97706", "v9": "#0284c7"}

    def __init__(self, tr, parent=None):
        super().__init__(parent)
        self.tr, self.preview, self.channel_index = tr, {}, 0
        self.setMinimumHeight(230)
        self.setAccessibleName(tr("v9.waveform"))

    def set_preview(self, preview):
        self.preview = preview or {}
        self.update()

    def set_channel(self, index):
        self.channel_index = max(0, min(7, int(index)))
        self.update()

    def display_tracks(self):
        tracks, times = {}, self.preview.get("time_s", [])[:750]
        for name in self.TRACKS:
            channels = self.preview.get(name)
            if not channels or self.channel_index >= len(channels):
                continue
            values = channels[self.channel_index][:len(times)]
            if len(values) != len(times) or not values or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
                continue
            mean = math.fsum(values) / len(values)
            tracks[name] = [value - mean for value in values]
        return times, tracks

    def shared_extent(self):
        _, tracks = self.display_tracks()
        return max((abs(v) for values in tracks.values() for v in values), default=0) or 1.0

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.palette().base())
        times, tracks = self.display_tracks()
        if len(times) < 2 or not tracks or times[-1] <= times[0]:
            painter.setPen(self.palette().text().color())
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.tr("v9.no_preview"))
            return
        rect, extent = QRectF(self.rect()).adjusted(66, 32, -16, -30), self.shared_extent()
        painter.setPen(QPen(self.palette().mid().color(), 1))
        painter.drawRect(rect)
        painter.drawLine(rect.left(), rect.center().y(), rect.right(), rect.center().y())
        painter.setPen(self.palette().text().color())
        painter.drawText(4, int(rect.top() + 12), f"+{extent:.3g}")
        painter.drawText(4, int(rect.bottom()), f"-{extent:.3g}")
        painter.drawText(int(rect.left()), self.height() - 8, f"{times[0]:.2f} s")
        painter.drawText(int(rect.right() - 58), self.height() - 8, f"{times[-1]:.2f} s")
        for index, (name, values) in enumerate(tracks.items()):
            painter.setPen(QPen(QColor(self.COLORS[name]), 1.2))
            painter.drawText(int(rect.left() + index * rect.width() / 3), 20, self.tr("v9.track." + name))
            path = QPainterPath()
            for point, (time, value) in enumerate(zip(times, values)):
                x = rect.left() + (time - times[0]) / (times[-1] - times[0]) * rect.width()
                y = rect.center().y() - value / extent * rect.height() / 2
                if point:
                    path.lineTo(x, y)
                else:
                    path.moveTo(x, y)
            painter.save()
            painter.setClipRect(rect)
            painter.drawPath(path)
            painter.restore()


class V9OfflinePanel(QWidget):
    def __init__(self, tr, session_dir, controller=None, *, native=True, parent=None):
        super().__init__(parent)
        self.tr, self.controller, self.native = tr, controller, native
        self.session_dir = Path(session_dir).expanduser().resolve()
        self.setObjectName("v9OfflinePanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(label(tr("v9.title"), "sectionTitle"))
        layout.addWidget(label(tr("v9.notice"), "notice"))
        controls = QHBoxLayout()
        self.device = QComboBox()
        self.device.addItem(tr("v9.gpu"), "cuda:0")
        self.device.addItem(tr("v9.cpu"), "cpu")
        self.device.setAccessibleName(tr("v9.device"))
        self.run_button = action(tr("v9.run"), self._start, True)
        self.cancel_button = action(tr("v9.cancel"), self._cancel)
        self.python_button = action(tr("v9.choose_python"), self._choose_python)
        for widget in (self.device, self.run_button, self.cancel_button, self.python_button):
            controls.addWidget(widget)
        layout.addLayout(controls)
        self.python_label, self.status_label = label(""), label("")
        self.error_label, self.processing_label = label("", "error"), label("")
        self.device_label, self.directory_label = label(""), label("", "path")
        for widget in (self.python_label, self.status_label, self.error_label, self.processing_label,
                       self.device_label, self.directory_label):
            layout.addWidget(widget)
        self.open_button = action(tr("v9.open_output"), self._open_output)
        layout.addWidget(self.open_button)
        self.results = QWidget()
        results_layout = QVBoxLayout(self.results)
        results_layout.setContentsMargins(0, 0, 0, 0)
        channel_row = QHBoxLayout()
        channel_row.addWidget(label(tr("v9.waveform")))
        self.channel = QComboBox()
        self.channel.addItems([f"CH{index}" for index in range(1, 9)])
        self.channel.setAccessibleName(tr("v9.channel"))
        channel_row.addWidget(self.channel)
        channel_row.addStretch()
        results_layout.addLayout(channel_row)
        self.waveform = V9ComparisonWaveform(tr)
        self.channel.currentIndexChanged.connect(self.waveform.set_channel)
        results_layout.addWidget(self.waveform)
        results_layout.addWidget(label(tr("v9.display_note"), "muted"))
        results_layout.addWidget(label(tr("v9.fft_title"), "sectionTitle"))
        results_layout.addWidget(label(tr("v9.fft_note"), "muted"))
        self.fft_summary = label("")
        results_layout.addWidget(self.fft_summary)
        self.trial_table = QTableWidget(0, 6)
        self.trial_table.setHorizontalHeaderLabels([tr("v9.trial"), tr("v9.presented"), tr("v9.gaze"),
            *(tr("v9.track." + name) for name in V9ComparisonWaveform.TRACKS)])
        self.trial_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.trial_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.trial_table.setMinimumHeight(150)
        results_layout.addWidget(self.trial_table)
        self.cca_title = label(tr("v9.cca_title"), "sectionTitle")
        self.cca_note = label(tr("v9.cca_note"), "muted")
        self.cca_summary = label("")
        for widget in (self.cca_title, self.cca_note, self.cca_summary):
            results_layout.addWidget(widget)
        self.cca_trial_table = QTableWidget(0, 7)
        self.cca_trial_table.setAccessibleName(tr("v9.cca_title"))
        self.cca_trial_table.setHorizontalHeaderLabels([
            tr("v9.trial"), tr("v9.presented"), tr("v9.gaze"), tr("v9.cca_window_samples"),
            *(tr("v9.cca_prediction", track=tr("v9.track." + name))
              for name in V9ComparisonWaveform.TRACKS)])
        self.cca_trial_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.cca_trial_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.cca_trial_table.setMinimumHeight(150)
        results_layout.addWidget(self.cca_trial_table)
        layout.addWidget(self.results)
        self._output_dir = self._rendered_result = None
        if controller is not None:
            controller.updated.connect(self._updated)
            controller.busy_changed.connect(self.refresh)
            if native:
                controller.restore(self.session_dir)
        self.refresh()

    def set_capture_busy(self, busy):
        if self.controller is not None:
            self.controller.set_capture_busy(busy)
        self.refresh()

    def _updated(self, session):
        if not session or session == str(self.session_dir):
            self.refresh()

    def _start(self):
        if self.controller is not None and self.native:
            self.controller.start(self.session_dir, self.device.currentData())

    def _cancel(self):
        if self.controller is not None and self.controller.session == str(self.session_dir):
            self.controller.cancel()

    def _choose_python(self):
        path, _ = QFileDialog.getOpenFileName(self, self.tr("v9.choose_python"), "", self.tr("v9.python_filter"))
        if path and self.controller is not None:
            self.controller.python_override = Path(path).expanduser().resolve()
            self.controller.updated.emit("")

    def _open_output(self):
        if self._output_dir is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._output_dir)))

    def refresh(self, *args):
        controller = self.controller
        eligible, reason = (controller.eligible(self.session_dir) if controller and self.native
                            else (False, "v9.unsupported"))
        state = controller.state_for(self.session_dir) if controller else {"phase": "idle"}
        busy = bool(controller and controller.busy)
        own_job = busy and controller.session == str(self.session_dir)
        interpreter = controller.ml_python() if controller else None
        python_ready = interpreter is not None and Path(interpreter).is_file()
        capture_busy = bool(controller and controller.capture_busy)
        self.run_button.setEnabled(eligible and python_ready and not busy and not capture_busy)
        self.cancel_button.setEnabled(own_job and state["phase"] != "cancelling")
        self.device.setEnabled(not busy and not capture_busy)
        self.python_button.setEnabled(controller is not None and not busy and not capture_busy)
        self.python_label.setText(self.tr("v9.python_path", path=str(interpreter)) if python_ready
                                  else self.tr("v9.python_missing"))
        phase = state.get("phase", "idle")
        if own_job:
            status = self.tr("v9." + phase)
            if state.get("progress"):
                stage = str(state["progress"].get("stage", ""))[:100]
                known = {"preflight", "v9", "traditional", "comparison", "personal_decoder", "verification", "complete"}
                status += " / " + (self.tr("v9.stage." + stage) if stage in known else stage)
        elif not eligible:
            status = self.tr("v9.unsupported") + (": " + self.tr(str(reason)) if reason and reason != "v9.unsupported" else "")
        elif capture_busy:
            status = self.tr("v9.capture_busy")
        elif busy:
            status = self.tr("v9.other_job")
        else:
            status = self.tr("v9." + phase)
        self.status_label.setText(status)
        self.error_label.setText(self.tr(state.get("error", "")))
        result = state.get("result") if phase == "complete" and eligible else None
        for widget in (self.results, self.open_button, self.processing_label, self.device_label, self.directory_label):
            widget.setVisible(result is not None)
        self._output_dir = Path(state["output_dir"]) if result is not None else None
        if result is None:
            self._rendered_result = None
            self.waveform.set_preview({})
            return
        if result is not self._rendered_result:
            self._rendered_result = result
            self._show_result(result)
        self.directory_label.setText(self.tr("v9.output_path", path=str(self._output_dir)))

    def _show_result(self, result):
        processing = result.get("processing", {})
        windows = processing.get("window_status_counts")
        if windows is None:
            recorded_windows = processing.get("windows", {})
            windows = (dict(Counter(str(window.get("status", "unknown"))
                                    for window in recorded_windows))
                       if isinstance(recorded_windows, list) else recorded_windows)
        self.processing_label.setText(self.tr("v9.processing", processed=processing.get("processed_samples", 0),
            preserved=processing.get("preserved_samples", 0)) + "\n" + self.tr("v9.windows",
            windows=json.dumps(windows, ensure_ascii=False)))
        gpu = result.get("gpu", {})
        if isinstance(gpu, dict):
            name, identifier = str(gpu.get("device_name") or ""), str(gpu.get("device") or "")
            device = f"{name} ({identifier})" if name and identifier and name != identifier else name or identifier or "—"
        else:
            device = str(gpu)
        self.device_label.setText(self.tr("v9.actual_device", device=device))
        self.waveform.set_preview(result.get("preview"))
        tracks, summaries = result.get("comparison", {}).get("tracks", {}), []
        for name in V9ComparisonWaveform.TRACKS:
            track = tracks.get(name, {})
            if track.get("available") is False or track.get("trial_count", 0) == 0:
                summaries.append(self.tr("v9.track_unavailable", track=self.tr("v9.track." + name),
                    reason=str(track.get("reason") or result.get("classification_unavailable_reason")
                               or self.tr("v9.no_fft_trials"))))
            elif track:
                summaries.append(self.tr("v9.match_summary", track=self.tr("v9.track." + name),
                    matched=track.get("presented_match_count", 0), total=track.get("trial_count", 0),
                    gaze=track.get("gaze_truth_count", 0)))
        self.fft_summary.setText("\n".join(summaries))
        trials = result.get("trials", [])
        self.trial_table.setRowCount(len(trials))
        for row, trial in enumerate(trials):
            gaze = trial.get("gaze_target_id")
            values = [str(trial.get("trial_index", "—")), str(trial.get("presented_target_id", "—")),
                str(gaze) if gaze is not None and gaze != "" else self.tr("v9.unlabeled")]
            for name in V9ComparisonWaveform.TRACKS:
                prediction = (trial.get("predictions", {}).get(name) or {})
                confidence = prediction.get("confidence")
                text = str(prediction.get("target_id") or self.tr("v9.unavailable"))
                if isinstance(confidence, (int, float)) and math.isfinite(confidence):
                    text += f" ({confidence:.3f})"
                values.append(text)
            for column, value in enumerate(values):
                self.trial_table.setItem(row, column, QTableWidgetItem(value))
        self._show_personal_decoder(result.get("personal_decoder"))

    def _show_personal_decoder(self, decoder):
        # This optional diagnostic has its own trials and counts; FFT stays independent.
        self.cca_trial_table.setRowCount(0)
        if not isinstance(decoder, dict):
            self.cca_summary.setText(self.tr("v9.cca_missing"))
            self.cca_trial_table.setVisible(False)
            return
        tracks = decoder.get("tracks") or {}
        summaries = []
        decoder_available = decoder.get("available") is True
        for name in V9ComparisonWaveform.TRACKS:
            track = tracks.get(name) or {}
            track_name = self.tr("v9.track." + name)
            if not decoder_available or track.get("available") is not True:
                reason = (track.get("reason") or decoder.get("unavailable_reason")
                          or self.tr("v9.cca_no_trials"))
                summaries.append(self.tr("v9.track_unavailable", track=track_name,
                                         reason=self.tr(str(reason))))
            else:
                summaries.append(self.tr("v9.cca_summary", track=track_name,
                    matched=track.get("presented_match_count", 0),
                    comparable=track.get("presented_comparable_count", 0),
                    decoded=track.get("trial_count", 0),
                    recorded=track.get("recorded_trial_count", 0)))
        self.cca_summary.setText("\n".join(summaries))
        trials = decoder.get("trials") or []
        self.cca_trial_table.setVisible(bool(trials))
        self.cca_trial_table.setRowCount(len(trials))
        for row, trial in enumerate(trials):
            gaze = trial.get("gaze_target_id")
            values = [str(trial.get("trial_index", "—")),
                str(trial.get("presented_target_id") or self.tr("v9.unlabeled")),
                str(gaze) if gaze is not None and gaze != "" else self.tr("v9.unlabeled"),
                str(trial.get("window_samples", "—"))]
            for column, value in enumerate(values):
                self.cca_trial_table.setItem(row, column, QTableWidgetItem(value))
            for column, name in enumerate(V9ComparisonWaveform.TRACKS, start=4):
                prediction = (trial.get("predictions") or {}).get(name) or {}
                status = (trial.get("prediction_status") or {}).get(name) or {}
                track = tracks.get(name) or {}
                available = (decoder_available and track.get("available") is True
                             and prediction.get("available") is True
                             and status.get("available") is not False)
                target = prediction.get("target_id")
                text = str(target) if available and target else self.tr("v9.unavailable")
                confidence = prediction.get("confidence")
                if (available and target and isinstance(confidence, (int, float))
                        and not isinstance(confidence, bool) and math.isfinite(confidence)):
                    text += f" ({confidence:.3f})"
                item = QTableWidgetItem(text)
                if not available or not target:
                    reason = (status.get("reason") or track.get("reason")
                              or decoder.get("unavailable_reason") or self.tr("v9.cca_no_trials"))
                    item.setToolTip(self.tr(str(reason)))
                self.cca_trial_table.setItem(row, column, item)
