"""Standalone, opt-in realtime EEG command application; no workstation coupling."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
import uuid

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QPushButton,
    QScrollArea, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from apps.realtime_command_display import CommandStimulusWindow
from apps.realtime_command_transport import JsonWorker
from eeg_tools.runtime_bindings import bound_file, runtime_config
from eeg_tools.quality.commands import QualityCommandGate
from apps.workstation_ui.quality_panel import QualityPanel
from eeg_tools.realtime_commands import (
    CHANNEL_NAMES, CommandJournal, CommandWindowBuffer, PendingPredictionGate,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "NeuroStationQualityUpgrade" / "realtime"
DEFAULT_MODEL_PYTHON = bound_file("model_python", ROOT / ".venv-ml-gpu" / "Scripts" / "python.exe")
DEFAULT_CAPTURE_PYTHON = bound_file("capture_python", ROOT / ".venv" / "Scripts" / "python.exe")
if runtime_config().get("output_root"):
    DEFAULT_DATA_ROOT = Path(runtime_config()["output_root"]) / "realtime"
FLAGS = {
    "source": "realtime_eeg_model", "status": "formal", "experiment_only": True,
    "personal_validated": False, "runtime_enabled": False,
    "production_validated": False, "default_deployment_authorized": False, "executed": False,
}
COMMANDS = {10: "MOVE_LEFT", 12: "MOVE_RIGHT", 15: "SELECT"}
REALTIME_PROTOCOL = {
    "schema_version": 1, "status": "formal",
    "protocol_id": "realtime_commands_ssvep_10_15_12_v1",
    "targets": [
        {"position": "left", "frequency_hz": 10, "command": "MOVE_LEFT"},
        {"position": "center", "frequency_hz": 15, "command": "SELECT"},
        {"position": "right", "frequency_hz": 12, "command": "MOVE_RIGHT"},
    ],
    "sample_rate_hz": 250, "channel_names": list(CHANNEL_NAMES),
    "window_samples": 1250, "stride_samples": 250, "settle_samples": 250,
}


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _port(value):
    return str(value).strip().upper()


class RealtimeCommandsWindow(QMainWindow):
    """The UI owns all consent, lifecycle, deadline and output decisions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("NeuroStation 实时指令（质量升级支线 · 未合并）")
        self.resize(960, 820)
        self.buffer = CommandWindowBuffer()
        self.gate = PendingPredictionGate(timeout_s=2.0)
        self.quality_gate = QualityCommandGate()
        self.capture_worker = JsonWorker(self)
        self.model_worker = JsonWorker(self)
        self.capture_worker.message.connect(self._capture_message)
        self.model_worker.message.connect(self._model_message)
        self.capture_worker.failed.connect(lambda why: self._fatal("采集进程：" + why))
        self.model_worker.failed.connect(lambda why: self._fatal("模型进程：" + why))
        self.capture_worker.finished.connect(self._capture_finished)
        self.model_worker.finished.connect(self._model_finished)
        self.state = "idle"
        self.session_dir = None
        self.journal = None
        self.display = None
        self.preflight_report = None
        self.model_ready = False
        self.calibrated = False
        self.capture_mode = None
        self._settings = None
        self._inflight = None
        self._quality_waiting_window = None
        self._next_request_id = 0
        self._next_event_id = 0
        self._capture_waiting = self._model_waiting = False
        self._model_started = None
        self._capture_started = None
        self._last_samples = None
        self._silent = False
        self._settle_remaining = 0
        self._command_deadline = None
        self._closing = False
        self._allow_close = False
        self._journal_broken = False
        self._failure_reason = None
        self._stdout_broken = False
        self._udp = None
        self._build_ui()
        self.escape_shortcut = QShortcut(QKeySequence("Esc"), self)
        self.escape_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.escape_shortcut.activated.connect(lambda: self.stop("escape_pressed"))
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        self._refresh_controls()

    def _build_ui(self):
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        self.setCentralWidget(scroll)
        scroll.setWidget(content)
        intro = QLabel("8 通道 · 250 Hz · 每 5 秒连续窗口，每新增 1 秒更新。\n"
                       "刺激布局：左 10 Hz → MOVE_LEFT；中 15 Hz → SELECT；右 12 Hz → MOVE_RIGHT。\n"
                       "原个人测试失败，结果倾向 10 Hz（MOVE_LEFT），实际输出可能偏向一类。\n"
                       "实验模型未通过个人验证；只输出指令，不执行键盘或设备操作。\n"
                       "打开应用不会连接硬件、自动校对或闪烁。")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        settings = QGroupBox("连接与保存")
        form = QFormLayout(settings)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.port_edit = QLineEdit("AUTO")
        self.config_edit = QLineEdit(str(ROOT / "configs" / "channel_config_v1_auto.json"))
        self.data_root_edit = QLineEdit(str(DEFAULT_DATA_ROOT))
        self.model_python_edit = QLineEdit(str(DEFAULT_MODEL_PYTHON))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "cuda"])
        self.screen_combo = QComboBox()
        for index, screen in enumerate(QApplication.screens()):
            geometry = screen.geometry()
            self.screen_combo.addItem(
                f"{index + 1} · {screen.name()} · {geometry.width()}×{geometry.height()} · {screen.refreshRate():.1f} Hz", screen)
        form.addRow("串口（AUTO 或 COM）", self.port_edit)
        form.addRow("通道配置", self._path_row(self.config_edit, "配置"))
        form.addRow("数据根目录", self._path_row(self.data_root_edit, "目录"))
        form.addRow("模型 Python", self._path_row(self.model_python_edit, "Python"))
        form.addRow("模型设备", self.device_combo)
        form.addRow("刺激屏幕", self.screen_combo)
        self.udp_check = QCheckBox("发送到本机 UDP 127.0.0.1:8765（有效期 1500 ms）")
        self.udp_check.setChecked(False)
        form.addRow(self.udp_check)
        layout.addWidget(settings)
        self.settings_group = settings
        self.electrode_check = QCheckBox("我已核对实际电极：Fp1 / Fp2 / C3 / C4 / P7 / P8 / O1 / O2，以及参考和偏置连接")
        self.risk_check = QCheckBox("我已了解闪烁可能引起不适及光敏风险；如不适立即按 Esc 或停止")
        for check in (self.electrode_check, self.risk_check):
            check.setChecked(False)
            check.stateChanged.connect(self._consent_changed)
            layout.addWidget(check)
        actions = QHBoxLayout()
        self.preflight_button = QPushButton("1. 设备预检并加载模型")
        self.calibrate_button = QPushButton("2. 打开全屏并校对")
        self.stop_button = QPushButton("停止")
        self.preflight_button.clicked.connect(self._start_preflight)
        self.calibrate_button.clicked.connect(self._start_calibration)
        self.stop_button.clicked.connect(lambda: self.stop("user_stop"))
        for button in (self.preflight_button, self.calibrate_button, self.stop_button):
            actions.addWidget(button)
        layout.addLayout(actions)
        hint = QLabel("校对通过后，仅在原全屏窗口按 Enter 开始。采集就绪后才启动闪烁。\n"
                      "Esc / Q 或本页停止按钮中止；断点及超时立即显示 UNKNOWN。")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.status_label = QLabel("待机：尚未连接硬件")
        self.status_label.setWordWrap(True)
        self.session_label = QLabel("会话：尚未创建")
        self.session_label.setWordWrap(True)
        self.buffer_label = QLabel("连续窗口：0 / 1250 点")
        layout.addWidget(self.status_label)
        layout.addWidget(self.session_label)
        layout.addWidget(self.buffer_label)
        self.quality_label = QLabel("信号质量：等待采样（不代表电极接触验收）")
        self.quality_label.setWordWrap(True)
        layout.addWidget(self.quality_label)
        self.quality_panel = QualityPanel(self)
        layout.addWidget(self.quality_panel)
        self.command_label = QLabel("UNKNOWN")
        self.command_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.command_label.setStyleSheet("font-size: 42px; font-weight: bold; padding: 16px; color: #8a6a16;")
        layout.addWidget(self.command_label)
        self.probability_label = QLabel("概率：10 Hz — / 12 Hz — / 15 Hz —")
        self.probability_label.setWordWrap(True)
        layout.addWidget(self.probability_label)
        self.reason_label = QLabel("尚无推理结果")
        self.reason_label.setWordWrap(True)
        layout.addWidget(self.reason_label)
        self.recent_table = QTableWidget(0, 5)
        self.recent_table.setHorizontalHeaderLabels(["时间", "指令", "原因", "请求", "代次"])
        self.recent_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.recent_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.recent_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.recent_table.setMinimumHeight(220)
        layout.addWidget(self.recent_table)
        layout.addStretch()

    def _path_row(self, edit, kind):
        row = QWidget()
        box = QHBoxLayout(row)
        box.setContentsMargins(0, 0, 0, 0)
        button = QPushButton("选择")
        def browse():
            if kind == "目录":
                value = QFileDialog.getExistingDirectory(self, "选择数据根目录", edit.text())
            else:
                value, _ = QFileDialog.getOpenFileName(self, "选择" + kind, edit.text())
            if value:
                edit.setText(value)
        button.clicked.connect(browse)
        box.addWidget(edit, 1)
        box.addWidget(button)
        return row

    def _refresh_controls(self):
        idle = self.state == "idle" and not self._closing
        self.settings_group.setEnabled(idle)
        self.preflight_button.setEnabled(idle and self._consented())
        self.calibrate_button.setEnabled(self.state == "prepared" and self._consented()
                                         and self.model_ready and not self._closing
                                         and (self.display is None or self.display.mode != "calibration"))
        self.stop_button.setEnabled(self.state != "idle")

    def _consented(self):
        return self.electrode_check.isChecked() and self.risk_check.isChecked()

    def _consent_changed(self, *_):
        if not self._consented() and self.state not in ("idle", "stopping"):
            self.stop("consent_withdrawn")
        self._refresh_controls()

    def _event(self, event, **fields):
        self._append("event", {"type": event, "timestamp_s": time.time(),
                               "monotonic_s": time.monotonic(), **fields})

    def _append(self, kind, record):
        if self.journal is None or self._journal_broken:
            return
        try:
            self.journal.append(kind, record)
        except Exception as exc:
            self._journal_broken = True
            self._fatal("journal_write_failed: " + str(exc))

    def _fatal(self, reason):
        # Preserve the original failure while both workers finish cleanup.
        if self._failure_reason is None:
            self._failure_reason = str(reason)
        self.status_label.setText("已中止：" + self._failure_reason)
        self.stop(self._failure_reason)

    def _configuration(self):
        config = Path(self.config_edit.text().strip()).expanduser().resolve()
        python = Path(self.model_python_edit.text().strip()).expanduser().resolve()
        capture_python = DEFAULT_CAPTURE_PYTHON if DEFAULT_CAPTURE_PYTHON.is_file() else Path(sys.executable)
        if not config.is_file() or not python.is_file():
            raise ValueError("通道配置或模型 Python 文件不存在")
        if self.screen_combo.currentData() is None:
            raise ValueError("请选择有效刺激屏幕")
        return {"port": self.port_edit.text().strip() or "AUTO", "config": config,
                "config_hash": hashlib.sha256(config.read_bytes()).hexdigest(),
                "python": python, "capture_python": capture_python,
                "device": self.device_combo.currentText(),
                "screen": self.screen_combo.currentData(), "udp": self.udp_check.isChecked()}

    def _start_preflight(self):
        if self.state != "idle" or self._closing or not self._consented():
            return
        try:
            settings = self._configuration()
            root = Path(self.data_root_edit.text().strip()).expanduser().resolve()
            if not self.data_root_edit.text().strip():
                raise ValueError("请填写数据根目录")
            root.mkdir(parents=True, exist_ok=True)
            session = root / (datetime.now().strftime("%Y%m%d_%H%M%S_%f_") + uuid.uuid4().hex[:10])
            session.mkdir(exist_ok=False)
            journal = CommandJournal(session)
        except Exception as exc:
            self.status_label.setText("无法开始预检：" + str(exc))
            return
        self._settings, self.session_dir, self.journal = settings, session, journal
        self._journal_broken = False
        self._failure_reason = None
        self.preflight_report = None
        self.model_ready = self.calibrated = False
        self._inflight = None
        self._quality_waiting_window = None
        self._next_event_id = 0
        self._settle_remaining = 0
        self.quality_gate.reset_session()
        self.buffer.clear("new_session")
        self.gate.invalidate("new_session")
        self.capture_mode = "preflight"
        self._model_started = time.monotonic()
        self._last_samples = self._capture_started = None
        self._silent = False
        self.state = "preflight"
        self.session_label.setText("会话：" + str(session))
        self.quality_label.setText("信号质量：等待采样（不代表电极接触验收）")
        self.status_label.setText("正在预检设备并加载模型；尚未开始连续采集或闪烁")
        self._refresh_controls()
        self._event("session_started", port=settings["port"], channel_config=str(settings["config"]),
                    channel_config_sha256=settings["config_hash"], device=settings["device"],
                    model_python=str(settings["python"]), udp_enabled=settings["udp"],
                    electrode_confirmed=True, photosensitivity_consent=True,
                    protocol_id=REALTIME_PROTOCOL["protocol_id"], protocol_status="formal", **FLAGS)
        self._event("realtime_protocol", **{**FLAGS, **REALTIME_PROTOCOL})
        if self.state != "preflight":
            return
        try:
            self._model_waiting = True
            self.model_worker.start(settings["python"], ROOT / "scripts" / "run_realtime_command_inference.py",
                                    ["--device", settings["device"]], ROOT, session / "model.stderr.log")
            if self.state == "preflight":
                self._start_capture_worker("preflight", settings["port"])
        except Exception as exc:
            if not self.model_worker.running:
                self._model_waiting = False
            self._fatal("worker_start_failed: " + str(exc))

    def _start_capture_worker(self, mode, port):
        settings = self._settings
        self.capture_mode = mode
        self._capture_waiting = True
        try:
            self.capture_worker.start(
                settings["capture_python"], ROOT / "scripts" / "run_realtime_capture.py",
                ["--mode", mode, "--port", port, "--channel-config", str(settings["config"]),
                 "--session-dir", str(self.session_dir)], ROOT, self.session_dir / "capture.stderr.log")
        except Exception:
            self._capture_waiting = False
            raise

    def _valid_preflight(self):
        report = self.preflight_report
        if not isinstance(report, dict) or self._settings is None:
            return False
        completed = report.get("completed_timestamp_s")
        try:
            inner = report.get("report", {})
            if not isinstance(inner, dict):
                return False
            checks = inner.get("checks", [])
            if not isinstance(checks, list):
                return False
            if inner and inner.get("status") not in ("passed", "warning"):
                return False
            current_hash = hashlib.sha256(self._settings["config"].read_bytes()).hexdigest()
            selected = _port(report.get("port", ""))
            requested = _port(self._settings["port"])
            return (report.get("status") in ("passed", "warning")
                    and not report.get("cancelled", False)
                    and _finite(completed) and 0 <= time.time() - completed <= 300
                    and selected not in ("", "AUTO")
                    and (requested == "AUTO" or requested == selected)
                    and report.get("channel_config_sha256") == current_hash == self._settings["config_hash"]
                    and all(isinstance(check, dict) and check.get("status") not in
                            ("failed", "degraded", "cancelled") for check in checks))
        except (OSError, TypeError, AttributeError):
            return False

    def _maybe_prepared(self):
        if self.state != "preflight" or self.capture_worker.running or self._capture_waiting:
            return
        if self.preflight_report is not None and not self._valid_preflight():
            self._fatal("preflight_invalid_or_expired")
        elif self.model_ready and self._valid_preflight():
            self.state = "prepared"
            warning = self.preflight_report.get("status") == "warning"
            self.status_label.setText("预检与模型已就绪" + ("（设备报告警告，请查看会话记录）" if warning else "")
                                      + "；请手动打开全屏校对")
            self._refresh_controls()

    def _start_calibration(self):
        if self.state != "prepared" or not self._consented() or not self.model_ready:
            return
        if not self._valid_preflight():
            self._fatal("preflight_invalid_or_expired")
            return
        try:
            if self.display is None:
                self.display = CommandStimulusWindow(self._settings["screen"])
                self.display.calibration_finished.connect(self._calibration_finished)
                self.display.timing_failed.connect(lambda why: self._fatal("display_timing_failed: " + why))
                self.display.aborted.connect(lambda: self.stop("display_aborted"))
                self.display.frame_record.connect(lambda frame: self._append("frame", frame))
                self.display.start_requested.connect(self._request_start)
            self.calibrated = False
            started = self.display.start_calibration()
            self._event("calibration_started", started=bool(started))
            if not started and self.state != "stopping":
                self._fatal("calibration_start_failed")
        except Exception as exc:
            self._fatal("display_failed: " + str(exc))
        self._refresh_controls()

    def _calibration_finished(self, report):
        self._event("calibration_finished", report=report)
        if self.state == "stopping":
            return
        self.calibrated = isinstance(report, dict) and report.get("passed") is True
        if not self.calibrated:
            self._fatal("calibration_failed")
            return
        self.status_label.setText("屏幕校对通过；请在原全屏窗口按 Enter 开始采集与刺激")
        self._refresh_controls()

    def _request_start(self):
        if self.state != "prepared":
            return
        if (not self._consented() or not self.model_ready or not self.model_worker.running
                or not self.calibrated or self.display is None or self.capture_worker.running
                or not getattr(self.display, "calibration_valid", self.calibrated)
                or not self._valid_preflight()):
            self._fatal("start_requirements_failed")
            return
        report = self.display.calibration_report
        if not isinstance(report, dict) or report.get("passed") is not True:
            self._fatal("calibration_missing")
            return
        self.state = "capture_starting"
        self._capture_started = time.monotonic()
        self.buffer.clear("capture_starting")
        self.gate.invalidate("capture_starting")
        self._event("capture_requested", port=self.preflight_report["port"], calibration=report)
        self.status_label.setText("正在启动连续采集；采集就绪后开始闪烁")
        self._refresh_controls()
        try:
            if self.state == "capture_starting":
                self._start_capture_worker("capture", self.preflight_report["port"])
        except Exception as exc:
            self._fatal("capture_start_failed: " + str(exc))

    def _capture_message(self, message):
        if not isinstance(message, dict):
            self._fatal("capture_invalid_message")
            return
        kind = message.get("type")
        if kind == "quality_update":
            self._quality_update(message.get("quality_v2"))
            return
        if kind == "samples":
            self._samples(message)
            return
        self._event("capture_message", message=message)
        if self.state in ("idle", "stopping"):
            return
        if kind == "preflight":
            if self.state != "preflight" or self.capture_mode != "preflight":
                self._fatal("unexpected_preflight")
                return
            self.preflight_report = message.get("report")
            if not self._valid_preflight():
                self._fatal("preflight_failed_or_cancelled")
            else:
                self._maybe_prepared()
        elif kind == "capture_ready":
            if self.state != "capture_starting" or not self._capture_ready_matches(message):
                self._fatal("capture_ready_mismatch")
                return
            self.state = "running"
            self._last_samples = time.monotonic()
            self._silent = False
            try:
                started = self.display.start_flicker()
            except Exception as exc:
                self._fatal("display_start_failed: " + str(exc))
                return
            if not started:
                self._fatal("display_start_failed")
                return
            if self.state == "running":
                self.buffer.clear("stimulus_started")
                self.gate.invalidate("stimulus_started")
                self._settle_remaining = 250
                self._event("stimulus_settling", settle_samples=250, sample_rate_hz=250)
                self.status_label.setText("正在采集与刺激；先丢弃 250 点启动过渡，再累计 1250 点连续 EEG")
                self._refresh_controls()
        elif kind == "error":
            code = str(message.get("code") or "unknown")
            detail = str(message.get("message") or "未提供详细原因")
            stage = str(message.get("stage") or "")
            port = str(message.get("port") or "")
            context = " / ".join(value for value in (stage, port) if value)
            reason = "capture_error: " + code
            if context:
                reason += " [" + context + "]"
            self._fatal(reason + ": " + detail)
        elif kind == "warning":
            if message.get("code") == "no_data" and self.state == "running":
                self._discontinuity("no_samples", clear=True)
                self._silent = True
        elif kind == "capture_stopped" and self.state in ("running", "capture_starting"):
            self.stop("capture_stopped")

    def _capture_ready_matches(self, message):
        try:
            return (self._valid_preflight() and self._consented() and self.model_ready
                    and self.calibrated and self.display is not None
                    and getattr(self.display, "calibration_valid", self.calibrated)
                    and message.get("status") == "formal"
                    and message.get("recording_status") == "recording"
                    and message.get("sample_rate_hz") == 250
                    and type(message.get("sample_rate_hz")) is int
                    and message.get("channel_names") == list(CHANNEL_NAMES)
                    and message.get("channel_config_sha256") == self._settings["config_hash"]
                    and _port(message.get("port", "")) == _port(self.preflight_report["port"])
                    and Path(message.get("session_dir", "")).resolve() == self.session_dir)
        except (ValueError, TypeError, OSError):
            return False

    def _update_quality(self, message):
        """Show acquisition counters and batch RMS without inferring contact quality."""
        quality = message.get("quality")
        quality = quality if isinstance(quality, dict) else {}
        def counter(key):
            value = quality.get(key)
            return str(value) if type(value) is int and value >= 0 else "—"
        samples = message.get("samples")
        rms = []
        for index, name in enumerate(CHANNEL_NAMES):
            value = None
            try:
                if isinstance(samples, list) and len(samples) == 8:
                    row = samples[index]
                    if isinstance(row, list) and 1 <= len(row) <= 250 and all(_finite(point) for point in row):
                        value = math.hypot(*row) / math.sqrt(len(row))
            except (TypeError, ValueError, OverflowError):
                pass
            magnitude = f"{value:.3g}" if value is not None and math.isfinite(value) else "无效"
            rms.append(f"{name} {magnitude}")
        status = quality.get("status", "未知")
        self.quality_label.setText(
            f"信号质量报告：{status} · 累计缺包 {counter('packet_loss_count')} / "
            f"重复 {counter('packet_duplicate_count')} / 无效包 {counter('invalid_package_count')} / "
            f"非有限值 {counter('nonfinite_count')}\n"
            "RMS（本批原始幅值）：" + " / ".join(rms)
            + "\n仅展示信号质量，不代表电极接触验收。")

    def _quality_update(self, quality):
        if self.state != "running":return
        if self.quality_gate.observe(quality):
            self._discontinuity("processing_or_stream_changed", clear=True)
        self.quality_panel.update_quality(quality)
        waiting=self._quality_waiting_window
        if waiting is not None:
            self._attempt_window(*waiting, log_rejection=False)

    def _samples(self, message):
        if self.state != "running":
            return
        self._update_quality(message)
        quality = message.get("quality_v2")
        if quality is not None or message.get("quality_delivery") != "async":
            if self.quality_gate.observe(quality):
                self._discontinuity("processing_or_stream_changed", clear=True)
            self.quality_panel.update_quality(quality)
        metadata = {key: value for key, value in message.items() if key != "samples"}
        self._event("sample_batch", metadata=metadata)
        if self.state != "running":
            return
        samples = message.get("samples")
        if (not isinstance(samples, list) or len(samples) != 8
                or not isinstance(samples[0], list) or not 1 <= len(samples[0]) <= 250):
            self._discontinuity("invalid_capture_batch", clear=True)
            return
        now = time.monotonic()
        received = message.get("received_monotonic_s")
        if not _finite(received) or now - received > 1.0 or received - now > .1:
            self._discontinuity("stale_capture_batch", clear=True)
            return
        if self._last_samples is not None and now - self._last_samples >= 1.0 and not self._silent:
            self._discontinuity("no_samples", clear=True)
        self._last_samples, self._silent = now, False
        if self._settle_remaining:
            dropped = min(self._settle_remaining, len(samples[0]))
            self._settle_remaining -= dropped
            self._event("settle_batch_discarded", discarded_samples=dropped,
                        remaining_samples=self._settle_remaining)
            if dropped == len(samples[0]):
                self.buffer_label.setText(f"启动过渡：还需丢弃 {self._settle_remaining} 点；连续窗口 0 / 1250 点")
                return
            try:
                message = self._slice_batch(message, dropped)
            except (TypeError, ValueError, IndexError):
                self._discontinuity("invalid_settle_batch", clear=True)
                return
        generation = self.buffer.generation
        window = self.buffer.add_batch(message)
        if self.buffer.generation != generation:
            self._discontinuity(self.buffer.last_reset_reason, clear=False)
        self.buffer_label.setText(f"连续窗口：{self.buffer.sample_count} / 1250 点 · 代次 {self.buffer.generation}")
        if window is None or self.state != "running" or not self.model_ready:
            return
        self._attempt_window(window, now)

    def _attempt_window(self, window, ready_at, *, log_rejection=True):
        now=time.monotonic()
        self._quality_waiting_window=None
        if self.state != "running" or not self.model_ready or self._last_samples is None:return
        if now-ready_at>2 or now-self._last_samples>=1:
            self._publish_unknown("quality_window_wait_expired")
            return
        if self._inflight is not None or self.gate.pending is not None:
            self._event("window_skipped", reason="model_inflight", sample_end=window["sample_end"])
            return
        quality_decision = self.quality_gate.assess(window)
        if not quality_decision["accepted"]:
            if quality_decision["reason"] in ("quality_unavailable","quality_coverage_gap","quality_coverage_incomplete"):
                self._quality_waiting_window=(window,ready_at)
            if log_rejection:
                self._publish_unknown(quality_decision["reason"], request={"window_metadata": {k:v for k,v in window.items() if k != "samples"}, "quality_v2": quality_decision})
            return
        ids = self.gate.begin(self._next_request_id)
        if ids is None:
            return
        self._next_request_id += 1
        # The ONLY fields transmitted to the inference worker are this whitelist.
        request = {"type": "window", **ids, "sample_rate_hz": window["sample_rate_hz"],
                   "channel_names": window["channel_names"], "samples": window["samples"]}
        self._inflight = {**ids, "started_at": now, "quality_v2": quality_decision,
                          "window_metadata": {key: value for key, value in window.items() if key != "samples"}}
        self._event("inference_requested", **ids, buffer_generation=self.buffer.generation,
                    window_metadata={key: value for key, value in window.items() if key != "samples"})
        if self.state != "running":
            return
        if not self.model_worker.send(request):
            self._fatal("model_send_failed")

    @staticmethod
    def _slice_batch(message, offset):
        """Slice actual points and every per-point identity together after settling."""
        count = len(message["samples"][0])
        result = dict(message)
        if any(not isinstance(row, list) or len(row) != count for row in message["samples"]):
            raise ValueError("invalid_samples")
        result["samples"] = [row[offset:] for row in message["samples"]]
        for field in ("sample_indices", "package_numbers", "timestamps_s", "timestamps"):
            if field in result:
                if not isinstance(result[field], list) or len(result[field]) != count:
                    raise ValueError("invalid_per_sample_metadata")
                result[field] = result[field][offset:]
        indices = result.get("sample_indices")
        if indices:
            if "sample_start" in result:
                result["sample_start"] = indices[0]
            if "sample_end" in result:
                result["sample_end"] = indices[-1]
        elif "sample_start" in result:
            result["sample_start"] += offset
        if "sample_count" in result:
            result["sample_count"] = count - offset
        return result

    def _discontinuity(self, reason, *, clear):
        self._quality_waiting_window = None
        if clear:
            self.buffer.clear(reason)
        self.gate.invalidate(reason)
        self._event("window_invalidated", reason=reason, buffer_generation=self.buffer.generation,
                    pending_generation=self.gate.generation)
        self._publish_unknown(reason)
        self.buffer_label.setText(f"连续窗口：{self.buffer.sample_count} / 1250 点 · 代次 {self.buffer.generation}")

    def _model_message(self, message):
        if not isinstance(message, dict):
            self._fatal("model_invalid_message")
            return
        self._event("model_message", message=message)
        if self.state in ("idle", "stopping"):
            return
        kind = message.get("type")
        if kind == "ready":
            if self.model_ready or self.state != "preflight" or not self._ready_matches(message):
                self._fatal("model_ready_mismatch")
                return
            self.model_ready = True
            self._maybe_prepared()
        elif kind == "prediction":
            inflight = self._inflight
            if (inflight is None or any(type(message.get(key)) is not int or
                                       message.get(key) != inflight[key] for key in ("request_id", "generation"))):
                self._event("prediction_discarded", reason="unmatched_physical_request", response=message)
                return
            self._inflight = None
            audit = self.gate.resolve(message)
            if not audit["accepted"]:
                self._event("prediction_discarded", **audit)
                if audit["reason"] == "prediction_timeout" and self.state == "running":
                    self._publish_unknown("prediction_timeout", message, request=inflight)
                return
            if self.state == "running" and (self._last_samples is None or
                                             time.monotonic() - self._last_samples >= 1.0):
                self._silent = True
                self._discontinuity("no_samples", clear=True)
            if self.state != "running" or self._silent:
                self._event("prediction_discarded", reason="capture_not_active", response=message)
                return
            # Resolve alone only associates IDs; validate the actual command separately.
            accepted = self._accepted_prediction(message)
            reason = "accepted" if accepted else str(message.get("reason", "invalid_model_prediction"))
            if not accepted and reason == "accepted":
                reason = "invalid_model_prediction"
            quality_decision = self.quality_gate.assess(inflight.get("window_metadata", {}),
                processing_id=inflight.get("quality_v2", {}).get("processing_id"))
            if not quality_decision["accepted"]:
                accepted, reason = False, quality_decision["reason"]
            command = message["prediction"]["command"] if accepted else "UNKNOWN"
            if accepted:
                command, reason = self.quality_gate.latch(command)
            self._publish(command, reason, message, request=inflight)
        elif kind == "error":
            self._fatal("model_error: " + str(message.get("error", "unknown")))
        elif kind == "stopped":
            self._fatal("model_stopped_unexpectedly")
        else:
            self._fatal("unexpected_model_message")

    @staticmethod
    def _ready_matches(message):
        gate = message.get("command_gate", {})
        ood = message.get("ood_gate", {})
        return (all(type(message.get(key)) is type(value) and message.get(key) == value for key, value in FLAGS.items())
                and message.get("model_loaded") is True
                and message.get("channel_names") == list(CHANNEL_NAMES)
                and message.get("sample_rate_hz") == 250 and message.get("sample_count") == 1250
                and message.get("training_sample_count") == 1250 and message.get("input_features") == 72
                and isinstance(gate, dict) and gate.get("min_probability") == .65
                and gate.get("min_probability_margin") == .15
                and isinstance(ood, dict) and ood.get("maxabs_limit") == 8
                and ood.get("p95abs_limit") == 4 and ood.get("heuristic") is True
                and ood.get("validated") is False)

    @staticmethod
    def _accepted_prediction(message):
        if not all(type(message.get(key)) is type(value) and message.get(key) == value for key, value in FLAGS.items()):
            return False
        prediction, raw, ood = (message.get(key) for key in ("prediction", "raw_prediction", "ood"))
        probabilities = message.get("probabilities")
        if (message.get("model_loaded") is not True or message.get("window_matches_training") is not True
                or message.get("sample_count") != 1250 or message.get("training_sample_count") != 1250
                or not isinstance(prediction, dict) or prediction.get("accepted") is not True
                or prediction.get("executed") is not False or not isinstance(raw, dict)
                or not isinstance(ood, dict) or not isinstance(probabilities, list) or len(probabilities) != 3
                or not all(_finite(value) and 0 <= value <= 1 for value in probabilities)
                or abs(sum(probabilities) - 1) > 1e-6):
            return False
        order = sorted(range(3), key=lambda i: -probabilities[i])
        frequency = (10, 12, 15)[order[0]]
        return (message.get("reason") == "accepted" and raw.get("accepted") is True
                and raw.get("executed") is False and raw.get("predicted_frequency_hz") == frequency
                and raw.get("command") == prediction.get("command") == COMMANDS[frequency]
                and raw.get("probabilities") == probabilities and probabilities[order[0]] >= .65
                and probabilities[order[0]] - probabilities[order[1]] >= .15
                and ood.get("passed") is True and ood.get("heuristic") is True
                and ood.get("validated") is False and ood.get("maxabs_limit") == 8
                and ood.get("p95abs_limit") == 4 and _finite(ood.get("maxabs"))
                and _finite(ood.get("p95abs")) and 0 <= ood["maxabs"] <= 8 and 0 <= ood["p95abs"] <= 4)

    def _publish_unknown(self, reason, raw=None, request=None):
        self._publish("UNKNOWN", reason, raw, request=request)

    def _publish(self, command, reason, raw=None, *, request=None):
        if command != "UNKNOWN" and self.state != "running":
            return
        now, wall = time.monotonic(), time.time()
        event_id = self._next_event_id
        self._next_event_id += 1
        accepted = command in COMMANDS.values()
        request = request or {}
        raw = raw if isinstance(raw, dict) else None
        metadata = request.get("window_metadata", {})
        record = {
            **FLAGS, "type": "command", "command": command, "accepted": accepted,
            "quality_v2": request.get("quality_v2"),
            "session_id": self.session_dir.name if self.session_dir else None, "event_id": event_id,
            "inference_latency_ms": max(0., (now - request["started_at"]) * 1000)
                if _finite(request.get("started_at")) else None,
            "prediction": {"command": command, "accepted": accepted, "executed": False},
            "reason": str(reason), "timestamp_s": wall, "monotonic_s": now,
            "request_id": raw.get("request_id") if raw else request.get("request_id"),
            "generation": raw.get("generation") if raw else request.get("generation", self.gate.generation),
            "started_at": request.get("started_at"),
            "request_started_monotonic_s": request.get("started_at"),
            "sample_start": metadata.get("sample_start"), "sample_end": metadata.get("sample_end"),
            "window_metadata": metadata, "raw_prediction": raw,
            "probabilities": raw.get("probabilities") if raw else None,
            "ood": raw.get("ood") if raw else None,
            "valid_for_ms": 1500 if accepted else 0,
            "valid_until_timestamp_s": wall + (1.5 if accepted else 0),
            "expires_at_ms": int(wall * 1000) + (1500 if accepted else 0),
            "session_dir": str(self.session_dir) if self.session_dir else None,
        }
        self._command_deadline = now + 1.5 if accepted else None
        self.command_label.setText(command)
        self.command_label.setStyleSheet("font-size: 42px; font-weight: bold; padding: 16px; color: "
                                        + ("#18794e;" if accepted else "#8a6a16;"))
        self.reason_label.setText("原因：" + str(reason))
        probabilities = record["probabilities"]
        if (isinstance(probabilities, list) and len(probabilities) == 3
                and all(_finite(value) for value in probabilities)):
            self.probability_label.setText("概率：" + " / ".join(
                f"{hz} Hz {probability:.1%}" for hz, probability in zip((10, 12, 15), probabilities)))
        else:
            self.probability_label.setText("概率：10 Hz — / 12 Hz — / 15 Hz —")
        if self.display is not None:
            self.display.set_status_text(f"{command} · {reason}；Esc / Q 停止")
        self.recent_table.insertRow(0)
        for column, value in enumerate((datetime.now().strftime("%H:%M:%S"), command, reason,
                                        record["request_id"], record["generation"])):
            self.recent_table.setItem(0, column, QTableWidgetItem("—" if value is None else str(value)))
        if self.recent_table.rowCount() > 100:
            self.recent_table.removeRow(100)
        self._append("command", record)
        if accepted and self.state != "running":
            return
        data = (json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
        if not self._stdout_broken:
            try:
                sys.stdout.write(data.decode("utf-8"))
                sys.stdout.flush()
            except (OSError, UnicodeError):
                self._stdout_broken = True
                self._fatal("command_stdout_failed")
                return
        if self._settings and self._settings["udp"]:
            # Keep datagrams small; complete raw evidence remains in commands.ndjson.
            udp_record = {key: value for key, value in record.items()
                          if key not in ("window_metadata", "raw_prediction", "ood", "session_dir")}
            try:
                if self._udp is None:
                    self._udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    self._udp.setblocking(False)
                self._udp.sendto(json.dumps(udp_record, ensure_ascii=True, allow_nan=False,
                                            separators=(",", ":")).encode(), ("127.0.0.1", 8765))
            except OSError as exc:
                self._event("udp_send_failed", reason=str(exc))
                self._fatal("udp_send_failed")

    def _tick(self):
        now = time.monotonic()
        if self.state == "stopping":
            self._finish_shutdown()
            return
        if self.state == "idle":
            return
        if not self.model_ready and self._model_started is not None and now - self._model_started >= 180:
            self._fatal("model_ready_timeout_180s")
            return
        if self._inflight is not None and now - self._inflight["started_at"] >= 10:
            self._fatal("model_response_timeout_10s")
            return
        pending = self._inflight
        if self.gate.expire() and self.state == "running":
            self._publish_unknown("prediction_timeout", request=pending)
            self._event("prediction_timeout", physical_inflight=pending)
        if self.state == "capture_starting" and self._capture_started is not None and now - self._capture_started >= 15:
            self._fatal("capture_ready_timeout")
            return
        if self.state == "running":
            if self.display is None or self.display.mode != "flicker":
                self._fatal("display_not_flickering")
                return
            if self._last_samples is None or now - self._last_samples >= 1.0:
                if not self._silent:
                    self._silent = True
                    self._discontinuity("no_samples", clear=True)
            if self._command_deadline is not None and now >= self._command_deadline:
                self._publish_unknown("command_expired")
        if self.state in ("preflight", "prepared") and self.preflight_report is not None:
            if not self._valid_preflight():
                self._fatal("preflight_invalid_or_expired")

    def _capture_finished(self, code):
        self._capture_waiting = False
        self._event("capture_finished", exit_code=code, mode=self.capture_mode)
        if self.state == "stopping":
            self._finish_shutdown()
        elif self.capture_mode == "preflight" and self.state == "preflight":
            if code != 0 or not self._valid_preflight():
                self._fatal("preflight_worker_failed")
            else:
                self._maybe_prepared()
        elif self.state != "idle":
            self._fatal("capture_exited: " + str(code))

    def _model_finished(self, code):
        self._model_waiting = False
        self.model_ready = False
        self._event("model_finished", exit_code=code)
        if self.state == "stopping":
            self._finish_shutdown()
        elif self.state != "idle":
            self._fatal("model_exited: " + str(code))

    def stop(self, reason="user_stop"):
        if self.state == "stopping":
            self._finish_shutdown()
            return
        if self.state == "idle":
            if self._closing:
                self._allow_close = True
                QTimer.singleShot(0, self.close)
            return
        self.state = "stopping"
        self._command_deadline = None
        self.buffer.clear(reason)
        self.gate.invalidate(reason)
        self._publish_unknown(reason, request=self._inflight)
        self._event("stop_requested", reason=reason)
        if self.display is not None:
            try:
                self.display.stop_flicker()
            except Exception as exc:
                self._event("display_stop_failed", reason=str(exc))
        self.calibrated = False
        self.model_ready = False
        self.status_label.setText("正在停止并等待两个进程退出：" + str(reason))
        self._refresh_controls()
        self.capture_worker.stop()
        self.model_worker.stop()
        self._finish_shutdown()

    def _finish_shutdown(self):
        if (self.state != "stopping" or self.capture_worker.running or self.model_worker.running
                or self._capture_waiting or self._model_waiting):
            return
        if self.display is not None:
            display, self.display = self.display, None
            display.close()
        if self.journal is not None:
            journal, self.journal = self.journal, None
            try:
                journal.close()
                if (self.session_dir / "session.json").is_file():
                    from eeg_tools.quality.reporting import build_session_report
                    report = build_session_report(self.session_dir)
                    self.session_label.setText("会话：" + str(self.session_dir) + "\n报告：" + report["output_dir"])
                if self._failure_reason is None:
                    self.status_label.setText("已停止；会话与独立命令清单已保存")
                else:
                    self.status_label.setText("已中止；记录已保存。原因：" + self._failure_reason
                                              + "\n采集日志：" + str(self.session_dir / "capture.stderr.log"))
            except Exception as exc:
                self.status_label.setText("已停止；命令清单写入失败：" + str(exc))
        if self._udp is not None:
            self._udp.close()
            self._udp = None
        self._inflight = None
        self._quality_waiting_window = None
        self.preflight_report = None
        self._model_started = self._capture_started = self._last_samples = None
        self.capture_mode = None
        self._settings = None
        self.state = "idle"
        self.electrode_check.setChecked(False)
        self.risk_check.setChecked(False)
        self.buffer_label.setText("连续窗口：0 / 1250 点")
        self._refresh_controls()
        if self._closing:
            self._allow_close = True
            QTimer.singleShot(0, self.close)

    def closeEvent(self, event):
        if self._allow_close or (self.state == "idle" and not self.capture_worker.running
                                 and not self.model_worker.running and self.journal is None):
            self.timer.stop()
            event.accept()
            return
        event.ignore()
        self._closing = True
        self.stop("window_closed")


def main(argv=None):
    app = QApplication.instance() or QApplication(sys.argv if argv is None else argv)
    window = RealtimeCommandsWindow()
    window.show()
    if "--smoke-test" in (sys.argv if argv is None else argv):
        QTimer.singleShot(500, window.close)
    return app.exec()


if __name__ == "__main__":
    # Direct module execution uses the same formal configuration gate as the script.
    from scripts.validate_release_configs import validate_release_configs
    validate_release_configs()
    raise SystemExit(main())
