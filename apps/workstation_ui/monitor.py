"""Operator-only EEG acquisition dashboard; no hardware access or stimulus rendering."""
from __future__ import annotations

import math
import time

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QHeaderView, QHBoxLayout, QMainWindow, QProgressBar,
    QScrollArea, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from .components import KeyValues, Section, WaveformWidget, action, label
from .gateway import CaptureConfig, DeviceInfo, Phase, TaskSnapshot, format_duration
from .monitor_model import AcquisitionMonitorModel


class AcquisitionMonitorWindow(QMainWindow):
    dismissed = Signal()
    cancel_requested = Signal()

    def __init__(self, tr, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.tr = tr
        self.model = AcquisitionMonitorModel()
        self.last_sample_at: float | None = None
        self.started_at = time.monotonic()
        self.target_screen = None
        self.target_screen_name = ""
        self.setWindowTitle(tr("monitor.title"))
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.resize(1440, 900)
        self.setMinimumSize(640, 480)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.setCentralWidget(scroll)
        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)
        header = QHBoxLayout()
        header.addWidget(label(tr("monitor.title"), "pageTitle"), 1)
        header.addWidget(action(tr("action.stop_task"), self.cancel_requested.emit))
        layout.addLayout(header)
        self.status = label(tr("monitor.waiting"), "notice")
        layout.addWidget(self.status)
        summary = Section(tr("monitor.session"))
        summary_layout = QHBoxLayout()
        self.session = KeyValues([(tr(key), "—") for key in (
            "field.participant", "monitor.device", "monitor.rate", "monitor.channels", "monitor.preview_samples",
        )])
        self.task = KeyValues([(tr(key), "—") for key in (
            "task.phase", "task.elapsed", "task.remaining", "task.trial", "task.target", "task.events",
        )])
        summary_layout.addWidget(self.session, 1)
        summary_layout.addWidget(self.task, 1)
        summary.layout.addLayout(summary_layout)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setFormat("%p%")
        summary.layout.addWidget(self.progress)
        layout.addWidget(summary)
        details = QHBoxLayout()
        waveform_section = Section(tr("task.waveform"))
        self.waveform = WaveformWidget(
            tr("task.waveform"), tr("task.waveform_badge"), empty_text=tr("task.waveform_empty"),
        )
        waveform_section.layout.addWidget(self.waveform, 1)
        details.addWidget(waveform_section, 3)
        channel_section = Section(tr("monitor.channel_metrics"))
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels([tr(key) for key in (
            "monitor.channel", "monitor.mean", "monitor.rms", "monitor.peak_to_peak", "monitor.valid",
        )])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumWidth(420)
        self.table.setMinimumHeight(320)
        channel_section.layout.addWidget(self.table, 1)
        self.invalid_label = label("", "muted")
        channel_section.layout.addWidget(self.invalid_label)
        details.addWidget(channel_section, 2)
        layout.addLayout(details, 1)
        self.screen_label = label("", "muted")
        layout.addWidget(self.screen_label)
        layout.addWidget(label(tr("monitor.metrics_note"), "muted"))
        layout.addWidget(label(tr("monitor.unsupported_note"), "muted"))

    def show_on_screen(self, screen) -> None:
        """Create/move the native window before maximizing; do not steal stimulus focus."""
        self.target_screen = screen
        self.target_screen_name = screen.name()
        self.winId()
        self.windowHandle().setScreen(screen)
        self.setGeometry(screen.availableGeometry())
        self.showMaximized()

    def update_samples(self, names, rate: int, samples: list[list[float]]) -> None:
        self.model.configure(names, rate)
        self.waveform.configure(names, rate)
        self.model.append(samples)
        self.waveform.push_samples(samples)
        self.last_sample_at = time.monotonic()
        self.table.setRowCount(len(self.model.channel_names))
        for row, (name, metric) in enumerate(zip(self.model.channel_names, self.model.metrics())):
            values = [name, *(
                "—" if value is None else f"{value:.1f}"
                for value in (metric.mean_uv, metric.rms_uv, metric.peak_to_peak_uv)
            ), f"{metric.valid_samples} / {metric.valid_samples + metric.invalid_samples}"]
            for column, value in enumerate(values):
                item = self.table.item(row, column)
                if item is None:
                    item = QTableWidgetItem()
                    self.table.setItem(row, column, item)
                item.setText(value)
        self.invalid_label.setText(self.tr("monitor.invalid_values", count=self.model.invalid_value_count))
        self.session.values[4].setText(f"{self.model.sample_count:,}")

    def update_snapshot(self, snapshot: TaskSnapshot, config: CaptureConfig, device: DeviceInfo) -> None:
        self.waveform.set_active(snapshot.active)
        phase_key = (
            "task.preparing" if snapshot.phase is Phase.COUNTDOWN
            else "task.rest" if snapshot.resting else "task.stimulus_visual"
        )
        session_values = [
            config.participant, f"{device.name} · {device.port or config.port}",
            f"{self.model.sample_rate_hz or device.sample_rate} Hz",
            str(len(self.model.channel_names) or device.channels), f"{self.model.sample_count:,}",
        ]
        task_values = [
            self.tr(phase_key), format_duration(snapshot.elapsed), format_duration(math.ceil(snapshot.remaining)),
            f"{snapshot.trial} / {snapshot.trial_count}",
            f"{snapshot.frequency} Hz" if snapshot.trial else "—", str(snapshot.event_count),
        ]
        for nodes, values in ((self.session.values, session_values), (self.task.values, task_values)):
            for node, value in zip(nodes, values):
                node.setText(value)
        self.progress.setValue(round(snapshot.progress * 10))
        now = time.monotonic()
        if not snapshot.active:
            text = self.tr("monitor.finished")
        elif self.last_sample_at is None:
            key = "monitor.stale" if snapshot.phase is Phase.RUNNING and now - self.started_at > 2 else "monitor.waiting"
            text = self.tr(key)
        elif now - self.last_sample_at > 2:
            text = self.tr("monitor.stale")
        else:
            text = self.tr("monitor.receiving")
        self.status.setText(text)
        stimulus = snapshot.screen_name or self.tr("monitor.primary_screen")
        # QScreen wrappers may be destroyed during hotplug; keep only its name
        # for presentation rather than dereferencing it while updating labels.
        monitor = self.target_screen_name or "—"
        self.screen_label.setText(self.tr("monitor.screen_mapping", stimulus=stimulus, monitor=monitor))

    def closeEvent(self, event):
        self.dismissed.emit()
        super().closeEvent(event)
