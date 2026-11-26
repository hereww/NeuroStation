"""Operator-only EEG acquisition dashboard; no hardware access or stimulus rendering."""
from __future__ import annotations

import math
import time

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView, QHeaderView, QHBoxLayout, QMainWindow, QProgressBar,
    QScrollArea, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from .components import KeyValues, Section, WaveformWidget, action, label
from .gateway import CaptureConfig, DeviceInfo, Phase, TaskSnapshot, format_duration
from .monitor_model import AcquisitionMonitorModel
from .quality_panel import QualityPanel


class AcquisitionMonitorWindow(QMainWindow):
    dismissed = Signal()
    cancel_requested = Signal()

    def __init__(self, tr, parent=None):
        # An owned/transient window can be kept on its parent's screen by
        # the window manager. The main window explicitly owns our lifecycle.
        super().__init__(None, Qt.WindowType.Window)
        if parent is not None:
            self.setStyleSheet(parent.styleSheet())
        self.tr = tr
        self.model = AcquisitionMonitorModel()
        self.last_sample_at: float | None = None
        self.started_at = time.monotonic()
        self.target_screen = None
        self.target_screen_name = ""
        self.setWindowTitle(tr("monitor.title"))
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        self.resize(1280, 720)
        self.setMinimumSize(640, 480)
        content = QWidget()
        self.setCentralWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)
        header = QHBoxLayout()
        header.addWidget(label(tr("monitor.title"), "sectionTitle"), 1)
        self.fullscreen_button = action(tr("monitor.fullscreen"), self.toggle_fullscreen)
        header.addWidget(self.fullscreen_button)
        header.addWidget(action(tr("action.stop_task"), self.cancel_requested.emit))
        self.fullscreen_shortcut = QShortcut(QKeySequence("F11"), self)
        self.fullscreen_shortcut.activated.connect(self.toggle_fullscreen)
        self.escape_shortcut = QShortcut(QKeySequence("Escape"), self)
        self.escape_shortcut.activated.connect(self.leave_fullscreen)
        layout.addLayout(header)
        self.status = label(tr("monitor.waiting"), "notice")
        layout.addWidget(self.status)
        self.ssvep_prediction = label("", "notice")
        self.ssvep_prediction.hide()
        layout.addWidget(self.ssvep_prediction)
        summary_layout = QHBoxLayout()
        self.session = KeyValues([(tr(key), "—") for key in (
            "field.participant", "monitor.device", "monitor.rate", "monitor.channels", "monitor.preview_samples",
        )], columns=2)
        self.task = KeyValues([(tr(key), "—") for key in (
            "task.phase", "task.elapsed", "task.remaining", "task.trial", "task.target", "task.events",
        )], columns=2)
        for values in (self.session, self.task):
            values.layout().setVerticalSpacing(2)
            values.layout().setHorizontalSpacing(12)
        summary_layout.addWidget(self.session, 1)
        summary_layout.addWidget(self.task, 1)
        layout.addLayout(summary_layout)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setFormat("%p%")
        self.progress.setMaximumHeight(16)
        layout.addWidget(self.progress)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        self.quality_panel = QualityPanel(self, tr=tr)
        # Eight rows and the selected-channel spectrum must fit a 720p
        # operator display without the workstation's spacious table padding.
        self.quality_panel.table.setStyleSheet("QTableView::item { padding: 2px 4px; }")
        self.quality_panel.table.verticalHeader().setMinimumSectionSize(18)
        self.quality_panel.table.verticalHeader().setDefaultSectionSize(22)
        self.quality_panel.table.setMinimumHeight(220)
        self.quality_panel.table.setMaximumHeight(240)
        self.quality_panel.plot.setMinimumHeight(120)
        quality_scroll = QScrollArea()
        quality_scroll.setWidgetResizable(True)
        quality_scroll.setWidget(self.quality_panel)
        self.tabs.addTab(quality_scroll, tr("quality.title"))
        raw_page = QWidget()
        details = QHBoxLayout(raw_page)
        details.setContentsMargins(0, 0, 0, 0)
        waveform_section = Section(tr("task.waveform"))
        self.waveform = WaveformWidget(
            tr("task.waveform"), tr("task.waveform_badge"), empty_text=tr("task.waveform_empty"),
        )
        self.waveform.setMinimumHeight(240)
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
        self.table.setMinimumWidth(280)
        self.table.setMinimumHeight(200)
        channel_section.layout.addWidget(self.table, 1)
        self.invalid_label = label("", "muted")
        channel_section.layout.addWidget(self.invalid_label)
        details.addWidget(channel_section, 2)
        self.tabs.addTab(raw_page, tr("task.waveform"))
        info = QWidget()
        info_layout = QVBoxLayout(info)
        info_layout.addWidget(label(tr("monitor.metrics_note"), "muted"))
        info_layout.addWidget(label(tr("monitor.unsupported_note"), "muted"))
        info_layout.addStretch()
        self.tabs.addTab(info, tr("monitor.info"))
        self.screen_label = label("", "muted")
        layout.addWidget(self.screen_label)

    def show_on_screen(self, screen) -> None:
        """Fill the selected extended screen without activating over the stimulus."""
        self.target_screen = screen
        self.target_screen_name = screen.name()
        self.hide()
        self.setWindowState(Qt.WindowState.WindowNoState)
        self.setScreen(screen)
        self.setGeometry(screen.geometry())
        # Realize a normal window on this screen first. On Windows with
        # scaled displays, a direct hidden -> full-screen transition can
        # otherwise use the primary display's cached native placement.
        self.show()
        self.windowHandle().setScreen(screen)
        self.setGeometry(screen.geometry())
        self.showFullScreen()
        self.fullscreen_button.setText(self.tr("monitor.windowed"))

    def leave_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
            if self.target_screen is not None:
                self.setGeometry(self.target_screen.availableGeometry())
            self.showMaximized()
            self.fullscreen_button.setText(self.tr("monitor.fullscreen"))

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.leave_fullscreen()
        elif self.target_screen is not None:
            self.show_on_screen(self.target_screen)

    def update_samples(self, names, rate: int, samples: list[list[float]], stream_ids=None) -> None:
        self.model.configure(names, rate)
        self.waveform.configure(names, rate)
        self.model.append(samples)
        self.waveform.push_samples(samples, stream_ids=stream_ids)
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
        self.ssvep_prediction.setVisible(snapshot.active and config.continuous_ssvep)
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
