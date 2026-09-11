"""Qt pages; all acquisition commands are emitted to the window's gateway owner."""
from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import math
import re

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout, QLineEdit,
    QSpinBox, QComboBox, QCheckBox, QFileDialog, QProgressBar, QStyle,
    QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem,
)

from .components import Page, Section, KeyValues, AppTile, StaticTargets, WaveformWidget, action, label
from .gateway import CaptureConfig, CaptureMode, TaskSnapshot, Phase, Dataset, format_duration


def _readonly_table(headers: tuple[str, ...]) -> QTableWidget:
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(list(headers))
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    table.setAlternatingRowColors(True)
    # Fill rows before enabling sorting; QTableWidget can move the active row
    # while individual cells are being inserted when sorting is already on.
    table.setSortingEnabled(False)
    table.setWordWrap(False)
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setStretchLastSection(True)
    table.setMinimumHeight(72)
    return table


def _table_item(value: object) -> QTableWidgetItem:
    item = QTableWidgetItem(str(value))
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    item.setToolTip(str(value))
    return item


def _format_file_size(size: int | None) -> str:
    if size is None:
        return "—"
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    if size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    return f"{size / (1024 * 1024 * 1024):.2f} GB"


def _dataset_file_rows(result: Dataset) -> list[tuple[str, str, str, str]]:
    """Build a metadata-only file table; raw EEG content is never loaded."""

    names = result.files
    if not names and result.path.is_dir():
        try:
            names = tuple(
                sorted(
                    path.relative_to(result.path).as_posix()
                    for path in result.path.rglob("*")
                    if path.is_file() and path.name not in {"session.json", "session.json.pending"}
                )
            )
        except OSError:
            names = ()

    source_metadata: dict[str, dict[str, object]] = {}
    metadata_path = result.path / "session.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        raw_files = metadata.get("raw_files", ())
        if isinstance(raw_files, list):
            source_metadata = {
                str(item["name"]): item
                for item in raw_files
                if isinstance(item, dict) and item.get("name")
            }
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass

    rows: list[tuple[str, str, str, str]] = []
    for name in names:
        path = result.path / name
        # Imported sessions reserve session.json for workstation metadata and
        # retain an original source session.json as source_session.json.
        if result.imported and name == "session.json" and (result.path / "source_session.json").is_file():
            path = result.path / "source_session.json"
        elif not path.is_file() and name == "session.json":
            path = result.path / "source_session.json"
        size: int | None = None
        modified_ns: int | None = None
        try:
            stat = path.stat()
            size = stat.st_size
            modified_ns = stat.st_mtime_ns
        except OSError:
            detail = source_metadata.get(name, {})
            try:
                size = int(detail.get("size_bytes"))
            except (TypeError, ValueError):
                size = None
            try:
                modified_ns = int(detail.get("modified_ns"))
            except (TypeError, ValueError):
                modified_ns = None
        suffix = Path(name).suffix.lower().lstrip(".")
        kind = suffix.upper() if suffix else "FILE"
        modified = "—"
        if modified_ns is not None:
            try:
                modified = datetime.fromtimestamp(modified_ns / 1_000_000_000).astimezone().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            except (OSError, OverflowError, ValueError):
                pass
        rows.append((name, kind, _format_file_size(size), modified))
    return rows


def _dataset_raw_file_names(result: Dataset) -> tuple[str, ...]:
    names = result.files
    if not names and result.path.is_dir():
        try:
            names = tuple(
                sorted(
                    path.relative_to(result.path).as_posix()
                    for path in result.path.rglob("*")
                    if path.is_file()
                )
            )
        except OSError:
            names = ()
    return tuple(
        name
        for name in names
        if (
            re.fullmatch(r"BrainFlow-RAW_.*\.csv", name, re.IGNORECASE)
            or re.fullmatch(r"OpenBCI-RAW-.*\.txt", name, re.IGNORECASE)
            or Path(name).name.lower() == "raw_brainflow.tsv"
        )
    )


def _read_data_preview(path: Path, limit: int = 100) -> tuple[tuple[str, ...], list[list[str]]]:
    """Read only a bounded prefix for the Excel-like read-only data preview."""

    rows: list[list[str]] = []
    headers: list[str] | None = None
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            first_line = handle.readline()
            if not first_line:
                return (), []
            delimiter = "\t" if "\t" in first_line else ","
            handle.seek(0)
            reader = csv.reader(handle, delimiter=delimiter)
            for row in reader:
                if not row or not any(cell.strip() for cell in row):
                    continue
                first = row[0].strip()
                if first.startswith("%"):
                    continue
                try:
                    float(first)
                except ValueError:
                    if headers is None:
                        headers = row
                    continue
                rows.append(row)
                if len(rows) >= limit:
                    break
    except (OSError, UnicodeError, csv.Error):
        return (), []

    column_count = max((len(row) for row in rows), default=len(headers or ()))
    if column_count == 0:
        return (), []
    if headers is None:
        headers = [f"Column {index}" for index in range(1, column_count + 1)]
    elif len(headers) < column_count:
        headers.extend(
            f"Column {index}" for index in range(len(headers) + 1, column_count + 1)
        )
    normalized = [row + [""] * (column_count - len(row)) for row in rows]
    return tuple(headers[:column_count]), normalized


class HomePage(Page):
    def __init__(self, tr, navigate):
        super().__init__(tr, tr("home.ready"), tr("home.subtitle"))
        device = Section("OpenBCI Cyton")
        device.layout.addWidget(label("AUTO · 8 CH · 250 Hz"))
        device.layout.addWidget(label(tr("device.connected"), "muted"))
        self.layout.addWidget(device)
        self.layout.addWidget(action(tr("home.check"), lambda: navigate("live")))
        self.layout.addWidget(action(tr("home.apps"), lambda: navigate("apps"), True))
        self.layout.addStretch()


class DevicesPage(Page):
    def __init__(self, tr, navigate):
        super().__init__(tr, tr("nav.devices"), tr("device.serial_note"))
        device = Section("OpenBCI Cyton · AUTO")
        device.layout.addWidget(KeyValues([
            (tr("device.title"), tr("device.connected")),
            (tr("device.channels"), "8 CH / 250 Hz"),
            (tr("device.transport"), "USB Dongle"),
            (tr("device.owner"), "NeuroStation"),
        ]))
        self.layout.addWidget(device)
        mapping = Section(tr("device.mapping"))
        mapping.layout.addWidget(label("CH 1–4: Fp1 · Fp2 · C3 · C4\nCH 5–8: P7 · P8 · O1 · O2"))
        mapping.layout.addWidget(label(tr("device.mapping_note"), "muted"))
        self.layout.addWidget(mapping)
        self.layout.addWidget(action(tr("action.live"), lambda: navigate("live"), True))
        self.layout.addStretch()


class LivePage(Page):
    start_requested = Signal(object)
    stop_requested = Signal()
    marker_requested = Signal()

    def __init__(self, tr, config: CaptureConfig):
        super().__init__(tr, tr("nav.live"), tr("live.subtitle"))
        self.config = config
        self.layout.addWidget(label(tr("live.test_mode"), "notice"))
        self.layout.addWidget(label(tr("live.status"), "muted"))
        settings = Section(tr("live.settings"))
        form = QFormLayout()
        self.participant = QLineEdit(config.participant)
        self.name = QLineEdit(tr("live.test_name"))
        form.addRow(tr("field.participant"), self.participant)
        form.addRow(tr("field.name"), self.name)
        settings.layout.addLayout(form)
        controls = QHBoxLayout()
        self.start_button = action(tr("live.start"), self._start, True)
        self.stop_button = action(tr("live.stop"), self.stop_requested.emit)
        self.marker_button = action(tr("live.marker"), self.marker_requested.emit)
        for button in (self.start_button, self.stop_button, self.marker_button):
            controls.addWidget(button)
        settings.layout.addLayout(controls)
        self.status = label(tr("live.idle"))
        settings.layout.addWidget(self.status)
        self.layout.addWidget(settings)
        wave = Section(tr("live.waveform"))
        wave.layout.addWidget(label(tr("live.display"), "muted"))
        self.waveform = WaveformWidget(tr("live.waveform"), tr("live.waveform_badge"))
        wave.layout.addWidget(self.waveform)
        wave.layout.addWidget(label(tr("live.filter_note"), "muted"))
        self.layout.addWidget(wave)

    def _start(self):
        self.start_requested.emit(CaptureConfig(
            participant=self.participant.text().strip(),
            name=self.name.text().strip(),
            save_directory=self.config.save_directory,
            mode=CaptureMode.DEMO,
        ))

    def update_snapshot(self, snapshot: TaskSnapshot):
        manual = snapshot.protocol == "manual" and snapshot.phase == Phase.RUNNING
        self.start_button.setEnabled(not snapshot.active)
        self.stop_button.setEnabled(manual)
        self.marker_button.setEnabled(manual)
        self.participant.setEnabled(not snapshot.active)
        self.name.setEnabled(not snapshot.active)
        self.waveform.set_active(manual)
        self.status.setText(f"{self.tr('live.recording')} · {format_duration(snapshot.elapsed)} · "
            +self.tr("live.markers", count=snapshot.event_count) if manual else self.tr("live.idle"))


class AppsPage(Page):
    def __init__(self, tr, navigate):
        super().__init__(tr, tr("nav.apps"), tr("apps.subtitle"))
        grid = QGridLayout()
        grid.setSpacing(18)
        entries = [
            ("apps.ssvep", "apps.ssvep_note", QStyle.StandardPixmap.SP_ComputerIcon, lambda: navigate("ssvep")),
            ("apps.rest", "apps.rest_note", QStyle.StandardPixmap.SP_MediaPlay, lambda: navigate("live")),
            ("apps.motor", "apps.soon", QStyle.StandardPixmap.SP_DialogApplyButton, None),
            ("apps.p300", "apps.soon", QStyle.StandardPixmap.SP_MediaVolume, None),
        ]
        for index, (title, subtitle, icon_type, callback) in enumerate(entries):
            grid.addWidget(AppTile(tr(title), tr(subtitle), icon_type, callback), index//2, index%2)
        self.layout.addLayout(grid)
        device = Section("OpenBCI Cyton · AUTO · 8 CH · 250 Hz")
        device.layout.addWidget(label(tr("device.connected"), "muted"))
        self.layout.addWidget(device)
        self.layout.addWidget(label(tr("apps.flow"), "muted"))
        self.layout.addStretch()


class SSVEPPage(Page):
    start_requested = Signal(object, float)
    serial_scan_requested = Signal()

    def __init__(self, tr, config: CaptureConfig, navigate):
        super().__init__(tr, "SSVEP", tr("ssvep.subtitle"))
        self.layout.addWidget(action(tr("action.back_apps"), lambda: navigate("apps")))
        self.layout.addWidget(label(tr("ssvep.description"), "muted"))
        self.layout.addWidget(label(tr("ssvep.defaults"), "muted"))
        self.layout.addWidget(label(tr("ssvep.safety"), "notice"))
        section = Section(tr("ssvep.parameters"))
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.participant = QLineEdit(config.participant)
        self.name = QLineEdit(config.name)
        self.stimulus = self._spin(1, 30, config.stimulus_seconds, tr("field.seconds"))
        self.rest = self._spin(0, 30, config.rest_seconds, tr("field.seconds"))
        self.repetitions = self._spin(1, 10, config.repetitions)
        self.mode = QComboBox()
        for mode in CaptureMode:
            if mode is CaptureMode.IMPORTED_OPENBCI:
                continue
            self.mode.addItem(tr("mode." + mode.value), mode)
        self.mode.setCurrentIndex(max(0, self.mode.findData(config.mode)))
        self.port = QLineEdit(config.port)
        self.port.setPlaceholderText("AUTO")
        port_picker = QWidget()
        port_picker_layout = QHBoxLayout(port_picker)
        port_picker_layout.setContentsMargins(0, 0, 0, 0)
        port_picker_layout.addWidget(self.port, 1)
        port_picker_layout.addWidget(action(tr("action.scan_ports"), self.serial_scan_requested.emit))
        self.port_status = label(tr("field.port_auto"), "muted")
        self.screen = self._spin(0, 15, config.screen_index)
        self.speed = QComboBox()
        for value in (1, 4, 8, 16):
            self.speed.addItem(f"{value}×", value)
        self.speed.setCurrentIndex(2)
        self.save = QLineEdit(str(config.save_directory))
        picker = QWidget()
        picker_layout = QHBoxLayout(picker)
        picker_layout.setContentsMargins(0, 0, 0, 0)
        picker_layout.addWidget(self.save, 1)
        picker_layout.addWidget(action(tr("action.browse"), self._choose_directory))
        self.channel = QLineEdit(str(config.channel_config or ""))
        self.channel.setPlaceholderText(tr("field.channel_manual_placeholder"))
        self.channel_manual = QCheckBox(tr("field.channel_manual"))
        self.channel_manual.setChecked(config.channel_config is not None)
        self.channel_browse = None
        self.channel_picker = QWidget()
        channel_picker_layout = QHBoxLayout(self.channel_picker)
        channel_picker_layout.setContentsMargins(0, 0, 0, 0)
        channel_picker_layout.addWidget(self.channel, 1)
        self.channel_browse = action(tr("action.browse"), self._choose_channel_config)
        channel_picker_layout.addWidget(self.channel_browse)
        channel_controls = QWidget()
        channel_controls_layout = QVBoxLayout(channel_controls)
        channel_controls_layout.setContentsMargins(0, 0, 0, 0)
        channel_controls_layout.setSpacing(6)
        self.channel_auto_label = label(tr("field.channel_auto"), "muted")
        channel_controls_layout.addWidget(self.channel_auto_label)
        channel_controls_layout.addWidget(self.channel_picker)
        self.acknowledge = QCheckBox(tr("ssvep.acknowledge"))
        self.acknowledge.setChecked(config.acknowledge_flicker_risk)
        self.allow_draft = QCheckBox(tr("ssvep.allow_draft"))
        self.allow_draft.setChecked(config.allow_draft_hardware_config)
        for key, widget in (("field.participant", self.participant), ("field.name", self.name),
                            ("field.mode", self.mode), ("field.port", port_picker),
                            ("field.screen", self.screen),
                            ("field.stimulus", self.stimulus), ("field.rest", self.rest),
                            ("field.repetitions", self.repetitions), ("field.refresh", label("60 Hz")),
                            ("field.speed", self.speed), ("field.save", picker),
                            ("field.channel_config", channel_controls)):
            form.addRow(tr(key), widget)
        form.addRow("", self.channel_manual)
        form.addRow("", self.acknowledge)
        form.addRow("", self.allow_draft)
        section.layout.addLayout(form)
        section.layout.addWidget(self.port_status)
        section.layout.addWidget(label(tr("ssvep.refresh_note"), "muted"))
        self.estimate = label("", "estimate")
        section.layout.addWidget(self.estimate)
        section.layout.addWidget(label(tr("ssvep.formula"), "muted"))
        self.layout.addWidget(section)
        self.layout.addWidget(StaticTargets(tr))
        self.layout.addWidget(KeyValues([(tr("device.channels"), "OpenBCI Cyton · AUTO · 8 CH / 250 Hz"),
            (tr("ssvep.frequencies"), "10 / 12 / 15 / 20 Hz"), ("Marker", tr("ssvep.marker_map"))]))
        self.error = label("", "error")
        self.layout.addWidget(self.error)
        self.layout.addWidget(label(tr("ssvep.countdown_note"), "muted"))
        self.start_button = action(tr("action.start_task"), self._start, True)
        self.start_button.setObjectName("startTaskButton")
        self.layout.addWidget(self.start_button)
        for spin in (self.stimulus, self.rest, self.repetitions):
            spin.valueChanged.connect(self._update_estimate)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.channel_manual.toggled.connect(self._channel_manual_changed)
        self._mode_changed()
        self._update_estimate()

    @staticmethod
    def _spin(minimum: int, maximum: int, value: int, suffix="") -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSuffix(suffix)
        return spin

    def config(self) -> CaptureConfig:
        mode = CaptureMode(self.mode.currentData())
        channel_config = (
            self.channel.text().strip()
            if mode == CaptureMode.CYTON and self.channel_manual.isChecked()
            else ""
        )
        return CaptureConfig(participant=self.participant.text().strip(), name=self.name.text().strip(),
            stimulus_seconds=self.stimulus.value(), rest_seconds=self.rest.value(),
            repetitions=self.repetitions.value(), save_directory=Path(self.save.text()).expanduser(),
            mode=CaptureMode(self.mode.currentData()), port=self.port.text().strip(),
            screen_index=self.screen.value(),
            acknowledge_flicker_risk=self.acknowledge.isChecked(),
            channel_config=Path(channel_config).expanduser() if channel_config else None,
            allow_draft_hardware_config=self.allow_draft.isChecked())

    def _update_estimate(self):
        config = self.config()
        self.estimate.setText(self.tr("ssvep.estimate", total=format_duration(config.total_seconds),
            recording=format_duration(config.recording_seconds), trials=config.trials))

    def _choose_directory(self):
        path = QFileDialog.getExistingDirectory(self, self.tr("field.save"), self.save.text())
        if path:
            self.save.setText(path)

    def _choose_channel_config(self):
        path, _selected = QFileDialog.getOpenFileName(
            self, self.tr("field.channel_config"), self.channel.text(), "JSON (*.json)"
        )
        if path:
            self.channel.setText(path)

    def set_detected_ports(self, ports: tuple[dict[str, str], ...] | list[dict[str, str]]) -> None:
        devices = [str(item.get("device", "")).strip() for item in ports if item.get("device")]
        if devices:
            self.port.setText("AUTO")
            self.port_status.setText(self.tr("field.port_detected", ports=", ".join(devices)))
        else:
            self.port.setText("AUTO")
            self.port_status.setText(self.tr("field.port_none"))

    def _mode_changed(self):
        mode = CaptureMode(self.mode.currentData())
        demo = mode == CaptureMode.DEMO
        cyton = mode == CaptureMode.CYTON
        preview = mode == CaptureMode.VISUAL_PREVIEW
        self.port.setEnabled(cyton)
        self.channel_manual.setEnabled(cyton)
        self.channel_auto_label.setVisible(cyton)
        manual_channels = cyton and self.channel_manual.isChecked()
        self.channel_picker.setVisible(manual_channels)
        # Keep this control enabled for keyboard/accessibility tooling while
        # the advanced path picker itself stays hidden in automatic mode.
        self.channel.setEnabled(cyton)
        self.channel.setReadOnly(not manual_channels)
        if self.channel_browse is not None:
            self.channel_browse.setEnabled(manual_channels)
        if not cyton and self.channel_manual.isChecked():
            self.channel_manual.blockSignals(True)
            self.channel_manual.setChecked(False)
            self.channel_manual.blockSignals(False)
        self.allow_draft.setEnabled(cyton)
        self.screen.setEnabled(not demo)
        self.acknowledge.setEnabled(not demo)
        self.speed.setEnabled(demo)
        if not demo:
            self.speed.setCurrentIndex(self.speed.findData(1))
        if preview:
            self.port.clear()

    def _channel_manual_changed(self, checked: bool):
        cyton = CaptureMode(self.mode.currentData()) == CaptureMode.CYTON
        manual_channels = cyton and checked
        self.channel_picker.setVisible(manual_channels)
        self.channel.setEnabled(cyton)
        self.channel.setReadOnly(not manual_channels)
        if self.channel_browse is not None:
            self.channel_browse.setEnabled(manual_channels)

    def _start(self):
        try:
            config = self.config()
            config.validate()
        except ValueError as error:
            self.error.setText(self.tr(str(error)))
            return
        self.error.clear()
        speed = float(self.speed.currentData()) if config.mode == CaptureMode.DEMO else 1.0
        self.start_requested.emit(config, speed)


class TaskPage(Page):
    cancel_requested = Signal()

    def __init__(self, tr):
        super().__init__(tr, "SSVEP")
        self.cancel_button = action(tr("action.cancel"), self.cancel_requested.emit)
        self.layout.addWidget(self.cancel_button)
        self.countdown_section = Section(tr("task.prepare_note"))
        self.countdown = label("5", "countdown")
        self.countdown.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.countdown.setMinimumHeight(130)
        self.countdown_section.layout.addWidget(self.countdown)
        self.layout.addWidget(self.countdown_section)
        self.run_section = Section(tr("task.running"))
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setFormat("%p%")
        self.run_section.layout.addWidget(self.progress)
        self.stats = KeyValues([(tr(key), "—") for key in (
            "task.elapsed", "task.remaining", "task.trial", "task.target", "task.phase", "task.events", "task.demo")])
        self.run_section.layout.addWidget(self.stats)
        self.capture_label = label("", "muted")
        self.run_section.layout.addWidget(self.capture_label)
        self.speed_label = label("", "muted")
        self.run_section.layout.addWidget(self.speed_label)
        self.layout.addWidget(self.run_section)
        self.layout.addWidget(StaticTargets(tr))
        self.layout.addStretch()

    def update_snapshot(self, snapshot: TaskSnapshot, config: CaptureConfig):
        countdown = snapshot.phase == Phase.COUNTDOWN
        self.title_label.setText("SSVEP · "+self.tr("task.preparing" if countdown else "task.running"))
        self.countdown_section.setVisible(countdown)
        self.run_section.setVisible(not countdown)
        self.cancel_button.setText(self.tr("action.cancel" if countdown else "action.stop_task"))
        self.countdown.setText(str(snapshot.countdown))
        self.progress.setValue(round(snapshot.progress*10))
        phase_key = "task.rest" if snapshot.resting else (
            "task.stimulus" if config.mode == CaptureMode.DEMO else "task.stimulus_visual"
        )
        values = [format_duration(snapshot.elapsed), format_duration(math.ceil(snapshot.remaining)),
            f"{snapshot.trial} / {snapshot.trial_count}", f"{snapshot.target} / {snapshot.frequency} Hz",
            self.tr(phase_key),
            str(snapshot.event_count), format_duration(snapshot.demo_elapsed)]
        for node, value in zip(self.stats.values, values):
            node.setText(value)
        self.capture_label.setText(self.tr("task.capture." + config.mode.value))
        if config.mode == CaptureMode.DEMO:
            self.speed_label.setText(self.tr("task.speed", speed=f"{snapshot.speed:g}",
                duration=format_duration(config.recording_seconds)))
        else:
            self.speed_label.setText(self.tr("task.realtime", duration=format_duration(config.recording_seconds)))


class ResultPage(Page):
    def __init__(self, tr, result: Dataset, navigate):
        title_key = {
            "completed": "result.title",
            "aborted": "result.aborted_title",
            "error": "result.error_title",
        }.get(result.status, "result.title")
        capture_test = result.protocol == "manual" or result.origin == "capture_test"
        if capture_test:
            status_key = "result.capture_test"
            notice_key = "result.capture_test_notice"
            path_key = "result.capture_test_path"
        elif result.source == CaptureMode.DEMO:
            status_key = "result.simulated_saved" if result.persisted else "result.simulated"
            notice_key = "result.notice_written" if result.persisted else "result.notice"
            path_key = "result.path_written" if result.persisted else "result.path"
        elif result.source == CaptureMode.VISUAL_PREVIEW:
            status_key = (
                "result.preview_saved"
                if result.status == "completed"
                else "result." + result.status + "_saved"
            )
            notice_key = "result.preview_notice"
            path_key = "result.path_written"
        else:
            status_key = (
                "result." + result.source.value + "_saved"
                if result.status == "completed"
                else "result." + result.status + "_saved"
            )
            notice_key = "result.recorded_notice"
            path_key = "result.recorded_path"
        simulated_visual = result.source in (CaptureMode.DEMO, CaptureMode.VISUAL_PREVIEW)
        sample_key = "result.test_samples" if capture_test else (
            "result.samples" if simulated_visual else "result.recorded_samples"
        )
        values_key = "result.test_values" if capture_test else (
            "result.values" if simulated_visual else "result.recorded_values"
        )
        super().__init__(tr, tr(title_key), f"{result.name} · {tr(status_key)}")
        self.layout.addWidget(label(tr(notice_key), "notice"))
        self.layout.addWidget(KeyValues([
            (tr("field.participant"), result.participant),
            (tr("result.recording"), format_duration(result.recording_seconds)),
            (tr("result.preparation"), format_duration(result.preparation_seconds)),
            (tr("result.demo"), format_duration(math.ceil(result.demo_seconds))),
            (tr("result.trials"), str(result.trials)),
            (tr("result.channels"), str(result.channel_count)),
            (tr(sample_key), f"{result.samples_per_channel:,}"),
            (tr(values_key), f"{result.sample_values:,}"),
            (tr("result.events"), str(result.event_count)),
        ]))
        path = Section(tr(path_key))
        path_value = tr("result.capture_test_memory") if capture_test else str(result.path)
        path.layout.addWidget(label(path_value, "path"))
        self.layout.addWidget(path)
        if result.error:
            error_section = Section(tr("result.error_details"))
            error_section.layout.addWidget(label(result.error, "error"))
            self.layout.addWidget(error_section)
        self.layout.addWidget(action(tr("app.workstation")+" / "+tr("app.datasets"), lambda: navigate("datasets"), True))
        self.layout.addWidget(action(tr("action.configure"), lambda: navigate("ssvep")))
        self.layout.addStretch()


class DatasetSummaryPage(Page):
    def __init__(self, tr, result: Dataset, navigate):
        super().__init__(tr, tr("dataset_summary.title"), result.name)
        self._result_path = result.path
        self.layout.addWidget(label(tr("dataset_summary.readonly"), "notice"))
        source_name = (
            tr("dataset_summary.capture_test")
            if result.protocol == "manual" or result.origin == "capture_test"
            else tr("dataset_summary.imported") if result.imported else tr("dataset_summary.acquired")
        )
        session_rows = (
            (tr("dataset_summary.session"), result.name),
            (tr("dataset_summary.source"), source_name),
            (tr("field.participant"), result.participant or tr("dataset_summary.unlabeled")),
            (tr("dataset_summary.markers"), tr("dataset_summary.no_markers") if result.imported else str(result.event_count)),
            (tr("result.channels"), str(result.channel_count)),
            (tr("dataset_summary.sampling_rate"), f"{result.sampling_rate_hz:g} Hz"),
            (tr("dataset_summary.duration"), format_duration(result.recording_seconds)),
            (tr("dataset_summary.samples_per_channel"), f"{result.samples_per_channel:,}"),
            (tr("dataset_summary.file_count"), str(len(_dataset_file_rows(result)))),
            (
                tr("dataset_summary.workstation_copy"),
                tr("result.capture_test_memory")
                if result.protocol == "manual" or result.origin == "capture_test"
                else str(result.path),
            ),
            (tr("dataset_summary.original_source"), result.source_path or "—"),
        )
        session_table = _readonly_table(
            (tr("dataset_summary.field"), tr("dataset_summary.value"))
        )
        session_table.setObjectName("datasetSessionTable")
        session_table.setAccessibleName(tr("dataset_summary.session_table"))
        session_table.setColumnWidth(0, 210)
        session_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        for key, value in session_rows:
            row = session_table.rowCount()
            session_table.insertRow(row)
            session_table.setItem(row, 0, _table_item(key))
            session_table.setItem(row, 1, _table_item(value))
        session_table.resizeRowsToContents()
        session_table.setSortingEnabled(True)
        session_table.setMinimumHeight(min(420, max(150, session_table.sizeHintForRow(0) * len(session_rows) + 44)))
        self.layout.addWidget(session_table)

        files = Section(tr("dataset_summary.files"))
        file_rows = _dataset_file_rows(result)
        file_table = _readonly_table(
            (
                tr("dataset_summary.file_name"),
                tr("dataset_summary.file_type"),
                tr("dataset_summary.file_size"),
                tr("dataset_summary.file_modified"),
            )
        )
        file_table.setObjectName("datasetFilesTable")
        file_table.setAccessibleName(tr("dataset_summary.file_table"))
        file_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        for column in (1, 2, 3):
            file_table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        for values in file_rows:
            row = file_table.rowCount()
            file_table.insertRow(row)
            for column, value in enumerate(values):
                file_table.setItem(row, column, _table_item(value))
        file_table.resizeRowsToContents()
        file_table.setSortingEnabled(True)
        file_table.setMinimumHeight(min(360, max(84, file_table.sizeHintForRow(0) * max(1, len(file_rows)) + 44)))
        files.layout.addWidget(file_table)
        self.layout.addWidget(files)

        raw_names = _dataset_raw_file_names(result)
        if raw_names:
            preview = Section(tr("dataset_summary.data_preview"))
            preview.layout.addWidget(label(tr("dataset_summary.data_preview_note"), "muted"))
            self.preview_file = QComboBox()
            self.preview_file.addItems(list(raw_names))
            self.preview_file.setAccessibleName(tr("dataset_summary.preview_file"))
            preview.layout.addWidget(self.preview_file)
            headers, values = _read_data_preview(result.path / raw_names[0])
            self.preview_table = _readonly_table(headers or (tr("dataset_summary.no_columns"),))
            self.preview_table.setObjectName("datasetRawPreviewTable")
            self.preview_table.setAccessibleName(tr("dataset_summary.raw_preview_table"))
            preview.layout.addWidget(self.preview_table)
            self._set_preview_table(headers, values)
            self.preview_file.currentTextChanged.connect(self._preview_file_changed)
            self.layout.addWidget(preview)
        self.layout.addWidget(action(tr("app.datasets"), lambda: navigate("datasets"), True))
        self.layout.addStretch()

    def _preview_file_changed(self, name: str) -> None:
        headers, values = _read_data_preview(self._result_path / name)
        self._set_preview_table(headers, values)

    def _set_preview_table(
        self, headers: tuple[str, ...], values: list[list[str]]
    ) -> None:
        table = self.preview_table
        table.setSortingEnabled(False)
        table.clearContents()
        table.setColumnCount(len(headers) or 1)
        table.setHorizontalHeaderLabels(list(headers) or [self.tr("dataset_summary.no_columns")])
        table.setRowCount(0)
        for values_row in values:
            row = table.rowCount()
            table.insertRow(row)
            for column, value in enumerate(values_row):
                table.setItem(row, column, _table_item(value))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.setMinimumHeight(min(440, max(108, 30 * min(10, max(1, len(values))) + 44)))
        table.setSortingEnabled(True)


class DatasetsPage(Page):
    import_requested = Signal(object)

    def __init__(self, tr, datasets: tuple[Dataset, ...], latest: str | None, show_result, navigate,
                 import_busy: bool = False, import_status: str = "", import_directory: str = ""):
        super().__init__(tr, tr("app.datasets"), tr("datasets.subtitle"))
        controls = QHBoxLayout()
        controls.addWidget(action(tr("nav.apps"), lambda: navigate("apps")))
        self.import_button = action(tr("datasets.import_openbci"), self._choose_import_directory, True)
        self.import_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.import_button.setEnabled(not import_busy)
        controls.addWidget(self.import_button)
        self.layout.addLayout(controls)
        self.import_status = label(import_status, "muted")
        self.import_status.setVisible(bool(import_status))
        self.layout.addWidget(self.import_status)
        self.import_directory = import_directory
        if not datasets:
            self.layout.addWidget(label(tr("datasets.empty"), "muted"))
        for dataset in reversed(datasets):
            item = Section(dataset.name)
            item.setObjectName("latestDataset" if dataset.id == latest else "section")
            if dataset.protocol == "manual" or dataset.origin == "capture_test":
                status_key = "result.capture_test"
            elif dataset.id == latest:
                status_key = "datasets.latest." + dataset.source.value
            elif dataset.source == CaptureMode.DEMO:
                status_key = "result.simulated_saved" if dataset.persisted else "result.simulated"
            elif dataset.source == CaptureMode.VISUAL_PREVIEW:
                status_key = (
                    "result.preview_saved"
                    if dataset.status == "completed"
                    else "result." + dataset.status + "_saved"
                )
            else:
                status_key = "result." + dataset.source.value + "_saved"
            item.layout.addWidget(label(tr(status_key), "muted"))
            item.layout.addWidget(label(tr("datasets.details", participant=dataset.participant,
                duration=format_duration(dataset.recording_seconds), samples=f"{dataset.samples_per_channel:,}",
                events=dataset.event_count)))
            item.layout.addWidget(label(
                tr("result.capture_test_memory")
                if dataset.protocol == "manual" or dataset.origin == "capture_test"
                else str(dataset.path),
                "path",
            ))
            item.layout.addWidget(action(tr("action.view_dataset"), lambda _checked=False, d=dataset: show_result(d)))
            self.layout.addWidget(item)
        self.layout.addStretch()

    def _choose_import_directory(self):
        path = QFileDialog.getExistingDirectory(
            self, self.tr("datasets.import_openbci"), self.import_directory
        )
        if path:
            self.import_requested.emit(path)


class InfoPage(Page):
    def __init__(self, tr, kind: str, openbci_status=None, launch_openbci=None):
        if kind == "openbci":
            super().__init__(tr, tr("nav.openbci"), tr("openbci.subtitle"))
            if openbci_status is not None:
                self.layout.addWidget(KeyValues([
                    (tr("openbci.source"), tr("status.ready") if openbci_status.source_ready else tr("status.missing")),
                    (tr("openbci.chinese"), tr("status.ready") if openbci_status.overlay_ready else tr("status.missing")),
                    (tr("openbci.executable"), tr("status.ready") if openbci_status.executable_ready else tr("status.not_built")),
                    (tr("openbci.revision"), openbci_status.revision or "—"),
                ]))
            for key in ("openbci.notice", "openbci.tools"):
                section = Section()
                section.layout.addWidget(label(tr(key)))
                self.layout.addWidget(section)
            if launch_openbci is not None:
                launch = action(tr("openbci.launch"), launch_openbci, True)
                launch.setEnabled(bool(openbci_status and openbci_status.executable_ready))
                self.layout.addWidget(launch)
        else:
            super().__init__(tr, tr("nav.integrations"), tr("integration.subtitle"))
            for key in ("integration.brainflow", "integration.openbci", "integration.lsl", "integration.analysis"):
                section = Section()
                section.layout.addWidget(label(tr(key)))
                self.layout.addWidget(section)
        self.layout.addStretch()
