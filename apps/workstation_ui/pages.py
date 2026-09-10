"""Qt pages; all acquisition commands are emitted to the window's gateway owner."""
from __future__ import annotations

from pathlib import Path
import math

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout, QLineEdit,
    QSpinBox, QComboBox, QCheckBox, QFileDialog, QProgressBar, QStyle,
)

from .components import Page, Section, KeyValues, AppTile, StaticTargets, WaveformWidget, action, label
from .gateway import CaptureConfig, CaptureMode, TaskSnapshot, Phase, Dataset, format_duration


class HomePage(Page):
    def __init__(self, tr, navigate):
        super().__init__(tr, tr("home.ready"), tr("home.subtitle"))
        device = Section("OpenBCI Cyton")
        device.layout.addWidget(label("COM5 · 8 CH · 250 Hz"))
        device.layout.addWidget(label(tr("device.connected"), "muted"))
        self.layout.addWidget(device)
        self.layout.addWidget(action(tr("home.check"), lambda: navigate("live")))
        self.layout.addWidget(action(tr("home.apps"), lambda: navigate("apps"), True))
        self.layout.addStretch()


class DevicesPage(Page):
    def __init__(self, tr, navigate):
        super().__init__(tr, tr("nav.devices"), tr("device.serial_note"))
        device = Section("OpenBCI Cyton · COM5")
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
        self.layout.addWidget(label("OpenBCI Cyton · COM5 · 8 CH · 250 Hz"))
        self.layout.addWidget(label(tr("live.status"), "muted"))
        settings = Section(tr("live.settings"))
        form = QFormLayout()
        self.participant = QLineEdit(config.participant)
        self.name = QLineEdit(tr("apps.rest"))
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
        wave.layout.addWidget(WaveformWidget(tr("live.waveform")))
        wave.layout.addWidget(label(tr("live.filter_note"), "muted"))
        self.layout.addWidget(wave)

    def _start(self):
        self.start_requested.emit(CaptureConfig(participant=self.participant.text().strip(),
            name=self.name.text().strip(), save_directory=self.config.save_directory))

    def update_snapshot(self, snapshot: TaskSnapshot):
        manual = snapshot.protocol == "manual" and snapshot.phase == Phase.RUNNING
        self.start_button.setEnabled(not snapshot.active)
        self.stop_button.setEnabled(manual)
        self.marker_button.setEnabled(manual)
        self.participant.setEnabled(not snapshot.active)
        self.name.setEnabled(not snapshot.active)
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
        device = Section("OpenBCI Cyton · COM5 · 8 CH · 250 Hz")
        device.layout.addWidget(label(tr("device.connected"), "muted"))
        self.layout.addWidget(device)
        self.layout.addWidget(label(tr("apps.flow"), "muted"))
        self.layout.addStretch()


class SSVEPPage(Page):
    start_requested = Signal(object, float)

    def __init__(self, tr, config: CaptureConfig, navigate):
        super().__init__(tr, "SSVEP", tr("ssvep.subtitle"))
        self.layout.addWidget(action(tr("action.back_apps"), lambda: navigate("apps")))
        self.layout.addWidget(label(tr("ssvep.description"), "muted"))
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
            self.mode.addItem(tr("mode." + mode.value), mode)
        self.mode.setCurrentIndex(max(0, self.mode.findData(config.mode)))
        self.port = QLineEdit(config.port)
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
        channel_picker = QWidget()
        channel_picker_layout = QHBoxLayout(channel_picker)
        channel_picker_layout.setContentsMargins(0, 0, 0, 0)
        channel_picker_layout.addWidget(self.channel, 1)
        channel_picker_layout.addWidget(action(tr("action.browse"), self._choose_channel_config))
        self.acknowledge = QCheckBox(tr("ssvep.acknowledge"))
        self.acknowledge.setChecked(config.acknowledge_flicker_risk)
        self.allow_draft = QCheckBox(tr("ssvep.allow_draft"))
        self.allow_draft.setChecked(config.allow_draft_hardware_config)
        for key, widget in (("field.participant", self.participant), ("field.name", self.name),
                            ("field.mode", self.mode), ("field.port", self.port),
                            ("field.screen", self.screen),
                            ("field.stimulus", self.stimulus), ("field.rest", self.rest),
                            ("field.repetitions", self.repetitions), ("field.refresh", label("60 Hz")),
                            ("field.speed", self.speed), ("field.save", picker),
                            ("field.channel_config", channel_picker)):
            form.addRow(tr(key), widget)
        form.addRow("", self.acknowledge)
        form.addRow("", self.allow_draft)
        section.layout.addLayout(form)
        section.layout.addWidget(label(tr("ssvep.refresh_note"), "muted"))
        self.estimate = label("", "estimate")
        section.layout.addWidget(self.estimate)
        section.layout.addWidget(label(tr("ssvep.formula"), "muted"))
        self.layout.addWidget(section)
        self.layout.addWidget(StaticTargets(tr))
        self.layout.addWidget(KeyValues([(tr("device.channels"), "OpenBCI Cyton · COM5 · 8 CH / 250 Hz"),
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
        channel_config = self.channel.text().strip()
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

    def _mode_changed(self):
        mode = CaptureMode(self.mode.currentData())
        demo = mode == CaptureMode.DEMO
        cyton = mode == CaptureMode.CYTON
        self.port.setEnabled(cyton)
        self.channel.setEnabled(cyton)
        self.allow_draft.setEnabled(cyton)
        self.screen.setEnabled(not demo)
        self.acknowledge.setEnabled(not demo)
        self.speed.setEnabled(demo)
        if not demo:
            self.speed.setCurrentIndex(self.speed.findData(1))

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
        if result.source == CaptureMode.DEMO:
            status_key = "result.simulated_saved" if result.persisted else "result.simulated"
            notice_key = "result.notice_written" if result.persisted else "result.notice"
            path_key = "result.path_written" if result.persisted else "result.path"
        else:
            status_key = (
                "result." + result.source.value + "_saved"
                if result.status == "completed"
                else "result." + result.status + "_saved"
            )
            notice_key = "result.recorded_notice"
            path_key = "result.recorded_path"
        sample_key = "result.samples" if result.source is CaptureMode.DEMO else "result.recorded_samples"
        values_key = "result.values" if result.source is CaptureMode.DEMO else "result.recorded_values"
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
        path.layout.addWidget(label(str(result.path), "path"))
        self.layout.addWidget(path)
        if result.error:
            error_section = Section(tr("result.error_details"))
            error_section.layout.addWidget(label(result.error, "error"))
            self.layout.addWidget(error_section)
        self.layout.addWidget(action(tr("app.workstation")+" / "+tr("app.datasets"), lambda: navigate("datasets"), True))
        self.layout.addWidget(action(tr("action.configure"), lambda: navigate("ssvep")))
        self.layout.addStretch()


class DatasetsPage(Page):
    def __init__(self, tr, datasets: tuple[Dataset, ...], latest: str | None, show_result, navigate):
        super().__init__(tr, tr("app.datasets"), tr("datasets.subtitle"))
        self.layout.addWidget(action(tr("nav.apps"), lambda: navigate("apps")))
        if not datasets:
            self.layout.addWidget(label(tr("datasets.empty"), "muted"))
        for dataset in reversed(datasets):
            item = Section(dataset.name)
            item.setObjectName("latestDataset" if dataset.id == latest else "section")
            if dataset.id == latest:
                status_key = "datasets.latest." + dataset.source.value
            elif dataset.source == CaptureMode.DEMO:
                status_key = "result.simulated_saved" if dataset.persisted else "result.simulated"
            else:
                status_key = "result." + dataset.source.value + "_saved"
            item.layout.addWidget(label(tr(status_key), "muted"))
            item.layout.addWidget(label(tr("datasets.details", participant=dataset.participant,
                duration=format_duration(dataset.recording_seconds), samples=f"{dataset.samples_per_channel:,}",
                events=dataset.event_count)))
            item.layout.addWidget(label(str(dataset.path), "path"))
            item.layout.addWidget(action(tr("action.summary"), lambda _checked=False, d=dataset: show_result(d)))
            self.layout.addWidget(item)
        self.layout.addStretch()


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
