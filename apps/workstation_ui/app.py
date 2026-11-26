"""Desktop shell and UI command routing; one gateway, one task timer."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import threading

from PySide6.QtCore import QObject, QSettings, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QIcon, QPalette, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QListWidget,
    QListWidgetItem, QComboBox, QStackedWidget, QScrollArea, QMessageBox, QFileDialog, QPushButton,
)

from .components import action, label
from .four_condition_page import FourConditionPage
from .gateway import CaptureGateway, CaptureConfig, CaptureMode, Phase, Dataset
from .i18n import Translator
from .monitor import AcquisitionMonitorWindow
from .ssvep_models import ModelsPage
from .v9_offline import V9OfflineController
from .quality_workspace import ChannelCheckPage, QualityWorkspaceController, QualityWorkspacePage
from .ssvep_replay import EEGProcessingPage, ReplayController
from neurostation_display import resolve_monitor_screen
from .pages import (
    HomePage, DevicesPage, AppsPage, SSVEPPage, TaskPage,
    ResultPage, DatasetsPage, DatasetTrashPage, InfoPage,
    DatasetSummaryPage, UserManagementPage, UserDialog, DiagnosticsPage,
)
from neurostation_diagnostics import DiagnosticStore
class _PreflightWorker(QObject):
    finished = Signal(object)

    def __init__(self, gateway: CaptureGateway, port: str = "AUTO"):
        super().__init__()
        self.gateway = gateway
        self.port = port

    @Slot()
    def run(self):
        try:
            report = self.gateway.preflight_cyton(port=self.port)
        except Exception as error:
            report = {"status": "failed", "error": str(error)}
        if not isinstance(report, dict):
            report = {"status": "failed"}
        self.finished.emit({"requested_port": self.port, **report})


class _ChannelCalibrationWorker(QObject):
    finished = Signal(object)
    samples = Signal(object)

    def __init__(self, gateway: CaptureGateway, channel_number: int, port: str, seconds: float | None):
        super().__init__()
        self.gateway = gateway
        self.channel_number = channel_number
        self.seconds = seconds
        self.port = port
        self.cancel_event = threading.Event()

    def request_stop(self) -> None:
        self.cancel_event.set()

    def _emit_samples(self, payload):
        if isinstance(payload, dict):
            self.samples.emit({"channel": self.channel_number, **payload})

    @Slot()
    def run(self):
        try:
            result = self.gateway.test_cyton_channel(
                self.channel_number,
                seconds=self.seconds,
                port=self.port,
                sample_callback=self._emit_samples,
                cancel_event=self.cancel_event,
            )
        except Exception as error:
            self.finished.emit(error)
            return
        self.finished.emit(result)


class _ImportWorker(QObject):
    finished = Signal(object)

    def __init__(self, gateway: CaptureGateway, source_root: Path):
        super().__init__()
        self.gateway = gateway
        self.source_root = source_root

    @Slot()
    def run(self):
        try:
            report = self.gateway.import_openbci_recordings(self.source_root)
        except Exception as error:  # forwarded to the GUI thread for display
            self.finished.emit(error)
            return
        self.finished.emit(report)


NAVIGATION = (
    "home", "devices", "users", "apps", "quality_details", "processing", "models", "datasets", "trash", "openbci",
    "integrations", "diagnostics",
)


def _has_only_clean_packet_timestamp_jitter(report: dict) -> bool:
    """Return whether a warning is only host-arrival timestamp jitter.

    Cyton packet continuity is the meaningful loss signal. Host timestamps can
    arrive in bursts over USB/radio without any board samples being lost.
    """
    checks = report.get("checks")
    if not isinstance(checks, list):
        return False

    timestamp_warning = False
    for check in checks:
        if not isinstance(check, dict):
            return False
        if str(check.get("status") or "") == "passed":
            continue
        if str(check.get("name") or "") != "timestamps":
            return False
        if str(check.get("status") or "") != "warning":
            return False
        metrics = check.get("metrics")
        if not isinstance(metrics, dict):
            return False
        timestamp_warning = True
        if not metrics.get("packet_sequence_available"):
            return False
        if any(
            int(metrics.get(key, 0) or 0) != 0
            for key in (
                "packet_loss_count",
                "packet_sequence_mismatch_count",
                "packet_duplicate_count",
                "timestamp_non_monotonic_count",
            )
        ):
            return False
        if bool(metrics.get("timestamp_missing", False)):
            return False

    return timestamp_warning


def product_stylesheet(dark: bool) -> str:
    background, surface, line, ink, muted, accent, active = (
        ("#151a20", "#1b222a", "#34434e", "#e0e9ef", "#a6b6c1", "#69c8d3", "#233b41")
        if dark else ("#ffffff", "#f5f7f8", "#dfe6e9", "#20313d", "#677781", "#087c88", "#e4f3f3")
    )
    return f"""
        QWidget {{ color: {ink}; background-color: {background}; font-size: 13px; font-family: "Microsoft YaHei UI", "Microsoft YaHei", "Noto Sans CJK SC", "PingFang SC", sans-serif; }}
        QMainWindow {{ background: {background}; }}
        QLabel {{ background: transparent; }}
        QLabel#pageTitle {{ font-size: 23px; font-weight: 500; }}
        QLabel#sectionTitle {{ font-weight: 500; }}
        QLabel#muted, QLabel#path {{ color: {muted}; font-size: 12px; }}
        QLabel#notice, QLabel#qualitySourceBanner {{ background: {surface}; padding: 12px; border-left: 3px solid #b58b3a; }}
        QLabel#error {{ color: {'#f39393' if dark else '#b44242'}; }}
        QLabel#estimate {{ font-size: 15px; color: {accent}; }}
        QLabel#channelCheckBadge {{ background: {surface}; border: 1px solid {line}; border-radius: 6px; padding: 9px 12px; color: {muted}; }}
        QLabel#channelCheckBadge[state="passed"] {{ color: {'#9fdfbb' if dark else '#17633c'}; }}
        QLabel#channelCheckBadge[state="warning"] {{ color: {'#ffbe82' if dark else '#9a4509'}; }}
        QLabel#channelGuideChannel {{ background: {surface}; border-radius: 4px; padding: 2px 5px; font-weight: 600; }}
        QLabel#channelGuideChannel[selected="true"] {{ background: {active}; color: {accent}; }}
        QLabel#countdown {{ font-size: 64px; color: {accent}; }}
        QLabel#staticTarget {{ background: {background}; border: 1px solid {line}; border-radius: 5px; font-size: 18px; }}
        QFrame#section {{ border: 1px solid {line}; border-radius: 6px; }}
        QFrame#latestDataset {{ background: {active}; border: 1px solid {accent}; border-radius: 6px; }}
        QFrame#latestDataset QWidget {{ background: transparent; }}
        QWidget#sidebar {{ background: {surface}; border-right: 1px solid {line}; }}
        QWidget#sidebar QLabel, QListWidget {{ background: transparent; }}
        QListWidget {{ border: 0; outline: 0; }}
        QListWidget::item {{ border-radius: 5px; padding: 12px 7px; margin-bottom: 4px; }}
        QListWidget::item:selected {{ background: {active}; color: {accent}; }}
        QPushButton {{ border: 1px solid {line}; border-radius: 5px; padding: 7px 12px; background: {background}; }}
        QPushButton:hover {{ background: {surface}; }}
        QPushButton:disabled {{ color: {muted}; background: {surface}; }}
        QPushButton:focus {{ border: 2px solid {accent}; }}
        QPushButton#primaryButton, QPushButton#startTaskButton {{ background: {accent}; color: {background}; border-color: {accent}; }}
        QToolButton#appTile {{ border: 1px solid {line}; border-radius: 12px; padding: 20px 10px; background: {background}; }}
        QToolButton#appTile:hover {{ background: {active}; }}
        QToolButton#appTile:disabled {{ color: {muted}; background: {surface}; }}
        QLineEdit, QSpinBox, QComboBox {{ border: 1px solid {line}; border-radius: 4px; padding: 7px; min-height: 22px; }}
        QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{ border-color: {accent}; }}
        QScrollArea {{ border: 0; }}
        QProgressBar {{ background: {surface}; border: 0; border-radius: 3px; min-height: 20px; text-align: center; }}
        QProgressBar::chunk {{ background: {active}; border-radius: 3px; }}
    """


class MainWindow(QMainWindow):
    def __init__(
        self,
        gateway: CaptureGateway,
        locale: str | None = "zh-CN",
        timer_enabled: bool = True,
        persist_settings: bool = False,
        save_directory_override: Path | None = None,
        settings: QSettings | None = None,
        initial_mode: CaptureMode | None = None,
        initial_port: str | None = None,
        diagnostic_store: DiagnosticStore | None = None,
    ):
        super().__init__()
        self.gateway = gateway
        self.diagnostic_store = diagnostic_store or DiagnosticStore()
        self.diagnostic_store.install_exception_hook()
        self.diagnostic_store.info(
            "ui",
            "workstation window initialized",
            context={"gateway": type(self.gateway).__name__},
        )
        self.persist_settings = persist_settings
        self.settings = settings or QSettings("NeuroStation", "NeuroStation")
        chosen_locale = locale
        if chosen_locale is None and persist_settings:
            chosen_locale = str(self.settings.value("ui/language", "zh-CN"))
        self.tr = Translator(chosen_locale or "zh-CN")
        self.timer_enabled = timer_enabled
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.poll)
        self.cancel_shortcut = QShortcut(QKeySequence("Esc"), self)
        self.cancel_shortcut.activated.connect(self.cancel_task)
        self.current_page = "apps"
        self.draft_config = self.gateway.config
        if persist_settings:
            self.draft_config = replace(self.draft_config, realtime_model_id=str(
                self.settings.value('capture/realtime_model_id', self.draft_config.realtime_model_id)))
        if initial_mode is not None:
            self.draft_config = replace(
                self.draft_config,
                mode=CaptureMode.CYTON,
                port=(initial_port or self.draft_config.port),
            )
        if save_directory_override is not None:
            self.draft_config = replace(
                self.draft_config,
                save_directory=save_directory_override.expanduser().resolve(),
            )
        elif persist_settings:
            saved_directory = str(self.settings.value("capture/save_directory", ""))
            if saved_directory and Path(saved_directory).expanduser().is_absolute():
                self.draft_config = replace(
                    self.draft_config,
                    save_directory=Path(saved_directory).expanduser(),
                )
        self.result: Dataset | None = None
        self._showing_dataset_summary = False
        self.latest_dataset: str | None = None
        self.import_thread: QThread | None = None
        self.import_worker: _ImportWorker | None = None
        self.preflight_thread: QThread | None = None
        self.preflight_worker: _PreflightWorker | None = None
        self._channel_check_report: dict | None = None
        self._close_after_workers = False
        self.channel_calibration_thread: QThread | None = None
        self.channel_calibration_worker: _ChannelCalibrationWorker | None = None
        self.quality_controller = QualityWorkspaceController(self)
        self.quality_controller.changed.connect(self._update_quality_banner)
        self.quality_controller.busy_changed.connect(lambda _busy: self._update_controls())
        self.replay_controller = ReplayController(self)
        self.replay_controller.busy_changed.connect(lambda _busy: self._update_controls())
        self.offline_controller = V9OfflineController(self)
        self.offline_controller.busy_changed.connect(lambda _busy: self._update_controls())
        self._channel_calibration_index: int | None = None
        self._pending_ssvep: tuple[CaptureConfig, float] | None = None
        self._preflight_ready = False
        self.import_status = ""
        self._dataset_search = ""
        self._dataset_source = ""
        self._dataset_status = ""
        self._dataset_signal = ""
        self.monitor_window: AcquisitionMonitorWindow | None = None
        self._monitor_suppressed = False
        self.pages: dict[str, QWidget] = {}
        self.screens: dict[str, QScrollArea] = {}
        self.sidebar: QWidget | None = None
        self.sidebar_summary = None
        self.platform_label = None
        self.resize(1160, 850)
        self.setMinimumSize(320, 560)
        icon_path = Path(__file__).resolve().parents[2] / "assets" / "neurostation.svg"
        if icon_path.is_file():
            self.setWindowIcon(QIcon(str(icon_path)))
        self._build_shell()
        QApplication.instance().screenRemoved.connect(self._monitor_screen_removed)
        if getattr(self.gateway, "auto_import_default", False):
            QTimer.singleShot(0, self._auto_import_default)
        if persist_settings:
            geometry = self.settings.value("window/geometry")
            if geometry is not None:
                self.restoreGeometry(geometry)

    def _build_shell(self):
        self.setWindowTitle(self.tr("app.title"))
        application = QApplication.instance()
        dark = application.palette().color(QPalette.ColorRole.Window).lightness() < 128
        self.setStyleSheet(product_stylesheet(dark))
        previous = self.takeCentralWidget()
        if previous:
            previous.deleteLater()
        outer = QWidget()
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        chrome = QHBoxLayout()
        chrome.setContentsMargins(18, 12, 18, 12)
        chrome.addWidget(label("NeuroStation", "sectionTitle"))
        chrome.addStretch()
        self.mode_banner_label = label("", "muted")
        chrome.addWidget(self.mode_banner_label)
        self.language_selector = language = QComboBox()
        language.addItem("简体中文", "zh-CN")
        language.addItem("English", "en-US")
        language.setCurrentIndex(0 if self.tr.locale == "zh-CN" else 1)
        language.currentIndexChanged.connect(lambda index: self.change_language(language.itemData(index)))
        chrome.addWidget(language)
        outer_layout.addLayout(chrome)
        body = QHBoxLayout()
        body.setSpacing(0)
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(186)
        self.sidebar = sidebar
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(12, 18, 12, 16)
        side_layout.addWidget(label(self.tr("app.workspace"), "muted"))
        self.navigation = QListWidget()
        self.navigation.setAccessibleName(self.tr("app.workspace"))
        for key in NAVIGATION:
            item = QListWidgetItem(self.tr("nav."+key))
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setToolTip(self.tr("nav." + key))
            self.navigation.addItem(item)
        self.navigation.currentRowChanged.connect(self._nav_selected)
        side_layout.addWidget(self.navigation, 1)
        device = self.gateway.device
        self.device_summary = label(
            f"{device.name}\n{device.port} · {device.channels} CH · {device.sample_rate} Hz",
            "muted",
        )
        self.sidebar_summary = self.device_summary
        side_layout.addWidget(self.device_summary)
        self.platform_label = label("Windows / Linux / macOS", "muted")
        side_layout.addWidget(self.platform_label)
        body.addWidget(sidebar)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        breadcrumb = QHBoxLayout()
        breadcrumb.setContentsMargins(20, 10, 20, 0)
        breadcrumb.addWidget(action(self.tr("app.workstation"), lambda: self.navigate("home")))
        breadcrumb.addWidget(label("/"))
        self.breadcrumb_title = label("", "muted")
        breadcrumb.addWidget(self.breadcrumb_title)
        breadcrumb.addStretch()
        content_layout.addLayout(breadcrumb)
        self.active_banner = QWidget()
        banner_layout = QHBoxLayout(self.active_banner)
        banner_layout.setContentsMargins(24, 8, 24, 0)
        banner_layout.addWidget(label(self.tr("app.active_task"), "muted"), 1)
        banner_layout.addWidget(action(self.tr("app.return_task"), self.return_to_task))
        content_layout.addWidget(self.active_banner)
        self.stack = QStackedWidget()
        content_layout.addWidget(self.stack, 1)
        body.addWidget(content, 1)
        outer_layout.addLayout(body, 1)
        self.setCentralWidget(outer)
        self.pages = {}
        self.screens = {}
        self._replace_page("home", HomePage(self.tr, self.navigate, self.gateway.device))
        self._replace_page(
            "devices",
            DevicesPage(
                self.tr,
                self.navigate,
                self.preflight_cyton,
                self.calibrate_cyton_channel,
                self.save_channel_calibration,
                self._channel_config_value(),
                self._protocol_status(),
                channel_test_stop=self.stop_calibrate_cyton_channel,
                channel_preview=lambda index: self.calibrate_cyton_channel(index, seconds=None),
                device_info=self.gateway.device,
            ),
        )
        self._replace_page("users", self._build_users_page())
        self._replace_page("apps", AppsPage(self.tr, self.navigate, self.gateway.device))
        self._replace_page("quality", ChannelCheckPage(
            self.tr, self.preflight_cyton, lambda: self.navigate("quality_details")))
        self._replace_page("quality_details", QualityWorkspacePage(
            self.tr, self.quality_controller, lambda: self.navigate("quality")))
        detail = SSVEPPage(self.tr, self.draft_config, self.navigate, self.gateway.users)
        models = ModelsPage(self.tr)
        self._replace_page('models', models)
        models.apply_requested.connect(self._use_model)
        models.imported.connect(lambda _key: detail.refresh_realtime_models(detail.realtime_model.currentData()))
        models.busy_changed.connect(self._update_controls)
        detail.realtime_model.currentIndexChanged.connect(self._remember_model)
        detail.start_requested.connect(self.start_ssvep)
        detail.serial_scan_requested.connect(self.scan_serial_ports)
        detail.create_user_requested.connect(self.create_user_from_capture)
        detail.refresh_users_requested.connect(self.refresh_users)
        detail.mode.currentIndexChanged.connect(lambda _index: self._capture_form_changed())
        detail.port.textChanged.connect(lambda _text: self._capture_form_changed())
        self._replace_page("ssvep", detail)
        self._replace_page("ssvep_four_condition", FourConditionPage(self.tr, self.navigate))
        task = TaskPage(self.tr)
        task.cancel_requested.connect(self.cancel_task)
        task.monitor_requested.connect(self.show_acquisition_monitor)
        self._replace_page("task", task)
        # The task poller owns the consuming live-preview reader. Both windows
        # receive the same batch, even when the task page is hidden.
        self._replace_page(
            "openbci",
            InfoPage(
                self.tr,
                "openbci",
                self.gateway.openbci_status,
                self.launch_openbci,
            ),
        )
        self._replace_page("integrations", InfoPage(self.tr, "integrations"))
        self._replace_page(
            "diagnostics",
            DiagnosticsPage(
                self.tr,
                lambda: self.diagnostic_store.build_report(self.gateway),
                lambda: self.diagnostic_store.events(200),
                self._refresh_diagnostics,
                self._export_diagnostics,
                self._clear_diagnostics,
            ),
        )
        self.navigate(self.current_page)
        self._apply_compact_layout()
        self._update_controls()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_compact_layout()

    def _apply_compact_layout(self):
        """Keep the navigation usable on narrow laptop and tiled windows."""

        if self.sidebar is None:
            return
        compact = self.width() < 720
        self.sidebar.setFixedWidth(68 if compact else 186)
        if self.sidebar_summary is not None:
            self.sidebar_summary.setVisible(not compact)
        if self.platform_label is not None:
            self.platform_label.setVisible(not compact)
        if compact:
            self.navigation.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.navigation.setToolTip(self.tr("app.workspace"))
        else:
            self.navigation.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self.navigation.setToolTip("")

    def _replace_page(self, key: str, page: QWidget):
        old = self.screens.get(key)
        if old:
            self.stack.removeWidget(old)
            old.deleteLater()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        self.stack.addWidget(scroll)
        self.screens[key] = scroll
        self.pages[key] = page

    def _nav_selected(self, row: int):
        if 0 <= row < len(NAVIGATION):
            self.navigate(NAVIGATION[row])

    def navigate(self, key: str):
        if key == "ssvep" and self.gateway.snapshot.active and self.gateway.snapshot.protocol == "ssvep":
            key = "task"
        if key == "datasets":
            page = DatasetsPage(
                self.tr,
                self.gateway.datasets,
                self.latest_dataset,
                self.show_dataset,
                self.navigate,
                import_busy=self.import_thread is not None or self._analysis_busy(),
                import_status=self.import_status,
                import_directory=self._last_import_directory(),
                search_text=self._dataset_search,
                source_value=self._dataset_source,
                status_value=self._dataset_status,
                signal_value=self._dataset_signal,
                callbacks={
                    "rename": self.rename_dataset,
                    "delete": lambda dataset_id: self._offline_guarded_mutation(self.gateway.delete_dataset, dataset_id),
                    "refresh": self.refresh_datasets,
                },
            )
            page.import_requested.connect(self._start_import)
            page.filter_changed.connect(self._dataset_filters_changed)
            self._replace_page(key, page)
        elif key == "trash":
            page = DatasetTrashPage(
                self.tr,
                self.gateway.trashed_datasets,
                self.navigate,
                search_text=self._dataset_search,
                source_value=self._dataset_source,
                status_value=self._dataset_status,
                signal_value=self._dataset_signal,
                callbacks={
                    "restore": lambda dataset_id: self._offline_guarded_mutation(self.gateway.restore_dataset, dataset_id),
                    "purge": lambda dataset_id: self._offline_guarded_mutation(self.gateway.purge_dataset, dataset_id),
                    "refresh": self.refresh_datasets,
                },
            )
            page.filter_changed.connect(self._dataset_filters_changed)
            self._replace_page(key, page)
        elif key == 'models':
            models = self.pages['models']
            if models.process is None:
                models.refresh(self.pages['ssvep'].realtime_model.currentData())
        elif key == "processing":
            self._replace_page(key, EEGProcessingPage(self.tr, self.gateway.datasets, self.replay_controller,
                                                     self.configure_continuous_capture))
        elif key == "users":
            self._replace_page(key, self._build_users_page())
        elif key == "result":
            if self.result is None:
                key = "apps"
            else:
                page = DatasetSummaryPage if self._showing_dataset_summary else ResultPage
                self._replace_page(key, page(
                    self.tr, self.result, self.navigate, offline_controller=self.offline_controller,
                    replay_controller=self.replay_controller, quality_callback=self.open_dataset_quality,
                ))
        elif key == "diagnostics":
            self._refresh_diagnostics()
        self.current_page = key
        self.stack.setCurrentWidget(self.screens[key])
        nav_key = "apps" if key in ("ssvep", "ssvep_four_condition", "task", "result") else key
        if key == "quality":
            nav_key = "apps"
        elif key == "result" and self._showing_dataset_summary:
            nav_key = "datasets"
        self.breadcrumb_title.setText(self.pages[key].title_label.text())
        self.navigation.blockSignals(True)
        self.navigation.setCurrentRow(NAVIGATION.index(nav_key))
        self.navigation.blockSignals(False)
        self._update_controls()

    def _use_model(self, key):
        if self._offline_capture_busy() or self._analysis_busy():
            return
        self.pages['ssvep'].refresh_realtime_models(key)
        self._remember_model()
        self.configure_continuous_capture()

    def _remember_model(self):
        if self.persist_settings:
            self.settings.setValue('capture/realtime_model_id', self.pages['ssvep'].realtime_model.currentData())

    def configure_continuous_capture(self):
        if not self.gateway.snapshot.active:
            self.pages["ssvep"].continuous.setChecked(True)
        self.navigate("ssvep")

    def _dataset_filters_changed(self, search: str, source: str, status: str, signal: str):
        self._dataset_signal = signal
        self._dataset_search = search
        self._dataset_source = source
        self._dataset_status = status

    def rename_dataset(self, dataset_id: str, name: str) -> Dataset:
        dataset = self._offline_guarded_mutation(self.gateway.rename_dataset, dataset_id, name)
        if self.result is not None and self.result.id == dataset_id:
            self.result = dataset
        return dataset

    def refresh_datasets(self):
        self.gateway.refresh_datasets()
        if self.current_page in ("datasets", "trash"):
            self.navigate(self.current_page)

    def change_language(self, locale: str):
        if locale == self.tr.locale or self.channel_calibration_thread is not None:
            return
        four_condition_draft = self.pages["ssvep_four_condition"].draft_state()
        self.draft_config = self.pages["ssvep"].config()
        if self.persist_settings:
            self.settings.setValue("ui/language", locale)
        self.tr = Translator(locale)
        monitor_suppressed = self._monitor_suppressed
        self._dispose_acquisition_monitor()
        self._monitor_suppressed = monitor_suppressed
        self._build_shell()
        self.pages["ssvep_four_condition"].restore_draft(four_condition_draft)

    def _error(self, error: Exception):
        self.diagnostic_store.exception(error, source="ui")
        QMessageBox.warning(self, self.tr("error.title"), self.tr(str(error)))

    def _refresh_diagnostics(self):
        page = self.pages.get("diagnostics")
        if isinstance(page, DiagnosticsPage):
            page.refresh(
                self.diagnostic_store.build_report(self.gateway),
                self.diagnostic_store.events(200),
            )

    def _export_diagnostics(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("diagnostics.export_title"),
            "neurostation-diagnostics.json",
            "JSON (*.json)",
        )
        if not path:
            return
        try:
            output = self.diagnostic_store.export_report(Path(path), self.gateway)
        except Exception as error:
            self._error(error)
            return
        QMessageBox.information(
            self,
            self.tr("diagnostics.export_title"),
            self.tr("diagnostics.export_done", path=str(output)),
        )

    def _clear_diagnostics(self):
        self.diagnostic_store.clear()
        self._refresh_diagnostics()

    def _record(self, source: str, message: str, **context):
        self.diagnostic_store.info(source, message, context=context)

    def _start_timer(self):
        if self.timer_enabled and not self.timer.isActive():
            self.timer.start()

    def start_ssvep(self, config: CaptureConfig, speed: float, *, _preflight_ready: bool = False):
        if self._analysis_busy() or self.pages["models"].process is not None:
            self.pages["ssvep"].error.setText(self.tr("v9.busy_mutation"))
            return
        if not _preflight_ready and (self.gateway.snapshot.active or self.preflight_thread is not None
                                     or self.channel_calibration_thread is not None):
            return
        self._record(
            "capture",
            "ssvep start requested",
            mode=CaptureMode(config.mode).value,
            participant=config.participant,
            preflight_ready=_preflight_ready,
        )
        if CaptureMode(config.mode) == CaptureMode.CYTON and not _preflight_ready:
            self._pending_ssvep = (config, speed)
            self._preflight_ready = False
            self.preflight_cyton()
            return
        try:
            self.gateway.start_ssvep(config, speed)
        except (ValueError, RuntimeError) as error:
            self._error(error)
            return
        self.draft_config = config
        self.quality_controller.use_live()
        self._dispose_acquisition_monitor()
        self._monitor_suppressed = False
        self._preflight_ready = False
        if self.persist_settings:
            self.settings.setValue("capture/save_directory", str(config.save_directory))
        task_page = self.pages.get("task")
        if task_page is not None and hasattr(task_page, "set_quality_warning"):
            task_page.set_quality_warning("")
        self.navigate("task")
        self._start_timer()
        self._record("capture", "ssvep task started", mode=CaptureMode(config.mode).value)

    def preflight_cyton(self):
        if (self._analysis_busy() or self.gateway.snapshot.active or self.preflight_thread is not None
                or self.channel_calibration_thread is not None):
            return
        self._channel_check_report = None
        page = self.pages.get("devices")
        if page is not None and hasattr(page, "preflight_status"):
            page.preflight_status.setText(self.tr("device.preflight_running"))
        requested_port = "AUTO"
        if self._pending_ssvep is not None:
            requested_port = self._pending_ssvep[0].port or "AUTO"
        else:
            ssvep_page = self.pages.get("ssvep")
            if ssvep_page is not None and hasattr(ssvep_page, "port"):
                requested_port = ssvep_page.port.text().strip() or "AUTO"
        self.preflight_thread = QThread(self)
        self.preflight_worker = _PreflightWorker(self.gateway, requested_port)
        self.preflight_worker.moveToThread(self.preflight_thread)
        self.preflight_thread.started.connect(self.preflight_worker.run)
        self.preflight_worker.finished.connect(self._preflight_finished)
        self.preflight_worker.finished.connect(self.preflight_thread.quit)
        self.preflight_worker.finished.connect(self.preflight_worker.deleteLater)
        self.preflight_thread.finished.connect(self.preflight_thread.deleteLater)
        self.preflight_thread.finished.connect(self._preflight_thread_finished)
        self.preflight_thread.start()
        self._update_controls()

    def _channel_config_value(self) -> dict:
        loader = getattr(self.gateway, "load_channel_calibration", None)
        if callable(loader):
            try:
                value = loader()
                if isinstance(value, dict):
                    return value
            except Exception:
                pass
        path = getattr(self.gateway, "channel_config_path", None)
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _protocol_status(self) -> str:
        path = getattr(self.gateway, "protocol_path", None)
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return "unknown"
        return str(value.get("status") or "unknown") if isinstance(value, dict) else "unknown"

    def calibrate_cyton_channel(self, channel_index: int, *, seconds: float | None = 3.0):
        if self._analysis_busy() or self.gateway.snapshot.active or self.preflight_thread is not None:
            return
        if self.channel_calibration_thread is not None:
            return
        page = self.pages.get("devices")
        if page is None:
            return
        try:
            channel_index = int(channel_index)
        except (TypeError, ValueError):
            page.calibration_status.setText(self.tr("validation.calibration_channel"))
            return
        if not 0 <= channel_index < 8:
            page.calibration_status.setText(self.tr("validation.calibration_channel"))
            return
        port = page.calibration_port.text().strip() or "AUTO"
        self._channel_calibration_index = channel_index
        page.set_channel_test_busy(channel_index, True)
        page.calibration_status.setText(
            self.tr("device.calibration_running", channel=channel_index + 1)
        )
        self.channel_calibration_thread = QThread(self)
        self.channel_calibration_worker = _ChannelCalibrationWorker(
            self.gateway,
            channel_index + 1,
            port,
            seconds,
        )
        self.channel_calibration_worker.moveToThread(self.channel_calibration_thread)
        self.channel_calibration_thread.started.connect(self.channel_calibration_worker.run)
        self.channel_calibration_worker.samples.connect(
            page.append_channel_calibration_samples
        )
        self.channel_calibration_worker.finished.connect(self._channel_calibration_finished)
        self.channel_calibration_worker.finished.connect(self.channel_calibration_thread.quit)
        self.channel_calibration_worker.finished.connect(self.channel_calibration_worker.deleteLater)
        self.channel_calibration_thread.finished.connect(self.channel_calibration_thread.deleteLater)
        self.channel_calibration_thread.finished.connect(self._channel_calibration_thread_finished)
        self.channel_calibration_thread.start()
        self._update_controls()

    def stop_calibrate_cyton_channel(self):
        worker = self.channel_calibration_worker
        page = self.pages.get("devices")
        if worker is None:
            return
        worker.request_stop()
        if page is not None:
            page.set_channel_test_stopping(True)

    def _channel_calibration_finished(self, result):
        page = self.pages.get("devices")
        index = self._channel_calibration_index
        if page is None or index is None:
            return
        if isinstance(result, Exception):
            result = {
                "status": "failed",
                "channel": index + 1,
                "detail": str(result),
                "metrics": {},
            }
        if not isinstance(result, dict):
            result = {
                "status": "failed",
                "channel": index + 1,
                "detail": self.tr("validation.calibration_result"),
                "metrics": {},
            }
        if (self.channel_calibration_worker is not None and self.channel_calibration_worker.seconds is None
                and result.get("status") == "cancelled"):
            page.calibration_status.setText(self.tr("device.calibration_status.cancelled"))
        else:
            page.set_channel_test_result(index, result)
        self._record(
            "hardware",
            "cyton channel calibration finished",
            channel=index + 1,
            status=result.get("status", "unknown"),
        )

    def _channel_calibration_thread_finished(self):
        page = self.pages.get("devices")
        if page is not None:
            page.set_channel_test_busy(-1, False)
        self.channel_calibration_thread = None
        self.channel_calibration_worker = None
        self._channel_calibration_index = None
        if self._close_after_workers:
            self.close()
        else:
            self._update_controls()

    def save_channel_calibration(self, calibration: dict):
        try:
            result = self.gateway.save_channel_calibration(calibration)
        except (ValueError, RuntimeError, OSError) as error:
            self._error(error)
            return
        self._record(
            "hardware",
            "formal Cyton channel calibration saved",
            channel_config_path=result.get("channel_config_path", ""),
            protocol_path=result.get("protocol_path", ""),
        )
        page = self.pages.get("devices")
        if page is not None:
            page.set_calibration_saved(result)
            page.set_protocol_status(self._protocol_status())
        ssvep_page = self.pages.get("ssvep")
        if ssvep_page is not None and hasattr(ssvep_page, "allow_draft"):
            ssvep_page.allow_draft.setChecked(False)
        self.draft_config = replace(self.draft_config, allow_draft_hardware_config=False)
        self._refresh_diagnostics()

    def _preflight_finished(self, report: dict):
        self._record(
            "preflight",
            "cyton preflight finished",
            status=report.get("status", "unknown"),
            checks=len(report.get("checks", [])) if isinstance(report.get("checks"), list) else 0,
        )
        page = self.pages.get("devices")
        status = str(report.get("status", "unknown"))
        detail = str(report.get("error") or "")
        if not detail:
            checks = report.get("checks", [])
            if isinstance(checks, list):
                details: list[str] = []
                for check in checks:
                    if not isinstance(check, dict) or str(check.get("status")) == "passed":
                        continue
                    name = str(check.get("name") or "")
                    metrics = check.get("metrics")
                    if name == "timestamps" and isinstance(metrics, dict):
                        try:
                            details.append(self.tr(
                                "device.preflight_warning_timestamps",
                                count=int(metrics.get("timestamp_gap_count", 0) or 0),
                                ratio=round(float(metrics.get("timestamp_gap_ratio", 0.0) or 0.0) * 100, 1),
                                max_ms=round(float(metrics.get("timestamp_diff_max_s", 0.0) or 0.0) * 1000, 1),
                            ))
                            if metrics.get("packet_sequence_available"):
                                lost = int(metrics.get("packet_loss_count", 0) or 0)
                                mismatch = int(metrics.get("packet_sequence_mismatch_count", 0) or 0)
                                duplicate = int(metrics.get("packet_duplicate_count", 0) or 0)
                                if lost or mismatch or duplicate:
                                    details.append(self.tr(
                                        "device.preflight_packet_loss",
                                        lost=lost,
                                        mismatch=mismatch,
                                        duplicate=duplicate,
                                    ))
                                else:
                                    details.append(self.tr("device.preflight_packet_sequence_ok"))
                            continue
                        except (TypeError, ValueError):
                            pass
                    if name == "packet_sequence" and isinstance(metrics, dict):
                        try:
                            details.append(self.tr(
                                "device.preflight_packet_loss",
                                lost=int(metrics.get("packet_loss_count", 0) or 0),
                                mismatch=int(metrics.get("packet_sequence_mismatch_count", 0) or 0),
                                duplicate=int(metrics.get("packet_duplicate_count", 0) or 0),
                            ))
                            continue
                        except (TypeError, ValueError):
                            pass
                    if name == "channels":
                        warning_channels = metrics.get("warning_channels") if isinstance(metrics, dict) else None
                        channels = ", ".join(
                            f"CH{int(channel)}" for channel in warning_channels
                        ) if isinstance(warning_channels, list) and warning_channels else ""
                        details.append(self.tr(
                            "device.preflight_warning_channels",
                            channels=channels,
                        ))
                        continue
                    if check.get("detail"):
                        details.append(str(check["detail"]))
                detail = ("；" if self.tr.locale == "zh-CN" else "; ").join(details)
        status_text = self.tr(f"device.preflight_status.{status}")
        selected_port = str(report.get("selected_port") or "").strip()
        ssvep_page = self.pages.get("ssvep")
        if selected_port and ssvep_page is not None and hasattr(ssvep_page, "port"):
            ssvep_page.port.setText(selected_port)
            ssvep_page.port_status.setText(self.tr(
                "field.port_detected",
                ports=selected_port,
                selected=selected_port,
            ))
        self._channel_check_report = report
        if page is not None and hasattr(page, "preflight_status"):
            page.preflight_status.setText(
                self.tr("device.preflight_result", status=status_text, detail=detail)
            )
        pending = self._pending_ssvep
        self._pending_ssvep = None
        if status == "passed":
            self._preflight_ready = True
            if pending is not None:
                config, speed = pending
                if selected_port:
                    config = replace(config, port=selected_port)
                    self.draft_config = config
                self.start_ssvep(config, speed, _preflight_ready=True)
        elif pending is not None:
            self._preflight_ready = False
            if status == "warning":
                warning_detail = detail or self.tr(f"device.preflight_status.{status}")
                only_clean_timestamp_jitter = _has_only_clean_packet_timestamp_jitter(report)
                # OpenBCI GUI keeps streaming when packet/timestamp quality is
                # imperfect and surfaces loss statistics while acquisition
                # continues. Match that behavior for timestamp-only warnings.
                # More severe degraded results still require review; failed
                # handshake, empty stream, or invalid data remain hard stops.
                # Draft protocol/channel validation is enforced by the
                # gateway independently of this quality-warning path.
                message_key = (
                    "validation.preflight_warning_allowed"
                    if pending[0].allow_draft_hardware_config
                    else "validation.preflight_warning_started"
                )
                message = self.tr(
                    message_key,
                    port=selected_port or "AUTO",
                    detail=warning_detail,
                )
                self._preflight_ready = True
                config, speed = pending
                if selected_port:
                    config = replace(config, port=selected_port)
                    self.draft_config = config
                if only_clean_timestamp_jitter:
                    self._record(
                        "preflight",
                        "host timestamp jitter accepted; acquisition continues",
                        port=selected_port or "AUTO",
                        detail=warning_detail,
                        suppressed_ui_warning=True,
                        technical_validation=bool(config.allow_draft_hardware_config),
                    )
                    self.start_ssvep(config, speed, _preflight_ready=True)
                else:
                    self._record(
                        "preflight",
                        "quality warning accepted; acquisition continues",
                        port=selected_port or "AUTO",
                        detail=warning_detail,
                        technical_validation=bool(config.allow_draft_hardware_config),
                    )
                    self.start_ssvep(config, speed, _preflight_ready=True)
                    task_page = self.pages.get("task")
                    if task_page is not None and hasattr(task_page, "set_quality_warning"):
                        task_page.set_quality_warning(message)
                    if ssvep_page is not None and hasattr(ssvep_page, "error"):
                        ssvep_page.error.setText(message)
            elif status == "degraded":
                message = self.tr(
                    "validation.preflight_degraded_blocked",
                    port=selected_port or "AUTO",
                    detail=detail or self.tr("device.preflight_status.degraded"),
                )
                self._record(
                    "preflight",
                    "degraded quality requires review before acquisition",
                    port=selected_port or "AUTO",
                    detail=detail,
                )
                if ssvep_page is not None and hasattr(ssvep_page, "error"):
                    ssvep_page.error.setText(message)
            else:
                self._error(ValueError("validation.preflight_failed"))
        self._update_controls()

    def _preflight_thread_finished(self):
        self.preflight_thread = None
        self.preflight_worker = None
        if self._close_after_workers:
            self.close()
        else:
            self._update_controls()

    def scan_serial_ports(self):
        try:
            ports = self.gateway.scan_serial_ports()
        except Exception as error:
            self._error(error)
            return
        self._record("hardware", "serial ports scanned", count=len(ports))
        page = self.pages.get("ssvep")
        if page is not None and hasattr(page, "set_detected_ports"):
            page.set_detected_ports(ports)
            self._capture_form_changed()

    def _capture_form_changed(self):
        if self.gateway.snapshot.active:
            return
        page = self.pages.get("ssvep")
        if page is None or not hasattr(page, "config"):
            return
        try:
            self.draft_config = page.config()
        except (TypeError, ValueError):
            return
        self._update_controls()

    def _build_users_page(self):
        callbacks = {
            "next_id": self.gateway.next_user_id,
            "add": self.gateway.add_user,
            "update": lambda *args, **kwargs: self._offline_guarded_mutation(self.gateway.update_user, *args, **kwargs),
            "delete": lambda *args, **kwargs: self._offline_guarded_mutation(self.gateway.delete_user, *args, **kwargs),
            "restore": lambda *args, **kwargs: self._offline_guarded_mutation(self.gateway.restore_user, *args, **kwargs),
            "purge": lambda *args, **kwargs: self._offline_guarded_mutation(self.gateway.purge_user, *args, **kwargs),
            "refresh": self.refresh_users,
        }
        page = UserManagementPage(
            self.tr,
            self.gateway.users,
            self.gateway.trashed_users,
            callbacks,
        )
        page.export_requested.connect(self.export_public_users)
        page.table.itemSelectionChanged.connect(self._update_controls)
        page.trash_table.itemSelectionChanged.connect(self._update_controls)
        return page

    def export_public_users(self):
        path, _ = QFileDialog.getSaveFileName(
            self, self.tr("users.export_title"), "public-users.json", "JSON (*.json)"
        )
        if not path:
            return
        try:
            output = self.gateway.export_public_users(Path(path))
        except Exception as error:
            self._error(error)
            return
        QMessageBox.information(self, self.tr("users.export_title"), self.tr("users.export_done", path=str(output)))

    def refresh_users(self):
        page = self.pages.get("users")
        if isinstance(page, UserManagementPage):
            page.set_users(self.gateway.users, self.gateway.trashed_users)
        ssvep = self.pages.get("ssvep")
        if ssvep is not None and hasattr(ssvep, "set_users"):
            selected = ssvep.config().user_id
            ssvep.set_users(self.gateway.users, selected_id=selected)

    def create_user_from_capture(self):
        next_id = "U0001"
        if hasattr(self.gateway, "next_user_id"):
            next_id = self.gateway.next_user_id()
        dialog = UserDialog(self.tr, next_id=next_id)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        try:
            profile = self.gateway.add_user(dialog.profile())
        except ValueError as error:
            self._error(error)
            return
        self.refresh_users()
        self.pages["ssvep"].set_users(self.gateway.users, selected_id=profile.user_id)

    def launch_openbci(self):
        try:
            process_id = self.gateway.launch_openbci_workspace(self.tr.locale)
        except RuntimeError as error:
            self._error(error)
            return
        QMessageBox.information(
            self,
            self.tr("openbci.started_title"),
            self.tr("openbci.started", pid=process_id),
        )

    def cancel_task(self):
        if not self.gateway.snapshot.active:
            return
        self.timer.stop()
        snapshot = self.gateway.cancel()
        self._record("capture", "task cancelled", phase=snapshot.phase.value)
        if snapshot.result is not None:
            self.latest_dataset = snapshot.result.id
            self.show_result(snapshot.result)
        else:
            self.navigate("ssvep")
            self.pages["ssvep"].error.setText(self.tr("task.cancelled"))

    def return_to_task(self):
        self.navigate("task")

    def show_result(self, result: Dataset):
        self.result = result
        self._showing_dataset_summary = result.imported
        self.navigate("result")

    def open_dataset_quality(self, path):
        if self.quality_controller.select_session(path):
            self.navigate("quality_details")

    def show_dataset(self, result: Dataset):
        self.result = result
        self._showing_dataset_summary = True
        self.navigate("result")

    def poll(self):
        snapshot = self.gateway.tick()
        if snapshot.phase == Phase.COMPLETED and snapshot.result and snapshot.result.id != self.latest_dataset:
            self.timer.stop()
            self.latest_dataset = snapshot.result.id
            self.show_result(snapshot.result)
        elif snapshot.phase == Phase.CANCELLED and snapshot.result and snapshot.result.id != self.latest_dataset:
            self.timer.stop()
            self.latest_dataset = snapshot.result.id
            self.show_result(snapshot.result)
        elif snapshot.phase == Phase.FAILED:
            self.diagnostic_store.error(
                "capture",
                snapshot.error or "task failed",
                context={"protocol": snapshot.protocol},
            )
            self.timer.stop()
            if snapshot.result is not None:
                self.latest_dataset = snapshot.result.id
                self.show_result(snapshot.result)
            else:
                self.navigate("ssvep")
                self.pages["ssvep"].error.setText(snapshot.error or self.tr("validation.worker_failed"))
        elif not snapshot.active:
            self.timer.stop()
        self._update_controls()
        if snapshot.active:
            self._pump_live_samples()

    def _analysis_busy(self):
        return self.offline_controller.busy or self.quality_controller.busy or self.replay_controller.busy

    def _offline_capture_busy(self):
        return (self.gateway.snapshot.active or self.preflight_thread is not None
                or self.channel_calibration_thread is not None or self.import_thread is not None)

    def _offline_guarded_mutation(self, callback, *args, **kwargs):
        if self._analysis_busy():
            raise ValueError("v9.busy_mutation")
        return callback(*args, **kwargs)

    def _update_quality_banner(self):
        if getattr(self, "current_page", None) == "quality_details" and hasattr(self, "mode_banner_label"):
            self.mode_banner_label.setText(self.tr("quality.mode." + self.quality_controller.mode))

    def _update_controls(self):
        capture_busy = self._offline_capture_busy()
        self.offline_controller.set_capture_busy(capture_busy or self.replay_controller.busy or self.quality_controller.busy)
        self.replay_controller.set_capture_busy(capture_busy or self.offline_controller.busy or self.quality_controller.busy)
        self.quality_controller.set_capture_busy(capture_busy or self.offline_controller.busy or self.replay_controller.busy)
        offline_busy = self._analysis_busy()
        models = self.pages['models']
        models.set_capture_busy(capture_busy or offline_busy)
        self.language_selector.setEnabled(self.channel_calibration_thread is None and models.process is None)
        for key in ("datasets", "trash"):
            page = self.pages.get(key)
            if page is not None:
                for button in page.findChildren(QPushButton):
                    if button.objectName() in {"datasetMutationButton", "datasetRenameButton"}:
                        button.setEnabled(not offline_busy)
                import_button = getattr(page, "import_button", None)
                if import_button is not None:
                    import_button.setEnabled(not offline_busy and self.import_thread is None)
        users = self.pages.get("users")
        if users is not None:
            users._selection_changed()
            for name in ("delete_button", "edit_button", "restore_button", "purge_button"):
                button = getattr(users, name, None)
                if button is not None and offline_busy:
                    button.setEnabled(False)
        snapshot = self.gateway.snapshot
        device = self.gateway.device
        for key in ("home", "devices", "apps"):
            page = self.pages.get(key)
            setter = getattr(page, "set_device_info", None)
            if callable(setter):
                setter(device)
        # Before a task starts the selected form mode is meaningful; after a
        # task starts the gateway configuration becomes the source of truth.
        mode = CaptureMode(
            self.gateway.config.mode if snapshot.active else self.draft_config.mode
        )
        self.mode_banner_label.setText(self.tr("mode.banner", mode=self.tr("mode." + mode.value)))
        self._update_quality_banner()
        if snapshot.active:
            device_name, device_port, device_channels, device_rate = (
                device.name, device.port, device.channels, device.sample_rate
            )
        elif device.connected:
            device_name, device_port, device_channels, device_rate = (
                device.name, device.port, device.channels, device.sample_rate
            )
        else:
            device_name, device_port, device_channels, device_rate = (
                "OpenBCI Cyton", self.draft_config.port or "AUTO", 8, 250
            )
        self.device_summary.setText(
            f"{device_name}\n{device_port} · {device_channels} CH · {device_rate} Hz"
        )
        self.active_banner.setVisible(snapshot.active and self.current_page != "task")
        can_check = (not snapshot.active and self.preflight_thread is None
                     and self.channel_calibration_thread is None and not offline_busy)
        form = self.pages["ssvep"]
        form.start_button.setEnabled(can_check and models.process is None)
        check_page = self.pages["quality"]
        check_page.check_button.setEnabled(can_check)
        report = self._channel_check_report
        checking = self.preflight_thread is not None and report is None
        if report is not None:
            port = self.gateway.config.port if snapshot.active else form.port.text()
            port = port.strip().upper() or "AUTO"
            checked_ports = {str(report.get(key) or "").strip().upper()
                             for key in ("requested_port", "selected_port")}
            if port not in checked_ports:
                report = None
        summary = check_page.render(report, running=checking)
        form.channel_check_status.setText(summary)
        form.channel_check_status.setVisible(mode == CaptureMode.CYTON and (report is not None or checking))
        task = self.pages["task"]
        task.update_snapshot(snapshot, self.gateway.config)
        task.channel_check_status.setText(summary)
        task.channel_check_status.setVisible(
            snapshot.phase == Phase.COUNTDOWN and mode == CaptureMode.CYTON and report is not None)
        self._sync_acquisition_monitor(snapshot)

    def show_acquisition_monitor(self):
        self._monitor_suppressed = False
        self._sync_acquisition_monitor(self.gateway.snapshot)

    def _monitor_dismissed(self):
        # Closing this view never cancels acquisition; the task button reopens it.
        self._monitor_suppressed = True

    def _monitor_screen_removed(self, screen):
        window = self.monitor_window
        if window is not None and window.target_screen == screen:
            # Hide immediately, before Qt relocates an orphaned window to the
            # primary screen. The next poll can select another extended screen.
            window.hide()
            window.target_screen = None
            window.target_screen_name = ""

    def _dispose_acquisition_monitor(self):
        if self.monitor_window is not None:
            self.monitor_window.close()
            self.monitor_window.deleteLater()
            self.monitor_window = None

    def _sync_acquisition_monitor(self, snapshot):
        application = QApplication.instance()
        screen = resolve_monitor_screen(application, snapshot.screen_index)
        page = self.pages["task"]
        page.monitor_button.setEnabled(snapshot.active and screen is not None)
        page.monitor_notice.setText(self.tr(
            "monitor.auto_note" if screen is not None else "monitor.no_screen"
        ))
        if not snapshot.active or screen is None:
            if self.monitor_window is not None:
                self.monitor_window.hide()
                if screen is None:
                    self.monitor_window.target_screen = None
                    self.monitor_window.target_screen_name = ""
                self.monitor_window.update_snapshot(snapshot, self.gateway.config, self.gateway.device)
            return
        if self.monitor_window is None and not self._monitor_suppressed:
            self.monitor_window = AcquisitionMonitorWindow(self.tr, self)
            self.monitor_window.dismissed.connect(self._monitor_dismissed)
            self.monitor_window.cancel_requested.connect(self.cancel_task)
        window = self.monitor_window
        if window is None:
            return
        if not self._monitor_suppressed and (
            not window.isVisible() or window.target_screen != screen
        ):
            window.show_on_screen(screen)
        window.update_snapshot(snapshot, self.gateway.config, self.gateway.device)

    def _pump_live_samples(self):
        if CaptureMode(self.gateway.config.mode) is not CaptureMode.CYTON:
            return
        if self.gateway.config.continuous_ssvep:
            reader = getattr(self.gateway, "read_live_ssvep", None)
            row = reader() if reader else None
            text = self.tr("ssvep.waiting")
            if row:
                state = row["state"]
                value = f"{row['frequency_hz']} Hz" if state == "target" else self.tr("ssvep." + state)
                if row.get("quality") == "stale_input":
                    value = self.tr("ssvep.stale_input")
                text = self.tr("ssvep.decision", value=value, sample=row["end_sample"])
            actual = row.get('model') if row else None
            if isinstance(actual, dict) and all(isinstance(actual.get(k), str) for k in ('name', 'version', 'device')):
                identity = f"{actual['name']} · {actual['version']} · {actual['device']}"
            else:
                identity = self.tr('models.requested', name=self.pages['ssvep'].realtime_model.currentText())
            text = f'{identity}\n{text}'
            self.pages["task"].ssvep_prediction.setText(text)
            if self.monitor_window is not None:
                self.monitor_window.ssvep_prediction.setText(text)
        samples = self._read_live_waveform()
        if samples is None:
            return
        waveform = self.pages["task"].waveform
        waveform.push_samples(samples, stream_ids=getattr(self, "_quality_stream_ids", None))
        if self.monitor_window is not None:
            self.monitor_window.update_samples(
                waveform.model.CHANNEL_NAMES, waveform.model.SAMPLE_RATE_HZ, samples,
                stream_ids=getattr(self, "_quality_stream_ids", None),
            )
            self.monitor_window.update_snapshot(
                self.gateway.snapshot, self.gateway.config, self.gateway.device,
            )

    def _read_live_waveform(self, maximum_rows: int = 1000):
        reader = getattr(self.gateway, "read_live_waveform", None)
        if reader is None:
            return None
        payload = reader(maximum_rows)
        if not isinstance(payload, dict):
            return None
        self.quality_controller.accept_live(payload.get("quality_v2"))
        if self.monitor_window is not None and hasattr(self.monitor_window, "quality_panel"):
            self.monitor_window.quality_panel.update_quality(payload.get("quality_v2"))
        self._quality_stream_ids = payload.get("metadata_v2", {}).get("stream_id")
        samples = payload.get("samples")
        names = payload.get("channel_names")
        rate = payload.get("sample_rate_hz")
        if not isinstance(samples, list) or not samples or not isinstance(names, (list, tuple)):
            return None
        try:
            channel_names = tuple(str(item) for item in names)
            sample_rate = int(rate)
            if len(samples) != len(channel_names) or not isinstance(samples[0], (list, tuple)):
                return None
            count = len(samples[0])
            if not count or any(not isinstance(channel, (list, tuple)) or len(channel) != count for channel in samples):
                return None
            samples = [[float(value) for value in channel] for channel in samples]
        except (TypeError, ValueError, OverflowError):
            return None
        task_page = self.pages.get("task")
        if task_page is not None and hasattr(task_page, "waveform"):
            try:
                task_page.waveform.configure(channel_names, sample_rate)
            except ValueError:
                return None
        return samples

    def closeEvent(self, event):
        if any(thread is not None for thread in (
            self.preflight_thread, self.channel_calibration_thread, self.import_thread,
        )):
            self._pending_ssvep = None
            self._close_after_workers = True
            if self.channel_calibration_worker is not None:
                self.channel_calibration_worker.request_stop()
            event.ignore()
            return
        self._close_after_workers = False
        if not self.quality_controller.close():
            self.navigate("quality_details")
            event.ignore()
            return
        if not self.replay_controller.close() or not self.offline_controller.close():
            event.ignore()
            return
        self._record("ui", "workstation window closing")
        self.timer.stop()
        self._dispose_acquisition_monitor()
        self.gateway.cancel()
        close_gateway = getattr(self.gateway, "close", None)
        if callable(close_gateway):
            close_gateway()
        if self.persist_settings:
            self.settings.setValue("window/geometry", self.saveGeometry())
            page = self.pages.get("ssvep")
            if page is not None:
                self.settings.setValue("capture/save_directory", page.save.text())
            self.settings.sync()
        event.accept()

    def _last_import_directory(self) -> str:
        saved = str(self.settings.value("datasets/openbci_directory", ""))
        if saved and Path(saved).expanduser().is_dir():
            return saved
        return str(Path.home() / "Documents" / "OpenBCI_GUI" / "Recordings")

    def _auto_import_default(self):
        source = Path.home() / "Documents" / "OpenBCI_GUI" / "Recordings"
        if source.is_dir():
            self._start_import(source, automatic=True)

    def _start_import(self, source_root: Path | str, automatic: bool = False):
        if self._analysis_busy() or self.import_thread is not None:
            return
        source = Path(source_root).expanduser().resolve()
        self._record("datasets", "OpenBCI import started", source_root=str(source), automatic=automatic)
        if self.persist_settings:
            self.settings.setValue("datasets/openbci_directory", str(source))
        self.import_thread = QThread(self)
        self.import_worker = _ImportWorker(self.gateway, source)
        self.import_worker.moveToThread(self.import_thread)
        self.import_thread.started.connect(self.import_worker.run)
        self.import_worker.finished.connect(self._import_finished)
        self.import_worker.finished.connect(self.import_thread.quit)
        self.import_worker.finished.connect(self.import_worker.deleteLater)
        self.import_thread.finished.connect(self.import_thread.deleteLater)
        self.import_thread.finished.connect(self._import_thread_finished)
        self.import_thread.start()
        self._update_controls()
        if self.current_page == "datasets":
            self.navigate("datasets")

    def _import_thread_finished(self):
        self.import_thread = None
        self.import_worker = None
        if self._close_after_workers:
            self.close()
            return
        self._update_controls()
        if self.current_page == "datasets":
            self.navigate("datasets")

    def _import_finished(self, result):
        if isinstance(result, Exception):
            self.diagnostic_store.exception(
                result,
                source="datasets",
                message="OpenBCI import failed",
            )
            self.import_status = self.tr("datasets.import_failed", reason=str(result))
        else:
            self._record(
                "datasets",
                "OpenBCI import finished",
                imported=result.imported_count,
                skipped=result.skipped_count,
                failed=result.failed_count,
            )
            self.import_status = self.tr(
                "datasets.import_status",
                imported=result.imported_count,
                skipped=result.skipped_count,
                failed=result.failed_count,
            )
            if result.failed_count:
                self.import_status += " " + "; ".join(result.failures)
        self.gateway.refresh_datasets()
        if self.current_page == "datasets":
            self.navigate("datasets")
