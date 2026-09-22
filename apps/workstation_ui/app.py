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
    QListWidgetItem, QComboBox, QStackedWidget, QScrollArea, QMessageBox, QFileDialog,
)

from .components import action, label
from .gateway import CaptureGateway, CaptureConfig, CaptureMode, Phase, Dataset
from .i18n import Translator
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
            self.finished.emit(error)
            return
        self.finished.emit(report)


class _ChannelCalibrationWorker(QObject):
    finished = Signal(object)
    samples = Signal(object)

    def __init__(self, gateway: CaptureGateway, channel_number: int, port: str):
        super().__init__()
        self.gateway = gateway
        self.channel_number = channel_number
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
                seconds=None,
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
    "home", "devices", "users", "apps", "datasets", "trash", "openbci",
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
        QLabel#notice {{ background: {surface}; padding: 12px; border-left: 3px solid #b58b3a; }}
        QLabel#error {{ color: {'#f39393' if dark else '#b44242'}; }}
        QLabel#estimate {{ font-size: 15px; color: {accent}; }}
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
        self.channel_calibration_thread: QThread | None = None
        self.channel_calibration_worker: _ChannelCalibrationWorker | None = None
        self._channel_calibration_index: int | None = None
        self._pending_ssvep: tuple[CaptureConfig, float] | None = None
        self._preflight_ready = False
        self.import_status = ""
        self._dataset_search = ""
        self._dataset_source = ""
        self._dataset_status = ""
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
        language = QComboBox()
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
        breadcrumb.addWidget(action(self.tr("app.datasets"), lambda: self.navigate("datasets")))
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
                device_info=self.gateway.device,
            ),
        )
        self._replace_page("users", self._build_users_page())
        self._replace_page("apps", AppsPage(self.tr, self.navigate, self.gateway.device))
        detail = SSVEPPage(self.tr, self.draft_config, self.navigate, self.gateway.users)
        detail.start_requested.connect(self.start_ssvep)
        detail.serial_scan_requested.connect(self.scan_serial_ports)
        detail.create_user_requested.connect(self.create_user_from_capture)
        detail.refresh_users_requested.connect(self.refresh_users)
        detail.mode.currentIndexChanged.connect(lambda _index: self._capture_form_changed())
        detail.port.textChanged.connect(lambda _text: self._capture_form_changed())
        self._replace_page("ssvep", detail)
        task = TaskPage(self.tr)
        task.cancel_requested.connect(self.cancel_task)
        self._replace_page("task", task)
        task.waveform.set_sample_provider(self._read_live_waveform)
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
                import_busy=self.import_thread is not None,
                import_status=self.import_status,
                import_directory=self._last_import_directory(),
                search_text=self._dataset_search,
                source_value=self._dataset_source,
                status_value=self._dataset_status,
                callbacks={
                    "delete": self.gateway.delete_dataset,
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
                callbacks={
                    "restore": self.gateway.restore_dataset,
                    "purge": self.gateway.purge_dataset,
                    "refresh": self.refresh_datasets,
                },
            )
            page.filter_changed.connect(self._dataset_filters_changed)
            self._replace_page(key, page)
        elif key == "users":
            self._replace_page(key, self._build_users_page())
        elif key == "result":
            if self.result is None:
                key = "apps"
            else:
                page = DatasetSummaryPage if self._showing_dataset_summary else ResultPage
                self._replace_page(key, page(self.tr, self.result, self.navigate))
        elif key == "diagnostics":
            self._refresh_diagnostics()
        self.current_page = key
        self.stack.setCurrentWidget(self.screens[key])
        nav_key = "apps" if key in ("ssvep", "task", "result") else key
        self.navigation.blockSignals(True)
        self.navigation.setCurrentRow(NAVIGATION.index(nav_key))
        self.navigation.blockSignals(False)
        self._update_controls()

    def _dataset_filters_changed(self, search: str, source: str, status: str):
        self._dataset_search = search
        self._dataset_source = source
        self._dataset_status = status

    def refresh_datasets(self):
        self.gateway.refresh_datasets()
        if self.current_page in ("datasets", "trash"):
            self.navigate(self.current_page)

    def change_language(self, locale: str):
        if locale == self.tr.locale:
            return
        self.draft_config = self.pages["ssvep"].config()
        if self.persist_settings:
            self.settings.setValue("ui/language", locale)
        self.tr = Translator(locale)
        self._build_shell()

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
        if self.preflight_thread is not None or self.channel_calibration_thread is not None:
            return
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

    def calibrate_cyton_channel(self, channel_index: int):
        if self.gateway.snapshot.active or self.preflight_thread is not None:
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

    def _preflight_finished(self, result):
        if isinstance(result, Exception):
            report = {"status": "failed", "error": str(result), "checks": []}
        else:
            report = result if isinstance(result, dict) else {"status": "failed", "error": ""}
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
            "update": self.gateway.update_user,
            "delete": self.gateway.delete_user,
            "restore": self.gateway.restore_user,
            "purge": self.gateway.purge_user,
            "refresh": self.refresh_users,
        }
        page = UserManagementPage(
            self.tr,
            self.gateway.users,
            self.gateway.trashed_users,
            callbacks,
        )
        page.export_requested.connect(self.export_public_users)
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

    def _update_controls(self):
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
        if snapshot.active:
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
        self.pages["ssvep"].start_button.setEnabled(
            not snapshot.active and self.preflight_thread is None
        )
        self.pages["task"].update_snapshot(snapshot, self.gateway.config)

    def _read_live_waveform(self, maximum_rows: int = 1000):
        reader = getattr(self.gateway, "read_live_waveform", None)
        if reader is None:
            return None
        payload = reader(maximum_rows)
        if not isinstance(payload, dict):
            return None
        samples = payload.get("samples")
        names = payload.get("channel_names")
        rate = payload.get("sample_rate_hz")
        if not isinstance(samples, list) or not samples:
            return None
        try:
            channel_names = tuple(str(item) for item in names)
            sample_rate = int(rate)
        except (TypeError, ValueError):
            return None
        task_page = self.pages.get("task")
        if task_page is not None and hasattr(task_page, "waveform"):
            try:
                task_page.waveform.configure(channel_names, sample_rate)
            except ValueError:
                return None
        return samples

    def closeEvent(self, event):
        self._record("ui", "workstation window closing")
        self.timer.stop()
        if self.import_thread is not None:
            self.import_thread.quit()
            self.import_thread.wait(2000)
        if self.preflight_thread is not None:
            self.preflight_thread.quit()
            self.preflight_thread.wait(4000)
        if self.channel_calibration_worker is not None:
            self.channel_calibration_worker.request_stop()
        if self.channel_calibration_thread is not None:
            self.channel_calibration_thread.quit()
            self.channel_calibration_thread.wait(4000)
        self.gateway.cancel()
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
        if self.import_thread is not None:
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
        if self.current_page == "datasets":
            self.navigate("datasets")

    def _import_thread_finished(self):
        self.import_thread = None
        self.import_worker = None
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
