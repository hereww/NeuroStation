"""Desktop shell and UI command routing; one gateway, one task timer."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QSettings, QTimer, Qt
from PySide6.QtGui import QIcon, QPalette
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QListWidget,
    QListWidgetItem, QComboBox, QStackedWidget, QScrollArea, QMessageBox,
)

from .components import action, label
from .gateway import CaptureGateway, MockGateway, CaptureConfig, CaptureMode, Phase, Dataset
from .i18n import Translator
from .pages import (
    HomePage, DevicesPage, LivePage, AppsPage, SSVEPPage, TaskPage,
    ResultPage, DatasetsPage, InfoPage,
)


NAVIGATION = ("home", "devices", "live", "apps", "datasets", "openbci", "integrations")


def product_stylesheet(dark: bool) -> str:
    background, surface, line, ink, muted, accent, active = (
        ("#151a20", "#1b222a", "#34434e", "#e0e9ef", "#a6b6c1", "#69c8d3", "#233b41")
        if dark else ("#ffffff", "#f5f7f8", "#dfe6e9", "#20313d", "#677781", "#087c88", "#e4f3f3")
    )
    return f"""
        QWidget {{ color: {ink}; background-color: {background}; font-size: 13px; }}
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
        gateway: CaptureGateway | None = None,
        locale: str | None = "zh-CN",
        timer_enabled: bool = True,
        persist_settings: bool = False,
        save_directory_override: Path | None = None,
        settings: QSettings | None = None,
    ):
        super().__init__()
        self.gateway = gateway or MockGateway()
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
        self.current_page = "apps"
        self.draft_config = self.gateway.config
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
        self.latest_dataset: str | None = None
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
        self._replace_page("home", HomePage(self.tr, self.navigate))
        self._replace_page("devices", DevicesPage(self.tr, self.navigate))
        live = LivePage(self.tr, self.draft_config)
        live.start_requested.connect(self.start_manual)
        live.stop_requested.connect(self.stop_manual)
        live.marker_requested.connect(self.add_marker)
        self._replace_page("live", live)
        self._replace_page("apps", AppsPage(self.tr, self.navigate))
        detail = SSVEPPage(self.tr, self.draft_config, self.navigate)
        detail.start_requested.connect(self.start_ssvep)
        self._replace_page("ssvep", detail)
        task = TaskPage(self.tr)
        task.cancel_requested.connect(self.cancel_task)
        self._replace_page("task", task)
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
            self._replace_page(key, DatasetsPage(self.tr, self.gateway.datasets, self.latest_dataset,
                                                 self.show_result, self.navigate))
        elif key == "result":
            if self.result is None:
                key = "apps"
            else:
                self._replace_page(key, ResultPage(self.tr, self.result, self.navigate))
        self.current_page = key
        self.stack.setCurrentWidget(self.screens[key])
        nav_key = "apps" if key in ("ssvep", "task", "result") else key
        self.navigation.blockSignals(True)
        self.navigation.setCurrentRow(NAVIGATION.index(nav_key))
        self.navigation.blockSignals(False)
        self._update_controls()

    def change_language(self, locale: str):
        if locale == self.tr.locale:
            return
        self.draft_config = self.pages["ssvep"].config()
        if self.persist_settings:
            self.settings.setValue("ui/language", locale)
        self.tr = Translator(locale)
        self._build_shell()

    def _error(self, error: Exception):
        QMessageBox.warning(self, self.tr("error.title"), self.tr(str(error)))

    def _start_timer(self):
        if self.timer_enabled and not self.timer.isActive():
            self.timer.start()

    def start_ssvep(self, config: CaptureConfig, speed: float):
        try:
            self.gateway.start_ssvep(config, speed)
        except (ValueError, RuntimeError) as error:
            self._error(error)
            return
        self.draft_config = config
        if self.persist_settings:
            self.settings.setValue("capture/save_directory", str(config.save_directory))
        self.navigate("task")
        self._start_timer()

    def start_manual(self, config: CaptureConfig):
        try:
            self.gateway.start_manual(config)
        except (ValueError, RuntimeError) as error:
            self._error(error)
            return
        self._update_controls()
        self._start_timer()

    def stop_manual(self):
        try:
            result = self.gateway.stop_manual()
        except RuntimeError as error:
            self._error(error)
            return
        self.timer.stop()
        self.latest_dataset = result.id
        self.show_result(result)

    def add_marker(self):
        try:
            self.gateway.add_marker()
        except RuntimeError as error:
            self._error(error)
        self._update_controls()

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
        self.timer.stop()
        snapshot = self.gateway.cancel()
        if snapshot.result is not None:
            self.latest_dataset = snapshot.result.id
            self.show_result(snapshot.result)
        else:
            self.navigate("ssvep")
            self.pages["ssvep"].error.setText(self.tr("task.cancelled"))

    def return_to_task(self):
        self.navigate("live" if self.gateway.snapshot.protocol == "manual" else "task")

    def show_result(self, result: Dataset):
        self.result = result
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
        mode = CaptureMode(self.gateway.config.mode)
        self.mode_banner_label.setText(self.tr("mode.banner", mode=self.tr("mode." + mode.value)))
        device = self.gateway.device
        self.device_summary.setText(
            f"{device.name}\n{device.port} · {device.channels} CH · {device.sample_rate} Hz"
        )
        self.active_banner.setVisible(snapshot.active and self.current_page not in ("task", "live"))
        self.pages["live"].update_snapshot(snapshot)
        self.pages["ssvep"].start_button.setEnabled(not snapshot.active)
        self.pages["task"].update_snapshot(snapshot, self.gateway.config)

    def closeEvent(self, event):
        self.timer.stop()
        self.gateway.cancel()
        if self.persist_settings:
            self.settings.setValue("window/geometry", self.saveGeometry())
            page = self.pages.get("ssvep")
            if page is not None:
                self.settings.setValue("capture/save_directory", page.save.text())
            self.settings.sync()
        event.accept()
