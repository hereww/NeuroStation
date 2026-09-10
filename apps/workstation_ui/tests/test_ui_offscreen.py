"""Optional real-Qt smoke tests; automatically skipped when PySide6 is absent."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_QT = importlib.util.find_spec("PySide6") is not None


@unittest.skipUnless(HAS_QT, "PySide6 not installed; pure logic tests still run")
class QtOffscreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        from apps.workstation_ui.app import MainWindow
        from apps.workstation_ui.gateway import MockGateway
        self.now = 0.0
        self.gateway = MockGateway(lambda: self.now)
        self.window = MainWindow(self.gateway, timer_enabled=False)

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.application.processEvents()

    def test_navigation_language_and_static_render(self):
        from apps.workstation_ui.app import NAVIGATION
        self.window.show()
        for page in NAVIGATION:
            self.window.navigate(page)
            self.application.processEvents()
            self.assertEqual(self.window.current_page, page)
        self.window.change_language("en-US")
        self.assertEqual(self.window.tr("nav.apps"), "Acquisition apps")
        self.assertFalse(self.window.grab().isNull())

    def test_task_completion_dataset_and_timer_cleanup(self):
        from PySide6.QtWidgets import QFrame
        from apps.workstation_ui.gateway import CaptureConfig, Phase
        self.window.start_ssvep(CaptureConfig(), 8)
        self.assertEqual(self.window.current_page, "task")
        self.assertFalse(self.window.pages["ssvep"].start_button.isEnabled())
        self.now = 4.999
        self.window.poll()
        self.assertEqual(self.gateway.snapshot.phase, Phase.COUNTDOWN)
        self.now = 5
        self.window.poll()
        self.now = 16.625
        self.window.poll()
        self.assertEqual(self.window.current_page, "result")
        self.window.navigate("datasets")
        self.assertEqual(self.window.latest_dataset, self.gateway.datasets[0].id)
        self.assertIsNotNone(self.window.pages["datasets"].findChild(QFrame, "latestDataset"))
        self.assertFalse(self.window.timer.isActive())

    def test_cancel_and_close_stop_timer(self):
        from apps.workstation_ui.gateway import CaptureConfig, Phase
        self.window.start_ssvep(CaptureConfig(), 8)
        self.window.cancel_task()
        self.assertEqual(self.gateway.snapshot.phase, Phase.CANCELLED)
        self.assertFalse(self.window.timer.isActive())
        self.window.start_ssvep(CaptureConfig(), 8)
        self.window.close()
        self.assertFalse(self.gateway.snapshot.active)

    def test_ssvep_mode_selector_preserves_enum_and_requires_risk_ack(self):
        from apps.workstation_ui.gateway import CaptureMode

        page = self.window.pages["ssvep"]
        page.mode.setCurrentIndex(page.mode.findData(CaptureMode.SYNTHETIC))
        self.application.processEvents()
        self.assertFalse(page.speed.isEnabled())
        self.assertTrue(page.acknowledge.isEnabled())
        with self.assertRaisesRegex(ValueError, "validation.flicker_ack"):
            page.config().validate()
        page.acknowledge.setChecked(True)
        config = page.config()
        config.validate()
        self.assertIs(config.mode, CaptureMode.SYNTHETIC)
        page.mode.setCurrentIndex(page.mode.findData(CaptureMode.CYTON))
        self.application.processEvents()
        self.assertTrue(page.port.isEnabled())
        self.assertTrue(page.channel.isEnabled())

    def test_language_storage_and_window_geometry_persist(self):
        from PySide6.QtCore import QSettings
        from apps.workstation_ui.app import MainWindow
        from apps.workstation_ui.gateway import MockGateway

        with tempfile.TemporaryDirectory() as directory:
            settings_path = Path(directory) / "settings.ini"
            dataset_path = Path(directory) / "持久化数据"
            settings = QSettings(str(settings_path), QSettings.Format.IniFormat)
            first = MainWindow(
                MockGateway(lambda: self.now),
                locale="zh-CN",
                timer_enabled=False,
                persist_settings=True,
                settings=settings,
            )
            first.change_language("en-US")
            first.pages["ssvep"].save.setText(str(dataset_path))
            first.resize(780, 650)
            first.close()
            settings.sync()

            second = MainWindow(
                MockGateway(lambda: self.now),
                locale=None,
                timer_enabled=False,
                persist_settings=True,
                settings=QSettings(str(settings_path), QSettings.Format.IniFormat),
            )
            self.assertEqual("en-US", second.tr.locale)
            self.assertEqual(dataset_path, second.draft_config.save_directory)
            self.assertEqual((780, 650), (second.width(), second.height()))
            second.close()

    def test_narrow_window_keeps_sidebar_compact_and_content_scrollable(self):
        self.window.show()
        self.application.processEvents()
        self.window.resize(320, 560)
        self.application.processEvents()
        self.assertEqual(68, self.window.sidebar.width())
        self.assertFalse(self.window.sidebar_summary.isVisible())
        self.assertFalse(self.window.platform_label.isVisible())
        self.assertGreaterEqual(self.window.minimumWidth(), 320)
        self.window.resize(1024, 640)
        self.application.processEvents()
        self.assertEqual(186, self.window.sidebar.width())
        self.assertTrue(self.window.sidebar_summary.isVisible())


if __name__ == "__main__":
    unittest.main()
