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

    def test_default_ssvep_page_is_one_click_runnable(self):
        from apps.workstation_ui.gateway import CaptureMode, Phase

        page = self.window.pages["ssvep"]
        self.assertEqual(CaptureMode(page.mode.currentData()), CaptureMode.DEMO)
        self.assertTrue(page.config().save_directory.is_absolute())
        page._start()
        self.assertEqual(self.window.current_page, "task")
        self.assertEqual(self.gateway.snapshot.phase, Phase.COUNTDOWN)
        self.assertFalse(page.start_button.isEnabled())

        # A normal user can leave every field untouched and still reach the
        # completed result path through the same UI signal as a button click.
        self.now = 5
        self.window.poll()
        self.now = 16.625
        self.window.poll()
        self.assertEqual(self.window.current_page, "result")
        self.assertIsNotNone(self.window.result)
        self.assertFalse(self.window.result.persisted)

    def test_live_page_is_non_persistent_capture_test_with_animated_waveform(self):
        from apps.workstation_ui.gateway import CaptureMode

        page = self.window.pages["live"]
        self.window.navigate("live")
        self.window.show()
        self.application.processEvents()
        self.assertEqual("采集测试", page.title_label.text())
        self.assertTrue(page.waveform._timer.isActive())
        phase = page.waveform._phase
        page.waveform._advance()
        self.assertNotEqual(phase, page.waveform._phase)

        page._start()
        self.assertIs(CaptureMode.DEMO, self.gateway.config.mode)
        self.now = 3
        self.window.poll()
        self.window.stop_manual()
        self.assertEqual("result", self.window.current_page)
        self.assertEqual("capture_test", self.window.result.origin)
        self.assertFalse(self.window.result.persisted)
        self.assertFalse(self.window.result.path.exists())

    def test_cancel_and_close_stop_timer(self):
        from apps.workstation_ui.gateway import CaptureConfig, Phase
        self.window.start_ssvep(CaptureConfig(), 8)
        self.window.cancel_task()
        self.assertEqual(self.gateway.snapshot.phase, Phase.CANCELLED)
        self.assertFalse(self.window.timer.isActive())
        self.window.start_ssvep(CaptureConfig(), 8)
        self.window.close()
        self.assertFalse(self.gateway.snapshot.active)

    def test_cyton_start_runs_preflight_before_starting(self):
        from apps.workstation_ui.gateway import CaptureConfig, CaptureMode, Phase
        calls = []
        self.gateway.preflight_cyton = lambda port="AUTO": calls.append(port) or {"status": "passed", "checks": []}
        self.gateway.add_user(__import__("neurostation_contract", fromlist=["UserProfile"]).UserProfile(
            user_id="U0001", name="Test", age=30, medical_conditions=("none",)
        ))
        config = CaptureConfig(
            mode=CaptureMode.CYTON,
            port="COM5",
            user_id="U0001",
            participant="U0001",
            acknowledge_flicker_risk=True,
        )
        self.window.start_ssvep(config, 1)
        for _ in range(50):
            self.application.processEvents()
            if self.window.gateway.snapshot.phase == Phase.COUNTDOWN:
                break
        self.assertEqual(["COM5"], calls)
        self.assertEqual(Phase.COUNTDOWN, self.gateway.snapshot.phase)

    def test_cyton_failed_preflight_blocks_start(self):
        from apps.workstation_ui.gateway import CaptureConfig, CaptureMode, Phase
        from neurostation_contract import UserProfile
        errors = []
        self.window._error = errors.append
        self.gateway.add_user(UserProfile(
            user_id="U0002", name="Failed", age=30, medical_conditions=("none",)
        ))
        calls = []
        self.gateway.preflight_cyton = lambda port="AUTO": calls.append(port) or {
            "status": "failed", "error": "no device", "checks": []
        }
        config = CaptureConfig(
            mode=CaptureMode.CYTON,
            port="COM9",
            user_id="U0002",
            participant="U0002",
            acknowledge_flicker_risk=True,
        )
        self.window.start_ssvep(config, 1)
        for _ in range(50):
            self.application.processEvents()
            if self.window.preflight_thread is None:
                break
        self.assertEqual(["COM9"], calls)
        self.assertFalse(self.gateway.snapshot.active)
        self.assertEqual(1, len(errors))
        self.assertIsInstance(errors[0], ValueError)

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

    def test_preview_and_cyton_channel_mapping_defaults(self):
        from apps.workstation_ui.gateway import CaptureMode

        page = self.window.pages["ssvep"]
        page.mode.setCurrentIndex(page.mode.findData(CaptureMode.VISUAL_PREVIEW))
        self.application.processEvents()
        self.assertFalse(page.port.isEnabled())
        self.assertFalse(page.channel_manual.isEnabled())
        self.assertTrue(page.channel_auto_label.isHidden())
        self.assertIsNone(page.config().channel_config)
        with self.assertRaisesRegex(ValueError, "validation.flicker_ack"):
            page.config().validate()

        page.mode.setCurrentIndex(page.mode.findData(CaptureMode.CYTON))
        self.application.processEvents()
        self.assertTrue(page.port.isEnabled())
        self.assertFalse(page.channel_auto_label.isHidden())
        self.assertFalse(page.channel_manual.isChecked())
        self.assertIsNone(page.config().channel_config)
        page.channel_manual.setChecked(True)
        channel_path = Path.cwd() / "configs" / "cyton.json"
        page.channel.setText(str(channel_path))
        self.assertEqual(channel_path, page.config().channel_config)

    def test_serial_scan_updates_auto_status(self):
        page = self.window.pages["ssvep"]
        self.window.gateway.scan_serial_ports = lambda: (
            {"device": "COM5", "description": "USB serial"},
        )
        self.window.scan_serial_ports()
        self.assertEqual("AUTO", page.port.text())
        self.assertIn("COM5", page.port_status.text())

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

    def test_escape_does_not_cancel_when_idle(self):
        self.window.cancel_task()
        self.assertFalse(self.gateway.snapshot.active)
        self.assertEqual("apps", self.window.current_page)

    def test_task_page_exposes_current_trial_banner(self):
        from apps.workstation_ui.gateway import CaptureConfig
        self.window.start_ssvep(CaptureConfig(), 8)
        self.now = 5
        self.window.poll()
        self.assertIn("/ 12", self.window.pages["task"].trial_banner.text())

    def test_dataset_filters_persist_when_page_rebuilt(self):
        self.window.navigate("datasets")
        page = self.window.pages["datasets"]
        page.search.setText("U0000")
        page.source_filter.setCurrentIndex(page.source_filter.findData("demo"))
        page.status_filter.setCurrentIndex(page.status_filter.findData("completed"))
        self.window.navigate("apps")
        self.window.navigate("datasets")
        page = self.window.pages["datasets"]
        self.assertEqual("U0000", page.search.text())
        self.assertEqual("demo", page.source_filter.currentData())
        self.assertEqual("completed", page.status_filter.currentData())

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
