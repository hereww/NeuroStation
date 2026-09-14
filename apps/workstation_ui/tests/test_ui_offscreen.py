"""Optional real-Qt smoke tests; automatically skipped when PySide6 is absent."""
import importlib.util
import math
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
        from neurostation_diagnostics import DiagnosticStore
        self.now = 0.0
        self.gateway = MockGateway(lambda: self.now)
        self.diagnostic_store = DiagnosticStore(persist=False)
        self.window = MainWindow(
            self.gateway,
            timer_enabled=False,
            diagnostic_store=self.diagnostic_store,
        )

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

    def test_openbci_waveform_model_uses_rolling_buffers_and_display_copy(self):
        from apps.workstation_ui.components import WaveformDisplayModel

        model = WaveformDisplayModel()
        self.assertEqual(
            ("Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2"),
            model.CHANNEL_NAMES,
        )
        self.assertEqual(8, model.CHANNEL_COUNT)
        self.assertEqual(10, model.SAMPLES_PER_UPDATE)
        self.assertEqual(1250, model.DISPLAY_SAMPLES)
        self.assertEqual(5500, model.RAW_CAPACITY)
        self.assertEqual(200, model.Y_LIMIT_UV)

        source = [[float(sample + channel) for sample in range(6000)] for channel in range(8)]
        original_first_channel = list(source[0])
        model.append(source)

        self.assertEqual(6000, model.sample_index)
        self.assertTrue(all(len(channel) == model.RAW_CAPACITY for channel in model.raw_buffers))
        self.assertTrue(all(len(channel) == model.DISPLAY_SAMPLES for channel in model.filtered_buffers))
        self.assertEqual(original_first_channel, source[0])
        self.assertEqual(tuple(source[0][-model.RAW_CAPACITY:]), tuple(model.raw_buffers[0]))
        self.assertEqual(model.DISPLAY_SAMPLES, len(model.visible_samples(0)))
        self.assertNotEqual(tuple(source[0][-model.DISPLAY_SAMPLES:]), model.visible_samples(0))

    def test_openbci_display_filter_has_requested_bandpass_and_notches(self):
        from apps.workstation_ui.components import _OpenBCIDisplayFilter

        sample_rate = 250
        sample_count = 5_000

        def gain(frequency_hz: float) -> float:
            display_filter = _OpenBCIDisplayFilter(sample_rate)
            output = []
            for index in range(sample_count):
                value = math.sin(2.0 * math.pi * frequency_hz * index / sample_rate)
                output.extend(display_filter.process([value]))
            tail = output[-1_000:]
            return math.sqrt(2.0 * sum(value * value for value in tail) / len(tail))

        self.assertAlmostEqual(0.707, gain(5.0), delta=0.03)
        self.assertGreater(gain(10.0), 0.95)
        self.assertLess(gain(50.0), 0.02)
        self.assertLess(gain(60.0), 0.02)
        self.assertLess(gain(100.0), 0.02)

    def test_live_page_is_non_persistent_capture_test_with_openbci_waveform(self):
        from apps.workstation_ui.gateway import CaptureMode

        page = self.window.pages["live"]
        self.window.navigate("live")
        self.window.show()
        self.application.processEvents()
        self.assertEqual("采集测试", page.title_label.text())
        self.assertTrue(page.waveform._timer.isActive())
        self.assertEqual(40, page.waveform._timer.interval())
        self.assertEqual(8, page.waveform.model.CHANNEL_COUNT)
        self.assertEqual(
            ("Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2"),
            page.waveform.model.CHANNEL_NAMES,
        )
        self.assertEqual(0, page.waveform.model.sample_index)
        self.assertEqual(
            (0.0,) * page.waveform.model.DISPLAY_SAMPLES,
            page.waveform.model.visible_samples(0),
        )
        page.waveform._advance()
        self.assertEqual(0, page.waveform.model.sample_index)

        page._start()
        self.assertIs(CaptureMode.DEMO, self.gateway.config.mode)
        page.waveform._advance()
        self.assertEqual(10, page.waveform.model.sample_index)
        self.assertEqual(1250, len(page.waveform.model.visible_samples(0)))
        self.assertEqual(10, len(page.waveform.model.raw_buffers[0]))
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

    def test_cyton_preflight_warning_surfaces_check_detail(self):
        page = self.window.pages["devices"]
        self.window._preflight_finished({
            "status": "warning",
            "error": "",
            "checks": [{
                "name": "timestamps",
                "status": "warning",
                "detail": "Detected 9 timestamp gaps.",
                "metrics": {
                    "timestamp_gap_count": 9,
                    "timestamp_gap_ratio": 9 / 208,
                    "timestamp_diff_max_s": 0.012,
                },
            }],
        })
        self.assertIn("时间戳", page.preflight_status.text())
        self.assertIn("9", page.preflight_status.text())
        self.assertIn("12.0", page.preflight_status.text())

    def test_cyton_preflight_warning_continues_in_technical_validation(self):
        from apps.workstation_ui.gateway import CaptureConfig, CaptureMode, Phase
        from neurostation_contract import UserProfile
        self.gateway.add_user(UserProfile(
            user_id="U0003", name="Technical", age=30, medical_conditions=("none",)
        ))
        config = CaptureConfig(
            mode=CaptureMode.CYTON,
            port="COM7",
            user_id="U0003",
            participant="U0003",
            acknowledge_flicker_risk=True,
            allow_draft_hardware_config=True,
        )
        self.window._pending_ssvep = (config, 1)
        self.window._preflight_finished({
            "status": "warning",
            "error": "",
            "selected_port": "COM7",
            "checks": [{"name": "channels", "status": "warning", "metrics": {"warning_channels": [2]}}],
        })
        for _ in range(50):
            self.application.processEvents()
            if self.gateway.snapshot.phase == Phase.COUNTDOWN:
                break
        self.assertEqual(Phase.COUNTDOWN, self.gateway.snapshot.phase)

    def test_cyton_preflight_warning_does_not_block_formal_candidate(self):
        from apps.workstation_ui.gateway import CaptureConfig, CaptureMode, Phase
        from neurostation_contract import UserProfile
        self.gateway.add_user(UserProfile(
            user_id="U0004", name="Formal candidate", age=30, medical_conditions=("none",)
        ))
        config = CaptureConfig(
            mode=CaptureMode.CYTON,
            port="COM5",
            user_id="U0004",
            participant="U0004",
            acknowledge_flicker_risk=True,
            allow_draft_hardware_config=False,
        )
        self.window._pending_ssvep = (config, 1)
        self.window._preflight_finished({
            "status": "warning",
            "error": "",
            "selected_port": "COM5",
            "checks": [{"name": "timestamps", "status": "warning", "metrics": {
                "timestamp_gap_count": 22,
                "timestamp_gap_ratio": 0.031,
                "timestamp_diff_max_s": 0.3546,
                "packet_sequence_available": True,
                "packet_loss_count": 0,
                "packet_sequence_mismatch_count": 0,
                "packet_duplicate_count": 0,
            }}],
        })
        for _ in range(50):
            self.application.processEvents()
            if self.gateway.snapshot.phase == Phase.COUNTDOWN:
                break
        self.assertEqual(Phase.COUNTDOWN, self.gateway.snapshot.phase)
        self.assertIn("继续", self.window.pages["ssvep"].error.text())
        task_page = self.window.pages["task"]
        self.assertFalse(task_page.quality_notice.isHidden())
        self.assertIn("继续", task_page.quality_notice.text())

    def test_cyton_preflight_degraded_requires_review(self):
        from apps.workstation_ui.gateway import CaptureConfig, CaptureMode, Phase
        from neurostation_contract import UserProfile
        self.gateway.add_user(UserProfile(
            user_id="U0005", name="Degraded", age=30, medical_conditions=("none",)
        ))
        config = CaptureConfig(
            mode=CaptureMode.CYTON,
            port="COM5",
            user_id="U0005",
            participant="U0005",
            acknowledge_flicker_risk=True,
        )
        self.window._pending_ssvep = (config, 1)
        self.window._preflight_finished({
            "status": "degraded",
            "error": "",
            "selected_port": "COM5",
            "checks": [{"name": "timestamps", "status": "degraded", "detail": "large gap"}],
        })
        self.assertFalse(self.gateway.snapshot.active)
        self.assertNotEqual(Phase.COUNTDOWN, self.gateway.snapshot.phase)
        self.assertIn("复核", self.window.pages["ssvep"].error.text())

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

    def test_serial_scan_selects_detected_port(self):
        page = self.window.pages["ssvep"]
        self.window.gateway.scan_serial_ports = lambda: (
            {"device": "COM5", "description": "USB serial"},
        )
        self.window.scan_serial_ports()
        self.assertEqual("COM5", page.port.text())
        self.assertIn("COM5", page.port_status.text())

    def test_diagnostics_page_navigates_refreshes_and_shows_events(self):
        self.diagnostic_store.warning(
            "test",
            "diagnostic event visible",
            context={"attempt": 1},
        )
        self.window.navigate("diagnostics")
        page = self.window.pages["diagnostics"]

        self.assertGreater(page.checks.rowCount(), 0)
        self.assertIn("diagnostic event visible", page.events.toPlainText())
        self.assertIn("test", page.events.toPlainText())

        self.diagnostic_store.error("test", "second diagnostic event")
        self.window._refresh_diagnostics()
        self.assertIn("second diagnostic event", page.events.toPlainText())

        self.window.change_language("en-US")
        self.window.navigate("diagnostics")
        self.assertEqual("Diagnostics", self.window.pages["diagnostics"].title_label.text())

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
