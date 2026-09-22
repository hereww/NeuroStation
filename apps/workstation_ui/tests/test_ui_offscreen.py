"""Optional real-Qt smoke tests; automatically skipped when PySide6 is absent."""

import importlib.util
import math
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_QT = importlib.util.find_spec("PySide6") is not None

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "configs" / "protocols" / "ssvep_four_target_v2.json"
CHANNELS = ROOT / "configs" / "channel_config_v1_auto.json"


@unittest.skipUnless(HAS_QT, "PySide6 not installed; pure logic tests still run")
class QtOffscreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        from apps.workstation_ui.app import MainWindow
        from eeg_tools.workstation.desktop_gateway import DesktopGateway
        from neurostation_diagnostics import DiagnosticStore

        self.temp = tempfile.TemporaryDirectory()
        self.gateway = DesktopGateway(
            protocol_path=PROTOCOL,
            channel_config_path=CHANNELS,
            dataset_root=Path(self.temp.name) / "Datasets",
        )
        self.start_calls = []
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
        self.temp.cleanup()

    def _stub_start(self, config, speed=1):
        """Record routing without creating samples or a dataset."""

        from apps.workstation_ui.gateway import Phase, TaskSnapshot

        self.start_calls.append((config, speed))
        acquisition = self.gateway._acquisition
        acquisition.config = config
        acquisition._snapshot = TaskSnapshot(
            phase=Phase.COUNTDOWN,
            countdown=5,
            remaining=config.recording_seconds,
            trial_count=config.trials,
            speed=1,
        )
        self.gateway._active = acquisition
        return acquisition.snapshot

    def _add_user(self, user_id="U0001", name="Test user"):
        from neurostation_contract import UserProfile

        self.gateway.add_user(UserProfile(
            user_id=user_id,
            name=name,
            age=30,
            medical_conditions=("none",),
        ))
        self.window.refresh_users()

    def test_navigation_language_and_static_render(self):
        from apps.workstation_ui.app import NAVIGATION

        self.assertNotIn("live", NAVIGATION)
        self.assertNotIn("live", self.window.pages)
        self.window.show()
        for page in NAVIGATION:
            self.window.navigate(page)
            self.application.processEvents()
            self.assertEqual(self.window.current_page, page)
        self.window.change_language("en-US")
        self.assertEqual(self.window.tr("nav.apps"), "Acquisition apps")
        self.assertFalse(self.window.grab().isNull())

    def test_ssvep_form_is_cyton_only_and_runs_at_real_time(self):
        from apps.workstation_ui.gateway import CaptureMode

        page = self.window.pages["ssvep"]
        self.assertEqual(1, page.mode.count())
        self.assertEqual(CaptureMode.CYTON, page.mode.currentData())
        self.assertFalse(hasattr(page, "speed"))
        self.assertEqual(CaptureMode.CYTON, page.config().mode)
        self.assertTrue(page.port.isEnabled())
        self.assertFalse(page.channel_auto_label.isHidden())

        page.channel_manual.setChecked(True)
        channel_path = ROOT / "configs" / "channel_config_v1_auto.json"
        page.channel.setText(str(channel_path))
        self.assertEqual(channel_path, page.config().channel_config)

    def test_device_page_has_one_test_action_per_channel(self):
        self.window.navigate("devices")
        page = self.window.pages["devices"]
        self.assertEqual(8, page.calibration_table.rowCount())
        self.assertEqual(8, len(page.test_buttons))
        self.assertEqual(8, page.calibration_channel_selector.count())
        self.assertIsNotNone(page.head_map)
        self.assertIsNotNone(page.calibration_signal)
        self.assertEqual("开始脑电测试", page.calibration_test_toggle.text())
        self.assertEqual(
            ("Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2"),
            tuple(box.currentText() for box in page.position_boxes),
        )
        self.assertTrue(all(button.isEnabled() for button in page.test_buttons))

        page._select_calibration_channel(2)
        page._assign_map_position("C3")
        self.assertEqual("C3", page.position_boxes[2].currentText())
        page.append_channel_calibration_samples({
            "channel": 3,
            "channels": [[1.0], [2.0], [3.0, 4.0], [4.0], [5.0], [6.0], [7.0], [8.0]],
        })
        self.assertEqual(2, page.calibration_signal.sample_count)
        self.assertEqual(1.0, page.calibration_signal.variation)

        started = []
        stopped = []
        page.calibration_test_requested.disconnect()
        page.calibration_test_stop_requested.disconnect()
        page.calibration_test_requested.connect(started.append)
        page.calibration_test_stop_requested.connect(lambda: stopped.append(True))
        page._select_calibration_channel(2)
        page.calibration_test_toggle.click()
        self.assertEqual([2], started)

        page.set_channel_test_busy(2, True)
        self.assertEqual("停止脑电测试", page.calibration_test_toggle.text())
        page.calibration_test_toggle.click()
        self.assertEqual([True], stopped)
        self.assertTrue(all(not button.isEnabled() for button in page.test_buttons))
        page.set_channel_test_busy(-1, False)
        self.assertEqual("开始脑电测试", page.calibration_test_toggle.text())
        self.assertTrue(all(button.isEnabled() for button in page.test_buttons))

        page.set_channel_test_result(2, {
            "status": "passed",
            "detail": "ok",
            "metrics": {
                "finite_fraction": 1.0,
                "flat_fraction": 0.0,
                "saturation_fraction": 0.0,
            },
        })
        self.assertIn("通过", page.test_statuses[2].text())

    def test_capture_form_requires_a_real_user_and_photosensitivity_ack(self):
        from apps.workstation_ui.gateway import CaptureMode

        page = self.window.pages["ssvep"]
        self.assertEqual("", page.eye_side.currentData())
        self.assertIn("屏幕映射不可用", page.screen_mapping.text())
        self.assertIn("选择眼别", page.dataset_preview.text())
        page._start()
        self.assertTrue(page.error.text())
        self._add_user()
        page.acknowledge.setChecked(True)
        page._start()
        self.assertEqual(self.window.tr("validation.eye_side"), page.error.text())
        page.eye_side.setCurrentIndex(page.eye_side.findData("left"))
        self.assertIn("_左眼", page.dataset_preview.text())
        config = page.config()
        self.assertEqual(CaptureMode.CYTON, config.mode)
        config.validate()

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

    def test_cyton_start_runs_preflight_before_routing_to_worker(self):
        from apps.workstation_ui.gateway import CaptureConfig, CaptureMode, Phase

        self._add_user()
        self.gateway.preflight_cyton = lambda port="AUTO": {
            "status": "passed",
            "selected_port": port,
            "checks": [],
        }
        self.gateway.start_ssvep = self._stub_start
        config = CaptureConfig(
            mode=CaptureMode.CYTON,
            port="COM5",
            user_id="U0001",
            participant="U0001",
            name="real hardware routing",
            eye_side="left",
            acknowledge_flicker_risk=True,
        )
        self.window.start_ssvep(config, 1)
        for _ in range(50):
            self.application.processEvents()
            if self.window.preflight_thread is None:
                break
        self.assertEqual(1, len(self.start_calls))
        self.assertEqual("COM5", self.start_calls[0][0].port)
        self.assertEqual(1, self.start_calls[0][1])
        self.assertEqual(Phase.COUNTDOWN, self.gateway.snapshot.phase)

    def test_cyton_failed_preflight_blocks_worker_routing(self):
        from apps.workstation_ui.gateway import CaptureConfig, CaptureMode

        self._add_user()
        errors = []
        self.window._error = errors.append
        self.gateway.start_ssvep = self._stub_start
        self.gateway.preflight_cyton = lambda port="AUTO": {
            "status": "failed",
            "error": "no device",
            "checks": [],
        }
        config = CaptureConfig(
            mode=CaptureMode.CYTON,
            port="COM9",
            user_id="U0001",
            participant="U0001",
            name="blocked hardware",
            eye_side="left",
            acknowledge_flicker_risk=True,
        )
        self.window.start_ssvep(config, 1)
        for _ in range(50):
            self.application.processEvents()
            if self.window.preflight_thread is None:
                break
        self.assertEqual([], self.start_calls)
        self.assertEqual(1, len(errors))

    def test_cyton_preflight_timestamp_jitter_does_not_show_repeated_warning(self):
        from apps.workstation_ui.gateway import CaptureConfig, CaptureMode, Phase

        self._add_user()
        self.gateway.start_ssvep = self._stub_start
        config = CaptureConfig(
            mode=CaptureMode.CYTON,
            port="COM5",
            user_id="U0001",
            participant="U0001",
            name="warning hardware",
            eye_side="left",
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
        self.assertEqual(Phase.COUNTDOWN, self.gateway.snapshot.phase)
        self.assertEqual("", self.window.pages["ssvep"].error.text())
        self.assertTrue(self.window.pages["task"].quality_notice.isHidden())

    def test_cyton_preflight_packet_warning_is_visible_when_worker_continues(self):
        from apps.workstation_ui.gateway import CaptureConfig, CaptureMode, Phase

        self._add_user()
        self.gateway.start_ssvep = self._stub_start
        config = CaptureConfig(
            mode=CaptureMode.CYTON,
            port="COM5",
            user_id="U0001",
            participant="U0001",
            name="packet warning hardware",
            eye_side="left",
            acknowledge_flicker_risk=True,
            allow_draft_hardware_config=False,
        )
        self.window._pending_ssvep = (config, 1)
        self.window._preflight_finished({
            "status": "warning",
            "error": "",
            "selected_port": "COM5",
            "checks": [
                {"name": "timestamps", "status": "warning", "metrics": {
                    "timestamp_gap_count": 22,
                    "timestamp_gap_ratio": 0.031,
                    "timestamp_diff_max_s": 0.3546,
                    "packet_sequence_available": True,
                    "packet_loss_count": 1,
                    "packet_sequence_mismatch_count": 1,
                    "packet_duplicate_count": 0,
                }},
                {"name": "packet_sequence", "status": "warning", "metrics": {
                    "packet_sequence_available": True,
                    "packet_loss_count": 1,
                    "packet_sequence_mismatch_count": 1,
                    "packet_duplicate_count": 0,
                }},
            ],
        })
        self.assertEqual(Phase.COUNTDOWN, self.gateway.snapshot.phase)
        self.assertIn("继续", self.window.pages["ssvep"].error.text())
        self.assertFalse(self.window.pages["task"].quality_notice.isHidden())

    def test_cyton_preflight_degraded_requires_review(self):
        from apps.workstation_ui.gateway import CaptureConfig, CaptureMode, Phase

        self._add_user()
        config = CaptureConfig(
            mode=CaptureMode.CYTON,
            port="COM5",
            user_id="U0001",
            participant="U0001",
            name="degraded hardware",
            eye_side="left",
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

    def test_task_page_exposes_current_trial_banner(self):
        from apps.workstation_ui.gateway import CaptureConfig, Phase, TaskSnapshot

        config = CaptureConfig(
            participant="U0001",
            user_id="U0001",
            name="task view",
            eye_side="left",
            acknowledge_flicker_risk=True,
        )
        self.window.pages["task"].update_snapshot(
            TaskSnapshot(
                phase=Phase.RUNNING,
                trial=2,
                trial_count=12,
                frequency=12,
                elapsed=8,
                remaining=85,
                progress=8.6,
            ),
            config,
        )
        self.assertIn("/ 12", self.window.pages["task"].trial_banner.text())

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
        from eeg_tools.workstation.desktop_gateway import DesktopGateway

        settings_path = Path(self.temp.name) / "settings.ini"
        dataset_path = Path(self.temp.name) / "persistent-datasets"
        settings = QSettings(str(settings_path), QSettings.Format.IniFormat)
        first_gateway = DesktopGateway(
            protocol_path=PROTOCOL,
            channel_config_path=CHANNELS,
            dataset_root=Path(self.temp.name) / "first-datasets",
        )
        first = MainWindow(
            first_gateway,
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

        second_gateway = DesktopGateway(
            protocol_path=PROTOCOL,
            channel_config_path=CHANNELS,
            dataset_root=Path(self.temp.name) / "second-datasets",
        )
        second = MainWindow(
            second_gateway,
            locale=None,
            timer_enabled=False,
            persist_settings=True,
            settings=QSettings(str(settings_path), QSettings.Format.IniFormat),
        )
        self.assertEqual("en-US", second.tr.locale)
        self.assertEqual(dataset_path, second.draft_config.save_directory)
        self.assertEqual((780, 650), (second.width(), second.height()))
        second.close()

    def test_dataset_filters_persist_when_page_rebuilt(self):
        self.window.navigate("datasets")
        page = self.window.pages["datasets"]
        page.search.setText("U0001")
        page.source_filter.setCurrentIndex(page.source_filter.findData("cyton"))
        page.status_filter.setCurrentIndex(page.status_filter.findData("completed"))
        self.window.navigate("apps")
        self.window.navigate("datasets")
        page = self.window.pages["datasets"]
        self.assertEqual("U0001", page.search.text())
        self.assertEqual("cyton", page.source_filter.currentData())
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
