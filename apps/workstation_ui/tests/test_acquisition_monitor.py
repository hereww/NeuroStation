"""Secondary-screen selection, real-preview statistics and Qt integration tests."""
from dataclasses import replace
import importlib.util
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from apps.workstation_ui.monitor_model import AcquisitionMonitorModel
from neurostation_display import resolve_monitor_screen

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_QT = importlib.util.find_spec("PySide6") is not None
ROOT = Path(__file__).resolve().parents[3]


class MonitorModelTests(unittest.TestCase):
    def test_real_samples_and_centered_rms_without_mutation(self):
        model = AcquisitionMonitorModel()
        model.configure(("CH1", "CH2"), 250)
        samples = [[10.0, 12.0, 14.0], [float("nan"), float("inf"), 3.0]]
        model.append(samples)
        first, second = model.metrics()
        self.assertEqual(12.0, first.mean_uv)
        self.assertAlmostEqual(math.sqrt(8 / 3), first.rms_uv)
        self.assertEqual(4.0, first.peak_to_peak_uv)
        self.assertEqual((1, 2), (second.valid_samples, second.invalid_samples))
        self.assertEqual(2, model.invalid_value_count)
        self.assertEqual(3, model.sample_count)
        self.assertEqual([10.0, 12.0, 14.0], samples[0])
        self.assertTrue(math.isnan(samples[1][0]))

    def test_window_is_bounded_and_empty_is_not_zero(self):
        model = AcquisitionMonitorModel()
        model.configure(("CH1",), 250)
        self.assertIsNone(model.metrics()[0].rms_uv)
        model.append([list(range(600))])
        self.assertEqual(500, len(model.buffers[0]))
        self.assertEqual(600, model.sample_count)
        self.assertEqual(499, model.metrics()[0].peak_to_peak_uv)
        model.configure(("CH1",), 250)
        self.assertEqual(600, model.sample_count)
        model.configure(("CH2",), 125)
        self.assertEqual(0, model.sample_count)
        model.append([[float("nan")]])
        self.assertIsNone(model.metrics()[0].mean_uv)

    def test_rejects_malformed_batches_atomically(self):
        model = AcquisitionMonitorModel()
        model.configure(("CH1", "CH2"), 250)
        for samples in ([[1]], [[1], [2, 3]], [[1], ["bad"]]):
            with self.assertRaises(ValueError):
                model.append(samples)
            self.assertEqual(0, model.sample_count)
            self.assertTrue(all(not buffer for buffer in model.buffers))


class _Rect:
    def __init__(self, x, y=0, width=1920, height=1080):
        self.values = (x, y, width, height)

    def x(self): return self.values[0]
    def y(self): return self.values[1]
    def width(self): return self.values[2]
    def height(self): return self.values[3]


class _Screen:
    def __init__(self, x, **geometry): self.rect = _Rect(x, **geometry)
    def geometry(self): return self.rect


class _Application:
    def __init__(self, screens, primary): self.values, self.primary = screens, primary
    def screens(self): return self.values
    def primaryScreen(self): return self.primary


class MonitorScreenTests(unittest.TestCase):
    def test_primary_does_not_need_to_be_first_and_negative_positions_work(self):
        secondary, primary = _Screen(-1920), _Screen(0)
        app = _Application([secondary, primary], primary)
        self.assertIs(secondary, resolve_monitor_screen(app))
        self.assertIs(secondary, resolve_monitor_screen(app, 1))
        self.assertIs(secondary, resolve_monitor_screen(app, 9))

    def test_single_mirrored_and_overlapping_screens_do_not_cover_stimulus(self):
        primary = _Screen(0)
        for screens in ([primary], [primary, _Screen(0)], [primary, _Screen(100)]):
            self.assertIsNone(resolve_monitor_screen(_Application(screens, primary)))

    def test_stale_worker_index_never_redirects_monitor_to_primary(self):
        secondary, primary = _Screen(-1920), _Screen(0)
        app = _Application([secondary, primary], primary)
        self.assertIsNone(resolve_monitor_screen(app, 0))

    def test_skips_unusable_screens_and_honors_vertical_extended_layout(self):
        primary, monitor = _Screen(0), _Screen(0, y=-1080)
        app = _Application([primary, _Screen(1920, width=0), monitor], primary)
        self.assertIs(monitor, resolve_monitor_screen(app))


@unittest.skipUnless(HAS_QT, "PySide6 not installed")
class MonitorWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        from apps.workstation_ui.app import MainWindow
        from eeg_tools.workstation.desktop_gateway import DesktopGateway
        from neurostation_contract import CaptureConfig, Phase, TaskSnapshot
        from neurostation_diagnostics import DiagnosticStore

        self.temp = tempfile.TemporaryDirectory()
        self.gateway = DesktopGateway(
            protocol_path=ROOT / "configs/protocols/ssvep_four_target_v2.json",
            channel_config_path=ROOT / "configs/channel_config_v1_auto.json",
            dataset_root=Path(self.temp.name) / "Datasets",
        )
        self.window = MainWindow(self.gateway, timer_enabled=False, diagnostic_store=DiagnosticStore(persist=False))
        self.config = CaptureConfig(participant="U0001", name="Monitor test", eye_side="left", acknowledge_flicker_risk=True)
        acquisition = self.gateway._acquisition
        acquisition.config = self.config
        acquisition._snapshot = TaskSnapshot(phase=Phase.RUNNING, trial=2, trial_count=12, frequency=12, elapsed=8, remaining=85)
        self.gateway._active = acquisition
        self.secondary = patch("apps.workstation_ui.app.resolve_monitor_screen", return_value=self.application.primaryScreen())
        self.secondary.start()  # Route a logical second screen onto offscreen Qt for UI-only tests.
        self.window._update_controls()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.application.processEvents()
        self.secondary.stop()
        self.temp.cleanup()

    def test_one_reader_fans_out_identical_samples_even_with_task_page_hidden(self):
        payload = {"channel_names": ["CH1", "CH2"], "sample_rate_hz": 250, "samples": [[1, 2, 3], [4, 5, 6]]}
        with patch.object(self.gateway, "read_live_waveform", return_value=payload) as reader:
            self.window.navigate("devices")
            self.window.poll()
            reader.assert_called_once_with(1000)
        main_model = self.window.pages["task"].waveform.model
        monitor = self.window.monitor_window
        self.assertEqual(3, main_model.sample_index)
        self.assertEqual(3, monitor.waveform.model.sample_index)
        self.assertEqual(3, monitor.model.sample_count)
        self.assertEqual(tuple(main_model.raw_buffers[0]), tuple(monitor.model.buffers[0]))
        self.assertIsNone(self.window.pages["task"].waveform._sample_provider)
        self.assertEqual("CH1", monitor.table.item(0, 0).text())
        self.assertIn("12 Hz", monitor.task.values[4].text())
        self.assertIn("正在接收", monitor.status.text())

    def test_empty_or_malformed_payload_does_not_invent_samples(self):
        for payload in (None, {}, {"samples": [[1]], "channel_names": ["A", "B"], "sample_rate_hz": 250}):
            with patch.object(self.gateway, "read_live_waveform", return_value=payload):
                self.window.poll()
        monitor = self.window.monitor_window
        self.assertEqual(0, monitor.model.sample_count)
        self.assertEqual(0, monitor.table.rowCount())
        self.assertEqual(0, monitor.waveform.model.visible_sample_count)
        self.assertIn("等待", monitor.status.text())

    def test_dismiss_reopen_and_finish_do_not_create_another_task_timer(self):
        from neurostation_contract import Phase
        monitor = self.window.monitor_window
        monitor.close()
        self.window.poll()
        self.assertFalse(monitor.isVisible())
        self.assertTrue(self.gateway.snapshot.active)
        self.window.pages["task"].monitor_button.click()
        self.assertIs(monitor, self.window.monitor_window)
        self.assertTrue(monitor.isVisible())
        self.assertFalse(self.window.timer.isActive())
        self.gateway._acquisition._snapshot = replace(self.gateway.snapshot, phase=Phase.COMPLETED)
        self.window.poll()
        self.assertFalse(monitor.isVisible())
        self.assertFalse(monitor.waveform._timer.isActive())
        self.assertFalse(self.window.pages["task"].monitor_button.isEnabled())

    def test_single_screen_and_disconnect_leave_capture_untouched(self):
        monitor = self.window.monitor_window
        with patch("apps.workstation_ui.app.resolve_monitor_screen", return_value=None):
            self.window.poll()
            self.assertFalse(monitor.isVisible())
            self.assertTrue(self.gateway.snapshot.active)
            self.assertFalse(self.window.pages["task"].monitor_button.isEnabled())
            self.assertIn("未检测", self.window.pages["task"].monitor_notice.text())
        self.window.poll()
        self.assertTrue(monitor.isVisible())

    def test_freshness_warning_and_language_switch(self):
        monitor = self.window.monitor_window
        monitor.last_sample_at = 1.0
        with patch("apps.workstation_ui.monitor.time.monotonic", return_value=4.0):
            self.window._update_controls()
        self.assertIn("超过 2 秒", monitor.status.text())
        self.window.change_language("en-US")
        self.assertEqual("EEG acquisition monitor", self.window.monitor_window.windowTitle())
        self.assertEqual("Open secondary monitor", self.window.pages["task"].monitor_button.text())

    def test_hotplug_hides_immediately_and_does_not_dereference_removed_screen(self):
        monitor = self.window.monitor_window
        screen = monitor.target_screen
        self.window._monitor_screen_removed(screen)
        self.assertFalse(monitor.isVisible())
        self.assertIsNone(monitor.target_screen)
        self.assertEqual("", monitor.target_screen_name)
        self.assertTrue(self.gateway.snapshot.active)

    def test_language_change_keeps_manually_closed_monitor_closed(self):
        self.window.monitor_window.close()
        self.window.change_language("en-US")
        self.assertTrue(self._monitor_is_closed())
        self.window.show_acquisition_monitor()
        self.assertTrue(self.window.monitor_window.isVisible())

    def _monitor_is_closed(self):
        monitor = self.window.monitor_window
        return monitor is None or not monitor.isVisible()

    def test_monitor_stop_uses_existing_cancel_route(self):
        from neurostation_contract import Phase
        monitor = self.window.monitor_window
        def cancel():
            self.gateway._acquisition._snapshot = replace(self.gateway.snapshot, phase=Phase.CANCELLED)
            return self.gateway.snapshot
        with patch.object(self.gateway, "cancel", side_effect=cancel) as cancel_call:
            monitor.cancel_requested.emit()
            cancel_call.assert_called_once_with()
        self.assertFalse(self.gateway.snapshot.active)
        self.assertFalse(monitor.isVisible())

    def test_new_capture_resets_preview_counts(self):
        self.window.monitor_window.update_samples(["A"], 250, [[1, 2, 3]])
        def start(config, speed):
            self.gateway._acquisition.config = config
            self.gateway._acquisition._snapshot = replace(self.gateway.snapshot, trial=0, elapsed=0)
            return self.gateway.snapshot
        with patch.object(self.gateway, "start_ssvep", side_effect=start):
            self.window.start_ssvep(self.config, 1, _preflight_ready=True)
        self.assertEqual(0, self.window.monitor_window.model.sample_count)
