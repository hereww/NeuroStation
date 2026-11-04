"""Offscreen logic tests only: no physical calibration, flashing or headset I/O."""
from __future__ import annotations

from copy import deepcopy
import json
import math
import os
import time
import unittest
from unittest.mock import MagicMock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QEvent, QRect, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from apps.realtime_command_display import (
    CALIBRATION_FRAMES, WARMUP_FRAMES, CommandStimulusWindow, TARGETS,
    frame_plan, identity_failure, target_rectangles, timing_failure, timing_statistics,
)
from neurostation_display import ssvep_frame_is_lit


def synthetic_identity() -> dict:
    """Unit-test fixture, never a physical screen acceptance record."""
    geometry = {"x": 1920, "y": 0, "width": 1920, "height": 1080}
    return {
        "window_token": "synthetic-test-only", "context_generation": 1,
        "platform": "windows", "screen": {"name": "synthetic-test-only",
        "geometry": geometry, "refresh_rate_hz": 60.0},
        "window_geometry": deepcopy(geometry), "fullscreen": True,
        "selected_screen_matches": True,
        "graphics": {"context_valid": True, "vendor": "synthetic-vendor",
                     "renderer": "synthetic-hardware-fixture", "version": "synthetic-version",
                     "swap_interval": 1},
    }


class FramePlanTests(unittest.TestCase):
    def test_order_frequencies_commands_and_shared_frame_algorithm(self):
        self.assertEqual([(t.command, t.frequency_hz) for t in TARGETS],
                         [("MOVE_LEFT", 10), ("SELECT", 15), ("MOVE_RIGHT", 12)])
        for index in range(600):
            plan = frame_plan(index)
            self.assertEqual(len(plan), 3)
            for target in TARGETS:
                self.assertEqual(plan[target.command], ssvep_frame_is_lit(index, target.frequency_hz, 60))
        self.assertEqual(frame_plan(0), {t.command: True for t in TARGETS})
        self.assertGreater(len({tuple(frame_plan(i).values()) for i in range(60)}), 3)

    def test_cycle_counts_and_duty_cycle(self):
        for target in TARGETS:
            samples = [frame_plan(i)[target.command] for i in range(60)]
            self.assertEqual(sum(samples), 30)
            self.assertEqual(sum(samples[i] != samples[i - 1] for i in range(60)), 2 * target.frequency_hz)

    def test_negative_frame_rejected(self):
        with self.assertRaises(ValueError):
            frame_plan(-1)

    def test_target_rects_are_ordered_disjoint_and_fit(self):
        for width, height in [(1920, 1080), (1280, 720), (3840, 2160)]:
            rects = target_rectangles(width, height)
            for rect in rects:
                self.assertTrue(QRect(0, 0, width, height).contains(rect))
            self.assertLess(rects[0].right(), rects[1].left())
            self.assertLess(rects[1].right(), rects[2].left())


class TimingTests(unittest.TestCase):
    def test_median_percentiles_and_refresh(self):
        stats = timing_statistics([1 / 60] * 150)
        self.assertEqual(stats["interval_count"], 150)
        self.assertAlmostEqual(stats["median_interval_s"], 1 / 60)
        self.assertAlmostEqual(stats["p95_interval_s"], 1 / 60)
        self.assertAlmostEqual(stats["p99_interval_s"], 1 / 60)
        self.assertAlmostEqual(stats["measured_refresh_hz"], 60)
        self.assertEqual(stats["dropped_frames"], 0)
        self.assertEqual(timing_failure(stats), "")
        self.assertNotIn("passed", stats)

    def test_drop_budget_counts_missing_refresh_slots(self):
        one = timing_statistics([1 / 60] * 149 + [2 / 60])
        two = timing_statistics([1 / 60] * 148 + [2 / 60] * 2)
        self.assertEqual(one["dropped_frames"], 1)
        self.assertAlmostEqual(one["drop_ratio"], 1 / 151)
        self.assertEqual(timing_failure(one), "")
        self.assertIn("1%", timing_failure(two))
        exact = timing_statistics([1 / 60] * 98 + [2 / 60])
        self.assertEqual(exact["drop_ratio"], 0.01)
        self.assertEqual(timing_failure(exact), "")

    def test_refresh_tolerance(self):
        for hz in (59, 60, 61):
            self.assertEqual(timing_failure(timing_statistics([1 / hz] * 150)), "")
        for hz in (58, 62, 75):
            self.assertIn("刷新率", timing_failure(timing_statistics([1 / hz] * 150)))

    def test_invalid_intervals_fail_closed(self):
        for samples in ([], [0], [-1], [math.nan], [math.inf], [1 / 60, math.nan]):
            stats = timing_statistics(samples)
            self.assertIsNone(stats["measured_refresh_hz"])
            self.assertTrue(timing_failure(stats))
            json.dumps(stats, allow_nan=False)

    def test_quantiles_use_linear_interpolation(self):
        stats = timing_statistics([0.01, 0.02, 0.03, 0.04, 0.05])
        self.assertAlmostEqual(stats["p95_interval_s"], 0.048)
        self.assertAlmostEqual(stats["p99_interval_s"], 0.0496)


class IdentityTests(unittest.TestCase):
    def test_synthetic_identity_only_exercises_gate_logic(self):
        self.assertEqual(identity_failure(synthetic_identity()), "")

    def test_offscreen_minimal_and_unknown_rejected_even_with_gpu_strings(self):
        for platform in ("offscreen", "minimal", "vnc", "unknown", ""):
            identity = synthetic_identity()
            identity["platform"] = platform
            self.assertTrue(identity_failure(identity))

    def test_software_renderer_context_strings_and_vsync_fail_closed(self):
        for renderer in ("llvmpipe (LLVM)", "SwiftShader", "GDI Generic", "Microsoft Basic Render Driver",
                         "softpipe", "Software Rasterizer", "swrast"):
            identity = synthetic_identity()
            identity["graphics"]["renderer"] = renderer
            self.assertTrue(identity_failure(identity))
        for key, value in (("context_valid", False), ("vendor", ""), ("version", ""),
                           ("renderer", ""), ("swap_interval", 0), ("swap_interval", -1)):
            identity = synthetic_identity()
            identity["graphics"][key] = value
            self.assertTrue(identity_failure(identity))

    def test_screen_identity_must_be_fullscreen_on_selected_60hz_display(self):
        for key, value in (("fullscreen", False), ("selected_screen_matches", False)):
            identity = synthetic_identity()
            identity[key] = value
            self.assertTrue(identity_failure(identity))
        identity = synthetic_identity()
        identity["window_geometry"]["x"] += 1
        self.assertTrue(identity_failure(identity))
        for hz in (0, 58, 62, 120, math.nan):
            identity = synthetic_identity()
            identity["screen"]["refresh_rate_hz"] = hz
            self.assertTrue(identity_failure(identity))


class WindowLogicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        if cls.app.platformName() != "offscreen":
            raise unittest.SkipTest("these tests require the offscreen platform")

    def setUp(self):
        self.window = CommandStimulusWindow(self.app.primaryScreen())
        self.reports, self.failures, self.records, self.aborts, self.starts = [], [], [], [], []
        self.window.calibration_finished.connect(self.reports.append)
        self.window.timing_failed.connect(self.failures.append)
        self.window.frame_record.connect(self.records.append)
        self.window.aborted.connect(lambda: self.aborts.append(True))
        self.window.start_requested.connect(lambda: self.starts.append(True))

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def queue_synthetic_swap(self, now):
        window = self.window
        window._pending_frame = {
            "mode": window._mode, "generation": window._generation,
            "frame_index": window._next_frame_index,
            "target_lit": frame_plan(window._next_frame_index) if window._mode == "flicker" else None,
            "identity": window.snapshot_identity(),
        }
        with patch("apps.realtime_command_display.time.perf_counter", return_value=now):
            window._on_frame_swapped()

    def test_constructor_is_staged_static_and_never_claims_pass(self):
        self.assertFalse(self.window.isVisible())
        self.assertEqual(self.window.mode, "idle")
        self.assertIsNone(self.window.calibration_report)
        self.assertEqual(self.window.format().swapInterval(), 1)
        identity = self.window.snapshot_identity()
        self.assertEqual(identity["platform"], "offscreen")
        self.assertFalse(identity["graphics"]["context_valid"])
        json.dumps(identity, allow_nan=False)

    def test_offscreen_cannot_start_calibration_or_flicker(self):
        self.assertFalse(self.window.start_calibration())
        self.assertFalse(self.window.isVisible())
        self.assertEqual(len(self.reports), 1)
        report = self.reports[0]
        self.assertFalse(report["passed"])
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["frames_observed"], 0)
        self.assertIsNone(report["median_interval_s"])
        self.assertFalse(self.window.start_flicker())
        self.assertEqual(self.window.mode, "idle")
        json.dumps(report, allow_nan=False)

    def test_report_signal_and_property_are_copies(self):
        self.window.start_calibration()
        self.reports[0]["passed"] = True
        report = self.window.calibration_report
        self.assertFalse(report["passed"])
        report["passed"] = True
        self.assertFalse(self.window.calibration_report["passed"])
        self.assertFalse(self.window.start_flicker())

    def test_fake_passed_dict_cannot_override_offscreen_hardware_rejection(self):
        self.window._passed_calibration = {"passed": True, "identity": self.window.snapshot_identity(),
                                           "completed_monotonic_s": time.perf_counter()}
        self.assertFalse(self.window.start_flicker())
        self.assertIsNone(self.window._passed_calibration)

    def test_different_window_context_refresh_or_geometry_identity_rejected(self):
        for field, changed in (("window_token", "another-window"), ("context_generation", 9),
                               ("device_pixel_ratio", 2.0), ("fullscreen", True)):
            identity = self.window.snapshot_identity()
            identity[field] = changed
            self.window._passed_calibration = {"passed": True, "identity": identity,
                                               "completed_monotonic_s": time.perf_counter()}
            self.assertFalse(self.window.start_flicker())
            self.assertIn("改变", self.failures[-1])

    def test_static_status_does_not_change_identity_or_emit_start(self):
        identity = self.window.snapshot_identity()
        self.window.set_status_text("尚未闪烁，等待采集就绪")
        self.assertEqual(self.window.snapshot_identity(), identity)
        self.assertIn("等待采集", self.window.status_text)
        self.assertEqual(self.starts, [])

    def test_calibration_uses_180_swaps_discards_first_30_and_stays_static(self):
        window = self.window
        window._reset_frames("calibration")
        window._started_s = 0.0
        window._calibration_identity = window.snapshot_identity()
        # Bypass only the platform gate for callback arithmetic, never show a surface.
        with patch.object(window, "_presentation_failure", return_value=""):
            for index in range(CALIBRATION_FRAMES):
                self.queue_synthetic_swap((index + 1) / 60)
        report = self.reports[-1]
        self.assertEqual(report["frames_observed"], 180)
        self.assertEqual(report["warmup_frames"], 30)
        self.assertEqual(report["interval_count"], 150)
        self.assertEqual(len(self.records), 180)
        self.assertTrue(all(record["target_lit"] is None for record in self.records))
        self.assertEqual(window.mode, "idle")
        self.assertFalse(window.isVisible())
        self.assertEqual(self.starts, [])
        # The actual public gate still rejects the synthetic report on offscreen.
        self.assertFalse(window.start_flicker())

    def test_warmup_outliers_are_not_in_calibration_interval_stats(self):
        window = self.window
        window._reset_frames("calibration")
        window._started_s = 0.0
        window._calibration_identity = window.snapshot_identity()
        now = 0.0
        with patch.object(window, "_presentation_failure", return_value=""):
            for index in range(180):
                now += 0.04 if index < WARMUP_FRAMES else 1 / 60
                self.queue_synthetic_swap(now)
        self.assertEqual(self.reports[-1]["dropped_frames"], 0)
        self.assertGreater(self.records[-1]["dropped_frames"], 0)
        self.assertAlmostEqual(self.reports[-1]["median_interval_s"], 1 / 60)

    def test_calibration_timeout_without_a_single_swap(self):
        self.window._reset_frames("calibration")
        self.window._started_s = 1.0
        with patch("apps.realtime_command_display.time.perf_counter", return_value=13.001):
            self.window._check_calibration_timeout()
        self.assertFalse(self.reports[-1]["passed"])
        self.assertIn("12 秒", self.reports[-1]["reason"])
        self.assertEqual(self.window.mode, "idle")

    def test_late_180th_swap_does_not_pass_after_timeout(self):
        window = self.window
        window._reset_frames("calibration")
        window._started_s = 0.0
        window._frames_observed = 179
        with patch.object(window, "_presentation_failure", return_value=""):
            self.queue_synthetic_swap(12.01)
        self.assertFalse(self.reports[-1]["passed"])
        self.assertIn("12 秒", self.reports[-1]["reason"])

    def test_flicker_gap_stops_before_logging_and_never_schedules_another_frame(self):
        window = self.window
        window._reset_frames("flicker")
        window._started_s = 1.0
        window._last_swap_s = 1.0
        log_modes = []
        window.frame_record.connect(lambda _: log_modes.append(window.mode))
        with patch.object(window, "_presentation_failure", return_value=""):
            self.queue_synthetic_swap(1.101)
        self.assertEqual(window.mode, "idle")
        self.assertIn("0.10", self.failures[-1])
        self.assertEqual(log_modes, ["idle"])
        self.assertFalse(window._swap_watchdog.isActive())
        self.assertIsNone(window._pending_frame)
        self.assertEqual(self.records[-1]["target_lit"], frame_plan(0))

    def test_no_swap_watchdog_checks_first_frame_and_later_frames(self):
        for last in (None, 1.02):
            self.window._reset_frames("flicker")
            self.window._started_s = 1.0
            self.window._last_swap_s = last
            with patch("apps.realtime_command_display.time.perf_counter", return_value=1.13):
                self.window._check_swap_timeout()
            self.assertEqual(self.window.mode, "idle")
            self.assertIn("0.10", self.failures[-1])

    def test_every_120_flicker_frames_checks_drops(self):
        window = self.window
        window._reset_frames("flicker")
        window._started_s = 0.0
        now = 0.0
        with patch.object(window, "_presentation_failure", return_value=""):
            for index in range(120):
                now += 2 / 60 if index in (10, 40) else 1 / 60
                self.queue_synthetic_swap(now)
        self.assertEqual(window.mode, "idle")
        self.assertIn("1%", self.failures[-1])
        self.assertEqual(self.records[-1]["dropped_frames"], 2)
        self.assertEqual(self.records[11]["frame_index"], 12)

    def test_reentrant_log_handler_stop_does_not_schedule_next_frame(self):
        self.window._reset_frames("flicker")
        self.window._started_s = 0.0
        self.window.frame_record.connect(lambda _: self.window.stop_flicker())
        with patch.object(self.window, "_presentation_failure", return_value=""):
            self.queue_synthetic_swap(1 / 60)
        self.assertEqual(self.window.mode, "idle")
        self.assertFalse(self.window._swap_watchdog.isActive())
        self.assertEqual(self.window._next_frame_index, 0)

    def test_gpu_scissors_draw_all_three_same_frame_at_device_pixel_ratio(self):
        window = self.window
        window._reset_frames("flicker")
        window._next_frame_index = 7
        window._gl = MagicMock()
        device, painter = MagicMock(), MagicMock()
        with patch.object(window, "_presentation_failure", return_value=""), \
             patch.object(window, "width", return_value=1920), \
             patch.object(window, "height", return_value=1080), \
             patch.object(window, "devicePixelRatio", return_value=1.5), \
             patch("apps.realtime_command_display.QOpenGLPaintDevice", return_value=device), \
             patch("apps.realtime_command_display.QPainter", return_value=painter):
            window.paintGL()
        window._gl.glViewport.assert_called_once_with(0, 0, 2880, 1620)
        self.assertEqual(window._gl.glScissor.call_count, 3)
        self.assertEqual(window._gl.glClear.call_count, 4)
        self.assertEqual(window._pending_frame["target_lit"], frame_plan(7))
        for target, color in zip(TARGETS, window._gl.glClearColor.call_args_list[1:]):
            level = 1.0 if frame_plan(7)[target.command] else 0.0
            self.assertEqual(color.args, (level, level, level, 1.0))
        device.setDevicePixelRatio.assert_called_once_with(1.5)
        painter.end.assert_called_once()
        self.assertEqual(painter.drawText.call_count, 4)
        window._gl = None

    def test_start_requires_visible_exposed_foreground_even_with_matching_report(self):
        for reason in ("刺激窗口不可见或未曝光", "应用已失去前台活动状态"):
            self.window._passed_calibration = {"passed": True, "identity": self.window.snapshot_identity(),
                                           "completed_monotonic_s": time.perf_counter()}
            with patch.object(self.window, "_presentation_failure", return_value=reason):
                self.assertFalse(self.window.start_flicker())
            self.assertEqual(self.failures[-1], reason)
            self.assertIsNone(self.window._passed_calibration)

    def test_staged_invisible_window_ignores_app_deactivation(self):
        self.window._application_state_changed(Qt.ApplicationState.ApplicationInactive)
        self.assertEqual(self.aborts, [])
        self.assertFalse(self.window.isVisible())

    def test_stale_swap_is_not_logged(self):
        window = self.window
        window._reset_frames("flicker")
        window._pending_frame = {"generation": window._generation - 1, "mode": "flicker"}
        window._on_frame_swapped()
        self.assertEqual(self.records, [])

    def test_enter_space_are_explicit_start_requests_only_after_valid_calibration(self):
        for key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.window.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier))
        self.assertEqual(self.starts, [])
        self.window._passed_calibration = {"passed": True, "identity": self.window.snapshot_identity(),
                                           "completed_monotonic_s": time.perf_counter()}
        with patch.object(self.window, "_presentation_failure", return_value=""):
            for key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
                self.window.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier))
            self.window.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return,
                                                Qt.KeyboardModifier.NoModifier, "", True, 1))
        self.assertEqual(len(self.starts), 3)
        self.assertEqual(self.window.mode, "idle")
        self.assertFalse(self.window.isVisible())

    def install_synthetic_calibration(self, completed=700.0):
        report = {"passed": True, "status": "passed", "identity": self.window.snapshot_identity(),
                  "completed_monotonic_s": completed}
        self.window._passed_calibration = deepcopy(report)
        self.window._last_calibration = deepcopy(report)

    def test_freshness_boundary_gates_both_flicker_and_user_start_request(self):
        for now in (700.0, 999.999, 1000.0):
            with self.subTest(now=now), \
                 patch("apps.realtime_command_display.time.perf_counter", return_value=now), \
                 patch.object(self.window, "_presentation_failure", return_value=""):
                self.install_synthetic_calibration()
                self.assertTrue(self.window.calibration_valid)
                count = len(self.starts)
                self.window.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return,
                                                    Qt.KeyboardModifier.NoModifier))
                self.assertEqual(len(self.starts), count + 1)
                self.assertEqual(self.window.mode, "idle")
                self.assertTrue(self.window.start_flicker())
                self.assertEqual(self.window.mode, "flicker")
                self.assertFalse(self.window.isVisible())
                self.window.stop_flicker()

    def test_future_expired_or_invalid_time_rejects_both_start_paths(self):
        for completed in (699.999, 1000.001, None, math.nan, math.inf, -math.inf, True, "700"):
            for trigger in ("flicker", Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
                with self.subTest(completed=completed, trigger=trigger), \
                     patch("apps.realtime_command_display.time.perf_counter", return_value=1000.0), \
                     patch.object(self.window, "_presentation_failure", return_value=""):
                    self.install_synthetic_calibration(completed)
                    self.assertFalse(self.window.calibration_valid)
                    count = len(self.starts)
                    failures = len(self.failures)
                    if trigger == "flicker":
                        self.assertFalse(self.window.start_flicker())
                    else:
                        self.window.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, trigger,
                                                            Qt.KeyboardModifier.NoModifier))
                    self.assertEqual(len(self.starts), count)
                    self.assertEqual(len(self.failures), failures + 1)
                    self.assertEqual(self.window.mode, "idle")
                    self.assertIsNone(self.window._passed_calibration)
                    self.assertFalse(self.window._swap_watchdog.isActive())

    def test_missing_completion_and_invalid_current_clock_fail_closed(self):
        with patch.object(self.window, "_presentation_failure", return_value=""):
            self.install_synthetic_calibration()
            del self.window._passed_calibration["completed_monotonic_s"]
            self.assertFalse(self.window.calibration_valid)
            self.assertFalse(self.window.start_flicker())
            for now in (math.nan, math.inf, -math.inf):
                self.install_synthetic_calibration()
                with patch("apps.realtime_command_display.time.perf_counter", return_value=now):
                    self.assertFalse(self.window.calibration_valid)
                    self.assertFalse(self.window.start_flicker())

    def test_historical_pass_does_not_mean_calibration_currently_valid(self):
        with patch("apps.realtime_command_display.time.perf_counter", return_value=1000.0), \
             patch.object(self.window, "_presentation_failure", return_value=""):
            self.assertFalse(self.window.calibration_valid)
            self.install_synthetic_calibration()
            self.assertTrue(self.window.calibration_valid)
            with patch("apps.realtime_command_display.time.perf_counter", return_value=1000.001):
                self.assertFalse(self.window.calibration_valid)
                self.assertTrue(self.window.calibration_report["passed"])
            self.assertTrue(self.window.calibration_valid)
            valid_in_failed_handler = []
            self.window.timing_failed.connect(lambda _: valid_in_failed_handler.append(self.window.calibration_valid))
            self.window._fail("test timing failure")
            self.assertEqual(valid_in_failed_handler, [False])
            self.assertFalse(self.window.calibration_valid)
            self.assertTrue(self.window.calibration_report["passed"])
            self.install_synthetic_calibration()
            self.window._abort("test abort")
            self.assertFalse(self.window.calibration_valid)

    def test_calibration_valid_checks_identity_and_presentation(self):
        with patch("apps.realtime_command_display.time.perf_counter", return_value=1000.0):
            self.install_synthetic_calibration()
            self.assertFalse(self.window.calibration_valid)  # Real offscreen gate.
            with patch.object(self.window, "_presentation_failure", return_value=""):
                self.assertTrue(self.window.calibration_valid)
                self.window._passed_calibration["identity"]["context_generation"] += 1
                self.assertFalse(self.window.calibration_valid)

    def test_expired_repeated_start_flicker_does_not_bypass_gate(self):
        with patch.object(self.window, "_presentation_failure", return_value=""), \
             patch("apps.realtime_command_display.time.perf_counter", return_value=1000.001):
            self.install_synthetic_calibration()
            self.window._reset_frames("flicker")
            self.assertFalse(self.window.start_flicker())
            self.assertEqual(self.window.mode, "idle")
            self.assertIn("300", self.failures[-1])

    def test_escape_and_q_abort_once_and_invalidate_calibration(self):
        self.window._reset_frames("flicker")
        for key in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
            self.window.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier))
        self.assertEqual(self.aborts, [True])
        self.assertEqual(self.window.mode, "idle")
        self.assertIsNone(self.window._passed_calibration)

    def test_hide_loss_of_exposure_and_app_deactivation_abort(self):
        for cause in ("hide", "exposure", "app"):
            self.window._aborting = False
            self.window._reset_frames("flicker")
            self.window._passed_calibration = {"passed": True, "identity": self.window.snapshot_identity(),
                                           "completed_monotonic_s": time.perf_counter()}
            if cause == "hide":
                self.window.event(QEvent(QEvent.Type.Hide))
            elif cause == "exposure":
                self.window._exposed_once = True
                self.window.event(QEvent(QEvent.Type.Expose))
            else:
                self.window._application_state_changed(Qt.ApplicationState.ApplicationInactive)
            self.assertEqual(self.window.mode, "idle")
            self.assertIsNone(self.window._passed_calibration)
        self.assertEqual(len(self.aborts), 3)

    def test_screen_identity_change_invalidates_pass(self):
        self.window._passed_calibration = {"passed": True, "identity": synthetic_identity(),
                                           "completed_monotonic_s": time.perf_counter()}
        self.window._identity_changed()
        self.assertEqual(self.aborts, [True])
        self.assertIsNone(self.window._passed_calibration)
        self.assertFalse(self.window.start_flicker())


if __name__ == "__main__":
    unittest.main()
