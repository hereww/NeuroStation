"""Frame-swapped, three-target SSVEP surface; no acquisition or session I/O.

The controller owns consent, device/model readiness, capture and log persistence.
Connect ``start_requested`` to those checks, and call ``start_flicker`` only after
capture is ready. ``target_lit`` is stimulus audit data, never decoder input.
Constructing this window does not show it or start a stimulus. Calibration measures
software frame-swap callbacks; it does not verify optical output with a photodiode.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from statistics import median
import time
from typing import Any, Iterable

from PySide6.QtCore import QEvent, QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLPaintDevice, QOpenGLWindow

from neurostation_display import ssvep_frame_is_lit


@dataclass(frozen=True)
class CommandTarget:
    command: str
    frequency_hz: int
    title: str


TARGETS = (
    CommandTarget("MOVE_LEFT", 10, "左移"),
    CommandTarget("SELECT", 15, "选择"),
    CommandTarget("MOVE_RIGHT", 12, "右移"),
)
CALIBRATION_FRAMES = 180
WARMUP_FRAMES = 30
CALIBRATION_TIMEOUT_S = 12.0
CALIBRATION_MAX_AGE_S = 300.0
TIMING_CHECK_FRAMES = 120
MAX_GAP_S = 0.10
_NATIVE_PLATFORMS = {"windows", "xcb", "wayland", "cocoa"}
_SOFTWARE_RENDERERS = (
    "llvmpipe", "softpipe", "swiftshader", "software", "swrast",
    "microsoft basic render", "gdi generic", "lavapipe", "mesa offscreen",
)


def frame_plan(frame_index: int, refresh_rate_hz: int = 60) -> dict[str, bool]:
    """All three targets share this presentation frame and phase origin."""
    return {
        target.command: ssvep_frame_is_lit(frame_index, target.frequency_hz, refresh_rate_hz)
        for target in TARGETS
    }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def timing_statistics(intervals: Iterable[float], refresh_rate_hz: int = 60) -> dict[str, Any]:
    """Pure interval statistics, without claiming a physical display passed."""
    if refresh_rate_hz <= 0:
        raise ValueError("refresh_rate_hz must be positive")
    samples = [float(value) for value in intervals]
    valid = bool(samples) and all(math.isfinite(value) and value > 0 for value in samples)
    dropped = sum(max(0, math.floor(value * refresh_rate_hz + 0.5) - 1)
                  for value in samples) if valid else 0
    center = median(samples) if valid else None
    return {
        "interval_count": len(samples),
        "expected_refresh_hz": refresh_rate_hz,
        "measured_refresh_hz": 1.0 / center if center else None,
        "median_interval_s": center,
        "p95_interval_s": _percentile(samples, 0.95) if valid else None,
        "p99_interval_s": _percentile(samples, 0.99) if valid else None,
        "dropped_frames": dropped,
        "drop_ratio": dropped / (len(samples) + dropped) if valid else None,
        "valid_intervals": valid,
    }


def timing_failure(stats: dict[str, Any]) -> str:
    if not stats["valid_intervals"]:
        return "没有有效的帧交换间隔"
    if abs(stats["measured_refresh_hz"] - stats["expected_refresh_hz"]) > 1.0:
        return "实际刷新率与设定值不匹配（容差 ±1 Hz）"
    if stats["drop_ratio"] > 0.01:
        return "掉帧比例超过 1%"
    return ""


def identity_failure(identity: dict[str, Any], refresh_rate_hz: int = 60) -> str:
    """Reject unverifiable/non-native/software presentation before calibration."""
    if identity.get("platform") not in _NATIVE_PLATFORMS:
        return "屏幕校对需要原生显示平台，不能使用 offscreen/minimal 渲染"
    graphics = identity.get("graphics", {})
    if not graphics.get("context_valid") or not all(
        graphics.get(key) for key in ("vendor", "renderer", "version")
    ):
        return "缺少有效的真实 OpenGL 上下文信息"
    if any(name in str(graphics["renderer"]).lower() for name in _SOFTWARE_RENDERERS):
        return "屏幕校对不接受软件渲染器"
    if graphics.get("swap_interval") != 1:
        return "屏幕校对需要 swapInterval=1"
    screen = identity.get("screen", {})
    refresh = screen.get("refresh_rate_hz", 0)
    if not math.isfinite(refresh) or abs(refresh - refresh_rate_hz) > 1.0:
        return "屏幕报告的刷新率与设定值不匹配（容差 ±1 Hz）"
    if not identity.get("fullscreen") or identity.get("window_geometry") != screen.get("geometry"):
        return "刺激窗口必须覆盖所选屏幕全屏区域"
    if not identity.get("selected_screen_matches"):
        return "刺激窗口不在指定屏幕上"
    return ""


def _rect_identity(rect: QRect) -> dict[str, int]:
    return {"x": rect.x(), "y": rect.y(), "width": rect.width(), "height": rect.height()}


def target_rectangles(width: int, height: int) -> tuple[QRect, ...]:
    """Logical-pixel layout, converted to framebuffer pixels only by the renderer."""
    side = max(1, min(width // 5, height // 3))
    return tuple(QRect(round(width * (index + 0.5) / 3 - side / 2),
                       (height - side) // 2, side, side) for index in range(3))


class CommandStimulusWindow(QOpenGLWindow):
    calibration_finished = Signal(object)
    timing_failed = Signal(str)
    aborted = Signal()
    frame_record = Signal(object)
    start_requested = Signal()

    def __init__(self, screen, refresh_rate_hz: int = 60) -> None:
        if isinstance(refresh_rate_hz, bool) or not isinstance(refresh_rate_hz, int) or refresh_rate_hz < 30:
            raise ValueError("refresh_rate_hz must be an integer >= 30")
        if screen is None:
            raise ValueError("a specific QScreen is required")
        super().__init__(QOpenGLWindow.UpdateBehavior.NoPartialUpdate)
        self.refresh_rate_hz = refresh_rate_hz
        self._selected_screen = screen
        self._mode = "idle"
        self._generation = 0
        self._pending_frame: dict[str, Any] | None = None
        self._next_frame_index = 0
        self._frames_observed = 0
        self._last_swap_s: float | None = None
        self._started_s: float | None = None
        self._intervals: list[float] = []
        self._dropped_frames = 0
        self._calibration_identity: dict[str, Any] | None = None
        self._passed_calibration: dict[str, Any] | None = None
        self._last_calibration: dict[str, Any] | None = None
        self._gl = None
        self._graphics: dict[str, Any] = {}
        self._context_generation = 0
        self._aborting = False
        self._setup = False
        self._exposed_once = False
        self._status_text = "尚未闪烁，请先进行屏幕校对；Esc 退出"
        surface_format = QSurfaceFormat()
        surface_format.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
        surface_format.setSwapInterval(1)
        self.setFormat(surface_format)
        self.setScreen(screen)
        self.setTitle("NeuroStation 实时 SSVEP 屏幕校对")
        self.setCursor(Qt.CursorShape.BlankCursor)
        self._calibration_watchdog = QTimer(self)
        self._calibration_watchdog.setSingleShot(True)
        self._calibration_watchdog.setTimerType(Qt.TimerType.PreciseTimer)
        self._calibration_watchdog.timeout.connect(self._check_calibration_timeout)
        # Watchdogs only stop a failed surface; neither controls target phase.
        self._swap_watchdog = QTimer(self)
        self._swap_watchdog.setSingleShot(True)
        self._swap_watchdog.setTimerType(Qt.TimerType.PreciseTimer)
        self._swap_watchdog.timeout.connect(self._check_swap_timeout)
        self.frameSwapped.connect(self._on_frame_swapped)
        self.screenChanged.connect(self._identity_changed)
        self.xChanged.connect(self._identity_changed)
        self.yChanged.connect(self._identity_changed)
        self.widthChanged.connect(self._identity_changed)
        self.heightChanged.connect(self._identity_changed)
        self.windowStateChanged.connect(self._identity_changed)
        self.visibleChanged.connect(self._visibility_changed)
        screen.geometryChanged.connect(self._identity_changed)
        screen.refreshRateChanged.connect(self._identity_changed)
        screen.logicalDotsPerInchChanged.connect(self._identity_changed)
        screen.physicalDotsPerInchChanged.connect(self._identity_changed)
        application = QGuiApplication.instance()
        application.applicationStateChanged.connect(self._application_state_changed)
        application.screenRemoved.connect(self._screen_removed)
        application.aboutToQuit.connect(self._abort)

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def calibration_report(self) -> dict[str, Any] | None:
        """Historical software timing result; use calibration_valid to enable UI."""
        return deepcopy(self._last_calibration)

    @property
    def calibration_valid(self) -> bool:
        """Current freshness, identity and presentation gate; no side effects."""
        return not bool(self._calibration_failure())

    def _calibration_failure(self) -> str:
        report = self._passed_calibration
        if self._aborting or self._mode == "calibration" or not report or not report.get("passed"):
            return "本窗口尚无有效的屏幕校对通过记录"
        if report.get("identity") != self.snapshot_identity():
            return "屏幕或 OpenGL 上下文已改变，请重新校对"
        completed = report.get("completed_monotonic_s")
        if (isinstance(completed, bool) or not isinstance(completed, (int, float))
                or not math.isfinite(completed)):
            return "屏幕校对完成时间无效，请重新校对"
        age = time.perf_counter() - completed
        if not math.isfinite(age) or age < 0:
            return "屏幕校对完成时间在未来或时钟无效，请重新校对"
        if age > CALIBRATION_MAX_AGE_S:
            return "屏幕校对已过期（超过 300 秒），请重新校对"
        return self._presentation_failure()

    @property
    def status_text(self) -> str:
        return self._status_text

    def set_status_text(self, text: str) -> None:
        self._status_text = str(text)
        self.update()

    def snapshot_identity(self) -> dict[str, Any]:
        screen = self.screen()
        context = self.context()
        graphics = dict(self._graphics)
        graphics["context_valid"] = bool(context is not None and context.isValid())
        if context is not None:
            graphics["swap_interval"] = context.format().swapInterval()
        return {
            "window_token": str(id(self)),
            "context_generation": self._context_generation,
            "platform": QGuiApplication.platformName().lower(),
            "screen": {
                "name": screen.name(), "manufacturer": screen.manufacturer(),
                "model": screen.model(), "serial_number": screen.serialNumber(),
                "geometry": _rect_identity(screen.geometry()),
                "refresh_rate_hz": screen.refreshRate(),
                "device_pixel_ratio": screen.devicePixelRatio(),
            } if screen is not None else {},
            "selected_screen_matches": screen is self._selected_screen,
            "window_geometry": _rect_identity(self.geometry()),
            "device_pixel_ratio": self.devicePixelRatio(),
            "fullscreen": self.windowState() == Qt.WindowState.WindowFullScreen,
            "graphics": graphics,
        }

    def _presentation_failure(self) -> str:
        reason = identity_failure(self.snapshot_identity(), self.refresh_rate_hz)
        if reason:
            return reason
        if not self.isVisible() or not self.isExposed():
            return "刺激窗口不可见或未曝光"
        if QGuiApplication.applicationState() != Qt.ApplicationState.ApplicationActive:
            return "应用已失去前台活动状态"
        return ""

    def _reset_frames(self, mode: str) -> None:
        self._generation += 1
        self._mode = mode
        self._pending_frame = None
        self._next_frame_index = 0
        self._frames_observed = 0
        self._last_swap_s = None
        self._started_s = time.perf_counter()
        self._intervals = []
        self._dropped_frames = 0

    def start_calibration(self) -> bool:
        """Show a static full-screen surface and measure 180 real swaps."""
        if self._mode in ("calibration", "flicker"):
            return False
        self._passed_calibration = None
        self._last_calibration = None
        self._calibration_identity = None
        self._reset_frames("calibration")
        self._aborting = False
        self._exposed_once = False
        self._status_text = "尚未闪烁，正在校对屏幕时序；Esc 退出"
        self._calibration_watchdog.start(round(CALIBRATION_TIMEOUT_S * 1000))
        # A known non-native platform is rejected before any surface is shown.
        if QGuiApplication.platformName().lower() not in _NATIVE_PLATFORMS:
            self._fail("屏幕校对需要原生显示平台，不能使用 offscreen/minimal 渲染")
            return False
        self._setup = True
        try:
            self.setScreen(self._selected_screen)
            self.setGeometry(self._selected_screen.geometry())
            self.showFullScreen()
            self.raise_()
            self.requestActivate()
        finally:
            self._setup = False
        self.update()
        return True

    def start_flicker(self) -> bool:
        """Controller calls this only after consent/preflight/capture readiness."""
        reason = self._calibration_failure()
        if reason:
            self._fail(reason)
            return False
        if self._mode == "flicker":
            return True
        self._reset_frames("flicker")
        self._status_text = "三目标刺激进行中；Esc / Q 立即退出"
        self._swap_watchdog.start(round(MAX_GAP_S * 1000) + 1)
        self.update()
        return True

    def stop_flicker(self) -> None:
        """Stop scheduling swaps and return to static gray targets."""
        was_calibrating = self._mode == "calibration"
        self._generation += 1
        self._mode = "idle"
        self._pending_frame = None
        self._swap_watchdog.stop()
        self._calibration_watchdog.stop()
        self._status_text = "尚未闪烁，刺激已停止；Esc 退出"
        if was_calibrating:
            self._finish_calibration("校对已取消")
        self.update()

    def initializeGL(self) -> None:
        self._context_generation += 1
        context = self.context()
        self._gl = context.functions()
        self._gl.initializeOpenGLFunctions()
        def gl_text(name: int) -> str:
            value = self._gl.glGetString(name)
            return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value or "")
        self._graphics = {
            "backend": "OpenGL", "vendor": gl_text(0x1F00),
            "renderer": gl_text(0x1F01), "version": gl_text(0x1F02),
            "swap_interval": context.format().swapInterval(),
            "vsync_requested": True,
            "timestamp_source": "Qt frameSwapped callback / time.perf_counter",
        }
        context.aboutToBeDestroyed.connect(self._context_destroyed)
        self._gl.glDisable(0x0BD0)  # GL_DITHER
        self._gl.glDisable(0x0BE2)  # GL_BLEND
        self._identity_changed()

    def paintGL(self) -> None:
        if self._gl is None:
            return
        waiting_for_activation = (
            self._mode == "calibration" and self._calibration_identity is None
            and QGuiApplication.applicationState() != Qt.ApplicationState.ApplicationActive
        )
        if self._mode in ("calibration", "flicker") and not waiting_for_activation:
            reason = self._presentation_failure()
            if not reason and self._calibration_identity is not None:
                if self.snapshot_identity() != self._calibration_identity:
                    reason = "屏幕或 OpenGL 上下文已改变，请重新校对"
            if reason:
                self._fail(reason)
            elif self._calibration_identity is None:
                self._calibration_identity = self.snapshot_identity()
        lit = frame_plan(self._next_frame_index, self.refresh_rate_hz) if self._mode == "flicker" else None
        self._pending_frame = {
            "generation": self._generation, "mode": self._mode,
            "frame_index": self._next_frame_index, "target_lit": lit,
            "identity": self.snapshot_identity(),
        } if self._mode in ("calibration", "flicker") and not waiting_for_activation else None
        scale = self.devicePixelRatio()
        width, height = round(self.width() * scale), round(self.height() * scale)
        self._gl.glViewport(0, 0, width, height)
        self._gl.glDisable(0x0C11)  # GL_SCISSOR_TEST
        self._gl.glClearColor(0.08, 0.08, 0.08, 1.0)
        self._gl.glClear(0x00004000)  # GL_COLOR_BUFFER_BIT
        rectangles = target_rectangles(self.width(), self.height())
        self._gl.glEnable(0x0C11)
        for target, rect in zip(TARGETS, rectangles):
            left, right = round(rect.x() * scale), round((rect.x() + rect.width()) * scale)
            top, bottom = round(rect.y() * scale), round((rect.y() + rect.height()) * scale)
            self._gl.glScissor(left, height - bottom, right - left, bottom - top)
            level = (1.0 if lit[target.command] else 0.0) if lit is not None else 0.35
            self._gl.glClearColor(level, level, level, 1.0)
            self._gl.glClear(0x00004000)
        self._gl.glDisable(0x0C11)
        device = QOpenGLPaintDevice(QSize(width, height))
        device.setDevicePixelRatio(scale)
        painter = QPainter(device)
        try:
            painter.setPen(QColor("white"))
            painter.setFont(QFont("Microsoft YaHei", max(12, min(24, self.width() // 55))))
            for target, rect in zip(TARGETS, rectangles):
                # Labels stay outside the modulated rectangles.
                label = QRect(rect.x() - 20, rect.bottom() + 12, rect.width() + 40, 85)
                painter.drawText(label, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                                 f"{target.title}  {target.frequency_hz} Hz\n{target.command}")
            painter.drawText(QRect(20, 25, self.width() - 40, max(60, self.height() // 5)),
                             Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                             self._status_text)
        finally:
            painter.end()

    def _on_frame_swapped(self) -> None:
        now = time.perf_counter()
        pending = self._pending_frame
        self._pending_frame = None
        if not pending or pending["generation"] != self._generation or pending["mode"] != self._mode:
            return
        reason = self._presentation_failure()
        if not reason and pending["identity"] != self.snapshot_identity():
            reason = "帧交换期间屏幕身份改变，请重新校对"
        if reason:
            self._abort(reason)
            return
        interval = None if self._last_swap_s is None else now - self._last_swap_s
        dropped = max(0, math.floor(interval * self.refresh_rate_hz + 0.5) - 1) if interval is not None else 0
        self._dropped_frames += dropped
        self._frames_observed += 1
        record = {key: value for key, value in pending.items() if key != "generation"}
        record.update(swap_monotonic_s=now, interval_s=interval,
                      dropped_since_previous=dropped, dropped_frames=self._dropped_frames)
        self._last_swap_s = now
        # A gap already present at the first flicker swap is also unsafe.
        gap = interval if interval is not None else now - self._started_s
        failure = "帧交换间隔超过 0.10 秒，刺激已停止" if self._mode == "flicker" and gap > MAX_GAP_S else ""
        if interval is not None and (self._mode == "flicker" or self._frames_observed > WARMUP_FRAMES):
            self._intervals.append(interval)
        if self._mode == "calibration" and now - self._started_s >= CALIBRATION_TIMEOUT_S:
            failure = "屏幕校对超过 12 秒，未在期限内完成 180 帧"
        if self._mode == "flicker" and self._frames_observed % TIMING_CHECK_FRAMES == 0:
            failure = failure or timing_failure(timing_statistics(self._intervals, self.refresh_rate_hz))
            self._intervals.clear()
        if failure:
            # Notify/stop before emitting an audit record that controller code can consume.
            self._fail(failure)
            self.frame_record.emit(record)
            return
        generation = self._generation
        self.frame_record.emit(record)
        # A direct signal handler may stop/close the surface while logging a frame.
        if self._generation != generation:
            return
        if self._mode == "calibration" and self._frames_observed >= CALIBRATION_FRAMES:
            self._finish_calibration()
            return
        self._next_frame_index += 1 + dropped if self._mode == "flicker" else 1
        if self._mode == "flicker":
            self._swap_watchdog.start(round(MAX_GAP_S * 1000) + 1)
        self.update()

    def _finish_calibration(self, reason: str = "") -> None:
        stats = timing_statistics(self._intervals, self.refresh_rate_hz)
        reason = reason or self._presentation_failure()
        if not reason and self._calibration_identity != self.snapshot_identity():
            reason = "校对期间屏幕或 OpenGL 身份改变"
        if not reason and self._frames_observed < CALIBRATION_FRAMES:
            reason = "校对帧数不足 180 帧"
        reason = reason or timing_failure(stats)
        report = {
            **stats, "status": "failed" if reason else "passed", "passed": not bool(reason),
            "reason": reason, "identity": deepcopy(self._calibration_identity or self.snapshot_identity()),
            "frames_observed": self._frames_observed, "warmup_frames": WARMUP_FRAMES,
            "completed_monotonic_s": time.perf_counter(),
        }
        self._generation += 1
        self._pending_frame = None
        self._mode = "idle"
        self._calibration_watchdog.stop()
        self._passed_calibration = deepcopy(report) if report["passed"] else None
        self._last_calibration = deepcopy(report)
        self._status_text = ("尚未闪烁，校对完成，按 Enter / 空格开始，Esc 退出"
                             if report["passed"] else f"尚未闪烁，屏幕校对未通过：{reason}；Esc 退出")
        self.update()
        self.calibration_finished.emit(deepcopy(report))

    def _fail(self, reason: str) -> None:
        was_calibrating = self._mode == "calibration"
        self._passed_calibration = None
        if was_calibrating:
            self._finish_calibration(reason)
        else:
            self.stop_flicker()
        self._status_text = f"尚未闪烁，未通过：{reason}；Esc 退出"
        self.timing_failed.emit(reason)
        self.update()

    def _check_calibration_timeout(self) -> None:
        if self._mode == "calibration":
            elapsed = time.perf_counter() - self._started_s
            if elapsed >= CALIBRATION_TIMEOUT_S:
                self._fail("屏幕校对超过 12 秒，未在期限内完成 180 帧")
            else:
                self._calibration_watchdog.start(max(1, math.ceil((CALIBRATION_TIMEOUT_S - elapsed) * 1000)))

    def _check_swap_timeout(self) -> None:
        if self._mode == "flicker":
            reference = self._last_swap_s if self._last_swap_s is not None else self._started_s
            remaining = MAX_GAP_S - (time.perf_counter() - reference)
            if remaining < 0:
                self._fail("超过 0.10 秒未收到帧交换，刺激已停止")
            else:
                self._swap_watchdog.start(max(1, math.ceil(remaining * 1000) + 1))

    def _identity_changed(self, *_args) -> None:
        if self._setup:
            return
        baseline = self._passed_calibration or ({"identity": self._calibration_identity}
                                               if self._calibration_identity else None)
        if baseline and self.snapshot_identity() != baseline["identity"]:
            self._abort("全屏位置、屏幕几何、刷新率或 OpenGL 身份已改变")

    def _visibility_changed(self, visible: bool) -> None:
        if not visible and not self._setup:
            self._abort("刺激窗口已隐藏")

    def _application_state_changed(self, state) -> None:
        if state == Qt.ApplicationState.ApplicationActive:
            if self._mode == "calibration":
                self.update()
        elif not self._setup and (self.isVisible() or self._mode != "idle" or self._passed_calibration):
            self._abort("应用已失去前台活动状态")

    def _screen_removed(self, screen) -> None:
        if screen is self._selected_screen:
            self._abort("指定屏幕已断开")

    def _context_destroyed(self) -> None:
        self._abort("OpenGL 上下文已销毁")
        self._graphics = {}
        self._gl = None

    def _abort(self, reason: str = "用户退出") -> None:
        if self._aborting:
            return
        guarded = self._mode in ("calibration", "flicker") or self._passed_calibration is not None
        self._aborting = True
        self._passed_calibration = None
        if self._mode == "calibration":
            self._finish_calibration(reason)
        self.stop_flicker()
        # Even an uncalibrated visible window must clear controller I/O on exit.
        self.aborted.emit()
        if guarded:
            self._status_text = f"尚未闪烁，已中止：{reason}"
        self.close()

    def event(self, event) -> bool:
        # QWindow has no QWidget hideEvent/closeEvent; intercept its native events.
        if hasattr(self, "_mode"):
            kind = event.type()
            if kind == QEvent.Type.Close and not self._aborting:
                self._abort("刺激窗口已关闭")
            elif kind == QEvent.Type.Hide and not self._setup:
                self._abort("刺激窗口已隐藏")
            elif kind == QEvent.Type.Expose:
                if self.isExposed():
                    self._exposed_once = True
                elif self._exposed_once and not self._setup:
                    self._abort("刺激窗口已失去曝光")
            elif kind == QEvent.Type.DevicePixelRatioChange:
                self._identity_changed()
        return super().event(event)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
            event.accept()
            self._abort()
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            event.accept()
            if (not event.isAutoRepeat() and self._mode == "idle"
                    and self._passed_calibration and self._passed_calibration["passed"]):
                reason = self._calibration_failure()
                if reason:
                    self._fail(reason)
                else:
                    self.start_requested.emit()
        else:
            super().keyPressEvent(event)
