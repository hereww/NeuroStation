"""Isolated BrainFlow SSVEP acquisition worker with optional Qt stimulus output."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable

from eeg_tools.config import ConfigError, validate_channel_config
from eeg_tools.session_files import (
    format_raw_number,
    iso_now,
    sha256,
    write_brainflow_tsv,
    write_column_definitions,
    write_events,
    write_json,
    write_manifest,
)
from neurostation_contract import (
    PRODUCT_DESCRIPTION,
    PRODUCT_NAME,
    PRODUCT_SEMVER,
    PRODUCT_VERSION,
    RELEASE_DATE,
    dataset_name_for_eye,
    normalize_dataset_name_base,
)

from .ssvep import SSVEPProtocol, SSVEPProtocolError
from .device_discovery import candidate_serial_ports
from neurostation_display import (
    center_divider_geometry,
    eye_half_geometry,
    resolve_eye_regions,
    ssvep_frame_is_lit,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "configs" / "protocols" / "ssvep_four_target_v2.json"
DEFAULT_CHANNELS = ROOT / "configs" / "channel_config_v1_auto.json"


class UserAbort(Exception):
    pass


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.pending")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Windows may briefly deny replacement while the desktop process is
    # reading the previous status file. Retry the atomic swap instead of
    # failing an otherwise healthy acquisition.
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.005)


class ProgressReporter:
    def __init__(self, path: Path | None) -> None:
        self.path = path.resolve() if path else None
        self._last_emit_at = 0.0
        self._last_phase = ""

    def emit(self, **value: Any) -> None:
        now = time.monotonic()
        phase = str(value.get("phase", ""))
        terminal = phase in {"completed", "aborted", "error"}
        if not terminal and phase == self._last_phase and now - self._last_emit_at < 0.1:
            return
        payload = {"schema_version": 1, "updated_at": iso_now(), **value}
        if self.path:
            _atomic_json(self.path, payload)
        # stdout is consumed by desktop runners and CI on Windows, where the
        # active console encoding may not represent diagnostic text. JSON is
        # still Unicode-safe because the payload file is written as UTF-8.
        print(json.dumps(payload, ensure_ascii=True), flush=True)
        self._last_emit_at = now
        self._last_phase = phase


def _read_channel_config(path: Path) -> tuple[dict[str, Any], list[str]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigError(f"Channel configuration does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ConfigError(f"Invalid channel configuration: {error}") from error
    if not isinstance(value, dict):
        raise ConfigError("Channel configuration root must be an object")
    return value, validate_channel_config(value, path)


def _session_directory(root: Path) -> tuple[str, Path]:
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    base = datetime.now().strftime("session_%Y%m%d_%H%M%S_%f")[:-3]
    for suffix in range(1000):
        session_id = base if suffix == 0 else f"{base}_{suffix:03d}"
        candidate = root / session_id
        try:
            candidate.mkdir()
            return session_id, candidate
        except FileExistsError:
            continue
    raise RuntimeError("Unable to allocate a unique session directory")


def _screen_metadata(screen: Any, screen_index: int) -> dict[str, Any]:
    geometry = screen.geometry()
    refresh_rate = float(screen.refreshRate()) if screen.refreshRate() else 0.0
    return {
        "screen_index": int(screen_index),
        "name": str(screen.name()),
        "geometry": {
            "x": int(geometry.x()),
            "y": int(geometry.y()),
            "width": int(geometry.width()),
            "height": int(geometry.height()),
        },
        "device_pixel_ratio": float(screen.devicePixelRatio()),
        "refresh_rate_hz": refresh_rate,
    }


def _screen_geometry_text(metadata: dict[str, Any]) -> str:
    geometry = metadata.get("geometry")
    if not isinstance(geometry, dict):
        return ""
    return json.dumps(geometry, ensure_ascii=False, separators=(",", ":"))


def _screen_mapping_text(display: dict[str, Any]) -> str:
    mapping = {side: display.get(side) for side in ("left", "right")}
    if not any(value is not None for value in mapping.values()):
        return ""
    return json.dumps(mapping, ensure_ascii=False, separators=(",", ":"), default=str)


def _create_stimulus_window(eye_side: str):
    from PySide6.QtCore import QRect, Qt
    from PySide6.QtGui import QColor, QFont, QKeyEvent, QPainter, QSurfaceFormat
    from PySide6.QtOpenGL import QOpenGLWindow
    from PySide6.QtWidgets import QApplication

    application = QApplication.instance() or QApplication(sys.argv[:1])

    mapping = resolve_eye_regions(application)
    active = mapping[eye_side]
    abort_state = {"requested": False}
    surface_format = QSurfaceFormat()
    surface_format.setSwapInterval(1)

    class StimulusWindow(QOpenGLWindow):
        def __init__(self) -> None:
            super().__init__(QOpenGLWindow.UpdateBehavior.NoPartialUpdate)
            self.setFormat(surface_format)
            self._gaze_marks: list[tuple[int, float]] = []
            self._gl = None
            self._frame_swapped_count = 0
            self._last_frame_swapped_s: float | None = None
            self._state_generation = 0
            self._painted_generation = 0
            self._swapped_generation = 0
            self.rendering_metadata: dict[str, Any] = {}
            self.target_index: int | None = None
            self.lit = False
            self.message = ""
            self.message_flash = False
            self.frameSwapped.connect(self._on_frame_swapped)
            self.setCursor(Qt.CursorShape.BlankCursor)

        @property
        def abort_requested(self) -> bool:
            return bool(abort_state["requested"])

        @property
        def frame_swapped_count(self) -> int:
            return self._frame_swapped_count

        @property
        def last_frame_swapped_s(self) -> float | None:
            return self._last_frame_swapped_s

        @property
        def swapped_generation(self) -> int:
            return self._swapped_generation

        def initializeGL(self) -> None:
            self._gl = self.context().functions()
            get_string = self._gl.glGetString
            self.rendering_metadata = {
                "backend": "OpenGL",
                "vendor": str(get_string(0x1F00) or ""),
                "renderer": str(get_string(0x1F01) or ""),
                "version": str(get_string(0x1F02) or ""),
                "swap_interval": int(self.context().format().swapInterval()),
                "vsync_requested": True,
                "frame_timestamp_source": "Qt frameSwapped signal callback",
            }
            self._gl.glDisable(0x0BD0)  # GL_DITHER
            self._gl.glDisable(0x0BE2)  # GL_BLEND

        def _on_frame_swapped(self) -> None:
            self._last_frame_swapped_s = time.perf_counter()
            self._swapped_generation = self._painted_generation
            self._frame_swapped_count += 1

        def keyPressEvent(self, event: QKeyEvent) -> None:
            if event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
                abort_state["requested"] = True
            else:
                target_keys = {
                    Qt.Key.Key_1: 0,
                    Qt.Key.Key_2: 1,
                    Qt.Key.Key_3: 2,
                    Qt.Key.Key_4: 3,
                }
                target_index = target_keys.get(event.key())
                if target_index is not None:
                    # This is an explicit manual/self-report annotation. It
                    # is never treated as proof that the eyes were on target.
                    self._gaze_marks.append((target_index, time.perf_counter()))
                else:
                    super().keyPressEvent(event)

        def paintGL(self) -> None:
            if self._gl is None:
                return
            self._painted_generation = self._state_generation
            scale = self.devicePixelRatio()
            width = max(1, round(self.width() * scale))
            height = max(1, round(self.height() * scale))
            self._gl.glViewport(0, 0, width, height)
            self._gl.glDisable(0x0C11)  # GL_SCISSOR_TEST
            self._gl.glClearColor(0.0, 0.0, 0.0, 1.0)
            self._gl.glClear(0x00004000)  # GL_COLOR_BUFFER_BIT

            half = eye_half_geometry(0, 0, width, height, eye_side)
            if self.lit or (self.message and self.message_flash):
                self._gl.glEnable(0x0C11)  # GL_SCISSOR_TEST
                self._gl.glScissor(half["x"], 0, half["width"], height)
                self._gl.glClearColor(1.0, 1.0, 1.0, 1.0)
                self._gl.glClear(0x00004000)

            divider = center_divider_geometry(0, 0, width, height)
            self._gl.glEnable(0x0C11)  # GL_SCISSOR_TEST
            self._gl.glScissor(divider["x"], 0, divider["width"], height)
            self._gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            self._gl.glClear(0x00004000)
            self._gl.glDisable(0x0C11)  # GL_SCISSOR_TEST

            if self.message:
                active_rect = eye_half_geometry(0, 0, self.width(), self.height(), eye_side)
                painter = QPainter(self)
                painter.setPen(QColor("black") if self.message_flash else QColor("white"))
                painter.setFont(QFont("Sans Serif", max(28, min(active_rect["width"], active_rect["height"]) // 12)))
                painter.drawText(
                    QRect(
                        active_rect["x"],
                        active_rect["y"],
                        active_rect["width"],
                        active_rect["height"],
                    ),
                    Qt.AlignmentFlag.AlignCenter,
                    self.message,
                )
                painter.end()

        def show_message(self, message: str, *, flash: bool = False) -> None:
            self._state_generation += 1
            self.message = message
            self.target_index = None
            self.lit = False
            self.message_flash = bool(flash and int(time.perf_counter() * 4) % 2)
            self.update()

        def show_target(self, target_index: int, lit: bool) -> int:
            self._state_generation += 1
            self.message = ""
            self.target_index = target_index
            self.lit = lit
            self.update()
            return self._state_generation

        def consume_gaze_marks(self) -> list[tuple[int, float]]:
            marks = list(self._gaze_marks)
            self._gaze_marks.clear()
            return marks

    window = StimulusWindow()
    window.setScreen(active["screen"])
    window.setGeometry(active["screen"].geometry())
    window.showFullScreen()
    window.raise_()
    window.requestActivate()
    application.processEvents()
    rendering = window.rendering_metadata
    if not window.context() or not window.context().isValid() or not rendering.get("renderer"):
        window.close()
        raise RuntimeError("visual stimulus requires a working OpenGL context")
    renderer = str(rendering.get("renderer", "")).lower()
    software_renderers = (
        "llvmpipe", "softpipe", "swiftshader", "software rasterizer",
        "microsoft basic render driver", "gdi generic",
    )
    if any(name in renderer for name in software_renderers):
        window.close()
        raise RuntimeError(f"visual stimulus requires GPU rendering; got {rendering['renderer']}")
    if rendering.get("swap_interval") != 1:
        window.close()
        raise RuntimeError("visual stimulus requires vertical synchronization (swap interval 1)")
    return application, window, mapping


def _wait_for_frame_swap(
    application: Any,
    window: Any,
    previous_swap_count: int,
    *,
    cancel_file: Path | None = None,
    timeout_s: float = 1.0,
    minimum_generation: int | None = None,
) -> float:
    from PySide6.QtCore import QEventLoop, QTimer

    def target_swapped() -> bool:
        return (
            window.frame_swapped_count > previous_swap_count
            and (
                minimum_generation is None
                or window.swapped_generation >= minimum_generation
            )
        )

    if target_swapped():
        return float(window.last_frame_swapped_s)

    event_loop = QEventLoop()
    timeout = QTimer()
    timeout.setSingleShot(True)
    timeout.timeout.connect(event_loop.quit)
    abort_monitor = QTimer()
    abort_monitor.setInterval(10)
    abort_monitor.timeout.connect(
        lambda: event_loop.quit()
        if (cancel_file is not None and cancel_file.exists())
        or bool(window.abort_requested)
        else None
    )

    def on_frame_swapped() -> None:
        if target_swapped():
            event_loop.quit()

    window.frameSwapped.connect(on_frame_swapped)
    timeout.start(max(1, round(timeout_s * 1000)))
    abort_monitor.start()
    try:
        event_loop.exec()
    finally:
        timeout.stop()
        abort_monitor.stop()
        window.frameSwapped.disconnect(on_frame_swapped)

    _check_abort(cancel_file, window)
    if not target_swapped():
        raise RuntimeError("timed out waiting for a VSync frame swap")
    return float(window.last_frame_swapped_s)


def _wait_until_frame_deadline(
    deadline: float,
    *,
    cancel_file: Path | None,
    window: Any,
    render_lead_s: float = 0.001,
) -> None:
    from PySide6.QtCore import QEventLoop, QTimer, Qt

    remaining_ms = (deadline - render_lead_s - time.perf_counter()) * 1000
    if remaining_ms <= 0:
        _check_abort(cancel_file, window)
        return

    event_loop = QEventLoop()
    timer = QTimer()
    timer.setSingleShot(True)
    timer.setTimerType(Qt.TimerType.PreciseTimer)
    timer.timeout.connect(event_loop.quit)
    abort_monitor = QTimer()
    abort_monitor.setInterval(10)
    abort_monitor.timeout.connect(
        lambda: event_loop.quit()
        if (cancel_file is not None and cancel_file.exists())
        or bool(window.abort_requested)
        else None
    )
    timer.start(max(1, round(remaining_ms)))
    abort_monitor.start()
    try:
        event_loop.exec()
    finally:
        timer.stop()
        abort_monitor.stop()
    _check_abort(cancel_file, window)


def _prepare_cyton_board(
    board_id: int,
    params: Any,
    requested_port: str,
    *,
    retries: int = 2,
    retry_delay_s: float = 0.5,
) -> tuple[Any, str, list[dict[str, str]]]:
    """Prepare the first Cyton endpoint accepted by BrainFlow.

    BrainFlow's native error ``BOARD_NOT_READY_ERROR:7`` is commonly caused by
    a wrong COM port or another process holding the dongle. Trying every
    discovered endpoint turns that opaque failure into a reliable automatic
    scan while preserving explicit-port behavior.
    """

    from brainflow.board_shim import BoardShim

    requested_value = (requested_port or "AUTO").strip()
    if (
        os.name == "nt"
        and requested_value
        and requested_value.upper() != "AUTO"
        and not re.fullmatch(r"(?:\\\\\.\\)?COM\d+", requested_value, re.IGNORECASE)
    ):
        raise RuntimeError(
            f"Invalid Windows serial port '{requested_value}'. "
            "Use a COM port such as COM5."
        )

    candidates = candidate_serial_ports(requested_port)
    if not candidates:
        raise RuntimeError(
            "No serial ports were detected. Connect the OpenBCI USB dongle, "
            "close OpenBCI GUI/other serial monitors, then retry."
        )
    if retries < 0:
        raise ValueError("retries must be non-negative")
    if retry_delay_s < 0:
        raise ValueError("retry_delay_s must be non-negative")
    failures: list[dict[str, str]] = []
    for candidate in candidates:
        for attempt in range(retries + 1):
            params.serial_port = candidate.device
            board = None
            try:
                board = BoardShim(board_id, params)
                board.prepare_session()
                return board, candidate.device, failures
            except Exception as error:
                failures.append(
                    {
                        "port": candidate.device,
                        "attempt": str(attempt + 1),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                if board is not None:
                    try:
                        board.release_session()
                    except Exception:
                        pass
                message = str(error).lower()
                retryable = any(
                    marker in message
                    for marker in (
                        "board_not_ready_error:7",
                        "welcome characters",
                        "unable to prepare streaming session",
                    )
                )
                if not retryable or attempt >= retries:
                    break
                if retry_delay_s:
                    time.sleep(retry_delay_s)
    attempted = ", ".join(item["port"] for item in failures)
    detail = "; ".join(f"{item['port']}: {item['error']}" for item in failures)
    # BrainFlow logs the literal "welcome characters" message, but its Python
    # exception often only exposes the stable BOARD_NOT_READY_ERROR:7 wrapper.
    # Treat both forms as the same radio-handshake failure so the user gets a
    # useful hardware diagnosis in the saved session metadata as well as in the
    # console/UI error banner.
    no_welcome = bool(failures) and all(
        (
            "welcome characters" in item["error"].lower()
            or "board_not_ready_error:7" in item["error"].lower()
            or "unable to prepare streaming session" in item["error"].lower()
        )
        for item in failures
    )
    if no_welcome:
        diagnosis = (
            "串口可以打开，但没有收到 Cyton 欢迎字符；请确认 Cyton 主板已上电、"
            "无线 USB dongle 与主板已配对且距离合适，板卡开关处于 PC/运行位置，"
            "然后重新上电。若 COM5 对应的是其他 USB-UART 设备，请在界面中选择正确的端口。"
        )
    else:
        diagnosis = (
            "请确认 USB dongle 已连接、驱动正常，并关闭 OpenBCI GUI 或其他占用串口的程序。"
        )
    raise RuntimeError(
        "BrainFlow 无法准备 OpenBCI Cyton 串流（BOARD_NOT_READY_ERROR:7）。"
        f" 已扫描：{attempted or '无'}。{detail} {diagnosis}"
    )


def _check_abort(cancel_file: Path | None, window: Any | None) -> None:
    if cancel_file and cancel_file.exists():
        raise UserAbort
    if window is not None and window.abort_requested:
        raise UserAbort


def _wait_phase(
    seconds: float,
    *,
    cancel_file: Path | None,
    application: Any | None,
    window: Any | None,
    update: Callable[[float], None] | None = None,
) -> None:
    started = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - started
        if update:
            update(elapsed)
        if elapsed >= seconds:
            return
        if application is not None:
            application.processEvents()
        _check_abort(cancel_file, window)
        time.sleep(min(0.01, max(0.0, seconds - elapsed)))


def _present_stimulus(
    *,
    application: Any,
    window: Any,
    target_index: int,
    frequency_hz: int,
    duration_s: float,
    refresh_rate_hz: int,
    cancel_file: Path | None,
    trial_index: int,
    eye_side: str,
    screen_index: int,
    dataset_name_base: str,
    dataset_name: str,
    screen_name: str,
    screen_geometry: str,
    screen_mapping: str,
    stimulus_region: str,
    frame_rows: list[dict[str, Any]],
    progress: Callable[[float], None],
    on_gaze_mark: Callable[[int, float], None] | None = None,
    on_stimulus_onset: Callable[[float], None] | None = None,
    on_stimulus_offset: Callable[[float], None] | None = None,
) -> None:
    frame_total = max(1, round(duration_s * refresh_rate_hz))
    stimulus_started: float | None = None
    previous_actual: float | None = None
    frame_index = 0
    rendering = getattr(window, "rendering_metadata", {})

    def drain_gaze_marks() -> None:
        if on_gaze_mark is None or not hasattr(window, "consume_gaze_marks"):
            return
        for marked_target_index, marked_at in window.consume_gaze_marks():
            on_gaze_mark(marked_target_index, marked_at)

    while frame_index < frame_total:
        drain_gaze_marks()
        previous_swap_count = window.frame_swapped_count
        lit = ssvep_frame_is_lit(frame_index, frequency_hz, refresh_rate_hz)
        target_generation = window.show_target(target_index, lit)
        actual = _wait_for_frame_swap(
            application,
            window,
            previous_swap_count,
            cancel_file=cancel_file,
            minimum_generation=target_generation,
        )
        if stimulus_started is None:
            stimulus_started = actual
            if on_stimulus_onset is not None:
                on_stimulus_onset(actual)
        actual_s = actual - stimulus_started
        interval = None if previous_actual is None else actual - previous_actual
        frame_step = 1 if interval is None else max(1, round(interval * refresh_rate_hz))
        dropped = max(0, frame_step - 1)
        frame_rows.append(
            {
                "trial_index": trial_index,
                "eye_side": eye_side,
                "dataset_name_base": dataset_name_base,
                "dataset_name": dataset_name,
                "screen_index": screen_index,
                "screen_name": screen_name,
                "screen_geometry": screen_geometry,
                "screen_mapping": screen_mapping,
                "stimulus_region": stimulus_region,
                "graphics_vendor": rendering.get("vendor", ""),
                "graphics_renderer": rendering.get("renderer", ""),
                "graphics_version": rendering.get("version", ""),
                "swap_interval": rendering.get("swap_interval", ""),
                "frame_index": frame_index,
                "scheduled_s": frame_index / refresh_rate_hz,
                "actual_s": actual_s,
                "frame_interval_ms": "" if interval is None else interval * 1000,
                "lateness_ms": (actual_s - frame_index / refresh_rate_hz) * 1000,
                "dropped_since_previous": dropped,
                "lit": int(lit),
            }
        )
        previous_actual = actual
        frame_index += frame_step
        if frame_index % max(1, refresh_rate_hz // 10) == 0:
            progress(min(duration_s, actual_s))
    final_deadline = (stimulus_started or time.perf_counter()) + duration_s
    blank_request_deadline = final_deadline - (1.0 / refresh_rate_hz)
    _wait_phase(
        max(0.0, blank_request_deadline - time.perf_counter()),
        cancel_file=cancel_file,
        application=application,
        window=window,
    )
    previous_swap_count = window.frame_swapped_count
    blank_generation = window.show_target(target_index, False)
    offset_actual = _wait_for_frame_swap(
        application,
        window,
        previous_swap_count,
        cancel_file=cancel_file,
        minimum_generation=blank_generation,
    )
    offset_s = offset_actual - (stimulus_started or offset_actual)
    interval = None if previous_actual is None else offset_actual - previous_actual
    frame_step = 1 if interval is None else max(1, round(interval * refresh_rate_hz))
    frame_rows.append(
        {
            "trial_index": trial_index,
            "eye_side": eye_side,
            "dataset_name_base": dataset_name_base,
            "dataset_name": dataset_name,
            "screen_index": screen_index,
            "screen_name": screen_name,
            "screen_geometry": screen_geometry,
            "screen_mapping": screen_mapping,
            "stimulus_region": stimulus_region,
            "graphics_vendor": rendering.get("vendor", ""),
            "graphics_renderer": rendering.get("renderer", ""),
            "graphics_version": rendering.get("version", ""),
            "swap_interval": rendering.get("swap_interval", ""),
            "frame_index": frame_total,
            "scheduled_s": duration_s,
            "actual_s": offset_s,
            "frame_interval_ms": "" if interval is None else interval * 1000,
            "lateness_ms": (offset_s - duration_s) * 1000,
            "dropped_since_previous": max(0, frame_step - 1),
            "lit": 0,
        }
    )
    drain_gaze_marks()
    if on_stimulus_offset is not None:
        on_stimulus_offset(offset_actual)


def _write_frame_log(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = (
        "trial_index",
        "eye_side",
        "dataset_name_base",
        "dataset_name",
        "screen_index",
        "screen_name",
        "screen_geometry",
        "screen_mapping",
        "stimulus_region",
        "graphics_vendor",
        "graphics_renderer",
        "graphics_version",
        "swap_interval",
        "frame_index",
        "scheduled_s",
        "actual_s",
        "frame_interval_ms",
        "lateness_ms",
        "dropped_since_previous",
        "lit",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("\t".join(columns) + "\n")
        for row in rows:
            stream.write("\t".join(str(row.get(column, "")) for column in columns) + "\n")


def _frame_timing_summary(
    frame_rows: list[dict[str, Any]], *, expected_refresh_rate_hz: float | None = None
) -> dict[str, Any]:
    intervals = [
        float(row["frame_interval_ms"])
        for row in frame_rows
        if row.get("frame_interval_ms") not in (None, "")
    ]
    sorted_intervals = sorted(intervals)
    median = None
    p95 = None
    if sorted_intervals:
        middle = len(sorted_intervals) // 2
        median = (
            sorted_intervals[middle]
            if len(sorted_intervals) % 2
            else (sorted_intervals[middle - 1] + sorted_intervals[middle]) / 2
        )
        p95 = sorted_intervals[max(0, math.ceil(len(sorted_intervals) * 0.95) - 1)]
    summary = {
        "frame_count": len(frame_rows),
        "dropped_frame_count": sum(int(row.get("dropped_since_previous", 0)) for row in frame_rows),
        "min_frame_interval_ms": min(sorted_intervals) if sorted_intervals else None,
        "median_frame_interval_ms": median,
        "p95_frame_interval_ms": p95,
        "max_frame_interval_ms": max(sorted_intervals) if sorted_intervals else None,
    }
    if expected_refresh_rate_hz is not None:
        if expected_refresh_rate_hz <= 0:
            raise ValueError("expected refresh rate must be positive")
        expected_interval_ms = 1000.0 / expected_refresh_rate_hz
        effective_refresh_rate_hz = 1000.0 / median if median and median > 0 else None
        cadence_ok = (
            median is not None
            and abs(median - expected_interval_ms) <= expected_interval_ms * 0.1
            and p95 is not None
            and p95 <= expected_interval_ms * 1.5
            and summary["dropped_frame_count"] == 0
        )
        summary.update(
            {
                "expected_refresh_rate_hz": expected_refresh_rate_hz,
                "expected_frame_interval_ms": expected_interval_ms,
                "effective_refresh_rate_hz": effective_refresh_rate_hz,
                "cadence_status": "pass" if cadence_ok else "review",
            }
        )
    return summary


def _build_quality_report(
    *,
    raw_data: Any,
    board_id: int | None,
    frame_rows: list[dict[str, Any]],
    recording_status: str,
    error: str | None,
) -> dict[str, Any]:
    """Summarize acquisition integrity without altering the raw stream."""

    import numpy as np
    from brainflow.board_shim import BoardShim

    if board_id is None:
        return {
            "schema_version": 1,
            "status": recording_status,
            "recording_status": recording_status,
            "error": error,
            "sampling_rate_hz": 0,
            "channel_count": 0,
            "samples_per_channel": 0,
            "duration_s": 0.0,
            "effective_rate_hz": 0.0,
            "timestamp_diff_median_s": None,
            "timestamp_gap_count": 0,
            "channels": {},
            "frame_count": len(frame_rows),
            "dropped_frame_count": sum(int(row.get("dropped_since_previous", 0)) for row in frame_rows),
        }
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    timestamp_channel = BoardShim.get_timestamp_channel(board_id)
    sampling_rate = int(BoardShim.get_sampling_rate(board_id))
    sample_count = int(raw_data.shape[1]) if getattr(raw_data, "ndim", 0) == 2 else 0
    timestamp_values = (
        raw_data[timestamp_channel, :]
        if sample_count and raw_data.shape[0] > timestamp_channel
        else np.empty(0)
    )
    finite_timestamps = timestamp_values[np.isfinite(timestamp_values)]
    timestamp_diffs = np.diff(finite_timestamps)
    positive_diffs = timestamp_diffs[timestamp_diffs > 0]
    duration = (
        float(finite_timestamps[-1] - finite_timestamps[0])
        if len(finite_timestamps) > 1
        else 0.0
    )
    per_channel: dict[str, dict[str, float | None]] = {}
    for channel_number, channel_index in enumerate(eeg_channels, start=1):
        if sample_count and channel_index < raw_data.shape[0]:
            values = raw_data[channel_index, :]
            finite = values[np.isfinite(values)]
            differences = np.diff(finite)
            per_channel[f"CH{channel_number}"] = {
                "finite_fraction": float(np.mean(np.isfinite(values))),
                "flat_fraction": float(np.mean(np.abs(differences) < 1e-9))
                if len(differences)
                else 1.0,
                "rms": float(np.sqrt(np.mean(np.square(finite)))) if len(finite) else None,
            }
        else:
            per_channel[f"CH{channel_number}"] = {
                "finite_fraction": 0.0,
                "flat_fraction": 1.0,
                "rms": None,
            }
    dropped_frames = sum(int(row.get("dropped_since_previous", 0)) for row in frame_rows)
    return {
        "schema_version": 1,
        "status": "ok" if recording_status == "completed" and sample_count else recording_status,
        "recording_status": recording_status,
        "error": error,
        "sampling_rate_hz": sampling_rate,
        "channel_count": len(eeg_channels),
        "samples_per_channel": sample_count,
        "duration_s": duration,
        "effective_rate_hz": float((len(finite_timestamps) - 1) / duration)
        if duration > 0
        else 0.0,
        "timestamp_diff_median_s": float(np.median(positive_diffs))
        if len(positive_diffs)
        else None,
        "timestamp_gap_count": int(np.sum(timestamp_diffs > (1.5 / sampling_rate))),
        "channels": per_channel,
        "frame_count": len(frame_rows),
        "dropped_frame_count": dropped_frames,
    }


def _annotate_event_samples(
    events: list[dict[str, Any]], raw_data: Any, board_id: int | None
) -> None:
    """Attach the first matching marker sample and device time to each event."""

    if board_id is None or getattr(raw_data, "ndim", 0) != 2 or raw_data.size == 0:
        return
    import numpy as np
    from brainflow.board_shim import BoardShim

    try:
        marker_channel = int(BoardShim.get_marker_channel(board_id))
        timestamp_channel = int(BoardShim.get_timestamp_channel(board_id))
    except Exception:
        return
    if marker_channel >= raw_data.shape[0]:
        return
    marker_values = raw_data[marker_channel, :]
    timestamps = (
        raw_data[timestamp_channel, :]
        if timestamp_channel < raw_data.shape[0]
        else np.zeros(raw_data.shape[1])
    )
    used_by_marker: dict[int, int] = {}
    session_start_time: float | None = None
    for event in events:
        try:
            marker = int(event.get("marker_code"))
        except (TypeError, ValueError):
            continue
        matches = np.flatnonzero(np.isclose(marker_values, marker, atol=1e-6))
        start = used_by_marker.get(marker, 0)
        if start >= len(matches):
            continue
        sample_index = int(matches[start])
        used_by_marker[marker] = start + 1
        sample_time = float(timestamps[sample_index])
        event["sample_index"] = sample_index
        event["sample_time_s"] = round(sample_time, 9)
        if event.get("event_name") == "session_start":
            session_start_time = sample_time
        if session_start_time is not None:
            event["recording_time_s"] = round(sample_time - session_start_time, 9)


class _LiveWaveformWriter:
    """Write a bounded-latency EEG preview without draining BrainFlow's buffer."""

    def __init__(self, path: Path, board: Any, board_id: int) -> None:
        from brainflow.board_shim import BoardShim

        self.path = path
        self.board = board
        self.eeg_rows = tuple(int(value) for value in BoardShim.get_eeg_channels(board_id))
        self.timestamp_row = int(BoardShim.get_timestamp_channel(board_id))
        try:
            self.marker_row = int(BoardShim.get_marker_channel(board_id))
        except Exception:
            self.marker_row = -1
        self.sample_index = 0
        self.last_timestamp: float | None = None
        self.handle = path.open("w", encoding="utf-8", newline="")
        self.handle.write(
            "\t".join(
                ["sample_index", "timestamp_s", "marker"]
                + [f"eeg_ch{index}" for index in range(1, len(self.eeg_rows) + 1)]
            )
            + "\n"
        )
        self.handle.flush()

    def pump(self, maximum_samples: int = 512) -> int:
        import numpy as np

        getter = getattr(self.board, "get_current_board_data", None)
        if getter is None:
            return 0
        try:
            data = getter(maximum_samples)
        except Exception:
            return 0
        if getattr(data, "ndim", 0) != 2 or data.shape[1] == 0:
            return 0
        if self.timestamp_row < 0 or self.timestamp_row >= data.shape[0]:
            return 0
        timestamps = np.asarray(data[self.timestamp_row, :], dtype=float)
        valid = np.isfinite(timestamps)
        if self.last_timestamp is not None:
            valid &= timestamps > self.last_timestamp + 1e-9
        indexes = np.flatnonzero(valid)
        if indexes.size == 0:
            return 0
        for column in indexes:
            timestamp = float(timestamps[column])
            marker = (
                float(data[self.marker_row, column])
                if 0 <= self.marker_row < data.shape[0]
                else 0.0
            )
            eeg = [
                float(data[row, column]) if 0 <= row < data.shape[0] else float("nan")
                for row in self.eeg_rows
            ]
            values = [
                self.sample_index,
                format_raw_number(timestamp),
                format_raw_number(marker),
            ]
            values.extend(format_raw_number(value) for value in eeg)
            self.handle.write("\t".join(str(value) for value in values) + "\n")
            self.sample_index += 1
            self.last_timestamp = timestamp
        self.handle.flush()
        return int(indexes.size)

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--channel-config", type=Path, default=DEFAULT_CHANNELS)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--participant", required=True)
    parser.add_argument("--session-name", required=True)
    parser.add_argument("--user-id", default="")
    parser.add_argument("--user-name", default="")
    parser.add_argument(
        "--board",
        choices=("cyton",),
        default="cyton",
        help="Physical acquisition board. Only OpenBCI Cyton is supported.",
    )
    parser.add_argument("--port", default="AUTO")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--stimulus-seconds", type=float)
    parser.add_argument("--rest-seconds", type=float)
    parser.add_argument("--countdown-seconds", type=float)
    parser.add_argument("--eye-side", choices=("left", "right"), required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--acknowledge-flicker-risk", action="store_true")
    parser.add_argument("--allow-draft-protocol", action="store_true")
    parser.add_argument("--allow-draft-channel-config", action="store_true")
    parser.add_argument("--status-file", type=Path)
    parser.add_argument("--cancel-file", type=Path)
    parser.add_argument("--preflight-file", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.countdown_seconds is not None and arguments.countdown_seconds < 0:
        parser.error("--countdown-seconds must be non-negative")
    if not arguments.headless and not arguments.acknowledge_flicker_risk:
        parser.error("visual stimulus requires --acknowledge-flicker-risk")
    try:
        dataset_name_base = normalize_dataset_name_base(arguments.session_name)
        dataset_name = dataset_name_for_eye(dataset_name_base, arguments.eye_side)
    except ValueError as error:
        parser.error(str(error))
    try:
        protocol = SSVEPProtocol.load(arguments.protocol).with_runtime_parameters(
            repetitions=arguments.repetitions,
            stimulus_s=arguments.stimulus_seconds,
            rest_s=arguments.rest_seconds,
        )
    except SSVEPProtocolError as error:
        parser.error(str(error))
    protocol_source = json.loads(arguments.protocol.read_text(encoding="utf-8"))
    preflight_report: dict[str, Any] | None = None
    if arguments.preflight_file is not None and arguments.preflight_file.is_file():
        try:
            loaded_preflight = json.loads(arguments.preflight_file.read_text(encoding="utf-8"))
            if isinstance(loaded_preflight, dict):
                preflight_report = loaded_preflight
        except (OSError, json.JSONDecodeError):
            preflight_report = None
    protocol_status = str(protocol_source.get("status", ""))
    protocol_is_draft = protocol_status.lower().startswith("draft")
    if (
        arguments.board == "cyton"
        and str(protocol_source.get("status", "")).lower().startswith("draft")
        and not arguments.allow_draft_protocol
    ):
        parser.error("Cyton acquisition refuses a draft protocol; approve it or use --allow-draft-protocol for technical validation")

    channel_config: dict[str, Any] | None = None
    channel_warnings: list[str] = []
    channel_is_draft = False
    if arguments.board == "cyton":
        try:
            channel_config, channel_warnings = _read_channel_config(arguments.channel_config)
        except ConfigError as error:
            parser.error(str(error))
        channel_is_draft = bool(channel_warnings)
        if channel_warnings and not arguments.allow_draft_channel_config:
            parser.error(
                "Cyton acquisition refuses the draft/incomplete channel map: "
                + "; ".join(channel_warnings)
            )

    try:
        import numpy as np
        from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
    except ModuleNotFoundError as error:
        parser.error(
            f"Missing dependency '{error.name}'. Install BrainFlow and NumPy from requirements.txt"
        )

    countdown_s = (
        float(protocol.countdown_s)
        if arguments.countdown_seconds is None
        else arguments.countdown_seconds
    )
    status_file = arguments.status_file.resolve() if arguments.status_file else None
    cancel_file = arguments.cancel_file.resolve() if arguments.cancel_file else None
    reporter = ProgressReporter(status_file)
    session_id, session_dir = _session_directory(arguments.output_root)
    started_at = iso_now()
    session_started_monotonic = time.perf_counter()
    events: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    board = None
    selected_port = arguments.port if arguments.board == "cyton" else None
    selected_screen_index: int | None = None
    display_metadata: dict[str, Any] = {}
    prepared = False
    streaming = False
    application = None
    window = None
    recording_started: float | None = None
    acquisition_started: float | None = None
    live_waveform_path = (
        status_file.parent / "live_waveform.tsv" if status_file is not None else None
    )
    live_waveform_writer: _LiveWaveformWriter | None = None
    completed_trials = 0
    status = "error"
    error_message: str | None = None
    exit_code = 1
    raw_data = np.empty((0, 0))
    validation_mode = (
        "technical_validation_override"
        if protocol_is_draft or channel_is_draft or arguments.allow_draft_protocol or arguments.allow_draft_channel_config
        else "formal_candidate"
    )
    protocol_provenance = {
        "source_path": str(arguments.protocol.resolve()),
        "protocol_id": protocol_source.get("protocol_id", ""),
        "schema_version": protocol_source.get("schema_version"),
        "status": protocol_status,
        "sha256": sha256(arguments.protocol),
    }
    channel_provenance = None
    if channel_config is not None:
        channel_provenance = {
            "source_path": str(arguments.channel_config.resolve()),
            "config_version": channel_config.get("config_version", ""),
            "profile_type": channel_config.get("profile_type", ""),
            "sha256": sha256(arguments.channel_config),
            "warnings": list(channel_warnings),
        }

    params = BrainFlowInputParams()
    board_id = BoardIds.CYTON_BOARD
    params.serial_port = arguments.port

    def elapsed_recording() -> float:
        return 0.0 if recording_started is None else time.perf_counter() - recording_started

    def pump_live_waveform() -> None:
        if live_waveform_writer is not None:
            live_waveform_writer.pump()

    def add_event(
        name: str,
        marker: int,
        trial_index: int = -1,
        target_id: str = "",
        frequency_hz: int | str = "",
        *,
        source: str = "software_protocol",
        label_source: str = "",
        event_monotonic: float | None = None,
    ) -> None:
        if board is not None and streaming:
            board.insert_marker(float(marker))
        event_id = len(events) + 1
        planned_target = target_id if name in {"stimulus_onset", "stimulus_offset"} else ""
        presented_target = planned_target
        gaze_target = target_id if name == "gaze_target_mark" else ""
        events.append(
            {
                "event_id": event_id,
                "event_name": name,
                "eye_side": arguments.eye_side,
                "dataset_name_base": dataset_name_base,
                "dataset_name": dataset_name,
                "screen_index": selected_screen_index if selected_screen_index is not None else "",
                "screen_name": str((display_metadata.get("active") or {}).get("name") or ""),
                "screen_geometry": _screen_geometry_text(display_metadata.get("active") or {}),
                "screen_mapping": _screen_mapping_text(display_metadata),
                "stimulus_region": _screen_geometry_text(display_metadata.get("active_region") or {}),
                "graphics_vendor": (display_metadata.get("rendering") or {}).get("vendor", ""),
                "graphics_renderer": (display_metadata.get("rendering") or {}).get("renderer", ""),
                "graphics_version": (display_metadata.get("rendering") or {}).get("version", ""),
                "swap_interval": (display_metadata.get("rendering") or {}).get("swap_interval", ""),
                "source": source,
                "label_source": label_source,
                "trial_index": trial_index,
                "target_id": target_id,
                "planned_target_id": planned_target,
                "presented_target_id": presented_target,
                "gaze_target_id": gaze_target,
                "eeg_predicted_target_id": "",
                "target_confidence": "",
                "frequency_hz": frequency_hz,
                "marker_code": marker,
                "sample_index": "",
                "sample_time_s": "",
                "recording_time_s": "",
                "wall_time_iso": iso_now(),
                "monotonic_s": round(
                    (event_monotonic or time.perf_counter()) - session_started_monotonic,
                    6,
                ),
            }
        )

    def emit(phase: str, **extra: Any) -> None:
        reporter.emit(
            phase=phase,
            session_id=session_id,
            output_dir=str(session_dir),
            board=arguments.board,
            recording_elapsed_s=round(elapsed_recording(), 3),
            recording_duration_s=protocol.recording_duration_s,
            completed_trials=completed_trials,
            event_count=len(events),
            serial_port=selected_port,
            requested_serial_port=arguments.port if arguments.board == "cyton" else None,
            screen_index=selected_screen_index,
            eye_side=arguments.eye_side,
            dataset_name_base=dataset_name_base,
            dataset_name=dataset_name,
            display=display_metadata,
            screen_mapping={
                "left": display_metadata.get("left"),
                "right": display_metadata.get("right"),
            },
            stimulus_region=display_metadata.get("active_region"),
            **extra,
        )

    try:
        emit("preparing")
        board, selected_port, _port_failures = _prepare_cyton_board(
            board_id, params, arguments.port
        )
        prepared = True
        if arguments.headless:
            display_metadata = {
                "mode": "headless",
                "eye_side": arguments.eye_side,
                "active": None,
                "inactive": None,
                "left": None,
                "right": None,
                "active_region": None,
                "refresh_rate_hz": 0.0,
                "protocol_refresh_rate_hz": int(protocol.refresh_rate_hz),
                "refresh_rate_status": "not_applicable",
            }
        if not arguments.headless:
            application, window, screen_mapping = _create_stimulus_window(
                arguments.eye_side
            )
            selected_screen_index = int(screen_mapping[arguments.eye_side]["screen_index"])
            screen = screen_mapping[arguments.eye_side]["screen"]
            inactive_side = "right" if arguments.eye_side == "left" else "left"
            selected_display = _screen_metadata(screen, selected_screen_index)
            regions = {
                side: {**selected_display, "region_geometry": screen_mapping[side]["region_geometry"]}
                for side in ("left", "right")
            }
            refresh_rate = selected_display["refresh_rate_hz"]
            display_metadata = {
                "mode": "primary_screen_halves",
                "eye_side": arguments.eye_side,
                "active": selected_display,
                "inactive": regions[inactive_side],
                "active_region": {"geometry": regions[arguments.eye_side]["region_geometry"]},
                "left": regions["left"],
                "right": regions["right"],
                "refresh_rate_hz": refresh_rate,
                "protocol_refresh_rate_hz": int(protocol.refresh_rate_hz),
                "rendering": dict(window.rendering_metadata),
                "refresh_rate_status": (
                    "unknown" if refresh_rate <= 0 else
                    "matched" if abs(refresh_rate - protocol.refresh_rate_hz) <= 1.0 else "mismatch"
                ),
            }
            if refresh_rate <= 0 or abs(refresh_rate - protocol.refresh_rate_hz) > 1.0:
                raise RuntimeError(
                    "visual stimulus refresh rate does not match the protocol "
                    f"({refresh_rate:g} Hz detected, {protocol.refresh_rate_hz} Hz required)"
                )

        # Start the EEG stream before the visible countdown. The countdown is
        # therefore present in the raw recording, while session_start remains
        # the analysis anchor for the first experimental trial.
        if board is not None:
            board.start_stream()
            streaming = True
            if live_waveform_path is not None:
                live_waveform_writer = _LiveWaveformWriter(
                    live_waveform_path, board, int(board_id)
                )
            acquisition_started = time.perf_counter()
            add_event(
                "acquisition_start",
                protocol.acquisition_start_marker,
                source="acquisition_stream",
                label_source="board_marker",
            )

        countdown_started = time.perf_counter()

        def countdown_update(elapsed: float) -> None:
            pump_live_waveform()
            remaining = max(0, math.ceil(countdown_s - elapsed))
            if window is not None:
                window.show_message(str(remaining), flash=True)
            emit("countdown", countdown_remaining_s=remaining)

        _wait_phase(
            countdown_s,
            cancel_file=cancel_file,
            application=application,
            window=window,
            update=countdown_update,
        )
        recording_started = time.perf_counter()
        add_event(
            "session_start",
            protocol.session_start_marker,
            source="protocol_timeline",
            label_source="board_marker" if board is not None else "software_clock",
        )

        trials = protocol.build_trials()
        target_ids = [target_id for target_id, _frequency in protocol.targets]
        for trial in trials:
            target_index = target_ids.index(trial.target_id)

            def stimulus_progress(within: float) -> None:
                pump_live_waveform()
                emit(
                    "running",
                    trial_index=trial.index,
                    trial_count=protocol.trial_count,
                    target_id=trial.target_id,
                    target_index=target_index,
                    frequency_hz=trial.frequency_hz,
                    trial_phase="stimulus",
                    trial_elapsed_s=round(within, 3),
                )

            if arguments.headless:
                add_event(
                    "stimulus_onset",
                    trial.onset_marker,
                    trial.index,
                    trial.target_id,
                    trial.frequency_hz,
                )
                _wait_phase(
                    protocol.stimulus_s,
                    cancel_file=cancel_file,
                    application=None,
                    window=None,
                    update=stimulus_progress,
                )
                add_event(
                    "stimulus_offset",
                    trial.offset_marker,
                    trial.index,
                    trial.target_id,
                    trial.frequency_hz,
                )
            else:
                def gaze_mark(target_index: int, marked_at: float) -> None:
                    if not 0 <= target_index < len(target_ids):
                        return
                    marked_target_id = target_ids[target_index]
                    add_event(
                        "gaze_target_mark",
                        protocol.gaze_marker_base + target_index + 1,
                        trial.index,
                        marked_target_id,
                        trial.frequency_hz,
                        source="manual_gaze_annotation",
                        label_source="keyboard_self_report_or_operator",
                        event_monotonic=marked_at,
                    )

                _present_stimulus(
                    application=application,
                    window=window,
                    target_index=target_index,
                    frequency_hz=trial.frequency_hz,
                    duration_s=protocol.stimulus_s,
                    refresh_rate_hz=protocol.refresh_rate_hz,
                    cancel_file=cancel_file,
                    trial_index=trial.index,
                    eye_side=arguments.eye_side,
                    screen_index=selected_screen_index or 0,
                    dataset_name_base=dataset_name_base,
                    dataset_name=dataset_name,
                    screen_name=str((display_metadata.get("active") or {}).get("name") or ""),
                    screen_geometry=_screen_geometry_text(display_metadata.get("active") or {}),
                    screen_mapping=_screen_mapping_text(display_metadata),
                    stimulus_region=_screen_geometry_text(display_metadata.get("active_region") or {}),
                    frame_rows=frame_rows,
                    progress=stimulus_progress,
                    on_gaze_mark=gaze_mark,
                    on_stimulus_onset=lambda actual: add_event(
                        "stimulus_onset",
                        trial.onset_marker,
                        trial.index,
                        trial.target_id,
                        trial.frequency_hz,
                        event_monotonic=actual,
                    ),
                    on_stimulus_offset=lambda actual: add_event(
                        "stimulus_offset",
                        trial.offset_marker,
                        trial.index,
                        trial.target_id,
                        trial.frequency_hz,
                        event_monotonic=actual,
                    ),
                )
            completed_trials += 1
            if trial.index < protocol.trial_count - 1 and protocol.post_trial_rest_s:
                if window is not None:
                    window.show_message("+")

                def rest_progress(within: float) -> None:
                    pump_live_waveform()
                    emit(
                        "running",
                        trial_index=trial.index,
                        trial_count=protocol.trial_count,
                        target_id=trial.target_id,
                        target_index=target_index,
                        frequency_hz=trial.frequency_hz,
                        trial_phase="rest",
                        trial_elapsed_s=round(within, 3),
                    )

                _wait_phase(
                    protocol.post_trial_rest_s,
                    cancel_file=cancel_file,
                    application=application,
                    window=window,
                    update=rest_progress,
                )

        add_event("session_end", protocol.session_end_marker)
        time.sleep(0.1)
        status = "completed"
        exit_code = 0
    except UserAbort:
        add_event("abort", protocol.abort_marker)
        time.sleep(0.1)
        status = "aborted"
        exit_code = 130
    except Exception as error:
        error_message = f"{type(error).__name__}: {error}"
        try:
            add_event("error", protocol.abort_marker)
        except Exception:
            pass
        status = "error"
    finally:
        pump_live_waveform()
        if live_waveform_writer is not None:
            live_waveform_writer.close()
        if window is not None:
            window.close()
        if board is not None:
            if streaming:
                try:
                    board.stop_stream()
                except Exception:
                    pass
                try:
                    raw_data = board.get_board_data()
                except Exception:
                    raw_data = np.empty((0, 0))
            if prepared:
                try:
                    board.release_session()
                except Exception:
                    pass

    _annotate_event_samples(events, raw_data, board_id)
    raw_path = session_dir / "raw_brainflow.tsv"
    sample_rate_hz = 0 if board_id is None else int(BoardShim.get_sampling_rate(board_id))
    column_definitions = write_brainflow_tsv(
        raw_path, raw_data, board_id, sampling_rate_hz=sample_rate_hz
    )
    columns_path = session_dir / "raw_columns.tsv"
    write_column_definitions(columns_path, column_definitions)
    events_path = session_dir / "events.tsv"
    write_events(events_path, events)
    frames_path = session_dir / "frame_timing.tsv"
    _write_frame_log(frames_path, frame_rows)
    protocol_path = session_dir / "protocol.json"
    protocol_value = asdict(protocol)
    protocol_value["source"] = str(protocol.source) if protocol.source else None
    write_json(protocol_path, protocol_value)
    display_metadata["frame_timing"] = _frame_timing_summary(
        frame_rows, expected_refresh_rate_hz=protocol.refresh_rate_hz
    )
    source_config_path = session_dir / "ssvep_config.json"
    write_json(
        source_config_path,
        {
            **protocol_source,
            "runtime_overrides": {
                "repetitions": arguments.repetitions,
                "stimulus_seconds": arguments.stimulus_seconds,
                "rest_seconds": arguments.rest_seconds,
                "countdown_seconds": arguments.countdown_seconds,
                "eye_side": arguments.eye_side,
                "dataset_name_base": dataset_name_base,
                "dataset_name": dataset_name,
            },
            "provenance": protocol_provenance,
            "eye_side": arguments.eye_side,
            "dataset_name_base": dataset_name_base,
            "dataset_name": dataset_name,
            "display": display_metadata,
        },
    )
    quality_path = session_dir / "quality.json"
    quality_report = _build_quality_report(
        raw_data=raw_data,
        board_id=board_id,
        frame_rows=frame_rows,
        recording_status=status,
        error=error_message,
    )
    quality_report["frame_timing"] = display_metadata["frame_timing"]
    quality_report["graphics"] = display_metadata.get("rendering", {})
    write_json(quality_path, quality_report)
    output_files = [
        raw_path,
        columns_path,
        events_path,
        frames_path,
        protocol_path,
        source_config_path,
        quality_path,
    ]
    if channel_config is not None:
        channels_path = session_dir / "channel_config.json"
        write_json(channels_path, channel_config)
        output_files.append(channels_path)

    sample_count = int(raw_data.shape[1]) if raw_data.ndim == 2 else 0
    acquisition_duration = (
        max(0.0, time.perf_counter() - acquisition_started)
        if acquisition_started is not None
        else 0.0
    )
    session_path = session_dir / "session.json"
    session_value = {
        "schema_version": 1,
        "session_id": session_id,
        "status": status,
        "participant_id": arguments.participant,
        "session_name": dataset_name,
        "dataset_name_base": dataset_name_base,
        "eye_side": arguments.eye_side,
        "user_id": arguments.user_id,
        "user_name": arguments.user_name,
        "user_link_status": "active" if arguments.user_id else "unlinked",
        "board": arguments.board,
        "serial_port": selected_port,
        "requested_serial_port": arguments.port if arguments.board == "cyton" else None,
        "screen_index": selected_screen_index,
        "display": display_metadata,
        "screen_mapping": {
            "left": display_metadata.get("left"),
            "right": display_metadata.get("right"),
        },
        "stimulus_region": display_metadata.get("active_region"),
        "sampling_rate_hz": 0 if board_id is None else int(BoardShim.get_sampling_rate(board_id)),
        "channel_count": 0 if board_id is None else len(BoardShim.get_eeg_channels(board_id)),
        "started_at": started_at,
        "ended_at": iso_now(),
        "recording_duration_s": round(elapsed_recording(), 3),
        "protocol_duration_s": round(elapsed_recording(), 3),
        "acquisition_duration_s": round(acquisition_duration, 3),
        "acquisition_includes_countdown": bool(acquisition_started is not None),
        "expected_duration_s": protocol.recording_duration_s,
        "expected_samples_per_channel": protocol.expected_samples_per_channel,
        "completed_trials": completed_trials,
        "trial_count": protocol.trial_count,
        "recorded_samples_per_channel": sample_count,
        "event_count": len(events),
        "frame_count": len(frame_rows),
        "simulated": False,
        "headless": arguments.headless,
        "error": error_message,
        "channel_config_warnings": channel_warnings,
        "validation_mode": validation_mode,
        "validation_flags": {
            "allow_draft_protocol": bool(arguments.allow_draft_protocol),
            "allow_draft_channel_config": bool(arguments.allow_draft_channel_config),
        },
        "protocol_provenance": protocol_provenance,
        "channel_config_provenance": channel_provenance,
        "preflight": preflight_report,
        "application": {
            "name": PRODUCT_NAME,
            "version": PRODUCT_VERSION,
            "semantic_version": PRODUCT_SEMVER,
            "release_date": RELEASE_DATE,
            "description": PRODUCT_DESCRIPTION,
        },
    }
    write_json(session_path, session_value)
    output_files.append(session_path)
    if preflight_report is not None:
        preflight_path = session_dir / "preflight.json"
        write_json(preflight_path, preflight_report)
        output_files.append(preflight_path)
    manifest_path = session_dir / "manifest.csv"
    write_manifest(manifest_path, session_id, output_files)

    emit(
        status,
        status=status,
        result={
            "session_id": session_id,
            "output_dir": str(session_dir),
            "recorded_samples_per_channel": sample_count,
            "event_count": len(events),
            "completed_trials": completed_trials,
            "simulated": False,
            "persisted": True,
            "error": error_message,
        },
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
