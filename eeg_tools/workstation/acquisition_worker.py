"""Isolated BrainFlow SSVEP acquisition worker with optional Qt stimulus output."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

from eeg_tools.config import ConfigError, validate_channel_config
from eeg_tools.session_files import iso_now, write_events, write_json, write_manifest
from neurostation_contract import (
    PRODUCT_DESCRIPTION,
    PRODUCT_NAME,
    PRODUCT_SEMVER,
    PRODUCT_VERSION,
    RELEASE_DATE,
)

from .ssvep import SSVEPProtocol, SSVEPProtocolError
from .device_discovery import candidate_serial_ports


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
        print(json.dumps(payload, ensure_ascii=False), flush=True)
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


def _create_stimulus_window(screen_index: int, *, fullscreen_flicker: bool = False):
    from PySide6.QtCore import Qt, QRect
    from PySide6.QtGui import QColor, QFont, QKeyEvent, QPainter, QPaintEvent
    from PySide6.QtWidgets import QApplication, QWidget

    application = QApplication.instance() or QApplication(sys.argv[:1])

    class StimulusWindow(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.abort_requested = False
            self.target_index: int | None = None
            self.lit = False
            self.fullscreen_flicker = fullscreen_flicker
            self.message = ""
            self.message_flash = False
            self.setCursor(Qt.CursorShape.BlankCursor)
            self.setStyleSheet("background: black;")

        def keyPressEvent(self, event: QKeyEvent) -> None:
            if event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
                self.abort_requested = True
            else:
                super().keyPressEvent(event)

        def paintEvent(self, event: QPaintEvent) -> None:
            painter = QPainter(self)
            if self.message:
                painter.fillRect(self.rect(), QColor("white") if self.message_flash else QColor("black"))
            else:
                background = (
                    QColor("white")
                    if self.fullscreen_flicker and self.lit
                    else QColor("black")
                )
                painter.fillRect(self.rect(), background)
            if self.message:
                painter.setPen(QColor("black") if self.message_flash else QColor("white"))
                painter.setFont(QFont("Sans Serif", max(28, self.height() // 12)))
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.message)
                return
            side = min(self.width(), self.height()) // 5
            positions = (
                (self.width() // 4, self.height() // 3),
                (self.width() * 3 // 4, self.height() // 3),
                (self.width() // 4, self.height() * 2 // 3),
                (self.width() * 3 // 4, self.height() * 2 // 3),
            )
            for index, (center_x, center_y) in enumerate(positions):
                rectangle = QRect(center_x - side // 2, center_y - side // 2, side, side)
                if self.fullscreen_flicker:
                    selected = index == self.target_index
                    color = (
                        QColor("black")
                        if selected and self.lit
                        else QColor("white")
                        if selected
                        else QColor(226, 226, 226)
                        if self.lit
                        else QColor(28, 28, 28)
                    )
                else:
                    color = (
                        QColor("white")
                        if index == self.target_index and self.lit
                        else QColor(28, 28, 28)
                    )
                painter.fillRect(rectangle, color)
                painter.setPen(
                    QColor(90, 90, 90)
                    if not self.fullscreen_flicker
                    else QColor(120, 120, 120)
                    if self.lit
                    else QColor(150, 150, 150)
                )
                painter.drawRect(rectangle)
            patch_side = max(24, min(self.width(), self.height()) // 16)
            patch = QRect(self.width() - patch_side - 12, self.height() - patch_side - 12, patch_side, patch_side)
            painter.fillRect(patch, QColor("white") if self.lit else QColor("black"))

        def show_message(self, message: str, *, flash: bool = False) -> None:
            self.message = message
            self.target_index = None
            self.lit = False
            self.message_flash = bool(flash and int(time.perf_counter() * 4) % 2)
            self.repaint()
            application.processEvents()

        def show_target(self, target_index: int, lit: bool) -> None:
            self.message = ""
            self.target_index = target_index
            self.lit = lit
            self.repaint()
            application.processEvents()

    screens = application.screens()
    if not screens:
        raise RuntimeError("No usable display was detected for the SSVEP stimulus")
    # A saved monitor index can become stale after docking/undocking. Always
    # fall back to the first Qt screen so a task remains startable.
    selected_index = screen_index if 0 <= screen_index < len(screens) else 0
    window = StimulusWindow()
    window.setScreen(screens[selected_index])
    window.setGeometry(screens[selected_index].geometry())
    window.showFullScreen()
    application.processEvents()
    return application, window


def _prepare_cyton_board(
    board_id: int,
    params: Any,
    requested_port: str,
) -> tuple[Any, str, list[dict[str, str]]]:
    """Prepare the first Cyton endpoint accepted by BrainFlow.

    BrainFlow's native error ``BOARD_NOT_READY_ERROR:7`` is commonly caused by
    a wrong COM port or another process holding the dongle. Trying every
    discovered endpoint turns that opaque failure into a reliable automatic
    scan while preserving explicit-port behavior.
    """

    from brainflow.board_shim import BoardShim

    candidates = candidate_serial_ports(requested_port)
    if not candidates:
        raise RuntimeError(
            "No serial ports were detected. Connect the OpenBCI USB dongle, "
            "close OpenBCI GUI/other serial monitors, then retry."
        )
    failures: list[dict[str, str]] = []
    for candidate in candidates:
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
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            if board is not None:
                try:
                    board.release_session()
                except Exception:
                    pass
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
    frame_rows: list[dict[str, Any]],
    progress: Callable[[float], None],
) -> None:
    frames_per_cycle = refresh_rate_hz // frequency_hz
    frame_total = max(1, round(duration_s * refresh_rate_hz))
    started = time.perf_counter()
    previous_actual: float | None = None
    for frame_index in range(frame_total):
        deadline = started + frame_index / refresh_rate_hz
        while True:
            now = time.perf_counter()
            if now >= deadline:
                break
            application.processEvents()
            _check_abort(cancel_file, window)
            time.sleep(min(0.002, deadline - now))
        actual = time.perf_counter()
        lit = (frame_index % frames_per_cycle) < (frames_per_cycle / 2)
        window.show_target(target_index, lit)
        interval = None if previous_actual is None else actual - previous_actual
        dropped = 0 if interval is None else max(0, round(interval * refresh_rate_hz) - 1)
        frame_rows.append(
            {
                "trial_index": trial_index,
                "frame_index": frame_index,
                "scheduled_s": frame_index / refresh_rate_hz,
                "actual_s": actual - started,
                "lateness_ms": (actual - deadline) * 1000,
                "dropped_since_previous": dropped,
                "lit": int(lit),
            }
        )
        previous_actual = actual
        if frame_index % max(1, refresh_rate_hz // 10) == 0:
            progress(min(duration_s, actual - started))
    final_deadline = started + duration_s
    _wait_phase(
        max(0.0, final_deadline - time.perf_counter()),
        cancel_file=cancel_file,
        application=application,
        window=window,
    )
    window.show_target(target_index, False)


def _write_frame_log(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = (
        "trial_index",
        "frame_index",
        "scheduled_s",
        "actual_s",
        "lateness_ms",
        "dropped_since_previous",
        "lit",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("\t".join(columns) + "\n")
        for row in rows:
            stream.write("\t".join(str(row[column]) for column in columns) + "\n")


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--channel-config", type=Path, default=DEFAULT_CHANNELS)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--participant", required=True)
    parser.add_argument("--session-name", required=True)
    parser.add_argument("--board", choices=("cyton", "synthetic", "demo"), default="cyton")
    parser.add_argument("--port", default="AUTO")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--stimulus-seconds", type=float)
    parser.add_argument("--rest-seconds", type=float)
    parser.add_argument("--countdown-seconds", type=float)
    parser.add_argument("--screen-index", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--acknowledge-flicker-risk", action="store_true")
    parser.add_argument("--allow-draft-protocol", action="store_true")
    parser.add_argument("--allow-draft-channel-config", action="store_true")
    parser.add_argument("--status-file", type=Path)
    parser.add_argument("--cancel-file", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.countdown_seconds is not None and arguments.countdown_seconds < 0:
        parser.error("--countdown-seconds must be non-negative")
    if not arguments.headless and not arguments.acknowledge_flicker_risk:
        parser.error("visual stimulus requires --acknowledge-flicker-risk")
    try:
        protocol = SSVEPProtocol.load(arguments.protocol).with_runtime_parameters(
            repetitions=arguments.repetitions,
            stimulus_s=arguments.stimulus_seconds,
            rest_s=arguments.rest_seconds,
        )
    except SSVEPProtocolError as error:
        parser.error(str(error))
    protocol_source = json.loads(arguments.protocol.read_text(encoding="utf-8"))
    if (
        arguments.board == "cyton"
        and str(protocol_source.get("status", "")).lower().startswith("draft")
        and not arguments.allow_draft_protocol
    ):
        parser.error("Cyton acquisition refuses a draft protocol; approve it or use --allow-draft-protocol for technical validation")

    channel_config: dict[str, Any] | None = None
    channel_warnings: list[str] = []
    if arguments.board == "cyton":
        try:
            channel_config, channel_warnings = _read_channel_config(arguments.channel_config)
        except ConfigError as error:
            parser.error(str(error))
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
    selected_screen_index = arguments.screen_index
    prepared = False
    streaming = False
    application = None
    window = None
    recording_started: float | None = None
    completed_trials = 0
    status = "error"
    error_message: str | None = None
    exit_code = 1
    raw_data = np.empty((0, 0))

    params = BrainFlowInputParams()
    board_id = (
        None
        if arguments.board == "demo"
        else (BoardIds.SYNTHETIC_BOARD if arguments.board == "synthetic" else BoardIds.CYTON_BOARD)
    )
    if arguments.board == "cyton":
        params.serial_port = arguments.port

    def elapsed_recording() -> float:
        return 0.0 if recording_started is None else time.perf_counter() - recording_started

    def add_event(
        name: str,
        marker: int,
        trial_index: int = -1,
        target_id: str = "",
        frequency_hz: int | str = "",
    ) -> None:
        if board is not None and streaming:
            board.insert_marker(float(marker))
        events.append(
            {
                "event_name": name,
                "trial_index": trial_index,
                "target_id": target_id,
                "frequency_hz": frequency_hz,
                "marker_code": marker,
                "wall_time_iso": iso_now(),
                "monotonic_s": round(time.perf_counter() - session_started_monotonic, 6),
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
            **extra,
        )

    try:
        emit("preparing")
        if board_id is not None:
            if arguments.board == "cyton":
                board, selected_port, _port_failures = _prepare_cyton_board(
                    board_id, params, arguments.port
                )
                prepared = True
            else:
                board = BoardShim(board_id, params)
                board.prepare_session()
                prepared = True
        if not arguments.headless:
            application, window = _create_stimulus_window(
                arguments.screen_index,
                fullscreen_flicker=arguments.board == "demo",
            )
            # Qt has already resolved an invalid saved index to screen 0 in
            # _create_stimulus_window; retain the effective value in metadata.
            selected_screen_index = (
                arguments.screen_index
                if 0 <= arguments.screen_index < len(application.screens())
                else 0
            )

        countdown_started = time.perf_counter()

        def countdown_update(elapsed: float) -> None:
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
        if board is not None:
            board.start_stream()
            streaming = True
        recording_started = time.perf_counter()
        add_event("session_start", protocol.session_start_marker)

        trials = protocol.build_trials()
        target_ids = [target_id for target_id, _frequency in protocol.targets]
        for trial in trials:
            target_index = target_ids.index(trial.target_id)
            add_event(
                "stimulus_onset",
                trial.onset_marker,
                trial.index,
                trial.target_id,
                trial.frequency_hz,
            )

            def stimulus_progress(within: float) -> None:
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
                _wait_phase(
                    protocol.stimulus_s,
                    cancel_file=cancel_file,
                    application=None,
                    window=None,
                    update=stimulus_progress,
                )
            else:
                _present_stimulus(
                    application=application,
                    window=window,
                    target_index=target_index,
                    frequency_hz=trial.frequency_hz,
                    duration_s=protocol.stimulus_s,
                    refresh_rate_hz=protocol.refresh_rate_hz,
                    cancel_file=cancel_file,
                    trial_index=trial.index,
                    frame_rows=frame_rows,
                    progress=stimulus_progress,
                )
            add_event(
                "stimulus_offset",
                trial.offset_marker,
                trial.index,
                trial.target_id,
                trial.frequency_hz,
            )
            completed_trials += 1
            if trial.index < protocol.trial_count - 1 and protocol.post_trial_rest_s:
                if window is not None:
                    window.show_message("+")

                def rest_progress(within: float) -> None:
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

    raw_path = session_dir / "raw_brainflow.tsv"
    if raw_data.size:
        # BrainFlow's native writer cannot open non-ASCII paths on Windows.
        # Python's file handle keeps dataset paths Unicode-safe on all platforms.
        with raw_path.open("w", encoding="utf-8", newline="") as stream:
            np.savetxt(stream, raw_data, delimiter="\t", fmt="%.10g")
    else:
        raw_path.write_text("", encoding="utf-8")
    events_path = session_dir / "events.tsv"
    write_events(events_path, events)
    frames_path = session_dir / "frame_timing.tsv"
    _write_frame_log(frames_path, frame_rows)
    protocol_path = session_dir / "protocol.json"
    protocol_value = asdict(protocol)
    protocol_value["source"] = str(protocol.source) if protocol.source else None
    write_json(protocol_path, protocol_value)
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
            },
        },
    )
    quality_path = session_dir / "quality.json"
    write_json(
        quality_path,
        _build_quality_report(
            raw_data=raw_data,
            board_id=board_id,
            frame_rows=frame_rows,
            recording_status=status,
            error=error_message,
        ),
    )
    output_files = [raw_path, events_path, frames_path, protocol_path, source_config_path, quality_path]
    if channel_config is not None:
        channels_path = session_dir / "channel_config.json"
        write_json(channels_path, channel_config)
        output_files.append(channels_path)

    sample_count = int(raw_data.shape[1]) if raw_data.ndim == 2 else 0
    session_path = session_dir / "session.json"
    session_value = {
        "schema_version": 1,
        "session_id": session_id,
        "status": status,
        "participant_id": arguments.participant,
        "session_name": arguments.session_name,
        "board": arguments.board,
        "serial_port": selected_port,
        "requested_serial_port": arguments.port if arguments.board == "cyton" else None,
        "screen_index": selected_screen_index,
        "sampling_rate_hz": 0 if board_id is None else int(BoardShim.get_sampling_rate(board_id)),
        "channel_count": 0 if board_id is None else len(BoardShim.get_eeg_channels(board_id)),
        "started_at": started_at,
        "ended_at": iso_now(),
        "recording_duration_s": round(elapsed_recording(), 3),
        "expected_duration_s": protocol.recording_duration_s,
        "completed_trials": completed_trials,
        "trial_count": protocol.trial_count,
        "recorded_samples_per_channel": sample_count,
        "event_count": len(events),
        "frame_count": len(frame_rows),
        "simulated": arguments.board in {"synthetic", "demo"},
        "headless": arguments.headless,
        "error": error_message,
        "channel_config_warnings": channel_warnings,
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
            "simulated": arguments.board in {"synthetic", "demo"},
            "persisted": True,
            "error": error_message,
        },
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
