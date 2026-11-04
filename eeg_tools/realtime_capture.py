"""Real Cyton continuous acquisition for the standalone command UI.

stdout is an NDJSON transport. Electrode names describe the supplied configuration;
this worker never claims that physical placement has been verified. Hardware is
loaded lazily and test adapters are available only through Python injection.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
import json
import logging
import math
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any, Callable, TextIO

import numpy as np

from eeg_tools import config as config_tools
from eeg_tools.session_files import (
    brainflow_column_definitions, format_raw_number, sha256,
    write_column_definitions, write_json, write_manifest,
)

CHANNEL_NAMES = ("Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2")
SAMPLE_RATE = 250
PREFLIGHT_MAX_AGE_S = 300.0
POLL_S = 0.05
LOG = logging.getLogger(__name__)


def _utc(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat(timespec="milliseconds")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


class StopController:
    """The reader owns no Qt or BrainFlow objects; the acquisition thread releases."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self.reason = ""
        self._lock = threading.Lock()

    def request(self, reason: str = "stop_command") -> None:
        with self._lock:
            if not self.event.is_set():
                self.reason = reason
                self.event.set()


def start_stdin_reader(stream: TextIO, stop: StopController) -> threading.Thread:
    def read() -> None:
        try:
            while not stop.event.is_set():
                line = stream.readline()
                if not line:
                    stop.request("stdin_eof")
                    return
                try:
                    message = json.loads(line)
                except (ValueError, TypeError):
                    LOG.warning("Ignoring malformed stdin JSON")
                    continue
                if isinstance(message, dict) and message.get("type") == "stop":
                    stop.request("stop_command")
                    return
        except Exception:
            LOG.exception("stdin reader failed")
            stop.request("stdin_error")

    reader = threading.Thread(target=read, name="cyton-stdin-reader", daemon=True)
    reader.start()
    return reader


class NDJSONEmitter:
    def __init__(self, stream: TextIO, stop: StopController) -> None:
        self.stream = stream
        self.stop = stop
        self.disconnected = False

    def __call__(self, value: dict[str, Any]) -> None:
        if self.disconnected:
            return
        try:
            self.stream.write(json.dumps(_json_safe(value), ensure_ascii=True, allow_nan=False) + "\n")
            self.stream.flush()
        except (BrokenPipeError, OSError, ValueError):
            self.disconnected = True
            self.stop.request("stdout_closed")


def _port(value: str) -> str:
    # Windows device-path spelling and ordinary COM spelling identify the same port.
    return str(value).strip().removeprefix("\\\\.\\").upper()


def load_channel_config(path: Path) -> tuple[dict[str, Any], str]:
    # Hash exactly the bytes parsed, avoiding a config/hash race.
    import hashlib

    contents = path.read_bytes()
    value = json.loads(contents.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Channel configuration must be an object")
    warnings = config_tools.validate_channel_config(value, path)
    complete_validator = getattr(config_tools, "_validate_complete_channel_map", None)
    if complete_validator is not None:
        complete_validator(value, path)
    if warnings:
        raise ValueError("Draft/incomplete channel configuration: " + "; ".join(warnings))
    for key in ("status", "config_version", "profile_type"):
        label = str(value.get(key, "")).lower()
        if any(marker in label for marker in ("draft", "mvp", "template", "formal_candidate")):
            raise ValueError(f"Draft/template channel configuration: {key}")
    if value.get("status") not in (None, "formal"):
        raise ValueError("Channel configuration status must be formal")
    if value["board"] != "OpenBCI Cyton" or value["sampling_rate_hz"] != SAMPLE_RATE:
        raise ValueError("A real OpenBCI Cyton with 250 Hz configuration is required")
    channels = value["channels"]
    positions = tuple(item["electrode_position"] for item in channels)
    if positions != CHANNEL_NAMES or [item["gui_index"] for item in channels] != list(range(8)):
        raise ValueError("The complete eight-channel Cyton map must be Fp1,Fp2,C3,C4,P7,P8,O1,O2 in input order")
    for index, item in enumerate(channels, 1):
        if "board_input" in item and item["board_input"] != f"N{index}P":
            raise ValueError("Channel board_input does not match Cyton input order")
    return value, hashlib.sha256(contents).hexdigest()


def run_preflight(
    port: str, channel_config: Path, session_dir: Path, *,
    stop: StopController | None = None,
    emit: Callable[[dict[str, Any]], None] | None = None,
    preflight_runner: Callable[..., Any] | None = None,
    wall_time: Callable[[], float] = time.time,
) -> int:
    stop = stop or StopController()
    emit = emit or NDJSONEmitter(sys.stdout, stop)
    started = wall_time()
    if any((session_dir / name).exists() for name in
           ("session.json", "quality.json", "manifest.csv", "raw_brainflow.tsv", "raw_columns.tsv")):
        emit({"type": "error", "code": "session_exists",
              "message": "Cannot replace preflight evidence in an existing recording"})
        return 2
    digest = ""
    positions: list[str] = []
    report: dict[str, Any] = {"status": "failed", "requested_port": port,
                              "selected_port": "", "checks": [], "error": ""}
    try:
        digest = sha256(channel_config)
        _, digest = load_channel_config(channel_config)
        positions = list(CHANNEL_NAMES)
        if stop.event.is_set():
            raise RuntimeError("Preflight cancelled before opening hardware")
        if preflight_runner is None:
            from eeg_tools.workstation.preflight import run_cyton_preflight
            preflight_runner = run_cyton_preflight
        # The existing helper only observes cancellation inside its callback path.
        with contextlib.redirect_stdout(sys.stderr):
            result = preflight_runner(
                port, 3.0, cancel_event=stop.event,
                sample_callback=lambda _batch: None, sleep_fn=stop.event.wait,
            )
        report = result.as_dict() if hasattr(result, "as_dict") else dict(result)
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        LOG.error("Preflight failed: %s", error)
    finished = wall_time()
    selected = str(report.get("selected_port", ""))
    status = report.get("status", "failed")
    if status not in {"passed", "warning"} or stop.event.is_set():
        status = "failed"
    if status != "failed" and (not selected or _port(selected) == "AUTO"):
        status = "failed"
        report["error"] = "Preflight did not select a concrete serial port"
    envelope = {
        "schema_version": 1, "type": "cyton_realtime_preflight", "status": status,
        "requested_port": port, "port": selected,
        "channel_config_sha256": digest, "configured_positions": positions,
        "position_verification": "configuration_only",
        "started_at_utc": _utc(started), "completed_at_utc": _utc(finished),
        "completed_timestamp_s": finished, "cancelled": stop.event.is_set(),
        "report": _json_safe(report),
    }
    try:
        session_dir.mkdir(parents=True, exist_ok=True)
        write_json(session_dir / "preflight.json", envelope)
    except Exception as error:
        emit({"type": "error", "code": "preflight_write_failed", "message": str(error)})
        return 2
    emit({"type": "preflight", "report": envelope})
    return 2 if status == "failed" else 0


def validate_preflight(session_dir: Path, digest: str, port: str, now: float) -> dict[str, Any]:
    value = json.loads((session_dir / "preflight.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1 or value.get("type") != "cyton_realtime_preflight":
        raise ValueError("Missing or unsupported realtime preflight schema")
    inner = value.get("report")
    if value.get("status") not in {"passed", "warning"} or not isinstance(inner, dict) or inner.get("status") not in {"passed", "warning"} or value.get("cancelled"):
        raise ValueError("Preflight must be passed or warning; failed/degraded/cancelled is refused")
    if any(check.get("status") in {"failed", "degraded", "cancelled"} for check in inner.get("checks", [])):
        raise ValueError("Preflight contains failed/degraded checks")
    if value.get("channel_config_sha256") != digest or value.get("configured_positions") != list(CHANNEL_NAMES):
        raise ValueError("Preflight channel configuration hash/map mismatch")
    selected = str(value.get("port", ""))
    if not selected or _port(selected) == "AUTO" or _port(inner.get("selected_port", "")) != _port(selected):
        raise ValueError("Preflight must identify the selected real serial port")
    if _port(port) != "AUTO" and _port(port) != _port(selected):
        raise ValueError("Preflight port mismatch")
    timestamp = value.get("completed_timestamp_s")
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
        raise ValueError("Preflight completion timestamp is invalid")
    if not 0 <= now - timestamp <= PREFLIGHT_MAX_AGE_S:
        raise ValueError("Preflight is stale (maximum 300 seconds) or dated in the future")
    return value


@dataclass(frozen=True)
class CaptureHardware:
    """Python-only injection boundary. CLI always loads real BrainFlow Cyton."""

    board_shim: Any
    board_id: int
    params: Any
    prepare_board: Callable[..., tuple[Any, str, list[dict[str, str]]]]

    @classmethod
    def real(cls) -> "CaptureHardware":
        from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
        from eeg_tools.workstation.acquisition_worker import _prepare_cyton_board
        # Reopening after preflight can miss Cyton's first welcome response.
        # Retry the validated endpoint only; AUTO has already been resolved.
        return cls(BoardShim, int(BoardIds.CYTON_BOARD), BrainFlowInputParams(),
                   partial(_prepare_cyton_board, retries=2, retry_delay_s=0.5))


class _Quality:
    def __init__(self, eeg_rows: list[int], package_row: int, timestamp_row: int) -> None:
        self.eeg_rows, self.package_row, self.timestamp_row = eeg_rows, package_row, timestamp_row
        self.sample_count = 0
        self.counts = dict.fromkeys(("packet_loss_count", "packet_duplicate_count", "packet_sequence_mismatch_count",
                                   "invalid_package_count", "nonfinite_count", "timestamp_nonfinite_count",
                                   "timestamp_nonincreasing_count"), 0)
        self.channels = {name: dict.fromkeys(("nonfinite_count", "finite_count", "flat_pair_count", "finite_pair_count", "saturation_count"), 0) for name in CHANNEL_NAMES}
        self.previous_eeg = np.full(8, np.nan)
        self.previous_package: int | None = None
        self.previous_timestamp: float | None = None
        self.no_data_warning_count = 0

    def update(self, raw: np.ndarray) -> None:
        self.sample_count += raw.shape[1]
        self.counts["nonfinite_count"] += int(np.count_nonzero(~np.isfinite(raw)))
        for value in raw[self.package_row]:
            if not np.isfinite(value) or value != int(value) or not 0 <= value < 256:
                self.counts["invalid_package_count"] += 1
                self.previous_package = None
                continue
            current = int(value)
            if self.previous_package is not None:
                delta = (current - self.previous_package) % 256
                if delta != 1:
                    self.counts["packet_sequence_mismatch_count"] += 1
                    if delta == 0:
                        self.counts["packet_duplicate_count"] += 1
                    else:
                        self.counts["packet_loss_count"] += delta - 1
            self.previous_package = current
        for value in raw[self.timestamp_row]:
            if not np.isfinite(value):
                self.counts["timestamp_nonfinite_count"] += 1
                self.previous_timestamp = None
                continue
            if self.previous_timestamp is not None and value <= self.previous_timestamp:
                self.counts["timestamp_nonincreasing_count"] += 1
            self.previous_timestamp = float(value)
        for offset, (name, row) in enumerate(zip(CHANNEL_NAMES, self.eeg_rows)):
            values = raw[row]
            finite = np.isfinite(values)
            counts = self.channels[name]
            counts["finite_count"] += int(finite.sum())
            counts["nonfinite_count"] += int((~finite).sum())
            counts["saturation_count"] += int(np.count_nonzero(finite & (np.abs(values) >= 187000.0)))
            previous = np.concatenate(([self.previous_eeg[offset]], values[:-1]))
            pairs = finite & np.isfinite(previous)
            counts["finite_pair_count"] += int(pairs.sum())
            with np.errstate(invalid="ignore", over="ignore"):
                counts["flat_pair_count"] += int(np.count_nonzero(pairs & (np.abs(values - previous) < 1e-9)))
            self.previous_eeg[offset] = values[-1]

    def report(self, failed: bool = False) -> dict[str, Any]:
        channels = {}
        for name, counts in self.channels.items():
            channels[name] = {**counts,
                "flat_fraction": counts["flat_pair_count"] / max(1, counts["finite_pair_count"]),
                "saturation_fraction": counts["saturation_count"] / max(1, counts["finite_count"])}
        warning = self.sample_count == 0 or any(self.counts.values()) or self.no_data_warning_count or any(
            item["flat_fraction"] >= 0.95 or item["saturation_fraction"] >= 0.95 for item in channels.values())
        return {"status": "failed" if failed else "warning" if warning else "passed",
                "sample_count": self.sample_count, **self.counts, "channels": channels,
                "no_data_warning_count": self.no_data_warning_count,
                "flat_threshold_uv": 1e-9, "saturation_threshold_uv": 187000.0,
                "json_nonfinite_representation": "null"}


def run_capture(
    port: str, channel_config: Path, session_dir: Path, *,
    stop: StopController | None = None,
    emit: Callable[[dict[str, Any]], None] | None = None,
    hardware: CaptureHardware | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    wall_time: Callable[[], float] = time.time,
    wait: Callable[[float], Any] | None = None,
) -> int:
    stop = stop or StopController()
    emit = emit or NDJSONEmitter(sys.stdout, stop)
    wait = wait or stop.event.wait
    session_dir = session_dir.resolve()
    session_dir.mkdir(parents=True, exist_ok=True)
    output_names = ("session.json", "quality.json", "manifest.csv", "raw_brainflow.tsv", "raw_columns.tsv")
    if any((session_dir / name).exists() for name in output_names):
        emit({"type": "error", "code": "session_exists", "message": "Recording output already exists; choose a new session directory"})
        return 2
    # Exclusive reservation also serializes two captures targeting the same directory.
    try:
        with (session_dir / "session.json").open("x", encoding="utf-8") as reservation:
            reservation.write("{}\n")
    except FileExistsError:
        emit({"type": "error", "code": "session_exists", "message": "Session already reserved"})
        return 2
    started = wall_time()
    session: dict[str, Any] = {
        "schema_version": 1, "source": "standalone_realtime_commands", "status": "formal",
        "recording_status": "started", "session_id": session_dir.name,
        "started_at_utc": _utc(started), "started_timestamp_s": started,
        "requested_port": port, "port": "", "channel_names": list(CHANNEL_NAMES),
        "sample_rate_hz": SAMPLE_RATE, "position_verification": "configuration_only",
        "capture_started": False, "cancelled": False, "completed": False,
        "sample_count": 0, "error": "", "error_stage": "", "stop_reason": "", "cleanup_errors": [],
    }
    owned_files = [session_dir / "session.json"]
    board = None
    streaming = False
    stream_attempted = False
    raw_handle = None
    quality: _Quality | None = None
    raw_writer = None
    row_count = 0
    error_message = ""
    preflight: dict[str, Any] | None = None

    def consume(data: Any) -> None:
        received_at = monotonic()
        raw = np.asarray(data, dtype=float)
        if raw.ndim != 2 or raw.shape[0] != row_count:
            raise ValueError(f"Invalid BrainFlow matrix shape {raw.shape}; expected {row_count} rows")
        assert quality is not None
        for begin in range(0, raw.shape[1], SAMPLE_RATE):
            chunk = raw[:, begin:begin + SAMPLE_RATE]
            first = quality.sample_count
            # Write every original BrainFlow row, without timestamp deduplication or padding.
            for offset in range(chunk.shape[1]):
                raw_writer.writerow([first + offset] + [format_raw_number(value) for value in chunk[:, offset]])
            raw_handle.flush()
            quality.update(chunk)
            emit(_json_safe({
                "type": "samples", "channel_names": list(CHANNEL_NAMES), "sample_rate_hz": SAMPLE_RATE,
                "samples": chunk[quality.eeg_rows].tolist(),
                "sample_indices": list(range(first, quality.sample_count)),
                "package_numbers": [int(value) if np.isfinite(value) and value == int(value) and 0 <= value < 256 else None for value in chunk[quality.package_row]],
                "timestamps_s": chunk[quality.timestamp_row].tolist(),
                "received_monotonic_s": received_at, "quality": quality.report(),
            }))

    phase = "configuration_validation"
    try:
        write_json(session_dir / "session.json", session)
        config, digest = load_channel_config(channel_config)
        session["channel_config_sha256"] = digest
        session["configured_positions"] = list(CHANNEL_NAMES)
        session["channel_configuration"] = config
        phase = "preflight_validation"
        preflight = validate_preflight(session_dir, digest, port, wall_time())
        selected_port = preflight["port"]
        session["port"] = selected_port
        if not stop.event.is_set():
            phase = "hardware_metadata"
            hardware = hardware or CaptureHardware.real()
            api, board_id = hardware.board_shim, hardware.board_id
            eeg_rows = list(api.get_eeg_channels(board_id))
            rate = int(api.get_sampling_rate(board_id))
            row_count = int(api.get_num_rows(board_id))
            package_row = int(api.get_package_num_channel(board_id))
            timestamp_row = int(api.get_timestamp_channel(board_id))
            if len(eeg_rows) != 8 or len(set(eeg_rows)) != 8 or rate != SAMPLE_RATE:
                raise ValueError("BrainFlow Cyton must expose exactly eight EEG channels at 250 Hz")
            rows = eeg_rows + [package_row, timestamp_row]
            if len(set(rows)) != 10 or not all(0 <= row < row_count for row in rows):
                raise ValueError("Invalid BrainFlow Cyton row metadata")
            quality = _Quality(eeg_rows, package_row, timestamp_row)
            session["board_id"] = board_id
            definitions = brainflow_column_definitions(board_id, row_count, sampling_rate_hz=rate)
            raw_handle = (session_dir / "raw_brainflow.tsv").open("x", encoding="utf-8", newline="")
            owned_files.append(session_dir / "raw_brainflow.tsv")
            raw_writer = csv.writer(raw_handle, delimiter="\t", lineterminator="\n")
            raw_writer.writerow([item["column_name"] for item in definitions])
            raw_handle.flush()
            write_column_definitions(session_dir / "raw_columns.tsv", definitions)
            owned_files.append(session_dir / "raw_columns.tsv")
            phase = "device_handshake"
            with contextlib.redirect_stdout(sys.stderr):
                board, actual_port, failures = hardware.prepare_board(board_id, hardware.params, selected_port)
            session["handshake_attempt_failures"] = failures
            if _port(actual_port) != _port(selected_port):
                raise ValueError("Handshake selected a port different from preflight")
            if not stop.event.is_set():
                phase = "stream_start"
                stream_attempted = True
                with contextlib.redirect_stdout(sys.stderr):
                    board.start_stream()
                streaming = True
                session["capture_started"] = True
                session["recording_status"] = "recording"
                write_json(session_dir / "session.json", session)
                emit({"type": "capture_ready", "port": actual_port, "requested_port": port,
                      "board_id": board_id, "channel_names": list(CHANNEL_NAMES), "sample_rate_hz": rate,
                      "channel_config_sha256": digest, "session_dir": str(session_dir),
                      "started_at_utc": session["started_at_utc"], "status": "formal", "recording_status": "recording"})
                last_data = monotonic()
                warned = False
                phase = "stream_read"
                while not stop.event.is_set():
                    with contextlib.redirect_stdout(sys.stderr):
                        data = board.get_board_data()
                    consume(data)
                    now = monotonic()
                    if np.shape(data)[1]:
                        last_data, warned = now, False
                    else:
                        silence = now - last_data
                        if silence >= 1.5 and not warned:
                            quality.no_data_warning_count += 1
                            emit({"type": "warning", "code": "no_data", "message": "No Cyton data for 1.5 seconds", "empty_duration_s": silence})
                            warned = True
                        if silence > 5.0:
                            raise RuntimeError("No Cyton data for more than 5 seconds")
                    wait(POLL_S)
    except (Exception, KeyboardInterrupt) as error:
        if isinstance(error, KeyboardInterrupt):
            stop.request("keyboard_interrupt")
        else:
            error_message = f"{type(error).__name__}: {error}"
            session["error_stage"] = phase
            LOG.error("Capture failed at %s (%s): %s", phase, session["port"] or port, error_message)
            emit({"type": "error", "code": "capture_failed", "message": error_message,
                  "stage": phase, "port": session["port"] or port})
    finally:
        if board is not None:
            if stream_attempted:
                try:
                    with contextlib.redirect_stdout(sys.stderr):
                        board.stop_stream()
                except Exception as error:
                    session["cleanup_errors"].append(f"stop_stream: {error}")
                if streaming:
                    try:
                        with contextlib.redirect_stdout(sys.stderr):
                            tail = board.get_board_data()
                        consume(tail)
                    except Exception as error:
                        session["cleanup_errors"].append(f"final_drain: {error}")
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    board.release_session()
            except Exception as error:
                session["cleanup_errors"].append(f"release_session: {error}")
        if raw_handle is not None:
            try:
                raw_handle.close()
            except Exception as error:
                session["cleanup_errors"].append(f"raw_close: {error}")
    if session["cleanup_errors"]:
        error_message = "; ".join(filter(None, [error_message, *session["cleanup_errors"]]))
    reason = stop.reason or ("error" if error_message else "stopped")
    cancelled = not error_message and (reason != "stop_command" or not session["capture_started"])
    recording_status = "failed" if error_message else "cancelled" if cancelled else "completed"
    finished = wall_time()
    final_quality = (quality or _Quality([], 0, 0)).report(bool(error_message))
    if not session["capture_started"] and not error_message:
        final_quality["status"] = "warning"
    session.update(recording_status=recording_status, sample_count=final_quality["sample_count"],
                   stopped_at_utc=_utc(finished), completed_at_utc=_utc(finished),
                   completed_timestamp_s=finished, cancelled=cancelled, completed=recording_status == "completed",
                   stop_reason=reason, error=error_message, quality_status=final_quality["status"])
    try:
        write_json(session_dir / "quality.json", final_quality)
        owned_files.append(session_dir / "quality.json")
        # Hash the immutable raw file after flushing/closing; manifest hashes all saved artifacts.
        session["raw_brainflow_sha256"] = sha256(session_dir / "raw_brainflow.tsv") if raw_handle else None
        write_json(session_dir / "session.json", session)
        if (session_dir / "preflight.json").is_file():
            owned_files.append(session_dir / "preflight.json")
        write_manifest(session_dir / "manifest.csv", session_dir.name, owned_files)
    except Exception as error:
        LOG.exception("Unable to finalize capture artifacts")
        emit({"type": "error", "code": "recording_write_failed", "message": str(error)})
        return 2
    emit({"type": "capture_stopped", "recording_status": recording_status,
          "sample_count": session["sample_count"], "stop_reason": reason,
          "error": error_message, "session_dir": str(session_dir)})
    return 2 if error_message else 0


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)

    def _print_message(self, message: str, file: Any = None) -> None:
        if message:
            sys.stderr.write(message)


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "capture"), required=True)
    parser.add_argument("--port", default="AUTO")
    parser.add_argument("--channel-config", type=Path, required=True)
    parser.add_argument("--session-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    stop = StopController()
    emit = NDJSONEmitter(sys.stdout, stop)
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    try:
        args = build_parser().parse_args(argv)
    except ValueError as error:
        emit({"type": "error", "code": "invalid_arguments", "message": str(error)})
        return 2
    previous_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, lambda sig, _frame: stop.request(f"signal_{sig}"))
    start_stdin_reader(sys.stdin, stop)
    try:
        runner = run_preflight if args.mode == "preflight" else run_capture
        return runner(args.port, args.channel_config.resolve(), args.session_dir.resolve(), stop=stop, emit=emit)
    except (Exception, KeyboardInterrupt) as error:
        LOG.exception("Realtime worker failed")
        emit({"type": "error", "code": "worker_failed", "message": str(error) or type(error).__name__})
        return 2
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
