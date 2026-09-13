"""Cyton preflight checks shared by diagnostics and the desktop gateway."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import time
from typing import Any, Callable


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    status: str
    detail: str
    metrics: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreflightReport:
    status: str
    requested_port: str
    selected_port: str = ""
    checks: tuple[PreflightCheck, ...] = ()
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "requested_port": self.requested_port,
            "selected_port": self.selected_port,
            "checks": [item.as_dict() for item in self.checks],
            "error": self.error,
        }


def run_cyton_preflight(
    requested_port: str = "AUTO",
    seconds: float = 3.0,
    *,
    board_shim: Any | None = None,
    board_ids: Any | None = None,
    params: Any | None = None,
    prepare_board: Callable[..., tuple[Any, str, list[dict[str, str]]]] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> PreflightReport:
    """Open Cyton, collect a short sample, and return actionable checks.

    Dependencies are imported lazily so importing the desktop UI never opens a
    device or requires a native BrainFlow runtime.
    """
    if seconds <= 0:
        raise ValueError("seconds must be greater than zero")
    if board_shim is None or board_ids is None:
        from brainflow.board_shim import BoardIds, BoardShim
        board_ids = BoardIds
        board_shim = BoardShim
    if params is None:
        from brainflow.board_shim import BrainFlowInputParams
        params = BrainFlowInputParams()
    if prepare_board is None:
        from .acquisition_worker import _prepare_cyton_board
        prepare_board = _prepare_cyton_board

    checks: list[PreflightCheck] = []
    board = None
    prepared = False
    streaming = False
    selected_port = ""
    try:
        board, selected_port, failures = prepare_board(
            board_ids.CYTON_BOARD, params, requested_port
        )
        prepared = True
        checks.append(PreflightCheck(
            "handshake", "passed", "Cyton handshake succeeded.",
            {"attempt_failures": failures, "selected_port": selected_port},
        ))
        board.start_stream()
        streaming = True
        sleep_fn(seconds)
        data = board.get_board_data()
        import numpy as np
        eeg_channels = board_shim.get_eeg_channels(board_ids.CYTON_BOARD)
        timestamp_channel = board_shim.get_timestamp_channel(board_ids.CYTON_BOARD)
        sampling_rate = int(board_shim.get_sampling_rate(board_ids.CYTON_BOARD))
        samples = int(data.shape[1]) if getattr(data, "ndim", 0) == 2 else 0
        checks.append(PreflightCheck(
            "sample_count", "passed" if samples > 0 else "failed",
            f"Collected {samples} samples per channel.",
            {"samples_per_channel": samples, "sampling_rate_hz": sampling_rate},
        ))
        if samples == 0:
            return PreflightReport("failed", requested_port, selected_port, tuple(checks), "No EEG samples were returned.")
        timestamps = data[timestamp_channel, :]
        finite_timestamps = timestamps[np.isfinite(timestamps)]
        diffs = np.diff(finite_timestamps)
        gaps = int(np.sum(diffs > (1.5 / sampling_rate))) if len(diffs) else 0
        checks.append(PreflightCheck(
            "timestamps", "passed" if gaps == 0 else "warning",
            "Timestamp stream is continuous." if gaps == 0 else f"Detected {gaps} timestamp gaps.",
            {"timestamp_gap_count": gaps},
        ))
        channel_metrics = []
        quality_warning = False
        for index in eeg_channels:
            values = data[index, :]
            finite = values[np.isfinite(values)]
            differences = np.diff(finite)
            flat_fraction = float(np.mean(np.abs(differences) < 1e-9)) if len(differences) else 1.0
            if flat_fraction >= 0.95:
                quality_warning = True
            channel_metrics.append({"flat_fraction": flat_fraction, "finite_fraction": float(np.mean(np.isfinite(values)))})
        checks.append(PreflightCheck(
            "channels", "warning" if quality_warning else "passed",
            "At least one channel appears mostly flat." if quality_warning else "Channel samples are finite and changing.",
            {"channel_count": len(eeg_channels), "channels": channel_metrics},
        ))
        overall = "warning" if any(item.status == "warning" for item in checks) else "passed"
        return PreflightReport(overall, requested_port, selected_port, tuple(checks))
    except Exception as error:
        checks.append(PreflightCheck("connection", "failed", f"{type(error).__name__}: {error}"))
        return PreflightReport("failed", requested_port, selected_port, tuple(checks), str(error))
    finally:
        if board is not None and streaming:
            try:
                board.stop_stream()
            except Exception:
                pass
        if board is not None and prepared:
            try:
                board.release_session()
            except Exception:
                pass
