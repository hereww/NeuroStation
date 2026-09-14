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
    default_prepare = prepare_board is None
    if default_prepare:
        from .acquisition_worker import _prepare_cyton_board
        prepare_board = _prepare_cyton_board

    checks: list[PreflightCheck] = []
    board = None
    prepared = False
    streaming = False
    selected_port = ""
    try:
        # A Cyton radio link can miss the first welcome window immediately
        # after a previous session is released. The production prepare helper
        # retries each candidate itself; injected test adapters use this small
        # wrapper so transient failures remain covered without coupling the
        # test contract to the production helper's optional arguments.
        prepare_failures: list[dict[str, str]] = []
        failures: list[dict[str, str]] = []
        if default_prepare:
            board, selected_port, failures = prepare_board(
                board_ids.CYTON_BOARD, params, requested_port
            )
        else:
            for attempt in range(3):
                failures = []
                try:
                    board, selected_port, failures = prepare_board(
                        board_ids.CYTON_BOARD, params, requested_port
                    )
                    break
                except Exception as error:
                    prepare_failures.extend(
                        [{**item, "attempt": str(attempt + 1)} for item in failures]
                        if failures
                        else [{
                            "port": str(requested_port),
                            "error": f"{type(error).__name__}: {error}",
                            "attempt": str(attempt + 1),
                        }]
                    )
                    if attempt == 2:
                        raise
                    sleep_fn(0.5)
        prepared = True
        checks.append(PreflightCheck(
            "handshake", "passed", "Cyton handshake succeeded.",
            {
                "attempt_failures": prepare_failures + failures,
                "selected_port": selected_port,
            },
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
        positive_diffs = diffs[diffs > 0]
        gap_threshold_s = 1.5 / sampling_rate
        gaps = int(np.sum(diffs > gap_threshold_s)) if len(diffs) else 0
        gap_ratio = float(gaps / len(diffs)) if len(diffs) else 0.0
        max_gap_s = float(np.max(positive_diffs)) if len(positive_diffs) else 0.0
        non_monotonic = int(np.sum(diffs <= 0)) if len(diffs) else 0
        timestamp_missing = len(finite_timestamps) != len(timestamps)
        if timestamp_missing or non_monotonic:
            timestamp_status = "failed"
        elif gap_ratio >= 0.05 or max_gap_s >= 1.0:
            timestamp_status = "failed"
        elif gap_ratio >= 0.01 or max_gap_s >= 0.5:
            timestamp_status = "degraded"
        elif gaps:
            timestamp_status = "warning"
        else:
            timestamp_status = "passed"

        # OpenBCI GUI tracks Cyton packet loss from the board's sample index,
        # not from host arrival timestamps. BrainFlow exposes that value as
        # package_num; keep it separate so bursty delivery does not get
        # misreported as dropped EEG samples.
        packet_metrics: dict[str, Any] = {
            "packet_sequence_available": False,
            "packet_loss_count": 0,
            "packet_loss_ratio": 0.0,
            "packet_sequence_mismatch_count": 0,
            "packet_duplicate_count": 0,
        }
        try:
            package_channel = int(
                board_shim.get_package_num_channel(board_ids.CYTON_BOARD)
            )
        except (AttributeError, TypeError, ValueError):
            package_channel = -1
        if 0 <= package_channel < data.shape[0]:
            package_values = np.asarray(data[package_channel, :], dtype=float)
            finite_packages = package_values[np.isfinite(package_values)]
            if len(finite_packages) > 1:
                package_numbers = np.rint(finite_packages).astype(int) % 256
                deltas = (package_numbers[1:] - package_numbers[:-1]) % 256
                expected = (package_numbers[:-1] + 1) % 256
                mismatches = package_numbers[1:] != expected
                positive_jumps = deltas[mismatches][deltas[mismatches] > 0]
                lost_count = int(np.sum(positive_jumps - 1)) if len(positive_jumps) else 0
                duplicate_count = int(np.sum(deltas == 0))
                mismatch_count = int(np.sum(mismatches))
                packet_metrics = {
                    "packet_sequence_available": True,
                    "packet_loss_count": lost_count,
                    "packet_loss_ratio": float(
                        lost_count / (lost_count + len(package_numbers))
                    ) if lost_count + len(package_numbers) else 0.0,
                    "packet_sequence_mismatch_count": mismatch_count,
                    "packet_duplicate_count": duplicate_count,
                }
        # BrainFlow timestamps can reflect bursty host delivery even when the
        # Cyton packet/sample index is perfectly continuous. OpenBCI GUI uses
        # that board sequence as its loss signal, so downgrade a timestamp-only
        # degraded result to a warning when no packets are missing, duplicated,
        # or out of order. Keep non-finite/reversed timestamps and extreme
        # failures as hard failures.
        packet_sequence_clean = (
            bool(packet_metrics.get("packet_sequence_available"))
            and int(packet_metrics.get("packet_loss_count", 0) or 0) == 0
            and int(packet_metrics.get("packet_sequence_mismatch_count", 0) or 0) == 0
            and int(packet_metrics.get("packet_duplicate_count", 0) or 0) == 0
        )
        if (
            timestamp_status in {"degraded", "failed"}
            and packet_sequence_clean
            and not timestamp_missing
            and not non_monotonic
            and max_gap_s < 1.0
        ):
            # BrainFlow host timestamps can arrive in bursts even when the
            # Cyton board sequence is continuous. OpenBCI GUI keeps streaming
            # in this case and reports the quality separately. Treat the same
            # bounded, timestamp-only condition as a warning so a transient
            # USB/radio delivery burst cannot block acquisition.
            timestamp_status = "warning"

        timestamp_detail = (
            "Timestamp stream is continuous."
            if timestamp_status == "passed" else
            f"Detected {gaps} timestamp gaps ({gap_ratio:.1%}); "
            f"maximum interval {max_gap_s * 1000:.1f} ms."
        )
        if timestamp_missing:
            timestamp_detail += f" {len(timestamps) - len(finite_timestamps)} non-finite timestamps."
        if non_monotonic:
            timestamp_detail += f" {non_monotonic} non-monotonic intervals."
        checks.append(PreflightCheck(
            "timestamps", timestamp_status,
            timestamp_detail,
            {
                "timestamp_gap_count": gaps,
                "timestamp_gap_ratio": gap_ratio,
                "timestamp_gap_threshold_s": gap_threshold_s,
                "timestamp_diff_max_s": max_gap_s,
                "timestamp_non_monotonic_count": non_monotonic,
                "timestamp_missing": timestamp_missing,
                **packet_metrics,
            },
        ))
        if packet_metrics["packet_sequence_available"]:
            packet_loss_count = int(packet_metrics["packet_loss_count"])
            packet_mismatch_count = int(packet_metrics["packet_sequence_mismatch_count"])
            packet_duplicate_count = int(packet_metrics["packet_duplicate_count"])
            packet_warning = bool(packet_loss_count or packet_mismatch_count or packet_duplicate_count)
            packet_detail = (
                "Packet sequence is continuous."
                if not packet_warning
                else (
                    f"Detected {packet_loss_count} lost samples, "
                    f"{packet_mismatch_count} sequence mismatches, "
                    f"and {packet_duplicate_count} duplicates."
                )
            )
            checks.append(PreflightCheck(
                "packet_sequence",
                "warning" if packet_warning else "passed",
                packet_detail,
                packet_metrics,
            ))
        channel_metrics = []
        quality_warning = False
        warning_channels: list[int] = []
        for channel_number, index in enumerate(eeg_channels, start=1):
            values = data[index, :]
            finite = values[np.isfinite(values)]
            differences = np.diff(finite)
            flat_fraction = float(np.mean(np.abs(differences) < 1e-9)) if len(differences) else 1.0
            saturation_fraction = (
                float(np.mean(finite <= -187000.0)) if len(finite) else 0.0
            )
            if flat_fraction >= 0.95 or saturation_fraction >= 0.95:
                quality_warning = True
                warning_channels.append(channel_number)
            channel_metrics.append({
                "channel": channel_number,
                "flat_fraction": flat_fraction,
                "saturation_fraction": saturation_fraction,
                "finite_fraction": float(np.mean(np.isfinite(values))),
            })
        checks.append(PreflightCheck(
            "channels", "warning" if quality_warning else "passed",
            "At least one channel appears mostly flat or saturated." if quality_warning else "Channel samples are finite and changing.",
            {
                "channel_count": len(eeg_channels),
                "warning_channels": warning_channels,
                "channels": channel_metrics,
            },
        ))
        statuses = {item.status for item in checks}
        overall = (
            "failed" if "failed" in statuses else
            "degraded" if "degraded" in statuses else
            "warning" if "warning" in statuses else
            "passed"
        )
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
