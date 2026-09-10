import argparse
import json
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture a short OpenBCI Cyton sample and report signal metrics as JSON."
    )
    parser.add_argument("--port", default="COM5", help="Cyton serial port (default: COM5)")
    parser.add_argument(
        "--seconds", type=float, default=15.0,
        help="Capture duration in seconds (default: 15)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable BrainFlow's development logger"
    )
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be greater than zero")

    try:
        import numpy as np
        from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
    except ModuleNotFoundError as exc:
        parser.error(
            f"Missing dependency '{exc.name}'. Run: python -m pip install -r requirements.txt"
        )

    if args.debug:
        BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    params.serial_port = args.port
    board = BoardShim(BoardIds.CYTON_BOARD, params)
    session_prepared = False
    stream_started = False
    data = np.empty((0, 0))

    try:
        board.prepare_session()
        session_prepared = True
        board.start_stream()
        stream_started = True
        time.sleep(args.seconds)
        data = board.get_board_data()
    except Exception as error:
        # Keep hardware failures machine-readable for the workstation and CI.
        # In particular, a busy serial port should be actionable without asking
        # users to interpret a native BrainFlow traceback.
        failure = {
            "status": "error",
            "port": args.port,
            "requested_seconds": args.seconds,
            "error_type": type(error).__name__,
            "error": str(error),
            "hint": (
                "Close OpenBCI GUI or another serial monitor, verify the port, "
                "and retry."
                if session_prepared is False
                else "The Cyton stream failed after connection; inspect the USB cable and board power."
            ),
        }
        print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stdout)
        return 3
    finally:
        if stream_started:
            try:
                board.stop_stream()
            except Exception:
                pass
        if session_prepared:
            board.release_session()

    eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD)
    timestamp_channel = BoardShim.get_timestamp_channel(BoardIds.CYTON_BOARD)
    sampling_rate = BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD)
    eeg = data[eeg_channels, :]
    timestamps = data[timestamp_channel, :]

    finite_timestamps = timestamps[np.isfinite(timestamps)]
    duration = (
        float(finite_timestamps[-1] - finite_timestamps[0])
        if len(finite_timestamps) > 1
        else 0.0
    )
    timestamp_diffs = np.diff(finite_timestamps)
    positive_diffs = timestamp_diffs[timestamp_diffs > 0]
    gaps = int(np.sum(timestamp_diffs > (1.5 / sampling_rate)))

    channels = []
    for index, values in enumerate(eeg, start=1):
        finite_values = values[np.isfinite(values)]
        differences = np.diff(finite_values)
        channels.append(
            {
                "channel": index,
                "finite_fraction": float(np.mean(np.isfinite(values))),
                "mean": float(np.mean(finite_values)) if len(finite_values) else None,
                "std": float(np.std(finite_values)) if len(finite_values) else None,
                "min": float(np.min(finite_values)) if len(finite_values) else None,
                "max": float(np.max(finite_values)) if len(finite_values) else None,
                "flat_fraction": float(np.mean(np.abs(differences) < 1e-9))
                if len(differences)
                else 1.0,
                "near_previous_saturation_fraction": float(
                    np.mean(finite_values <= -187000.0)
                ) if len(finite_values) else None,
            }
        )

    result = {
        "status": "ok" if data.shape[1] else "no_data",
        "port": args.port,
        "samples": int(data.shape[1]),
        "channels": int(len(eeg_channels)),
        "sampling_rate_hz": int(sampling_rate),
        "duration_s": duration,
        "effective_rate_hz": float((len(timestamps) - 1) / duration)
        if duration > 0
        else 0.0,
        "timestamp_diff_median_s": float(np.median(positive_diffs))
        if len(positive_diffs)
        else None,
        "timestamp_gap_count": gaps,
        "channel_metrics": channels,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if data.shape[1] else 2


if __name__ == "__main__":
    raise SystemExit(main())
