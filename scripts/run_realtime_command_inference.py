"""Realtime EEG command inference worker (stdin/stdout NDJSON).

Run: python scripts/run_realtime_command_inference.py [--device cpu|cuda]
The default run is analysis_outputs/wang2016_spectral_mlp_20261015.

Each UTF-8 input line, including its newline, must be <= 512 KiB. A window is
exactly this schema (samples contains eight rows of 1250 finite real numbers):
    {"type":"window","request_id":7,"generation":3,"sample_rate_hz":250,
     "channel_names":["Fp1","Fp2","C3","C4","P7","P8","O1","O2"],
     "samples":[...]}
No target, labels, extra metadata, resampling, cropping or padding is accepted.
Only nonnegative integer association IDs are accepted. A stop request may omit
both IDs, or include either/both:
    {"type":"stop","request_id":8,"generation":3}

A complete synthetic interface example can be generated without loading a model:
    import json
    import numpy as np
    t = np.arange(1250) / 250
    request = {"type": "window", "request_id": 7, "generation": 3,
               "sample_rate_hz": 250,
               "channel_names": ["Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2"],
               "samples": [np.sin(2*np.pi*10*t + c/7).tolist() for c in range(8)]}
    print(json.dumps(request, separators=(",", ":")))
    print(json.dumps({"type": "stop", "request_id": 8, "generation": 3}))
These are protocol fixtures, not personal validation or command accuracy evidence.

Outputs are ready/prediction/error/stopped JSON envelopes. Accepted predictions
are MOVE_LEFT (10 Hz), MOVE_RIGHT (12 Hz), or SELECT (15 Hz); rejected windows
produce UNKNOWN with a reason. raw_prediction/probabilities/ood retain model
and OOD evidence when available. This worker never executes the commands.
stdout is JSON only, diagnostics are bounded on stderr, and load failure exits 2.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
from typing import BinaryIO, TextIO

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eeg_tools.realtime_command_inference import (
    DEFAULT_RUN_DIR, FLAGS, RealtimeCommandInference, unknown_response,
)

MAX_LINE_BYTES = 512 * 1024
MAX_LOG_BYTES = 16 * 1024


class _BoundedLog(io.TextIOBase):
    """One UTF-8 byte budget shared by loading and the entire worker lifetime."""

    def __init__(self, stream: TextIO):
        self.stream = stream
        self.remaining = MAX_LOG_BYTES
        self.truncated = False

    def write(self, text: str) -> int:
        marker = "\n[stderr log limit reached]\n"
        data = text.encode("utf-8", errors="replace")
        if self.truncated:
            return len(text)
        budget = max(0, self.remaining - len(marker.encode("utf-8")))
        if len(data) <= budget:
            self.stream.write(text)
            self.remaining -= len(data)
        else:
            prefix = data[:budget].decode("utf-8", errors="ignore")
            self.stream.write(prefix + marker)
            self.remaining = 0
            self.truncated = True
        return len(text)

    def flush(self) -> None:
        self.stream.flush()


def _emit(output: TextIO, value: dict) -> None:
    output.write(json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n")
    output.flush()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _bad_constant(value):
    raise ValueError("nonfinite_json_constant")


def serve(engine: RealtimeCommandInference, input_stream: BinaryIO, output: TextIO, errors: TextIO) -> int:
    """Reject/drain oversized lines in bounded chunks and continue until stop/EOF."""
    logs = errors if isinstance(errors, _BoundedLog) else _BoundedLog(errors)
    with redirect_stdout(logs), redirect_stderr(logs):
        ready = engine.ready()
    _emit(output, ready)
    while True:
        line = input_stream.readline(MAX_LINE_BYTES + 1)
        if not line:
            return 0
        if len(line) > MAX_LINE_BYTES:
            while line and not line.endswith(b"\n"):
                line = input_stream.readline(MAX_LINE_BYTES + 1)
            _emit(output, unknown_response(reason="line_too_large"))
            continue
        try:
            request = json.loads(line.decode("utf-8"), object_pairs_hook=_unique_object,
                                 parse_constant=_bad_constant)
        except (ValueError, UnicodeError, RecursionError):
            _emit(output, unknown_response(reason="invalid_json"))
            continue
        if isinstance(request, dict) and request.get("type") == "stop":
            if set(request) <= {"type", "request_id", "generation"} and all(
                type(request[key]) is int and request[key] >= 0
                for key in ("request_id", "generation") if key in request
            ):
                response = unknown_response(request, reason="stop_requested")
                _emit(output, {**FLAGS, "type": "stopped", "request_id": response["request_id"],
                               "generation": response["generation"]})
                return 0
            _emit(output, unknown_response(request, reason="invalid_stop_request"))
            continue
        with redirect_stdout(logs), redirect_stderr(logs):
            response = engine.infer(request)
        _emit(output, response)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


def main(argv=None, *, input_stream=None, output=None, errors=None) -> int:
    input_stream = sys.stdin.buffer if input_stream is None else input_stream
    output = sys.stdout if output is None else output
    errors = sys.stderr if errors is None else errors
    logs = _BoundedLog(errors)
    try:
        with redirect_stdout(logs), redirect_stderr(logs):
            args = make_parser().parse_args(argv)
    except SystemExit as exc:
        if exc.code:
            _emit(output, {**FLAGS, "type": "error", "error": "invalid_arguments", "model_loaded": False})
        return int(exc.code)
    try:
        with redirect_stdout(logs), redirect_stderr(logs):
            engine = RealtimeCommandInference(args.run_dir, device=args.device)
    except Exception as exc:
        logs.write(f"Realtime EEG model load failed: {type(exc).__name__}: {exc}\n")
        logs.flush()
        _emit(output, {**FLAGS, "type": "error", "error": "model_load_failed", "model_loaded": False})
        return 2
    return serve(engine, input_stream, output, logs)


if __name__ == "__main__":
    raise SystemExit(main())
