"""Pure EEG windowing, prediction association, and independent command auditing.

Samples are channel-major. Indices and packet counters define continuity;
host arrival times never define sample identity. No model or GUI is imported.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
import time
from typing import Any, Mapping

CHANNEL_NAMES = ("Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2")
SAMPLE_RATE_HZ = 250
WINDOW_SAMPLES = 1250
STRIDE_SAMPLES = 250
CYTON_SATURATION_UV = 187000.0


def _integer(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _finite(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


class CommandWindowBuffer:
    """Bounded five-second buffer; return at most the latest window per batch.

    Required metadata: channel_names, sample_rate_hz, package_numbers, and
    sample_indices OR sample_start (inclusive). Optional timestamps/timestamps_s
    are per sample; host_timestamp/received_monotonic_s describe batch arrival.
    Conflicting aliases are rejected. sample_end is inclusive.
    Invalid batches are discarded atomically. Gaps start a new contiguous
    suffix; duplicate/out-of-order samples are discarded. Every reset advances
    generation, allowing a UI to invalidate an outstanding prediction.
    A flat channel means flat_samples consecutive values within flat_epsilon.
    Any channel at abs(value) >= 187000 microvolts clears the buffer; only
    fresh samples after that saturated point may enter the next window.
    """

    def __init__(self, *, channel_names=CHANNEL_NAMES, flat_samples=250,
                 flat_epsilon=1e-12):
        self.channel_names = tuple(channel_names)
        if (len(self.channel_names) != 8 or len(set(self.channel_names)) != 8
                or not all(isinstance(name, str) and name for name in self.channel_names)):
            raise ValueError("eight unique channel names required")
        if not _integer(flat_samples) or not 2 <= flat_samples <= WINDOW_SAMPLES:
            raise ValueError("flat_samples must be between 2 and 1250")
        if not _finite(flat_epsilon) or flat_epsilon < 0:
            raise ValueError("flat_epsilon must be finite and nonnegative")
        self.flat_samples = int(flat_samples)
        self.flat_epsilon = float(flat_epsilon)
        self._points = deque(maxlen=WINDOW_SAMPLES)
        self.generation = 0
        self.last_reset_reason = "initial"
        self._reset_state()

    @property
    def sample_count(self):
        return len(self._points)

    @property
    def reset_count(self):
        return self.generation

    @property
    def reset_serial(self):
        return self.reset_count

    def _reset_state(self):
        self._points.clear()
        self._last_index = None
        self._last_packet = None
        self._flat_anchors = None
        self._flat_runs = [0] * 8
        self._new_samples = 0
        self._emitted = False
        self._metadata_signature = None

    def clear(self, reason="cleared"):
        self._reset_state()
        self.generation += 1
        self.last_reset_reason = str(reason)

    def _validate(self, message):
        if not isinstance(message, Mapping):
            raise ValueError("invalid_batch")
        if message.get("sample_rate_hz") != SAMPLE_RATE_HZ or isinstance(message.get("sample_rate_hz"), bool):
            raise ValueError("sample_rate_mismatch")
        if tuple(message.get("channel_names", ())) != self.channel_names:
            raise ValueError("channel_order_mismatch")
        samples = message.get("samples")
        if not isinstance(samples, (list, tuple)) or len(samples) != 8:
            raise ValueError("invalid_samples")
        if not all(isinstance(channel, (list, tuple)) for channel in samples):
            raise ValueError("invalid_samples")
        count = len(samples[0])
        if not count or any(len(channel) != count for channel in samples):
            raise ValueError("invalid_samples")
        if not all(_finite(value) for channel in samples for value in channel):
            raise ValueError("nonfinite_samples")
        indices = message.get("sample_indices")
        if indices is None:
            start = message.get("sample_start")
            if not _integer(start) or start < 0:
                raise ValueError("invalid_sample_indices")
            indices = list(range(int(start), int(start) + count))
        if (not isinstance(indices, (list, tuple)) or len(indices) != count
                or not all(_integer(value) and value >= 0 for value in indices)):
            raise ValueError("invalid_sample_indices")
        for field, expected in (("sample_start", indices[0]), ("sample_end", indices[-1])):
            if field in message and (not _integer(message[field]) or message[field] != expected):
                raise ValueError("invalid_sample_indices")
        packets = message.get("package_numbers")
        if (not isinstance(packets, (list, tuple)) or len(packets) != count
                or not all(_integer(value) and 0 <= value < 256 for value in packets)):
            raise ValueError("invalid_package_numbers")
        if ("timestamps" in message and "timestamps_s" in message
                and message["timestamps"] != message["timestamps_s"]):
            raise ValueError("conflicting_timestamps")
        timestamps = message.get("timestamps", message.get("timestamps_s"))
        if timestamps is None:
            timestamps = [None] * count
        elif (not isinstance(timestamps, (list, tuple)) or len(timestamps) != count
              or not all(_finite(value) for value in timestamps)):
            raise ValueError("invalid_timestamps")
        if ("host_timestamp" in message and "received_monotonic_s" in message
                and message["host_timestamp"] != message["received_monotonic_s"]):
            raise ValueError("conflicting_timestamps")
        host = message.get("host_timestamp", message.get("received_monotonic_s"))
        if host is not None and not _finite(host):
            raise ValueError("invalid_timestamps")
        source = message.get("timestamp_source", "unspecified")
        if not isinstance(source, str):
            raise ValueError("invalid_timestamps")
        return samples, indices, packets, timestamps, host, source

    def add_batch(self, message) -> dict | None:
        try:
            samples, indices, packets, timestamps, host, source = self._validate(message)
        except (ValueError, TypeError, OverflowError) as error:
            self.clear(str(error) if isinstance(error, ValueError) else "invalid_batch")
            return None
        signature = (source, any(message.get(key) is not None for key in ("timestamps", "timestamps_s")))
        if self._metadata_signature is not None and signature != self._metadata_signature:
            self.clear("timestamp_metadata_changed")
        self._metadata_signature = signature
        for column, (index, packet, timestamp) in enumerate(zip(indices, packets, timestamps)):
            index, packet = int(index), int(packet)
            if self._last_index is not None:
                if index <= self._last_index or packet == self._last_packet:
                    # Keep the previous high-water mark to reject replay overlap.
                    previous_index, previous_packet = self._last_index, self._last_packet
                    self.clear("duplicate_sample" if index == previous_index or packet == previous_packet
                               else "out_of_order_sample")
                    self._last_index, self._last_packet = previous_index, previous_packet
                    continue
                if index != self._last_index + 1 or packet != (self._last_packet + 1) % 256:
                    self.clear("sample_gap")
            values = tuple(float(samples[channel][column]) for channel in range(8))
            if any(abs(value) >= CYTON_SATURATION_UV for value in values):
                self.clear("saturated_sample")
                self._last_index, self._last_packet = index, packet
                continue
            if self._flat_anchors is None:
                self._flat_anchors = list(values)
                self._flat_runs = [1] * 8
            else:
                for channel, value in enumerate(values):
                    if abs(value - self._flat_anchors[channel]) <= self.flat_epsilon:
                        self._flat_runs[channel] += 1
                    else:
                        self._flat_anchors[channel] = value
                        self._flat_runs[channel] = 1
            self._last_index, self._last_packet = index, packet
            if max(self._flat_runs) >= self.flat_samples:
                self.clear("flat_channel")
                # This bad point is a boundary, never part of a new window.
                self._last_index, self._last_packet = index, packet
                continue
            self._points.append((values, index, packet, timestamp, host))
            self._new_samples += 1
        self._metadata_signature = signature
        if self.sample_count < WINDOW_SAMPLES or (self._emitted and self._new_samples < STRIDE_SAMPLES):
            return None
        self._emitted = True
        self._new_samples = 0
        points = list(self._points)
        return {
            "samples": [[point[0][channel] for point in points] for channel in range(8)],
            "channel_names": list(self.channel_names), "sample_rate_hz": SAMPLE_RATE_HZ,
            "sample_count": WINDOW_SAMPLES, "sample_start": points[0][1],
            "sample_end": points[-1][1], "sample_indices": [point[1] for point in points],
            "package_numbers": [point[2] for point in points],
            "timestamps": [point[3] for point in points], "timestamp_source": source,
            "timestamp_start": points[0][3], "timestamp_end": points[-1][3],
            "host_timestamp_start": points[0][4], "host_timestamp_end": points[-1][4],
        }


class PendingPredictionGate:
    """Associate one pending request with a generation and monotonic deadline.

    begin returns request_id/generation, or None while another request is pending.
    resolve returns an audit envelope; only accepted=True may reach the UI.
    Its raw_prediction is a detached copy of the complete received response.
    No confidence or model-command validation is performed here.
    """

    def __init__(self, timeout_s=2.0, *, clock=time.monotonic):
        if not _finite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")
        self.timeout_s = float(timeout_s)
        self._clock = clock
        self.generation = 0
        self._pending = None
        self._used_request_ids = set()
        self.last_reset_reason = "initial"

    @property
    def pending(self):
        return deepcopy(self._pending)

    def _now(self, now):
        value = self._clock() if now is None else now
        if not _finite(value):
            raise ValueError("now must be finite")
        return float(value)

    def begin(self, request_id, *, now=None):
        if not _integer(request_id) or request_id < 0:
            raise ValueError("request_id must be a nonnegative integer")
        now = self._now(now)
        self.expire(now=now)
        if self._pending is not None:
            return None
        if request_id in self._used_request_ids:
            raise ValueError("request_id already used in this generation")
        self._used_request_ids.add(int(request_id))
        self._pending = {"request_id": int(request_id), "generation": self.generation,
                         "started_at": now, "deadline": now + self.timeout_s}
        return {"request_id": int(request_id), "generation": self.generation}

    def invalidate(self, reason="stopped"):
        self._pending = None
        self.generation += 1
        self._used_request_ids.clear()
        self.last_reset_reason = str(reason)

    def expire(self, *, now=None):
        now = self._now(now)
        if self._pending is not None and now >= self._pending["deadline"]:
            self.invalidate("prediction_timeout")
            return True
        return False

    def resolve(self, prediction, *, now=None):
        expired = self.expire(now=now)
        audit = {"accepted": False, "reason": "stale_prediction",
                 "raw_prediction": deepcopy(prediction)}
        if expired:
            audit["reason"] = "prediction_timeout"
            return audit
        if not isinstance(prediction, Mapping):
            audit["reason"] = "invalid_prediction"
            return audit
        request_id, generation = prediction.get("request_id"), prediction.get("generation")
        if not _integer(request_id) or not _integer(generation):
            audit["reason"] = "invalid_prediction"
            return audit
        if self._pending is None or any(prediction.get(key) != self._pending[key]
                                        for key in ("request_id", "generation")):
            return audit
        self._pending = None
        audit.update(accepted=True, reason="accepted")
        return audit


class CommandJournal:
    """Exclusive NDJSON command logs with an independent, hash-bound manifest.

    Existing command logs are never truncated. append_* accepts JSON mappings;
    serialization with allow_nan=False finishes before any bytes are written.
    close is idempotent and never reads or edits a raw acquisition manifest.
    """

    FILENAMES = {"event": "command_events.ndjson", "command": "commands.ndjson",
                 "frame": "command_frames.ndjson"}

    def __init__(self, session_dir):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.session_dir / "command_manifest.json"
        if self.manifest_path.exists():
            raise FileExistsError(self.manifest_path)
        self._handles = {}
        self._counts = {kind: 0 for kind in self.FILENAMES}
        self._closed = False
        self._manifest = None
        try:
            for kind, filename in self.FILENAMES.items():
                self._handles[kind] = (self.session_dir / filename).open("xb")
        except BaseException:
            for handle in self._handles.values():
                handle.close()
            raise

    def append(self, kind, record):
        if self._closed:
            raise ValueError("journal is closed")
        if kind not in self.FILENAMES:
            raise ValueError("unknown journal kind")
        if not isinstance(record, Mapping):
            raise TypeError("journal records must be mappings")
        data = (json.dumps(dict(record), ensure_ascii=False, allow_nan=False,
                           separators=(",", ":")) + "\n").encode("utf-8")
        handle = self._handles[kind]
        handle.write(data)
        handle.flush()
        self._counts[kind] += 1

    def append_event(self, record):
        self.append("event", record)

    def append_command(self, record):
        self.append("command", record)

    def append_frame(self, record):
        self.append("frame", record)

    def close(self):
        if self._manifest is not None:
            return deepcopy(self._manifest)
        self._closed = True
        for handle in self._handles.values():
            handle.close()
        files = {}
        for kind, filename in self.FILENAMES.items():
            digest = hashlib.sha256()
            path = self.session_dir / filename
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            files[kind] = {"path": filename, "records": self._counts[kind],
                           "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}
        manifest = {"schema_version": 1, "status": "formal", "type": "command_journal",
                    "closed": True, "hash_algorithm": "sha256", "files": files}
        data = (json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2)
                + "\n").encode("utf-8")
        with self.manifest_path.open("xb") as handle:
            handle.write(data)
            handle.flush()
        self._manifest = manifest
        return deepcopy(manifest)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
