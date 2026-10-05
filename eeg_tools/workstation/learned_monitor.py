"""Background shadow worker for learned EEG quality optimization.

This worker is deliberately separate from the Qt process. It waits for the
acquisition worker to finish, then reads the immutable raw BrainFlow TSV and
writes learned_denoise_v1 outputs under the session's derived directory.
Failure is recorded as a derived status and never changes acquisition quality
or raw files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from eeg_tools.session_files import write_json

TERMINAL_PHASES = {"completed", "aborted", "error"}
MODEL_VERSION = "learned_denoise_v1"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".pending")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state(output: Path, status: str, reason: str = "", **extra: Any) -> dict[str, Any]:
    value = {"schema_version": 1, "status": status, "model_version": MODEL_VERSION, "reason": reason, **extra}
    _atomic_json(output / "status.json", value)
    if status != "completed":
        event = {"event": "shadow_state", **value, "original_quality_unchanged": True}
        with (output / "quality_events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    return value


def _load_raw(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = list(reader.fieldnames or [])
        eeg_fields = sorted(
            (name for name in fields if name.casefold().startswith("eeg_ch")),
            key=lambda name: int(name[6:]) if name[6:].isdigit() else 0,
        )
        if not eeg_fields:
            raise ValueError("raw_brainflow.tsv has no eeg_ch columns")
        samples: list[list[float]] = [[] for _ in eeg_fields]
        indexes: list[int] = []
        timestamps: list[float] = []
        for row in reader:
            try:
                indexes.append(int(float(row.get("sample_index", len(indexes)))))
            except (TypeError, ValueError):
                indexes.append(len(indexes))
            try:
                timestamps.append(float(row.get("timestamp_s", "nan")))
            except (TypeError, ValueError):
                timestamps.append(float("nan"))
            for target, name in zip(samples, eeg_fields):
                try:
                    target.append(float(row.get(name, "nan")))
                except (TypeError, ValueError):
                    target.append(float("nan"))
    return (
        np.asarray(samples, dtype=np.float32),
        np.asarray(indexes, dtype=np.int64),
        np.asarray(timestamps, dtype=np.float64),
        eeg_fields,
    )


def _assess_continuity(raw_path: Path, timestamps: np.ndarray) -> dict[str, Any]:
    """Conservative Cyton packet/gap check; equal batch timestamps are allowed.

    Sequential sample_index is an export counter, not packet-loss evidence.
    Absence of observed gaps does not certify acquisition data integrity.
    """
    with raw_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        has_packets = "package_num" in (reader.fieldnames or [])
        packets = []
        if has_packets:
            for row in reader:
                try:
                    packets.append(float(row.get("package_num", "nan")))
                except (TypeError, ValueError):
                    packets.append(float("nan"))
    packet_values = np.asarray(packets, dtype=float)
    packet_valid = np.isfinite(packet_values) & (packet_values >= 0) & (packet_values < 256)
    packet_valid &= packet_values == np.floor(packet_values)
    pairs = packet_valid[1:] & packet_valid[:-1]
    steps = np.diff(np.where(packet_valid, packet_values, 0)) % 256
    jumps = int(np.count_nonzero(pairs & (steps != 1)))
    time_pairs = np.isfinite(timestamps[1:]) & np.isfinite(timestamps[:-1])
    # Host timestamps may arrive in batches. Only an obvious >1s forward gap
    # is flagged here; repeated/non-strict timestamps are not called loss.
    time_gaps = int(np.count_nonzero(time_pairs & (np.diff(timestamps) > 1.0)))
    fully_assessed = packet_values.size >= 2 and bool(np.all(packet_valid))
    suspect = bool(jumps or time_gaps)
    return {
        "status": "suspected_discontinuity" if suspect else (
            "no_discontinuity_detected" if fully_assessed else "not_assessed"
        ),
        "packet_continuity": "checked" if fully_assessed else "not_assessed",
        "packet_discontinuity_count": jumps,
        "timestamp_large_gap_count": time_gaps,
        "timestamp_gap_threshold_s": 1.0,
        "missing_data_repaired": False,
        "data_integrity_certified": False,
    }


def _infer_signal(model_path: Path, config: dict[str, Any], signal: np.ndarray) -> dict[str, np.ndarray]:
    # Import optional ML dependencies only after acquisition has finished.
    from eeg_tools.learned_denoise import apply_model_to_array, load_model

    bundle = load_model(model_path, config, device="cpu")
    return apply_model_to_array(bundle, signal, np.isfinite(signal))


def process_session(session_dir: Path, config_path: Path, model_path: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    try:
        from eeg_tools.learned_denoise import load_config

        config = load_config(config_path)
        if not bool(config.get("runtime", {}).get("enabled", True)):
            return _state(output, "disabled", "configuration_disabled")
        if str(config.get("inference", {}).get("mode", "shadow")) != "shadow":
            return _state(output, "disabled", "non_shadow_mode_refused")
        if not model_path.is_file():
            return _state(output, "fallback", "model_missing", model_path=str(model_path))
        raw_path = session_dir / "raw_brainflow.tsv"
        if not raw_path.is_file():
            return _state(output, "fallback", "raw_data_missing", raw_path=str(raw_path))
        metadata = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
        session_rate = metadata.get("sampling_rate_hz") if isinstance(metadata, dict) else None
        try:
            rate = float(session_rate)
        except (TypeError, ValueError):
            rate = float("nan")
        if not np.isfinite(rate) or rate <= 0:
            return _state(output, "fallback", "session_sample_rate_unknown")
        if rate != float(config["sample_rate_hz"]):
            return _state(output, "fallback", "sample_rate_mismatch", session_sample_rate_hz=rate)
        signal, indexes, timestamps, channel_names = _load_raw(raw_path)
        if signal.ndim != 2 or signal.shape[1] == 0:
            return _state(output, "fallback", "raw_data_empty")
        if signal.shape[0] != int(config["channel_count"]):
            return _state(output, "fallback", "channel_count_mismatch")
        integrity = _assess_continuity(raw_path, timestamps)
        if integrity["status"] == "suspected_discontinuity":
            return _state(output, "fallback", "suspected_missing_or_duplicate_data", data_integrity_assessment=integrity)
        if float(np.mean(~np.isfinite(signal))) > float(config["inference"].get("max_invalid_fraction", 0.2)):
            return _state(output, "fallback", "excessive_invalid_samples", data_integrity_assessment=integrity)
        result = _infer_signal(model_path, config, signal)
        valid = result["valid_mask"]
        artifact = result["artifact_probability"]
        confidence = result["confidence"]
        replacement = result["replacement_mask"]
        rejected = ~np.isfinite(confidence) | (confidence <= 0)
        artifact_assessed = valid & ~rejected[None, :] & np.isfinite(artifact)
        np.savez_compressed(
            output / "denoised_signal.npz",
            raw=signal,
            denoised=result["denoised"],
            model_denoised=result["model_denoised"],
            artifact_probability=artifact,
            confidence=confidence,
            replacement_mask=replacement,
            valid_mask=valid,
            sample_index=indexes,
            timestamp_s=timestamps,
            channel_names=np.asarray(channel_names),
            sample_rate_hz=np.asarray(int(config["sample_rate_hz"])),
        )
        event = {
            "event": "session_quality_summary",
            "data_integrity_assessment": integrity,
            "original_quality_unchanged": True,
            "status": "completed",
            "sample_count": int(signal.shape[1]),
            "channel_count": int(signal.shape[0]),
            "valid_fraction": float(np.mean(valid)) if valid.size else 0.0,
            "artifact_probability_mean": float(np.mean(artifact[artifact_assessed])) if np.any(artifact_assessed) else None,
            "artifact_probability_max": float(np.max(artifact[artifact_assessed])) if np.any(artifact_assessed) else None,
            "confidence_mean": float(np.nanmean(confidence)) if confidence.size else 0.0,
            "replacement_fraction": float(np.mean(replacement)) if replacement.size else 0.0,
            "inference_rejected_fraction": float(np.mean(rejected)),
            "artifact_estimate_status": "incomplete_or_unknown" if np.any(rejected) or np.any(~valid) else "model_estimate_only",
            "unknown_or_invalid": bool(np.any(~valid) or np.any(rejected)),
        }
        (output / "quality_events.jsonl").write_text(json.dumps(event, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "status": "formal",
            "model_version": MODEL_VERSION,
            "inference_mode": "shadow",
            "model_path": str(model_path),
            "model_sha256": _sha256(model_path),
            "config_path": str(config_path),
            "input_raw": str(raw_path),
            "raw_is_immutable": True,
        }
        write_json(output / "model_manifest.json", manifest)
        return _state(output, "completed", summary=event)
    except Exception as error:  # Optional worker must not affect acquisition.
        return _state(output, "fallback", f"{type(error).__name__}: {error}")


def run(arguments: argparse.Namespace) -> int:
    output_root = Path(arguments.output_root).expanduser().resolve()
    status_path = Path(arguments.status_file).expanduser().resolve()
    # The status file lives in the per-acquisition runtime directory. Never
    # write failure state to a shared output-root/derived directory.
    output = status_path.parent / MODEL_VERSION
    deadline = time.monotonic() + float(arguments.timeout_seconds)
    terminal: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        try:
            value = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and str(value.get("phase") or value.get("status") or "") in TERMINAL_PHASES:
                terminal = value
                break
        except (OSError, ValueError):
            pass
        time.sleep(float(arguments.poll_interval))
    if terminal is None:
        _state(output, "fallback", "acquisition_status_timeout")
        return 0
    result = terminal.get("result") if isinstance(terminal.get("result"), dict) else {}
    raw_output_dir = result.get("output_dir")
    if not isinstance(raw_output_dir, str) or not raw_output_dir.strip():
        _state(output, "fallback", "session_output_missing")
        return 0
    session_dir = Path(raw_output_dir).expanduser().resolve()
    if session_dir == output_root or not session_dir.is_relative_to(output_root):
        _state(output, "fallback", "session_output_outside_root")
        return 0
    if not session_dir.is_dir():
        _state(output, "fallback", "session_output_missing")
        return 0
    process_session(session_dir, Path(arguments.config), Path(arguments.model), session_dir / "derived" / MODEL_VERSION)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--timeout-seconds", type=float, default=21600.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
