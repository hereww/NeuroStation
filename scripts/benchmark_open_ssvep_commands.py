"""Benchmark a transparent SSVEP classifier on an exploratory public dataset.

This tool is deliberately separate from the personal Cyton decoder.  It uses
only the public-dataset stimulus frequency labels for offline evaluation and
emits *textual* command candidates to stdout; it never sends keyboard, shell,
or desktop events and never enables a production runtime.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np
from scipy import signal as scipy_signal


FALSE_RUNTIME_FLAGS = {
    "runtime_enabled": False,
    "default_deployment_authorized": False,
    "personal_gaze_accuracy_claim": False,
}
DEFAULT_COMMANDS = {
    7: "MOVE_UP",
    11: "MOVE_DOWN",
    13: "MOVE_LEFT",
    17: "MOVE_RIGHT",
    19: "SELECT",
    23: "BACK",
}


@dataclass(frozen=True)
class Trial:
    index: int
    participant: int
    session: int
    frequency_hz: int
    eeg: np.ndarray  # (channels, samples)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_scalar(value: np.ndarray | object) -> object:
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else value


def load_single_trials(path: Path, *, allowed_frequencies: Sequence[int] | None = None) -> list[Trial]:
    """Load only single-frequency trials, never treating online decoder output as truth."""

    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        required = {"eeg", "participant", "session", "stimulation_type", "frequency_hz"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"NPZ is missing required fields: {sorted(missing)}")
        eeg = np.asarray(data["eeg"])
        participant = np.asarray(data["participant"])
        session = np.asarray(data["session"])
        stimulation_type = np.asarray(data["stimulation_type"])
        frequency_hz = np.asarray(data["frequency_hz"])
        sample_rate = int(np.asarray(_as_scalar(data["sampling_rate_hz"])).item()) if "sampling_rate_hz" in data else 512
        exploratory_only = bool(np.asarray(_as_scalar(data["exploratory_only"])).item()) if "exploratory_only" in data else False
        runtime_enabled = bool(np.asarray(_as_scalar(data["runtime_enabled"])).item()) if "runtime_enabled" in data else False

    if eeg.ndim != 3 or eeg.shape[1] < 1 or eeg.shape[2] < 32:
        raise ValueError(f"expected EEG array (trials, channels, samples), got {eeg.shape}")
    if sample_rate <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    if runtime_enabled:
        raise ValueError("refusing an NPZ marked runtime_enabled=true")
    if not exploratory_only:
        raise ValueError("refusing an NPZ without exploratory_only=true")

    allowed = set(int(value) for value in allowed_frequencies) if allowed_frequencies else None
    trials: list[Trial] = []
    for index in range(eeg.shape[0]):
        kind = str(stimulation_type[index])
        if kind != "single":
            continue
        frequency = int(frequency_hz[index, 0])
        if frequency <= 0 or (allowed is not None and frequency not in allowed):
            continue
        array = np.asarray(eeg[index], dtype=np.float64)
        if not np.isfinite(array).all():
            continue
        trials.append(
            Trial(
                index=int(index),
                participant=int(participant[index]),
                session=int(session[index]),
                frequency_hz=frequency,
                eeg=array,
            )
        )
    if not trials:
        raise ValueError("no finite single-frequency trials matched the requested filters")
    return trials


def _inv_sqrt(matrix: np.ndarray, regularization: float) -> np.ndarray:
    matrix = (matrix + matrix.T) * 0.5
    values, vectors = np.linalg.eigh(matrix + regularization * np.eye(matrix.shape[0]))
    values = np.maximum(values, regularization)
    return (vectors * (1.0 / np.sqrt(values))) @ vectors.T


def cca_score(epoch: np.ndarray, reference: np.ndarray, *, regularization: float = 1e-6) -> float:
    """Return the first regularized canonical correlation for (samples, channels)."""

    x = np.asarray(epoch, dtype=np.float64)
    y = np.asarray(reference, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("CCA inputs must be two 2-D arrays with equal sample counts")
    x = scipy_signal.detrend(x, axis=0, type="linear")
    y = y - np.mean(y, axis=0, keepdims=True)
    x_std = np.std(x, axis=0, keepdims=True)
    x = x / np.maximum(x_std, 1e-9)
    n = max(1, x.shape[0] - 1)
    cxx = (x.T @ x) / n
    cyy = (y.T @ y) / n
    cxy = (x.T @ y) / n
    wx = _inv_sqrt(cxx, regularization)
    wy = _inv_sqrt(cyy, regularization)
    singular_values = np.linalg.svd(wx @ cxy @ wy, compute_uv=False)
    return float(np.clip(singular_values[0], 0.0, 1.0))


def reference_matrix(sample_count: int, sample_rate_hz: int, frequency_hz: int, harmonics: int) -> np.ndarray:
    time = np.arange(sample_count, dtype=np.float64) / float(sample_rate_hz)
    columns: list[np.ndarray] = []
    for harmonic in range(1, harmonics + 1):
        angular = 2.0 * np.pi * float(frequency_hz) * harmonic * time
        columns.extend((np.sin(angular), np.cos(angular)))
    return np.column_stack(columns)


def bandpass(epoch: np.ndarray, sample_rate_hz: int, low_hz: float, high_hz: float) -> np.ndarray:
    if not 0.0 < low_hz < high_hz < sample_rate_hz / 2.0:
        raise ValueError("invalid bandpass limits")
    sos = scipy_signal.butter(4, [low_hz, high_hz], btype="bandpass", fs=sample_rate_hz, output="sos")
    # padlen is intentionally bounded so short diagnostic windows still work.
    padlen = min(max(0, epoch.shape[-1] - 1), 3 * (2 * sos.shape[0] + 1))
    return scipy_signal.sosfiltfilt(sos, epoch, axis=-1, padlen=padlen)


def score_trial(
    trial: Trial,
    target_frequencies: Sequence[int],
    *,
    sample_rate_hz: int,
    window_seconds: float,
    harmonics: int,
    low_hz: float,
    high_hz: float,
) -> dict[str, object]:
    sample_count = int(round(window_seconds * sample_rate_hz))
    if sample_count <= 32 or trial.eeg.shape[-1] < sample_count:
        raise ValueError(f"trial {trial.index} is shorter than requested {window_seconds:g}s window")
    epoch = np.asarray(trial.eeg[:, :sample_count], dtype=np.float64)
    filtered = bandpass(epoch, sample_rate_hz, low_hz, high_hz)
    scores: dict[str, float] = {}
    for frequency in target_frequencies:
        refs = reference_matrix(sample_count, sample_rate_hz, int(frequency), harmonics)
        scores[str(int(frequency))] = cca_score(filtered.T, refs)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], int(item[0])))
    best_id, best_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    return {
        "predicted_frequency_hz": int(best_id),
        "scores": scores,
        "best_score": float(best_score),
        "margin": float(best_score - second_score),
    }


def _group_trials(trials: Iterable[Trial]) -> dict[tuple[int, int], dict[int, list[Trial]]]:
    grouped: dict[tuple[int, int], dict[int, list[Trial]]] = defaultdict(lambda: defaultdict(list))
    for trial in trials:
        grouped[(trial.participant, trial.session)][trial.frequency_hz].append(trial)
    return grouped


def parse_command_map(raw: str | None, frequencies: Sequence[int]) -> dict[int, str]:
    if not raw:
        return {frequency: DEFAULT_COMMANDS.get(frequency, f"TARGET_{frequency}HZ") for frequency in frequencies}
    result: dict[int, str] = {}
    for item in raw.split(","):
        if "=" not in item:
            raise ValueError("--command-map entries must use FREQUENCY=COMMAND")
        frequency_text, command = item.split("=", 1)
        frequency = int(frequency_text.strip())
        command = command.strip()
        if frequency <= 0 or not command or any(char.isspace() for char in command):
            raise ValueError(f"invalid command map entry: {item!r}")
        result[frequency] = command
    missing = [frequency for frequency in frequencies if frequency not in result]
    if missing:
        raise ValueError(f"command map is missing frequencies: {missing}")
    return result


def benchmark(
    source: Path,
    *,
    output: Path,
    frequencies: Sequence[int] | None = None,
    calibration_repetitions: int = 2,
    evaluation_repetitions: int = 2,
    sample_rate_hz: int = 512,
    window_seconds: float = 4.0,
    harmonics: int = 3,
    low_hz: float = 5.0,
    high_hz: float = 45.0,
    score_threshold: float = 0.08,
    margin_threshold: float = 0.005,
    command_map: str | None = None,
    emit_commands: bool = False,
) -> dict[str, object]:
    trials = load_single_trials(source, allowed_frequencies=frequencies)
    found = sorted({trial.frequency_hz for trial in trials})
    targets = sorted(set(int(value) for value in (frequencies or found)))
    if not targets or not set(targets).issubset(set(found)):
        raise ValueError(f"requested frequencies are not fully present; available={found}")
    if calibration_repetitions < 0 or evaluation_repetitions <= 0:
        raise ValueError("calibration_repetitions must be >=0 and evaluation_repetitions must be >0")
    if sample_rate_hz != 512:
        raise ValueError("this public-dataset adapter expects the converted 512 Hz source")
    actions = parse_command_map(command_map, targets)

    groups = _group_trials(trials)
    rows: list[dict[str, object]] = []
    command_rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for (participant, session), by_frequency in sorted(groups.items()):
        missing = [frequency for frequency in targets if len(by_frequency.get(frequency, [])) < calibration_repetitions + evaluation_repetitions]
        if missing:
            skipped.append({"participant": participant, "session": session, "missing_frequencies": missing})
            continue
        for frequency in targets:
            ordered = by_frequency[frequency]
            evaluation = ordered[calibration_repetitions : calibration_repetitions + evaluation_repetitions]
            for trial in evaluation:
                prediction = score_trial(
                    trial,
                    targets,
                    sample_rate_hz=sample_rate_hz,
                    window_seconds=window_seconds,
                    harmonics=harmonics,
                    low_hz=low_hz,
                    high_hz=high_hz,
                )
                accepted = bool(prediction["best_score"] >= score_threshold and prediction["margin"] >= margin_threshold)
                predicted_frequency = int(prediction["predicted_frequency_hz"])
                row = {
                    "participant": participant,
                    "session": session,
                    "source_trial_index": trial.index,
                    "recorded_stimulus_frequency_hz": frequency,
                    "prediction": prediction,
                    "accepted": accepted,
                    "accepted_label_match": bool(accepted and predicted_frequency == frequency),
                    "recorded_label_used_in_forward": False,
                    "command": actions[predicted_frequency] if accepted else None,
                    "command_emitted": bool(emit_commands and accepted),
                }
                rows.append(row)
                if emit_commands and accepted:
                    command_rows.append({
                        "command": actions[predicted_frequency],
                        "frequency_hz": predicted_frequency,
                        "participant": participant,
                        "session": session,
                        "source_trial_index": trial.index,
                        "executed": False,
                        "sink": "stdout_jsonl",
                    })

    accepted_rows = [row for row in rows if row["accepted"]]
    match_rows = [row for row in accepted_rows if row["accepted_label_match"]]
    per_target = {}
    confusion = {str(frequency): {str(predicted): 0 for predicted in targets} for frequency in targets}
    for frequency in targets:
        target_rows = [row for row in rows if row["recorded_stimulus_frequency_hz"] == frequency]
        for row in target_rows:
            predicted = int(row["prediction"]["predicted_frequency_hz"])
            confusion[str(frequency)][str(predicted)] = confusion[str(frequency)].get(str(predicted), 0) + 1
        accepted_target_rows = [row for row in target_rows if row["accepted"]]
        per_target[str(frequency)] = {
            "evaluation_trial_count": len(target_rows),
            "accepted_trial_count": len(accepted_target_rows),
            "accepted_label_match_count": sum(row["accepted_label_match"] for row in accepted_target_rows),
            "all_trial_match_count": sum(int(row["prediction"]["predicted_frequency_hz"]) == frequency for row in target_rows),
        }
    report = {
        "schema_version": 1,
        "status": "formal",
        "completion_status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "source_kind": "public_open_dataset_exploratory",
        "label_semantics": "stimulus frequency label from converted public dataset; not gaze truth",
        "sample_rate_hz": sample_rate_hz,
        "targets_hz": targets,
        "split": {
            "calibration_repetitions_reserved_per_target": calibration_repetitions,
            "evaluation_repetitions_per_target": evaluation_repetitions,
            "participant_session_groups_evaluated": sorted({f"P{row['participant']:02d}_S{row['session']}" for row in rows}),
            "skipped_groups": skipped,
        },
        "decoder": {
            "method": "regularized CCA with sine/cosine references",
            "harmonics": harmonics,
            "bandpass_hz": [low_hz, high_hz],
            "window_seconds": window_seconds,
            "score_threshold": score_threshold,
            "margin_threshold": margin_threshold,
        },
        "summary": {
            "evaluation_trial_count": len(rows),
            "accepted_trial_count": len(accepted_rows),
            "unknown_trial_count": len(rows) - len(accepted_rows),
            "accepted_label_match_count": len(match_rows),
            "accepted_label_match_rate": (len(match_rows) / len(accepted_rows)) if accepted_rows else 0.0,
            "all_trial_match_rate": (sum(int(row["prediction"]["predicted_frequency_hz"]) == row["recorded_stimulus_frequency_hz"] for row in rows) / len(rows)) if rows else 0.0,
            "per_target": per_target,
            "confusion_matrix": confusion,
        },
        "command_output": {
            "enabled": emit_commands,
            "mapping": {str(frequency): actions[frequency] for frequency in targets},
            "executed": False,
            "safety": "textual JSONL candidates only; no keyboard, shell, or desktop side effects",
            "rows": command_rows,
        },
        "trials": rows,
        "limitations": [
            "This is an offline public-dataset diagnostic, not personal calibration.",
            "Public data are not in the Cyton 250 Hz / personal channel domain.",
            "Open-dataset stimulus labels are a presentation proxy and do not establish gaze accuracy.",
            "Unknown/rejected windows must remain no-command in any future runtime adapter.",
        ],
        "exploratory_only": True,
        **FALSE_RUNTIME_FLAGS,
    }
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {output}")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Converted exploratory public-data NPZ")
    parser.add_argument("--output", required=True, type=Path, help="New JSON report path")
    parser.add_argument("--frequencies", default=None, help="Comma-separated target frequencies, e.g. 7,11,13,17,19,23")
    parser.add_argument("--calibration-repetitions", type=int, default=2)
    parser.add_argument("--evaluation-repetitions", type=int, default=2)
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--harmonics", type=int, default=3)
    parser.add_argument("--low-hz", type=float, default=5.0)
    parser.add_argument("--high-hz", type=float, default=45.0)
    parser.add_argument("--score-threshold", type=float, default=0.08)
    parser.add_argument("--margin-threshold", type=float, default=0.005)
    parser.add_argument("--command-map", default=None, help="Comma-separated FREQUENCY=COMMAND entries")
    parser.add_argument("--emit-commands", action="store_true", help="Print accepted textual command JSONL candidates")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        frequencies = None if args.frequencies is None else [int(value) for value in args.frequencies.split(",") if value.strip()]
        report = benchmark(
            args.source,
            output=args.output,
            frequencies=frequencies,
            calibration_repetitions=args.calibration_repetitions,
            evaluation_repetitions=args.evaluation_repetitions,
            window_seconds=args.window_seconds,
            harmonics=args.harmonics,
            low_hz=args.low_hz,
            high_hz=args.high_hz,
            score_threshold=args.score_threshold,
            margin_threshold=args.margin_threshold,
            command_map=args.command_map,
            emit_commands=args.emit_commands,
        )
    except (FileExistsError, FileNotFoundError, ValueError, OSError) as exc:
        print(json.dumps({"status": "formal", "completion_status": "incomplete", "error": str(exc), **FALSE_RUNTIME_FLAGS}, ensure_ascii=False), file=sys.stderr)
        return 1
    summary = report["summary"]
    print(json.dumps({"status": "formal", "completion_status": "complete", "report": str(args.output.resolve()), "summary": summary, **FALSE_RUNTIME_FLAGS}, ensure_ascii=False))
    if args.emit_commands:
        for row in report["command_output"]["rows"]:
            print("COMMAND " + json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
