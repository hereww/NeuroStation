#!/usr/bin/env python3
"""Offline SSVEP decoding benchmark for the open MultiFrequency_Mu2024 data.

This is an analysis-only tool. It does not read or modify formal NeuroStation
configuration, does not enable any runtime model, and never uses MATLAB row 10
(the dataset's online decoder output) as a label.

The benchmark is deliberately restricted to Session 1/T1 single-frequency
trials containing 19 Hz or 23 Hz. The converter has already resolved trigger
cleanup and stimulus-frequency labels. We evaluate transparent reference-CCA
argmax decoding with participant-grouped holdout summaries and an offline
command adapter (19 Hz -> CMD_TARGET_19HZ, 23 Hz -> CMD_TARGET_23HZ). The
commands are report-only and are never sent to the workstation or hardware.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import signal

CANDIDATE_FREQUENCIES = (19.0, 23.0)
COMMANDS = {19.0: "CMD_TARGET_19HZ", 23.0: "CMD_TARGET_23HZ"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_dataset(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as z:
        required = {"eeg", "frequency_hz", "participant", "session", "test_method", "stimulation_type", "source_file"}
        missing = sorted(required - set(z.files))
        if missing:
            raise ValueError(f"converted dataset is missing fields: {missing}")
        data = {key: z[key].copy() for key in z.files}
    eeg = np.asarray(data["eeg"])
    if eeg.ndim != 3 or eeg.shape[1] != 6 or eeg.shape[2] < 512:
        raise ValueError(f"expected eeg [trials, 6, samples], got {eeg.shape}")
    frequencies = np.asarray(data["frequency_hz"])
    if frequencies.ndim != 2 or frequencies.shape[0] != eeg.shape[0]:
        raise ValueError("frequency_hz must have one row per trial")
    return data


def _select_single_frequency_trials(data: dict[str, np.ndarray]) -> np.ndarray:
    freq = np.asarray(data["frequency_hz"])
    session = np.asarray(data["session"])
    test_method = np.asarray(data["test_method"])
    stimulation_type = np.asarray(data["stimulation_type"])
    first = freq[:, 0].astype(float)
    padded = freq[:, 1:]
    # The converter stores -1 as padding for frequency tuples; some older
    # NPZ writers also truncated the fixed-width "single" string to "singl".
    single = np.all((padded < 0) | ~np.isfinite(padded), axis=1)
    keep = (
        (session == 1)
        & (test_method == "T1")
        & np.char.startswith(stimulation_type.astype(str), "singl")
        & single
        & np.isin(first, CANDIDATE_FREQUENCIES)
    )
    if not np.any(keep):
        raise ValueError("no Session 1 single-frequency 19/23 Hz trials found")
    return np.flatnonzero(keep)


def _filter_trials(eeg: np.ndarray, fs: float) -> np.ndarray:
    """Zero-phase 5-45 Hz band-pass; analysis-only, no saved transformed data."""
    sos = signal.butter(4, (5.0, 45.0), btype="bandpass", fs=fs, output="sos")
    filtered = np.empty_like(eeg, dtype=np.float64)
    for i in range(eeg.shape[0]):
        filtered[i] = signal.sosfiltfilt(sos, eeg[i], axis=-1)
    return filtered


def _references(frequency: float, harmonics: int, n_samples: int, fs: float) -> np.ndarray:
    t = np.arange(n_samples, dtype=np.float64) / fs
    columns = []
    for harmonic in range(1, harmonics + 1):
        phase = 2.0 * np.pi * frequency * harmonic * t
        columns.extend((np.sin(phase), np.cos(phase)))
    return np.stack(columns, axis=1)


def _cca_score(x: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> float:
    """Return the largest regularized canonical correlation."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - np.mean(x, axis=0, keepdims=True)
    y = y - np.mean(y, axis=0, keepdims=True)
    n = max(1, x.shape[0] - 1)
    sxx = (x.T @ x) / n
    syy = (y.T @ y) / n
    sxy = (x.T @ y) / n
    # Symmetric eigendecomposition gives stable inverse square roots without
    # adding sklearn as a project dependency.
    def inv_sqrt(matrix: np.ndarray) -> np.ndarray:
        vals, vecs = np.linalg.eigh(matrix)
        vals = np.maximum(vals, ridge)
        return (vecs * (1.0 / np.sqrt(vals))) @ vecs.T
    whitened = inv_sqrt(sxx) @ sxy @ inv_sqrt(syy)
    return float(np.clip(np.linalg.svd(whitened, compute_uv=False)[0], 0.0, 1.0))


def _decode_trial(trial: np.ndarray, fs: float, window_seconds: float, harmonics: int) -> tuple[float, dict[float, float]]:
    n = int(round(window_seconds * fs))
    x = trial[:, :n].T
    scores = {f: _cca_score(x, _references(f, harmonics, n, fs)) for f in CANDIDATE_FREQUENCIES}
    prediction = max(scores, key=scores.get)
    return prediction, scores


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = []
    for target in CANDIDATE_FREQUENCIES:
        mask = y_true == target
        recalls.append(float(np.mean(y_pred[mask] == target)) if np.any(mask) else float("nan"))
    return float(np.nanmean(recalls))


def _evaluate_condition(eeg: np.ndarray, labels: np.ndarray, participants: np.ndarray, fs: float, window_seconds: float, harmonics: int) -> dict:
    predictions = []
    scores = []
    for trial in eeg:
        pred, trial_scores = _decode_trial(trial, fs, window_seconds, harmonics)
        predictions.append(pred)
        scores.append(trial_scores)
    y_pred = np.asarray(predictions, dtype=float)
    accuracy = float(np.mean(y_pred == labels))
    balanced = _balanced_accuracy(labels, y_pred)
    per_participant = {}
    for p in sorted(np.unique(participants).tolist()):
        mask = participants == p
        per_participant[str(int(p))] = {
            "n": int(np.sum(mask)),
            "accuracy": float(np.mean(y_pred[mask] == labels[mask])),
            "balanced_accuracy": _balanced_accuracy(labels[mask], y_pred[mask]),
        }
    command_pred = np.asarray([COMMANDS[float(x)] for x in y_pred], dtype=object)
    command_true = np.asarray([COMMANDS[float(x)] for x in labels], dtype=object)
    score_matrix = np.asarray([[s[19.0], s[23.0]] for s in scores], dtype=float)
    top = np.max(score_matrix, axis=1)
    second = np.min(score_matrix, axis=1)
    margin = (top - second) / np.maximum(second, 1e-12)
    gate_rows = {}
    for threshold in (0.0, 0.05, 0.10, 0.20):
        accepted = margin >= threshold
        gate_rows[str(threshold)] = {
            "accepted": int(np.sum(accepted)),
            "coverage": float(np.mean(accepted)),
            "accepted_accuracy": float(np.mean(y_pred[accepted] == labels[accepted])) if np.any(accepted) else None,
            "accepted_commands": int(np.sum(accepted)),
            "unknown_commands": int(np.sum(~accepted)),
        }
    confusion = {
        "true_19_pred_19": int(np.sum((labels == 19) & (y_pred == 19))),
        "true_19_pred_23": int(np.sum((labels == 19) & (y_pred == 23))),
        "true_23_pred_19": int(np.sum((labels == 23) & (y_pred == 19))),
        "true_23_pred_23": int(np.sum((labels == 23) & (y_pred == 23))),
    }
    return {
        "window_seconds": window_seconds,
        "harmonics": harmonics,
        "n_trials": int(labels.size),
        "chance_accuracy": 0.5,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "confusion": confusion,
        "command_mapping": {"19.0": COMMANDS[19.0], "23.0": COMMANDS[23.0]},
        "raw_command_accuracy": float(np.mean(command_pred == command_true)),
        "margin_gate": gate_rows,
        "participant_grouped_holdout": per_participant,
    }


def _build_command_events(
    filtered: np.ndarray,
    selected: np.ndarray,
    labels: np.ndarray,
    participants: np.ndarray,
    fs: float,
    condition: dict,
    margin_threshold: float,
) -> list[dict]:
    """Re-run only the selected condition and emit offline command candidates."""
    events = []
    window_seconds = float(condition["window_seconds"])
    harmonics = int(condition["harmonics"])
    for local_index, trial in enumerate(filtered):
        predicted, scores = _decode_trial(trial, fs, window_seconds, harmonics)
        ordered = sorted(scores.values(), reverse=True)
        margin = (ordered[0] - ordered[1]) / max(ordered[1], 1e-12)
        accepted = bool(margin >= margin_threshold)
        events.append(
            {
                "source_trial_index": int(selected[local_index]),
                "participant": int(participants[local_index]),
                "prediction": {
                    "predicted_frequency_hz": float(predicted),
                    "scores": {str(key): float(value) for key, value in scores.items()},
                    "margin": float(margin),
                    "accepted": accepted,
                    "rejection_reason": None if accepted else "margin_below_threshold",
                    "recorded_label_used_in_forward": False,
                },
                "presented_frequency_proxy_hz": float(labels[local_index]),
                "command": COMMANDS[predicted] if accepted else None,
                "command_emitted": accepted,
                "executed": False,
                "sink": "stdout_jsonl",
            }
        )
    return events


def run(npz_path: Path, output_path: Path, fs: float = 512.0, command_margin_threshold: float = 0.20) -> dict:
    data = _load_dataset(npz_path)
    selected = _select_single_frequency_trials(data)
    eeg = np.asarray(data["eeg"])[selected].astype(np.float64)
    labels = np.asarray(data["frequency_hz"])[selected, 0].astype(float)
    participants = np.asarray(data["participant"])[selected].astype(int)
    if not np.all(np.isfinite(eeg)):
        raise ValueError("selected EEG contains non-finite values")
    filtered = _filter_trials(eeg, fs)
    conditions = []
    for harmonics in (1, 2, 3):
        for window_seconds in (1.0, 2.0, 3.0, 4.0, 5.0):
            conditions.append(_evaluate_condition(filtered, labels, participants, fs, window_seconds, harmonics))
    best = max(conditions, key=lambda x: (x["accuracy"], x["balanced_accuracy"], -x["window_seconds"], -x["harmonics"]))
    command_events = _build_command_events(
        filtered,
        selected,
        labels,
        participants,
        fs,
        best,
        command_margin_threshold,
    )
    report = {
        "status": "complete",
        "analysis_only": True,
        "runtime_enabled": False,
        "default_deployment_authorized": False,
        "formal_configuration_modified": False,
        "model_enabled": False,
        "source": {
            "npz": str(npz_path.resolve()),
            "npz_sha256": _sha256(npz_path),
            "sampling_rate_hz": fs,
            "external_channel_names": ["PO3", "POz", "PO4", "O1", "Oz", "O2"],
            "source_dataset": "MultiFrequency_Mu2024",
            "source_dataset_root": r"D:\SSVEP\MultiFrequency_Mu2024",
            "selected_scope": "Session 1 / T1 / single-frequency / 19 or 23 Hz",
            "label_source": "frequency_hz derived from row-8 stimulus index and dataset session/test mapping",
            "row_10_used_as_label": False,
            "n_selected_trials": int(selected.size),
            "participants": sorted(np.unique(participants).astype(int).tolist()),
            "class_counts": {str(int(f)): int(np.sum(labels == f)) for f in CANDIDATE_FREQUENCIES},
        },
        "method": {
            "preprocessing": "offline zero-phase Butterworth 5-45 Hz, order 4",
            "decoder": "regularized reference CCA argmax over 19/23 Hz",
            "harmonics_tested": [1, 2, 3],
            "windows_seconds_tested": [1, 2, 3, 4, 5],
            "grouping": "participant-grouped summaries; fixed reference-CCA has no participant-fit parameters",
            "command_adapter": "19 Hz -> CMD_TARGET_19HZ; 23 Hz -> CMD_TARGET_23HZ; margin-gated predictions can be UNKNOWN",
        },
        "best_condition": best,
        "conditions": conditions,
        "command_output": {
            "enabled": True,
            "margin_threshold": float(command_margin_threshold),
            "mapping": {"19.0": COMMANDS[19.0], "23.0": COMMANDS[23.0]},
            "accepted_command_count": int(sum(event["command_emitted"] for event in command_events)),
            "unknown_count": int(sum(not event["command_emitted"] for event in command_events)),
            "executed": False,
            "safety": "offline textual JSONL candidates only; no keyboard, shell, workstation, or hardware side effects",
            "events": command_events,
        },
        "interpretation": {
            "label_semantics": "stimulus frequency/presentation label, not gaze truth and not independent intent ground truth",
            "external_domain_gap": [
                "external data are 512 Hz and six EEG channels (PO3, POz, PO4, O1, Oz, O2)",
                "personal NeuroStation data are 250 Hz and eight channels",
                "external single-frequency Session 1 uses 19 Hz/23 Hz, while the personal protocol under discussion uses 20 Hz",
                "external PO3/POz/PO4/O1/Oz/O2 montage is not equivalent by label to the personal CH5-CH8 mapping; only O1/O2 overlap conceptually and hardware placement still needs verification",
            ],
            "condition_selection_note": "The best window/harmonics row is exploratory selection on this benchmark dataset; it is not an independent held-out model-selection claim.",
            "deployment_statement": "This benchmark does not authorize transfer to personal runtime or direct hardware commands.",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {output_path}")
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sampling-rate-hz", type=float, default=512.0)
    p.add_argument("--command-margin-threshold", type=float, default=0.20)
    p.add_argument("--emit-commands", action="store_true", help="print accepted/rejected offline command candidates as JSONL")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(
        args.npz.expanduser().resolve(),
        args.output.expanduser().resolve(),
        args.sampling_rate_hz,
        args.command_margin_threshold,
    )
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve()), "best_condition": report["best_condition"], "command_output": {key: value for key, value in report["command_output"].items() if key != "events"}}, ensure_ascii=False, indent=2))
    if args.emit_commands:
        for event in report["command_output"]["events"]:
            print("COMMAND " + json.dumps(event, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
