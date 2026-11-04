#!/usr/bin/env python3
"""Convert the external MultiFrequency_Mu2024 MAT files for exploration only.

This adapter is deliberately disjoint from NeuroStation's formal configuration,
default desktop runtime, and personal calibration paths.  It mirrors the
trigger dead-time and participant/session cleanup rules in the dataset's
bundled ``dataset_processData.m`` and emits fixed five-second EEG windows.

The command defaults to a safe, non-converting preview.  A conversion requires
an explicit participant and/or session selection plus ``--output-directory``.
The source MAT files are opened read-only and are never modified.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.io import loadmat

FS_HZ = 512
TRIAL_SECONDS = 5
TRIAL_SAMPLES = FS_HZ * TRIAL_SECONDS
BASE_FREQUENCIES_HZ: tuple[int, ...] = (7, 11, 13, 17, 19, 23)
EXPECTED_TRIALS_BY_SESSION: dict[int, int] = {
    1: 24,
    2: 45,
    3: 45,
    4: 45,
    5: 45,
    6: 40,
    7: 40,
    8: 40,
    9: 40,
}

# The values are 1-based raw trigger ordinals, copied from the official MATLAB
# script.  Keeping this table here rather than importing the exploratory script
# makes this converter independently auditable and disjoint.
MATLAB_TRIGGER_FIXES: dict[tuple[int, int], int | tuple[int, ...]] = {
    (1, 2): 45,
    (3, 1): tuple(range(1, 13)) + tuple(range(15, 27)),
    (3, 2): tuple(range(2, 17)) + tuple(range(18, 33)) + tuple(range(35, 50)),
    (3, 3): tuple(range(1, 16)) + tuple(range(17, 47)),
    (3, 6): tuple(range(3, 23)) + tuple(range(25, 45)),
    (14, 4): tuple(range(4, 49)),
    (22, 7): tuple(range(1, 21)) + tuple(range(22, 42)),
    (31, 5): tuple(range(10, 55)),
}


@dataclass(frozen=True)
class TrialRecord:
    participant: int
    session: int
    test_method: str
    stimulation_type: str
    stimulus_index: int
    frequency_tuple: tuple[int, ...]
    onset: int
    raw_trigger_ordinal: int
    corrected_ordinal: int
    correction_rule: str
    source_file: str
    eeg: np.ndarray


def parse_mat_filename(path: Path) -> tuple[int, int]:
    """Parse a strict ``P##_Ses#.mat`` source filename."""
    match = re.fullmatch(r"P(\d+)_Ses([1-9])\.mat", path.name)
    if not match:
        raise ValueError(f"Unexpected MultiFrequency_Mu2024 filename: {path.name}")
    participant = int(match.group(1))
    session = int(match.group(2))
    if not 1 <= participant <= 35:
        raise ValueError(f"Participant must be in 1..35, got {participant}")
    return participant, session


def find_trigger_onsets(trigger: Sequence[float], fs: int = FS_HZ) -> np.ndarray:
    """Return MATLAB ``findOnset1`` onsets as zero-based sample positions.

    The reference function starts with ``lastCall=0`` and uses one-based MATLAB
    sample indices, so the first eligible sample is MATLAB index ``fs``.  A
    detected onset blocks subsequent detections for ``fs-1`` intervening
    samples; an onset exactly ``fs`` samples later is eligible.
    """
    if fs <= 0:
        raise ValueError("fs must be positive")
    onsets: list[int] = []
    last_call_1based = 0
    for matlab_index, value in enumerate(trigger, start=1):
        if matlab_index - last_call_1based < fs:
            continue
        if float(value) >= 1.0:
            onsets.append(matlab_index - 1)
            last_call_1based = matlab_index
    return np.asarray(onsets, dtype=np.int64)


# Compatibility-friendly alias for callers that use the exploratory script's
# terminology while keeping the implementation in this standalone converter.
find_onsets_like_matlab = find_trigger_onsets


def correction_rule_name(participant: int, session: int) -> str:
    selection = MATLAB_TRIGGER_FIXES.get((participant, session))
    if selection is None:
        return "none"
    if isinstance(selection, int):
        return f"P{participant:02d}_Ses{session}_keep_first_{selection}"
    ranges = _format_ordinal_selection(selection)
    return f"P{participant:02d}_Ses{session}_keep_{ranges}"


def _format_ordinal_selection(selection: Sequence[int]) -> str:
    values = list(selection)
    ranges: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def apply_trigger_correction(
    participant: int, session: int, onsets: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, str]:
    """Apply the official participant/session ordinal cleanup.

    Returns ``(corrected_onsets, raw_ordinals, correction_rule)``.  Both
    ordinals are one-based, matching the MATLAB script and making provenance
    unambiguous: ``raw_ordinals`` points back into the dead-time-filtered list,
    while the corrected ordinal is the position in the final trial sequence.
    """
    raw_onsets = np.asarray(onsets, dtype=np.int64)
    selection = MATLAB_TRIGGER_FIXES.get((participant, session))
    rule = correction_rule_name(participant, session)
    if selection is None:
        raw_ordinals = np.arange(1, raw_onsets.size + 1, dtype=np.int64)
    elif isinstance(selection, int):
        if raw_onsets.size < selection:
            raise ValueError(
                f"{rule} requires at least {selection} raw onsets, found {raw_onsets.size}"
            )
        raw_ordinals = np.arange(1, selection + 1, dtype=np.int64)
    else:
        raw_ordinals = np.asarray(selection, dtype=np.int64)
        if raw_ordinals.size == 0 or raw_ordinals.min() < 1 or raw_ordinals.max() > raw_onsets.size:
            raise ValueError(
                f"{rule} references raw onsets 1..{raw_ordinals.max(initial=0)}, "
                f"but only {raw_onsets.size} are available"
            )
    return raw_onsets[raw_ordinals - 1], raw_ordinals, rule


# Existing exploratory callers may use this simpler name/signature.
def apply_bundled_trigger_fix(
    participant: int, session: int, onsets: Sequence[int]
) -> np.ndarray:
    return apply_trigger_correction(participant, session, onsets)[0]


def stimulation_type_for_session(session: int) -> str:
    if session == 1:
        return "single"
    if 2 <= session <= 5:
        return "dual"
    if 6 <= session <= 9:
        return "tri"
    raise ValueError(f"Session must be in 1..9, got {session}")


def target_combinations(session: int) -> list[tuple[int, ...]]:
    """Return lexicographic frequency combinations for a session's order."""
    stimulation_type = stimulation_type_for_session(session)
    order = {"single": 1, "dual": 2, "tri": 3}[stimulation_type]
    return list(itertools.combinations(BASE_FREQUENCIES_HZ, order))


def frequency_tuple_for_stimulus(session: int, stimulus_index: int) -> tuple[int, ...]:
    combinations = target_combinations(session)
    if not 1 <= stimulus_index <= len(combinations):
        raise ValueError(
            f"Session {session} stimulus_index must be in 1..{len(combinations)}, "
            f"got {stimulus_index}"
        )
    return combinations[stimulus_index - 1]


def _official_test_sequence(participant: int) -> tuple[str, ...]:
    """Reproduce ``genTestSequence_MFSSVEP`` as method labels."""
    seq_1f = [1, 1, 1, 1]
    seq_2f_templates = (
        [1, 2, 3, 2, 3, 1, 3, 1, 2, 1, 2, 3],
        [3, 1, 2, 2, 3, 1, 1, 3, 2, 2, 1, 3],
        [3, 2, 1, 1, 3, 2, 3, 2, 1, 2, 1, 3],
        [2, 1, 3, 1, 3, 2, 3, 2, 1, 2, 1, 3],
        [3, 2, 1, 1, 3, 2, 2, 3, 1, 1, 2, 3],
        [3, 1, 2, 2, 3, 1, 3, 1, 2, 1, 2, 3],
        [3, 1, 2, 1, 2, 3, 2, 3, 1, 3, 1, 2],
        [2, 3, 1, 1, 2, 3, 3, 2, 1, 1, 3, 2],
        [2, 1, 3, 3, 2, 1, 2, 1, 3, 1, 3, 2],
    )
    seq_3f_templates = ([31, 32, 32, 31, 32, 31, 31, 32], [32, 31, 31, 32, 31, 32, 32, 31])
    seq_2f = [value + 20 for value in seq_2f_templates[(participant - 1) % 9]]
    seq_3f = seq_3f_templates[(participant - 1) % 2]
    return tuple(f"T{value}" for value in (*seq_1f, *seq_2f, *seq_3f))


def test_methods_for_session(participant: int, session: int) -> tuple[str, ...]:
    sequence = _official_test_sequence(participant)
    if session == 1:
        return sequence[0:4]
    if 2 <= session <= 5:
        start = 4 + 3 * (session - 2)
        return sequence[start : start + 3]
    if 6 <= session <= 9:
        start = 4 + 12 + 2 * (session - 6)
        return sequence[start : start + 2]
    raise ValueError(f"Session must be in 1..9, got {session}")


def _parse_selected_values(values: Iterable[str] | None, *, name: str, maximum: int) -> tuple[int, ...] | None:
    if values is None:
        return None
    parsed: set[int] = set()
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                number = int(token)
            except ValueError as exc:
                raise ValueError(f"Invalid {name}: {token!r}") from exc
            if not 1 <= number <= maximum:
                raise ValueError(f"{name} must be in 1..{maximum}, got {number}")
            parsed.add(number)
    if not parsed:
        raise ValueError(f"At least one {name} is required when the option is supplied")
    return tuple(sorted(parsed))


def select_source_files(
    dataset_root: Path,
    participants: Sequence[int] | None,
    sessions: Sequence[int] | None,
) -> list[Path]:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    selected_participants = set(participants) if participants is not None else None
    selected_sessions = set(sessions) if sessions is not None else None
    files: list[Path] = []
    for path in sorted(dataset_root.glob("P*_Ses*.mat")):
        try:
            participant, session = parse_mat_filename(path)
        except ValueError:
            continue
        if selected_participants is not None and participant not in selected_participants:
            continue
        if selected_sessions is not None and session not in selected_sessions:
            continue
        files.append(path)
    if not files:
        raise FileNotFoundError(
            f"No P##_Ses#.mat files match the requested range under {dataset_root}"
        )
    return files


def _load_session_matrix(path: Path, session: int) -> np.ndarray:
    loaded = loadmat(path, squeeze_me=True, struct_as_record=False)
    variable_name = f"data_ses{session}"
    if variable_name not in loaded:
        raise KeyError(f"{path.name} missing variable {variable_name!r}")
    data = np.asarray(loaded[variable_name])
    if data.ndim != 2 or data.shape[0] < 10:
        raise ValueError(f"{path.name}: expected at least 10 rows, got {data.shape}")
    if data.shape[1] < TRIAL_SAMPLES:
        raise ValueError(f"{path.name}: recording has fewer than {TRIAL_SAMPLES} samples")
    if not np.issubdtype(data.dtype, np.number):
        raise TypeError(f"{path.name}: expected numeric matrix, got {data.dtype}")
    return data


def convert_file(path: Path) -> list[TrialRecord]:
    participant, session = parse_mat_filename(path)
    data = _load_session_matrix(path, session)
    # MATLAB rows 2:7 are zero-based Python rows 1:7: six EEG channels.
    eeg_channels = data[1:7]
    trigger = data[7]  # MATLAB row 8; never use row 10 for labels.
    raw_onsets = find_trigger_onsets(trigger)
    corrected_onsets, raw_ordinals, correction_rule = apply_trigger_correction(
        participant, session, raw_onsets
    )
    combinations = target_combinations(session)
    methods = test_methods_for_session(participant, session)
    trials_per_method = len(combinations)
    stimulation_type = stimulation_type_for_session(session)
    records: list[TrialRecord] = []
    for corrected_zero_based, (onset, raw_ordinal) in enumerate(
        zip(corrected_onsets, raw_ordinals), start=1
    ):
        corrected_ordinal = corrected_zero_based
        stimulus_index = int(round(float(trigger[int(onset)])))
        frequency_tuple = frequency_tuple_for_stimulus(session, stimulus_index)
        method_index = (corrected_ordinal - 1) // trials_per_method
        if method_index >= len(methods):
            raise ValueError(
                f"{path.name}: corrected ordinal {corrected_ordinal} exceeds official "
                f"test sequence for session {session}"
            )
        end = int(onset) + TRIAL_SAMPLES
        if end > data.shape[1]:
            raise ValueError(
                f"{path.name}: trial at onset {int(onset)} lacks a complete "
                f"{TRIAL_SAMPLES}-sample window"
            )
        records.append(
            TrialRecord(
                participant=participant,
                session=session,
                test_method=methods[method_index],
                stimulation_type=stimulation_type,
                stimulus_index=stimulus_index,
                frequency_tuple=frequency_tuple,
                onset=int(onset),
                raw_trigger_ordinal=int(raw_ordinal),
                corrected_ordinal=corrected_ordinal,
                correction_rule=correction_rule,
                source_file=path.name,
                eeg=np.asarray(eeg_channels[:, int(onset) : end]),
            )
        )
    expected = EXPECTED_TRIALS_BY_SESSION[session]
    if len(records) != expected:
        raise ValueError(
            f"{path.name}: official cleanup produced {len(records)} trials, expected {expected}"
        )
    return records


def _matches_frequency_filter(record: TrialRecord, near20_only: bool, include_adjacent_20: bool) -> bool:
    frequencies = set(record.frequency_tuple)
    if near20_only:
        return bool(frequencies.intersection({17, 19, 23}))
    if include_adjacent_20:
        return bool(frequencies.intersection({19, 23}))
    return True


def convert_files(
    files: Sequence[Path], *, near20_only: bool = False, include_adjacent_20: bool = False
) -> list[TrialRecord]:
    records: list[TrialRecord] = []
    for path in files:
        records.extend(
            record
            for record in convert_file(path)
            if _matches_frequency_filter(record, near20_only, include_adjacent_20)
        )
    if not records:
        raise ValueError("Frequency filters selected zero trials")
    return records


def _frequency_string(frequencies: Sequence[int]) -> str:
    return "+".join(str(value) for value in frequencies)


def _records_to_arrays(records: Sequence[TrialRecord]) -> dict[str, np.ndarray]:
    # External MAT files are float64; float32 is sufficient for the exploratory
    # cache and halves disk/memory use without changing the source files.
    eeg = np.stack([record.eeg for record in records], axis=0).astype(np.float32, copy=False)
    max_order = 3
    frequency_hz = np.full((len(records), max_order), -1, dtype=np.int16)
    for index, record in enumerate(records):
        frequency_hz[index, : len(record.frequency_tuple)] = record.frequency_tuple
    return {
        "eeg": eeg,
        "participant": np.asarray([record.participant for record in records], dtype=np.int16),
        "session": np.asarray([record.session for record in records], dtype=np.int8),
        "test_method": np.asarray([record.test_method for record in records], dtype="U4"),
        "stimulation_type": np.asarray([record.stimulation_type for record in records], dtype="U6"),
        "stimulus_index": np.asarray([record.stimulus_index for record in records], dtype=np.int16),
        "frequency_tuple": np.asarray(
            [_frequency_string(record.frequency_tuple) for record in records], dtype="U8"
        ),
        "frequency_hz": frequency_hz,
        "onset": np.asarray([record.onset for record in records], dtype=np.int64),
        "raw_trigger_ordinal": np.asarray(
            [record.raw_trigger_ordinal for record in records], dtype=np.int16
        ),
        "corrected_ordinal": np.asarray(
            [record.corrected_ordinal for record in records], dtype=np.int16
        ),
        "correction_rule": np.asarray(
            [record.correction_rule for record in records], dtype="U128"
        ),
        "source_file": np.asarray([record.source_file for record in records], dtype="U32"),
        "sampling_rate_hz": np.asarray(FS_HZ, dtype=np.int16),
        "trial_samples": np.asarray(TRIAL_SAMPLES, dtype=np.int32),
        "status": np.asarray("formal", dtype="U6"),
        "exploratory_only": np.asarray(True, dtype=np.bool_),
        "runtime_enabled": np.asarray(False, dtype=np.bool_),
        "default_deployment_authorized": np.asarray(False, dtype=np.bool_),
    }


def _record_manifest(record: TrialRecord) -> dict[str, object]:
    return {
        "participant": record.participant,
        "session": record.session,
        "test_method": record.test_method,
        "stimulation_type": record.stimulation_type,
        "stimulus_index": record.stimulus_index,
        "frequency_tuple": list(record.frequency_tuple),
        "onset": record.onset,
        "raw_trigger_ordinal": record.raw_trigger_ordinal,
        "corrected_ordinal": record.corrected_ordinal,
        "correction_rule": record.correction_rule,
        "source_file": record.source_file,
    }


def write_conversion(
    records: Sequence[TrialRecord],
    *,
    output_directory: Path,
    dataset_root: Path,
    participants: Sequence[int] | None,
    sessions: Sequence[int] | None,
    near20_only: bool,
    include_adjacent_20: bool,
) -> tuple[Path, Path]:
    """Write a new NPZ + manifest pair without ever overwriting either file."""
    output_directory = output_directory.expanduser().resolve()
    npz_path = output_directory / "multifrequency_mu2024_exploratory.npz"
    manifest_path = output_directory / "manifest.json"
    if npz_path.exists() or manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {npz_path if npz_path.exists() else manifest_path}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    arrays = _records_to_arrays(records)
    np.savez_compressed(npz_path, **arrays)
    manifest = {
        "status": "formal",
        "exploratory_only": True,
        "runtime_enabled": False,
        "default_deployment_authorized": False,
        "personal_gaze_accuracy_claim": False,
        "purpose": "external dataset exploration and pretraining only",
        "label_scope": "dataset stimulus/frequency labels only; not personal calibration truth",
        "converter": "scripts/convert_multifrequency_mu2024.py",
        "source_root": str(dataset_root.expanduser().resolve()),
        "sampling_rate_hz": FS_HZ,
        "trial_seconds": TRIAL_SECONDS,
        "trial_samples": TRIAL_SAMPLES,
        "eeg_rows": "MATLAB rows 2:7 (six channels)",
        "trigger_row": "MATLAB row 8 (stimulus index); MATLAB row 10 is not used as ground truth",
        "base_frequencies_hz": list(BASE_FREQUENCIES_HZ),
        "selection": {
            "participants": list(participants) if participants is not None else None,
            "sessions": list(sessions) if sessions is not None else None,
            "near20_only": near20_only,
            "include_adjacent_20": include_adjacent_20,
        },
        "npz_file": npz_path.name,
        "record_count": len(records),
        "fields": [
            "participant",
            "session",
            "test_method",
            "stimulation_type",
            "stimulus_index",
            "frequency_tuple",
            "onset",
            "raw_trigger_ordinal",
            "corrected_ordinal",
            "correction_rule",
            "source_file",
        ],
        "trial_metadata": [_record_manifest(record) for record in records],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return npz_path, manifest_path


def preview(
    dataset_root: Path, files: Sequence[Path], participants: Sequence[int] | None, sessions: Sequence[int] | None
) -> dict[str, object]:
    estimated = sum(EXPECTED_TRIALS_BY_SESSION[parse_mat_filename(path)[1]] for path in files)
    return {
        "mode": "preview",
        "source_root": str(dataset_root.expanduser().resolve()),
        "source_file_count": len(files),
        "source_files": [path.name for path in files],
        "participants": list(participants) if participants is not None else None,
        "sessions": list(sessions) if sessions is not None else None,
        "estimated_corrected_trials_before_frequency_filter": estimated,
        "message": "No MAT arrays were converted or written. Select --participant and/or --session plus --output-directory to convert.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"D:\SSVEP\MultiFrequency_Mu2024"),
        help="Directory containing P##_Ses#.mat files.",
    )
    parser.add_argument(
        "--participant",
        action="append",
        metavar="N[,N...]",
        help="Explicit participant selection; repeat or use comma-separated values (1..35).",
    )
    parser.add_argument(
        "--session",
        action="append",
        metavar="N[,N...]",
        help="Explicit session selection; repeat or use comma-separated values (1..9).",
    )
    parser.add_argument(
        "--near20-only",
        action="store_true",
        help="Keep only combinations containing 17, 19, or 23 Hz.",
    )
    parser.add_argument(
        "--include-adjacent-20",
        action="store_true",
        help="Without --near20-only, keep combinations containing at least 19 or 23 Hz.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="New output directory for multifrequency_mu2024_exploratory.npz and manifest.json.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print a safe file/trial-count preview without loading or writing trial arrays.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        participants = _parse_selected_values(args.participant, name="participant", maximum=35)
        sessions = _parse_selected_values(args.session, name="session", maximum=9)
        root = args.dataset_root.expanduser().resolve()
        files = select_source_files(root, participants, sessions)
        is_preview = args.preview or args.output_directory is None
        if args.output_directory is not None and participants is None and sessions is None:
            raise ValueError(
                "Refusing an unbounded conversion: explicitly select --participant and/or --session, "
                "or omit --output-directory for a safe preview"
            )
        if args.preview and args.output_directory is not None:
            raise ValueError("--preview cannot be combined with --output-directory")
        if is_preview:
            print(json.dumps(preview(root, files, participants, sessions), ensure_ascii=False, indent=2))
            return 0
        records = convert_files(
            files,
            near20_only=args.near20_only,
            include_adjacent_20=args.include_adjacent_20,
        )
        npz_path, manifest_path = write_conversion(
            records,
            output_directory=args.output_directory,
            dataset_root=root,
            participants=participants,
            sessions=sessions,
            near20_only=args.near20_only,
            include_adjacent_20=args.include_adjacent_20,
        )
        print(json.dumps({"npz": str(npz_path), "manifest": str(manifest_path), "record_count": len(records)}, ensure_ascii=False, indent=2))
        return 0
    except (FileExistsError, FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

