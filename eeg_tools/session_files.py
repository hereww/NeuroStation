"""Writers used by an EEG acquisition session."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


EVENT_FIELDS = [
    "event_id",
    "event_name",
    "eye_side",
    "dataset_name_base",
    "dataset_name",
    "screen_index",
    "screen_name",
    "screen_geometry",
    "screen_mapping",
    "source",
    "label_source",
    "trial_index",
    "target_id",
    "planned_target_id",
    "presented_target_id",
    "gaze_target_id",
    "eeg_predicted_target_id",
    "target_confidence",
    "frequency_hz",
    "marker_code",
    "sample_index",
    "sample_time_s",
    "recording_time_s",
    "wall_time_iso",
    "monotonic_s",
]


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_events(path: Path, events: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(events)


def brainflow_column_definitions(
    board_id: int | None, row_count: int, *, sampling_rate_hz: int = 0
) -> list[dict[str, Any]]:
    """Return stable headers and meanings for BrainFlow's row-oriented matrix.

    BrainFlow exposes a matrix whose rows differ by board.  The acquisition
    file is written sample-wise, so this function is the single source of
    truth for its column order and makes the otherwise opaque row indexes
    inspectable without loading a board-specific schema in the UI.
    """

    definitions: list[dict[str, Any]] = [
        {
            "column_name": "sample_index",
            "brainflow_row": "",
            "semantic": "Sequential sample number in this saved file",
            "unit": "count",
            "role": "index",
        }
    ]
    names = [
        {
            "column_name": f"board_row_{index}",
            "brainflow_row": index,
            "semantic": "Unclassified BrainFlow board row",
            "unit": "board-defined",
            "role": "raw",
        }
        for index in range(max(0, row_count))
    ]

    if board_id is not None:
        try:
            from brainflow.board_shim import BoardShim

            def assign(index: int, name: str, semantic: str, unit: str, role: str) -> None:
                if 0 <= index < len(names):
                    names[index] = {
                        "column_name": name,
                        "brainflow_row": index,
                        "semantic": semantic,
                        "unit": unit,
                        "role": role,
                    }

            for index in ("get_package_num_channel",):
                getter = getattr(BoardShim, index, None)
                if getter is not None:
                    try:
                        assign(int(getter(board_id)), "package_num", "BrainFlow packet number", "count", "quality")
                    except Exception:
                        pass
            getter = getattr(BoardShim, "get_eeg_channels", None)
            if getter is not None:
                try:
                    for channel_number, row in enumerate(getter(board_id), start=1):
                        assign(int(row), f"eeg_ch{channel_number}", f"EEG channel {channel_number}", "board-unit", "eeg")
                except Exception:
                    pass
            getter = getattr(BoardShim, "get_accel_channels", None)
            if getter is not None:
                try:
                    for axis, row in zip(("x", "y", "z"), getter(board_id)):
                        assign(int(row), f"accel_{axis}", f"Accelerometer {axis.upper()}", "board-unit", "aux")
                except Exception:
                    pass
            getter = getattr(BoardShim, "get_analog_channels", None)
            if getter is not None:
                try:
                    for channel_number, row in enumerate(getter(board_id), start=1):
                        assign(int(row), f"analog_ch{channel_number}", f"Analog auxiliary channel {channel_number}", "board-unit", "aux")
                except Exception:
                    pass
            getter = getattr(BoardShim, "get_other_channels", None)
            if getter is not None:
                try:
                    for channel_number, row in enumerate(getter(board_id), start=1):
                        assign(int(row), f"other_ch{channel_number}", f"Other auxiliary channel {channel_number}", "board-unit", "aux")
                except Exception:
                    pass
            for getter_name, name, semantic, unit, role in (
                ("get_timestamp_channel", "timestamp_s", "Device timestamp", "seconds", "time"),
                ("get_marker_channel", "marker", "Event marker inserted into the board stream", "marker-code", "event"),
            ):
                getter = getattr(BoardShim, getter_name, None)
                if getter is not None:
                    try:
                        assign(int(getter(board_id)), name, semantic, unit, role)
                    except Exception:
                        pass
        except Exception:
            # A metadata writer must still work for a session opened on a
            # machine without the optional BrainFlow runtime.
            pass

    definitions.extend(names)
    if sampling_rate_hz:
        for item in definitions:
            item["sampling_rate_hz"] = sampling_rate_hz
    return definitions


def write_brainflow_tsv(
    path: Path, raw_data: Any, board_id: int | None, *, sampling_rate_hz: int = 0
) -> list[dict[str, Any]]:
    """Write a sample-wise BrainFlow matrix with an explicit header row."""

    row_count = int(raw_data.shape[0]) if getattr(raw_data, "ndim", 0) == 2 else 0
    sample_count = int(raw_data.shape[1]) if getattr(raw_data, "ndim", 0) == 2 else 0
    definitions = brainflow_column_definitions(
        board_id, row_count, sampling_rate_hz=sampling_rate_hz
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow([item["column_name"] for item in definitions])
        for sample_index in range(sample_count):
            writer.writerow(
                [sample_index]
                + [f"{float(raw_data[row, sample_index]):.10g}" for row in range(row_count)]
            )
    return definitions


def write_column_definitions(path: Path, definitions: list[dict[str, Any]]) -> None:
    fields = ["column_name", "brainflow_row", "semantic", "unit", "role", "sampling_rate_hz"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(definitions)


def write_manifest(path: Path, session_id: str, files: list[Path]) -> None:
    fields = ["session_id", "file_name", "file_path", "size_bytes", "sha256"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for file_path in files:
            writer.writerow(
                {
                    "session_id": session_id,
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "size_bytes": file_path.stat().st_size,
                    "sha256": sha256(file_path),
                }
            )
