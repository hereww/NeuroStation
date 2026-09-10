"""Writers used by an EEG acquisition session."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


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
    fields = [
        "event_name",
        "trial_index",
        "target_id",
        "frequency_hz",
        "marker_code",
        "wall_time_iso",
        "monotonic_s",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(events)


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
