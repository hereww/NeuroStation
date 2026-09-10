"""Crash-conscious metadata repository for workstation acquisition datasets."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from eeg_tools.session_files import write_manifest

from .ssvep import SSVEPProtocol


@dataclass(frozen=True)
class DatasetRecord:
    session_id: str
    status: str
    participant_id: str
    session_name: str
    output_dir: Path
    duration_s: float
    completed_trials: int
    expected_samples_per_channel: int
    recorded_samples_per_channel: int
    event_count: int
    simulated: bool
    source: str = "demo"


class DatasetRepository:
    def __init__(self, root: Path):
        self.root = root

    @staticmethod
    def default_root() -> Path:
        override = os.environ.get("NEUROSTATION_DATASETS")
        if override:
            return Path(override).expanduser()
        return Path.home() / "Documents" / "NeuroStation" / "Datasets"

    def create_simulated(
        self,
        protocol: SSVEPProtocol,
        *,
        participant_id: str,
        session_name: str,
        status: str = "completed",
        completed_trials: int | None = None,
    ) -> DatasetRecord:
        now = datetime.now().astimezone()
        session_id = now.strftime("session_%Y%m%d_%H%M%S_%f")[:-3]
        output_dir = self.root / session_id
        output_dir.mkdir(parents=True, exist_ok=False)
        completed = protocol.trial_count if completed_trials is None else completed_trials
        event_count = 2 + completed * 2
        record = DatasetRecord(
            session_id=session_id,
            status=status,
            participant_id=participant_id,
            session_name=session_name,
            output_dir=output_dir,
            duration_s=protocol.recording_duration_s,
            completed_trials=completed,
            expected_samples_per_channel=protocol.expected_samples_per_channel,
            recorded_samples_per_channel=0,
            event_count=event_count,
            simulated=True,
            source="demo",
        )

        protocol_path = output_dir / "protocol.json"
        protocol_payload: dict[str, Any] = {
            "protocol_id": protocol.protocol_id,
            "source": str(protocol.source) if protocol.source else None,
            "countdown_s": protocol.countdown_s,
            "recording_duration_s": protocol.recording_duration_s,
            "trial_count": protocol.trial_count,
            "targets": [
                {"id": target_id, "frequency_hz": frequency}
                for target_id, frequency in protocol.targets
            ],
        }
        protocol_path.write_text(
            json.dumps(protocol_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        events_path = output_dir / "events.tsv"
        event_lines = ["event_name\ttrial_index\ttarget_id\tfrequency_hz\tmarker_code"]
        event_lines.append(f"session_start\t-1\t\t\t{protocol.session_start_marker}")
        for trial in protocol.build_trials()[:completed]:
            event_lines.append(
                f"stimulus_onset\t{trial.index}\t{trial.target_id}\t"
                f"{trial.frequency_hz}\t{trial.onset_marker}"
            )
            event_lines.append(
                f"stimulus_offset\t{trial.index}\t{trial.target_id}\t"
                f"{trial.frequency_hz}\t{trial.offset_marker}"
            )
        terminal_name = "session_end" if status == "completed" else "abort"
        terminal_marker = (
            protocol.session_end_marker if status == "completed" else protocol.abort_marker
        )
        event_lines.append(f"{terminal_name}\t-1\t\t\t{terminal_marker}")
        events_path.write_text("\n".join(event_lines) + "\n", encoding="utf-8")

        session_path = output_dir / "session.json"
        session_payload = asdict(record)
        session_payload["output_dir"] = str(output_dir)
        session_payload["created_at"] = now.isoformat(timespec="milliseconds")
        session_payload["notice"] = (
            "UI acceptance simulation: no hardware EEG samples were recorded."
        )
        temporary_path = output_dir / "session.json.pending"
        temporary_path.write_text(
            json.dumps(session_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_path.replace(session_path)

        manifest_path = output_dir / "manifest.csv"
        write_manifest(
            manifest_path, session_id, [session_path, protocol_path, events_path]
        )
        return record

    def list_records(self) -> list[DatasetRecord]:
        if not self.root.exists():
            return []
        records: list[DatasetRecord] = []
        for session_path in self.root.glob("session_*/session.json"):
            try:
                value = json.loads(session_path.read_text(encoding="utf-8"))
                output_dir = Path(value.get("output_dir") or session_path.parent)
                duration = float(
                    value.get("duration_s", value.get("recording_duration_s", 0))
                )
                sampling_rate = int(value.get("sampling_rate_hz", 250))
                expected_duration = float(value.get("expected_duration_s", duration))
                if value.get("source"):
                    source = str(value["source"])
                elif value.get("board") == "synthetic":
                    source = "synthetic"
                elif value.get("board") == "cyton":
                    source = "cyton"
                else:
                    source = "demo" if value.get("simulated", False) else "cyton"
                records.append(
                    DatasetRecord(
                        session_id=value["session_id"],
                        status=value["status"],
                        participant_id=value["participant_id"],
                        session_name=value["session_name"],
                        output_dir=output_dir,
                        duration_s=duration,
                        completed_trials=int(value["completed_trials"]),
                        expected_samples_per_channel=int(
                            value.get(
                                "expected_samples_per_channel",
                                round(expected_duration * sampling_rate),
                            )
                        ),
                        recorded_samples_per_channel=int(
                            value["recorded_samples_per_channel"]
                        ),
                        event_count=int(value["event_count"]),
                        simulated=bool(value["simulated"]),
                        source=source,
                    )
                )
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
        return sorted(records, key=lambda item: item.session_id, reverse=True)
