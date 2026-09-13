"""Crash-conscious metadata repository for workstation acquisition datasets."""

from __future__ import annotations

import json
import os
import hashlib
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from neurostation_contract import (
    OpenBCIImportReport,
    PRODUCT_DESCRIPTION,
    PRODUCT_NAME,
    PRODUCT_SEMVER,
    PRODUCT_VERSION,
    RELEASE_DATE,
)

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
    sampling_rate_hz: int = 250
    channel_count: int = 8
    files: tuple[str, ...] = ()
    source_path: str = ""
    imported: bool = False
    origin: str = "acquired"
    fingerprint: str = ""
    user_id: str = ""
    user_name: str = ""
    user_link_status: str = "unlinked"


@dataclass(frozen=True)
class _RawFileStatistics:
    path: Path
    kind: str
    samples: int
    channel_count: int
    sampling_rate_hz: int


class DatasetRepository:
    def __init__(self, root: Path):
        self.root = root

    @staticmethod
    def default_root() -> Path:
        override = os.environ.get("NEUROSTATION_DATASETS")
        if override:
            return Path(override).expanduser()
        return Path.home() / "Documents" / "NeuroStation" / "Datasets"

    @staticmethod
    def default_openbci_recordings_root() -> Path:
        return Path.home() / "Documents" / "OpenBCI_GUI" / "Recordings"

    def create_simulated(
        self,
        protocol: SSVEPProtocol,
        *,
        participant_id: str,
        session_name: str,
        status: str = "completed",
        completed_trials: int | None = None,
        user_id: str = "",
        user_name: str = "",
        user_link_status: str = "unlinked",
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
            sampling_rate_hz=protocol.sampling_rate_hz,
            channel_count=protocol.channel_count,
            user_id=user_id,
            user_name=user_name,
            user_link_status=user_link_status,
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
        session_payload["application"] = {
            "name": PRODUCT_NAME,
            "version": PRODUCT_VERSION,
            "semantic_version": PRODUCT_SEMVER,
            "release_date": RELEASE_DATE,
            "description": PRODUCT_DESCRIPTION,
        }
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

    def import_openbci_recordings(self, source_root: Path) -> OpenBCIImportReport:
        """Copy OpenBCI GUI recordings into this repository atomically."""

        source_root = Path(source_root).expanduser()
        if not source_root.is_dir():
            return OpenBCIImportReport(
                failed_count=1,
                failures=(f"{source_root}: OpenBCI recordings directory was not found",),
            )

        sessions = self._openbci_session_directories(source_root)
        if not sessions:
            return OpenBCIImportReport()

        imports_root = self.root / "imports"
        imports_root.mkdir(parents=True, exist_ok=True)
        existing_fingerprints = self._import_fingerprints(imports_root)
        imported = skipped = 0
        failures: list[str] = []
        for session_directory in sessions:
            try:
                fingerprint = self._directory_fingerprint(session_directory)
                if fingerprint in existing_fingerprints:
                    skipped += 1
                    continue
                record, raw_files, copied_files = self._describe_openbci_session(
                    session_directory, fingerprint
                )
                destination = self._next_import_destination(
                    imports_root, record.session_id
                )
                record = replace(record, session_id=destination.name, output_dir=destination)
                self._copy_openbci_session(
                    session_directory, destination, record, copied_files, raw_files
                )
                existing_fingerprints.add(fingerprint)
                imported += 1
            except (OSError, ValueError, UnicodeError) as error:
                failures.append(f"{session_directory.name}: {error}")
        return OpenBCIImportReport(
            imported_count=imported,
            skipped_count=skipped,
            failed_count=len(failures),
            failures=tuple(failures),
        )

    def list_records(self) -> list[DatasetRecord]:
        if not self.root.exists():
            return []
        records: list[DatasetRecord] = []
        session_paths = list(self.root.glob("session_*/session.json"))
        session_paths.extend((self.root / "imports").glob("*/session.json"))
        for session_path in session_paths:
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
                imported = bool(
                    value.get("imported", False)
                    or value.get("origin") == "imported_openbci"
                    or source == "imported_openbci"
                )
                records.append(
                    DatasetRecord(
                        session_id=str(value.get("session_id") or session_path.parent.name),
                        status=str(value.get("status") or "completed"),
                        participant_id=str(value.get("participant_id") or ""),
                        session_name=str(value.get("session_name") or session_path.parent.name),
                        output_dir=output_dir,
                        duration_s=duration,
                        completed_trials=int(value.get("completed_trials", 0)),
                        expected_samples_per_channel=int(
                            value.get(
                                "expected_samples_per_channel",
                                round(expected_duration * sampling_rate),
                            )
                        ),
                        recorded_samples_per_channel=int(
                            value.get("recorded_samples_per_channel", 0)
                        ),
                        event_count=int(value.get("event_count", 0)),
                        simulated=bool(value.get("simulated", False)),
                        source=source,
                        sampling_rate_hz=sampling_rate,
                        channel_count=int(value.get("channel_count", 8) or 8),
                        files=DatasetRepository._record_file_names(value),
                        source_path=str(
                            value.get("source_path")
                            or value.get("source_directory")
                            or ""
                        ),
                        imported=imported,
                        origin=str(
                            value.get(
                                "origin",
                                "imported_openbci" if imported else "acquired",
                            )
                        ),
                        fingerprint=str(value.get("fingerprint") or ""),
                        user_id=str(value.get("user_id") or ""),
                        user_name=str(value.get("user_name") or ""),
                        user_link_status=str(value.get("user_link_status") or ("active" if value.get("user_id") else "unlinked")),
                    )
                )
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
        return sorted(records, key=lambda item: item.session_id, reverse=True)

    @staticmethod
    def _openbci_session_directories(source_root: Path) -> list[Path]:
        if source_root.name.startswith("OpenBCISession_"):
            return [source_root]
        return sorted(
            (
                path
                for path in source_root.iterdir()
                if path.is_dir() and path.name.startswith("OpenBCISession_")
            ),
            key=lambda path: path.name,
        )

    @staticmethod
    def _directory_fingerprint(source_directory: Path) -> str:
        digest = hashlib.sha256()
        files = sorted(
            (item for item in source_directory.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(source_directory).as_posix(),
        )
        for path in files:
            stat = path.stat()
            relative = path.relative_to(source_directory).as_posix()
            file_digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    file_digest.update(chunk)
            digest.update(
                f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\0{file_digest.hexdigest()}\n".encode("utf-8")
            )
        return digest.hexdigest()

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _is_brainflow_raw(path: Path) -> bool:
        return bool(re.fullmatch(r"BrainFlow-RAW_.*\.csv", path.name, re.IGNORECASE))

    @staticmethod
    def _is_openbci_raw(path: Path) -> bool:
        return bool(re.fullmatch(r"OpenBCI-RAW-.*\.txt", path.name, re.IGNORECASE))

    def _describe_openbci_session(
        self, source_directory: Path, fingerprint: str
    ) -> tuple[DatasetRecord, tuple[Path, ...], tuple[Path, ...]]:
        copied_files = tuple(
            sorted(
                (item for item in source_directory.rglob("*") if item.is_file()),
                key=lambda item: item.relative_to(source_directory).as_posix(),
            )
        )
        raw_files = tuple(
            path
            for path in copied_files
            if self._is_brainflow_raw(path) or self._is_openbci_raw(path)
        )
        if not raw_files:
            raise ValueError("no BrainFlow-RAW CSV or OpenBCI-RAW TXT file was found")

        statistics: list[_RawFileStatistics] = []
        invalid_files: list[str] = []
        for raw_path in raw_files:
            try:
                current = (
                    self._summarize_brainflow_csv(raw_path)
                    if self._is_brainflow_raw(raw_path)
                    else self._summarize_openbci_txt(raw_path)
                )
            except (OSError, ValueError, UnicodeError) as error:
                invalid_files.append(f"{raw_path.name}: {error}")
                continue
            if current.samples:
                statistics.append(current)
            else:
                invalid_files.append(f"{raw_path.name}: contains no data rows")
        if not statistics:
            detail = "; ".join(invalid_files) or "no data rows"
            raise ValueError(detail)

        # OpenBCI GUI commonly writes the same stream in both formats. Count
        # BrainFlow shards when available, otherwise count the GUI text shards.
        preferred_kind = "brainflow_csv" if any(
            item.kind == "brainflow_csv" for item in statistics
        ) else "openbci_txt"
        selected = [item for item in statistics if item.kind == preferred_kind]
        samples = sum(item.samples for item in selected)
        sampling_rate = next(
            (item.sampling_rate_hz for item in selected if item.sampling_rate_hz > 0),
            250,
        )
        channel_count = next(
            (item.channel_count for item in selected if item.channel_count > 0), 8
        )
        base_id = "imported_openbci_" + re.sub(
            r"[^A-Za-z0-9_.-]+", "_", source_directory.name
        )
        file_names = tuple(
            path.relative_to(source_directory).as_posix() for path in copied_files
        )
        return (
            DatasetRecord(
                session_id=base_id,
                status="completed",
                participant_id="",
                session_name=source_directory.name,
                output_dir=self.root / "imports" / base_id,
                duration_s=samples / sampling_rate if sampling_rate else 0.0,
                completed_trials=0,
                expected_samples_per_channel=samples,
                recorded_samples_per_channel=samples,
                event_count=0,
                simulated=False,
                source="imported_openbci",
                sampling_rate_hz=sampling_rate,
                channel_count=channel_count,
                files=file_names,
                source_path=str(source_directory.resolve()),
                imported=True,
                origin="imported_openbci",
                fingerprint=fingerprint,
                user_link_status="unlinked",
            ),
            raw_files,
            copied_files,
        )

    @staticmethod
    def _summarize_brainflow_csv(path: Path) -> _RawFileStatistics:
        samples = 0
        columns = 0
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            for line in handle:
                value = line.strip()
                if not value or value.startswith("%"):
                    continue
                row = value.split("\t") if "\t" in value else value.split(",")
                try:
                    float(row[0].strip())
                except (IndexError, ValueError):
                    continue
                if len(row) < 2:
                    continue
                samples += 1
                columns = max(columns, len(row))
        channels = 8 if columns >= 9 else max(0, columns - 1)
        return _RawFileStatistics(path, "brainflow_csv", samples, channels, 250)

    @staticmethod
    def _summarize_openbci_txt(path: Path) -> _RawFileStatistics:
        samples = 0
        channels = 8
        sampling_rate = 250
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            for line in handle:
                value = line.strip()
                if not value:
                    continue
                if value.startswith("%"):
                    channel_match = re.search(
                        r"Number of channels\s*=\s*(\d+)", value, re.I
                    )
                    rate_match = re.search(r"Sample Rate\s*=\s*(\d+)", value, re.I)
                    if channel_match:
                        channels = int(channel_match.group(1))
                    if rate_match:
                        sampling_rate = int(rate_match.group(1))
                    continue
                row = value.split(",")
                try:
                    float(row[0].strip())
                except (IndexError, ValueError):
                    continue
                if len(row) >= 2:
                    samples += 1
        if samples == 0:
            raise ValueError("contains no readable data rows")
        return _RawFileStatistics(path, "openbci_txt", samples, channels, sampling_rate)

    @staticmethod
    def _next_import_destination(imports_root: Path, base_id: str) -> Path:
        destination = imports_root / base_id
        suffix = 2
        while destination.exists():
            destination = imports_root / f"{base_id}-{suffix}"
            suffix += 1
        return destination

    def _copy_openbci_session(
        self,
        source_directory: Path,
        destination: Path,
        record: DatasetRecord,
        copied_files: tuple[Path, ...],
        raw_files: tuple[Path, ...],
    ) -> None:
        temporary = Path(
            tempfile.mkdtemp(prefix=".openbci-import-", dir=destination.parent)
        )
        try:
            for source_file in copied_files:
                relative = source_file.relative_to(source_directory)
                target = temporary / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, target)
            existing_metadata = temporary / "session.json"
            if existing_metadata.exists():
                existing_metadata.replace(temporary / "source_session.json")
            saved_record = asdict(record)
            saved_record["output_dir"] = str(destination)
            saved_record["created_at"] = datetime.now().astimezone().isoformat(
                timespec="milliseconds"
            )
            saved_record["source_directory"] = str(source_directory.resolve())
            saved_record["raw_files"] = [
                {
                    "name": path.relative_to(source_directory).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "modified_ns": path.stat().st_mtime_ns,
                    "content_sha256": self._file_sha256(path),
                }
                for path in raw_files
            ]
            saved_record["copied_file_count"] = len(copied_files)
            pending = temporary / "session.json.pending"
            pending.write_text(
                json.dumps(saved_record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            pending.replace(temporary / "session.json")
            temporary.replace(destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    @staticmethod
    def _import_fingerprints(imports_root: Path) -> set[str]:
        fingerprints: set[str] = set()
        for session_path in imports_root.glob("*/session.json"):
            try:
                value = json.loads(session_path.read_text(encoding="utf-8"))
                fingerprint = str(value.get("fingerprint") or "")
                if fingerprint:
                    fingerprints.add(fingerprint)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return fingerprints

    @staticmethod
    def _record_file_names(value: dict[str, Any]) -> tuple[str, ...]:
        raw = value.get("files", value.get("raw_files", ()))
        if not isinstance(raw, (list, tuple)):
            return ()
        result: list[str] = []
        for item in raw:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict) and item.get("name"):
                result.append(str(item["name"]))
        return tuple(result)
