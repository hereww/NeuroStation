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
)


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
    source: str = "cyton"
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
    deleted_at: str = ""


@dataclass(frozen=True)
class _RawFileStatistics:
    path: Path
    kind: str
    samples: int
    channel_count: int
    sampling_rate_hz: int


class DatasetRepository:
    def __init__(self, root: Path):
        # Keep the caller's absolute spelling. Windows can expose the same
        # temp directory through long and 8.3 paths; returning one spelling
        # consistently avoids leaking that representation change to callers.
        self.root = self._absolute_path(root)

    @property
    def trash_root(self) -> Path:
        return self.root / "Trash" / "Datasets"

    @staticmethod
    def default_root() -> Path:
        override = os.environ.get("NEUROSTATION_DATASETS")
        if override:
            return Path(override).expanduser()
        return Path.home() / "Documents" / "NeuroStation" / "Datasets"

    @staticmethod
    def default_openbci_recordings_root() -> Path:
        return Path.home() / "Documents" / "OpenBCI_GUI" / "Recordings"

    def create_simulated(self, *args, **kwargs) -> DatasetRecord:
        """Reject the removed simulated-session API."""

        raise RuntimeError("validation.real_hardware_only")

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
        existing_fingerprints.update(self._import_fingerprints(self.trash_root))
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
                    imports_root, record.session_id, self.trash_root
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
        trashed_records = self.list_trashed_records()
        trashed_ids = {record.session_id for record in trashed_records}
        trashed_fingerprints = {
            record.fingerprint
            for record in trashed_records
            if record.fingerprint
        }
        records = [
            record
            for path in self._active_session_paths()
            if (record := self._read_record(path, force_path=path.parent)) is not None
            and not record.deleted_at
            and record.session_id not in trashed_ids
            and (
                not record.fingerprint
                or record.fingerprint not in trashed_fingerprints
            )
        ]
        return sorted(records, key=lambda item: item.session_id, reverse=True)

    def list_trashed_records(self) -> list[DatasetRecord]:
        """Return dataset directories moved to the recoverable trash."""

        if not self.trash_root.is_dir():
            return []
        records = [
            record
            for path in sorted(self.trash_root.glob("*/session.json"))
            if (record := self._read_record(path, force_path=path.parent)) is not None
        ]
        return sorted(
            records, key=lambda item: item.deleted_at or item.session_id, reverse=True
        )

    def delete_record(self, session_id: str) -> DatasetRecord:
        source = self._find_active_session_path(session_id)
        if source is None:
            raise ValueError("validation.dataset_not_found")
        source = self._checked_path(source, self.root)
        target = self.trash_root / source.name
        if target.exists():
            raise RuntimeError("validation.dataset_trash_exists")
        self.trash_root.mkdir(parents=True, exist_ok=True)
        session_path = source / "session.json"
        value = self._read_session_value(session_path)
        now = datetime.now().astimezone().isoformat(timespec="milliseconds")
        value["trash_original_path"] = str(source.resolve())
        value["trash_original_display_path"] = str(source)
        value["output_dir"] = str(target)
        value["deleted_at"] = now
        self._write_session_value(session_path, value)
        try:
            shutil.move(str(source), str(target))
        except OSError:
            # Restore metadata when the move itself fails so a retry does not
            # leave an active record looking like it is already trashed.
            value["output_dir"] = str(source)
            value.pop("trash_original_path", None)
            value.pop("trash_original_display_path", None)
            value.pop("deleted_at", None)
            try:
                self._write_session_value(session_path, value)
            except OSError:
                pass
            raise
        record = self._read_record(target / "session.json", force_path=target)
        if record is None:
            raise RuntimeError("validation.dataset_invalid")
        return record

    def restore_record(self, session_id: str) -> DatasetRecord:
        source = self._find_trashed_session_path(session_id)
        if source is None:
            raise ValueError("validation.dataset_not_found")
        source = self._checked_path(source, self.trash_root)
        value = self._read_session_value(source / "session.json")
        original_value = str(value.get("trash_original_path") or "").strip()
        if not original_value:
            raise RuntimeError("validation.dataset_restore_path")
        original = self._checked_path(Path(original_value), self.root)
        display_value = str(value.get("trash_original_display_path") or "").strip()
        if display_value:
            display_path = self._checked_path(Path(display_value), self.root)
            if display_path.resolve() != original.resolve():
                raise RuntimeError("validation.dataset_restore_path")
            original = display_path
        if original.resolve().is_relative_to(self.trash_root.resolve()):
            raise RuntimeError("validation.dataset_restore_path")
        if original.exists():
            raise RuntimeError("validation.dataset_restore_conflict")
        original.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(original))
        value["output_dir"] = str(original)
        value.pop("trash_original_path", None)
        value.pop("trash_original_display_path", None)
        value.pop("deleted_at", None)
        self._write_session_value(original / "session.json", value)
        record = self._read_record(original / "session.json", force_path=original)
        if record is None:
            raise RuntimeError("validation.dataset_invalid")
        return record

    def purge_record(self, session_id: str) -> None:
        source = self._find_trashed_session_path(session_id)
        if source is None:
            raise ValueError("validation.dataset_not_found")
        source = self._checked_path(source, self.trash_root)
        shutil.rmtree(source)

    def delete_dataset(self, dataset_id: str) -> DatasetRecord:
        return self.delete_record(dataset_id)

    def restore_dataset(self, dataset_id: str) -> DatasetRecord:
        return self.restore_record(dataset_id)

    def purge_dataset(self, dataset_id: str) -> None:
        self.purge_record(dataset_id)

    def _active_session_paths(self) -> list[Path]:
        paths = list(self.root.glob("session_*/session.json"))
        paths.extend((self.root / "imports").glob("*/session.json"))
        return paths

    def _find_active_session_path(self, session_id: str) -> Path | None:
        wanted = str(session_id)
        for path in self._active_session_paths():
            if path.parent.name == wanted:
                return path.parent
            try:
                value = self._read_session_value(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if str(value.get("session_id") or "") == wanted:
                return path.parent
        return None

    def _find_trashed_session_path(self, session_id: str) -> Path | None:
        wanted = str(session_id)
        for path in self.trash_root.glob("*/session.json"):
            if path.parent.name == wanted:
                return path.parent
            try:
                value = self._read_session_value(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if str(value.get("session_id") or "") == wanted:
                return path.parent
        return None

    @staticmethod
    def _checked_path(path: Path, parent: Path) -> Path:
        candidate = DatasetRepository._absolute_path(path)
        resolved = candidate.resolve()
        root = parent.expanduser().resolve()
        if resolved == root or not resolved.is_relative_to(root):
            raise RuntimeError("validation.dataset_path")
        return candidate

    @staticmethod
    def _absolute_path(path: Path) -> Path:
        return Path(path).expanduser().absolute()

    @staticmethod
    def _read_session_value(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("session metadata must be an object")
        return value

    @staticmethod
    def _write_session_value(path: Path, value: dict[str, Any]) -> None:
        pending = path.with_name(path.name + ".pending")
        pending.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        pending.replace(path)

    @classmethod
    def _read_record(
        cls, session_path: Path, *, force_path: Path | None = None
    ) -> DatasetRecord | None:
        try:
            value = cls._read_session_value(session_path)
            output_dir = (
                force_path or Path(value.get("output_dir") or session_path.parent)
            ).expanduser().absolute()
            duration = float(
                value.get("duration_s", value.get("recording_duration_s", 0))
            )
            sampling_rate = int(value.get("sampling_rate_hz", 250))
            expected_duration = float(value.get("expected_duration_s", duration))
            if value.get("source"):
                source = str(value["source"])
            elif value.get("board") == "synthetic" or value.get("simulated", False):
                source = "synthetic"
            elif value.get("board") == "cyton":
                source = "cyton"
            else:
                source = "cyton"
            imported = bool(
                value.get("imported", False)
                or value.get("origin") == "imported_openbci"
                or source == "imported_openbci"
            )
            return DatasetRecord(
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
                files=cls._record_file_names(value),
                source_path=str(
                    value.get("source_path") or value.get("source_directory") or ""
                ),
                imported=imported,
                origin=str(
                    value.get("origin", "imported_openbci" if imported else "acquired")
                ),
                fingerprint=str(value.get("fingerprint") or ""),
                user_id=str(value.get("user_id") or ""),
                user_name=str(value.get("user_name") or ""),
                user_link_status=str(
                    value.get("user_link_status")
                    or ("active" if value.get("user_id") else "unlinked")
                ),
                deleted_at=str(value.get("deleted_at") or ""),
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

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
    def _next_import_destination(
        imports_root: Path, base_id: str, *reserved_roots: Path
    ) -> Path:
        destination = imports_root / base_id
        suffix = 2
        roots = (imports_root, *reserved_roots)
        while any((root / destination.name).exists() for root in roots):
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
