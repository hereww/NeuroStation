"""Composition adapter between the independent desktop UI and workstation core."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from neurostation_contract import (
    CaptureConfig,
    CaptureMode,
    Dataset,
    DeviceInfo,
    OpenBCIImportReport,
    OpenBCIWorkspaceStatus,
    Phase,
    TaskSnapshot,
)

from .dataset import DatasetRecord, DatasetRepository
from .openbci_workspace import OpenBCIWorkspaceError, OpenBCIWorkspaceManager
from .process_gateway import AcquisitionProcessGateway
from .device_discovery import discover_serial_ports
from eeg_tools.config import ConfigError, validate_channel_config
from .users import UserProfile, UserRegistry


ROOT = Path(__file__).resolve().parents[2]


class MetadataGateway:
    """Own users and persisted dataset metadata for the desktop facade.

    This gateway never creates acquisition data. Real Cyton acquisition is
    owned exclusively by ``AcquisitionProcessGateway``.
    """

    def __init__(
        self,
        *,
        protocol_path: Path,
        dataset_root: Path | None = None,
    ) -> None:
        self.protocol_path = protocol_path.resolve()
        self.device = DeviceInfo()
        self.config = CaptureConfig(
            save_directory=(dataset_root or DatasetRepository.default_root()).resolve()
        )
        self.repository = DatasetRepository(Path(self.config.save_directory))
        self.auto_import_default = dataset_root is None
        self.last_import_report = OpenBCIImportReport()
        self._snapshot = TaskSnapshot()
        self._datasets: list[Dataset] = []
        self._trashed_transient: list[Dataset] = []
        self._last_result_id: str | None = None
        self._openbci = OpenBCIWorkspaceManager(ROOT)
        self._users = UserRegistry(Path(self.config.save_directory))
        self._load_existing(Path(self.config.save_directory))

    @property
    def snapshot(self) -> TaskSnapshot:
        return self._snapshot

    @property
    def datasets(self) -> tuple[Dataset, ...]:
        return tuple(self._datasets)

    @property
    def trashed_datasets(self) -> tuple[Dataset, ...]:
        persisted = tuple(
            self._dataset_from_record(record)
            for record in reversed(self.repository.list_trashed_records())
        )
        return persisted + tuple(self._trashed_transient)

    @property
    def users(self) -> tuple[UserProfile, ...]:
        return self._users.users

    @property
    def trashed_users(self) -> tuple[UserProfile, ...]:
        return self._users.trash

    def next_user_id(self) -> str:
        return self._users.next_user_id()

    def add_user(self, profile: UserProfile) -> UserProfile:
        return self._users.add(profile)

    def update_user(self, original_id: str, profile: UserProfile) -> UserProfile:
        return self._users.update(original_id, profile)

    def delete_user(self, user_id: str, *, move_data: bool) -> UserProfile:
        profile = self._users.delete(user_id, move_data=move_data)
        self.refresh_datasets()
        return profile

    def restore_user(self, user_id: str) -> UserProfile:
        profile = self._users.restore(user_id)
        self.refresh_datasets()
        return profile

    def purge_user(self, user_id: str) -> None:
        self._users.purge(user_id)
        self.refresh_datasets()

    def export_public_users(self, path: Path) -> Path:
        return self._users.export_public_snapshot(path)

    @property
    def openbci_status(self) -> OpenBCIWorkspaceStatus:
        state = self._openbci.status()
        return OpenBCIWorkspaceStatus(
            source_ready=state.source_ready,
            overlay_ready=state.overlay_ready,
            executable_ready=state.executable_ready,
            revision=state.revision,
        )

    def launch_openbci_workspace(self, locale: str) -> int:
        self._ensure_available()
        try:
            return self._openbci.launch(
                locale=locale,
                dataset_root=Path(self.config.save_directory),
            )
        except OpenBCIWorkspaceError as error:
            raise RuntimeError(str(error)) from error

    def _ensure_available(self) -> None:
        if self.snapshot.active:
            raise RuntimeError("validation.busy")

    def start_ssvep(self, config: CaptureConfig, speed: float = 1) -> TaskSnapshot:
        raise ValueError("validation.real_hardware_only")

    def cancel(self) -> TaskSnapshot:
        return self._snapshot

    def tick(self) -> TaskSnapshot:
        return self._snapshot

    def read_live_waveform(self, maximum_rows: int = 1000):
        return None

    def _load_existing(self, root: Path) -> None:
        self.repository = DatasetRepository(root)
        for record in reversed(self.repository.list_records()):
            self._remember(self._dataset_from_record(record))

    def import_openbci_recordings(self, source_root: Path) -> OpenBCIImportReport:
        self._ensure_available()
        report = self.repository.import_openbci_recordings(source_root)
        self.last_import_report = report
        self.refresh_datasets()
        return report

    def refresh_datasets(self) -> None:
        transient = [dataset for dataset in self._datasets if not dataset.persisted]
        self._datasets = []
        for record in reversed(self.repository.list_records()):
            self._remember(self._dataset_from_record(record))
        for dataset in transient:
            self._remember(dataset)

    def delete_dataset(self, dataset_id: str) -> Dataset:
        self._ensure_available()
        for index, dataset in enumerate(self._datasets):
            if dataset.id == dataset_id and not dataset.persisted:
                deleted = replace(
                    dataset,
                    deleted_at=datetime.now(timezone.utc).astimezone().isoformat(
                        timespec="milliseconds"
                    ),
                )
                self._trashed_transient.append(deleted)
                self._datasets.pop(index)
                return deleted
        record = self.repository.delete_record(dataset_id)
        self.refresh_datasets()
        return self._dataset_from_record(record)

    def restore_dataset(self, dataset_id: str) -> Dataset:
        self._ensure_available()
        for index, dataset in enumerate(self._trashed_transient):
            if dataset.id == dataset_id:
                restored = replace(dataset, deleted_at="")
                self._datasets.append(restored)
                self._trashed_transient.pop(index)
                return restored
        record = self.repository.restore_record(dataset_id)
        self.refresh_datasets()
        return self._dataset_from_record(record)

    def purge_dataset(self, dataset_id: str) -> None:
        self._ensure_available()
        for index, dataset in enumerate(self._trashed_transient):
            if dataset.id == dataset_id:
                self._trashed_transient.pop(index)
                return
        self.repository.purge_record(dataset_id)
        self.refresh_datasets()

    def _remember(self, dataset: Dataset) -> None:
        if any(existing.id == dataset.id for existing in self._datasets):
            return
        self._datasets.append(dataset)
        self._last_result_id = dataset.id

    @staticmethod
    def _dataset_from_record(
        record: DatasetRecord, *, demo_seconds: float = 0.0
    ) -> Dataset:
        try:
            source = CaptureMode(record.source)
        except ValueError:
            # Older workstation builds used several free-form source labels.
            # Keep those records visible instead of allowing one unknown value
            # to abort the entire dataset refresh.
            source = (
                CaptureMode.IMPORTED_OPENBCI
                if record.imported or record.origin == "imported_openbci"
                else CaptureMode.CYTON
            )
        quality: dict[str, object] = {}
        try:
            loaded_quality = json.loads((record.output_dir / "quality.json").read_text(encoding="utf-8"))
            if isinstance(loaded_quality, dict):
                quality = loaded_quality
        except (OSError, json.JSONDecodeError):
            pass
        channels = quality.get("channels", {})
        flat_channel_count = sum(
            1 for item in channels.values()
            if isinstance(item, dict) and float(item.get("flat_fraction", 0.0) or 0.0) >= 0.95
        ) if isinstance(channels, dict) else 0
        return Dataset(
            id=record.session_id,
            name=record.session_name,
            participant=record.participant_id,
            protocol="ssvep",
            recording_seconds=record.duration_s,
            preparation_seconds=5,
            demo_seconds=demo_seconds,
            trials=record.completed_trials,
            samples_per_channel=record.recorded_samples_per_channel,
            event_count=record.event_count,
            path=record.output_dir.resolve(),
            created_at=record.session_id,
            simulated=record.simulated,
            persisted=True,
            source=source,
            status=record.status,
            channel_count=record.channel_count,
            sampling_rate_hz=record.sampling_rate_hz,
            files=record.files,
            source_path=record.source_path,
            imported=record.imported,
            origin=record.origin,
            user_id=record.user_id,
            user_name=record.user_name,
            user_link_status=record.user_link_status,
            quality_status=str(quality.get("status") or "unknown"),
            timestamp_gap_count=int(quality.get("timestamp_gap_count", 0) or 0),
            dropped_frame_count=int(quality.get("dropped_frame_count", 0) or 0),
            flat_channel_count=flat_channel_count,
            deleted_at=record.deleted_at,
        )


class DesktopGateway:
    """Desktop facade for real OpenBCI Cyton acquisition and datasets."""

    def __init__(
        self,
        *,
        protocol_path: Path,
        channel_config_path: Path,
        dataset_root: Path | None = None,
    ) -> None:
        self._metadata = MetadataGateway(
            protocol_path=protocol_path,
            dataset_root=dataset_root,
        )
        self._acquisition = AcquisitionProcessGateway(
            protocol_path=protocol_path,
            channel_config_path=channel_config_path,
        )
        self.auto_import_default = dataset_root is None
        self._active = self._metadata

    @property
    def snapshot(self) -> TaskSnapshot:
        return self._active.snapshot

    @property
    def config(self) -> CaptureConfig:
        return self._active.config

    @property
    def device(self) -> DeviceInfo:
        return self._active.device

    @property
    def datasets(self) -> tuple[Dataset, ...]:
        combined: list[Dataset] = []
        seen: set[str] = set()
        for dataset in (*self._metadata.datasets, *self._acquisition.datasets):
            if dataset.id not in seen:
                combined.append(dataset)
                seen.add(dataset.id)
        return tuple(combined)

    @property
    def users(self) -> tuple[UserProfile, ...]:
        return self._metadata.users

    @property
    def trashed_users(self) -> tuple[UserProfile, ...]:
        return self._metadata.trashed_users

    @property
    def trashed_datasets(self) -> tuple[Dataset, ...]:
        return self._metadata.trashed_datasets

    def next_user_id(self) -> str:
        return self._metadata.next_user_id()

    def add_user(self, profile: UserProfile) -> UserProfile:
        return self._metadata.add_user(profile)

    def update_user(self, original_id: str, profile: UserProfile) -> UserProfile:
        return self._metadata.update_user(original_id, profile)

    def delete_user(self, user_id: str, *, move_data: bool) -> UserProfile:
        profile = self._metadata.delete_user(user_id, move_data=move_data)
        self.refresh_datasets()
        return profile

    def restore_user(self, user_id: str) -> UserProfile:
        profile = self._metadata.restore_user(user_id)
        self.refresh_datasets()
        return profile

    def purge_user(self, user_id: str) -> None:
        self._metadata.purge_user(user_id)
        self.refresh_datasets()

    def export_public_users(self, path: Path) -> Path:
        return self._metadata.export_public_users(path)

    @property
    def openbci_status(self) -> OpenBCIWorkspaceStatus:
        return self._metadata.openbci_status

    def preflight_cyton(self, seconds: float = 3.0, port: str = "AUTO") -> dict[str, object]:
        return self._acquisition.preflight_cyton(seconds, port)

    def start_ssvep(self, config: CaptureConfig, speed: float = 1) -> TaskSnapshot:
        self._ensure_available()
        config.validate()
        if CaptureMode(config.mode) is not CaptureMode.CYTON:
            raise ValueError("validation.real_hardware_only")
        if speed != 1:
            raise ValueError("validation.production_speed")
        profile = self._metadata._users.get(config.user_id)
        if profile is None or profile.status != "active":
            raise ValueError("validation.user_not_found")
        if profile.is_demo:
            raise ValueError("validation.user_demo")
        if CaptureMode(config.mode) == CaptureMode.CYTON and not config.allow_draft_hardware_config:
            channel_path = Path(config.channel_config or self._acquisition.channel_config_path).resolve()
            try:
                protocol_value = json.loads(
                    self._acquisition.protocol_path.read_text(encoding="utf-8")
                )
                channel_value = json.loads(channel_path.read_text(encoding="utf-8"))
                if not isinstance(protocol_value, dict) or not isinstance(channel_value, dict):
                    raise ConfigError("configuration roots must be objects")
                warnings = []
                if str(protocol_value.get("status", "")).lower().startswith("draft"):
                    warnings.append("protocol is draft")
                warnings.extend(validate_channel_config(channel_value, channel_path))
            except (ConfigError, OSError, ValueError, json.JSONDecodeError) as error:
                raise ValueError("validation.hardware_config") from error
            if warnings:
                raise ValueError("validation.hardware_draft")
        config = replace(
            config,
            participant=profile.user_id,
            user_id=profile.user_id,
            user_name=profile.name,
        )
        self._active = self._acquisition
        return self._acquisition.start_ssvep(config, 1)

    def cancel(self) -> TaskSnapshot:
        return self._active.cancel()

    def tick(self) -> TaskSnapshot:
        return self._active.tick()

    def read_live_waveform(self, maximum_rows: int = 1000):
        if self._active is self._acquisition:
            return self._acquisition.read_live_waveform(maximum_rows)
        return None

    def launch_openbci_workspace(self, locale: str) -> int:
        self._ensure_available()
        return self._metadata.launch_openbci_workspace(locale)

    def import_openbci_recordings(self, source_root: Path) -> OpenBCIImportReport:
        self._ensure_available()
        return self._metadata.import_openbci_recordings(source_root)

    def refresh_datasets(self) -> None:
        self._metadata.refresh_datasets()

    def delete_dataset(self, dataset_id: str) -> Dataset:
        self._ensure_available()
        result = self._metadata.delete_dataset(dataset_id)
        self._acquisition.discard_dataset(dataset_id)
        return result

    def restore_dataset(self, dataset_id: str) -> Dataset:
        self._ensure_available()
        result = self._metadata.restore_dataset(dataset_id)
        self._acquisition.discard_dataset(dataset_id)
        return result

    def purge_dataset(self, dataset_id: str) -> None:
        self._ensure_available()
        self._metadata.purge_dataset(dataset_id)
        self._acquisition.discard_dataset(dataset_id)

    def scan_serial_ports(self) -> tuple[dict[str, str], ...]:
        return tuple(item.as_dict() for item in discover_serial_ports())

    def _ensure_available(self) -> None:
        if self.snapshot.active:
            raise RuntimeError("validation.busy")
