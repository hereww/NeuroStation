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
from .ssvep import SSVEPProtocol, SSVEPProtocolError
from eeg_tools.config import ConfigError, is_confirmed_position, validate_channel_config
from eeg_tools.session_files import iso_now
from neurostation_contract import (
    default_user_channel_config_path,
    default_user_protocol_path,
)
from .users import UserProfile, UserRegistry


ROOT = Path(__file__).resolve().parents[2]


def _write_json_pending(path: Path, value: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return temporary


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    temporary = _write_json_pending(path, value)
    temporary.replace(path)


def _write_json_pair_atomic(
    updates: tuple[tuple[Path, dict[str, object]], ...],
) -> None:
    """Commit related JSON files together and restore the previous pair on failure."""

    previous: dict[Path, bytes | None] = {}
    pending: list[tuple[Path, Path]] = []
    try:
        for path, value in updates:
            previous[path] = path.read_bytes() if path.is_file() else None
            pending.append((path, _write_json_pending(path, value)))
        for path, temporary in pending:
            temporary.replace(path)
    except Exception:
        for path, content in previous.items():
            try:
                if content is None:
                    path.unlink(missing_ok=True)
                else:
                    restore = path.with_name(path.name + ".restore")
                    restore.write_bytes(content)
                    restore.replace(path)
            except OSError:
                pass
        raise
    finally:
        for _, temporary in pending:
            temporary.unlink(missing_ok=True)


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
            created_at=record.recorded_at or record.session_id,
            eye_side=record.eye_side,
            screen_index=record.screen_index,
            screen_name=record.screen_name,
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
    def protocol_path(self) -> Path:
        return self._acquisition.protocol_path

    @property
    def channel_config_path(self) -> Path:
        return self._acquisition.channel_config_path

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

    def test_cyton_channel(
        self,
        channel_number: int,
        seconds: float | None = 3.0,
        port: str = "AUTO",
        sample_callback=None,
        cancel_event=None,
    ) -> dict[str, object]:
        return self._acquisition.test_cyton_channel(
            channel_number,
            seconds,
            port,
            sample_callback=sample_callback,
            cancel_event=cancel_event,
        )

    def load_channel_calibration(self) -> dict[str, object] | None:
        path = default_user_channel_config_path()
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def save_channel_calibration(self, calibration: dict[str, object]) -> dict[str, object]:
        """Persist a reviewed channel map and activate the formal protocol copy."""

        if not isinstance(calibration, dict):
            raise ValueError("validation.calibration_config")
        channels = calibration.get("channels")
        if not isinstance(channels, list) or len(channels) != 8:
            raise ValueError("validation.calibration_channels")
        if any(not isinstance(item, dict) for item in channels):
            raise ValueError("validation.calibration_config")
        positions = [
            str(item.get("electrode_position") or "").strip()
            for item in channels
        ]
        if len(positions) != 8 or any(not is_confirmed_position(position) for position in positions):
            raise ValueError("validation.calibration_missing")
        if len(set(position.casefold() for position in positions)) != len(positions):
            raise ValueError("validation.calibration_duplicate")
        tests = calibration.get("tests")
        if (
            not isinstance(tests, list)
            or len(tests) != 8
            or any(
                not isinstance(item, dict) or str(item.get("status") or "") != "passed"
                for item in tests
            )
        ):
            raise ValueError("validation.calibration_test_required")
        for key in ("reference", "bias", "ground"):
            auxiliary = calibration.get(key)
            position = auxiliary.get("position") if isinstance(auxiliary, dict) else None
            if not is_confirmed_position(position):
                raise ValueError("validation.calibration_aux_missing")
        if not bool(calibration.get("protocol_reviewed")):
            raise ValueError("validation.calibration_protocol_ack")

        completed_at = iso_now()
        channel_path = default_user_channel_config_path()
        channel_value = {
            **calibration,
            "config_version": "CHANNEL-V1-FORMAL",
            "profile_type": "calibrated",
            "board": "OpenBCI Cyton",
            "serial_port": str(calibration.get("serial_port") or "AUTO"),
            "sampling_rate_hz": 250,
            "channel_order": [f"CH{index}" for index in range(1, 9)],
            "calibration": {
                "completed_at": completed_at,
                "operator_confirmed": True,
                "tests": tests,
            },
        }
        channel_value.pop("protocol_reviewed", None)
        try:
            channel_warnings = validate_channel_config(channel_value, channel_path)
        except ConfigError as error:
            raise ValueError("validation.calibration_validation") from error
        if channel_warnings:
            raise ValueError("validation.calibration_validation")

        try:
            protocol_value = json.loads(
                self._acquisition.protocol_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("validation.hardware_config") from error
        if not isinstance(protocol_value, dict):
            raise ValueError("validation.hardware_config")
        try:
            protocol = SSVEPProtocol.load(self._acquisition.protocol_path)
        except (OSError, SSVEPProtocolError) as error:
            raise ValueError("validation.hardware_config") from error
        if (
            protocol.refresh_rate_hz != 60
            or protocol.sampling_rate_hz != 250
            or protocol.channel_count != 8
        ):
            raise ValueError("validation.calibration_validation")
        protocol_value["status"] = "formal_candidate"
        protocol_value["formal_approval"] = {
            "approved_at": completed_at,
            "operator_confirmed": True,
            "screen_refresh_rate_hz": 60,
            "channel_config_path": str(channel_path),
        }
        protocol_path = default_user_protocol_path()
        _write_json_pair_atomic(((channel_path, channel_value), (protocol_path, protocol_value)))

        # Activate the user-scoped files immediately; a restart is not required.
        self._metadata.protocol_path = protocol_path
        self._acquisition.protocol_path = protocol_path
        self._acquisition.channel_config_path = channel_path
        return {
            "channel_config_path": str(channel_path),
            "protocol_path": str(protocol_path),
            "completed_at": completed_at,
        }

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
