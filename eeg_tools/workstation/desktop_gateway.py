"""Composition adapter between the independent desktop UI and workstation core."""

from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from uuid import uuid4

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
from .gateway import WorkstationGateway
from .openbci_workspace import OpenBCIWorkspaceError, OpenBCIWorkspaceManager
from .process_gateway import AcquisitionProcessGateway
from .device_discovery import discover_serial_ports
from eeg_tools.config import ConfigError, validate_channel_config
from .task import TaskPhase
from .users import UserProfile, UserRegistry


ROOT = Path(__file__).resolve().parents[2]


class MetadataSimulationGateway:
    """Run the reviewed SSVEP flow and persist non-EEG acceptance metadata.

    This adapter deliberately remains a simulation: it never opens a serial port
    and never labels expected sample counts as recorded EEG. It does, however,
    create the session directory shown by the result page.
    """

    def __init__(
        self,
        *,
        protocol_path: Path,
        dataset_root: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.protocol_path = protocol_path.resolve()
        self._clock = clock
        self.device = DeviceInfo()
        self.config = CaptureConfig(
            save_directory=(dataset_root or DatasetRepository.default_root()).resolve()
        )
        self.repository = DatasetRepository(Path(self.config.save_directory))
        self.auto_import_default = dataset_root is None
        self.last_import_report = OpenBCIImportReport()
        self._snapshot = TaskSnapshot()
        self._service: WorkstationGateway | None = None
        self._manual_markers = 0
        self._datasets: list[Dataset] = []
        self._trashed_transient: list[Dataset] = []
        self._task_started_at = 0.0
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

    def start_ssvep(self, config: CaptureConfig, speed: float = 8) -> TaskSnapshot:
        self._ensure_available()
        config.validate()
        if not math.isfinite(speed) or not 1 <= speed <= 32:
            raise ValueError("validation.speed")
        root = Path(config.save_directory).expanduser().resolve()
        self._load_existing(root)
        self._users = UserRegistry(root)
        profile = self._users.get(config.user_id)
        if profile is None or profile.status != "active":
            raise ValueError("validation.user_not_found")
        if config.mode != CaptureMode.DEMO and profile.is_demo:
            raise ValueError("validation.user_demo")
        self.config = replace(
            config,
            participant=profile.user_id,
            user_id=profile.user_id,
            user_name=profile.name,
        )
        self._manual_markers = 0
        self._service = WorkstationGateway(
            protocol_path=self.protocol_path,
            dataset_root=root,
            clock=self._clock,
        )
        core = self._service.start_ssvep(
            participant_id=profile.user_id,
            session_name=config.name,
            user_id=profile.user_id,
            user_name=profile.name,
            user_link_status="active",
            repetitions=config.repetitions,
            stimulus_s=config.stimulus_seconds,
            rest_s=config.rest_seconds,
            simulation_speed=speed,
        )
        self._task_started_at = self._clock()
        self._last_result_id = None
        self._snapshot = self._map_snapshot(core, speed)
        return self._snapshot

    def start_manual(self, config: CaptureConfig) -> TaskSnapshot:
        self._ensure_available()
        config.validate()
        self._service = None
        self.config = config
        self._manual_markers = 0
        self._task_started_at = self._clock()
        self._snapshot = TaskSnapshot(
            phase=Phase.RUNNING,
            protocol="manual",
            speed=1,
            event_count=1,
        )
        return self._snapshot

    def stop_manual(self) -> Dataset:
        if self._snapshot.phase != Phase.RUNNING or self._snapshot.protocol != "manual":
            raise RuntimeError("validation.manual_stop")
        current = self.tick()
        now = datetime.now(timezone.utc)
        identifier = f"manual-{now:%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"
        result = Dataset(
            id=identifier,
            name=self.config.name,
            participant=self.config.participant,
            protocol="manual",
            recording_seconds=current.elapsed,
            preparation_seconds=0,
            demo_seconds=current.elapsed,
            trials=0,
            samples_per_channel=int(current.elapsed * 250),
            event_count=self._manual_markers + 2,
            path=Path(self.config.save_directory).expanduser().resolve() / identifier,
            created_at=now.isoformat(),
            simulated=True,
            persisted=False,
            source=CaptureMode.DEMO,
            origin="capture_test",
            user_id=self.config.user_id,
            user_name=self.config.user_name,
            user_link_status="active",
        )
        self._remember(result)
        self._snapshot = TaskSnapshot(
            phase=Phase.COMPLETED,
            protocol="manual",
            elapsed=current.elapsed,
            event_count=result.event_count,
            result=result,
            speed=1,
        )
        return result

    def add_marker(self) -> int:
        if self._snapshot.phase != Phase.RUNNING or self._snapshot.protocol != "manual":
            raise RuntimeError("validation.manual_marker")
        self._manual_markers += 1
        return self.tick().event_count

    def cancel(self) -> TaskSnapshot:
        if self._snapshot.phase == Phase.RUNNING and self._snapshot.protocol == "manual":
            self._snapshot = TaskSnapshot(phase=Phase.CANCELLED, protocol="manual")
            return self._snapshot
        if self._service is not None and self._snapshot.active:
            core = self._service.abort_task()
            # Keep the aborted session attached to the terminal snapshot so
            # the UI can show its saved path immediately, just like the
            # subprocess-backed Synthetic/Cyton gateway does.
            self._snapshot = self._map_snapshot(core, self._snapshot.speed)
            if self._snapshot.result is not None:
                self._remember(self._snapshot.result)
        return self._snapshot

    def tick(self) -> TaskSnapshot:
        if self._snapshot.phase == Phase.RUNNING and self._snapshot.protocol == "manual":
            elapsed = max(0.0, self._clock() - self._task_started_at)
            self._snapshot = TaskSnapshot(
                phase=Phase.RUNNING,
                protocol="manual",
                elapsed=elapsed,
                demo_elapsed=elapsed,
                speed=1,
                event_count=1 + self._manual_markers,
            )
            return self._snapshot
        if self._service is None or not self._snapshot.active:
            return self._snapshot
        core = self._service.poll_task()
        speed = self._snapshot.speed
        self._snapshot = self._map_snapshot(core, speed)
        if self._snapshot.result is not None:
            self._remember(self._snapshot.result)
        return self._snapshot

    def read_live_waveform(self, maximum_rows: int = 1000):
        return None

    def _map_snapshot(self, core, speed: float) -> TaskSnapshot:
        phase = {
            TaskPhase.READY: Phase.IDLE,
            TaskPhase.COUNTDOWN: Phase.COUNTDOWN,
            TaskPhase.RUNNING: Phase.RUNNING,
            TaskPhase.COMPLETED: Phase.COMPLETED,
            TaskPhase.ABORTED: Phase.CANCELLED,
        }[core.phase]
        protocol = self._service.active_protocol if self._service else None
        trial_count = protocol.trial_count if protocol else self.config.trials
        trial_index = core.current_trial_index
        within = 0.0
        resting = False
        if protocol and trial_index is not None:
            within = max(
                0.0,
                core.recording_elapsed_s
                - protocol.pre_session_rest_s
                - trial_index * protocol.seconds_per_trial,
            )
            resting = within >= protocol.pre_trial_rest_s + protocol.stimulus_s
        event_count = 0
        if phase is Phase.RUNNING and trial_index is not None:
            event_count = 3 + trial_index * 2 + int(resting)
        elif phase is Phase.COMPLETED and protocol:
            event_count = protocol.expected_event_count
        result = None
        if core.result is not None:
            result = self._dataset_from_record(
                core.result,
                demo_seconds=max(0.0, self._clock() - self._task_started_at),
            )
        target_number = 0
        if protocol and core.current_target_id:
            ids = [target_id for target_id, _ in protocol.targets]
            target_number = ids.index(core.current_target_id) + 1
        return TaskSnapshot(
            phase=phase,
            protocol="ssvep",
            countdown=max(0, math.ceil(core.countdown_remaining_s)),
            elapsed=core.recording_elapsed_s,
            remaining=core.recording_remaining_s,
            demo_elapsed=max(0.0, self._clock() - self._task_started_at),
            progress=core.progress * 100,
            trial=0 if trial_index is None else trial_index + 1,
            trial_count=trial_count,
            target=target_number,
            frequency=core.current_frequency_hz or 0,
            resting=resting,
            event_count=event_count,
            speed=speed,
            result=result,
        )

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
            samples_per_channel=(
                record.expected_samples_per_channel
                if source == CaptureMode.DEMO
                else record.recorded_samples_per_channel
            ),
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
    """One desktop facade for demo, synthetic BrainFlow, and Cyton modes."""

    def __init__(
        self,
        *,
        protocol_path: Path,
        channel_config_path: Path,
        dataset_root: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._simulation = MetadataSimulationGateway(
            protocol_path=protocol_path,
            dataset_root=dataset_root,
            clock=clock,
        )
        self._acquisition = AcquisitionProcessGateway(
            protocol_path=protocol_path,
            channel_config_path=channel_config_path,
        )
        self.auto_import_default = dataset_root is None
        self._active = self._simulation

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
        for dataset in (*self._simulation.datasets, *self._acquisition.datasets):
            if dataset.id not in seen:
                combined.append(dataset)
                seen.add(dataset.id)
        return tuple(combined)

    @property
    def users(self) -> tuple[UserProfile, ...]:
        return self._simulation.users

    @property
    def trashed_users(self) -> tuple[UserProfile, ...]:
        return self._simulation.trashed_users

    @property
    def trashed_datasets(self) -> tuple[Dataset, ...]:
        return self._simulation.trashed_datasets

    def next_user_id(self) -> str:
        return self._simulation.next_user_id()

    def add_user(self, profile: UserProfile) -> UserProfile:
        return self._simulation.add_user(profile)

    def update_user(self, original_id: str, profile: UserProfile) -> UserProfile:
        return self._simulation.update_user(original_id, profile)

    def delete_user(self, user_id: str, *, move_data: bool) -> UserProfile:
        profile = self._simulation.delete_user(user_id, move_data=move_data)
        self.refresh_datasets()
        return profile

    def restore_user(self, user_id: str) -> UserProfile:
        profile = self._simulation.restore_user(user_id)
        self.refresh_datasets()
        return profile

    def purge_user(self, user_id: str) -> None:
        self._simulation.purge_user(user_id)
        self.refresh_datasets()

    def export_public_users(self, path: Path) -> Path:
        return self._simulation.export_public_users(path)

    @property
    def openbci_status(self) -> OpenBCIWorkspaceStatus:
        return self._simulation.openbci_status

    def preflight_cyton(self, seconds: float = 3.0, port: str = "AUTO") -> dict[str, object]:
        return self._acquisition.preflight_cyton(seconds, port)

    def start_ssvep(self, config: CaptureConfig, speed: float = 8) -> TaskSnapshot:
        self._ensure_available()
        config.validate()
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
        profile = self._simulation._users.get(config.user_id)
        if profile is None or profile.status != "active":
            raise ValueError("validation.user_not_found")
        if CaptureMode(config.mode) != CaptureMode.DEMO and profile.is_demo:
            raise ValueError("validation.user_demo")
        config = replace(
            config,
            participant=profile.user_id,
            user_id=profile.user_id,
            user_name=profile.name,
        )
        if CaptureMode(config.mode) == CaptureMode.DEMO:
            self._active = self._simulation
            return self._simulation.start_ssvep(config, speed)
        self._active = self._acquisition
        return self._acquisition.start_ssvep(config, speed)

    def start_manual(self, config: CaptureConfig) -> TaskSnapshot:
        self._ensure_available()
        self._active = self._simulation
        return self._simulation.start_manual(config)

    def stop_manual(self) -> Dataset:
        return self._simulation.stop_manual()

    def add_marker(self) -> int:
        return self._simulation.add_marker()

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
        return self._simulation.launch_openbci_workspace(locale)

    def import_openbci_recordings(self, source_root: Path) -> OpenBCIImportReport:
        self._ensure_available()
        return self._simulation.import_openbci_recordings(source_root)

    def refresh_datasets(self) -> None:
        self._simulation.refresh_datasets()

    def delete_dataset(self, dataset_id: str) -> Dataset:
        self._ensure_available()
        result = self._simulation.delete_dataset(dataset_id)
        self._acquisition.discard_dataset(dataset_id)
        return result

    def restore_dataset(self, dataset_id: str) -> Dataset:
        self._ensure_available()
        result = self._simulation.restore_dataset(dataset_id)
        self._acquisition.discard_dataset(dataset_id)
        return result

    def purge_dataset(self, dataset_id: str) -> None:
        self._ensure_available()
        self._simulation.purge_dataset(dataset_id)
        self._acquisition.discard_dataset(dataset_id)

    def scan_serial_ports(self) -> tuple[dict[str, str], ...]:
        return tuple(item.as_dict() for item in discover_serial_ports())

    def _ensure_available(self) -> None:
        if self.snapshot.active:
            raise RuntimeError("validation.busy")
