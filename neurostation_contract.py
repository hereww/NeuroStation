"""Stable data contract shared by the independent UI and acquisition adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import os
from pathlib import Path
import re
from typing import Protocol
from datetime import datetime
from dataclasses import asdict
from typing import Any


PRODUCT_NAME = "NeuroStation"
PRODUCT_VERSION = "MVP1.0.3"
PRODUCT_SEMVER = "1.0.3"
PRODUCT_DESCRIPTION = (
    "Windows-first EEG acquisition workstation for OpenBCI Cyton SSVEP capture, "
    "session metadata, quality reports, and dataset review."
)
RELEASE_DATE = "2026-09-16"


FREQUENCIES = (10, 12, 15, 20)
PREPARATION_SECONDS = 5
SAMPLE_RATE = 250
CHANNEL_COUNT = 8
GENDERS = ("male", "female", "other", "unspecified")
MEDICAL_OPTIONS = (
    "none",
    "photosensitive_epilepsy",
    "cardiovascular",
    "hypertension",
    "diabetes",
    "other",
)


def is_confirmed_position(value: Any) -> bool:
    """Return whether a position is an operator-confirmed placement label."""

    if not isinstance(value, str):
        return False
    normalized = value.strip().casefold()
    return normalized not in {"", "unspecified", "unknown", "unassigned", "未指定", "未设置"}


@dataclass(frozen=True)
class UserProfile:
    user_id: str
    name: str
    age: int
    gender: str = "unspecified"
    medical_conditions: tuple[str, ...] = ("none",)
    medical_other: str = ""
    created_at: str = ""
    updated_at: str = ""
    status: str = "active"
    is_demo: bool = False
    deleted_at: str = ""

    @property
    def display_name(self) -> str:
        return f"{self.user_id} · {self.name}"

    def validate(self) -> None:
        if not self.user_id.strip():
            raise ValueError("validation.user_id")
        if not self.name.strip():
            raise ValueError("validation.user_name")
        if type(self.age) is not int or not 0 <= self.age <= 150:
            raise ValueError("validation.user_age")
        if self.gender not in GENDERS:
            raise ValueError("validation.user_gender")
        conditions = tuple(dict.fromkeys(self.medical_conditions))
        if not conditions or any(item not in MEDICAL_OPTIONS for item in conditions):
            raise ValueError("validation.user_medical")
        if "none" in conditions and len(conditions) > 1:
            raise ValueError("validation.user_medical_none")
        if "other" in conditions and not self.medical_other.strip():
            raise ValueError("validation.user_medical_other")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["medical_conditions"] = list(self.medical_conditions)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "UserProfile":
        conditions = value.get("medical_conditions", ("none",))
        if isinstance(conditions, str):
            conditions = (conditions,)
        if not isinstance(conditions, (list, tuple)):
            conditions = ("none",)
        return cls(
            user_id=str(value.get("user_id") or ""),
            name=str(value.get("name") or ""),
            age=int(value.get("age", 0) or 0),
            gender=str(value.get("gender") or "unspecified"),
            medical_conditions=tuple(str(item) for item in conditions),
            medical_other=str(value.get("medical_other") or ""),
            created_at=str(value.get("created_at") or ""),
            updated_at=str(value.get("updated_at") or ""),
            status=str(value.get("status") or "active"),
            is_demo=bool(value.get("is_demo", False)),
            deleted_at=str(value.get("deleted_at") or ""),
        )


class CaptureMode(str, Enum):
    # The legacy values remain readable so older datasets can still be shown
    # in the dataset browser.  New acquisition requests are restricted to
    # CYTON by CaptureConfig.validate().
    DEMO = "demo"
    VISUAL_PREVIEW = "preview"
    SYNTHETIC = "synthetic"
    CYTON = "cyton"
    IMPORTED_OPENBCI = "imported_openbci"


class EyeSide(str, Enum):
    LEFT = "left"
    RIGHT = "right"

    @property
    def label_zh(self) -> str:
        return "左眼" if self is EyeSide.LEFT else "右眼"


def normalize_dataset_name_base(name: str) -> str:
    """Remove trailing eye suffixes and return the canonical base name."""

    base = re.sub(r"(?:_(?:左眼|右眼))+$", "", str(name).strip())
    if not base:
        raise ValueError("validation.identity")
    return base


def dataset_name_for_eye(name: str, eye_side: EyeSide | str) -> str:
    """Return a stable dataset name with exactly one Chinese eye suffix."""

    try:
        side = EyeSide(eye_side)
    except ValueError as error:
        raise ValueError("validation.eye_side") from error
    base = normalize_dataset_name_base(name)
    return f"{base}_{side.label_zh}"


def default_save_directory() -> Path:
    override = os.environ.get("NEUROSTATION_DATASETS")
    if override:
        # Resolve the environment override once at the boundary so every
        # default-created CaptureConfig is valid on all three desktop OSes.
        return Path(override).expanduser().resolve()
    return (Path.home() / "Documents" / "NeuroStation" / "Datasets").resolve()


def default_user_settings_directory() -> Path:
    """Return the writable per-user directory for calibrated acquisition settings."""

    return (Path.home() / "Documents" / PRODUCT_NAME / "Settings").resolve()


def default_user_channel_config_path() -> Path:
    return default_user_settings_directory() / "channel_config_v1.json"


def default_user_protocol_path() -> Path:
    return default_user_settings_directory() / "ssvep_four_target_v2.json"


def format_duration(seconds: float) -> str:
    value = max(0, int(seconds))
    return f"{value // 3600:02}:{value // 60 % 60:02}:{value % 60:02}"


@dataclass(frozen=True)
class CaptureConfig:
    participant: str = ""
    name: str = "SSVEP"
    user_id: str = ""
    user_name: str = ""
    stimulus_seconds: int = 5
    rest_seconds: int = 3
    repetitions: int = 3
    # Use a factory rather than evaluating the path at import time. This keeps
    # NEUROSTATION_DATASETS and per-user home directories reliable in packaged
    # apps, test runners, and long-lived desktop processes.
    save_directory: Path = field(default_factory=default_save_directory)
    refresh_rate: int = 60
    mode: CaptureMode = CaptureMode.CYTON
    port: str = "AUTO"
    eye_side: EyeSide | str = ""
    acknowledge_flicker_risk: bool = False
    channel_config: Path | None = None
    allow_draft_hardware_config: bool = False

    def validate(self) -> None:
        if not self.participant.strip() or not self.name.strip():
            raise ValueError("validation.identity")
        for value, low, high in (
            (self.stimulus_seconds, 1, 30),
            (self.rest_seconds, 0, 30),
            (self.repetitions, 1, 10),
        ):
            if type(value) is not int or not low <= value <= high:
                raise ValueError("validation.parameters")
        if not Path(self.save_directory).expanduser().is_absolute():
            raise ValueError("validation.path")
        if self.refresh_rate != 60:
            raise ValueError("validation.refresh")
        try:
            mode = CaptureMode(self.mode)
        except ValueError as error:
            raise ValueError("validation.mode") from error
        if mode is not CaptureMode.CYTON:
            raise ValueError("validation.real_hardware_only")
        try:
            EyeSide(self.eye_side)
        except ValueError as error:
            raise ValueError("validation.eye_side") from error
        if not self.port.strip():
            raise ValueError("validation.port")
        if not self.acknowledge_flicker_risk:
            raise ValueError("validation.flicker_ack")
        if self.channel_config is not None:
            channel_path = Path(self.channel_config).expanduser()
            if not channel_path.is_absolute():
                raise ValueError("validation.channel_path")

    @property
    def trials(self) -> int:
        return len(FREQUENCIES) * self.repetitions

    @property
    def recording_seconds(self) -> int:
        return self.trials * self.stimulus_seconds + (self.trials - 1) * self.rest_seconds

    @property
    def total_seconds(self) -> int:
        return PREPARATION_SECONDS + self.recording_seconds

    @property
    def expected_events(self) -> int:
        return 3 + 2 * self.trials

    @property
    def dataset_name(self) -> str:
        return dataset_name_for_eye(self.name, self.eye_side)


class Phase(str, Enum):
    IDLE = "idle"
    COUNTDOWN = "countdown"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class DeviceInfo:
    name: str = "OpenBCI Cyton"
    port: str = "AUTO"
    channels: int = CHANNEL_COUNT
    sample_rate: int = SAMPLE_RATE
    connected: bool = False
    simulated: bool = False


@dataclass(frozen=True)
class OpenBCIWorkspaceStatus:
    source_ready: bool = False
    overlay_ready: bool = False
    executable_ready: bool = False
    revision: str = ""


@dataclass(frozen=True)
class Dataset:
    id: str
    name: str
    participant: str
    protocol: str
    recording_seconds: float
    preparation_seconds: int
    demo_seconds: float
    trials: int
    samples_per_channel: int
    event_count: int
    path: Path
    created_at: str
    eye_side: str = ""
    screen_index: int | None = None
    screen_name: str = ""
    simulated: bool = False
    persisted: bool = False
    source: CaptureMode = CaptureMode.CYTON
    status: str = "completed"
    channel_count: int = CHANNEL_COUNT
    error: str = ""
    sampling_rate_hz: int = SAMPLE_RATE
    files: tuple[str, ...] = ()
    source_path: str = ""
    imported: bool = False
    origin: str = "acquired"
    user_id: str = ""
    user_name: str = ""
    user_link_status: str = "unlinked"
    validation_mode: str = "technical_validation"
    protocol_status: str = ""
    protocol_sha256: str = ""
    channel_config_sha256: str = ""
    quality_status: str = "unknown"
    timestamp_gap_count: int = 0
    dropped_frame_count: int = 0
    flat_channel_count: int = 0
    deleted_at: str = ""

    @property
    def sample_values(self) -> int:
        return self.samples_per_channel * self.channel_count


@dataclass(frozen=True)
class TaskSnapshot:
    phase: Phase = Phase.IDLE
    protocol: str = "ssvep"
    countdown: int = PREPARATION_SECONDS
    elapsed: float = 0
    remaining: float = 0
    demo_elapsed: float = 0
    progress: float = 0
    trial: int = 0
    trial_count: int = 0
    target: int = 0
    frequency: int = 0
    resting: bool = False
    event_count: int = 0
    speed: float = 1
    eye_side: str = ""
    screen_index: int | None = None
    screen_name: str = ""
    result: Dataset | None = None
    error: str = ""

    @property
    def active(self) -> bool:
        return self.phase in (Phase.COUNTDOWN, Phase.RUNNING)


@dataclass(frozen=True)
class OpenBCIImportReport:
    """Outcome of one immutable OpenBCI GUI recording import attempt."""

    imported_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    failures: tuple[str, ...] = ()


class CaptureGateway(Protocol):
    device: DeviceInfo
    config: CaptureConfig

    @property
    def datasets(self) -> tuple[Dataset, ...]: ...

    @property
    def snapshot(self) -> TaskSnapshot: ...

    @property
    def openbci_status(self) -> OpenBCIWorkspaceStatus: ...

    def preflight_cyton(self, seconds: float = 3.0, port: str = "AUTO") -> dict[str, Any]: ...

    def test_cyton_channel(
        self,
        channel_number: int,
        seconds: float = 3.0,
        port: str = "AUTO",
        sample_callback=None,
    ) -> dict[str, Any]: ...

    def load_channel_calibration(self) -> dict[str, Any] | None: ...
    def save_channel_calibration(self, calibration: dict[str, Any]) -> dict[str, Any]: ...

    def start_ssvep(self, config: CaptureConfig, speed: float = 1) -> TaskSnapshot: ...
    def cancel(self) -> TaskSnapshot: ...
    def tick(self) -> TaskSnapshot: ...
    def read_live_waveform(self, maximum_rows: int = 1000) -> dict[str, Any] | None: ...
    def launch_openbci_workspace(self, locale: str) -> int: ...
    def import_openbci_recordings(self, source_root: Path) -> OpenBCIImportReport: ...
    def refresh_datasets(self) -> None: ...
    def delete_dataset(self, dataset_id: str) -> Dataset: ...
    def restore_dataset(self, dataset_id: str) -> Dataset: ...
    def purge_dataset(self, dataset_id: str) -> None: ...

    @property
    def trashed_datasets(self) -> tuple[Dataset, ...]: ...
    def export_public_users(self, path: Path) -> Path: ...

    def scan_serial_ports(self) -> tuple[dict[str, str], ...]: ...
