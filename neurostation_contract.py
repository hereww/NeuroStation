"""Stable data contract shared by the independent UI and acquisition adapters."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
from typing import Protocol


FREQUENCIES = (10, 12, 15, 20)
PREPARATION_SECONDS = 5
SAMPLE_RATE = 250
CHANNEL_COUNT = 8


class CaptureMode(str, Enum):
    DEMO = "demo"
    SYNTHETIC = "synthetic"
    CYTON = "cyton"


def default_save_directory() -> Path:
    override = os.environ.get("NEUROSTATION_DATASETS")
    if override:
        return Path(override).expanduser()
    return Path.home() / "Documents" / "NeuroStation" / "Datasets"


def format_duration(seconds: float) -> str:
    value = max(0, int(seconds))
    return f"{value // 3600:02}:{value // 60 % 60:02}:{value % 60:02}"


@dataclass(frozen=True)
class CaptureConfig:
    participant: str = "P001"
    name: str = "SSVEP"
    stimulus_seconds: int = 5
    rest_seconds: int = 3
    repetitions: int = 3
    save_directory: Path = default_save_directory()
    refresh_rate: int = 60
    mode: CaptureMode = CaptureMode.DEMO
    port: str = "COM5"
    screen_index: int = 0
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
        if type(self.screen_index) is not int or self.screen_index < 0:
            raise ValueError("validation.screen")
        if mode == CaptureMode.CYTON and not self.port.strip():
            raise ValueError("validation.port")
        if mode != CaptureMode.DEMO and not self.acknowledge_flicker_risk:
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
        return 2 + 2 * self.trials


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
    port: str = "COM5"
    channels: int = CHANNEL_COUNT
    sample_rate: int = SAMPLE_RATE
    connected: bool = True
    simulated: bool = True


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
    simulated: bool = True
    persisted: bool = False
    source: CaptureMode = CaptureMode.DEMO
    status: str = "completed"
    channel_count: int = CHANNEL_COUNT
    error: str = ""

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
    speed: float = 8
    result: Dataset | None = None
    error: str = ""

    @property
    def active(self) -> bool:
        return self.phase in (Phase.COUNTDOWN, Phase.RUNNING)


class CaptureGateway(Protocol):
    device: DeviceInfo
    config: CaptureConfig

    @property
    def datasets(self) -> tuple[Dataset, ...]: ...

    @property
    def snapshot(self) -> TaskSnapshot: ...

    @property
    def openbci_status(self) -> OpenBCIWorkspaceStatus: ...

    def start_ssvep(self, config: CaptureConfig, speed: float = 8) -> TaskSnapshot: ...
    def start_manual(self, config: CaptureConfig) -> TaskSnapshot: ...
    def stop_manual(self) -> Dataset: ...
    def add_marker(self) -> int: ...
    def cancel(self) -> TaskSnapshot: ...
    def tick(self) -> TaskSnapshot: ...
    def launch_openbci_workspace(self, locale: str) -> int: ...
