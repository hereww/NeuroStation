"""UI-facing contract exports and a deterministic, hardware-free mock."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from time import monotonic
from typing import Callable
from uuid import uuid4

from neurostation_contract import (
    CHANNEL_COUNT,
    FREQUENCIES,
    PREPARATION_SECONDS,
    SAMPLE_RATE,
    CaptureConfig,
    CaptureGateway,
    CaptureMode,
    Dataset,
    DeviceInfo,
    OpenBCIImportReport,
    OpenBCIWorkspaceStatus,
    Phase,
    TaskSnapshot,
    default_save_directory,
    format_duration,
)


__all__ = [
    "CHANNEL_COUNT",
    "FREQUENCIES",
    "PREPARATION_SECONDS",
    "SAMPLE_RATE",
    "CaptureConfig",
    "CaptureGateway",
    "CaptureMode",
    "Dataset",
    "DeviceInfo",
    "MockGateway",
    "OpenBCIImportReport",
    "OpenBCIWorkspaceStatus",
    "Phase",
    "TaskSnapshot",
    "default_save_directory",
    "format_duration",
]


class MockGateway:
    """No device access, file writes, or timers; the host calls tick()."""

    def __init__(self, clock: Callable[[], float] = monotonic) -> None:
        self._clock = clock
        self.device = DeviceInfo()
        self.config = CaptureConfig()
        self._snapshot = TaskSnapshot()
        self._datasets: list[Dataset] = []
        self._task_start = 0.0
        self._run_start = 0.0
        self._manual_markers = 0

    @property
    def snapshot(self) -> TaskSnapshot:
        return self._snapshot

    @property
    def datasets(self) -> tuple[Dataset, ...]:
        return tuple(self._datasets)

    @property
    def openbci_status(self) -> OpenBCIWorkspaceStatus:
        return OpenBCIWorkspaceStatus()

    def launch_openbci_workspace(self, locale: str) -> int:
        raise RuntimeError("validation.openbci_unavailable")

    def import_openbci_recordings(self, source_root):
        return OpenBCIImportReport()

    def refresh_datasets(self) -> None:
        return None

    def _ensure_available(self) -> None:
        if self.snapshot.active:
            raise RuntimeError("validation.busy")

    def start_ssvep(self, config: CaptureConfig, speed: float = 8) -> TaskSnapshot:
        self._ensure_available()
        config.validate()
        if not math.isfinite(speed) or not 1 <= speed <= 32:
            raise ValueError("validation.speed")
        self.config = config
        self._task_start = self._clock()
        self._manual_markers = 0
        self._snapshot = TaskSnapshot(
            phase=Phase.COUNTDOWN,
            speed=speed,
            trial_count=config.trials,
            remaining=config.recording_seconds,
        )
        return self.snapshot

    def start_manual(self, config: CaptureConfig) -> TaskSnapshot:
        self._ensure_available()
        config.validate()
        self.config = config
        self._task_start = self._run_start = self._clock()
        self._manual_markers = 0
        self._snapshot = TaskSnapshot(
            phase=Phase.RUNNING,
            protocol="manual",
            speed=1,
            event_count=1,
        )
        return self.snapshot

    def tick(self) -> TaskSnapshot:
        previous = self.snapshot
        now = self._clock()
        if previous.phase == Phase.COUNTDOWN:
            remaining = PREPARATION_SECONDS - (now - self._task_start)
            if remaining > 0:
                self._snapshot = TaskSnapshot(
                    phase=Phase.COUNTDOWN,
                    countdown=math.ceil(remaining),
                    speed=previous.speed,
                    trial_count=self.config.trials,
                    remaining=self.config.recording_seconds,
                )
                return self.snapshot
            # Begin when the UI actually processes the deadline. A suspended UI
            # must not silently skip an entire task while no run screen is shown.
            self._run_start = now
            previous = TaskSnapshot(phase=Phase.RUNNING, speed=previous.speed)
        if previous.phase != Phase.RUNNING:
            return self.snapshot
        elapsed = max(0.0, (now - self._run_start) * previous.speed)
        if previous.protocol == "manual":
            self._snapshot = TaskSnapshot(
                phase=Phase.RUNNING,
                protocol="manual",
                elapsed=elapsed,
                demo_elapsed=now - self._task_start,
                speed=1,
                event_count=1 + self._manual_markers,
            )
            return self.snapshot
        config = self.config
        elapsed = min(elapsed, config.recording_seconds)
        if elapsed >= config.recording_seconds:
            result = self._make_dataset(
                "ssvep", elapsed, config.expected_events, config.trials
            )
            self._snapshot = TaskSnapshot(
                phase=Phase.COMPLETED,
                elapsed=elapsed,
                progress=100,
                trial=config.trials,
                trial_count=config.trials,
                event_count=config.expected_events,
                demo_elapsed=now - self._task_start,
                speed=previous.speed,
                result=result,
            )
            return self.snapshot
        index = min(
            config.trials - 1,
            int(elapsed // (config.stimulus_seconds + config.rest_seconds)),
        )
        within = elapsed - index * (config.stimulus_seconds + config.rest_seconds)
        resting = within >= config.stimulus_seconds
        self._snapshot = TaskSnapshot(
            phase=Phase.RUNNING,
            elapsed=elapsed,
            remaining=config.recording_seconds - elapsed,
            demo_elapsed=now - self._task_start,
            progress=elapsed / config.recording_seconds * 100,
            trial=index + 1,
            trial_count=config.trials,
            target=index % 4 + 1,
            frequency=FREQUENCIES[index % 4],
            resting=resting,
            event_count=2 + index * 2 + int(resting),
            speed=previous.speed,
        )
        return self.snapshot

    def add_marker(self) -> int:
        if self.snapshot.phase != Phase.RUNNING or self.snapshot.protocol != "manual":
            raise RuntimeError("validation.manual_marker")
        self._manual_markers += 1
        return self.tick().event_count

    def stop_manual(self) -> Dataset:
        if self.snapshot.phase != Phase.RUNNING or self.snapshot.protocol != "manual":
            raise RuntimeError("validation.manual_stop")
        current = self.tick()
        result = self._make_dataset(
            "manual", current.elapsed, self._manual_markers + 2, 0
        )
        self._snapshot = TaskSnapshot(
            phase=Phase.COMPLETED,
            protocol="manual",
            elapsed=current.elapsed,
            event_count=result.event_count,
            result=result,
            speed=1,
        )
        return result

    def cancel(self) -> TaskSnapshot:
        if self.snapshot.active:
            self._snapshot = TaskSnapshot(
                phase=Phase.CANCELLED, protocol=self.snapshot.protocol
            )
        return self.snapshot

    def _make_dataset(
        self, protocol: str, duration: float, events: int, trials: int
    ) -> Dataset:
        now = datetime.now(timezone.utc)
        identifier = f"{protocol}-{now:%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"
        result = Dataset(
            id=identifier,
            name=self.config.name,
            participant=self.config.participant,
            protocol=protocol,
            recording_seconds=duration,
            preparation_seconds=PREPARATION_SECONDS if protocol == "ssvep" else 0,
            demo_seconds=self._clock() - self._task_start,
            trials=trials,
            samples_per_channel=int(duration * SAMPLE_RATE),
            event_count=events,
            path=self.config.save_directory.expanduser().resolve() / identifier,
            created_at=now.isoformat(),
            source=CaptureMode.DEMO,
        )
        self._datasets.append(result)
        return result
