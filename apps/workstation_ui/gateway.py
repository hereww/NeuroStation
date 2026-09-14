"""UI-facing contract exports and a deterministic, hardware-free mock."""

from __future__ import annotations

from dataclasses import replace
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
from neurostation_contract import GENDERS, MEDICAL_OPTIONS, UserProfile


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


class _MemoryUserRegistry:
    def __init__(self):
        self._users = {
            "U0000": UserProfile(
                user_id="U0000", name="演示用户", age=0, is_demo=True,
                created_at="memory", updated_at="memory",
            )
        }
        self._trash = {}

    @property
    def users(self):
        return tuple(sorted(self._users.values(), key=lambda item: item.user_id))

    @property
    def trash(self):
        return tuple(sorted(self._trash.values(), key=lambda item: item.user_id))

    def next_user_id(self):
        values = [int(key[1:]) for key in (*self._users, *self._trash) if key.startswith("U") and key[1:].isdigit()]
        return f"U{max(values or [0]) + 1:04d}"

    def get(self, user_id):
        return self._users.get(user_id)

    def add(self, profile):
        profile = replace(
            profile,
            user_id=profile.user_id.strip() or self.next_user_id(),
            name=profile.name.strip(),
            status="active",
            created_at=profile.created_at or _now(),
            updated_at=_now(),
            deleted_at="",
        )
        profile.validate()
        if profile.user_id in self._users or profile.user_id in self._trash:
            raise ValueError("validation.user_id_duplicate")
        self._users[profile.user_id] = profile
        return profile

    def update(self, original_id, profile):
        if original_id not in self._users:
            raise ValueError("validation.user_not_found")
        profile = replace(profile, user_id=profile.user_id.strip(), name=profile.name.strip(),
                          updated_at=_now(), status="active", deleted_at="")
        profile.validate()
        if profile.user_id != original_id and (
            profile.user_id in self._users or profile.user_id in self._trash
        ):
            raise ValueError("validation.user_id_duplicate")
        del self._users[original_id]
        self._users[profile.user_id] = profile
        return profile

    def delete(self, user_id, *, move_data=False):
        if user_id == "U0000":
            raise ValueError("validation.user_not_found")
        profile = self._users.pop(user_id, None)
        if profile is None:
            raise ValueError("validation.user_not_found")
        self._trash[user_id] = replace(profile, status="trash", deleted_at=_now(), updated_at=_now())
        return profile

    def restore(self, user_id):
        profile = self._trash.pop(user_id, None)
        if profile is None:
            raise ValueError("validation.user_not_found")
        self._users[user_id] = replace(profile, status="active", deleted_at="", updated_at=_now())
        return profile

    def purge(self, user_id):
        if user_id not in self._trash:
            raise ValueError("validation.user_not_found")
        del self._trash[user_id]


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
        self._users = _MemoryUserRegistry()

    @property
    def snapshot(self) -> TaskSnapshot:
        return self._snapshot

    @property
    def datasets(self) -> tuple[Dataset, ...]:
        return tuple(self._datasets)

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
        return self._users.delete(user_id, move_data=move_data)

    def restore_user(self, user_id: str) -> UserProfile:
        return self._users.restore(user_id)

    def purge_user(self, user_id: str) -> None:
        self._users.purge(user_id)

    def export_public_users(self, path: Path) -> Path:
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        def redacted(profile):
            return {
                "user_id": profile.user_id,
                "status": profile.status,
                "is_demo": profile.is_demo,
                "created_at": profile.created_at,
                "updated_at": profile.updated_at,
                "screening_recorded": bool(profile.medical_conditions),
            }
        output.write_text(
            json.dumps({
                "schema_version": 1,
                "users": [redacted(item) for item in self._users.users],
                "trash": [redacted(item) for item in self._users.trash],
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output

    @property
    def openbci_status(self) -> OpenBCIWorkspaceStatus:
        return OpenBCIWorkspaceStatus()

    def launch_openbci_workspace(self, locale: str) -> int:
        raise RuntimeError("validation.openbci_unavailable")

    def import_openbci_recordings(self, source_root):
        return OpenBCIImportReport()

    def refresh_datasets(self) -> None:
        return None

    def scan_serial_ports(self) -> tuple[dict[str, str], ...]:
        return ()

    def _ensure_available(self) -> None:
        if self.snapshot.active:
            raise RuntimeError("validation.busy")

    def preflight_cyton(self, seconds: float = 3.0, port: str = "AUTO") -> dict[str, object]:
        return {"status": "unavailable", "requested_port": "AUTO", "checks": []}

    def start_ssvep(self, config: CaptureConfig, speed: float = 8) -> TaskSnapshot:
        self._ensure_available()
        config.validate()
        profile = self._users.get(config.user_id)
        if profile is None or profile.status != "active":
            raise ValueError("validation.user_not_found")
        if config.mode != CaptureMode.DEMO and profile.is_demo:
            raise ValueError("validation.user_demo")
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
            event_count=1,
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
            previous = TaskSnapshot(phase=Phase.RUNNING, speed=previous.speed, event_count=2)
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
                event_count=3 + index * 2 + int(resting),
            speed=previous.speed,
        )
        return self.snapshot

    def read_live_waveform(self, maximum_rows: int = 1000):
        return None

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
            origin="capture_test" if protocol == "manual" else "acquired",
            user_id=self.config.user_id,
            user_name=self.config.user_name,
            user_link_status="active",
        )
        self._datasets.append(result)
        return result
