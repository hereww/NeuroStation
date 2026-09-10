"""Deterministic UI-facing SSVEP task state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .dataset import DatasetRecord, DatasetRepository
from .ssvep import SSVEPProtocol


class TaskPhase(str, Enum):
    READY = "ready"
    COUNTDOWN = "countdown"
    RUNNING = "running"
    COMPLETED = "completed"
    ABORTED = "aborted"


@dataclass(frozen=True)
class TaskSnapshot:
    phase: TaskPhase
    countdown_remaining_s: float
    recording_elapsed_s: float
    recording_remaining_s: float
    progress: float
    current_trial_index: int | None
    current_target_id: str | None
    current_frequency_hz: int | None
    result: DatasetRecord | None


class SSVEPTask:
    def __init__(
        self,
        protocol: SSVEPProtocol,
        repository: DatasetRepository,
        *,
        participant_id: str,
        session_name: str,
        simulation_speed: float = 1.0,
    ):
        if simulation_speed <= 0:
            raise ValueError("simulation_speed must be greater than zero")
        if not participant_id.strip() or not session_name.strip():
            raise ValueError("participant_id and session_name are required")
        self.protocol = protocol
        self.repository = repository
        self.participant_id = participant_id.strip()
        self.session_name = session_name.strip()
        self.simulation_speed = simulation_speed
        self.phase = TaskPhase.READY
        self._started_at: float | None = None
        self._recording_started_at: float | None = None
        self._result: DatasetRecord | None = None

    def start(self, now_s: float) -> TaskSnapshot:
        if self.phase is not TaskPhase.READY:
            raise RuntimeError("task can only be started once")
        self._started_at = now_s
        self.phase = TaskPhase.COUNTDOWN
        return self.snapshot(now_s)

    def abort(self, now_s: float) -> TaskSnapshot:
        if self.phase not in (TaskPhase.COUNTDOWN, TaskPhase.RUNNING):
            raise RuntimeError("only an active task can be aborted")
        current = self.snapshot(now_s)
        completed_trials = current.current_trial_index or 0
        self._result = self.repository.create_simulated(
            self.protocol,
            participant_id=self.participant_id,
            session_name=self.session_name,
            status="aborted",
            completed_trials=completed_trials,
        )
        self.phase = TaskPhase.ABORTED
        return self.snapshot(now_s)

    def snapshot(self, now_s: float) -> TaskSnapshot:
        if self._started_at is None:
            return TaskSnapshot(
                phase=self.phase,
                countdown_remaining_s=float(self.protocol.countdown_s),
                recording_elapsed_s=0.0,
                recording_remaining_s=self.protocol.recording_duration_s,
                progress=0.0,
                current_trial_index=None,
                current_target_id=None,
                current_frequency_hz=None,
                result=self._result,
            )

        wall_elapsed = max(0.0, now_s - self._started_at)
        if self.phase is TaskPhase.COUNTDOWN and wall_elapsed >= self.protocol.countdown_s:
            self.phase = TaskPhase.RUNNING
            # A suspended UI must not silently skip the acquisition when it wakes.
            self._recording_started_at = now_s

        recording_elapsed = (
            0.0
            if self._recording_started_at is None
            else max(0.0, now_s - self._recording_started_at) * self.simulation_speed
        )
        recording_elapsed = min(recording_elapsed, self.protocol.recording_duration_s)

        if (
            self.phase is TaskPhase.RUNNING
            and recording_elapsed >= self.protocol.recording_duration_s
        ):
            self._result = self.repository.create_simulated(
                self.protocol,
                participant_id=self.participant_id,
                session_name=self.session_name,
            )
            self.phase = TaskPhase.COMPLETED

        countdown_remaining = max(0.0, self.protocol.countdown_s - wall_elapsed)
        progress = recording_elapsed / self.protocol.recording_duration_s
        remaining = max(0.0, self.protocol.recording_duration_s - recording_elapsed)
        current_index: int | None = None
        current_target: str | None = None
        current_frequency: int | None = None
        if self.phase is TaskPhase.RUNNING:
            task_elapsed = max(0.0, recording_elapsed - self.protocol.pre_session_rest_s)
            index = min(
                int(task_elapsed / self.protocol.seconds_per_trial),
                self.protocol.trial_count - 1,
            )
            trial = self.protocol.build_trials()[index]
            current_index = index
            current_target = trial.target_id
            current_frequency = trial.frequency_hz

        return TaskSnapshot(
            phase=self.phase,
            countdown_remaining_s=countdown_remaining,
            recording_elapsed_s=recording_elapsed,
            recording_remaining_s=remaining,
            progress=progress,
            current_trial_index=current_index,
            current_target_id=current_target,
            current_frequency_hz=current_frequency,
            result=self._result,
        )
