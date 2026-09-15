"""Removed legacy simulated task API.

Real acquisition is owned by ``AcquisitionProcessGateway``. These names stay
importable for callers that have not migrated yet, but they cannot create a
task or write a simulated session.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


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
    result: Any


class SSVEPTask:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("validation.real_hardware_only")
