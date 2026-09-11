"""Stable UI gateway for acquisition apps and task state."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .dataset import DatasetRecord, DatasetRepository
from .ssvep import SSVEPProtocol
from .task import SSVEPTask, TaskPhase, TaskSnapshot


@dataclass(frozen=True)
class AcquisitionApp:
    app_id: str
    name_zh: str
    name_en: str
    icon: str
    available: bool
    description_zh: str


class WorkstationGateway:
    """The only application service a desktop UI needs for the MVP flow."""

    API_VERSION = 1

    def __init__(
        self,
        *,
        protocol_path: Path,
        dataset_root: Path,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.protocol_path = protocol_path
        self.repository = DatasetRepository(dataset_root)
        self.clock = clock
        self._task: SSVEPTask | None = None

    def list_acquisition_apps(self) -> tuple[AcquisitionApp, ...]:
        return (
            AcquisitionApp(
                app_id="ssvep",
                name_zh="SSVEP 屏幕闪烁采集",
                name_en="SSVEP visual flicker",
                icon="grid-2x2",
                available=True,
                description_zh="四目标视觉稳态诱发电位采集",
            ),
            AcquisitionApp(
                app_id="resting_state",
                name_zh="静息态采集",
                name_en="Resting-state EEG",
                icon="brain",
                available=False,
                description_zh="睁眼/闭眼静息态协议，尚未启用",
            ),
            AcquisitionApp(
                app_id="p300",
                name_zh="P300 采集",
                name_en="P300 oddball",
                icon="sparkles",
                available=False,
                description_zh="Oddball 事件相关电位协议，尚未启用",
            ),
            AcquisitionApp(
                app_id="custom",
                name_zh="自定义协议",
                name_en="Custom protocol",
                icon="settings-2",
                available=False,
                description_zh="可视化协议编辑器，尚未启用",
            ),
        )

    @property
    def active_protocol(self) -> SSVEPProtocol | None:
        return self._task.protocol if self._task is not None else None

    def ssvep_details(
        self,
        *,
        repetitions: int | None = None,
        stimulus_s: float | None = None,
        rest_s: float | None = None,
        frequencies_hz: tuple[int, ...] | None = None,
    ) -> dict:
        protocol = SSVEPProtocol.load(self.protocol_path).with_runtime_parameters(
            repetitions=repetitions,
            stimulus_s=stimulus_s,
            rest_s=rest_s,
            frequencies_hz=frequencies_hz,
        )
        return {
            "protocol_id": protocol.protocol_id,
            "countdown_s": protocol.countdown_s,
            "refresh_rate_hz": protocol.refresh_rate_hz,
            "sampling_rate_hz": protocol.sampling_rate_hz,
            "channel_count": protocol.channel_count,
            "frequencies_hz": protocol.frequencies_hz,
            "repetitions": protocol.repetitions,
            "trial_count": protocol.trial_count,
            "stimulus_s": protocol.stimulus_s,
            "pre_trial_rest_s": protocol.pre_trial_rest_s,
            "post_trial_rest_s": protocol.post_trial_rest_s,
            "recording_duration_s": protocol.recording_duration_s,
            "total_duration_s": protocol.total_duration_s,
            "expected_samples_per_channel": protocol.expected_samples_per_channel,
            "expected_event_count": protocol.expected_event_count,
            "dataset_root": str(self.repository.root),
        }

    def start_ssvep(
        self,
        *,
        participant_id: str,
        session_name: str,
        repetitions: int | None = None,
        stimulus_s: float | None = None,
        rest_s: float | None = None,
        frequencies_hz: tuple[int, ...] | None = None,
        simulation_speed: float = 1.0,
    ) -> TaskSnapshot:
        if self._task and self._task.phase in (TaskPhase.COUNTDOWN, TaskPhase.RUNNING):
            raise RuntimeError("an acquisition task is already active")
        protocol = SSVEPProtocol.load(self.protocol_path).with_runtime_parameters(
            repetitions=repetitions,
            stimulus_s=stimulus_s,
            rest_s=rest_s,
            frequencies_hz=frequencies_hz,
        )
        self._task = SSVEPTask(
            protocol,
            self.repository,
            participant_id=participant_id,
            session_name=session_name,
            simulation_speed=simulation_speed,
        )
        return self._task.start(self.clock())

    def poll_task(self) -> TaskSnapshot:
        if self._task is None:
            raise RuntimeError("no acquisition task has been started")
        return self._task.snapshot(self.clock())

    def abort_task(self) -> TaskSnapshot:
        if self._task is None:
            raise RuntimeError("no acquisition task has been started")
        return self._task.abort(self.clock())

    def list_datasets(self) -> list[dict]:
        return [self._serialize_record(record) for record in self.repository.list_records()]

    def import_openbci_recordings(self, source_root: Path):
        return self.repository.import_openbci_recordings(source_root)

    def refresh_datasets(self) -> None:
        return None

    @staticmethod
    def _serialize_record(record: DatasetRecord) -> dict:
        value = asdict(record)
        value["output_dir"] = str(record.output_dir)
        return value
