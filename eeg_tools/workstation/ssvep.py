"""Versioned SSVEP protocol model used by the workstation UI and runner."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


class SSVEPProtocolError(ValueError):
    """Raised when an SSVEP protocol cannot be executed safely."""


def _number(value: Any, name: str, *, minimum: float, allow_equal: bool = True) -> float:
    valid_type = isinstance(value, (int, float)) and not isinstance(value, bool)
    valid_range = valid_type and (
        value >= minimum if allow_equal else value > minimum
    )
    if not valid_type or not valid_range:
        relation = "at least" if allow_equal else "greater than"
        raise SSVEPProtocolError(f"{name} must be {relation} {minimum}")
    return float(value)


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SSVEPProtocolError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class SSVEPTrial:
    index: int
    target_id: str
    frequency_hz: int
    onset_marker: int
    offset_marker: int


@dataclass(frozen=True)
class SSVEPProtocol:
    protocol_id: str
    countdown_s: int
    refresh_rate_hz: int
    sampling_rate_hz: int
    channel_count: int
    targets: tuple[tuple[str, int], ...]
    pre_session_rest_s: float
    pre_trial_rest_s: float
    stimulus_s: float
    post_trial_rest_s: float
    repetitions: int
    random_seed: int
    session_start_marker: int
    acquisition_start_marker: int
    session_end_marker: int
    abort_marker: int
    onset_marker_base: int
    offset_marker_base: int
    gaze_marker_base: int
    source: Path | None = None

    @classmethod
    def load(cls, path: Path) -> "SSVEPProtocol":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise SSVEPProtocolError(f"Protocol file does not exist: {path}") from exc
        except json.JSONDecodeError as exc:
            raise SSVEPProtocolError(
                f"Invalid protocol JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
            ) from exc
        if not isinstance(value, dict):
            raise SSVEPProtocolError("Protocol root must be an object")
        return cls.from_dict(value, source=path)

    @classmethod
    def from_dict(
        cls, value: dict[str, Any], *, source: Path | None = None
    ) -> "SSVEPProtocol":
        if value.get("schema_version") != 2:
            raise SSVEPProtocolError("schema_version must be 2")
        display = value.get("display")
        hardware = value.get("hardware")
        trial = value.get("trial")
        events = value.get("events")
        if not all(isinstance(item, dict) for item in (display, hardware, trial, events)):
            raise SSVEPProtocolError("display, hardware, trial and events must be objects")

        protocol_id = value.get("protocol_id")
        if not isinstance(protocol_id, str) or not protocol_id.strip():
            raise SSVEPProtocolError("protocol_id must be a non-empty string")
        countdown_s = _positive_integer(value.get("countdown_s"), "countdown_s")
        refresh_rate_hz = _positive_integer(
            display.get("refresh_rate_hz"), "display.refresh_rate_hz"
        )
        sampling_rate_hz = _positive_integer(
            hardware.get("sampling_rate_hz"), "hardware.sampling_rate_hz"
        )
        channel_count = _positive_integer(
            hardware.get("channel_count"), "hardware.channel_count"
        )

        target_values = value.get("targets")
        if not isinstance(target_values, list) or not target_values:
            raise SSVEPProtocolError("targets must be a non-empty array")
        targets: list[tuple[str, int]] = []
        for index, target in enumerate(target_values):
            if not isinstance(target, dict):
                raise SSVEPProtocolError(f"targets[{index}] must be an object")
            target_id = target.get("id")
            frequency = target.get("frequency_hz")
            if not isinstance(target_id, str) or not target_id:
                raise SSVEPProtocolError(f"targets[{index}].id must be a non-empty string")
            frequency = _positive_integer(frequency, f"targets[{index}].frequency_hz")
            if refresh_rate_hz % frequency:
                raise SSVEPProtocolError(
                    f"Target frequency {frequency} Hz does not divide refresh rate "
                    f"{refresh_rate_hz} Hz"
                )
            targets.append((target_id, frequency))
        if len({item[0] for item in targets}) != len(targets):
            raise SSVEPProtocolError("target ids must be unique")
        if len({item[1] for item in targets}) != len(targets):
            raise SSVEPProtocolError("target frequencies must be unique")

        repetitions = _positive_integer(trial.get("repetitions"), "trial.repetitions")
        random_seed = trial.get("random_seed")
        if isinstance(random_seed, bool) or not isinstance(random_seed, int):
            raise SSVEPProtocolError("trial.random_seed must be an integer")

        return cls(
            protocol_id=protocol_id,
            countdown_s=countdown_s,
            refresh_rate_hz=refresh_rate_hz,
            sampling_rate_hz=sampling_rate_hz,
            channel_count=channel_count,
            targets=tuple(targets),
            pre_session_rest_s=_number(
                trial.get("pre_session_rest_s"), "trial.pre_session_rest_s", minimum=0
            ),
            pre_trial_rest_s=_number(
                trial.get("pre_trial_rest_s"), "trial.pre_trial_rest_s", minimum=0
            ),
            stimulus_s=_number(
                trial.get("stimulus_s"),
                "trial.stimulus_s",
                minimum=0,
                allow_equal=False,
            ),
            post_trial_rest_s=_number(
                trial.get("post_trial_rest_s"), "trial.post_trial_rest_s", minimum=0
            ),
            repetitions=repetitions,
            random_seed=random_seed,
            session_start_marker=_positive_integer(
                events.get("session_start"), "events.session_start"
            ),
            acquisition_start_marker=_positive_integer(
                events.get("acquisition_start"), "events.acquisition_start"
            ),
            session_end_marker=_positive_integer(
                events.get("session_end"), "events.session_end"
            ),
            abort_marker=_positive_integer(events.get("abort"), "events.abort"),
            onset_marker_base=_positive_integer(
                events.get("stimulus_onset_base"), "events.stimulus_onset_base"
            ),
            offset_marker_base=_positive_integer(
                events.get("stimulus_offset_base"), "events.stimulus_offset_base"
            ),
            gaze_marker_base=_positive_integer(
                events.get("gaze_target_base"), "events.gaze_target_base"
            ),
            source=source,
        )

    @property
    def frequencies_hz(self) -> tuple[int, ...]:
        return tuple(frequency for _, frequency in self.targets)

    @property
    def trial_count(self) -> int:
        return len(self.targets) * self.repetitions

    @property
    def seconds_per_trial(self) -> float:
        return self.pre_trial_rest_s + self.stimulus_s + self.post_trial_rest_s

    @property
    def recording_duration_s(self) -> float:
        # The inter-trial rest is intentionally omitted after the final stimulus.
        return (
            self.pre_session_rest_s
            + self.trial_count * (self.pre_trial_rest_s + self.stimulus_s)
            + max(0, self.trial_count - 1) * self.post_trial_rest_s
        )

    @property
    def total_duration_s(self) -> float:
        return self.countdown_s + self.recording_duration_s

    @property
    def expected_samples_per_channel(self) -> int:
        return round(self.recording_duration_s * self.sampling_rate_hz)

    @property
    def expected_event_count(self) -> int:
        return 3 + self.trial_count * 2

    def build_trials(self) -> tuple[SSVEPTrial, ...]:
        ordered = list(self.targets) * self.repetitions
        random.Random(self.random_seed).shuffle(ordered)
        return tuple(
            SSVEPTrial(
                index=index,
                target_id=target_id,
                frequency_hz=frequency,
                onset_marker=self.onset_marker_base + frequency,
                offset_marker=self.offset_marker_base + frequency,
            )
            for index, (target_id, frequency) in enumerate(ordered)
        )

    def with_runtime_parameters(
        self,
        *,
        repetitions: int | None = None,
        stimulus_s: float | None = None,
        rest_s: float | None = None,
        frequencies_hz: tuple[int, ...] | None = None,
    ) -> "SSVEPProtocol":
        next_repetitions = (
            self.repetitions
            if repetitions is None
            else _positive_integer(repetitions, "repetitions")
        )
        next_stimulus = (
            self.stimulus_s
            if stimulus_s is None
            else _number(stimulus_s, "stimulus_s", minimum=0, allow_equal=False)
        )
        next_rest = (
            self.post_trial_rest_s
            if rest_s is None
            else _number(rest_s, "rest_s", minimum=0)
        )
        next_targets = self.targets
        if frequencies_hz is not None:
            if len(frequencies_hz) != len(self.targets):
                raise SSVEPProtocolError(
                    "frequencies_hz must contain one value for every target"
                )
            validated: list[int] = []
            for index, frequency in enumerate(frequencies_hz):
                value = _positive_integer(frequency, f"frequencies_hz[{index}]")
                if self.refresh_rate_hz % value:
                    raise SSVEPProtocolError(
                        f"Target frequency {value} Hz does not divide refresh rate "
                        f"{self.refresh_rate_hz} Hz"
                    )
                validated.append(value)
            if len(set(validated)) != len(validated):
                raise SSVEPProtocolError("target frequencies must be unique")
            next_targets = tuple(
                (target_id, frequency)
                for (target_id, _), frequency in zip(self.targets, validated)
            )
        return replace(
            self,
            repetitions=next_repetitions,
            stimulus_s=next_stimulus,
            post_trial_rest_s=next_rest,
            targets=next_targets,
        )
