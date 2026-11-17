"""Offline four-condition planning; never opens a device or stimulus window.

Schema 3 is deliberately separate from the schema-2 frequency-class protocol.
A formal design is not a parameter lock, a hardware check, or a quality verdict.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from neurostation_protocol_errors import SSVEPProtocolError

CONDITIONS = ("L_FIXED", "R_FIXED", "L_TO_R", "R_TO_L")
PARAMETERS = (
    "refresh_rate_hz", "frequency_hz", "duty_cycle", "left_x", "right_x", "y",
    "target_width", "target_height", "shape", "motion_mode", "gaze_instruction",
    "background_level", "on_level", "off_level", "cue_s", "baseline_s",
    "stimulus_s", "rest_s", "trials_per_block", "blocks_per_condition",
    "block_rest_s", "condition_rest_s", "makeup_policy",
)
IMPLEMENTATION = "four-condition-frame-plan/1"


def parameter_digest(parameters: dict[str, Any]) -> str:
    """Bind confirmation to every parameter, not only frequency or duration."""
    return hashlib.sha256(json.dumps(parameters, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class ConditionTrial:
    index: int
    condition: str
    block_id: int
    index_in_block: int
    frequency_hz: float
    onset_marker: int
    offset_marker: int
    cue_onset_s: float
    baseline_onset_s: float
    stimulus_onset_s: float
    stimulus_offset_s: float
    rest_end_s: float
    break_after_s: float
    label_source: str = "planned_condition_not_verified_gaze"


@dataclass(frozen=True)
class ConditionFrame:
    frame_index: int
    scheduled_s: float
    x: float
    y: float
    lit: bool
    target_visible: bool
    phase: str


@dataclass(frozen=True)
class FourConditionProtocol:
    # Immutable canonical JSON also isolates caller mutations after validation.
    canonical_json: str

    @classmethod
    def load(cls, path: Path) -> "FourConditionProtocol":
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            raise SSVEPProtocolError(f"Invalid four-condition protocol {path}: {exc}") from exc

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FourConditionProtocol":
        if not isinstance(value, dict):
            raise SSVEPProtocolError("Protocol root must be an object")
        if value.get("schema_version") != 3 or value.get("kind") != "ssvep_four_condition":
            raise SSVEPProtocolError("Expected schema 3 ssvep_four_condition")
        if value.get("status") != "formal":
            raise SSVEPProtocolError("Four-condition design status must be formal")
        if value.get("protocol_id") != "ssvep_four_condition_v1":
            raise SSVEPProtocolError("Unsupported four-condition protocol_id")
        if value.get("condition_order") != list(CONDITIONS):
            raise SSVEPProtocolError("condition_order must retain the confirmed fixed order")
        p = value.get("parameters")
        if not isinstance(p, dict) or set(p) != set(PARAMETERS):
            raise SSVEPProtocolError("parameters must contain exactly the versioned parameter keys")
        enums = {
            "shape": ("rectangle", "circle"), "motion_mode": ("smooth_horizontal",),
            "gaze_instruction": ("central_fixation_attention_follow",),
            "makeup_policy": ("none_preserve_all_trials",),
        }
        for key, x in p.items():
            if x is None:
                continue
            if key in enums:
                if x not in enums[key]:
                    raise SSVEPProtocolError(f"Unsupported {key}; requires a separate implementation/version")
            elif isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
                raise SSVEPProtocolError(f"{key} must be a finite number")
            elif key in ("refresh_rate_hz", "trials_per_block", "blocks_per_condition"):
                if not isinstance(x, int) or x <= 0:
                    raise SSVEPProtocolError(f"{key} must be a positive integer")
            elif key in ("left_x", "right_x", "y", "background_level", "on_level", "off_level"):
                if not 0 <= x <= 1:
                    raise SSVEPProtocolError(f"{key} must be within [0, 1]")
            elif key in ("target_width", "target_height", "duty_cycle"):
                if not 0 < x < 1:
                    raise SSVEPProtocolError(f"{key} must be within (0, 1)")
            elif x <= 0:
                raise SSVEPProtocolError(f"{key} must be positive")
        def known(*keys):
            return all(p[k] is not None for k in keys)
        if known("left_x", "right_x") and p["left_x"] >= p["right_x"]:
            raise SSVEPProtocolError("left_x must precede right_x")
        if known("left_x", "right_x", "target_width"):
            half = p["target_width"] / 2
            if p["left_x"] < half or p["right_x"] > 1 - half:
                raise SSVEPProtocolError("Target extends outside horizontal viewport")
        if known("y", "target_height"):
            half = p["target_height"] / 2
            if not half <= p["y"] <= 1 - half:
                raise SSVEPProtocolError("Target extends outside vertical viewport")
        if known("on_level", "off_level") and p["on_level"] <= p["off_level"]:
            raise SSVEPProtocolError("on_level must exceed off_level")
        if known("frequency_hz", "refresh_rate_hz"):
            cycle = p["refresh_rate_hz"] / p["frequency_hz"]
            if not math.isfinite(cycle) or cycle < 2 or not math.isclose(cycle, round(cycle), abs_tol=1e-9, rel_tol=0):
                raise SSVEPProtocolError("frequency must produce an integral frame cycle >= 2")
            if known("duty_cycle"):
                on = cycle * p["duty_cycle"]
                if not 1 <= on <= cycle - 1 or not math.isclose(on, round(on), abs_tol=1e-9, rel_tol=0):
                    raise SSVEPProtocolError("duty_cycle must produce integral on/off frame counts")
        if known("stimulus_s", "refresh_rate_hz"):
            frames = p["stimulus_s"] * p["refresh_rate_hz"]
            if not math.isfinite(frames) or frames < 2 or not math.isclose(frames, round(frames), abs_tol=1e-9, rel_tol=0):
                raise SSVEPProtocolError("stimulus_s must produce an integral frame count >= 2")
        if known("trials_per_block", "blocks_per_condition"):
            count = 4 * p["trials_per_block"] * p["blocks_per_condition"]
            if count > 10000:
                raise SSVEPProtocolError("Offline planner limit is 10000 trials")
            times = ("cue_s", "baseline_s", "stimulus_s", "rest_s", "block_rest_s", "condition_rest_s")
            if known(*times) and not math.isfinite(count * sum(p[k] for k in times)):
                raise SSVEPProtocolError("Planned duration exceeds finite numeric range")
        confirmation = value.get("parameter_confirmation")
        if confirmation is not None:
            if (not isinstance(confirmation, dict)
                or not isinstance(confirmation.get("confirmed_by"), str)
                or not confirmation["confirmed_by"].strip()
                or any(x is None for x in p.values())
                or confirmation.get("parameter_sha256") != parameter_digest(p)):
                raise SSVEPProtocolError("Parameter confirmation is incomplete or stale")
        try:
            return cls(json.dumps(value, sort_keys=True, allow_nan=False))
        except (ValueError, TypeError) as exc:
            raise SSVEPProtocolError("Protocol must contain finite JSON values") from exc

    @property
    def parameters(self) -> dict[str, Any]:
        return json.loads(self.canonical_json)["parameters"]

    @property
    def pending(self) -> tuple[str, ...]:
        result = [k for k, v in self.parameters.items() if v is None]
        if json.loads(self.canonical_json).get("parameter_confirmation") is None:
            result.append("parameter_confirmation")
        return tuple(result)

    @property
    def processing_identity(self) -> str:
        return hashlib.sha256((IMPLEMENTATION + self.canonical_json).encode()).hexdigest()

    def require_locked_parameters(self) -> None:
        if self.pending:
            raise SSVEPProtocolError("Parameters not locked: " + ", ".join(self.pending))

    def build_trials(self) -> tuple[ConditionTrial, ...]:
        self.require_locked_parameters()
        p = self.parameters
        result = []
        clock = 0.0
        for condition_index, condition in enumerate(CONDITIONS):
            for block in range(p["blocks_per_condition"]):
                for within in range(p["trials_per_block"]):
                    baseline = clock + p["cue_s"]
                    onset = baseline + p["baseline_s"]
                    offset = onset + p["stimulus_s"]
                    end = offset + p["rest_s"]
                    pause = 0.0
                    if within == p["trials_per_block"] - 1:
                        if block < p["blocks_per_condition"] - 1:
                            pause = p["block_rest_s"]
                        elif condition_index < len(CONDITIONS) - 1:
                            pause = p["condition_rest_s"]
                    result.append(ConditionTrial(
                        len(result), condition, condition_index * p["blocks_per_condition"] + block,
                        within, p["frequency_hz"], 101 + condition_index, 201 + condition_index,
                        clock, baseline, onset, offset, end, pause))
                    clock = end + pause
        return tuple(result)

    def frame(self, condition: str, frame_index: int) -> ConditionFrame:
        """Planned coordinates, not observed flips. Terminal frame blanks at endpoint.

        Frame indices may skip forward after a delayed display callback; positions
        remain monotonic. Reset to the start is allowed only while target is hidden.
        """
        self.require_locked_parameters()
        if condition not in CONDITIONS:
            raise SSVEPProtocolError("Unknown condition")
        p = self.parameters
        total = round(p["stimulus_s"] * p["refresh_rate_hz"])
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or not 0 <= frame_index <= total:
            raise SSVEPProtocolError("frame_index must be within the trial including terminal blank")
        progress = min(frame_index / (total - 1), 1.0)
        left, right = p["left_x"], p["right_x"]
        x = {"L_FIXED": left, "R_FIXED": right,
             "L_TO_R": left + (right - left) * progress,
             "R_TO_L": right - (right - left) * progress}[condition]
        cycle = round(p["refresh_rate_hz"] / p["frequency_hz"])
        terminal = frame_index == total
        lit = not terminal and frame_index % cycle < round(cycle * p["duty_cycle"])
        return ConditionFrame(frame_index, frame_index / p["refresh_rate_hz"], x, p["y"],
                              lit, not terminal, "stimulus_offset" if terminal else "stimulus")
