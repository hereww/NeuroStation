"""Injected display-only trial runner. Does not discover devices or create windows.

The host must perform hardware/display/risk preflight before calling this runner.
Callbacks must use the same monotonic clock. Journal writes are synchronous: if a
write fails, presentation stops rather than silently losing timing evidence.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Callable, Protocol

from neurostation_four_condition import ConditionFrame, ConditionTrial, FourConditionProtocol


class PresentationCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class DisplayState:
    phase: str
    condition: str
    message: str = ''
    frame: ConditionFrame | None = None


@dataclass(frozen=True)
class FrameSwap:
    generation: int
    monotonic_s: float


class ConditionDisplay(Protocol):
    def submit(self, state: DisplayState) -> int: ...
    def wait_swap(self, generation: int) -> FrameSwap: ...
    def emergency_blank(self) -> None: ...


def present_trial(
    protocol: FourConditionProtocol, trial: ConditionTrial, *, session_id: str,
    display: ConditionDisplay, clock: Callable[[], float],
    wait_until: Callable[[float], None], cancelled: Callable[[], bool],
    journal: Callable[[dict], None],
) -> dict:
    """Present one cue/baseline/stimulus/rest trial and record observed callbacks.

    Callback times are NOT physical light-onset times. EEG associations remain
    explicitly unknown until the capture owner resolves them. Trial completion
    is not signal quality acceptance. No automatic makeup trials are scheduled.
    """
    protocol.require_locked_parameters()
    if not session_id or trial not in protocol.build_trials():
        raise ValueError('A session ID and an unmodified planned trial are required')
    p = protocol.parameters
    identity = protocol.processing_identity
    last_swap = None
    total = round(p['stimulus_s'] * p['refresh_rate_hz'])
    base = {'session_id': session_id, 'block_id': trial.block_id, 'trial_index': trial.index,
            'condition': trial.condition, 'frequency_hz': trial.frequency_hz,
            'processing_identity': identity, 'label_source': trial.label_source,
            'coordinate_space': 'full_selected_display_normalized',
            'eeg_sample_index': None, 'eeg_association': 'not_associated',
            'physical_optical_time_s': None, 'optical_timing_verified': False}

    def check():
        if cancelled():
            raise PresentationCancelled('Operator or host requested abort')

    def write(kind, **values):
        journal({**base, 'kind': kind, **values})

    def event(name, actual, *, source='display_swap_callback', **extra):
        write('event', event=name, monotonic_s=actual, time_source=source, **extra)

    def wait(deadline):
        check()
        wait_until(deadline)
        check()

    def show(state):
        nonlocal last_swap
        check()
        generation = display.submit(state)
        observed = display.wait_swap(generation)
        check()
        if observed.generation != generation:
            raise RuntimeError('Stale or superseded display generation')
        if (not math.isfinite(observed.monotonic_s)
            or (last_swap is not None and observed.monotonic_s <= last_swap)):
            raise RuntimeError('Display callback clock must be finite and strictly increasing')
        last_swap = observed.monotonic_s
        return observed.monotonic_s

    try:
        event('trial_start', clock(), source='software_monotonic_clock')
        cue = show(DisplayState('cue', trial.condition, trial.condition))
        event('cue_onset', cue)
        wait(cue + p['cue_s'])
        baseline = show(DisplayState('baseline', trial.condition, '+'))
        event('baseline_onset', baseline)
        wait(baseline + p['baseline_s'])
        onset = None
        previous = None
        frame_index = 0
        endpoint_presented = False
        presented = 0
        while frame_index < total:
            if onset is not None:
                wait(onset + frame_index / p['refresh_rate_hz'])
            frame = protocol.frame(trial.condition, frame_index)
            actual = show(DisplayState('stimulus', trial.condition, frame=frame))
            if onset is None:
                onset = actual
                event('stimulus_onset', actual, marker_code=trial.onset_marker, x=frame.x, y=frame.y)
            interval = None if previous is None else actual - previous
            write('frame', frame_index=frame_index, scheduled_s=frame.scheduled_s,
                  monotonic_s=actual, actual_relative_s=actual-onset,
                  time_source='display_swap_callback', x=frame.x, y=frame.y,
                  lit=frame.lit, target_visible=frame.target_visible, phase=frame.phase,
                  frame_interval_s=interval,
                  dropped_since_previous=0 if interval is None else max(0, round(interval*p['refresh_rate_hz'])-1),
                  lateness_s=actual-onset-frame.scheduled_s)
            endpoint_presented = frame_index == total - 1
            presented += 1
            previous = actual
            # Never replay missed frames or move backwards after a late swap.
            frame_index = max(frame_index+1, math.floor((actual-onset)*p['refresh_rate_hz'] + 1e-7)+1)
        wait(onset + p['stimulus_s'])
        blank = protocol.frame(trial.condition, total)
        offset = show(DisplayState('stimulus_offset', trial.condition, frame=blank))
        write('frame', frame_index=total, scheduled_s=p['stimulus_s'], monotonic_s=offset,
              actual_relative_s=offset-onset, time_source='display_swap_callback',
              x=blank.x, y=blank.y, lit=False, target_visible=False, phase='stimulus_offset',
              frame_interval_s=offset-previous,
              dropped_since_previous=max(0, round((offset-previous)*p['refresh_rate_hz'])-1),
              lateness_s=offset-onset-p['stimulus_s'])
        event('stimulus_offset', offset, marker_code=trial.offset_marker,
              actual_duration_s=offset-onset, endpoint_frame_presented=endpoint_presented,
              x=blank.x, y=blank.y)
        rest = show(DisplayState('rest', trial.condition, '+'))
        event('rest_onset', rest)
        wait(rest + p['rest_s'])
        event('trial_complete', clock(), source='software_monotonic_clock',
              trial_status='completed', signal_quality='unknown', valid_trial=None)
        return {'status': 'completed', 'presented_frames': presented, 'valid_trial': None,
                'endpoint_frame_presented': endpoint_presented}
    except Exception as error:
        # Hide/blank even when the swap, abort monitor, or evidence writer failed.
        try:
            display.emergency_blank()
        except Exception as blank_error:
            error.add_note(f'Emergency blank failed: {type(blank_error).__name__}; display state unknown')
        try:
            event('trial_abort' if isinstance(error, PresentationCancelled) else 'trial_error',
                  clock(), source='software_monotonic_clock',
                  trial_status='aborted' if isinstance(error, PresentationCancelled) else 'error',
                  reason=type(error).__name__, blank_optical_time_s=None, valid_trial=None)
        except Exception as log_error:
            error.add_note(f'Terminal journal unavailable: {type(log_error).__name__}')
        raise
