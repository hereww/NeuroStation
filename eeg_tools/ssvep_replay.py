"""Continuous SSVEP replay and post-hoc, interval-level scoring of Cyton sessions."""
from __future__ import annotations

import csv
import json
import shutil
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np

from .continuous_ssvep import (ContinuousDetector, DecisionGate, FREQUENCIES, FS, Gate,
                               HOP, WINDOW, METHOD, MAX_CLOCK_STEP_S, SPECTRAL_RATIO, evidence_settings, sample_quality)
from .session_files import sha256


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_tsv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def read_numeric(path, names):
    with path.open(encoding='utf-8-sig') as f:
        header = f.readline().rstrip('\r\n').split('\t')
        return np.loadtxt(f, delimiter='\t', usecols=[header.index(n) for n in names], ndmin=2)


@dataclass
class Signal:
    index: np.ndarray
    packet: np.ndarray
    clock: np.ndarray
    eeg: np.ndarray

    @classmethod
    def load(cls, root):
        data = read_numeric(root/'raw_brainflow.tsv', ['sample_index', 'package_num', 'timestamp_s',
                            *(f'eeg_ch{i}' for i in range(1, 9))])
        if not len(data) or not np.all(np.isfinite(data[:, 0]) & (data[:, 0] == np.floor(data[:, 0]))):
            raise ValueError('Missing or invalid sample coordinates')
        return cls(data[:, 0].astype(np.int64), data[:, 1], data[:, 2], data[:, 3:])

    def replay(self, config, block_size):
        if block_size <= 0:
            raise ValueError('Chunk size must be positive')
        detector = ContinuousDetector(config)
        for start in range(0, len(self.index), block_size):
            s = slice(start, start + block_size)
            yield from detector.push(self.index[s], self.packet[s], self.clock[s], self.eeg[s])


@dataclass
class Interval:
    trial: int
    kind: str
    start: int | None
    stop: int | None
    expected_hz: int | None
    reason: str = ''
    samples: int = 0
    transport_breaks: int = 0
    max_clock_step_ms: float | None = None


def integer(value):
    try:
        number = float(value)
        return int(number) if number.is_integer() else None
    except (ValueError, TypeError):
        return None


def verified_event(event, signal, markers, targets, marker_base):
    if event is None:
        return 'missing_or_duplicate_event'
    pos = integer(event['sample_index'])
    row = np.searchsorted(signal.index, pos) if pos is not None else len(markers)
    if row >= len(markers) or signal.index[row] != pos or markers[row] != integer(event['marker_code']):
        return 'unaligned_raw_marker'
    hz = integer(event['frequency_hz'])
    if hz not in FREQUENCIES or targets.get(event['presented_target_id']) != hz or integer(event['marker_code']) != marker_base + hz:
        return 'unverified_target'
    return ''


def intervals(root, signal):
    protocol = read_json(root/'protocol.json')
    session = read_json(root/'session.json')
    targets = dict(protocol['targets'])
    markers = read_numeric(root/'raw_brainflow.tsv', ['marker'])[:, 0]
    events = defaultdict(lambda: defaultdict(list))
    for e in read_tsv(root/'events.tsv'):
        if e['event_name'] in ('stimulus_onset', 'stimulus_offset'):
            trial = integer(e['trial_index'])
            if trial is None or trial < 0:
                raise ValueError('Unusable trial_index on stimulus event')
            events[trial][e['event_name']].append(e)
    frames = defaultdict(list)
    observed_trials = set(events)
    if (root/'frame_timing.tsv').is_file():
        for r in read_tsv(root/'frame_timing.tsv'):
            trial = integer(r['trial_index'])
            if trial is None or trial < 0:
                continue
            observed_trials.add(trial)
            # In older recordings a dark flicker frame has no offset semantics.
            if r.get('presentation_phase') == 'offset':
                frames[trial].append(r.get('lit'))
    # Aborted sessions can have an observed partial trial after their completed prefix.
    completed = integer(session.get('completed_trials')) or 0
    if session.get('status') == 'completed':
        planned = integer(session.get('trial_count'))
        if planned is None:
            planned = (integer(protocol.get('repetitions')) or 0) * len(targets)
        completed = max(completed, planned)
    observed_trials.update(range(completed))
    display = session.get('display')
    headless = session.get('headless') is True or (isinstance(display, dict) and display.get('mode') == 'headless')
    issues = sample_quality(signal.index, signal.packet, signal.clock, signal.eeg)
    delta = np.diff(signal.clock, prepend=signal.clock[0])
    result, trials = [], []
    for trial in sorted(observed_trials):
        pair = events[trial]
        on, off = (rows[0] if len(rows) == 1 else None for rows in
                   (pair['stimulus_onset'], pair['stimulus_offset']))
        start, stop = (integer(e['sample_index']) if e else None for e in (on, off))
        reasons = [verified_event(e, signal, markers, targets, protocol[key]) for e, key in
                   ((on, 'onset_marker_base'), (off, 'offset_marker_base'))]
        hz = integer(on['frequency_hz']) if on else None
        if on and off and (on['presented_target_id'] != off['presented_target_id'] or hz != integer(off['frequency_hz'])):
            reasons.append('mismatched_target')
        reason = 'headless_no_presentation' if headless else '|'.join(dict.fromkeys(filter(None, reasons)))
        trials.append(Interval(trial, 'stimulus', start, stop, hz, reason))
    for prev, nxt in zip(trials, trials[1:]):
        if prev.stop is not None and nxt.start is not None and prev.stop > nxt.start:
            prev.reason = nxt.reason = 'overlapping_or_unordered_trials'
    for t in trials:
        if t.start is None or t.stop is None or t.stop <= t.start:
            t.reason = t.reason or 'invalid_interval'
    for i, t in enumerate(trials):
        result.append(t)
        if i + 1 < len(trials) and float(protocol.get('post_trial_rest_s', 0)) > 0:
            nxt = trials[i+1]
            verified = (not t.reason and not nxt.reason and nxt.trial == t.trial + 1
                        and frames[t.trial] == ['0'])
            result.append(Interval(t.trial, 'rest', t.stop, nxt.start, None,
                                   '' if verified else 'unverified_rest_boundaries_or_offset_frame'))
    marker_codes = {protocol[key] + f for key in ('onset_marker_base', 'offset_marker_base') for f in FREQUENCIES}
    stimulus_mark = np.isin(markers, list(marker_codes))
    for seg in result:
        if seg.start is None or seg.stop is None or seg.stop <= seg.start:
            seg.reason = seg.reason or 'invalid_interval'
            continue
        lo, hi = np.searchsorted(signal.index, [seg.start, seg.stop])
        seg.samples = int(hi-lo)
        seg.transport_breaks = int(np.count_nonzero(issues[lo:hi]))
        if hi > lo:
            maximum = float(np.max(delta[lo:hi])) * 1000
            seg.max_clock_step_ms = round(maximum, 3) if np.isfinite(maximum) else None
        if seg.samples != seg.stop - seg.start or seg.transport_breaks:
            seg.reason = seg.reason or 'incomplete_or_discontinuous_eeg'
        if np.any(stimulus_mark[lo+1:hi]):
            seg.reason = seg.reason or 'unexpected_stimulus_marker'
    return result


def score(segments, predictions):
    ends = [d.window.end_sample for d in predictions]
    rows = []
    for seg in segments:
        out = dict(**asdict(seg), result='excluded', decisions=0, unavailable=0, no_target=0,
                   first_hz=None, first_sample=None, first_correct_sample=None,
                   onset_to_first_target_s=None, onset_to_correct_s=None, wrong_to_correct_s=None)
        rows.append(out)
        if seg.start is None or seg.stop is None:
            continue
        part = predictions[bisect_left(ends, seg.start):bisect_left(ends, seg.stop)]
        out.update(decisions=len(part), unavailable=sum(d.state == 'unavailable' for d in part),
                   no_target=sum(d.state == 'no_target' for d in part))
        expected = seg.stop // HOP - seg.start // HOP
        if len(part) != expected or not part or out['unavailable']:
            issues = sorted({d.window.quality for d in part if d.state == 'unavailable'})
            out['reason'] = seg.reason or '|'.join(issues) or 'incomplete_decision_coverage'
        if out['reason']:
            continue
        first = next((d for d in part if d.hz is not None), None)
        if first is not None:
            out.update(first_hz=first.hz, first_sample=first.window.end_sample,
                       onset_to_first_target_s=(first.window.end_sample - seg.start) / FS)
        if seg.kind == 'rest':
            out['result'] = 'false_alarm' if first else 'quiet'
            continue
        out['result'] = 'miss' if not first else 'correct' if first.hz == seg.expected_hz else 'wrong'
        hit = next((d for d in part if d.hz == seg.expected_hz), None)
        if hit:
            out.update(first_correct_sample=hit.window.end_sample, onset_to_correct_s=(hit.window.end_sample-seg.start)/FS)
            if first.hz != hit.hz:
                out['wrong_to_correct_s'] = (hit.window.end_sample - first.window.end_sample)/FS
    return rows


def totals(rows):
    c = Counter(r['result'] for r in rows)
    stim = sum(c[k] for k in ('correct', 'wrong', 'miss'))
    rest = c['quiet'] + c['false_alarm']
    return dict(correct=c['correct'], wrong=c['wrong'], miss=c['miss'], stimulus_n=stim,
                quiet=c['quiet'], false_alarm=c['false_alarm'], rest_n=rest,
                stimulus_total=sum(r['kind'] == 'stimulus' for r in rows), rest_total=sum(r['kind'] == 'rest' for r in rows),
                recovered=sum(r['wrong_to_correct_s'] is not None for r in rows),
                recognition=c['correct']/stim if stim else None, false_alarm_rate=c['false_alarm']/rest if rest else None)


def calibrate(root, chunk, signal=None):
    if signal is None:
        signal = Signal.load(root)
    windows = [d.window for d in signal.replay(Gate(), chunk)]
    segments = intervals(root, signal)
    grid = []
    for floor, margin, n in product((.20, .25, .30, .35, .40, .45, .50), (.02, .04, .08), (2, 3)):
        config = Gate(floor, margin, n)
        gate = DecisionGate(config)
        metrics = totals(score(segments, [gate.apply(w) for w in windows]))
        if metrics['recognition'] is None:
            raise ValueError('Calibration session has no scorable stimulus trials')
        fa = metrics['false_alarm_rate']
        utility = metrics['recognition'] if fa is None else (metrics['recognition'] + 1 - fa) / 2
        grid.append(dict(gate=asdict(config), utility=utility, metrics=metrics))
    # Equal-utility ties prefer recognition, then keep the first candidate in grid order.
    best = max(grid, key=lambda r: (r['utility'], r['metrics']['recognition']))
    return Gate(**best['gate']), grid


def write_table(path, rows, fields=None):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def prediction_row(d):
    return dict(end_sample=d.window.end_sample, quality=d.window.quality, state=d.state,
                frequency_hz=d.hz, **{f'score_{hz}': s for hz, s in zip(FREQUENCIES, d.window.scores)},
                **{f'spectral_support_{hz}': int(ok) for hz, ok in zip(FREQUENCIES, d.window.spectral_support)})


PRED_FIELDS = ['end_sample', 'quality', 'state', 'frequency_hz', *(f'score_{hz}' for hz in FREQUENCIES),
               *(f'spectral_support_{hz}' for hz in FREQUENCIES)]


def validate_split(cal, test):
    early, late = [read_json(p/'session.json') for p in (cal, test)]
    early_start = datetime.fromisoformat(early['started_at'])
    late_start = datetime.fromisoformat(late['started_at'])
    if (cal.resolve() == test.resolve() or early['session_id'] == late['session_id']
            or early_start >= late_start
            or not early.get('participant_id') or early['participant_id'] != late['participant_id']):
        raise ValueError('Calibration requires an earlier, separate session of the same participant')
    if early.get('ended_at') and not early_start <= datetime.fromisoformat(early['ended_at']) <= late_start:
        raise ValueError('Calibration must finish before the evaluation session starts')
    return early, late


def validate_replay_protocol(root):
    protocol = read_json(root / 'protocol.json')
    if (protocol['sampling_rate_hz'] != FS or protocol['channel_count'] != 8 or
            sorted(f for _, f in protocol['targets']) != list(FREQUENCIES)):
        raise ValueError('This detector requires native 250 Hz, 8-channel, four-frequency sessions')


def _prepare_replay(test, calibration, output, chunk, *, save_grid):
    validate_replay_protocol(test)
    signal = Signal.load(test)
    early = {}
    personal_source = None
    if calibration is None:
        late = read_json(test / 'session.json')
    else:
        validate_replay_protocol(calibration)
        early, late = validate_split(calibration, test)
        history = Signal.load(calibration)
        # Compare measurements independently of clocks, markers and TSV formatting.
        if np.array_equal(history.eeg, signal.eeg, equal_nan=True):
            raise ValueError('Duplicated EEG samples in calibration and evaluation recordings')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Choose a new or empty output directory to preserve existing runs')
    output.mkdir(parents=True, exist_ok=True)
    configs = {'baseline': Gate()}
    if calibration is not None:
        configs['personal'], grid = calibrate(calibration, chunk, history)
        del history
        if save_grid:
            write_json(output/'calibration_grid.json', grid)
        personal_source = ('earlier session only, grid objective balanced trial recognition and rest specificity'
                           if grid[0]['metrics']['rest_n'] else 'earlier session only, grid objective trial recognition')
    lock = dict(method=METHOD, sampling_rate_hz=FS, frequencies_hz=FREQUENCIES, window=WINDOW, hop=HOP,
                max_clock_step_s=MAX_CLOCK_STEP_S, spectral_power_ratio=SPECTRAL_RATIO,
                evidence=evidence_settings(), gates={k: asdict(v) for k,v in configs.items()},
                participant_id=late.get('participant_id'), calibration_session=early.get('session_id'), test_session=late['session_id'],
                baseline_source='fixed engineering defaults, no participant fitting',
                personal_source=personal_source,
                source_sha256={name:sha256(Path(__file__).with_name(name)) for name in ('continuous_ssvep.py','ssvep_replay.py')})
    write_json(output/'frozen_models.json', lock)
    return signal, configs, lock, early, late


def process_session(test, calibration, output, chunk=1024):
    """Export continuous decisions with optional, verbatim presentation logs."""
    test, output = Path(test), Path(output)
    calibration = Path(calibration) if calibration is not None else None
    signal, configs, lock, _, session = _prepare_replay(test, calibration, output, chunk, save_grid=False)
    files = {}
    for name, config in configs.items():
        path = output / f'{name}_predictions.tsv'
        write_table(path, map(prediction_row, signal.replay(config, chunk)), PRED_FIELDS)
        files[name] = path.name

    shutil.copyfile(test / 'protocol.json', output / 'protocol.json')
    annotations = []
    for name in ('events.tsv', 'frame_timing.tsv'):
        if (test / name).is_file():
            shutil.copyfile(test / name, output / name)
            annotations.append(name)
    manifest = dict(schema_version=1, kind='ssvep_signal_export', session_id=lock['test_session'],
                    method=METHOD, sampling_rate_hz=FS, hop_samples=HOP, frequencies_hz=list(FREQUENCIES),
                    sample_count=len(signal.index), predictions=files, parameters='frozen_models.json',
                    annotations=annotations, protocol='protocol.json', columns=PRED_FIELDS,
                    recording={k: session[k] for k in ('status', 'completed_trials', 'trial_count', 'headless') if k in session},
                    display_mode=(session.get('display') or {}).get('mode') if isinstance(session.get('display'), dict) else None,
                    annotation_semantics='Presentation logs describe software events. Alignment uses original sample indices.',
                    alignment='end_sample is the inclusive zero-based sample_index in raw_brainflow.tsv. '
                              'Join on sample_index for the recorded timestamp_s.',
                    states={'target': 'frequency_hz is the detected frequency', 'no_target': 'valid window, no target detected',
                            'unavailable': 'window cannot be interpreted (see quality)'},
                    scores='score_* is detector evidence. spectral_support_* is 0 or 1. '
                           'Empty evidence fields indicate unavailable values.')
    # Publish the completion marker only after all data files are closed.
    from .streaming import atomic_json
    atomic_json(output / 'processing_manifest.json', manifest)
    return manifest


def compare_sessions(test, calibration, output, chunk=1024):
    test, output = Path(test), Path(output)
    calibration = Path(calibration) if calibration is not None else None
    signal, configs, lock, early, late = _prepare_replay(test, calibration, output, chunk, save_grid=True)
    sources = {'test': test}
    if calibration is not None:
        sources['calibration'] = calibration
    streams = {}
    for key, config in configs.items():
        streams[key] = list(signal.replay(config, chunk))
        write_table(output/f'{key}_predictions.tsv', map(prediction_row, streams[key]), PRED_FIELDS)
    segments = intervals(test, signal)  # Test labels are first opened after all prediction files are closed.
    all_rows = {k: score(segments, stream) for k,stream in streams.items()}
    details = []
    for i, baseline_row in enumerate(all_rows['baseline']):
        data = {k: baseline_row[k] for k in Interval.__dataclass_fields__}
        for mode, rows in all_rows.items():
            row = rows[i]
            if row['reason'] != baseline_row['reason']:
                raise RuntimeError('Compared arms have different scorable intervals')
            data.update({f'{mode}_{k}': v for k, v in row.items() if k not in Interval.__dataclass_fields__})
        details.append(data)
    write_table(output/'intervals.tsv', details, None if details else ['trial','kind','reason'])
    summary = dict(scoring_version=2, method=METHOD, calibration=early.get('session_id'), test=late['session_id'], sample_count=len(signal.index),
                   sides=[early.get('eye_side'), late.get('eye_side')],
                   metrics={k:totals(rows) for k,rows in all_rows.items()},
                   frequencies={k:{f:totals([r for r in rows if r['expected_hz']==f]) for f in FREQUENCIES} for k,rows in all_rows.items()},
                   quality_windows=dict(Counter(d.window.quality for d in streams['baseline'])),
                   exclusions=dict(Counter(r['reason'] for r in details if r['reason'])),
                   stream_states={k:dict(Counter(d.state for d in ds)) for k,ds in streams.items()},
                   raw_issues=dict(Counter(str(s) for s in sample_quality(signal.index, signal.packet, signal.clock, signal.eeg) if s)),
                   time_steps_over_6ms=int(np.sum(np.diff(signal.clock) > .006)),
                   inputs={group:{n:sha256(p/n) for n in ('raw_brainflow.tsv','events.tsv','frame_timing.tsv','protocol.json','session.json') if (p/n).is_file()}
                           for group,p in sources.items()})
    write_json(output/'metrics.json', summary)
    from .ssvep_replay_report import write_report
    write_report(output, lock, summary, details, test, calibration)
    return summary
