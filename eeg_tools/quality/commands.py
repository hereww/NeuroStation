"""Outer quality/freshness gate. Malformed evidence fails closed; model schema is unchanged."""
from collections import deque
from copy import deepcopy
import math
import time

REQUIRED_DIMENSIONS = ('integrity', 'signal_quality', 'coverage', 'stimulus_timing')


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _valid_record(result):
    return (isinstance(result, dict) and type(result.get('schema_version')) is int and result['schema_version'] == 2
            and all(isinstance(result.get(key), str) and bool(result[key]) for key in ('processing_id', 'stream_id'))
            and type(result.get('start')) is int and type(result.get('stop')) is int
            and 0 <= result['start'] < result['stop']
            and ('generation' not in result or type(result['generation']) is int and result['generation'] >= 0)
            and _finite(result.get('updated_monotonic_s'))
            and isinstance(result.get('dimensions'), dict)
            and all(result['dimensions'].get(key) in ('pass', 'fail', 'unknown') for key in REQUIRED_DIMENSIONS))


class QualityCommandGate:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.reset_session()

    def reset_session(self):
        self.records = deque(maxlen=4096)
        self.current_identity = self.current_stream = self.current_generation = None
        self.select_armed = True

    def _invalidate(self):
        changed = self.current_identity is not None or bool(self.records)
        self.records.clear()
        self.current_identity = self.current_stream = self.current_generation = None
        # Rejections/identity changes must not rearm SELECT.
        return changed

    def observe(self, result):
        if not _valid_record(result):
            return self._invalidate()
        new_identity, new_stream = result['processing_id'], result['stream_id']
        generation = result.get('generation')
        changed = self.current_identity is not None and (
            new_identity != self.current_identity or new_stream != self.current_stream
            or generation != self.current_generation)
        if not changed and self.records and result['start'] < self.records[-1]['stop']:
            # Duplicate/overlapping or late coverage cannot overrule earlier evidence.
            return self._invalidate()
        if self.records and result['start'] != self.records[-1]['stop']:
            changed = True
        if changed:
            self.records.clear()
        self.current_identity, self.current_stream, self.current_generation = new_identity, new_stream, generation
        keys = ('schema_version', 'start', 'stop', 'stream_id', 'processing_id', 'generation', 'dimensions', 'updated_monotonic_s')
        self.records.append(deepcopy({key: result[key] for key in keys if key in result}))
        return changed

    def assess(self, window, *, processing_id=None, now=None):
        now = self.clock() if now is None else now
        def reject(reason):
            return {'accepted': False, 'reason': reason, 'processing_id': self.current_identity}
        if not self.records:
            return reject('quality_unavailable')
        if not all(_valid_record(record) for record in self.records):
            return reject('quality_rejected_or_unassessable')
        updated = self.records[-1]['updated_monotonic_s']
        if not _finite(now) or not 0 <= now - updated <= 2:
            return reject('quality_stale')
        if not self.current_identity or (processing_id is not None and processing_id != self.current_identity):
            return reject('processing_identity_mismatch')
        if (not isinstance(window, dict) or type(window.get('sample_start')) is not int
                or type(window.get('sample_end')) is not int or window['sample_start'] < 0):
            return reject('unknown_window_interval')
        start, stop = window['sample_start'], window['sample_end'] + 1  # Legacy inclusive adapter, no coercion.
        if stop - start != 1250:
            return reject('quality_window_mismatch')
        cursor, previous_stop = start, None
        for record in self.records:
            if (record['stream_id'] != self.current_stream or record['processing_id'] != self.current_identity
                    or record.get('generation') != self.current_generation):
                return reject('processing_identity_mismatch')
            if previous_stop is not None and record['start'] < previous_stop:
                return reject('quality_coverage_overlap')
            previous_stop = record['stop']
            if record['stop'] <= start or record['start'] >= stop:
                continue
            if record['start'] > cursor:
                return reject('quality_coverage_gap')
            if any(record['dimensions'][key] != 'pass' for key in REQUIRED_DIMENSIONS):
                return reject('quality_rejected_or_unassessable')
            cursor = min(stop, record['stop'])
        if cursor >= stop:
            return {'accepted': True, 'reason': 'accepted', 'processing_id': self.current_identity,
                    'stream_id': self.current_stream, 'start': start, 'stop': stop}
        return reject('quality_coverage_incomplete')

    def latch(self, command):
        if command == 'SELECT':
            if not self.select_armed:
                return 'UNKNOWN', 'select_latched'
            self.select_armed = False
        elif command in ('MOVE_LEFT', 'MOVE_RIGHT'):
            self.select_armed = True
        return command, 'accepted' if command != 'UNKNOWN' else 'rejected'
