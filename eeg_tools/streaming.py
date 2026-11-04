"""Single-owner acquisition and append-only, recoverable raw block recording.

Saving is synchronous and precedes optional consumers. A disk failure is fatal;
consumer failures retain raw data and explicitly invalidate assessment. No sample
is dropped, deduplicated, interpolated, or indexed by a guessed device time.
"""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import numpy as np
from eeg_tools.runtime_bindings import sha256
from eeg_tools.quality.contracts import BlockMetadata, identity


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as f:
        json.dump(value, f, allow_nan=False, sort_keys=True)
        f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)


class BlockRecorder:
    def __init__(self, directory, *, session_id, channel_names, eeg_rows, package_row,
                 timestamp_row, sample_rate=250, raw_handle=None, raw_writer=None, clock=time.monotonic):
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=False)
        self.session_id, self.channels = session_id, tuple(channel_names)
        self.eeg_rows = tuple(eeg_rows)
        self.package_row, self.timestamp_row = package_row, timestamp_row
        self.sample_rate, self.clock = sample_rate, clock
        self.raw_handle, self.raw_writer = raw_handle, raw_writer
        self.journal = (self.directory/'blocks.jsonl').open('x', encoding='utf-8')
        self.received = self.written = self.durable = self.assessed = 0
        self.generation, self.sequence = 0, 0
        self.last_packet = self.last_timestamp = None
        self.previous_hash = '0'*64
        self.consumer_errors = []
        self.latencies = []
        self.closed = False
        self.failed = None

    @property
    def watermarks(self):
        return dict(received=self.received, written=self.written, durable=self.durable, assessed=self.assessed)

    def _boundary(self, packet, timestamp):
        reasons = []
        if not np.isfinite(packet) or packet != int(packet) or not 0 <= packet < 256:
            reasons.append('unknown_packet')
        elif self.last_packet is not None:
            delta = (int(packet) - self.last_packet) % 256
            if delta != 1:
                reasons.append('duplicate_packet' if delta == 0 else 'packet_discontinuity_or_restart')
        if not np.isfinite(timestamp):
            reasons.append('unknown_timestamp')
        elif self.last_timestamp is not None:
            delta = timestamp - self.last_timestamp
            if delta <= 0 or delta > 4/self.sample_rate:
                reasons.append('timestamp_discontinuity')
        self.last_packet = int(packet) if np.isfinite(packet) and packet == int(packet) and 0 <= packet < 256 else None
        self.last_timestamp = float(timestamp) if np.isfinite(timestamp) else None
        return reasons

    def append(self, data, consumer=None, received_at=None):
        if self.closed or self.failed:
            raise RuntimeError('recording not writable')
        raw = np.asarray(data, dtype=np.float64)
        if raw.ndim != 2 or max((*self.eeg_rows, self.package_row, self.timestamp_row)) >= raw.shape[0]:
            raise ValueError('invalid raw block shape')
        if not raw.shape[1]:
            return []
        arrival = self.clock() if received_at is None else received_at
        self.received += raw.shape[1]
        starts, gaps = [0], [[]]
        for i in range(raw.shape[1]):
            reasons = self._boundary(raw[self.package_row, i], raw[self.timestamp_row, i])
            if reasons:
                if i == 0:
                    gaps[0] = reasons
                else:
                    starts.append(i); gaps.append(reasons)
        starts.append(raw.shape[1])
        results = []
        for j, (left, right) in enumerate(zip(starts, starts[1:])):
            if gaps[j]:
                self.generation += 1
            block = raw[:, left:right].copy()
            meta = BlockMetadata(self.session_id, f'{self.session_id}:{self.generation}', self.generation,
                                 self.written, self.written + block.shape[1], self.channels,
                                 self.sample_rate, timestamp_source='brainflow_device_unverified', gaps=tuple(gaps[j])).to_dict()
            began = self.clock()
            try:
                filename = f'{self.sequence:08d}.npy'
                temporary = self.directory/(filename + '.tmp')
                with temporary.open('xb') as f:
                    np.save(f, block, allow_pickle=False); f.flush(); os.fsync(f.fileno())
                os.replace(temporary, self.directory/filename)
                if self.raw_writer is not None:
                    for offset in range(block.shape[1]):
                        self.raw_writer.writerow([self.written + offset, *[format(float(v), '.17g') for v in block[:, offset]]])
                    self.raw_handle.flush()
                self.written = meta['stop']
                record = {**meta, 'sequence': self.sequence, 'file': filename,
                          'sha256': sha256(self.directory/filename), 'previous_hash': self.previous_hash,
                          'received_monotonic_s': arrival, 'eeg_rows': list(self.eeg_rows),
                          'package_row': self.package_row, 'timestamp_row': self.timestamp_row}
                record['record_hash'] = identity(record)
                self.journal.write(json.dumps(record, allow_nan=False, sort_keys=True) + '\n')
                self.journal.flush()
                if self.raw_handle is not None:
                    os.fsync(self.raw_handle.fileno())
                os.fsync(self.journal.fileno())
                self.durable = self.written
                self.previous_hash = record['record_hash']
                self.sequence += 1
                block.flags.writeable = False
                consumer_record = {**record, "watermarks": self.watermarks}
                if consumer:
                    try:
                        consumer(block, consumer_record)
                        self.assessed = self.durable
                    except Exception as exc:
                        self.consumer_errors.append({'start':meta['start'], 'stop':meta['stop'],
                                                     'error':type(exc).__name__ + ': ' + str(exc)})
                        self.consumer_errors = self.consumer_errors[-32:]
                self.latencies.append(self.clock() - began)
                self.latencies = self.latencies[-2048:]
                atomic_json(self.directory/'watermarks.json', {**self.watermarks, 'status':'recording',
                            'consumer_errors':self.consumer_errors, 'processing_latency_s':self.latencies[-1],
                            'backpressure': self.latencies[-1] > block.shape[1]/self.sample_rate})
                results.append((block, {**record, 'watermarks':self.watermarks}))
            except Exception as exc:
                self.failed = type(exc).__name__ + ': ' + str(exc)
                raise
        return results

    def close(self, status='completed'):
        if self.closed:
            return
        try:
            self.journal.close()
            atomic_json(self.directory/'seal.json', {'schema_version':2,'status':'failed' if self.failed else status,
                        'error':self.failed, 'watermarks':self.watermarks, 'last_hash':self.previous_hash,
                        'consumer_errors':self.consumer_errors, 'quality_evidence':{p.name:sha256(p) for p in self.directory.glob('quality-*') if p.is_file()}})
        finally:
            self.closed = True


def recover_blocks(directory):
    """Read verified journal prefix only, never edit raw files or claim omitted tail complete."""
    directory = Path(directory)
    previous, expected, sequence = '0'*64, 0, 0
    records, issues = [], []
    with (directory/'blocks.jsonl').open(encoding='utf-8') as f:
        for line in f:
            try:
                r = json.loads(line)
                digest = r.pop('record_hash')
                if identity(r) != digest or r['previous_hash'] != previous:
                    raise ValueError('journal hash chain mismatch')
                r['record_hash'] = digest
                if r['sequence'] != sequence or r['start'] != expected or r['stop'] <= expected:
                    raise ValueError('journal sequence mismatch')
                if r['file'] != f'{sequence:08d}.npy' or sha256(directory/r['file']) != r['sha256']:
                    raise ValueError('block content mismatch')
                block = np.load(directory/r['file'], allow_pickle=False, mmap_mode='r')
                if block.ndim != 2 or block.shape[1] != r['stop']-r['start']:
                    raise ValueError('block shape mismatch')
                records.append(r)
                previous, expected, sequence = digest, r['stop'], sequence+1
            except (OSError, ValueError, KeyError, TypeError) as exc:
                issues.append(str(exc)); break
    seal = directory/'seal.json'
    complete = False
    if seal.is_file():
        try:
            s = json.loads(seal.read_text(encoding='utf-8'))
            complete = not issues and s['status']=='completed' and s['last_hash']==previous and s['watermarks']['received']==expected
        except (ValueError, KeyError, TypeError):
            issues.append('invalid seal')
    return {'records':records, 'verified_samples':expected, 'complete':complete, 'issues':issues,
            'uncommitted_files':[p.name for p in directory.glob('*.npy') if p.name not in {r['file'] for r in records}]}


def materialize_verified(directory):
    report = recover_blocks(directory)
    if report['issues']:
        raise ValueError('recording integrity failed: ' + repr(report['issues']))
    blocks = [np.load(Path(directory)/r['file'], allow_pickle=False, mmap_mode='r') for r in report['records']]
    if not blocks:
        return np.empty((0, 0))
    # Disk-backed legacy analysis view; raw acquisition memory is bounded.
    result = np.lib.format.open_memmap(Path(directory)/'legacy-analysis.npy', mode='w+', dtype=np.float64,
                                      shape=(blocks[0].shape[0], report['verified_samples']))
    for record, block in zip(report['records'], blocks):
        result[:, record['start']:record['stop']] = block
    result.flush()
    return result


class AcquisitionOwner:
    """One draining thread per board. UI requests health, never reads the board."""
    def __init__(self, board, recorder, consumer=None):
        self.board, self.recorder, self.consumer = board, recorder, consumer
        self.stop_event = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, name='NeuroStationCaptureOwner', daemon=True)

    def start(self):
        self.thread.start()

    def drain(self):
        data = self.board.get_board_data()
        for start in range(0, data.shape[1], 250):
            self.recorder.append(data[:, start:start+250], self.consumer)

    def _run(self):
        try:
            while not self.stop_event.is_set():
                self.drain()
                self.stop_event.wait(.02)
        except Exception as exc:
            self.error = exc

    def check(self):
        if self.error:
            raise RuntimeError('acquisition/save failed: ' + str(self.error)) from self.error

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError('capture owner did not stop; no second reader allowed')
        self.check()

    def final_drain(self):
        if self.thread.is_alive():
            raise RuntimeError('capture owner still active')
        self.check(); self.drain()
