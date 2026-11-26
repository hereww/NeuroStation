"""Single-owner acquisition and append-only, recoverable raw block recording.

Saving is synchronous and precedes bounded asynchronous production consumers. A disk failure is fatal;
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
import queue
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
    # Windows readers can briefly block replacement if they omit delete sharing.
    for delay in (0.005, 0.01, 0.02, 0.04, 0.08, None):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as exc:
            if delay is None or getattr(exc, 'winerror', None) not in (5, 32, 33):
                raise
            time.sleep(delay)


DEFERRED_ASSESSMENT = object()


class BlockRecorder:
    def __init__(self, directory, *, session_id, channel_names, eeg_rows, package_row,
                 timestamp_row, sample_rate=250, raw_handle=None, raw_writer=None, clock=time.monotonic, configuration_id=None, retain_raw=True):
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=False)
        self.session_id, self.channels = session_id, tuple(channel_names)
        self.eeg_rows = tuple(eeg_rows)
        self.package_row, self.timestamp_row = package_row, timestamp_row
        self.sample_rate, self.clock = sample_rate, clock
        self.configuration_id = configuration_id
        self.raw_handle, self.raw_writer = raw_handle, raw_writer
        self.retain_raw = retain_raw
        if not retain_raw and (raw_handle is not None or raw_writer is not None):
            raise ValueError("Raw writer supplied in results-only mode")
        self.next_sample = 0
        journal_name = 'blocks.jsonl' if retain_raw else 'intervals.jsonl'
        self.journal = (self.directory/journal_name).open('x', encoding='utf-8')
        self.received = self.written = self.durable = self.assessed = 0
        self.generation, self.sequence = 0, 0
        self.last_packet = self.last_timestamp = None
        self.previous_hash = '0'*64
        self.consumer_errors = []
        self.assessment_lock = threading.Lock()
        self.latencies = []
        self.closed = False
        self.failed = None

    @property
    def watermarks(self):
        return dict(received=self.received, written=self.written, durable=self.durable, assessed=self.assessed)

    def assessment_error(self, record, error):
        with self.assessment_lock:
            if self.closed:return
            self.consumer_errors.append({'start':record.get('start'),'stop':record.get('stop'),
                                         'error':str(error)})
            self.consumer_errors=self.consumer_errors[-32:]

    def acknowledge_assessment(self, record):
        with self.assessment_lock:
            if not self.closed and self.assessed==record['start'] and record['stop']<=self.next_sample:
                self.assessed=record['stop']

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
                                 self.next_sample, self.next_sample + block.shape[1], self.channels,
                                 self.sample_rate, timestamp_source='brainflow_device_unverified', gaps=tuple(gaps[j])).to_dict()
            began = self.clock()
            try:
                storage = {}
                if self.retain_raw:
                    filename = f'{self.sequence:08d}.npy'
                    temporary = self.directory/(filename + '.tmp')
                    with temporary.open('xb') as f:
                        np.save(f, block, allow_pickle=False); f.flush(); os.fsync(f.fileno())
                    os.replace(temporary, self.directory/filename)
                    if self.raw_writer is not None:
                        for offset in range(block.shape[1]):
                            self.raw_writer.writerow([meta['start'] + offset, *[format(float(v), '.17g') for v in block[:, offset]]])
                        self.raw_handle.flush()
                    self.written = meta['stop']
                    storage = {'file': filename, 'sha256': sha256(self.directory/filename)}
                record = {**meta, **storage, 'sequence': self.sequence, 'previous_hash': self.previous_hash,
                          'received_monotonic_s': arrival, 'eeg_rows': list(self.eeg_rows),
                          'configuration_id': self.configuration_id,
                          'package_row': self.package_row, 'timestamp_row': self.timestamp_row}
                record['record_hash'] = identity(record)
                self.journal.write(json.dumps(record, allow_nan=False, sort_keys=True) + '\n')
                self.journal.flush()
                if self.raw_handle is not None:
                    os.fsync(self.raw_handle.fileno())
                os.fsync(self.journal.fileno())
                self.durable = self.written
                self.next_sample = meta['stop']
                self.previous_hash = record['record_hash']
                self.sequence += 1
                block.flags.writeable = False
                consumer_record = {**record, "watermarks": self.watermarks}
                consumer_ok = False
                if consumer:
                    try:
                        outcome = consumer(block, consumer_record)
                        consumer_ok = outcome is not DEFERRED_ASSESSMENT
                        if consumer_ok:self.acknowledge_assessment(record)
                    except Exception as exc:
                        self.assessment_error(record,type(exc).__name__ + ': ' + str(exc))
                self.latencies.append(self.clock() - began)
                self.latencies = self.latencies[-2048:]
                atomic_json(self.directory/'watermarks.json', {**self.watermarks, 'status':'recording',
                            'consumer_errors':self.consumer_errors, 'processing_latency_s':self.latencies[-1],
                            'backpressure': self.latencies[-1] > block.shape[1]/self.sample_rate})
                results.append((block, {**record, 'watermarks':self.watermarks, 'consumer_ok':consumer_ok}))
            except Exception as exc:
                self.failed = type(exc).__name__ + ': ' + str(exc)
                raise
        return results

    def close(self, status='completed', *, quality_complete=True):
        if self.closed:
            return
        try:
            self.journal.close()
            atomic_json(self.directory/'seal.json', {'schema_version':2,'status':'failed' if self.failed else status,
                        'error':self.failed, 'watermarks':self.watermarks, 'last_hash':self.previous_hash,
                        'consumer_errors':self.consumer_errors, 'quality_complete':quality_complete, 'quality_evidence':{p.name:sha256(p) for p in (*self.directory.glob('quality-*'),*self.directory.glob('processing-*.json')) if p.is_file()} if quality_complete else {}})
        finally:
            self.closed = True


def recover_blocks(directory, *, cancelled=None):
    """Read verified journal prefix only, never edit raw files or claim omitted tail complete."""
    directory = Path(directory)
    previous, expected, sequence = '0'*64, 0, 0
    records, issues = [], []
    with (directory/'blocks.jsonl').open(encoding='utf-8') as f:
        for line in f:
            if cancelled and cancelled():raise InterruptedError("quality.cancelled")
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
                try:
                    if block.ndim != 2 or block.shape[1] != r['stop']-r['start']:
                        raise ValueError('block shape mismatch')
                finally:
                    block._mmap.close()
                records.append(r)
                previous, expected, sequence = digest, r['stop'], sequence+1
            except (OSError, ValueError, KeyError, TypeError) as exc:
                issues.append(str(exc)); break
    committed = {r['file'] for r in records}
    uncommitted = sorted(p.name for p in directory.glob('[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].npy') if p.name not in committed)
    temporary = sorted(p.name for p in directory.glob('*.tmp'))
    seal = directory/'seal.json'
    complete = False
    if seal.is_file():
        try:
            s = json.loads(seal.read_text(encoding='utf-8'))
            complete = not issues and not uncommitted and not temporary and s['status']=='completed' and s['last_hash']==previous and all(s['watermarks'][k]==expected for k in ('received','written','durable'))
        except (ValueError, KeyError, TypeError):
            issues.append('invalid seal')
    return {'records':records, 'verified_samples':expected, 'complete':complete, 'issues':issues,
            'uncommitted_files':uncommitted, 'temporary_files':temporary}


def materialize_verified(directory):
    report = recover_blocks(directory)
    if report['issues']:
        raise ValueError('recording integrity failed: ' + repr(report['issues']))
    records = report['records']
    if not records:
        return np.empty((0, 0))
    first = np.load(Path(directory)/records[0]['file'], allow_pickle=False, mmap_mode='r')
    row_count = first.shape[0]
    first._mmap.close()
    # Disk-backed legacy analysis view; keep only one raw-block mapping open at once.
    import uuid
    result = np.lib.format.open_memmap(Path(directory)/('legacy-analysis-'+uuid.uuid4().hex+'.npy'), mode='w+', dtype=np.float64,
                                      shape=(row_count, report['verified_samples']))
    for record in records:
        block = np.load(Path(directory)/record['file'], allow_pickle=False, mmap_mode='r')
        try:
            result[:, record['start']:record['stop']] = block
        finally:
            block._mmap.close()
    result.flush()
    return result


class AsyncConsumer:
    """Bounded optional work. A slow/failed quality consumer never stalls raw writes.

    Full queues refuse NEW work, record its exact interval, and leave the contiguous
    assessed watermark at the first missing interval. The next processed block's
    start/stream contract resets causal state after a skipped interval.
    """
    def __init__(self, recorder, consumer, max_pending=8, *, purpose="quality"):
        self.purpose = purpose
        if max_pending<1:raise ValueError('positive quality queue capacity required')
        self.recorder,self.consumer=recorder,consumer
        self.queue=queue.Queue(maxsize=max_pending)
        self.stop_event=threading.Event();self.active=None
        self.submitted=self.completed=self.dropped=0
        self.thread=threading.Thread(target=self._run,name=f'NeuroStation-{purpose}',daemon=True)
        self.thread.start()

    def __call__(self,block,record):
        if self.stop_event.is_set():raise RuntimeError('quality consumer stopped')
        try:
            self.queue.put_nowait((block,record));self.submitted+=1
        except queue.Full:
            self.dropped+=1
            self.recorder.assessment_error(record,f'{self.purpose}_queue_full: interval unassessed')
        return DEFERRED_ASSESSMENT

    def _run(self):
        while not self.stop_event.is_set() or not self.queue.empty():
            try:block,record=self.queue.get(timeout=.02)
            except queue.Empty:continue
            self.active=record
            try:
                self.consumer(block,record)
                if self.purpose == "quality":
                    self.recorder.acknowledge_assessment(record)
                self.completed += 1
            except Exception as exc:
                self.recorder.assessment_error(record,type(exc).__name__+': '+str(exc))
            finally:
                self.active=None;self.queue.task_done()

    def close(self,timeout=3.0):
        self.stop_event.set();self.thread.join(timeout=timeout)
        stopped=not self.thread.is_alive()
        if not stopped:
            self.recorder.assessment_error(self.active or {},f'{self.purpose}_shutdown_timeout: pending work is unknown')
        return stopped

    @property
    def status(self):
        return {'submitted':self.submitted,'completed':self.completed,'dropped':self.dropped,
                'pending':self.queue.qsize(),'active':self.active is not None,
                'capacity':self.queue.maxsize,'thread_stopped':not self.thread.is_alive()}


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
