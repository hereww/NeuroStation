"""Optional SSVEP consumer for already acquired BrainFlow blocks."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .continuous_ssvep import ContinuousDetector, FREQUENCIES, FS, Gate, METHOD, WINDOW, HOP, evidence_settings
from .ssvep_replay import PRED_FIELDS, prediction_row
from .streaming import atomic_json


def load_personal_gate(path: Path, participant: str) -> Gate:
    lock = json.loads(path.read_text(encoding='utf-8'))
    if lock.get('method') != METHOD or lock.get('evidence') != evidence_settings():
        raise ValueError('Incompatible SSVEP detector version. Re-run personal calibration in EEG Processing.')
    if (lock.get('sampling_rate_hz') != FS
            or lock.get('frequencies_hz') != list(FREQUENCIES) or lock.get('window') != WINDOW
            or lock.get('hop') != HOP or lock.get('participant_id') != participant):
        raise ValueError('Incompatible SSVEP model or participant')
    personal = lock.get('gates', {}).get('personal')
    if not isinstance(personal, dict):
        raise ValueError('Select a model from a replay with personal calibration')
    return Gate(**personal)


class LiveSSVEP:
    def __init__(self, root, eeg_rows, packet_row, clock_row, config, preview=None, *, prepared=None):
        self.detector, self.model = prepared if prepared is not None else (ContinuousDetector(config), {})
        self.eeg_rows = list(eeg_rows)
        self.packet_row, self.clock_row = packet_row, clock_row
        self.preview = preview
        self.path = root / 'continuous_ssvep_predictions.tsv'
        self.file = self.path.open('x', encoding='utf-8-sig', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=PRED_FIELDS, delimiter='\t', lineterminator='\n')
        self.writer.writeheader()
        self.file.flush()
        self.count = self.missing_samples = self.next_sample = 0

    def consume(self, data, record):
        start, stop = record['start'], record['stop']
        self.missing_samples += max(0, start - self.next_sample)
        self.next_sample = stop
        last = None
        for decision in self.detector.push(np.arange(start, stop), data[self.packet_row],
                                          data[self.clock_row], data[self.eeg_rows].T):
            last = prediction_row(decision)
            self.writer.writerow(last)
            self.count += 1
        if last is not None:
            self.file.flush()
            if self.preview is not None:
                atomic_json(self.preview, {**last, 'model': self.model, 'received_monotonic_s': record.get('received_monotonic_s')})

    def metadata(self, sample_count):
        return dict(method=self.model.get('id', METHOD), model=self.model, gate=asdict(self.detector.gate.config), path=self.path.name,
                    sample_index_basis='end_sample is inclusive, zero-based from acquisition start',
                    complete=(self.count > 0 and self.count == sample_count//HOP
                              and self.missing_samples == 0 and self.next_sample == sample_count),
                    decisions=self.count, expected_decisions=sample_count//HOP,
                    unprocessed_samples=self.missing_samples + max(0, sample_count-self.next_sample))
