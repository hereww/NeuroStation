"""Reproducible continuous SSVEP software audit. No hardware or recorded EEG."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eeg_tools import continuous_ssvep as current
from eeg_tools.ssvep_replay import Interval, score, totals


def case(seed, amplitude, *, background=None, random_phase=False, startup=None, gap=False):
    rng = np.random.default_rng(seed)
    fs, block = current.FS, 5 * current.FS
    count = 12 if amplitude is None else 8
    idx = np.arange(2 * fs + count * block * (1 if amplitude is None else 2))
    eeg = rng.normal(0, 1, (len(idx), 8))
    if background is not None or startup is not None:
        hz = background if background is not None else startup
        eeg[:, 0] += 1.2 * np.sin(2 * np.pi * hz * idx / fs + rng.uniform(-np.pi, np.pi))
    intervals = []
    start = 2 * fs
    for k in range(count):
        if amplitude is not None:
            hz = current.FREQUENCIES[k % 4]
            span = slice(start, start + block)
            phase = rng.uniform(-np.pi, np.pi) if random_phase else 0
            eeg[span, 0] += amplitude * np.sin(2 * np.pi * hz * idx[span] / fs + phase)
            intervals.append(Interval(k, 'stimulus', start, start + block, hz))
            start += block
        intervals.append(Interval(k, 'stimulus' if startup else 'rest', start, start + block, startup))
        start += block
    if gap:
        keep = (idx < 7 * fs) | (idx >= 7 * fs + 83)
        idx, eeg = idx[keep], eeg[keep]
    return idx, eeg, intervals


def run(module, first_seed, count):
    result = {}
    for name, amplitude, options in (
        ('noise', None, {}), ('strong', 1.2, {}), ('weak', .15, {}),
        ('weak_random_phase', .15, {'random_phase': True}),
        ('stationary_background', None, {'background': 10}),
        ('strong_over_background', 1.2, {'background': 10, 'random_phase': True}),
        ('startup_target', None, {'startup': 10}),
        ('gap_background', None, {'background': 10, 'gap': True}),
    ):
        records = []
        for seed in range(first_seed, first_seed + count):
            idx, eeg, segments = case(seed, amplitude, **options)
            stream = list(module.ContinuousDetector().push(idx, idx % 256, idx / current.FS, eeg))
            records.extend(score(segments, stream))
        result[name] = totals(records)
        delays = [r['onset_to_first_target_s'] for r in records if r['result'] == 'correct']
        result[name]['median_first_correct_s'] = float(np.median(delays)) if delays else None
        if options.get('background') and amplitude:
            result[name]['at_background_hz'] = totals([r for r in records if r['expected_hz'] == options['background']])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-source', type=Path, help='Optional trusted earlier detector module for comparison')
    parser.add_argument('--first-seed', type=int, default=8293)
    parser.add_argument('--count', type=int, default=5)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    modules = [current]
    if args.prior_source:
        spec = importlib.util.spec_from_file_location('prior_detector', args.prior_source)
        prior = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = prior
        spec.loader.exec_module(prior)
        modules.insert(0, prior)
    result = {'first_seed': args.first_seed, 'seed_count': args.count,
              'scope': 'synthetic EEG only; startup_target and stationary_background have identical samples; scoring begins at 2 s',
              'methods': {m.METHOD: run(m, args.first_seed, args.count) for m in modules}}
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(text + '\n', encoding='utf-8')
    else:
        print(text)
