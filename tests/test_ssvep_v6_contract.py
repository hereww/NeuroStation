"""Synthetic-only v6 safety/math tests; never read real EEG/NPZ/PT or CUDA.

Fixture .npz/.pt files contain inert bytes, not EEG/checkpoint serialization.
All fixture filesystem activity is constrained to a temporary C: directory.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from eeg_tools import ssvep_v6_contract as c


def config():
    # Exact fixed design embedded in the source, no real artifact reads in tests.
    return json.loads(c._EXPECTED_CONFIG_JSON)


def signal():
    t = np.arange(256, dtype=np.float64)
    return np.tile(np.sin(2 * np.pi * 12 * t / 256) + .3 * np.cos(2 * np.pi * 31 * t / 256), (8, 1))


def row(condition='transient', *, dataset='Nakanishi2015', subject=5, trial=0, a=1., b=.5, purpose='diagnostic', epoch=0):
    identity = condition == 'identity'
    return {'dataset': dataset, 'subject_id': subject, 'trial_index': trial, 'condition': condition,
            'seed': c._case_seed(dataset, subject, trial, condition, purpose, epoch, config()),
            'input_normalized_mse': float(0. if identity else a), 'output_normalized_mse': float(b),
            'relative_reduction': None if identity or a == 0 else (float(a) - float(b)) / float(a),
            'identity_normalized_change': float(b) if identity else None,
            'outcome': 'loss' if identity and b > 0 else 'tie' if identity or a == b else 'win' if b < a else 'loss',
            'protected_complex_energy_ratio': 0.0 if identity else None,
            'protected_weighted_abs_phase_radians': 0.0 if identity else None, 'undefined_phase_bin_count': 0}


class Fixture:
    """Generate 16 metadata identities; NO files at all for subjects 7-10."""
    def __init__(self, root, stage='training'):
        self.root, self.stage = root, stage
        cfg = config()
        for rel in c.INPUT_PATHS + c.NEW_SOURCE_PATHS:
            self.put(rel, ('synthetic inert placeholder: ' + rel).encode())
        self.put_json(c.CONFIG_RELATIVE, cfg)
        self.config_sha = self.digest(c.CONFIG_RELATIVE)
        self.put_json(c.PARTITION_RELATIVE, {
            'schema_version': 1, 'status': 'formal', 'stage': 'reference_identity_freeze',
            'partitions': {'train': [1, 2, 3, 4, 5, 6], 'validation': [7, 8], 'holdout': [9, 10]},
            'raw_reference_is_clean_ground_truth': False, 'runtime': {'enabled': False, 'production_validated': False}})
        partition_sha = self.digest(c.PARTITION_RELATIVE)
        catalog = []
        for dataset in c.DATASETS:
            for subject in range(1, 9):
                r = {'dataset': dataset, 'subject_id': subject, 'split': 'train' if subject <= 6 else 'validation',
                     'relative_path': f'{dataset}/subject_{subject:03d}.npz', 'trials': c.SHAPES[dataset][0]}
                if subject <= 6:
                    rel = f'{c.REFERENCE_DIR}/{dataset}/subject_{subject:03d}.npz'
                    npz_sha = self.digest(rel)
                    side = {'schema_version': 1, 'status': 'formal', 'reference_version': 'ssvep_reference_v5',
                        'dataset': dataset, 'subject_id': subject, 'split': 'train',
                        'reference_shape': list(c.SHAPES[dataset]), 'reference_dtype': 'float64',
                        'native_sample_rate_hz': c.RATES[dataset], 'onset_sample_zero_based': c.STARTS[dataset],
                        'native_units': None, 'source_lock_file_sha256': partition_sha,
                        'raw_reference_is_clean_ground_truth': False, 'runtime_enabled': False,
                        'npz_sha256': npz_sha, 'materialized_trials': c.SHAPES[dataset][0],
                        'source': {'path': 'D:/NEVER_ACCESS_SOURCE.mat'}}
                    self.put_json(rel + '.manifest.json', side)
                    r.update(npz_sha256=npz_sha, manifest_sha256=self.digest(rel + '.manifest.json'))
                else:
                    r['relative_path'] = 'D:/DO_NOT_RESOLVE_OUTER.npz'
                catalog.append(r)
        self.put_json(c.MANIFEST_RELATIVE, {
            'schema_version': 1, 'status': 'formal', 'reference_version': 'ssvep_reference_v5', 'build_completed': True,
            'source_lock_file_sha256': partition_sha, 'raw_reference_is_clean_ground_truth': False,
            'runtime': {'enabled': False, 'production_validated': False}, 'materialized_subjects': catalog,
            'source_root': 'D:/NEVER', 'holdout_sources': ['D:/NO_HOLDOUT_009.npz', 'D:/NO_HOLDOUT_010.npz']})
        self.protocol = {'schema_version': 1, 'status': 'formal',
            'experiment': 'learned_denoise_v6_ssvep_train_internal_only',
            'configuration': {'path': c.CONFIG_RELATIVE, 'sha256': self.config_sha},
            'model_design': cfg['model'], 'training_schedule': cfg['training'],
            'internal_gates': cfg['internal_research_gates'],
            'scope': {'fit_subjects': [1, 2, 3, 4], 'internal_selection_subjects': [5, 6],
                      'diagnostic_subjects': [5, 6], 'outer_validation_subjects_not_loaded': [7, 8],
                      'holdout_subjects_not_loaded': [9, 10], 'recorded_reference_is_clean_ground_truth': False,
                      'runtime_enabled': False, 'production_validated': False},
            'bound_files': {p: self.digest(p) for p in c.INPUT_PATHS}}
        self.put_json(c.PREREG_RELATIVE, self.protocol)
        self.prereg_sha = self.digest(c.PREREG_RELATIVE)
        files = c.INPUT_PATHS + (c.PREREG_RELATIVE,) + c.NEW_SOURCE_PATHS
        if stage == 'internal_evaluation':
            self.put(c.CHECKPOINT_RELATIVE, b'NOT A REAL CHECKPOINT')
            self.put_json(c.TRAIN_REPORT_RELATIVE, {'synthetic': True})
            files += (c.CHECKPOINT_RELATIVE, c.TRAIN_REPORT_RELATIVE)
        self.lock = {'schema_version': 1, 'status': 'formal',
            'stage': 'frozen_before_v6_training_waveform_read' if stage == 'training' else 'frozen_before_v6_internal_diagnostic',
            'scope': dict.fromkeys(c.SCOPE_FALSE_FLAGS, False),
            'waveforms_loaded': stage != 'training', 'checkpoint_loaded': stage != 'training',
            'files': {rel: self.digest(rel) for rel in files}}
        self.lock_rel = c.TRAIN_LOCK_RELATIVE if stage == 'training' else c.EVAL_LOCK_RELATIVE
        self.put_json(self.lock_rel, self.lock)

    def put(self, rel, data):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def put_json(self, rel, value):
        self.put(rel, json.dumps(value, allow_nan=False).encode())

    def digest(self, rel):
        return hashlib.sha256((self.root / rel).read_bytes()).hexdigest()

    def rebind(self, rel):
        self.protocol['bound_files'][rel] = self.digest(rel)
        self.put_json(c.PREREG_RELATIVE, self.protocol)
        self.prereg_sha = self.digest(c.PREREG_RELATIVE)
        self.lock['files'][rel] = self.digest(rel)
        self.lock['files'][c.PREREG_RELATIVE] = self.prereg_sha
        self.put_json(self.lock_rel, self.lock)

    def hashes(self):
        return patch.multiple(c, CONFIG_SHA256=self.config_sha, PREREG_SHA256=self.prereg_sha)


class TemporaryTest(unittest.TestCase):
    def setUp(self):
        # Absolute C: workspace prevents a user TEMP environment on D:.
        self.temporary = tempfile.TemporaryDirectory(prefix='v6_contract_synthetic_', dir=c.ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)


class ConfigJSONTests(TemporaryTest):
    def test_complete_exact_config_and_each_leaf_and_unknown_key(self):
        c.validate_config(config())
        def leaves(value, prefix=()):
            if type(value) is dict:
                for k, v in value.items():
                    yield from leaves(v, prefix + (k,))
            elif type(value) is list:
                for i, v in enumerate(value):
                    yield from leaves(v, prefix + (i,))
            else:
                yield prefix, value
        for path, value in leaves(config()):
            changed = config()
            node = changed
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = not value if type(value) is bool else value + 1 if type(value) in (int, float) else value + '_changed'
            with self.subTest(path=path), self.assertRaises(c.V6Error):
                c.validate_config(changed)
        for section in (None, 'model', 'corruption', 'training', 'loss', 'safety', 'seeds'):
            changed = config()
            (changed if section is None else changed[section])['UNKNOWN'] = 0
            with self.subTest(section=section), self.assertRaises(c.V6Error):
                c.validate_config(changed)

    def test_strict_bool_int_float_types_and_nonfinite(self):
        for key, value in [('input_channels', True), ('schema_version', True), ('window_samples', 256.0)]:
            changed = config()
            changed[key] = value
            with self.assertRaises(c.V6Error):
                c.validate_config(changed)
        for value in (0, 1, None, 'false'):
            changed = config()
            changed['model']['post_projection_clip'] = value
            with self.assertRaises(c.V6Error):
                c.validate_config(changed)
        for value in (float('nan'), float('inf'), float('-inf'), 0):
            changed = config()
            changed['model']['dropout'] = value
            with self.assertRaises(c.V6Error):
                c.validate_config(changed)

    def test_duplicate_nonfinite_overflow_and_invalid_json(self):
        path = self.root / 'input.json'
        for text in ('{"a":1,"a":2}', '{"a":{"b":1,"b":2}}', '{"a":NaN}', '{"a":Infinity}',
                     '{"a":-Infinity}', '{"a":1e999}', '[]', '{bad', ''):
            path.write_text(text, encoding='utf-8')
            with self.subTest(text=text), self.assertRaises(c.V6Error):
                c.read_json(self.root, 'input.json')
        path.write_bytes(b'\xef\xbb\xbf{"good":true}')
        self.assertEqual(c.read_json(self.root, 'input.json'), {'good': True})
        path.write_bytes(b'\xff')
        with self.assertRaises(c.V6Error):
            c.read_json(self.root, 'input.json')
        with self.assertRaises(c.V6Error):
            c.read_json(self.root, 'arbitrary.pt')


class PreflightTests(TemporaryTest):
    def guard_binary_open(self, predicate=lambda p: True):
        actual = os.open
        def guarded(path, *args, **kwargs):
            text = str(path).replace('\\', '/')
            if text.endswith(('.npz', '.pt')) and predicate(text):
                raise AssertionError('forbidden binary open: ' + text)
            return actual(path, *args, **kwargs)
        return patch.object(c.os, 'open', side_effect=guarded)

    def test_training_check_only_40_bindings_and_no_binary_open(self):
        fixture = Fixture(self.root)
        with fixture.hashes(), self.guard_binary_open():
            state = c.preflight(self.root)
            self.assertEqual(len(state.files), 40)
            self.assertEqual(len(state.selected), 12)
            self.assertEqual(len(state.deferred), 12)
            self.assertEqual(len(state.verified_hashes), 28)
            c.verify_frozen(state)
            report = state.report()
            self.assertEqual(report['deferred'], list(state.deferred))
            self.assertFalse(report['waveforms_loaded'])
            self.assertFalse(report['checkpoint_loaded'])
            self.assertFalse(report['cuda_initialized'])
            json.dumps(report, allow_nan=False)
            with self.assertRaises(c.V6Error):
                c.load_training_records(state)

    def test_diagnostic_check_only_42_and_runtime_only_5_6_and_pt_hash(self):
        fixture = Fixture(self.root, 'internal_evaluation')
        with fixture.hashes(), self.guard_binary_open():
            check = c.preflight(self.root, 'internal_evaluation', True)
            self.assertEqual(len(check.files), 42)
            self.assertEqual(len(check.selected), 4)
            self.assertEqual({r['subject_id'] for r in check.selected}, {5, 6})
            self.assertEqual(len(check.deferred), 13)
        fit = lambda p: p.endswith(tuple(f'subject_{s:03d}.npz' for s in range(1, 5)))
        with fixture.hashes(), self.guard_binary_open(fit):
            state = c.preflight(self.root, 'internal_evaluation', False)
            self.assertEqual(len(state.deferred), 8)
            self.assertIn(c.CHECKPOINT_RELATIVE, state.verified_hashes)
            self.assertEqual(sum(p.endswith('.npz') for p in state.verified_hashes), 4)
            c.verify_frozen(state)

    def test_check_only_never_stats_outer_holdout_or_d_metadata_source_paths(self):
        fixture = Fixture(self.root, 'internal_evaluation')
        actual = Path.lstat
        touched = []
        def guarded(path, *args, **kwargs):
            text = str(path).replace('\\', '/')
            touched.append(text)
            if text.lower().startswith('d:') or 'DO_NOT_RESOLVE_OUTER' in text or 'NO_HOLDOUT' in text:
                raise AssertionError('forbidden external metadata path touched')
            if any(f'subject_{s:03d}.npz' in text for s in (7, 8, 9, 10)):
                raise AssertionError('outer/holdout path stat forbidden')
            return actual(path, *args, **kwargs)
        with fixture.hashes(), patch.object(Path, 'lstat', guarded), self.guard_binary_open():
            state = c.preflight(self.root, 'internal_evaluation')
            c.verify_frozen(state)
        self.assertTrue(touched)

    def test_runtime_load_allowlists_training_and_diagnostic(self):
        from eeg_tools import ssvep_denoise_v5 as core
        for stage, subjects in [('training', (1, 2, 3, 4, 5, 6)), ('internal_evaluation', (5, 6))]:
            fixture = Fixture(self.root, stage)
            with fixture.hashes():
                state = c.preflight(self.root, stage, False)
                def records(root, *, subjects):
                    return [SimpleNamespace(dataset=d, subject_id=s, sample_rate_hz=c.RATES[d],
                        npz_sha256=state.files[f'{c.REFERENCE_DIR}/{d}/subject_{s:03d}.npz'],
                        sidecar_sha256=state.files[f'{c.REFERENCE_DIR}/{d}/subject_{s:03d}.npz.manifest.json'],
                        signal=np.broadcast_to(np.float64(0), c.SHAPES[d])) for d in c.DATASETS for s in subjects]
                with patch.object(core, 'load_reference_records', side_effect=records) as loader:
                    loaded = c.load_training_records(state)
                    loader.assert_called_once_with(state.root, subjects=subjects)
                    self.assertEqual(len(loaded), 2 * len(subjects))
                with patch.object(core, 'load_reference_records', return_value=[]):
                    with self.assertRaises(c.V6Error):
                        c.load_training_records(state)

    def test_frozen_hash_stat_lock_config_and_state_mutation(self):
        fixture = Fixture(self.root)
        with fixture.hashes():
            state = c.preflight(self.root)
            target = c.NEW_SOURCE_PATHS[0]
            (self.root / target).write_bytes(b'changed')
            with self.assertRaises(c.V6Error):
                c.verify_frozen(state)
        fixture = Fixture(self.root)
        with fixture.hashes():
            state = c.preflight(self.root)
            state.config['training']['epochs'] = 19
            with self.assertRaises(c.V6Error):
                c.verify_frozen(state)
        fixture = Fixture(self.root)
        with fixture.hashes():
            state = c.preflight(self.root)
            fixture.lock['files'][c.NEW_SOURCE_PATHS[1]] = 'f' * 64
            fixture.put_json(fixture.lock_rel, fixture.lock)
            with self.assertRaises(c.V6Error):
                c.verify_frozen(state)
        fixture = Fixture(self.root)
        with fixture.hashes():
            state = c.preflight(self.root)
            state.files[c.NEW_SOURCE_PATHS[1]] = 'f' * 64
            with self.assertRaises(c.V6Error):
                c.verify_frozen(state)
        fixture = Fixture(self.root)
        with fixture.hashes():
            state = c.preflight(self.root)
            rel = next(iter(state.deferred))
            (self.root / rel).write_bytes(b'changed inert bytes')
            with self.assertRaises(c.V6Error):
                c.verify_frozen(state)

    def test_absent_stage_markers_binding_scope_and_types_fail_closed(self):
        fixture = Fixture(self.root)
        for key, value in [('schema_version', True), ('status', 'draft'), ('stage', 'wrong'),
                           ('checkpoint_loaded', True), ('waveforms_loaded', True),
                           ('scope', {**fixture.lock['scope'], 'runtime_enabled': 0})]:
            broken = copy.deepcopy(fixture.lock)
            broken[key] = value
            fixture.put_json(fixture.lock_rel, broken)
            with fixture.hashes(), self.subTest(key=key), self.assertRaises(c.V6Error):
                c.preflight(self.root)
        for mutate in (lambda x: x['files'].pop(c.NEW_SOURCE_PATHS[0]),
                       lambda x: x['files'].update({'extra.py': 'a' * 64}),
                       lambda x: x['files'].update({c.CONFIG_RELATIVE: 'b' * 64}),
                       lambda x: x['files'].update({c.PREREG_RELATIVE: 'c' * 64}),
                       lambda x: x['files'].update({c.NEW_SOURCE_PATHS[0]: 'Z' * 64}),
                       lambda x: x.pop('scope')):
            broken = copy.deepcopy(fixture.lock)
            mutate(broken)
            fixture.put_json(fixture.lock_rel, broken)
            with fixture.hashes(), self.assertRaises(c.V6Error):
                c.preflight(self.root)
        with self.assertRaises(c.V6Error):
            c.preflight(self.root, 'outer')
        with self.assertRaises(c.V6Error):
            c.preflight(self.root, check_only=1)
        (self.root / fixture.lock_rel).unlink()
        with self.assertRaises(c.V6Error):
            c.preflight(self.root)

    def test_catalog_completeness_splits_shapes_units_fs_onset_and_flags(self):
        for name, value in [('native_sample_rate_hz', 250), ('onset_sample_zero_based', 125),
                            ('native_units', 'uV'), ('raw_reference_is_clean_ground_truth', True),
                            ('reference_shape', [180, 1114, 7]), ('subject_id', True), ('split', 'validation')]:
            fixture = Fixture(self.root)
            rel = f'{c.REFERENCE_DIR}/Nakanishi2015/subject_001.npz.manifest.json'
            side = json.loads((self.root / rel).read_text())
            side[name] = value
            fixture.put_json(rel, side)
            fixture.rebind(rel)
            # Rebind catalog sidecar digest so failure tests metadata, not just SHA.
            manifest = json.loads((self.root / c.MANIFEST_RELATIVE).read_text())
            manifest['materialized_subjects'][0]['manifest_sha256'] = fixture.digest(rel)
            fixture.put_json(c.MANIFEST_RELATIVE, manifest)
            fixture.rebind(c.MANIFEST_RELATIVE)
            with fixture.hashes(), self.subTest(name=name), self.assertRaises(c.V6Error):
                c.preflight(self.root)
        for mutation in ('missing', 'duplicate', 'outer-split'):
            fixture = Fixture(self.root)
            manifest = json.loads((self.root / c.MANIFEST_RELATIVE).read_text())
            if mutation == 'missing':
                manifest['materialized_subjects'].pop()
            elif mutation == 'duplicate':
                manifest['materialized_subjects'][-1] = manifest['materialized_subjects'][0]
            else:
                manifest['materialized_subjects'][-1]['split'] = 'holdout'
            fixture.put_json(c.MANIFEST_RELATIVE, manifest)
            fixture.rebind(c.MANIFEST_RELATIVE)
            with fixture.hashes(), self.assertRaises(c.V6Error):
                c.preflight(self.root)


class MathTests(unittest.TestCase):
    def test_seed_formula_training_epochs_datasets_and_condition_copy(self):
        ref, cfg = signal(), config()
        original = copy.deepcopy(cfg)
        for dataset in c.DATASETS:
            for purpose, subject, epoch in [('training', 1, 1), ('training', 4, 18),
                                             ('internal_selection', 5, 0), ('diagnostic', 6, 0)]:
                for condition in c.CONDITIONS:
                    noisy, seed = c.corrupt_case(ref, dataset, subject, 7, condition, cfg, purpose=purpose, epoch=epoch)
                    expected = cfg['seeds'][purpose] + c.DATASETS.index(dataset)*10000000 + subject*100000 + 70 + c.CONDITIONS.index(condition) + epoch*100000000
                    self.assertEqual(seed, expected)
                    self.assertEqual(noisy.dtype, np.float64)
                    self.assertEqual(noisy.shape, (8, 256))
                    self.assertFalse(np.shares_memory(ref, noisy))
                    if condition == 'identity':
                        np.testing.assert_array_equal(noisy, ref)
                    else:
                        self.assertFalse(np.array_equal(noisy, ref))
                    again, _ = c.corrupt_case(ref, dataset, subject, 7, condition, cfg, purpose=purpose, epoch=epoch)
                    np.testing.assert_array_equal(noisy, again)
        self.assertEqual(cfg, original)
        np.testing.assert_array_equal(ref, signal())

    def test_corruption_override_is_deep_copy_single_family(self):
        from eeg_tools import ssvep_denoise_v5 as core
        cfg, ref = config(), signal()
        original = copy.deepcopy(cfg)
        def corrupt(x, fs, override, seed):
            self.assertEqual(fs, 250)
            self.assertEqual(override['corruption']['identity_probability'], 0.0)
            self.assertEqual(override['corruption']['combine_probability'], 0.0)
            self.assertEqual(override['corruption']['families'], ['transient'])
            override['corruption']['transient']['amplitude_rms_range'][0] = 999
            x[0, 0] = 1000  # A defensive copied reference, not the caller's array.
            return x, False, ['transient']
        with patch.object(core, 'corrupt_window', side_effect=corrupt):
            c.corrupt_case(ref, 'Wearable', 5, 0, 'transient', cfg, purpose='diagnostic')
        self.assertEqual(cfg, original)
        np.testing.assert_array_equal(ref, signal())

    def test_purpose_subject_epoch_trial_type_and_window_allowlists(self):
        cfg, ref = config(), signal()
        for purpose, allowed, epoch in [('training', (1, 2, 3, 4), 1), ('internal_selection', (5, 6), 0), ('diagnostic', (5, 6), 0)]:
            for subject in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, True):
                if type(subject) is int and subject in allowed:
                    continue
                with self.subTest(purpose=purpose, subject=subject), self.assertRaises(c.V6Error):
                    c.corrupt_case(ref, 'Nakanishi2015', subject, 0, 'identity', cfg, purpose=purpose, epoch=epoch)
        for purpose, epoch in [('training', 0), ('training', 19), ('training', True), ('diagnostic', 1), ('internal_selection', -1), ('unknown', 0)]:
            with self.assertRaises(c.V6Error):
                c.corrupt_case(ref, 'Nakanishi2015', 1 if purpose == 'training' else 5, 0, 'identity', cfg, purpose=purpose, epoch=epoch)
        for trial in (-1, 180, True, 1.0):
            with self.assertRaises(c.V6Error):
                c.corrupt_case(ref, 'Nakanishi2015', 5, trial, 'identity', cfg, purpose='diagnostic')
        for reference in (np.zeros((8, 255)), ref.astype(np.float32), np.full((8, 256), np.nan)):
            with self.assertRaises(c.V6Error):
                c.corrupt_case(reference, 'Nakanishi2015', 5, 0, 'identity', cfg, purpose='diagnostic')

    def test_normalization_and_float64_residual_identity_restore(self):
        ref = signal() + np.arange(8, dtype=np.float64)[:, None] * 1e6
        n, mean, rms = c.normalize_window(ref)
        self.assertEqual(n.dtype, np.float32)
        self.assertEqual(mean.dtype, np.float64)
        self.assertEqual(rms.dtype, np.float64)
        np.testing.assert_array_equal(c.restore_window(ref, n, n.copy(), rms), ref)
        prediction = (n + np.float32(.001)).astype(np.float32)
        restored = c.restore_window(ref, n, prediction, rms)
        np.testing.assert_array_equal(restored, ref + (prediction.astype(np.float64) - n.astype(np.float64)) * rms)
        with self.assertRaises(c.V6Error):
            c.restore_window(ref, n, n.astype(np.float64), rms)
        with self.assertRaises(c.V6Error):
            c.restore_window(ref, n, n, np.zeros((8, 1), dtype=np.float64))

    def test_phase_zero_rotation_rectangular_mask_and_required_fs(self):
        ref = signal()
        metrics = c.case_metrics(ref, ref.copy(), ref.copy(), identity=True, sample_rate_hz=256)
        self.assertEqual(metrics['protected_complex_energy_ratio'], 0.0)
        self.assertEqual(metrics['protected_weighted_abs_phase_radians'], 0.0)
        self.assertEqual(metrics['undefined_phase_bin_count'], 0)
        self.assertEqual(metrics['identity_normalized_change'], 0.0)
        t = np.arange(256, dtype=np.float64)
        pure = np.tile(np.sin(2 * np.pi * 12 * t / 256), (8, 1))
        shifted = np.tile(np.sin(2 * np.pi * 12 * t / 256 + .1), (8, 1))
        m = c.case_metrics(pure, pure, shifted, identity=True, sample_rate_hz=256)
        self.assertAlmostEqual(m['protected_weighted_abs_phase_radians'], .1, places=12)
        self.assertAlmostEqual(m['protected_complex_energy_ratio'], 2 - 2 * math.cos(.1), places=12)
        high = np.tile(np.sin(2 * np.pi * 41 * t / 256), (8, 1))
        m = c.case_metrics(pure, pure, pure + high, identity=True, sample_rate_hz=250)
        self.assertLess(m['protected_complex_energy_ratio'], 1e-25)
        # At 250Hz bin 8 = 7.8125Hz is OUTSIDE; at 256Hz it is inclusive.
        low = np.tile(np.sin(2 * np.pi * 8 * t / 256), (8, 1))
        m250 = c.case_metrics(pure, pure, pure + low, identity=True, sample_rate_hz=250)
        m256 = c.case_metrics(pure, pure, pure + low, identity=True, sample_rate_hz=256)
        self.assertLess(m250['protected_complex_energy_ratio'], 1e-25)
        self.assertAlmostEqual(m256['protected_complex_energy_ratio'], 1., places=12)
        with self.assertRaises(TypeError):
            c.case_metrics(ref, ref, ref, identity=True)
        for fs in (True, 250.0, 128):
            with self.assertRaises(c.V6Error):
                c.case_metrics(ref, ref, ref, identity=True, sample_rate_hz=fs)

    def test_phase_empty_support_and_vanished_bin_are_undefined(self):
        ref = signal()
        m = c.case_metrics(ref, ref, np.zeros_like(ref), identity=True, sample_rate_hz=256)
        self.assertIsNone(m['protected_weighted_abs_phase_radians'])
        self.assertEqual(m['undefined_phase_bin_count'], 16)
        self.assertAlmostEqual(m['protected_complex_energy_ratio'], 1.)
        zero = np.zeros_like(ref)
        m = c.case_metrics(zero, zero, zero, identity=True, sample_rate_hz=256)
        self.assertIsNone(m['protected_complex_energy_ratio'])
        self.assertIsNone(m['protected_weighted_abs_phase_radians'])
        # One lost reference bin, while another remains: still fails phase.
        t = np.arange(256, dtype=np.float64)
        out = np.tile(np.sin(2*np.pi*12*t/256), (8, 1))
        m = c.case_metrics(ref, ref, out, identity=True, sample_rate_hz=256)
        self.assertIsNone(m['protected_weighted_abs_phase_radians'])
        self.assertEqual(m['undefined_phase_bin_count'], 8)
        json.dumps(m, allow_nan=False)
        r = {'dataset': 'Nakanishi2015', 'subject_id': 5, 'trial_index': 0, 'condition': 'identity',
             'seed': c._case_seed('Nakanishi2015', 5, 0, 'identity', 'diagnostic', 0, config()), **m}
        summary = c.summarize_cases([r])
        self.assertEqual(summary['identity']['undefined_phase_case_count'], 1)
        self.assertFalse(c.gate_summary(summary, config())['all_passed'])

    def test_nonidentity_metrics_and_finite_input_fail_closed(self):
        ref = signal()
        m = c.case_metrics(ref, ref + .5, ref + .25, sample_rate_hz=256)
        self.assertAlmostEqual(m['relative_reduction'], .75)
        self.assertIsNone(m['identity_normalized_change'])
        self.assertIsNone(m['protected_complex_energy_ratio'])
        self.assertIsNone(m['protected_weighted_abs_phase_radians'])
        self.assertEqual(m['undefined_phase_bin_count'], 0)
        with self.assertRaises(c.V6Error):
            c.case_metrics(ref, ref + .1, ref, identity=True, sample_rate_hz=256)
        with self.assertRaises(c.V6Error):
            c.case_metrics(ref, ref, np.full_like(ref, np.inf), sample_rate_hz=256)
        with self.assertRaises(c.V6Error):
            c.case_metrics(ref, ref, np.full_like(ref, 1e308), sample_rate_hz=256)


class SummaryGateTests(unittest.TestCase):
    def test_ratio_of_means_not_mean_reductions_and_exact_keys(self):
        summary = c.summarize_cases([row(a=1, b=.5), row(trial=1, a=9, b=9)])
        self.assertAlmostEqual(summary['transient']['ratio_of_mean_reduction'], .05)
        self.assertEqual(set(summary['transient']), {'case_count', 'mean_input_normalized_mse',
            'mean_output_normalized_mse', 'ratio_of_mean_reduction', 'mean_identity_normalized_change',
            'mean_protected_complex_energy_ratio', 'mean_protected_weighted_abs_phase_radians',
            'undefined_phase_case_count'})

    def test_mixed_normalized_description_requires_explicit_opt_in_and_partial_group_is_legal(self):
        rows = [row(a=1., b=.5), row(dataset='Wearable', a=9., b=9.)]
        rows[0].update(sample_rate_hz=256, native_raw_power=1e12)
        rows[1].update(sample_rate_hz=250, native_raw_power=1e-12)
        original = copy.deepcopy(rows)
        with self.assertRaises(c.V6Error):
            c.summarize_cases(rows)
        with self.assertRaises(c.V6Error):
            c.summarize_cases(rows, allow_mixed_datasets=False)
        summary = c.summarize_cases(rows, allow_mixed_datasets=True)
        self.assertEqual(set(summary), {'transient'})
        part = summary['transient']
        self.assertEqual(part['case_count'], 2)
        self.assertEqual(part['mean_input_normalized_mse'], 5.)
        self.assertEqual(part['mean_output_normalized_mse'], 4.75)
        self.assertAlmostEqual(part['ratio_of_mean_reduction'], .05)
        self.assertNotIn('native_raw_power', part)
        self.assertEqual(rows, original)
        self.assertEqual(c.summarize_cases([rows[0]], allow_mixed_datasets=True),
                         c.summarize_cases([rows[0]]))

    def test_mixed_identity_phase_uses_existing_native_fs_metrics_without_recomputing(self):
        t = np.arange(256, dtype=np.float64)
        ref = np.tile(np.sin(2 * np.pi * 12 * t / 256), (8, 1))
        output = ref + np.tile(np.sin(2 * np.pi * 8 * t / 256), (8, 1))
        rows = []
        for dataset in c.DATASETS:
            r = row(condition='identity', dataset=dataset)
            r.update(c.case_metrics(ref, ref, output, identity=True, sample_rate_hz=c.RATES[dataset]),
                     sample_rate_hz=c.RATES[dataset])
            rows.append(r)
        # Native fs changes the band mask: bin 8 included only at 256Hz.
        self.assertAlmostEqual(rows[0]['protected_complex_energy_ratio'], 1.)
        self.assertLess(rows[1]['protected_complex_energy_ratio'], 1e-25)
        with patch.object(c, 'case_metrics', side_effect=AssertionError('no metric recomputation')), \
             patch.object(np.fft, 'rfft', side_effect=AssertionError('no pooled spectrum')):
            part = c.summarize_cases(rows, allow_mixed_datasets=True)['identity']
        for source, target in [('output_normalized_mse', 'mean_output_normalized_mse'),
                               ('identity_normalized_change', 'mean_identity_normalized_change'),
                               ('protected_complex_energy_ratio', 'mean_protected_complex_energy_ratio'),
                               ('protected_weighted_abs_phase_radians', 'mean_protected_weighted_abs_phase_radians')]:
            self.assertEqual(part[target], (rows[0][source] + rows[1][source]) / 2.)
        self.assertAlmostEqual(part['mean_protected_complex_energy_ratio'], .5)
        self.assertEqual(part['undefined_phase_case_count'], 0)
        rows[1].update(protected_weighted_abs_phase_radians=None, undefined_phase_bin_count=1)
        part = c.summarize_cases(rows, allow_mixed_datasets=True)['identity']
        self.assertIsNone(part['mean_protected_weighted_abs_phase_radians'])
        self.assertEqual(part['undefined_phase_case_count'], 1)

    def test_mixed_opt_in_retains_identity_seed_rate_finite_and_strict_bool_checks(self):
        rows = [row(), row(dataset='Wearable')]
        for value in (0, 1, None, 'true', np.bool_(True)):
            with self.subTest(value=value), self.assertRaises(c.V6Error):
                c.summarize_cases([rows[0]], allow_mixed_datasets=value)
        for key, value in [('subject_id', 7), ('seed', rows[1]['seed'] + 100),
                           ('sample_rate_hz', 256), ('sample_rate_hz', 250.), ('sample_rate_hz', True),
                           ('sample_rate_hz', None), ('output_normalized_mse', float('nan'))]:
            bad = copy.deepcopy(rows)
            bad[1][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(c.V6Error):
                c.summarize_cases(bad, allow_mixed_datasets=True)
        with self.assertRaises(c.V6Error):
            c.summarize_cases(rows + [copy.deepcopy(rows[1])], allow_mixed_datasets=True)
        self.assertIn('DATASET SCOPE ONLY', c.gate_summary.__doc__)
        self.assertIn('MUST NOT', c.gate_summary.__doc__)

    def test_case_duplicates_seeds_subjects_and_consistency(self):
        r = row()
        with self.assertRaises(c.V6Error):
            c.summarize_cases([r, dict(r)])
        for key, value in [('seed', True), ('seed', r['seed'] + 100), ('subject_id', 7), ('subject_id', True),
                           ('trial_index', -1), ('relative_reduction', 0.), ('outcome', 'tie'),
                           ('input_normalized_mse', float('nan')), ('protected_complex_energy_ratio', 0.),
                           ('undefined_phase_bin_count', None)]:
            bad = dict(r)
            bad[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(c.V6Error):
                c.summarize_cases([bad])
        with self.assertRaises(c.V6Error):
            c.summarize_cases([r, row(dataset='Wearable')])
        with self.assertRaises(c.V6Error):
            c.summarize_cases([])
        # Valid subject subset and both legal internal seed purposes.
        self.assertEqual(c.summarize_cases([row(subject=6, purpose='internal_selection')])['transient']['case_count'], 1)
        self.assertEqual(c.summarize_cases([row(subject=2, purpose='training', epoch=18)])['transient']['case_count'], 1)
        r = row(condition='identity', b=0.)
        r.update(protected_weighted_abs_phase_radians=0.0, undefined_phase_bin_count=1)
        with self.assertRaises(c.V6Error):
            c.summarize_cases([r])

    def test_gate_rejects_invalid_summary_types_unknown_fields_and_negative_metrics(self):
        for field, value in [('case_count', True), ('mean_protected_complex_energy_ratio', -1.),
                             ('mean_identity_normalized_change', float('nan')),
                             ('undefined_phase_case_count', -1), ('mean_output_normalized_mse', False)]:
            summary = self.boundary_summary()
            summary['identity'][field] = value
            with self.subTest(field=field), self.assertRaises(c.V6Error):
                c.gate_summary(summary, config())
        summary = self.boundary_summary()
        summary['identity']['unknown'] = 0
        with self.assertRaises(c.V6Error):
            c.gate_summary(summary, config())
        summary = self.boundary_summary()
        summary['identity'] = []
        with self.assertRaises(c.V6Error):
            c.gate_summary(summary, config())

    def boundary_summary(self):
        rows = [row(condition='identity', b=.0005)] + [row(condition=condition, b=1.) for condition in c.CONDITIONS[1:]]
        rows[0].update(protected_complex_energy_ratio=1e-6, protected_weighted_abs_phase_radians=.02)
        summary = c.summarize_cases(rows)
        summary['high_frequency_burst']['ratio_of_mean_reduction'] = .05
        return summary

    def test_all_gate_boundaries_inclusive_and_any_failure_fails(self):
        cfg = config()
        summary = self.boundary_summary()
        gate = c.gate_summary(summary, cfg)
        self.assertTrue(gate['all_passed'])
        self.assertEqual(len(gate['gates']), 8)
        for condition, field, value in [
            ('identity', 'mean_identity_normalized_change', np.nextafter(.0005, math.inf)),
            ('identity', 'mean_protected_complex_energy_ratio', np.nextafter(1e-6, math.inf)),
            ('identity', 'mean_protected_weighted_abs_phase_radians', np.nextafter(.02, math.inf)),
            ('identity', 'undefined_phase_case_count', 1),
            ('high_frequency_burst', 'ratio_of_mean_reduction', np.nextafter(.05, -math.inf)),
            ('transient', 'ratio_of_mean_reduction', -1e-12),
            ('low_frequency_drift', 'ratio_of_mean_reduction', -1e-12),
            ('common_mode', 'ratio_of_mean_reduction', -1e-12)]:
            bad = copy.deepcopy(summary)
            bad[condition][field] = float(value) if type(value) is not int else value
            if field == 'mean_identity_normalized_change':
                bad[condition]['mean_output_normalized_mse'] = float(value)
            with self.subTest(condition=condition, field=field):
                self.assertFalse(c.gate_summary(bad, cfg)['all_passed'])
        for field in ('mean_protected_complex_energy_ratio', 'mean_protected_weighted_abs_phase_radians'):
            bad = copy.deepcopy(summary)
            bad['identity'][field] = None
            self.assertFalse(c.gate_summary(bad, cfg)['all_passed'])
        del summary['common_mode']
        self.assertFalse(c.gate_summary(summary, cfg)['all_passed'])


class PathPublicationTests(TemporaryTest):
    def test_lexical_paths_rejected_before_stat_and_safe_nested(self):
        invalid = ['', '..', 'a/../b', 'a\\..\\b', 'D:/forbidden', 'D:\\forbidden', 'C:relative',
                   '//server/share/file', '\\\\server\\share\\file', 'x.json:stream', 'x\x00y', 'NUL', 'CON.txt', 'ambiguous.']
        for text in invalid:
            with patch.object(Path, 'lstat', side_effect=AssertionError('must reject before stat')):
                with self.subTest(text=text), self.assertRaises(c.V6Error):
                    c.safe_path(self.root, text)
        self.assertEqual(c.safe_path(self.root, 'new/subdir/test.json'), self.root / 'new/subdir/test.json')
        with self.assertRaises(c.V6Error):
            c.safe_path(self.root, self.root.parent / 'outside.json')
        with self.assertRaises(c.V6Error):
            c.safe_path('D:/forbidden', 'file.json')

    def test_symlink_and_reparse_ancestors_rejected(self):
        target = self.root / 'target'
        target.mkdir()
        link = self.root / 'link'
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError:
            pass  # Windows may require symlink privilege; mocked reparse still tested.
        else:
            with self.assertRaises(c.V6Error):
                c.safe_path(self.root, 'link/file.json')
        actual = Path.lstat
        def reparse(path, *args, **kwargs):
            info = actual(path, *args, **kwargs)
            if path == target:
                return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400)
            return info
        with patch.object(Path, 'lstat', reparse), self.assertRaises(c.V6Error):
            c.safe_path(self.root, 'target/file.json')
        regular = self.root / 'regular'
        regular.write_bytes(b'x')
        with self.assertRaises(c.V6Error):
            c.safe_path(self.root, 'regular/child.json')

    def test_exclusive_publish_only_allowed_v6_and_existing_rejection(self):
        payloads = {c.CHECKPOINT_RELATIVE: b'inert', 'analysis_outputs/ssvep_v6_result.json': b'{}'}
        paths = c.publish_bundle(self.root, payloads)
        self.assertEqual([p.read_bytes() for p in paths], list(payloads.values()))
        for output in ['models/denoiser_v5_ssvep.pt', 'analysis_outputs/ssvep_v5.json',
                       'analysis_outputs/nested/ssvep_v6.json', 'configs/ssvep_v6.json', '../ssvep_v6.json']:
            with self.assertRaises(c.V6Error):
                c.assert_new_outputs(self.root, [output])
        with self.assertRaises(c.V6Error):
            c.publish_bundle(self.root, payloads)
        with self.assertRaises(c.V6Error):
            c.assert_new_outputs(self.root, ['analysis_outputs/ssvep_v6_x.json'] * 2)
        self.assertFalse(list(self.root.rglob('*.pending')))

    def test_racing_writer_second_link_partial_rollback_own_only(self):
        first, second = 'analysis_outputs/ssvep_v6_first.json', 'analysis_outputs/ssvep_v6_second.json'
        actual = os.link
        calls = 0
        def racing(source, target):
            nonlocal calls
            calls += 1
            if calls == 2:
                Path(target).write_bytes(b'competing writer')
            actual(source, target)
        with patch.object(c.os, 'link', side_effect=racing), self.assertRaises(FileExistsError):
            c.publish_bundle(self.root, {first: b'ours', second: b'ours second'})
        self.assertFalse((self.root / first).exists())
        self.assertEqual((self.root / second).read_bytes(), b'competing writer')
        self.assertFalse(list(self.root.rglob('*.pending')))

    def test_partial_failure_preserves_replaced_first_inode(self):
        first, second = 'analysis_outputs/ssvep_v6_first.json', 'analysis_outputs/ssvep_v6_second.json'
        actual = os.link
        calls = 0
        def racing(source, target):
            nonlocal calls
            calls += 1
            if calls == 2:
                (self.root / first).unlink()
                (self.root / first).write_bytes(b'new inode competitor')
                raise OSError('synthetic link failure')
            actual(source, target)
        with patch.object(c.os, 'link', side_effect=racing), self.assertRaises(OSError):
            c.publish_bundle(self.root, {first: b'ours', second: b'ours second'})
        self.assertEqual((self.root / first).read_bytes(), b'new inode competitor')
        self.assertFalse((self.root / second).exists())
        self.assertFalse(list(self.root.rglob('*.pending')))

    def test_first_link_race_and_fsync_failure_do_not_overwrite_or_leave_pending(self):
        target = 'analysis_outputs/ssvep_v6_first.json'
        actual = os.link
        def racing(source, path):
            Path(path).write_bytes(b'other writer')
            actual(source, path)
        with patch.object(c.os, 'link', side_effect=racing), self.assertRaises(FileExistsError):
            c.publish_bundle(self.root, {target: b'ours'})
        self.assertEqual((self.root / target).read_bytes(), b'other writer')
        target = 'analysis_outputs/ssvep_v6_fsync.json'
        with patch.object(c.os, 'fsync', side_effect=OSError('synthetic fsync fail')), self.assertRaises(OSError):
            c.publish_bundle(self.root, {target: b'ours'})
        self.assertFalse((self.root / target).exists())
        self.assertFalse(list(self.root.rglob('*.pending')))


if __name__ == '__main__':
    unittest.main()
