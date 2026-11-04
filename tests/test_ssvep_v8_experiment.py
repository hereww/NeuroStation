"""Synthetic CPU / mocked I/O and CUDA only. No NPZ/PT files are read.

Run with python -B -m unittest tests.test_ssvep_v8_experiment -v.
All synthetic locks remain in memory; no real lock/training/report is created.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from eeg_tools import ssvep_v8_experiment as api
from scripts import train_ssvep_denoiser_v8 as cli

CFG = json.loads(api._EXPECTED_CONFIG_JSON)
PROTOCOL = json.loads(api._EXPECTED_PREREG_JSON)
LOCK_SHA = 'a' * 64
ROOT = Path(__file__).absolute().parents[1]
# Authorized text metadata ONLY. Never read a source NPZ/PT or a real EEG file.
CFG_BYTES = (ROOT / api.CONFIG_RELATIVE).read_bytes()
PREREG_BYTES = (ROOT / api.PREREG_RELATIVE).read_bytes()


def synthetic_lock():
    files = dict(PROTOCOL['bound_files'])
    files[api.PREREG_RELATIVE] = api.PREREG_SHA256
    files.update(dict.fromkeys(api.NEW_SOURCE_PATHS, '1' * 64))
    return {'schema_version': 1, 'status': 'formal', 'stage': 'frozen_before_v8_training_waveform_read',
        'files': files, 'scope': {k: False for k in ('runtime_enabled', 'production_validated',
            'holdout_waveforms_loaded', 'outer_validation_waveforms_loaded', 'outer_validation_used_for_training_or_selection')},
        'checkpoint_loaded': False, 'waveforms_loaded': False, 'cuda_initialized': False}


def synthetic_state(check_only=False):
    return api.FrozenExperiment(ROOT, copy.deepcopy(CFG), copy.deepcopy(PROTOCOL),
                                synthetic_lock()['files'], LOCK_SHA, check_only)


@contextlib.contextmanager
def metadata_io(present=True, lock=None):
    content = {api.CONFIG_RELATIVE: CFG_BYTES, api.PREREG_RELATIVE: PREREG_BYTES,
        api.TRAIN_LOCK_RELATIVE: json.dumps(synthetic_lock() if lock is None else lock).encode('utf-8')}
    reads = []
    def read(root, rel):
        reads.append(str(rel))
        if str(rel) not in content:
            raise AssertionError('metadata-only attempted an unauthorized read: ' + str(rel))
        return content[str(rel)]
    def exists(path):
        return present if Path(path).name == Path(api.TRAIN_LOCK_RELATIVE).name else False
    with patch.object(api, '_read_bytes', side_effect=read), patch.object(api.os.path, 'lexists', side_effect=exists):
        yield reads, content


def synthetic_records():
    records = []
    for d in api.DOMAINS:
        t = np.arange(api.SHAPES[d][1], dtype=np.float64) / api.RATES[d]
        single = np.sin(2 * np.pi * 12 * t)[:, None] * np.arange(1, 9, dtype=np.float64)[None, :]
        # Broadcasted full contract shapes, without allocating real data/cohorts.
        signal = np.broadcast_to(single, api.SHAPES[d])
        for s in range(1, 7):
            rel = f'{api.REFERENCE_DIR}/{d}/subject_{s:03d}.npz'
            records.append(SimpleNamespace(dataset=d, subject_id=s, sample_rate_hz=api.RATES[d], signal=signal,
                npz_sha256=PROTOCOL['bound_files'][rel], sidecar_sha256=PROTOCOL['bound_files'][rel + '.manifest.json']))
    return records


def synthetic_case(d='Nakanishi2015', sid=5, trial=0, condition='identity', *, purpose='internal_selection', ratio=0.8):
    identity = condition == 'identity'
    return {'dataset': d, 'subject_id': sid, 'trial_index': trial, 'condition': condition,
        'purpose': purpose, 'epoch': 0, 'sample_rate_hz': api.RATES[d],
        'seed': api.case_seed(d, sid, trial, condition, CFG, purpose=purpose, epoch=0),
        'input_normalized_mse': 0.0 if identity else 1.0, 'output_normalized_mse': 0.0 if identity else ratio,
        'relative_reduction': None if identity else 1.0 - ratio, 'identity_normalized_change': 0.0 if identity else None,
        'outcome': 'tie' if identity else 'win' if ratio < 1 else 'loss' if ratio > 1 else 'tie',
        'protected_complex_energy_ratio': 0.0 if identity else None,
        'protected_weighted_abs_phase_radians': 0.0 if identity else None, 'undefined_phase_bin_count': 0}


def synthetic_cases(d, ratio=0.8):
    return [synthetic_case(d, s, t, c, ratio=ratio) for s in (5, 6) for t in range(api.SHAPES[d][0]) for c in api.CONDITIONS]


def synthetic_losses(cases):
    return {'total': 0.8, 'corrupted': 0.8, 'identity': 0.0, 'hf_low_branch_change': 0.0, 'change': 0.0,
        'corrupted_weight_sum': float(cases // 5 * 7), 'case_count': cases, 'corrupted_count': 4 * cases // 5,
        'identity_count': cases // 5, 'hf_count': cases // 5}


def synthetic_report():
    domains = {}
    for d in api.DOMAINS:
        cases = synthetic_cases(d)
        summary = api.summarize_cases(cases, CFG, domain=d)
        gates = api.gate_summary(summary, CFG)
        key = api.selection_key(summary, gates, 0.8, 1)
        n = api.SHAPES[d][0]
        history = [{'epoch': e, 'train_loss': synthetic_losses(n * 20), 'internal_loss': synthetic_losses(n * 10),
            'conditions': copy.deepcopy(summary), 'gates': copy.deepcopy(gates),
            'selection_key': list(api.selection_key(summary, gates, 0.8, e)), 'best_epoch': 1} for e in range(1, 19)]
        domains[d] = {'dataset': d, 'sample_rate_hz': api.RATES[d], 'fit_subjects': [1, 2, 3, 4],
            'internal_validation_subjects': [5, 6], 'fit_case_count': n * 20, 'internal_case_count': n * 10,
            'best_epoch': 1, 'best_selection_key': list(key), 'internal_gates_passed': gates['all_passed'],
            'selection_status': 'internal_gates_passed' if gates['all_passed'] else 'gates_failed',
            'history': history, 'best_case_rows': cases}
    return {**copy.deepcopy(api.REPORT_FIXED), 'created_at_utc': '2026-10-07T12:00:00+00:00',
        'training_lock_sha256': LOCK_SHA, 'source_binding': api._expected_source_binding(PROTOCOL['bound_files']),
        'domain_reports': domains, 'target_definition': CFG['corruption']['target_definition'],
        'development_history': PROTOCOL['development_history']}


def loss_tensors(conditions=None):
    conditions = list(api.CONDITIONS) if conditions is None else conditions
    n = len(conditions)
    target = torch.linspace(-1, 1, 256).reshape(1, 1, 256).expand(n, 8, 256).clone()
    noisy = target + 0.4
    identity = torch.tensor([c == 'identity' for c in conditions], dtype=torch.bool)
    noisy[identity] = target[identity]
    pred = target + 0.1
    aux = {'low_delta': torch.full_like(pred, 0.05), 'high_attenuation': torch.zeros(n, 8, 88)}
    return pred, target, noisy, identity, aux, conditions


class MetadataTests(unittest.TestCase):
    def test_frozen_metadata_hashes(self):
        self.assertEqual(hashlib.sha256(CFG_BYTES).hexdigest(), api.CONFIG_SHA256)
        self.assertEqual(hashlib.sha256(PREREG_BYTES).hexdigest(), api.PREREG_SHA256)
        api.validate_config(CFG)
        api.validate_preregistration(PROTOCOL, CFG)

    def test_exact_44_plus_1_plus_5(self):
        self.assertEqual(len(api.BOUND_PATHS), 56)
        self.assertEqual(len(api.TRAIN_LOCK_PATHS), 62)
        self.assertEqual(set(api.TRAIN_LOCK_PATHS), set(synthetic_lock()['files']))

    def test_check_only_no_lock_is_pending_not_training_permission(self):
        with metadata_io(False) as (reads, _):
            state = api.preflight()
        self.assertTrue(state.check_only)
        self.assertEqual(state.report()['readiness'], 'awaiting_62_entry_training_lock')
        self.assertEqual(set(reads), {api.CONFIG_RELATIVE, api.PREREG_RELATIVE})
        with self.assertRaises(api.V8Error):
            api._require_training_preflight(state)

    def test_check_only_existing_lock_reads_only_three_json(self):
        with metadata_io() as (reads, _), patch.object(api, '_torch', side_effect=AssertionError), patch.object(api, '_contract', side_effect=AssertionError):
            state = api.preflight()
            api.verify_frozen(state)
        self.assertEqual(set(reads), {api.CONFIG_RELATIVE, api.PREREG_RELATIVE, api.TRAIN_LOCK_RELATIVE})
        self.assertFalse(state.report()['binary_hashes_complete'])
        self.assertFalse(state.report()['cuda_initialized'])

    def test_run_without_lock_fails_before_contract_or_cuda(self):
        with metadata_io(False), patch.object(api, '_contract', side_effect=AssertionError), patch.object(api, '_torch', side_effect=AssertionError):
            with self.assertRaisesRegex(api.V8Error, 'lock required'):
                api.preflight(check_only=False)

    def test_wrong_stage_and_nonboolean_check_only(self):
        for kwargs in ({'stage': 'diagnostic'}, {'stage': 'runtime'}, {'stage': 'outer'}, {'check_only': 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(api.V8Error):
                api.preflight(**kwargs)

    def test_lock_wrong_stage(self):
        for stage in ('frozen_before_v6_training_waveform_read', 'frozen_before_v8_internal_diagnostic'):
            lock = synthetic_lock(); lock['stage'] = stage
            with self.subTest(stage=stage), self.assertRaises(api.V8Error):
                api.validate_training_lock(lock, PROTOCOL)

    def test_lock_requires_exact_62_no_outer_or_checkpoint(self):
        for change in ('missing', 'checkpoint', 'outer'):
            lock = synthetic_lock()
            if change == 'missing':
                del lock['files'][api.NEW_SOURCE_PATHS[0]]
            else:
                lock['files'][api.CHECKPOINT_RELATIVE if change == 'checkpoint' else 'outer/subject_007.npz'] = 'a' * 64
            with self.subTest(change=change), self.assertRaises(api.V8Error):
                api.validate_training_lock(lock, PROTOCOL)

    def test_lock_rejects_prereg_and_base_hash_changes(self):
        for p in (api.PREREG_RELATIVE, 'analysis_outputs/ssvep_v6_training_lock_20261007.json', api.CONFIG_RELATIVE):
            lock = synthetic_lock(); lock['files'][p] = '2' * 64
            with self.subTest(path=p), self.assertRaises(api.V8Error):
                api.validate_training_lock(lock, PROTOCOL)

    def test_lock_rejects_non_sha_types_and_uppercase(self):
        for value in ('A' * 64, 'f' * 63, True, None):
            lock = synthetic_lock(); lock['files'][api.NEW_SOURCE_PATHS[0]] = value
            with self.subTest(value=value), self.assertRaises(api.V8Error):
                api.validate_training_lock(lock, PROTOCOL)

    def test_nested_unsafe_scope_fails_closed(self):
        for flag in ('runtime_enabled', 'production', 'outer', 'holdout', 'waveforms_loaded', 'cuda_initialized', 'clean_ground_truth_used'):
            lock = synthetic_lock(); lock['extra'] = {'nested': {flag: True}}
            with self.subTest(flag=flag), self.assertRaises(api.V8Error):
                api.validate_training_lock(lock, PROTOCOL)

    def test_scope_boolean_is_not_integer_zero(self):
        lock = synthetic_lock(); lock['scope']['runtime_enabled'] = 0
        with self.assertRaises(api.V8Error): api.validate_training_lock(lock, PROTOCOL)

    def test_strict_json_duplicate_key(self):
        with self.assertRaises(api.V8Error): api._parse_json(b'{"a":1,"a":2}')

    def test_strict_json_nonfinite_and_not_object(self):
        for data in (b'{"a": NaN}', b'{"a": Infinity}', b'[]', b'{"a":1e999}', b'\xff'):
            with self.subTest(data=data), self.assertRaises(api.V8Error): api._parse_json(data)

    def test_config_no_mutation_or_overrides(self):
        for field, value in (('epochs', 19), ('device', 'cpu'), ('amp', True), ('allow_tf32', True)):
            cfg = copy.deepcopy(CFG); cfg['training'][field] = value
            with self.subTest(field=field), self.assertRaises(api.V8Error): api.validate_config(cfg)
        cfg = copy.deepcopy(CFG); cfg['model']['high_attenuation_mapping'] = 'sigmoid'
        with self.assertRaises(api.V8Error): api.validate_config(cfg)

    def test_verify_frozen_lock_changed_and_inmemory_state_changed(self):
        with metadata_io() as (_, content):
            state = api.preflight()
            content[api.TRAIN_LOCK_RELATIVE] += b' '
            with self.assertRaises(api.V8Error): api.verify_frozen(state)
        with metadata_io():
            state = api.preflight(); state.config['training']['epochs'] = 19
            with self.assertRaises(api.V8Error): api.verify_frozen(state)

    def test_full_freeze_source_hash_mismatch_before_data(self):
        with metadata_io(), patch.object(api, '_contract', side_effect=AssertionError):
            with self.assertRaises(AssertionError) as caught: api.preflight(check_only=False)
        self.assertIn('unauthorized read', str(caught.exception))

    def test_metadata_reader_refuses_npz_and_pt_before_stat(self):
        rel = f'{api.REFERENCE_DIR}/Nakanishi2015/subject_001.npz'
        with patch.object(api, '_stamp', side_effect=AssertionError):
            for p in (rel, api.CHECKPOINT_RELATIVE, 'D:/eeg/file.eeg'):
                with self.subTest(path=p), self.assertRaises(api.V8Error): api._read_bytes(ROOT, p)

    def test_default_cli_does_not_run(self):
        state = synthetic_state(True)
        with patch.object(api, 'assert_new_outputs') as guard, patch.object(api, 'preflight', return_value=state) as pre, patch.object(api, 'run_training', side_effect=AssertionError), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main([]), 0)
        guard.assert_called_once_with(cli.ROOT)
        pre.assert_called_once_with(root=cli.ROOT, check_only=True)

    def test_cli_explicit_check_only_same_default(self):
        with patch.object(api, 'assert_new_outputs'), patch.object(api, 'preflight', return_value=synthetic_state(True)) as pre, patch.object(api, 'run_training', side_effect=AssertionError), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(['--check-only']), 0)
        self.assertTrue(pre.call_args.kwargs['check_only'])

    def test_cli_forbidden_options_and_mutually_exclusive(self):
        for args in (['--run', '--check-only'], ['--device', 'cpu'], ['--runtime'], ['--outer'], ['--resume'], ['--config', 'x']):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.build_parser().parse_args(args)

    def test_cli_collision_precedes_preflight(self):
        with patch.object(api, 'assert_new_outputs', side_effect=api.V8Error('collision')), patch.object(api, 'preflight', side_effect=AssertionError), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(['--run']), 2)

    def test_fresh_import_no_torch_and_check_only_no_binary(self):
        code = "import sys; from eeg_tools import ssvep_v8_experiment as a; assert 'torch' not in sys.modules; s=a.preflight(); assert 'torch' not in sys.modules; assert not s.report()['waveforms_loaded']"
        # This subprocess calls metadata-only on the current workspace; no real
        # lock is needed, no binary I/O or CUDA is possible in this code path.
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class PathPublicationTests(unittest.TestCase):
    def test_foreign_paths_rejected_before_stat(self):
        paths = ['D:/eeg/x.json', 'd:x.json', '../x.json', r'a\..\x.json', r'\\server\share\x',
                 'x.json:stream', 'CON.json', 'a./x.json', 'a /x.json', 'x\x00.json']
        with patch.object(Path, 'lstat', side_effect=AssertionError):
            for p in paths:
                with self.subTest(path=p), self.assertRaises(api.V8Error): api.safe_path(ROOT, p)

    def test_outside_workspace_and_outer_holdout_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'workspace'; root.mkdir()
            for p in (str(Path(td) / 'elsewhere.json'), 'outer/a.json', 'holdout/a.json'):
                with self.subTest(path=p), self.assertRaises(api.V8Error): api.safe_path(root, p)

    def test_output_only_v8_analysis_reports_or_unique_checkpoint(self):
        for p in ('models/denoiser_v6_ssvep.pt', 'models/another_v8.pt', api.TRAIN_LOCK_RELATIVE,
                  api.PREREG_RELATIVE, 'analysis_outputs/ssvep_v8_lock_fake.json', 'analysis_outputs/ssvep_v8.csv',
                  'analysis_outputs/ssvep_v6_report.json', 'analysis_outputs/sub/ssvep_v8.json'):
            with self.subTest(path=p), self.assertRaises(api.V8Error): api._output_path(ROOT, p)
        api._output_path(ROOT, api.CHECKPOINT_RELATIVE)
        api._output_path(ROOT, 'analysis_outputs/ssvep_v8_diagnostic_20261007.json')
        api._output_path(ROOT, 'analysis_outputs/ssvep_v8_diagnostic_20261007.md')

    def test_symlink_or_reparse_rejected(self):
        fake = SimpleNamespace(st_mode=0o040755, st_file_attributes=0x400)
        with patch.object(Path, 'lstat', return_value=fake), self.assertRaises(api.V8Error):
            api.safe_path(ROOT, 'analysis_outputs/ssvep_v8.json')

    def test_publish_reports_exclusive_and_no_staged_debris(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); (root / 'analysis_outputs').mkdir()
            payloads = {'analysis_outputs/ssvep_v8_synthetic.json': b'{"synthetic":true}',
                        'analysis_outputs/ssvep_v8_synthetic.md': b'# Synthetic'}
            api.publish_bundle(root, payloads)
            for p, b in payloads.items(): self.assertEqual((root / p).read_bytes(), b)
            self.assertFalse(list(root.rglob('.ssvep-v8-stage-*')))
            with self.assertRaises(api.V8Error): api.publish_bundle(root, payloads)
            self.assertEqual((root / next(iter(payloads))).read_bytes(), next(iter(payloads.values())))

    def test_publish_race_no_clobber_rolls_back_only_own_first_link(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); (root / 'analysis_outputs').mkdir()
            p1, p2 = 'analysis_outputs/ssvep_v8_first.json', 'analysis_outputs/ssvep_v8_second.md'
            link = os.link
            def raced(source, dest):
                if Path(dest).name == Path(p2).name:
                    Path(dest).write_bytes(b'concurrent-owner')
                return link(source, dest)
            with patch.object(api.os, 'link', side_effect=raced), self.assertRaises(api.V8Error):
                api.publish_bundle(root, {p1: b'ours1', p2: b'ours2'})
            self.assertFalse((root / p1).exists())
            self.assertEqual((root / p2).read_bytes(), b'concurrent-owner')
            self.assertFalse(list(root.rglob('.ssvep-v8-stage-*')))

    def test_publish_existing_destination_rejected_without_staging(self):
        with patch.object(api.os.path, 'lexists', return_value=True), patch.object(api.tempfile, 'mkstemp', side_effect=AssertionError):
            with self.assertRaises(api.V8Error): api.publish_bundle(ROOT, {'analysis_outputs/ssvep_v8_x.json': b'x'})

    def test_publish_missing_parent_and_invalid_payloads(self):
        with tempfile.TemporaryDirectory() as td:
            for payload in ({}, {'analysis_outputs/ssvep_v8.json': ''}, {'analysis_outputs/ssvep_v8.json': b''}):
                with self.subTest(payload=payload), self.assertRaises(api.V8Error): api.publish_bundle(td, payload)
            with self.assertRaises(api.V8Error): api.publish_bundle(td, {'analysis_outputs/ssvep_v8_x.json': b'x'})

    def test_checkpoint_never_overwritten_or_opened_for_collision_check(self):
        with patch.object(api.os.path, 'lexists', return_value=True), patch.object(api.os, 'open', side_effect=AssertionError):
            with self.assertRaises(api.V8Error): api.assert_new_outputs(ROOT)

    def test_output_case_aliases_rejected(self):
        payload = {'analysis_outputs/ssvep_v8_alias.json': b'a', 'analysis_outputs/ssvep_v8_alias.JSON': b'b'}
        with self.assertRaises(api.V8Error): api.publish_bundle(ROOT, payload)


class SeedDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.records = synthetic_records()

    def test_seed_exact_large_integer_and_epoch_formula(self):
        expected = CFG['seeds']['training'] + 10000000 + 4 * 100000 + 239 * 10 + 3 + 18 * 100000000
        self.assertEqual(api.case_seed('Wearable', 4, 239, 'high_frequency_burst', CFG, purpose='training', epoch=18), expected)
        self.assertGreater(expected, 2 ** 32)

    def test_selection_and_diagnostic_seed_purposes_disjoint(self):
        a = api.case_seed('Wearable', 5, 0, 'identity', CFG, purpose='internal_selection', epoch=0)
        b = api.case_seed('Wearable', 5, 0, 'identity', CFG, purpose='diagnostic', epoch=0)
        self.assertEqual(b - a, CFG['seeds']['diagnostic'] - CFG['seeds']['internal_selection'])
        self.assertNotEqual(a, b)

    def test_subject_allowlist_no_outer_holdout_or_selection_training_mix(self):
        for purpose, sid in (('training', 5), ('training', 7), ('internal_selection', 1), ('diagnostic', 9), ('diagnostic', True)):
            with self.subTest(purpose=purpose, sid=sid), self.assertRaises(api.V8Error):
                api.case_seed('Nakanishi2015', sid, 0, 'identity', CFG, purpose=purpose, epoch=1 if purpose == 'training' else 0)

    def test_illegal_epochs_trials_and_condition_combination(self):
        for epoch in (0, 19, True, 1.0):
            with self.subTest(epoch=epoch), self.assertRaises(api.V8Error):
                api.case_seed('Nakanishi2015', 1, 0, 'identity', CFG, purpose='training', epoch=epoch)
        for trial in (-1, 180, True):
            with self.subTest(trial=trial), self.assertRaises(api.V8Error):
                api.case_seed('Nakanishi2015', 5, trial, 'identity', CFG, purpose='internal_selection', epoch=0)
        for purpose, epoch, cond in (('internal_selection', 1, 'identity'), ('diagnostic', 18, 'identity'), ('runtime', 0, 'identity'),
                                     ('internal_selection', 0, 'transient+common_mode')):
            with self.subTest(purpose=purpose), self.assertRaises(api.V8Error):
                api.case_seed('Nakanishi2015', 5, 0, cond, CFG, purpose=purpose, epoch=epoch)

    def test_training_internal_exact_case_counts_and_five_conditions(self):
        for d, expected in (('Nakanishi2015', (3600, 1800)), ('Wearable', (4800, 2400))):
            fit = api.ReferenceWindowDataset(self.records, d, CFG, purpose='training', epoch=1)
            val = api.ReferenceWindowDataset(self.records, d, CFG, purpose='internal_selection', epoch=0)
            self.assertEqual((len(fit), len(val)), expected)
            self.assertEqual([e[2] for e in fit.entries[:5]], list(api.CONDITIONS))
            self.assertEqual([r.subject_id for r in fit.records], [1, 2, 3, 4])
            self.assertEqual([r.subject_id for r in val.records], [5, 6])

    def test_dataset_fixed_window_no_data_io(self):
        ds = api.ReferenceWindowDataset(self.records, 'Wearable', CFG, purpose='training', epoch=1)
        with patch.object(api, '_read_bytes', side_effect=AssertionError), patch.object(np, 'load', side_effect=AssertionError):
            item = ds[0]
        r = ds.records[0]
        self.assertTrue(np.array_equal(item['ref64'], r.signal[0, 125:381].T))
        self.assertTrue(np.array_equal(item['noisy64'], item['ref64']))
        self.assertEqual(item['noisy'].dtype, np.float32)
        self.assertEqual(item['target'].dtype, np.float32)
        self.assertEqual(item['ref64'].dtype, np.float64)

    def test_case_corruption_identity_copy_and_single_family(self):
        ref = np.tile(np.linspace(-1, 1, 256), (8, 1)).astype(np.float64)
        out, _ = api.corrupt_case(ref, 'Nakanishi2015', 1, 0, 'identity', CFG, purpose='training', epoch=1)
        self.assertTrue(np.array_equal(out, ref)); self.assertIsNot(out, ref)
        for c in api.CONDITIONS[1:]:
            out, seed = api.corrupt_case(ref, 'Nakanishi2015', 1, 0, c, CFG, purpose='training', epoch=1)
            again, _ = api.corrupt_case(ref, 'Nakanishi2015', 1, 0, c, CFG, purpose='training', epoch=1)
            self.assertTrue(np.array_equal(out, again))
            self.assertGreater(seed, 2 ** 32)
        self.assertEqual(CFG['corruption']['combine_probability'], 0.0)

    def test_record_contract_rejects_duplicate_outer_dtype_and_bad_provenance(self):
        for mode in ('duplicate', 'outer', 'dtype', 'hash', 'rate'):
            records = [copy.copy(r) for r in self.records]
            if mode == 'duplicate': records[-1] = records[0]
            if mode == 'outer': records[0].subject_id = 7
            if mode == 'dtype': records[0].signal = records[0].signal.astype(np.float32)
            if mode == 'hash': records[0].npz_sha256 = 'z' * 64
            if mode == 'rate': records[0].sample_rate_hz = 250
            with self.subTest(mode=mode), self.assertRaises(api.V8Error):
                api.validate_records(records, CFG, files=synthetic_lock()['files'])

    def test_dataset_rejects_diagnostic_and_incomplete_cohort(self):
        with self.assertRaises(api.V8Error): api.ReferenceWindowDataset(self.records, 'Wearable', CFG, purpose='diagnostic', epoch=0)
        with self.assertRaises(api.V8Error): api.ReferenceWindowDataset(self.records[:-1], 'Wearable', CFG, purpose='training', epoch=1)

    def test_set_seed_truncates_only_numpy_overall_seed(self):
        seed = 7026107701
        fake = SimpleNamespace(manual_seed=Mock(), cuda=SimpleNamespace(manual_seed_all=Mock()))
        with patch.object(api, '_torch', return_value=fake), patch.object(api.random, 'seed') as rand, patch.object(np.random, 'seed') as nr:
            api.set_seed(seed)
        rand.assert_called_once_with(seed); nr.assert_called_once_with(seed % (2 ** 32))
        fake.manual_seed.assert_called_once_with(seed); fake.cuda.manual_seed_all.assert_called_once_with(seed)


class LossTests(unittest.TestCase):
    def test_loss_exact_weighted_equation_and_sufficient_statistics(self):
        pred, target, noisy, identity, aux, cond = loss_tensors()
        loss = api.compute_loss(pred, target, noisy, identity, aux, CFG, conditions=cond)
        var = (target - target.mean(-1, keepdim=True)).square().mean(-1, keepdim=True) + 1e-8
        out = ((pred - target).square() / var).mean((1, 2))
        base = ((noisy - target).square() / var).mean((1, 2))
        expected = (out[1] / base[1] + out[2] / base[2] + 4 * out[3] / base[3] + out[4] / base[4]) / 7
        self.assertAlmostEqual(float(loss['corrupted']), float(expected), places=6)
        self.assertEqual(loss['corrupted_weight_sum'], 7.0)
        self.assertEqual(loss['hf_count'], 1)
        self.assertAlmostEqual(float(loss['hf_low_branch_change']), 0.05 ** 2, places=7)
        expected_total = float(expected + 60 * out[0] + 20 * loss['hf_low_branch_change'] + .001 * loss['change'])
        self.assertAlmostEqual(float(loss['total']), expected_total, places=6)

    def test_identity_never_input_ratio_division(self):
        p, t, x, ident, aux, c = loss_tensors(['identity'])
        loss = api.compute_loss(p, t, x, ident, aux, CFG, conditions=c)
        self.assertEqual(float(loss['corrupted']), 0.0)
        self.assertAlmostEqual(float(loss['total']), 60 * float(loss['identity']) + .001 * float(loss['change']), places=6)

    def test_hf_low_penalty_not_divided_again_by_target_variance(self):
        p, t, x, ident, aux, c = loss_tensors(['high_frequency_burst'])
        t = t * 100; x = t + 0.4; p = t + 0.1
        loss = api.compute_loss(p, t, x, ident, aux, CFG, conditions=c)
        self.assertAlmostEqual(float(loss['hf_low_branch_change']), 0.0025, places=7)

    def test_all_other_conditions_and_no_identity_or_hf_masks(self):
        p, t, x, ident, aux, c = loss_tensors(['transient', 'common_mode'])
        loss = api.compute_loss(p, t, x, ident, aux, CFG, conditions=c)
        self.assertEqual(loss['corrupted_weight_sum'], 2.0)
        self.assertEqual(float(loss['identity']), 0.0)
        self.assertEqual(float(loss['hf_low_branch_change']), 0.0)

    def test_relative_mse_floor(self):
        p, t, x, ident, aux, c = loss_tensors(['transient'])
        x = t.clone()
        loss = api.compute_loss(p, t, x, ident, aux, CFG, conditions=c)
        var = (t - t.mean(-1, keepdim=True)).square().mean(-1, keepdim=True) + 1e-8
        expected = ((p - t).square() / var).mean() / .01
        self.assertAlmostEqual(float(loss['corrupted']), float(expected), places=5)

    def test_condition_identity_inconsistency_and_combinations_rejected(self):
        p, t, x, ident, aux, c = loss_tensors()
        for conditions in (['transient'] * 5, list(c[:-1]), ['identity', 'transient+common_mode', *c[2:]]):
            with self.subTest(conditions=conditions), self.assertRaises(api.V8Error):
                api.compute_loss(p, t, x, ident, aux, CFG, conditions=conditions)

    def test_aux_gate_accepts_per_frequency_rejects_amplification_and_window_scalar(self):
        p, t, x, ident, aux, c = loss_tensors()
        api.compute_loss(p, t, x, ident, aux, CFG, conditions=c)
        for high in (torch.full((5, 8, 88), -.01), torch.full((5, 8, 88), .81), torch.zeros(5, 8, 1)):
            a = dict(aux); a['high_attenuation'] = high
            with self.subTest(shape=high.shape), self.assertRaises(api.V8Error):
                api.compute_loss(p, t, x, ident, a, CFG, conditions=c)

    def test_loss_rejects_nonfinite_bad_dtype_shape_and_aux(self):
        p, t, x, ident, aux, c = loss_tensors()
        bad = p.clone(); bad[0, 0, 0] = float('nan')
        for pred in (bad, p.double(), p[:, :, :128], p[:0]):
            with self.subTest(shape=pred.shape), self.assertRaises(api.V8Error): api.compute_loss(pred, t, x, ident, aux, CFG, conditions=c)
        with self.assertRaises(api.V8Error): api.compute_loss(p, t, x, ident.float(), aux, CFG, conditions=c)
        with self.assertRaises(api.V8Error): api.compute_loss(p, t, x, ident, {'low_delta': p}, CFG, conditions=c)

    def test_loss_keeps_low_aux_gradient(self):
        p, t, x, ident, aux, c = loss_tensors(['high_frequency_burst'])
        aux['low_delta'].requires_grad_(); p.requires_grad_()
        loss = api.compute_loss(p, t, x, ident, aux, CFG, conditions=c)
        loss['total'].backward()
        self.assertGreater(float(aux['low_delta'].grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_accumulator_global_weighted_sums_not_mean_of_batch_means(self):
        p, t, x, ident, aux, c = loss_tensors()
        acc = api.LossAccumulator(CFG)
        for indices in ([0], [1, 3], [2, 4]):
            a = {k: v[indices] for k, v in aux.items()}
            acc.update(api.compute_loss(p[indices], t[indices], x[indices], ident[indices], a, CFG, conditions=[c[i] for i in indices]))
        result = acc.result()
        full = api.compute_loss(p, t, x, ident, aux, CFG, conditions=c)
        for k in ('corrupted', 'identity', 'hf_low_branch_change', 'change', 'total'):
            self.assertAlmostEqual(result[k], float(full[k]), places=6)
        self.assertEqual(result['case_count'], 5); self.assertEqual(result['corrupted_weight_sum'], 7.0)

    def test_empty_accumulator_fails(self):
        with self.assertRaises(api.V8Error): api.LossAccumulator(CFG).result()


class TinyModel(torch.nn.Module):
    """Synthetic model that asserts the only forward argument is x."""
    def __init__(self):
        super().__init__()
        self.delta = torch.nn.Parameter(torch.zeros(8))
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x)
        return x + self.delta[None, :, None]

    def forward_with_aux(self, x):
        return self.forward(x), {'low_delta': self.delta[None, :, None].expand_as(x),
                                'high_attenuation': x.new_zeros(x.shape[0], 8, 88)}


def synthetic_batch(purpose='internal_selection'):
    ds = api.ReferenceWindowDataset(synthetic_records(), 'Nakanishi2015', CFG,
        purpose=purpose, epoch=1 if purpose == 'training' else 0)
    return torch.utils.data.default_collate([ds[i] for i in range(5)])


def fake_runtime(available=True, initialized=False):
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: available, is_initialized=lambda: initialized,
        set_device=Mock()), set_num_threads=Mock(), use_deterministic_algorithms=Mock(),
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
            cudnn=SimpleNamespace(allow_tf32=True, benchmark=True, deterministic=False)))


class RunnerSelectionTests(unittest.TestCase):
    def test_forward_with_aux_receives_x_only_and_reuses_v6_C_metrics_restore(self):
        batch = synthetic_batch()
        model = TinyModel()
        v6 = api._contract()
        with patch.object(v6, 'restore_window', wraps=v6.restore_window) as restore, patch.object(v6, 'case_metrics', wraps=v6.case_metrics) as metrics, patch.object(api, '_read_bytes', side_effect=AssertionError), patch.object(np, 'load', side_effect=AssertionError):
            losses, rows = api.run_epoch(model, [batch], CFG, 'cpu')
        self.assertEqual(len(model.inputs), 1)
        self.assertEqual(model.inputs[0].shape, (5, 8, 256))
        self.assertEqual(restore.call_count, 5); self.assertEqual(metrics.call_count, 5)
        self.assertEqual(losses['case_count'], 5)
        self.assertTrue(all(r['purpose'] == 'internal_selection' and r['epoch'] == 0 for r in rows))
        self.assertEqual(rows[0]['seed'], CFG['seeds']['internal_selection'] + 500000)
        self.assertEqual(rows[0]['sample_rate_hz'], 256)

    def test_synthetic_optimizer_batch_no_labels_in_forward(self):
        model = TinyModel(); optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        losses, rows = api.run_epoch(model, [synthetic_batch('training')], CFG, 'cpu', optimizer)
        self.assertEqual(rows, [])
        self.assertEqual(losses['case_count'], 5)
        self.assertEqual(len(model.inputs), 1)
        self.assertTrue(torch.isfinite(model.delta).all())

    def test_batch_seed_rate_purpose_outer_rejected_before_forward(self):
        for field, value in (('seed', 1), ('sample_rate_hz', 250), ('subject_id', 7), ('epoch', 1)):
            batch = synthetic_batch(); batch[field][0] = value
            model = TinyModel()
            with self.subTest(field=field), self.assertRaises(api.V8Error): api.run_epoch(model, [batch], CFG, 'cpu')
            self.assertEqual(model.inputs, [])
        batch = synthetic_batch(); batch['purpose'] = ['diagnostic'] * 5
        with self.assertRaises(api.V8Error): api.run_epoch(TinyModel(), [batch], CFG, 'cpu')

    def test_batch_float_bool_identity_and_duplicate_scope_rejected(self):
        for dtype in (torch.float32, torch.bool):
            batch = synthetic_batch(); batch['subject_id'] = batch['subject_id'].to(dtype)
            with self.subTest(dtype=dtype), self.assertRaises(api.V8Error): api.run_epoch(TinyModel(), [batch], CFG, 'cpu')
        batch = synthetic_batch()
        with self.assertRaisesRegex(api.V8Error, 'duplicate'): api.run_epoch(TinyModel(), [batch, batch], CFG, 'cpu')

    def test_actual_training_cpu_and_check_only_fail_before_data_or_cuda(self):
        with patch.object(api, '_torch', side_effect=AssertionError), patch.object(api, 'validate_frozen', side_effect=AssertionError):
            with self.assertRaises(api.V8Error): api.train_domain([], 'Wearable', CFG, 'cpu', frozen_state=synthetic_state())
            with self.assertRaises(api.V8Error): api.run_training(synthetic_state(True))
            with self.assertRaises(api.V8Error): api.load_training_records(synthetic_state(True))
            with self.assertRaises(api.V8Error): api.publish_training_bundle(synthetic_state(True), {}, {})

    def test_fixed_cuda_environment_and_no_cpu_fallback(self):
        fake = fake_runtime()
        with patch.dict(os.environ, {}, clear=False), patch.object(api, '_torch', return_value=fake), patch.object(api, 'set_seed') as seed:
            os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
            self.assertEqual(api.configure_training(CFG), 'cuda:0')
            self.assertEqual(os.environ['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')
        fake.cuda.set_device.assert_called_once_with(0)
        fake.set_num_threads.assert_called_once_with(4)
        fake.use_deterministic_algorithms.assert_called_once_with(True)
        seed.assert_called_once_with(CFG['training']['seed'])
        self.assertFalse(fake.backends.cuda.matmul.allow_tf32); self.assertFalse(fake.backends.cudnn.allow_tf32)
        self.assertTrue(fake.backends.cudnn.deterministic); self.assertFalse(fake.backends.cudnn.benchmark)
        for f in (fake_runtime(False), fake_runtime(True, True)):
            with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}), patch.object(api, '_torch', return_value=f), self.assertRaises(api.V8Error):
                api.configure_training(CFG)

    def test_conflicting_cublas_fails_before_torch_import(self):
        with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':16:8'}), patch.object(api, '_torch', side_effect=AssertionError), self.assertRaises(api.V8Error):
            api.configure_training(CFG)

    def test_load_records_verify_then_v6_contract_order_mock_only(self):
        events = []
        records = synthetic_records()
        state = synthetic_state()
        base = SimpleNamespace(files={p: h for p, h in state.files.items() if p in api.BOUND_PATHS})
        contract = SimpleNamespace(preflight=Mock(side_effect=lambda **kw: (events.append('v6_preflight'), base)[1]),
            verify_frozen=Mock(side_effect=lambda s: events.append('v6_verify')),
            load_training_records=Mock(side_effect=lambda s: (events.append('v6_load_records'), records)[1]))
        with patch.object(api, 'validate_frozen', side_effect=lambda s: events.append('v8_verify')), patch.object(api, '_contract', return_value=contract):
            self.assertIs(api.load_training_records(state), records)
        self.assertEqual(events[:2], ['v8_verify', 'v6_preflight'])
        self.assertLess(events.index('v8_verify'), events.index('v6_load_records'))
        self.assertEqual(events[-1], 'v8_verify')
        contract.preflight.assert_called_once_with(root=state.root, stage='training', check_only=False)

    def test_failed_v8_freeze_never_calls_v6_or_loads_records(self):
        with patch.object(api, 'validate_frozen', side_effect=api.V8Error('changed')), patch.object(api, '_contract', side_effect=AssertionError):
            with self.assertRaises(api.V8Error): api.load_training_records(synthetic_state())

    def test_summary_adapter_preserves_v8_saved_seed_and_provenance(self):
        row = synthetic_case(); before = copy.deepcopy(row)
        summary = api.summarize_cases([row], CFG, domain='Nakanishi2015', complete=False)
        self.assertEqual(row, before)
        self.assertEqual(summary['identity']['case_count'], 1)
        self.assertGreater(row['seed'], 2 ** 32)

    def test_summary_requires_complete_C_no_fit_mixed_domain_or_duplicates(self):
        row = synthetic_case()
        with self.assertRaises(api.V8Error): api.summarize_cases([row], CFG, domain='Nakanishi2015')
        with self.assertRaises(api.V8Error): api.summarize_cases([row, row], CFG, domain='Nakanishi2015', complete=False)
        other = synthetic_case('Wearable')
        with self.assertRaises(api.V8Error): api.summarize_cases([row, other], CFG, domain='Nakanishi2015', complete=False)
        row['subject_id'] = 1
        with self.assertRaises(api.V8Error): api.summarize_cases([row], CFG, domain='Nakanishi2015', complete=False)

    def test_summary_cannot_relabel_diagnostic_seed_as_selection(self):
        row = synthetic_case(purpose='diagnostic')
        with self.assertRaises(api.V8Error): api.summarize_cases([row], CFG, domain='Nakanishi2015', complete=False)

    def test_selection_eligible_first_then_worst_ratio_identity_loss_epoch(self):
        rows = [synthetic_case(condition=c) for c in api.CONDITIONS]
        s = api.summarize_cases(rows, CFG, domain='Nakanishi2015', complete=False)
        g = api.gate_summary(s, CFG)
        best = api.selection_key(s, g, 2.0, 2)
        failed = api.selection_key(s, {**g, 'all_passed': False}, .01, 1)
        self.assertLess(best, failed)
        self.assertLess(api.selection_key(s, g, 2.0, 1), best)
        worse = copy.deepcopy(s); worse['transient']['mean_output_normalized_mse'] = 2.0
        self.assertLess(best, api.selection_key(worse, g, 0.0, 1))
        worse = copy.deepcopy(s); worse['identity']['mean_output_normalized_mse'] = .01
        self.assertLess(best, api.selection_key(worse, g, .01, 1))
        self.assertLess(api.selection_key(s, g, 1.0, 18), best)

    def test_selection_no_epoch_zero_extension_nonfinite_or_missing_conditions(self):
        rows = [synthetic_case(condition=c) for c in api.CONDITIONS]
        s = api.summarize_cases(rows, CFG, domain='Nakanishi2015', complete=False); g = api.gate_summary(s, CFG)
        for epoch in (0, 19, True):
            with self.subTest(epoch=epoch), self.assertRaises(api.V8Error): api.selection_key(s, g, 1.0, epoch)
        with self.assertRaises(api.V8Error): api.selection_key(s, g, float('inf'), 1)
        s.pop('transient')
        with self.assertRaises(api.V8Error): api.selection_key(s, g, 1.0, 1)

    def test_selection_ratio_of_means_not_mean_of_case_ratios(self):
        a = synthetic_case(condition='transient', ratio=.8)
        b = synthetic_case(trial=1, condition='transient', ratio=.8)
        a.update(input_normalized_mse=1.0, output_normalized_mse=.1, relative_reduction=.9)
        b.update(input_normalized_mse=9.0, output_normalized_mse=8.1, relative_reduction=(9.-8.1)/9.)
        s = api.summarize_cases([a, b], CFG, domain='Nakanishi2015', complete=False)
        self.assertAlmostEqual(s['transient']['ratio_of_mean_reduction'], 1 - 8.2/10)
        self.assertNotAlmostEqual(s['transient']['ratio_of_mean_reduction'], .5)

    def test_undefined_identity_phase_never_filled_as_zero_or_passed(self):
        rows = [synthetic_case(condition=c) for c in api.CONDITIONS]
        rows[0]['undefined_phase_bin_count'] = 1; rows[0]['protected_weighted_abs_phase_radians'] = None
        s = api.summarize_cases(rows, CFG, domain='Nakanishi2015', complete=False)
        self.assertIsNone(s['identity']['mean_protected_weighted_abs_phase_radians'])
        self.assertFalse(api.gate_summary(s, CFG)['all_passed'])


class CheckpointReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.report = synthetic_report()

    def checkpoint(self):
        ck = api.checkpoint_metadata(CFG, self.report, LOCK_SHA)
        ck['models'] = {d: {k: v.detach().clone() for k, v in TinyModel().state_dict().items()} for d in api.DOMAINS}
        return ck

    def test_complete_report_and_public_three_arg_schema(self):
        api.validate_training_report(self.report, CFG, LOCK_SHA)
        self.assertEqual(self.report['domain_reports']['Nakanishi2015']['internal_case_count'], 1800)
        self.assertEqual(self.report['domain_reports']['Wearable']['internal_case_count'], 2400)
        self.assertEqual(set(api.checkpoint_metadata(CFG, self.report, LOCK_SHA)), api.CHECKPOINT_FIELDS - {'models'})

    def test_report_strict_flags_source_hash_target_definition(self):
        for k, v in (('runtime_enabled', True), ('clean_ground_truth_used', True), ('target_definition', 'clean EEG'),
                     ('internal_subjects_are_development_data', False), ('training_lock_sha256', 'b' * 64)):
            report = copy.deepcopy(self.report); report[k] = v
            with self.subTest(field=k), self.assertRaises(api.V8Error): api.validate_training_report(report, CFG, LOCK_SHA)
        report = copy.deepcopy(self.report); report['source_binding'][0]['npz_sha256'] = 'b' * 64
        with self.assertRaises(api.V8Error): api.validate_training_report(report, CFG, LOCK_SHA)

    def test_report_cannot_claim_epoch_zero_or_extension(self):
        for mode in ('extend', 'zero', 'best_epoch', 'loss_weight'):
            report = copy.deepcopy(self.report); d = report['domain_reports']['Wearable']
            if mode == 'extend': d['history'].append(copy.deepcopy(d['history'][-1]))
            if mode == 'zero': d['history'][0]['epoch'] = 0
            if mode == 'best_epoch': d['best_epoch'] = 2
            if mode == 'loss_weight': d['history'][0]['train_loss']['corrupted_weight_sum'] = 1.0
            with self.subTest(mode=mode), self.assertRaises(api.V8Error): api.validate_training_report(report, CFG, LOCK_SHA)

    def test_report_requires_complete_selected_native_C_rows(self):
        for mode in ('missing', 'seed', 'domain', 'summary'):
            report = copy.deepcopy(self.report); d = report['domain_reports']['Nakanishi2015']
            if mode == 'missing': d['best_case_rows'].pop()
            if mode == 'seed': d['best_case_rows'][0]['seed'] += 1
            if mode == 'domain': d['best_case_rows'][0]['dataset'] = 'Wearable'
            if mode == 'summary': d['best_case_rows'][1]['output_normalized_mse'] = .7
            with self.subTest(mode=mode), self.assertRaises(ValueError): api.validate_training_report(report, CFG, LOCK_SHA)

    def test_checkpoint_CPU_eval_shapes_and_RNG_restored(self):
        ck = self.checkpoint()
        before = torch.get_rng_state().clone()
        builds = []
        def build(cfg, fs): builds.append(fs); return TinyModel()
        with patch.object(api, '_model_api', return_value=SimpleNamespace(build_model=build)):
            models = api.validate_checkpoint(ck, CFG, LOCK_SHA)
        self.assertEqual(builds, [256, 250]); self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(set(models), set(api.DOMAINS))
        self.assertTrue(all(not m.training and next(m.parameters()).device.type == 'cpu' for m in models.values()))

    def test_checkpoint_rejects_keys_metadata_runtime_and_state_schema(self):
        for k, value in (('model_version', 'learned_denoise_v6_ssvep'), ('config_sha256', '1' * 64),
                         ('runtime_enabled', True), ('state_schema', {})):
            ck = self.checkpoint(); ck[k] = value
            with self.subTest(key=k), self.assertRaises(api.V8Error): api.validate_checkpoint(ck, CFG, LOCK_SHA)
        ck = self.checkpoint(); ck['optimizer_state'] = {}
        with self.assertRaises(api.V8Error): api.validate_checkpoint(ck, CFG, LOCK_SHA)

    def test_checkpoint_rejects_missing_domain(self):
        ck = self.checkpoint(); del ck['models']['Wearable']
        with self.assertRaises(api.V8Error): api.validate_checkpoint(ck, CFG, LOCK_SHA)

    def test_checkpoint_rejects_nonfinite_dtype_shape_grad_and_extra_state_keys(self):
        with patch.object(api, '_model_api', return_value=SimpleNamespace(build_model=lambda c, fs: TinyModel())):
            for tensor in (torch.full((8,), float('nan')), torch.zeros(8, dtype=torch.float64),
                           torch.zeros(7), torch.zeros(8, requires_grad=True)):
                ck = self.checkpoint(); ck['models']['Nakanishi2015']['delta'] = tensor
                with self.subTest(shape=tensor.shape), self.assertRaises(api.V8Error): api.validate_checkpoint(ck, CFG, LOCK_SHA)
            ck = self.checkpoint(); ck['models']['Nakanishi2015']['unexpected'] = torch.zeros(1)
            with self.assertRaises(api.V8Error): api.validate_checkpoint(ck, CFG, LOCK_SHA)

    def test_buffer_loader_CPU_weights_only_strict_no_path_load(self):
        ck = self.checkpoint(); buffer = io.BytesIO(); torch.save(ck, buffer)
        real_load = torch.load
        with patch.object(api, '_model_api', return_value=SimpleNamespace(build_model=lambda c, fs: TinyModel())), patch.object(torch, 'load', wraps=real_load) as loader:
            loaded, models = api.load_checkpoint_cpu(buffer, CFG, LOCK_SHA)
        self.assertEqual(set(models), set(api.DOMAINS)); self.assertEqual(loaded['model_version'], api.MODEL_VERSION)
        self.assertIsInstance(loader.call_args.args[0], io.BytesIO)
        self.assertEqual(loader.call_args.kwargs, {'map_location': 'cpu', 'weights_only': True})
        with patch.object(torch, 'load', side_effect=AssertionError):
            with self.assertRaises(api.V8Error): api.load_checkpoint_cpu(ROOT / api.CHECKPOINT_RELATIVE, CFG, LOCK_SHA)

    def test_publish_training_bundle_uses_own_no_clobber_helper_with_fixed_paths(self):
        state = synthetic_state()
        models = {d: TinyModel().state_dict() for d in api.DOMAINS}
        with patch.object(api, 'validate_frozen'), patch.object(api, 'assert_new_outputs'), patch.object(api, '_model_api', return_value=SimpleNamespace(build_model=lambda c, fs: TinyModel())), patch.object(api, 'publish_bundle') as publish:
            api.publish_training_bundle(state, models, self.report)
        payloads = publish.call_args.args[1]
        self.assertEqual(set(payloads), {api.CHECKPOINT_RELATIVE, api.TRAIN_REPORT_RELATIVE})
        self.assertTrue(all(type(v) is bytes and v for v in payloads.values()))

    def test_real_model_checkpoint_CPU_synthetic_only(self):
        core = api._model_api()
        ck = api.checkpoint_metadata(CFG, self.report, LOCK_SHA)
        ck['models'] = {d: {k: v.detach().cpu().clone() for k, v in core.build_model(CFG, api.RATES[d]).state_dict().items()}
                        for d in api.DOMAINS}
        models = api.validate_checkpoint(ck, CFG, LOCK_SHA)
        with torch.inference_mode():
            for d, model in models.items():
                x = torch.zeros(1, 8, 256)
                self.assertTrue(torch.equal(model(x), x))


class FullMockScheduleTests(unittest.TestCase):
    def test_full_mock_18_epochs_no_warmstart_extension_and_gates_failed_saved(self):
        cfg, state, domain = copy.deepcopy(CFG), synthetic_state(), 'Wearable'
        n = api.SHAPES[domain][0]
        # Model, CUDA, DataLoader, optimizer and run_epoch are all inert mocks.
        # The train_domain control flow alone runs; NO optimizer step or data read.
        model = TinyModel()
        model.to = Mock(return_value=model)
        fake = fake_runtime()
        fake.float32 = torch.float32
        fake.cuda.current_device = lambda: 0
        fake.are_deterministic_algorithms_enabled = lambda: True
        fake.backends.cuda.matmul.allow_tf32 = False
        fake.backends.cudnn.allow_tf32 = False
        fake.backends.cudnn.benchmark = False
        fake.backends.cudnn.deterministic = True
        generator = SimpleNamespace(manual_seed=Mock(return_value='generator'))
        fake.Generator = Mock(return_value=generator)
        fake.utils = SimpleNamespace(data=SimpleNamespace(DataLoader=Mock(side_effect=lambda ds, **kw: ds)))
        optimizer = SimpleNamespace(step=Mock(side_effect=AssertionError))
        fake.optim = SimpleNamespace(AdamW=Mock(return_value=optimizer))
        class MockDataset:
            def __init__(self, records, d, c, *, purpose, epoch):
                self.purpose, self.epoch = purpose, epoch
                self.epochs = []
            def __len__(self): return n * (20 if self.purpose == 'training' else 10)
            def set_epoch(self, e): self.epochs.append(e); self.epoch = e
        summary = {c: {'mean_input_normalized_mse': 0.0 if c == 'identity' else 1.0,
                      'mean_output_normalized_mse': 0.0 if c == 'identity' else 1.1}
                   for c in api.CONDITIONS}
        epochs, rows = [], [{'synthetic_only': True}]
        def run(m, ds, c, device, opt=None):
            if opt is not None:
                epochs.append(ds.epoch)
                return synthetic_losses(n * 20), []
            return synthetic_losses(n * 10), rows
        with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}), patch.object(api, '_torch', return_value=fake), patch.object(api, 'validate_frozen') as verify, patch.object(api, 'validate_records'), patch.object(api, 'set_seed') as seed, patch.object(api, 'ReferenceWindowDataset', MockDataset), patch.object(api, '_model_api', return_value=SimpleNamespace(build_model=Mock(return_value=model))) as build, patch.object(api, 'run_epoch', side_effect=run) as epoch_runner, patch.object(api, 'summarize_cases', return_value=summary), patch.object(api, 'gate_summary', return_value={'all_passed': False}), contextlib.redirect_stdout(io.StringIO()):
            selected, report = api.train_domain([], domain, cfg, 'cuda:0', frozen_state=state)
        self.assertEqual(epochs, list(range(1, 19)))
        self.assertEqual(epoch_runner.call_count, 36)
        self.assertEqual(len(report['history']), 18)
        self.assertEqual(report['best_epoch'], 1)
        self.assertEqual(report['selection_status'], 'gates_failed')
        self.assertFalse(report['internal_gates_passed'])
        self.assertEqual(report['best_case_rows'], rows); self.assertIsNot(report['best_case_rows'], rows)
        self.assertEqual(verify.call_count, 20)
        seed.assert_called_once_with(CFG['training']['seed'])
        fake.optim.AdamW.assert_called_once()
        self.assertEqual(fake.optim.AdamW.call_args.kwargs, {'lr': .0005, 'weight_decay': .0001, 'foreach': False, 'fused': False})
        self.assertEqual(selected['delta'].device.type, 'cpu')
        self.assertFalse(selected['delta'].requires_grad)
        optimizer.step.assert_not_called()

    def test_mock_run_two_domains_verify_load_publish_order_no_real_training(self):
        state = synthetic_state()
        events = []
        def event(name, value):
            def action(*args, **kwargs): events.append(name); return value
            return action
        with patch.object(api, 'assert_new_outputs', side_effect=event('collision', None)), patch.object(api, 'validate_frozen', side_effect=event('verify', None)), patch.object(api, 'configure_training', side_effect=event('cuda_config', 'cuda:0')), patch.object(api, 'load_training_records', side_effect=event('load_records', [])), patch.object(api, 'train_domain', side_effect=lambda records, d, c, device, **kw: (events.append(d), ({}, {'mock': True}))[1]) as train, patch.object(api, 'make_training_report', side_effect=event('make_report', {'synthetic': True})), patch.object(api, 'publish_training_bundle', side_effect=event('publish', None)):
            report = api.run_training(state)
        self.assertEqual(events, ['collision', 'verify', 'cuda_config', 'verify', 'load_records', *api.DOMAINS, 'make_report', 'publish'])
        self.assertEqual(train.call_count, 2)
        self.assertEqual(report, {'synthetic': True})

    def test_report_preserves_all_failed_gates_instead_of_fake_acceptance(self):
        report = synthetic_report()
        for d in api.DOMAINS:
            dr = report['domain_reports'][d]
            rows = synthetic_cases(d, ratio=1.1)
            summary = api.summarize_cases(rows, CFG, domain=d)
            gates = api.gate_summary(summary, CFG)
            self.assertFalse(gates['all_passed'])
            for e, row in enumerate(dr['history'], 1):
                row['conditions'], row['gates'] = copy.deepcopy(summary), copy.deepcopy(gates)
                row['selection_key'] = list(api.selection_key(summary, gates, .8, e))
            dr.update(best_case_rows=rows, best_selection_key=dr['history'][0]['selection_key'],
                      selection_status='gates_failed', internal_gates_passed=False)
        api.validate_training_report(report, CFG, LOCK_SHA)
        self.assertFalse(report['runtime_enabled']); self.assertFalse(report['production_validated'])

    def test_accumulator_rejects_invalid_stats_without_partial_mutation(self):
        p,t,x,ident,aux,c = loss_tensors()
        valid = api.compute_loss(p,t,x,ident,aux,CFG,conditions=c)
        acc = api.LossAccumulator(CFG)
        bad = dict(valid); bad['corrupted_weight_sum'] = torch.tensor(1.0)
        with self.assertRaises(api.V8Error): acc.update(bad)
        with self.assertRaises(api.V8Error): acc.result()
        acc.update(valid)
        self.assertEqual(acc.result()['case_count'], 5)

    def test_C_rows_no_hidden_runtime_or_acceptance_fields(self):
        row = synthetic_case(); row['runtime_enabled'] = True
        with self.assertRaises(api.V8Error):
            api.summarize_cases([row], CFG, domain='Nakanishi2015', complete=False)

    def test_unknown_true_scope_is_never_accepted(self):
        lock = synthetic_lock(); lock['scope']['independent_acceptance'] = True
        with self.assertRaises(api.V8Error): api.validate_training_lock(lock, PROTOCOL)


if __name__ == '__main__':
    unittest.main()
