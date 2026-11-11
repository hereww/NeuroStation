"""Fixed v13 evaluator tests, using artificial evidence and tiny mock weights."""
from __future__ import annotations
from tests.synthetic_research_fixtures import research_contract
import contextlib
import copy
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scripts import evaluate_ssvep_denoiser_v13 as ev
from eeg_tools import ssvep_v13_runtime as api

ROOT = Path(__file__).absolute().parents[1]


def gate_evidence():
    plan = research_contract('v13-plan.json')
    public = {'conditions': {'identity': {'output_reference_normalized_change': 0.001}}}
    def aggregate():
        return {'output_reference_normalized_change': 0.001,
                'regions': {r: {'tracks': {'notch': {'clean_centered_mse': 1.0},
                                          'v13': {'clean_centered_mse': 0.2}},
                                'contributions': {'learned_notch_to_v13': {'relative_centered_error_reduction': 0.8}}}
                            for r in ev.REGIONS}}
    synthetic = {'conditions': {c: aggregate() for c in ev.SYNTHETIC_CONDITIONS},
                 'balanced_drift_and_combined': aggregate(), 'pooled_corrupted': aggregate(), 'cases': []}
    for hz in (10, 12, 15, 20):
        for condition in ('identity', 'line50'):
            harmonics = [{'frequency_hz': f, 'amplitude_ratio_per_channel': [1.0]*8,
                          'wrapped_phase_error_degrees_per_channel': [0.0]*8} for f in (hz, 2*hz, 7)]
            synthetic['cases'].append({'frequency_hz': hz, 'condition': condition,
                                      'regions': {r: {'tracks': {'v13': {'harmonics': copy.deepcopy(harmonics)}}}
                                                  for r in ev.REGIONS}})
    personal = [{'source_session': name, 'source_hashes_match': True,
                 'tracks': {t: {'recorded_trial_count': 12, 'available_presented_trial_count': 12,
                                'presented_match_count': 9 if t == 'v13' else 3} for t in ev.TRACKS}}
                for name in ('artificial_session_a', 'artificial_session_b')]
    contract = {'max_absolute_difference': 0.0, 'cases': [{'window_samples': n} for n in (512, 4096)]}
    metadata = {'selection_gate_passed': True, 'checkpoint_role': 'selected_research', **ev.FALSE_FLAGS}
    return public, synthetic, personal, contract, plan, metadata


class NumericalAndGateTests(unittest.TestCase):
    def test_joint_ls_known_phase_trend_harmonics_and_wrap(self):
        t = np.arange(4096)[:, None]/250
        phases = np.radians(np.linspace(178.8, 179.8, 8))[None, :]
        def wave(offset):
            return (100 + np.arange(8)[None, :] + 0.12*t
                    + 1.3*np.sin(2*np.pi*12*t+phases+offset)
                    + 0.3*np.sin(2*np.pi*24*t-0.7)
                    + 0.2*np.sin(2*np.pi*7*t+0.4))
        for start, end in ((0, 4096), (512, 3584)):
            reference = ev.joint_sine_fit(wave(0)[start:end], (12, 24, 7), start_sample=start)
            measured = ev.joint_sine_fit(wave(np.radians(1.6))[start:end], (12, 24, 7), start_sample=start)
            difference = ev.compare_fits(reference, measured)
            np.testing.assert_allclose(reference['components'][0]['amplitude_per_channel'], 1.3, atol=1e-12)
            np.testing.assert_allclose(reference['trend_per_second'], 0.12, atol=1e-12)
            np.testing.assert_allclose(difference[0]['wrapped_phase_error_degrees_per_channel'], 1.6, atol=1e-10)
            np.testing.assert_allclose(difference[1]['wrapped_phase_error_degrees_per_channel'], 0, atol=1e-10)

    def test_mean_error_is_separate_from_centered_error(self):
        clean = np.sin(np.arange(4096)[:, None]/10)+np.arange(8)[None, :]
        errors = ev.waveform_errors(clean, clean+5)
        self.assertLess(errors['clean_centered_normalized_mse'], 1e-25)
        self.assertAlmostEqual(errors['mean_offset_mse'], 25)
        self.assertAlmostEqual(errors['uncentered_mse'], 25)

    def test_zero_amplitude_is_undefined_phase_and_cannot_pass(self):
        fit = ev.joint_sine_fit(np.zeros((4096, 8)), (12, 24, 7))
        self.assertEqual(fit['components'][0]['phase_radians_per_channel'], [None]*8)
        evidence = gate_evidence()
        evidence[1]['cases'][0]['regions']['full']['tracks']['v13']['harmonics'][0] = ev.compare_fits(fit, fit)[0]
        gate = ev.research_gate(*evidence)
        self.assertFalse(gate['passed'])
        key = 'noisefree_and_line50_fundamental_phase_error_degrees_max'
        self.assertFalse(gate['checks'][key]['available'])

    def test_fixed_research_gate_and_each_required_failure(self):
        self.assertTrue(ev.research_gate(*gate_evidence())['passed'])
        for failure in ('phase', 'identity', 'long_identity', 'drift', 'missing_personal', 'lower_matches', 'failed_selection', 'nonfinite'):
            with self.subTest(failure=failure):
                public, synthetic, personal, contract, plan, metadata = gate_evidence()
                if failure == 'phase':
                    synthetic['cases'][0]['regions']['interior']['tracks']['v13']['harmonics'][0]['wrapped_phase_error_degrees_per_channel'][3] = 2.1
                elif failure == 'identity':
                    public['conditions']['identity']['output_reference_normalized_change'] = 0.005129
                elif failure == 'long_identity':
                    synthetic['conditions']['identity']['output_reference_normalized_change'] = 0.0021
                elif failure == 'drift':
                    synthetic['balanced_drift_and_combined']['regions']['full']['contributions']['learned_notch_to_v13']['relative_centered_error_reduction'] = 0.49
                elif failure == 'missing_personal':
                    personal = []
                elif failure == 'lower_matches':
                    personal[0]['tracks']['v13']['presented_match_count'] = 2
                elif failure == 'failed_selection':
                    metadata['selection_gate_passed'] = False
                elif failure == 'nonfinite':
                    synthetic['cases'][0]['regions']['full']['tracks']['v13']['harmonics'][1]['amplitude_ratio_per_channel'][0] = float('nan')
                gate = ev.research_gate(public, synthetic, personal, contract, plan, metadata)
                self.assertFalse(gate['passed'])
                for key in ev.FALSE_FLAGS:
                    self.assertIs(gate[key], False)
                if failure == 'missing_personal':
                    self.assertFalse(gate['checks']['personal_presented_match_count_not_lower_than_raw']['available'])
                if failure == 'failed_selection':
                    self.assertFalse(gate['checks']['selection_gate_passed_required']['passed'])

    def test_plan_thresholds_are_consumed_without_modifying_plan(self):
        evidence = gate_evidence()
        unchanged = copy.deepcopy(evidence[4])
        evidence[0]['conditions']['identity']['output_reference_normalized_change'] = 0.0019
        self.assertTrue(ev.research_gate(*evidence)['checks']['public_identity_normalized_change_max']['passed'])
        self.assertEqual(evidence[4], unchanged)


@unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'optional PyTorch unavailable')
class LoaderTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.tmp = tempfile.TemporaryDirectory(prefix='v13_loader_')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = {'model_version': 'learned_denoise_v13_personal', 'training': {'epochs': 16}}
        for name in (api.MODEL, api.CONFIG, api.LOCK, api.PLAN, api.REPORT, api.MANIFEST):
            path = self.root/name
            path.parent.mkdir(parents=True, exist_ok=True)
        for name, value in ((api.CONFIG, self.cfg), (api.LOCK, {'fixture': 'lock'}), (api.PLAN, {'fixture': 'plan'})):
            (self.root/name).write_text(json.dumps(value), encoding='utf-8')
        self.common = dict(schema_version=1, status='formal', model_version=self.cfg['model_version'],
                           config_sha256=api.canonical_sha(self.cfg), training_lock_sha256=ev.sha(self.root/api.LOCK),
                           research_plan_sha256=ev.sha(self.root/api.PLAN), selected_epoch=None,
                           selection_gate_passed=False, runtime_enabled=False, production_validated=False)
        self.model = torch.nn.Linear(1, 1)
        torch.save({**self.common, 'config': self.cfg, 'state_dict': self.model.state_dict()}, self.root/api.MODEL)
        self.report = {**self.common, 'status': 'training_completed', 'epochs': 16, 'diagnostic_epoch': 16,
                       'selected_metrics': None, 'checkpoint_role': ev.DIAGNOSTIC_ROLE,
                       'normalization_contract': api.NORMALIZATION, 'parameter_count': 2,
                       'history': [{'epoch': i, 'selection': {'selection_gate_passed': False}} for i in range(1, 17)]}
        self.manifest = {**self.common, 'checkpoint_role': ev.DIAGNOSTIC_ROLE, 'normalization_contract': api.NORMALIZATION,
                         'parameter_count': 2, 'checkpoint_sha256': ev.sha(self.root/api.MODEL)}
        self.sync()
        self.state = {'config_sha256': self.common['config_sha256'], 'training_lock_sha256': self.common['training_lock_sha256']}
        for target, value in (('load_config', self.cfg), ('preflight', self.state)):
            patcher = patch.object(api, target, return_value=value)
            patcher.start(); self.addCleanup(patcher.stop)
        patcher = patch.object(api, 'build_model', side_effect=lambda cfg: torch.nn.Linear(1, 1))
        self.build = patcher.start(); self.addCleanup(patcher.stop)

    def sync(self):
        (self.root/api.REPORT).write_text(json.dumps(self.report), encoding='utf-8')
        self.manifest['training_report_sha256'] = ev.sha(self.root/api.REPORT)
        (self.root/api.MANIFEST).write_text(json.dumps(self.manifest), encoding='utf-8')

    def test_default_rejection_is_not_caught_or_replaced(self):
        with patch.object(api, 'load_model', side_effect=ValueError('unbound/ineligible v13 artifact')) as loader:
            with self.assertRaisesRegex(ValueError, 'ineligible'):
                ev.load_evaluation_model(self.root, 'cpu')
        loader.assert_called_once_with(self.root, 'cpu')
        self.build.assert_not_called()

    def test_explicit_diagnostic_weights_only_and_role_flags(self):
        torch = self.torch
        with patch.object(torch, 'load', wraps=torch.load) as loader, patch.object(api, 'load_model') as normal:
            model, metadata = ev.load_evaluation_model(self.root, 'cpu', allow_failed_selection_diagnostic=True)
        self.assertTrue(loader.call_args.kwargs['weights_only'])
        self.assertEqual(loader.call_args.kwargs['map_location'], 'cpu')
        normal.assert_not_called()
        self.assertFalse(model.training)
        self.assertEqual(metadata['checkpoint_role'], ev.DIAGNOSTIC_ROLE)
        self.assertIsNone(metadata['selected_epoch'])
        self.assertEqual(metadata['diagnostic_epoch'], 16)
        self.assertIs(metadata['selection_gate_passed'], False)
        for key in ev.FALSE_FLAGS:
            self.assertIs(metadata[key], False)
        self.assertEqual(os.environ['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertEqual(torch.get_num_threads(), 4)
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)

    def test_diagnostic_rejects_tampered_sha_and_nonfinal_roles(self):
        for change in ('sha', 'role', 'epoch', 'selected', 'eligible_history', 'config'):
            with self.subTest(change=change):
                report, manifest = copy.deepcopy(self.report), copy.deepcopy(self.manifest)
                if change == 'sha': self.manifest['checkpoint_sha256'] = '0'*64
                elif change == 'role': self.manifest['checkpoint_role'] = 'selected_research'
                elif change == 'epoch': self.report['diagnostic_epoch'] = 15
                elif change == 'selected': self.manifest['selected_epoch'] = 1
                elif change == 'eligible_history': self.report['history'][0]['selection']['selection_gate_passed'] = True
                elif change == 'config': self.manifest['config_sha256'] = '0'*64
                self.sync()
                with self.assertRaises(ValueError):
                    ev.load_evaluation_model(self.root, 'cpu', allow_failed_selection_diagnostic=True)
                self.report, self.manifest = report, manifest
                self.sync()


class CLIAndPathTests(unittest.TestCase):
    def test_import_and_help_never_import_scientific_stack(self):
        code = '''
import sys, io, contextlib
class Block:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'scipy', 'torch'}:
            raise AssertionError('unexpected scientific import '+fullname)
sys.meta_path.insert(0, Block())
from scripts import evaluate_ssvep_denoiser_v13 as ev
with contextlib.redirect_stdout(io.StringIO()) as out:
    try: ev.main(['--help'])
    except SystemExit as exc: assert exc.code == 0
assert '--allow-failed-selection-diagnostic' in out.getvalue()
assert not {'torch','numpy','scipy'} & sys.modules.keys()
'''
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_existing_outside_and_overlap_paths_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); analysis = root/'analysis_outputs'; analysis.mkdir()
            existing = analysis/'existing'; existing.mkdir(); (existing/'sentinel').write_bytes(b'keep')
            for path in (analysis, existing, root/'outside'):
                with self.assertRaises(ValueError): ev.validate_paths(path, root=root)
            source = analysis/'source'; source.mkdir()
            with self.assertRaisesRegex(ValueError, 'overlap'):
                ev.validate_paths(source/'nested-output', [source], root=root)
            self.assertEqual((existing/'sentinel').read_bytes(), b'keep')

    def test_cli_returns_zero_for_complete_failed_gate_and_one_for_exception(self):
        completed = {'research_gate': {'passed': False}, 'checkpoint_role': ev.DIAGNOSTIC_ROLE, 'selection_gate_passed': False}
        with patch.object(ev, 'evaluate', return_value=completed) as run, contextlib.redirect_stdout(io.StringIO()):
            status = ev.main(['--output-directory', 'analysis_outputs/new', '--device', 'cuda:0', '--personal-session', 'one', '--personal-session', 'two', '--allow-failed-selection-diagnostic'])
        self.assertEqual(status, 0)
        self.assertEqual(run.call_args.args[1], ['one', 'two'])
        self.assertTrue(run.call_args.kwargs['allow_failed_selection_diagnostic'])
        with patch.object(ev, 'evaluate', side_effect=ValueError('ineligible')), contextlib.redirect_stderr(io.StringIO()) as out:
            status = ev.main(['--output-directory', 'analysis_outputs/new'])
        self.assertEqual(status, 1)
        self.assertEqual(json.loads(out.getvalue())['completion_status'], 'incomplete')
