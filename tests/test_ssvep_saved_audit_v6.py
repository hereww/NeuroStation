"""Standard-library synthetic tests: no real report, EEG, NPZ/PT or torch.

Full 4,200-case fixtures are scalar dictionaries in temporary C: directories.
Only the frozen configuration JSON is copied from the actual workspace.
"""
from __future__ import annotations

from collections import defaultdict
import contextlib
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from eeg_tools import ssvep_saved_audit_v6 as a
from scripts import audit_ssvep_saved_internal_v6 as cli

CONFIG_BYTES = (a.ROOT / a.CONFIG_RELATIVE).read_bytes()


def cfg():
    return json.loads(CONFIG_BYTES)


def cases():
    result = []
    for di, dataset in enumerate(('Nakanishi2015', 'Wearable')):
        for subject in (5, 6):
            for trial in range(180 if di == 0 else 240):
                for ki, condition in enumerate(('identity', 'transient', 'low_frequency_drift', 'high_frequency_burst', 'common_mode')):
                    identity = ki == 0
                    x = 0.0 if identity else (1.0 if di == 0 else 9.0)
                    y = 0.0002 if identity else x * 0.9
                    result.append({'dataset': dataset, 'subject_id': subject, 'trial_index': trial, 'condition': condition,
                        'seed': 2026107603 + di * 10000000 + subject * 100000 + trial * 10 + ki,
                        'window_samples': 256, 'window_start_sample': 38 if di == 0 else 125,
                        'sample_rate_hz': 256 if di == 0 else 250,
                        'input_normalized_mse': x, 'output_normalized_mse': y,
                        'relative_reduction': None if identity else (x - y) / x,
                        'identity_normalized_change': y if identity else None,
                        'outcome': 'loss' if identity else 'win',
                        'protected_complex_energy_ratio': 1e-7 if identity else None,
                        'protected_weighted_abs_phase_radians': .01 if identity else None,
                        'undefined_phase_bin_count': 0})
    return result


def expected_conditions(rows):
    """Fixture oracle uses naive sums, not audit statistics or production code."""
    result = {}
    for condition in ('identity', 'transient', 'low_frequency_drift', 'high_frequency_burst', 'common_mode'):
        subset = [row for row in rows if row['condition'] == condition]
        if not subset:
            continue
        def mean(field):
            values = [row[field] for row in subset]
            return None if None in values else float(sum(values) / len(values))
        x, y = mean('input_normalized_mse'), mean('output_normalized_mse')
        result[condition] = {'case_count': len(subset), 'mean_input_normalized_mse': x, 'mean_output_normalized_mse': y,
            'ratio_of_mean_reduction': None if condition == 'identity' or not x else (x - y) / x,
            'mean_identity_normalized_change': mean('identity_normalized_change'),
            'mean_protected_complex_energy_ratio': mean('protected_complex_energy_ratio'),
            'mean_protected_weighted_abs_phase_radians': mean('protected_weighted_abs_phase_radians'),
            'undefined_phase_case_count': sum(row['condition'] == 'identity' and
                (row['undefined_phase_bin_count'] != 0 or row['protected_complex_energy_ratio'] is None or
                 row['protected_weighted_abs_phase_radians'] is None) for row in subset)}
    return result


def expected_gates(summary):
    gates = {}
    for condition, field, bound, operation in (
        ('transient', 'ratio_of_mean_reduction', 0., '>='),
        ('low_frequency_drift', 'ratio_of_mean_reduction', 0., '>='),
        ('high_frequency_burst', 'ratio_of_mean_reduction', .05, '>='),
        ('common_mode', 'ratio_of_mean_reduction', 0., '>='),
        ('identity', 'mean_identity_normalized_change', .0005, '<='),
        ('identity', 'mean_protected_complex_energy_ratio', 1e-6, '<='),
        ('identity', 'mean_protected_weighted_abs_phase_radians', .02, '<=')):
        value = summary.get(condition, {}).get(field)
        passed = value is not None and (value >= bound if operation == '>=' else value <= bound)
        gates[condition + '.' + field] = {'passed': passed, 'value': value, 'threshold': bound, 'comparison': operation}
    value = summary.get('identity', {}).get('undefined_phase_case_count')
    gates['identity.phase_defined'] = {'passed': value == 0, 'value': value, 'threshold': 0, 'comparison': '=='}
    return {'all_passed': all(gate['passed'] for gate in gates.values()), 'gates': gates}


def saved_report(rows=None):
    rows = cases() if rows is None else rows
    buckets = {name: defaultdict(list) for name in ('dataset', 'condition', 'dataset_condition', 'subject', 'subject_condition')}
    for row in rows:
        dataset, subject, condition = row['dataset'], row['subject_id'], row['condition']
        for name, key in [('dataset', dataset), ('condition', condition), ('dataset_condition', dataset + '/' + condition),
                           ('subject', f'{dataset}/{subject}'), ('subject_condition', f'{dataset}/{subject}/{condition}')]:
            buckets[name][key].append(row)
    aggregates = {'global': expected_conditions(rows)}
    for name, groups in buckets.items():
        aggregates[name] = {key: expected_conditions(group) for key, group in groups.items()}
    gates = {dataset: expected_gates(aggregates['dataset'].get(dataset, {})) for dataset in ('Nakanishi2015', 'Wearable')}
    identity = [r for r in rows if r['condition'] == 'identity']
    undefined = sum(r['undefined_phase_bin_count'] > 0 or r['protected_complex_energy_ratio'] is None or
                    r['protected_weighted_abs_phase_radians'] is None for r in identity)
    cohort = {'complete_cohort': len(rows) == 4200, 'case_count': len(rows),
        'reference_trial_count': len({(r['dataset'], r['subject_id'], r['trial_index']) for r in rows}),
        'phase_case_count': len(identity), 'undefined_phase_case_count': undefined,
        'undefined_phase_bin_count': sum(r['undefined_phase_bin_count'] for r in identity),
        'phase_verifiable': len(identity) == 840 and undefined == 0}
    scope = {'training_performed_this_evaluation': False, 'checkpoint_prior_training': True,
        'epoch_selection_performed': False, 'parameter_selection_performed': False,
        'outer_validation_waveforms_loaded': False, 'holdout_waveforms_loaded': False,
        'outer_validation_used_for_training_or_selection': False, 'clean_ground_truth_used': False,
        'independent_acceptance': False, 'runtime_enabled': False, 'production_validated': False}
    report = {'schema_version': 1, 'status': 'formal', 'model_version': 'learned_denoise_v6_ssvep',
        'created_at_utc': '2026-10-07T00:00:00+00:00', 'stage': 'v6_internal_diagnostic',
        'diagnostic_seed': 2026107603, 'diagnostic_epoch': 0, 'batch_size': 64,
        'inference': 'float32_forward_float64_window_residual_restore_no_hann_ola',
        'frozen': {'config_sha256': a.CONFIG_SHA256, 'preregistration_sha256': a.PREREG_SHA256,
                   'training_lock_sha256': 'a'*64, 'evaluation_lock_sha256': 'b'*64,
                   'checkpoint_sha256': 'c'*64, 'training_report_sha256': 'd'*64},
        'preflight': {'check_only': False, 'stage': 'internal_evaluation',
                      'deferred': ['D:/NEVER_RESOLVE_METADATA.npz'], 'selected': []},
        'scope': scope, 'deployment': {'status': 'blocked', 'runtime_enabled': False, 'production_validated': False,
            'pending': ['known_validation_one_time_locked_screen', 'holdout_program']},
        'aggregate_policy': {'mixed_aggregates': ['global', 'condition'], 'mixed_aggregates_descriptive_normalized_only': True,
            'native_raw_power_pooled': False, 'gates_applied_only_to_dataset': True, 'independent_acceptance_claimed': False},
        'limitations': ['synthetic saved scalar fixture, no EEG/weights'],
        'check_only': False, 'inference_performed': True, 'checkpoint_loaded': True, 'waveforms_loaded': True,
        'cuda_initialized': True, 'gpu': {'device': 'cuda:0', 'device_name': 'synthetic claim only'},
        'checks': {'phase': 'executed'}, 'model_state_sha256_before': 'e'*64, 'model_state_sha256_after': 'e'*64,
        **cohort, 'aggregates': aggregates, 'dataset_gates': gates,
        'all_internal_gates_passed': all(g['all_passed'] for g in gates.values()) and cohort['complete_cohort'] and cohort['phase_verifiable'],
        'cases': rows}
    return report


class TemporaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='saved_v6_audit_synthetic_', dir=a.ROOT)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.put(a.CONFIG_RELATIVE, CONFIG_BYTES)
        for source in a.AUDIT_SOURCE_PATHS:
            self.put(source, b'# synthetic audit source placeholder\n')

    def put(self, rel, data):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def put_report(self, report=None):
        report = saved_report() if report is None else report
        self.put(a.REPORT_RELATIVE, json.dumps(report, allow_nan=False).encode())
        return report


class RecomputeTests(unittest.TestCase):
    def test_full_cohort_all_group_counts_means_ratio_of_means_and_not_acceptance(self):
        report = saved_report()
        audit = a.audit_report(report, cfg())
        self.assertTrue(audit['audit_passed'], audit['mismatches'][:2])
        self.assertEqual(audit['mismatch_count'], 0)
        r = audit['recomputed']
        self.assertEqual(r['case_count'], 4200)
        self.assertEqual(r['reference_trial_count'], 840)
        self.assertEqual(r['phase_case_count'], 840)
        self.assertEqual(r['group_case_counts']['dataset'], {'Nakanishi2015': 1800, 'Wearable': 2400})
        self.assertEqual(r['group_case_counts']['condition'], dict.fromkeys(a.CONDITIONS, 840))
        self.assertEqual(r['group_case_counts']['subject'], {'Nakanishi2015/5': 900, 'Nakanishi2015/6': 900,
                                                           'Wearable/5': 1200, 'Wearable/6': 1200})
        self.assertEqual(len(r['group_case_counts']['dataset_condition']), 10)
        self.assertEqual(len(r['group_case_counts']['subject_condition']), 20)
        self.assertTrue(r['all_internal_gates_passed'])
        for key in ('research_acceptance', 'independent_acceptance', 'runtime_enabled', 'production_validated',
                    'original_phase_numeric_values_verified', 'waveforms_loaded', 'checkpoint_loaded',
                    'cuda_initialized', 'model_inference_performed', 'binary_hashes_verified'):
            self.assertIs(audit[key], False)
        self.assertAlmostEqual(r['aggregates']['global']['transient']['mean_input_normalized_mse'], (360+480*9)/840)
        self.assertAlmostEqual(r['aggregates']['global']['transient']['ratio_of_mean_reduction'], .1)
        json.dumps(audit, allow_nan=False)

    def test_wrong_mean_count_ratio_and_each_group_drift_are_recorded(self):
        paths = [('global', 'transient'), ('dataset', 'Wearable', 'identity'),
                 ('condition', 'transient', 'transient'), ('dataset_condition', 'Nakanishi2015/transient', 'transient'),
                 ('subject', 'Wearable/6', 'identity'), ('subject_condition', 'Wearable/6/identity', 'identity')]
        for keys in paths:
            report = saved_report()
            node = report['aggregates']
            for key in keys:
                node = node[key]
            node['mean_output_normalized_mse'] += .001
            result = a.audit_report(report, cfg())
            self.assertFalse(result['audit_passed'])
            self.assertTrue(any('aggregates.' + '.'.join(keys) in r['path'] for r in result['mismatches']))
        for key, value in [('case_count', 999), ('ratio_of_mean_reduction', .3)]:
            report = saved_report()
            report['aggregates']['global']['transient'][key] = value
            self.assertFalse(a.audit_report(report, cfg())['audit_passed'])

    def test_independent_frozen_gates_inclusive_boundaries_no_tolerance(self):
        summary = {condition: {'case_count': 1, 'ratio_of_mean_reduction': 0.0}
                   for condition in ('transient', 'low_frequency_drift', 'high_frequency_burst', 'common_mode')}
        summary['high_frequency_burst']['ratio_of_mean_reduction'] = .05
        summary['identity'] = {'case_count': 1, 'mean_identity_normalized_change': .0005,
            'mean_protected_complex_energy_ratio': 1e-6, 'mean_protected_weighted_abs_phase_radians': .02,
            'undefined_phase_case_count': 0}
        gates = a._dataset_gate(summary, cfg())
        self.assertTrue(gates['all_passed'])
        self.assertEqual(len(gates['gates']), 8)
        for condition, field, changed in (
            ('high_frequency_burst', 'ratio_of_mean_reduction', math.nextafter(.05, -math.inf)),
            ('transient', 'ratio_of_mean_reduction', math.nextafter(0., -math.inf)),
            ('low_frequency_drift', 'ratio_of_mean_reduction', math.nextafter(0., -math.inf)),
            ('common_mode', 'ratio_of_mean_reduction', math.nextafter(0., -math.inf)),
            ('identity', 'mean_identity_normalized_change', math.nextafter(.0005, math.inf)),
            ('identity', 'mean_protected_complex_energy_ratio', math.nextafter(1e-6, math.inf)),
            ('identity', 'mean_protected_weighted_abs_phase_radians', math.nextafter(.02, math.inf)),
            ('identity', 'undefined_phase_case_count', 1)):
            bad = copy.deepcopy(summary)
            bad[condition][field] = changed
            with self.subTest(condition=condition, field=field):
                self.assertFalse(a._dataset_gate(bad, cfg())['all_passed'])
        for field in ('mean_protected_complex_energy_ratio', 'mean_protected_weighted_abs_phase_radians'):
            bad = copy.deepcopy(summary)
            bad['identity'][field] = None
            self.assertFalse(a._dataset_gate(bad, cfg())['all_passed'])

    def test_missing_case_and_trial_cohort_are_failures_even_if_report_self_consistent(self):
        for drop in (1, 5):
            report = saved_report(cases()[drop:])
            result = a.audit_report(report, cfg())
            self.assertFalse(result['audit_passed'])
            self.assertFalse(result['recomputed']['complete_cohort'])
            self.assertFalse(result['recomputed']['all_internal_gates_passed'])
            self.assertTrue(any(m['path'] == 'cases.cohort' for m in result['mismatches']))
            self.assertEqual(result['recomputed']['case_count'], 4200 - drop)
            if drop == 5:
                self.assertEqual(result['recomputed']['reference_trial_count'], 839)

    def test_tampered_gate_allpass_comparison_threshold_or_value(self):
        key = 'high_frequency_burst.ratio_of_mean_reduction'
        for field, value in [('passed', False), ('value', .9), ('threshold', .04), ('comparison', '<=')]:
            report = saved_report()
            report['dataset_gates']['Wearable']['gates'][key][field] = value
            audit = a.audit_report(report, cfg())
            self.assertFalse(audit['audit_passed'])
            self.assertTrue(any(m['path'].endswith(field) for m in audit['mismatches']))
        report = saved_report()
        report['all_internal_gates_passed'] = False
        self.assertFalse(a.audit_report(report, cfg())['audit_passed'])
        report = saved_report()
        report['dataset_gates']['Wearable']['gates'][key]['threshold'] = math.nextafter(.05, 0.)
        self.assertFalse(a.audit_report(report, cfg())['audit_passed'])  # no threshold tolerance

    def test_undefined_phase_and_null_means_fail_gates_but_can_be_consistent_json(self):
        rows = cases()
        rows[0].update(protected_weighted_abs_phase_radians=None, undefined_phase_bin_count=2)
        report = saved_report(rows)
        audit = a.audit_report(report, cfg())
        self.assertTrue(audit['audit_passed'], audit['mismatches'][:1])
        self.assertFalse(audit['recomputed']['all_internal_gates_passed'])
        self.assertFalse(audit['recomputed']['phase_verifiable'])
        self.assertEqual(audit['recomputed']['undefined_phase_case_count'], 1)
        self.assertEqual(audit['recomputed']['undefined_phase_bin_count'], 2)
        self.assertIsNone(audit['recomputed']['aggregates']['global']['identity']['mean_protected_weighted_abs_phase_radians'])
        rows[0].update(protected_complex_energy_ratio=None, undefined_phase_bin_count=0)
        audit = a.audit_report(saved_report(rows), cfg())
        self.assertTrue(audit['audit_passed'])
        self.assertEqual(audit['recomputed']['undefined_phase_case_count'], 1)
        self.assertFalse(audit['recomputed']['phase_verifiable'])
        report = saved_report(rows)
        report['aggregates']['dataset']['Nakanishi2015']['identity']['mean_protected_weighted_abs_phase_radians'] = 0.0
        self.assertFalse(a.audit_report(report, cfg())['audit_passed'])

    def test_mixed_description_never_masks_single_dataset_gate_failure(self):
        rows = cases()
        for row in rows:
            if row['condition'] == 'high_frequency_burst':
                x = row['input_normalized_mse']
                y = x * (.8 if row['dataset'] == 'Wearable' else 1.2)
                row.update(output_normalized_mse=y, relative_reduction=(x-y)/x,
                           outcome='win' if y<x else 'loss')
        report = saved_report(rows)
        audit = a.audit_report(report, cfg())
        self.assertTrue(audit['audit_passed'], audit['mismatches'][:1])
        self.assertGreater(audit['recomputed']['aggregates']['global']['high_frequency_burst']['ratio_of_mean_reduction'], .05)
        self.assertFalse(audit['recomputed']['dataset_gates']['Nakanishi2015']['all_passed'])
        self.assertTrue(audit['recomputed']['dataset_gates']['Wearable']['all_passed'])
        self.assertFalse(audit['recomputed']['all_internal_gates_passed'])

    def test_saved_case_ratio_outcome_identity_change_and_model_state_drift(self):
        for key, value, index in [('relative_reduction', .9, 1), ('outcome', 'tie', 1),
                                  ('identity_normalized_change', .0004, 0), ('input_normalized_mse', 1e-8, 0)]:
            report = saved_report()
            report['cases'][index][key] = value
            self.assertFalse(a.audit_report(report, cfg())['audit_passed'])
        report = saved_report()
        report['model_state_sha256_after'] = 'f' * 64
        self.assertFalse(a.audit_report(report, cfg())['audit_passed'])

    def test_duplicates_bad_provenance_nonfinite_and_illegal_phase_fail_closed(self):
        report = saved_report()
        report['cases'].append(dict(report['cases'][0]))
        with self.assertRaises(a.AuditError):
            a.audit_report(report, cfg())
        for field, value in [('subject_id', 7), ('subject_id', True), ('seed', 1), ('seed', True),
                             ('trial_index', 180), ('window_samples', 255), ('sample_rate_hz', 250),
                             ('window_start_sample', 125), ('input_normalized_mse', float('inf')),
                             ('output_normalized_mse', float('nan')), ('protected_weighted_abs_phase_radians', math.pi+1),
                             ('undefined_phase_bin_count', True), ('undefined_phase_bin_count', -1)]:
            report = saved_report()
            report['cases'][0][field] = value
            with self.subTest(field=field), self.assertRaises(a.AuditError):
                a.audit_report(report, cfg())
        report = saved_report()
        report['cases'][0]['undefined_phase_bin_count'] = 1  # numeric phase forbidden
        with self.assertRaises(a.AuditError):
            a.audit_report(report, cfg())
        report = saved_report()
        report['cases'][0]['protected_complex_energy_ratio'] = None  # undefined ref support => no numeric phase
        with self.assertRaises(a.AuditError):
            a.audit_report(report, cfg())
        report = saved_report()
        report['cases'][1]['protected_complex_energy_ratio'] = 0.0
        with self.assertRaises(a.AuditError):
            a.audit_report(report, cfg())
        report = saved_report()
        report['cases'][0]['waveform'] = [1.0]
        with self.assertRaises(a.AuditError):
            a.audit_report(report, cfg())

    def test_header_schema_config_drift_unsafe_scope_and_bool_types(self):
        for field, value in [('schema_version', True), ('check_only', True), ('status', 'draft'),
                             ('diagnostic_seed', True), ('cuda_initialized', 1)]:
            report = saved_report()
            report[field] = value
            with self.assertRaises(a.AuditError):
                a.audit_report(report, cfg())
        for key in ('runtime_enabled', 'production_validated', 'independent_acceptance', 'holdout_waveforms_loaded',
                    'outer_validation_waveforms_loaded'):
            report = saved_report()
            report['scope'][key] = True
            with self.assertRaises(a.AuditError):
                a.audit_report(report, cfg())
        report = saved_report()
        report['frozen']['config_sha256'] = 'f'*64
        with self.assertRaises(a.AuditError):
            a.audit_report(report, cfg())
        for mutation in ('gate', 'unknown', 'bool', 'nonfinite'):
            config = cfg()
            if mutation == 'gate':
                config['internal_research_gates']['identity_mean_normalized_change_max'] = .9
            elif mutation == 'unknown':
                config['extra'] = 1
            elif mutation == 'bool':
                config['input_channels'] = True
            else:
                config['training']['learning_rate'] = float('nan')
            with self.assertRaises(a.AuditError):
                a.audit_report(saved_report(), config)

    def test_consistent_phase_claim_cannot_establish_original_numeric_correctness(self):
        rows = cases()
        rows[0]['protected_weighted_abs_phase_radians'] = .019
        audit = a.audit_report(saved_report(rows), cfg())
        self.assertTrue(audit['audit_passed'])
        self.assertFalse(audit['original_phase_numeric_values_verified'])
        self.assertTrue(any('not independently recomputed' in limitation for limitation in audit['limitations']))

    def test_ratio_of_means_not_mean_of_percase_ratios(self):
        rows = cases()
        transient = [row for row in rows if row['dataset']=='Nakanishi2015' and row['condition']=='transient']
        for index, row in enumerate(transient):
            x, y = (1., .5) if index % 2 == 0 else (9., 9.)
            row.update(input_normalized_mse=x, output_normalized_mse=y, relative_reduction=(x-y)/x,
                       outcome='win' if y<x else 'tie')
        result = a.audit_report(saved_report(rows), cfg())
        self.assertTrue(result['audit_passed'])
        self.assertAlmostEqual(result['recomputed']['aggregates']['dataset']['Nakanishi2015']['transient']['ratio_of_mean_reduction'], .05)


class JSONPathPublicationTests(TemporaryTests):
    def test_strict_duplicate_nonfinite_overflow_and_json_only_extension(self):
        for payload in ('{"a":1,"a":2}', '{"a":{"x":1,"x":2}}', '{"a":NaN}', '{"a":Infinity}',
                        '{"a":-Infinity}', '{"a":1e999}', '{"a":' + '9'*400 + '}', '[]', '{bad'):
            with self.subTest(payload=payload[:35]), self.assertRaises(a.AuditError):
                a.strict_json_loads(payload)
        self.assertEqual(a.strict_json_loads(b'\xef\xbb\xbf{"a":true}'), {'a': True})
        for rel in ('fake.npz', 'fake.pt', 'source.mat'):
            with patch.object(Path, 'lstat', side_effect=AssertionError('binary input must fail before stat')):
                with self.assertRaises(a.AuditError):
                    a.read_json(self.root, rel)
        with self.assertRaises(a.AuditError):
            a.strict_json_loads(b'\xff')

    def test_unsafe_paths_before_stat_symlinks_reparse_and_output_prefix(self):
        for path in ('D:/forbidden.json', 'D:\\forbidden.json', '../escape.json', 'x/../y.json',
                     '//server/share/file.json', '\\\\server\\share\\file.json', 'C:relative.json',
                     'input.json:stream', '', 'x\x00y', 'NUL', 'CON.json', 'trailing.'):
            with patch.object(Path, 'lstat', side_effect=AssertionError('reject before stat')):
                with self.subTest(path=path), self.assertRaises(a.AuditError):
                    a.safe_path(self.root, path)
        with self.assertRaises(a.AuditError):
            a.safe_path(self.root, self.root.parent / 'outside.json')
        target = self.root / 'real_dir'
        target.mkdir()
        link = self.root / 'symlink'
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError:
            pass
        else:
            with self.assertRaises(a.AuditError):
                a.safe_path(self.root, 'symlink/report.json')
        actual = Path.lstat
        def reparse(path, *args, **kwargs):
            info = actual(path, *args, **kwargs)
            return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400) if path == target else info
        with patch.object(Path, 'lstat', reparse), self.assertRaises(a.AuditError):
            a.safe_path(self.root, 'real_dir/report.json')
        for prefix in ('analysis_outputs/ssvep_v5', 'models/ssvep_v6_saved_internal_audit',
                       'analysis_outputs/nested/ssvep_v6_saved_internal_audit', 'analysis_outputs/ssvep_v6_internal_diagnostic_20261007'):
            with self.assertRaises(a.AuditError):
                a.output_paths(self.root, prefix)

    def test_check_only_does_not_read_nonexistent_report_or_publish_and_bool_strict(self):
        actual = os.open
        opened = []
        def guarded(path, *args, **kwargs):
            opened.append(str(path))
            if str(path).endswith(a.REPORT_RELATIVE.replace('/', os.sep)):
                raise AssertionError('check-only must not open future report')
            return actual(path, *args, **kwargs)
        with patch.object(a.os, 'open', side_effect=guarded):
            result = a.audit_saved_report(self.root, check_only=True)
        self.assertFalse(result['report_read'])
        self.assertFalse(result['audit_performed'])
        self.assertFalse(result['audit_passed'])
        self.assertEqual(len(opened), 1)
        self.assertFalse((self.root / 'analysis_outputs').exists())
        for value in (0, 1, None, 'true'):
            with self.assertRaises(a.AuditError):
                a.audit_saved_report(self.root, check_only=value)

    def test_saved_fixture_audit_json_and_md_hashes_sources_and_no_binary_no_d_touch(self):
        self.put_report()
        actual_open, actual_stat = os.open, Path.lstat
        opened = []
        def guarded_open(path, *args, **kwargs):
            text = str(path).replace('\\', '/')
            opened.append(text)
            if text.lower().startswith('d:') or text.endswith(('.npz', '.pt', '.mat')):
                raise AssertionError('no binary or D: open')
            return actual_open(path, *args, **kwargs)
        def guarded_stat(path, *args, **kwargs):
            text = str(path).replace('\\', '/')
            if text.lower().startswith('d:') or text.endswith(('.npz', '.pt', '.mat')) or 'subject_007' in text or 'subject_009' in text:
                raise AssertionError('no external/binary/outer/holdout stat')
            return actual_stat(path, *args, **kwargs)
        with patch.object(a.os, 'open', side_effect=guarded_open), patch.object(Path, 'lstat', guarded_stat):
            result = a.audit_saved_report(self.root)
        self.assertTrue(result['audit_passed'])
        self.assertEqual(set(result['audit_source_sha256']), set(a.AUDIT_SOURCE_PATHS))
        self.assertEqual(result['inputs']['report_sha256'], hashlib.sha256((self.root/a.REPORT_RELATIVE).read_bytes()).hexdigest())
        self.assertTrue(all(path.endswith(('.json', '.py', '.pending')) for path in opened))
        output = json.loads((self.root / (a.DEFAULT_OUTPUT_PREFIX + '.json')).read_text())
        self.assertEqual(output, result)
        markdown = (self.root / (a.DEFAULT_OUTPUT_PREFIX + '.md')).read_text()
        self.assertIn('NOT independent acceptance', markdown)
        self.assertIn('Original phase numeric values verified from EEG: false', markdown)
        self.assertFalse(list(self.root.rglob('*.pending')))

    def test_config_sha_drift_collision_before_read_and_input_change_fail_closed(self):
        self.put(a.CONFIG_RELATIVE, CONFIG_BYTES + b' ')
        with self.assertRaises(a.AuditError):
            a.audit_saved_report(self.root, check_only=True)
        self.put(a.CONFIG_RELATIVE, CONFIG_BYTES)
        self.put(a.DEFAULT_OUTPUT_PREFIX + '.md', b'other writer')
        with patch.object(a, '_read_bytes', side_effect=AssertionError('collision must precede inputs')):
            with self.assertRaises(a.AuditError):
                a.audit_saved_report(self.root)
        (self.root / (a.DEFAULT_OUTPUT_PREFIX + '.md')).unlink()
        self.put_report()
        actual = a._read_bytes
        def changed(root, rel, *, source=False):
            data, stamp = actual(root, rel, source=source)
            if rel == a.AUDIT_SOURCE_PATHS[-1] and source:
                self.put(a.REPORT_RELATIVE, b'changed between audit and publication')
            return data, stamp
        with patch.object(a, '_read_bytes', side_effect=changed), self.assertRaises(a.AuditError):
            a.audit_saved_report(self.root)
        self.assertFalse((self.root / (a.DEFAULT_OUTPUT_PREFIX + '.json')).exists())

    def test_partial_link_race_rolls_back_only_own_files_preserves_competitor(self):
        audit = a.audit_report(saved_report(), cfg())
        original_link = os.link
        count = 0
        def racing(source, target):
            nonlocal count
            count += 1
            if count == 2:
                Path(target).write_bytes(b'competing writer')
            original_link(source, target)
        with patch.object(a.os, 'link', side_effect=racing), self.assertRaises(FileExistsError):
            a.publish_audit(self.root, a.DEFAULT_OUTPUT_PREFIX, audit)
        self.assertFalse((self.root / (a.DEFAULT_OUTPUT_PREFIX + '.json')).exists())
        self.assertEqual((self.root / (a.DEFAULT_OUTPUT_PREFIX + '.md')).read_bytes(), b'competing writer')
        self.assertFalse(list(self.root.rglob('*.pending')))

    def test_partial_failure_never_removes_replaced_competitor_inode(self):
        audit = a.audit_report(saved_report(), cfg())
        original_link = os.link
        count = 0
        def racing(source, target):
            nonlocal count
            count += 1
            if count == 2:
                first = self.root / (a.DEFAULT_OUTPUT_PREFIX + '.json')
                first.unlink()
                first.write_bytes(b'replaced competitor inode')
                raise OSError('synthetic publication failure')
            original_link(source, target)
        with patch.object(a.os, 'link', side_effect=racing), self.assertRaises(OSError):
            a.publish_audit(self.root, a.DEFAULT_OUTPUT_PREFIX, audit)
        self.assertEqual((self.root / (a.DEFAULT_OUTPUT_PREFIX + '.json')).read_bytes(), b'replaced competitor inode')
        self.assertFalse(list(self.root.rglob('*.pending')))

    def test_fsync_failure_refuses_publication_and_cleans_own_temp(self):
        audit = a.audit_report(saved_report(), cfg())
        with patch.object(a.os, 'fsync', side_effect=OSError('synthetic flush failure')), self.assertRaises(OSError):
            a.publish_audit(self.root, a.DEFAULT_OUTPUT_PREFIX, audit)
        self.assertFalse((self.root / (a.DEFAULT_OUTPUT_PREFIX + '.json')).exists())
        self.assertFalse(list(self.root.rglob('*.pending')))


class CLINoImportTests(TemporaryTests):
    def test_help_before_read_check_only_without_report_and_exit_codes(self):
        with patch.object(a, 'audit_saved_report', side_effect=AssertionError('help never audits')), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                cli.main(['--help'], root=self.root)
            self.assertEqual(raised.exception.code, 0)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(cli.main(['--check-only'], root=self.root), 0)
        self.assertFalse(json.loads(output.getvalue())['audit_performed'])
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main([], root=self.root), 2)  # still no synthetic report
        self.put_report()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main([], root=self.root), 0)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main([], root=self.root), 2)  # existing output refusal
        report = saved_report()
        report['dataset_gates']['Wearable']['all_passed'] = False
        self.put_report(report)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(['--output-prefix', 'analysis_outputs/ssvep_v6_saved_internal_audit_mismatch'], root=self.root), 1)
        saved = json.loads((self.root / 'analysis_outputs/ssvep_v6_saved_internal_audit_mismatch.json').read_text())
        self.assertFalse(saved['audit_passed'])
        self.assertGreater(saved['mismatch_count'], 0)

    def test_cold_subprocess_blocks_torch_numpy_and_all_production_v6_modules(self):
        self.put_report()
        code = r'''
import builtins, sys
from pathlib import Path
original = builtins.__import__
forbidden = ('torch', 'numpy', 'eeg_tools.ssvep_v6_contract', 'eeg_tools.ssvep_denoise_v6', 'eeg_tools.ssvep_v6_internal_eval')
def guarded(name, *args, **kwargs):
    if any(name == prefix or name.startswith(prefix + '.') for prefix in forbidden):
        raise AssertionError('forbidden dependency imported: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from scripts import audit_ssvep_saved_internal_v6 as cli
assert cli.main([], root=Path(sys.argv[1])) == 0
assert not any(name == p or name.startswith(p + '.') for name in sys.modules for p in forbidden)
print('STANDARD_LIBRARY_ONLY_OK')
'''
        result = subprocess.run([sys.executable, '-B', '-c', code, str(self.root)], cwd=a.ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('STANDARD_LIBRARY_ONLY_OK', result.stdout)


if __name__ == '__main__':
    unittest.main()
