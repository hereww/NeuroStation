"""Synthetic diagnostics and additive bundle compatibility; no personal data."""
import copy
import json
import unittest
import numpy as np
from unittest.mock import patch
from eeg_tools.workstation import personal_decoder_analysis as analysis
from tests import test_v9_offline_analysis as fixtures
from eeg_tools.workstation import v9_offline as backend


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        t = np.arange(2000) / 250
        self.signal = np.column_stack([np.sin(2 * np.pi * 12 * t + ch / 10)
            + 0.1 * np.cos(2 * np.pi * 50 * t + ch) for ch in range(8)])
        self.targets = [('A', 10), ('B', 12), ('C', 15), ('D', 20)]
        self.trial = {'trial_index': '1', 'presented_target_id': 'B', 'gaze_target_id': '',
                      'onset_sample_index': 250, 'offset_sample_index': 1750,
                      'window_start_sample_index': 375, 'unavailable_reason': None}
        self.tracks = dict.fromkeys(analysis.TRACKS, self.signal)

    def run_analysis(self, trials=None, tracks=None, reason=None):
        return analysis.analyze(tracks or self.tracks, self.targets, trials or [self.trial], reason)

    def test_synthetic_candidate_and_controls_are_not_accuracy_acceptance(self):
        result = self.run_analysis()
        for name in analysis.TRACKS:
            self.assertEqual('B', result['trials'][0]['predictions'][name]['target_id'])
            self.assertEqual(1, result['tracks'][name]['presented_match_count'])
        self.assertFalse(result['production_validated'])
        self.assertFalse(result['confidence_is_probability'])
        self.assertFalse(result['detection_validated'])
        self.assertTrue(result['development_data'])
        self.assertEqual(32, len(result['controls']['noise']))
        self.assertTrue(all(row['available'] and row['target_id'] for row in result['controls']['noise']))
        self.assertEqual([0, 1750], [row['start_sample'] for row in result['controls']['rest']])

    def test_labels_never_enter_forward_and_only_change_proxy_counts(self):
        from eeg_tools import personal_ssvep_decoder
        seen = []
        original = personal_ssvep_decoder.decode_window
        def spy(signal, rate, targets):
            seen.append((signal.shape, rate, list(targets)))
            return original(signal, rate, targets)
        with patch.object(personal_ssvep_decoder, 'decode_window', side_effect=spy):
            first = self.run_analysis()
            trial = {**self.trial, 'presented_target_id': 'A', 'gaze_target_id': 'A'}
            second = self.run_analysis([trial])
        self.assertEqual(first['trials'][0]['predictions'], second['trials'][0]['predictions'])
        self.assertEqual(1, first['tracks']['raw']['presented_match_count'])
        self.assertEqual(0, second['tracks']['raw']['presented_match_count'])
        self.assertTrue(all(targets == self.targets for _, _, targets in seen))

    def test_missing_offset_is_unavailable_and_no_rest_intervals_invented(self):
        result = self.run_analysis([{**self.trial, 'offset_sample_index': None,
                                    'window_start_sample_index': None, 'unavailable_reason': 'missing offset'}])
        self.assertFalse(result['available'])
        self.assertIsNone(result['trials'][0]['predictions']['raw']['target_id'])
        self.assertEqual([], result['controls']['rest'])
        self.assertIn('incomplete', result['controls']['rest_unavailable_reason'])

    def test_track_unavailability_does_not_replace_other_outputs(self):
        result = self.run_analysis(tracks={**self.tracks, 'traditional': None})
        self.assertTrue(result['tracks']['raw']['available'])
        self.assertFalse(result['tracks']['traditional']['available'])

    def test_validation_rejects_probability_and_configuration_changes(self):
        result = self.run_analysis()
        for key in ('production_validated', 'confidence_is_probability', 'detection_validated'):
            changed = copy.deepcopy(result); changed[key] = True
            with self.assertRaises(ValueError):
                analysis.validate_analysis(changed)
        changed = copy.deepcopy(result); changed['config']['sample_rate_hz'] = 256
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            analysis.validate_analysis(changed)


class BundleExtensionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OfflineBackendTests('test_flat_channel_no_traditional_or_fft')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_new_bundle_and_legacy_bundle_both_readable(self):
        f = self.fixture
        result = f.run_job()
        restored = backend.read_result(f.output, session_dir=f.source, root=f.root)
        self.assertEqual(result['personal_decoder'], restored['personal_decoder'])
        f.edit_result(lambda value: value['artifacts'].pop('personal_decoder/analysis.json'))
        with self.assertRaisesRegex(ValueError, 'receipts missing'):
            backend.read_result(f.output, session_dir=f.source, root=f.root)
        # Historical bundle schema and FIXED_PLAN remain exactly the same.
        result.pop('personal_decoder')
        for name in list(result['artifacts']):
            if name.startswith('personal_decoder/'):
                result['artifacts'].pop(name)
        (f.output / 'result.json').write_text(json.dumps(result), encoding='utf-8')
        legacy = backend.read_result(f.output, session_dir=f.source, root=f.root)
        self.assertNotIn('personal_decoder', legacy)
        self.assertEqual(backend.FIXED_PLAN, json.loads((f.output / 'plan.json').read_text()))

    def test_tampered_decoder_artifact_fails_read(self):
        f = self.fixture; f.run_job()
        (f.output / 'personal_decoder' / 'analysis.json').write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            backend.read_result(f.output, session_dir=f.source, root=f.root)


if __name__ == '__main__':
    unittest.main()
