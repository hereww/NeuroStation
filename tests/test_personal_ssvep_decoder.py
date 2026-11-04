"""Synthetic-only decoder tests; never access personal recordings."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from eeg_tools import personal_ssvep_decoder as decoder

TARGETS = [('ten', 10.0), ('twelve', 12.0), ('fifteen', 15.0), ('twenty', 20.0)]


def mixture(frequency=12.0, samples=1500, seed=411):
    rng = np.random.default_rng(seed)
    t = np.arange(samples) / 250.0
    # Common known-frequency source, independent channel noise, mains and drift.
    phase = rng.uniform(-np.pi, np.pi, (8, 1))
    known = rng.uniform(.8, 1.5, (8, 1)) * np.sin(2 * np.pi * frequency * t + phase)
    mains = .8 * np.sin(2 * np.pi * 50 * t + phase)
    drift = 1.5 * np.sin(2 * np.pi * .25 * t + phase)
    return known + mains + drift + .5 * rng.normal(size=(8, samples))


class PersonalSSVEPDecoderTests(unittest.TestCase):
    def test_known_frequency_mixtures_all_candidates(self):
        for identifier, frequency in TARGETS:
            with self.subTest(frequency=frequency):
                result = decoder.decode_window(mixture(frequency), 250, TARGETS)
                self.assertEqual(result['target_id'], identifier)
                self.assertGreater(result['scores'][identifier], .85)
                self.assertGreater(result['confidence'], .5)
                self.assertTrue(result['available'])
                self.assertFalse(result['detection_validated'])

    def test_two_frequency_mix_is_subspace_correlation_not_power_classifier(self):
        t = np.arange(2000) / 250
        rng = np.random.default_rng(3)
        # Both known sources can be extracted by spatial CCA, even when one
        # source has much lower amplitude; both should receive a high score.
        x = (rng.normal(size=(8, 1)) * np.sin(2 * np.pi * 10 * t)
             + .2 * rng.normal(size=(8, 1)) * np.sin(2 * np.pi * 15 * t)
             + .015 * rng.normal(size=(8, len(t))))
        result = decoder.decode_window(x, 250, TARGETS)
        self.assertGreater(result['scores']['ten'], .9)
        self.assertGreater(result['scores']['fifteen'], .9)
        self.assertLess(result['scores']['twelve'], .25)

    def test_noise_controls_never_claim_reliable_detection(self):
        rng = np.random.default_rng(909)
        margins = []
        maxima = []
        for _ in range(8):
            result = decoder.decode_window(rng.normal(size=(8, 1500)), 250, TARGETS)
            self.assertTrue(result['available'])
            self.assertIn(result['target_id'], dict(TARGETS))
            self.assertEqual(result['target_selection'], 'forced_argmax_diagnostic_only')
            self.assertEqual(result['available_definition'], 'computation_succeeded_only_not_ssvep_detection')
            for field in ('detection_validated', 'confidence_is_probability',
                          'production_validated', 'gaze_accuracy_claim'):
                self.assertIs(result[field], False)
            self.assertNotIn('reliable_target', result)
            self.assertNotIn('ssvep_detected', result)
            maxima.append(max(result['scores'].values()))
            margins.append(result['confidence'])
        # These are fixed synthetic controls, not a production detection cutoff.
        self.assertLess(np.mean(maxima), .35)
        self.assertLess(np.mean(margins), .4)

    def test_non_target_reference_control(self):
        rng = np.random.default_rng(208)
        t = np.arange(2000) / 250
        x = np.sin(2 * np.pi * 17 * t)[None, :] * rng.uniform(.5, 2, (8, 1))
        x += .08 * rng.normal(size=x.shape)
        result = decoder.decode_window(x, 250, TARGETS + [('seventeen', 17)])
        self.assertEqual(result['target_id'], 'seventeen')
        without = decoder.decode_window(x, 250, TARGETS)
        self.assertLess(max(without['scores'].values()), .4)
        self.assertFalse(without['detection_validated'])

    def test_only_first_harmonic_is_used(self):
        x = mixture(24)
        result = decoder.decode_window(x, 250, [('twelve', 12), ('twentyfour', 24)])
        self.assertEqual(result['target_id'], 'twentyfour')
        self.assertLess(result['scores']['twelve'], .3)
        self.assertGreater(result['scores']['twentyfour'], .85)

    def test_global_scale_sign_and_channel_dc_invariance(self):
        x = mixture()
        baseline = decoder.decode_window(x, 250, TARGETS)
        for scale in (1e-200, 1e-9, .25, 1e9, 1e200, -7):
            with self.subTest(scale=scale):
                result = decoder.decode_window(x * scale, 250, TARGETS)
                np.testing.assert_allclose(list(result['scores'].values()),
                                           list(baseline['scores'].values()), rtol=1e-8, atol=1e-9)
        dc = np.arange(8)[:, None] * 7000 + 10000
        shifted = decoder.decode_window(x + dc, 250, TARGETS)
        np.testing.assert_allclose(list(shifted['scores'].values()),
                                   list(baseline['scores'].values()), rtol=1e-7, atol=1e-8)

    def test_input_is_not_mutated(self):
        x = mixture()
        original = x.copy()
        decoder.decode_window(x, 250, TARGETS)
        np.testing.assert_array_equal(x, original)

    def test_minimum_window_and_rank_deficient_channels(self):
        x = np.tile(np.sin(2 * np.pi * 12 * np.arange(250) / 250), (8, 1))
        result = decoder.decode_window(x, 250, TARGETS)
        self.assertEqual(result['target_id'], 'twelve')
        self.assertEqual(result['samples'], 250)
        self.assertTrue(all(0 <= score <= 1 for score in result['scores'].values()))

    def test_invalid_sample_rates(self):
        for rate in (None, True, '250', 249.9, 256, 0, float('nan'), float('inf')):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                decoder.decode_window(mixture(), rate, TARGETS)

    def test_invalid_shapes_and_types(self):
        invalid = [[], np.ones(250), np.ones((7, 500)), np.ones((9, 500)),
                   np.ones((500, 8)), np.ones((8, 249)), np.ones((8, 500, 1)),
                   np.ones((8, 500), dtype=complex), np.ones((8, 500), dtype=bool),
                   [['bad'] * 500] * 8, [[1, 2], [3]], None]
        for x in invalid:
            with self.subTest(shape=getattr(x, 'shape', None)), self.assertRaises(ValueError):
                decoder.decode_window(x, 250, TARGETS)

    def test_nonfinite_and_flat_channels(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            x = mixture(); x[3, 200] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                decoder.decode_window(x, 250, TARGETS)
        for value in (0, 1000):
            x = mixture(); x[3] = value
            with self.assertRaisesRegex(ValueError, 'flat'):
                decoder.decode_window(x, 250, TARGETS)

    def test_numerically_degenerate_channel_rejected(self):
        x = mixture(); x[4] *= 1e-30
        with self.assertRaisesRegex(ValueError, 'degenerate'):
            decoder.decode_window(x, 250, TARGETS)

    def test_invalid_target_ids(self):
        for identifier in ('', ' ', '\n', 'a\n', ' a', 'a ', None, 42, True):
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                decoder.decode_window(mixture(), 250, [(identifier, 12)])
        with self.assertRaisesRegex(ValueError, 'IDs must be unique'):
            decoder.decode_window(mixture(), 250, [('a', 10), ('a', 12)])

    def test_invalid_frequencies(self):
        for frequency in (None, True, '12', 0, -10, 125, 126, complex(12, 0),
                          float('nan'), float('inf'), 10**400):
            with self.subTest(frequency=frequency), self.assertRaises(ValueError):
                decoder.decode_window(mixture(), 250, [('a', frequency)])
        with self.assertRaisesRegex(ValueError, 'frequencies must be unique'):
            decoder.decode_window(mixture(), 250, [('a', 12), ('b', 12.0)])
        with self.assertRaises(ValueError):
            decoder.decode_window(mixture(), 250, [('a', 5e-324)])

    def test_invalid_candidate_structures(self):
        for targets in (None, [], 'abc', {'a': 12}, [('a',)], [('a', 12, 'label')]):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                decoder.decode_window(mixture(), 250, targets)

    def test_margin_and_single_candidate_are_not_probabilities(self):
        result = decoder.decode_window(mixture(), 250, TARGETS)
        first, second = sorted(result['scores'].values(), reverse=True)[:2]
        self.assertAlmostEqual(result['confidence'], (first - second) / first)
        self.assertAlmostEqual(result['relative_score'], first / sum(result['scores'].values()))
        single = decoder.decode_window(mixture(), 250, [('anything', 12)])
        self.assertEqual(single['confidence'], 0)
        self.assertFalse(single['confidence_is_probability'])

    def test_forward_has_no_label_argument_and_ids_do_not_change_scores(self):
        self.assertEqual(list(inspect.signature(decoder.decode_window).parameters),
                         ['signal', 'sample_rate_hz', 'targets'])
        x = mixture()
        original = decoder.decode_window(x, 250, TARGETS)
        renamed = decoder.decode_window(x, 250, [(f'id{i}', f) for i, (_, f) in enumerate(TARGETS)])
        self.assertEqual(list(original['scores'].values()), list(renamed['scores'].values()))
        generated = decoder.decode_window(x, 250, (pair for pair in TARGETS))
        self.assertEqual(original, generated)
        with self.assertRaises(TypeError):
            decoder.decode_window(x, 250, TARGETS, labels=['twelve'])

    def test_config_snapshot_fingerprint_and_detachment(self):
        config = decoder.get_config()
        self.assertEqual(config, json.loads(decoder.CONFIG_PATH.read_text(encoding='utf-8')))
        canonical = json.dumps(config, sort_keys=True, separators=(',', ':'),
                               ensure_ascii=False, allow_nan=False).encode('utf-8')
        digest = hashlib.sha256(canonical).hexdigest()
        self.assertEqual(decoder.config_sha256(), digest)
        self.assertEqual(decoder.config_summary()['config'], config)
        self.assertEqual(decoder.config_summary()['config_sha256'], digest)
        self.assertEqual(decoder.decode_window(mixture(), 250, TARGETS)['config_sha256'], digest)
        config['preprocessing']['bandpass_hz'][0] = 5
        self.assertEqual(decoder.get_config()['preprocessing']['bandpass_hz'][0], 1)
        with self.assertRaises(TypeError):
            decoder.CONFIG['reference_harmonics'] = 2
        self.assertEqual(decoder.CONFIG['reference_harmonics'], 1)
        self.assertEqual(decoder.CONFIG['bandpass_hz'], (1, 45))

    def test_config_rejects_drift_types_missing_extra_and_duplicate_keys(self):
        base = decoder.get_config()
        variants = []
        for key, value in [('reference_harmonics', 2), ('sample_rate_hz', 256),
                           ('production_validated', True), ('detection_validated', 0),
                           ('schema_version', True), ('extra', 1)]:
            changed = json.loads(json.dumps(base)); changed[key] = value
            variants.append(json.dumps(changed))
        missing = base.copy(); missing.pop('method'); variants.append(json.dumps(missing))
        variants += ['null', '[]', '{', '{"status":"formal","status":"formal"}',
                     '{"reference_harmonics":NaN}']
        for contents in variants:
            with self.subTest(contents=contents[:80]), patch.object(Path, 'read_text', return_value=contents):
                with self.assertRaises(ValueError):
                    decoder.get_config()
                with self.assertRaises(ValueError):
                    decoder.decode_window(mixture(), 250, TARGETS)
        with patch.object(Path, 'read_text', side_effect=FileNotFoundError('missing')):
            with self.assertRaises(ValueError):
                decoder.validate_config()

    def test_config_path_keyword_and_whitespace_stability(self):
        contents = json.dumps(decoder.get_config(), sort_keys=True, indent=4)
        expected = decoder.config_sha256()
        with patch.object(Path, 'read_text', return_value=contents) as read:
            self.assertEqual(decoder.validate_config(config_path='snapshot.json')['status'], 'formal')
            self.assertEqual(decoder.config_sha256(config_path='snapshot.json'), expected)
            self.assertEqual(read.call_count, 2)

    def test_numerical_failures_are_not_silently_valid_outputs(self):
        for score in (float('nan'), float('inf'), -1e-3, 1.1):
            with patch.object(decoder, '_canonical_score', return_value=score), self.assertRaises(ValueError):
                decoder.decode_window(mixture(), 250, TARGETS)
        for error in (np.linalg.LinAlgError('failed'), FloatingPointError('overflow')):
            with patch.object(decoder, '_canonical_score', side_effect=error), self.assertRaisesRegex(ValueError, 'failure'):
                decoder.decode_window(mixture(), 250, TARGETS)

    def test_exact_ties_are_explicit_forced_argmax(self):
        with patch.object(decoder, '_canonical_score', return_value=.2):
            result = decoder.decode_window(mixture(), 250, TARGETS)
        self.assertEqual(result['target_id'], 'ten')
        self.assertEqual(result['confidence'], 0)
        self.assertFalse(result['detection_validated'])

    def test_stdlib_only_import_and_config_api_in_fresh_interpreter(self):
        code = '''import sys
from eeg_tools import personal_ssvep_decoder as d
assert not any(n in sys.modules for n in ('torch','scipy','numpy'))
assert d.get_config()['reference_harmonics'] == 1
assert len(d.config_sha256()) == 64
assert d.config_summary()['config']['status'] == 'formal'
assert not any(n in sys.modules for n in ('torch','scipy','numpy'))
'''
        completed = subprocess.run([sys.executable, '-B', '-c', code],
                                   cwd=Path(__file__).resolve().parents[1],
                                   capture_output=True, text=True, timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == '__main__':
    unittest.main()
