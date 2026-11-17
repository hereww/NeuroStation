"""Synthetic planning fixtures: values below are NOT calibrated participant settings."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest

from eeg_tools.workstation.four_condition import (
    CONDITIONS, FourConditionProtocol, parameter_digest,
)
from eeg_tools.workstation.ssvep import SSVEPProtocol, SSVEPProtocolError
from scripts.plan_ssvep_four_conditions import main
from scripts.validate_release_configs import validate_release_configs

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/protocols/ssvep_four_condition_v1.json'


def synthetic_value():
    value = json.loads(CONFIG.read_text(encoding='utf-8'))
    value['parameters'].update(
        refresh_rate_hz=60, frequency_hz=10, duty_cycle=0.5,
        left_x=0.2, right_x=0.8, y=0.5, target_width=0.1, target_height=0.1,
        shape='rectangle', motion_mode='smooth_horizontal',
        gaze_instruction='central_fixation_attention_follow',
        background_level=0.2, on_level=1, off_level=0,
        cue_s=1, baseline_s=1, stimulus_s=4, rest_s=2,
        trials_per_block=15, blocks_per_condition=2,
        block_rest_s=30, condition_rest_s=60, makeup_policy='none_preserve_all_trials',
    )
    value['parameter_confirmation'] = {
        'confirmed_by': 'synthetic unit-test fixture, not a participant calibration',
        'parameter_sha256': parameter_digest(value['parameters']),
    }
    return value


class FourConditionTests(unittest.TestCase):
    def setUp(self):
        self.value = synthetic_value()
        self.protocol = FourConditionProtocol.from_dict(self.value)

    def test_formal_design_is_not_ready_by_default(self):
        protocol = FourConditionProtocol.load(CONFIG)
        self.assertIn('frequency_hz', protocol.pending)
        self.assertIn('parameter_confirmation', protocol.pending)
        with self.assertRaisesRegex(SSVEPProtocolError, 'not locked'):
            protocol.build_trials()
        with self.assertRaises(SSVEPProtocolError):
            protocol.frame('L_FIXED', 0)

    def test_every_parameter_requires_confirmation(self):
        for key in self.value['parameters']:
            value = deepcopy(self.value)
            value['parameter_confirmation'] = None
            value['parameters'][key] = None
            with self.subTest(key=key):
                protocol = FourConditionProtocol.from_dict(value)
                self.assertIn(key, protocol.pending)
                with self.assertRaises(SSVEPProtocolError):
                    protocol.build_trials()

    def test_stale_confirmation_rejected(self):
        self.value['parameters']['stimulus_s'] = 5
        with self.assertRaisesRegex(SSVEPProtocolError, 'stale'):
            FourConditionProtocol.from_dict(self.value)

    def test_missing_confirmation_blocks_filled_parameters(self):
        self.value['parameter_confirmation'] = None
        self.assertEqual(('parameter_confirmation',), FourConditionProtocol.from_dict(self.value).pending)

    def test_condition_order_cannot_be_randomized(self):
        self.value['condition_order'].reverse()
        with self.assertRaises(SSVEPProtocolError):
            FourConditionProtocol.from_dict(self.value)

    def test_fixed_order_common_frequency_and_distinct_markers(self):
        trials = self.protocol.build_trials()
        self.assertEqual(120, len(trials))
        self.assertEqual([c for c in CONDITIONS for _ in range(30)], [t.condition for t in trials])
        self.assertEqual({10}, {t.frequency_hz for t in trials})
        self.assertEqual(4, len({t.onset_marker for t in trials}))
        self.assertEqual(4, len({t.offset_marker for t in trials}))
        self.assertTrue(all(t.label_source == 'planned_condition_not_verified_gaze' for t in trials))

    def test_phase_timing_and_breaks(self):
        trials = self.protocol.build_trials()
        self.assertEqual((0, 1, 2, 6, 8), (trials[0].cue_onset_s, trials[0].baseline_onset_s,
                         trials[0].stimulus_onset_s, trials[0].stimulus_offset_s, trials[0].rest_end_s))
        self.assertEqual(30, trials[14].break_after_s)
        self.assertEqual(60, trials[29].break_after_s)
        self.assertEqual(0, trials[-1].break_after_s)
        self.assertEqual(1260, trials[-1].rest_end_s)
        for before, after in zip(trials, trials[1:]):
            self.assertEqual(before.rest_end_s + before.break_after_s, after.cue_onset_s)
        self.assertEqual(list(range(8)), sorted({t.block_id for t in trials}))

    def test_fixed_coordinates(self):
        for c, expected in [('L_FIXED', .2), ('R_FIXED', .8)]:
            self.assertTrue(all(self.protocol.frame(c, f).x == expected for f in range(241)))

    def test_single_direction_endpoints_and_terminal_blank(self):
        for c, start, end, sign in [('L_TO_R', .2, .8, 1), ('R_TO_L', .8, .2, -1)]:
            frames = [self.protocol.frame(c, f) for f in range(241)]
            self.assertAlmostEqual(start, frames[0].x)
            self.assertAlmostEqual(end, frames[-2].x)
            self.assertAlmostEqual(end, frames[-1].x)
            self.assertTrue(all(sign * (b.x - a.x) >= -1e-12 for a, b in zip(frames, frames[1:])))
            self.assertFalse(frames[-1].lit)
            self.assertFalse(frames[-1].target_visible)
            self.assertEqual('stimulus_offset', frames[-1].phase)
            self.assertEqual(4, frames[-1].scheduled_s)

    def test_frame_skip_preserves_direction(self):
        frames = [self.protocol.frame('L_TO_R', i) for i in (0, 2, 7, 99, 238, 240)]
        self.assertEqual(sorted(f.x for f in frames), [f.x for f in frames])
        self.assertFalse(frames[-1].target_visible)

    def test_flicker_is_independent_of_condition(self):
        expected = [True, True, True, False, False, False] * 40
        for c in CONDITIONS:
            self.assertEqual(expected, [self.protocol.frame(c, i).lit for i in range(240)])

    def test_invalid_frames_and_conditions(self):
        for index in (-1, 241, True, 1.5):
            with self.assertRaises(SSVEPProtocolError):
                self.protocol.frame('L_TO_R', index)
        with self.assertRaises(SSVEPProtocolError):
            self.protocol.frame('BACK_AND_FORTH', 0)

    def test_invalid_numbers_rejected_before_confirmation(self):
        for x in (True, float('nan'), float('inf'), -1, 0, '10'):
            self.value['parameters']['frequency_hz'] = x
            with self.subTest(x=x), self.assertRaises(SSVEPProtocolError):
                FourConditionProtocol.from_dict(self.value)

    def test_resource_and_arithmetic_overflow_rejected(self):
        for key, value in [('trials_per_block', 1000000000), ('frequency_hz', 1e-308),
                           ('stimulus_s', 1e308), ('cue_s', 1e308)]:
            candidate = deepcopy(self.value)
            candidate['parameters'][key] = value
            with self.subTest(key=key), self.assertRaises(SSVEPProtocolError):
                FourConditionProtocol.from_dict(candidate)

    def test_non_integral_flicker_and_duration_rejected(self):
        for key, value in [('frequency_hz', 13), ('duty_cycle', .25), ('stimulus_s', .001)]:
            candidate = deepcopy(self.value)
            candidate['parameters'][key] = value
            with self.assertRaises(SSVEPProtocolError):
                FourConditionProtocol.from_dict(candidate)

    def test_unimplemented_variants_rejected(self):
        for key, value in [('motion_mode', 'stepwise'), ('gaze_instruction', 'eye_tracking'),
                           ('makeup_policy', 'repeat_until_valid'), ('shape', 'star')]:
            candidate = deepcopy(self.value)
            candidate['parameters'][key] = value
            with self.assertRaises(SSVEPProtocolError):
                FourConditionProtocol.from_dict(candidate)

    def test_invalid_geometry_rejected(self):
        for values in [dict(left_x=.9), dict(target_width=.8), dict(y=.01), dict(on_level=0)]:
            candidate = deepcopy(self.value)
            candidate['parameters'].update(values)
            with self.assertRaises(SSVEPProtocolError):
                FourConditionProtocol.from_dict(candidate)

    def test_parameter_schema_is_exact(self):
        self.value['parameters']['frequency_typo'] = 10
        with self.assertRaises(SSVEPProtocolError):
            FourConditionProtocol.from_dict(self.value)

    def test_caller_cannot_mutate_validated_plan(self):
        original = self.protocol.processing_identity
        self.value['parameters']['frequency_hz'] = 12
        parameters = self.protocol.parameters
        parameters['frequency_hz'] = 20
        self.assertEqual(10, self.protocol.parameters['frequency_hz'])
        self.assertEqual(original, self.protocol.processing_identity)

    def test_identity_changes_with_parameters(self):
        self.value['parameters']['stimulus_s'] = 5
        self.value['parameter_confirmation']['parameter_sha256'] = parameter_digest(self.value['parameters'])
        other = FourConditionProtocol.from_dict(self.value)
        self.assertNotEqual(self.protocol.processing_identity, other.processing_identity)

    def test_old_frequency_runner_rejects_spatial_protocol(self):
        with self.assertRaisesRegex(SSVEPProtocolError, 'schema_version must be 2'):
            SSVEPProtocol.load(CONFIG)
        old = SSVEPProtocol.load(ROOT / 'configs/protocols/ssvep_four_target_v2.json')
        self.assertEqual(40, len(old.build_trials()))
        self.assertEqual((10, 12, 15, 20), old.frequencies_hz)

    def test_release_reports_parameter_lock_separately(self):
        report = validate_release_configs(ROOT)
        self.assertEqual('ready', report['status'])
        self.assertIn(CONFIG.name, report['parameter_lock_pending'])

    def test_offline_inspection_is_pending_and_has_no_trials(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main([])
        self.assertEqual(2, code)
        report = json.loads(output.getvalue())
        self.assertNotIn('trials', report)
        self.assertFalse(report['hardware_started'])
        self.assertFalse(report['stimulus_presented'])
        self.assertFalse(report['live_execution_supported'])
        self.assertIsNone(report['valid_trial_count'])

    def test_cli_writes_locked_plan_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / 'input.json'
            target = Path(temp) / 'plan.json'
            config.write_text(json.dumps(self.value), encoding='utf-8')
            args = ['--protocol', str(config), '--output', str(target)]
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(0, main(args))
                saved = target.read_bytes()
                self.assertEqual(1, main(args))
            self.assertEqual(saved, target.read_bytes())
            self.assertEqual(120, json.loads(saved)['planned_trial_count'])


if __name__ == '__main__':
    unittest.main()
