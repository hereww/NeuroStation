"""Display-only fault tests: no Qt windows, serial ports, or physical flicker."""
from dataclasses import replace
import unittest

from eeg_tools.workstation.four_condition_presenter import (
    DisplayState, FrameSwap, PresentationCancelled, present_trial,
)
from neurostation_four_condition import FourConditionProtocol
from tests.test_ssvep_four_conditions import synthetic_value


class FakeDisplay:
    def __init__(self):
        self.now = 0.0
        self.generation = 0
        self.states = []
        self.blank_calls = 0
        self.delay_at = None
        self.stale = False
        self.fail_swap = False
        self.bad_time = False

    def submit(self, state):
        self.generation += 1
        self.states.append(state)
        return self.generation

    def wait_swap(self, generation):
        if self.fail_swap:
            raise TimeoutError('simulated missing display callback')
        self.now = max(self.now, getattr(self, 'previous', -1) + 1 / 60)
        frame = self.states[-1].frame
        if frame and frame.frame_index == self.delay_at:
            self.now += .25
        self.previous = self.now
        return FrameSwap(generation-1 if self.stale else generation,
                         float('nan') if self.bad_time else self.now)

    def wait(self, deadline):
        self.now = max(self.now, deadline)

    def emergency_blank(self):
        self.blank_calls += 1


class ConditionPresenterTests(unittest.TestCase):
    def setUp(self):
        self.protocol = FourConditionProtocol.from_dict(synthetic_value())
        self.display = FakeDisplay()
        self.rows = []
        self.cancelled = lambda: False

    def run_trial(self, index=60, journal=None):
        return present_trial(self.protocol, self.protocol.build_trials()[index], session_id='synthetic',
            display=self.display, clock=lambda: self.display.now, wait_until=self.display.wait,
            cancelled=lambda: self.cancelled(), journal=journal or self.rows.append)

    def test_complete_trial_events_and_unknown_quality(self):
        result = self.run_trial()
        self.assertEqual('completed', result['status'])
        self.assertIsNone(result['valid_trial'])
        events = [r for r in self.rows if r['kind'] == 'event']
        self.assertEqual(['trial_start','cue_onset','baseline_onset','stimulus_onset',
                          'stimulus_offset','rest_onset','trial_complete'], [r['event'] for r in events])
        self.assertEqual('software_monotonic_clock', events[0]['time_source'])
        self.assertEqual('display_swap_callback', events[3]['time_source'])
        self.assertEqual('unknown', events[-1]['signal_quality'])
        self.assertTrue(all(r['eeg_sample_index'] is None for r in self.rows))
        self.assertTrue(all(r['physical_optical_time_s'] is None for r in self.rows))
        self.assertTrue(all(not r['optical_timing_verified'] for r in self.rows))

    def test_end_blank_and_no_reverse_motion(self):
        self.run_trial()
        frames = [r for r in self.rows if r['kind'] == 'frame']
        self.assertEqual(241, len(frames))
        self.assertEqual(sorted(r['x'] for r in frames), [r['x'] for r in frames])
        self.assertAlmostEqual(.8, frames[-1]['x'])
        self.assertFalse(frames[-1]['target_visible'])
        self.assertFalse(frames[-1]['lit'])
        self.assertTrue(self.rows[-3]['endpoint_frame_presented'])

    def test_two_trials_reset_while_hidden(self):
        self.run_trial(60)
        first_count = len(self.display.states)
        self.run_trial(61)
        boundary = self.display.states[first_count-2:first_count+3]
        self.assertEqual(['stimulus_offset','rest','cue','baseline','stimulus'], [s.phase for s in boundary])
        self.assertFalse(boundary[0].frame.target_visible)
        self.assertIsNone(boundary[1].frame)
        self.assertIsNone(boundary[2].frame)
        self.assertAlmostEqual(.2, boundary[-1].frame.x)

    def test_dropped_callbacks_skip_without_reversing(self):
        self.display.delay_at = 10
        self.run_trial()
        frames = [r for r in self.rows if r['kind'] == 'frame']
        self.assertLess(len(frames), 241)
        self.assertTrue(any(r['dropped_since_previous'] > 0 for r in frames))
        self.assertEqual(sorted(r['x'] for r in frames), [r['x'] for r in frames])
        self.assertTrue(any(b['frame_index']-a['frame_index'] > 1 for a,b in zip(frames,frames[1:])))

    def test_late_end_does_not_claim_endpoint_was_presented(self):
        self.display.delay_at = 235
        result = self.run_trial()
        self.assertFalse(result['endpoint_frame_presented'])
        self.assertGreater(next(r for r in self.rows if r.get('event') == 'stimulus_offset')['actual_duration_s'], 4)

    def test_cancel_before_trial(self):
        self.cancelled = lambda: True
        with self.assertRaises(PresentationCancelled):
            self.run_trial()
        self.assertEqual([], self.display.states)
        self.assertEqual(1, self.display.blank_calls)
        self.assertEqual('trial_abort', self.rows[-1]['event'])

    def test_cancel_during_stimulus_blanks_without_completion(self):
        self.cancelled = lambda: len(self.display.states) >= 12
        with self.assertRaises(PresentationCancelled):
            self.run_trial()
        self.assertEqual(1, self.display.blank_calls)
        self.assertFalse(any(r.get('event') == 'trial_complete' for r in self.rows))
        self.assertIsNone(self.rows[-1]['blank_optical_time_s'])

    def test_invalid_swap_or_timeout_safe_stop(self):
        for attr in ('stale','bad_time','fail_swap'):
            with self.subTest(attr=attr):
                self.display = FakeDisplay()
                setattr(self.display, attr, True)
                with self.assertRaises((RuntimeError, TimeoutError)):
                    self.run_trial()
                self.assertEqual(1, self.display.blank_calls)
                self.assertEqual('trial_error', self.rows[-1]['event'])

    def test_journal_failure_safe_stop(self):
        def failing_writer(row):
            if row['kind'] == 'frame':
                raise OSError('simulated disk failure')
            self.rows.append(row)
        with self.assertRaises(OSError):
            self.run_trial(journal=failing_writer)
        self.assertEqual(1, self.display.blank_calls)
        self.assertEqual('trial_error', self.rows[-1]['event'])

    def test_every_journal_write_failure_preserves_original_exception(self):
        def failing_writer(row):
            raise OSError('unwritable')
        with self.assertRaisesRegex(OSError, 'unwritable'):
            self.run_trial(journal=failing_writer)
        self.assertEqual(1, self.display.blank_calls)

    def test_wrong_trial_cannot_present(self):
        trial = replace(self.protocol.build_trials()[0], condition='R_TO_L')
        with self.assertRaises(ValueError):
            present_trial(self.protocol, trial, session_id='synthetic', display=self.display,
                clock=lambda: 0, wait_until=lambda t: None, cancelled=lambda: False, journal=self.rows.append)
        self.assertEqual([], self.display.states)

    def test_adapter_uses_generation_and_hides_on_error(self):
        from unittest.mock import Mock, patch
        from eeg_tools.workstation.acquisition_worker import QtConditionDisplay
        window = Mock(frame_swapped_count=7, swapped_generation=9)
        window.show_condition.return_value = 9
        adapter = QtConditionDisplay(Mock(), window, self.protocol)
        self.assertEqual(9, adapter.submit(DisplayState('baseline', 'L_FIXED', '+')))
        with patch('eeg_tools.workstation.acquisition_worker._wait_for_frame_swap', return_value=1.5) as wait:
            self.assertEqual(FrameSwap(9, 1.5), adapter.wait_swap(9))
            self.assertEqual(9, wait.call_args.kwargs['minimum_generation'])
        adapter.emergency_blank()
        window.hide.assert_called_once()


if __name__ == '__main__':
    unittest.main()
