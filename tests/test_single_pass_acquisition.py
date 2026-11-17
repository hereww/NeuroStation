"""Single-pass acquisition tests; no device connection or flickering window."""
import csv
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from contextlib import redirect_stderr
import time

from neurostation_contract import (
    CaptureConfig, StimulusCondition, dataset_name_for_condition,
    resolve_stimulus_condition, stimulus_condition_value,
)
from neurostation_display import stimulus_geometry, eye_half_geometry, ssvep_frame_is_lit
from eeg_tools.workstation import acquisition_worker as worker
from eeg_tools.workstation.process_gateway import AcquisitionProcessGateway
from eeg_tools.workstation.dataset import DatasetRepository
from eeg_tools.session_files import write_events

ROOT = Path(__file__).resolve().parents[1]


class MotionContractTests(unittest.TestCase):
    def test_four_choices_preserve_counts_and_legacy_names(self):
        base = CaptureConfig(participant="U001", eye_side="left", acknowledge_flicker_risk=True)
        for mode in StimulusCondition:
            config = replace(base, eye_side=mode.start_side, stimulus_condition=mode)
            config.validate()
            self.assertEqual(base.trials, config.trials)
            self.assertEqual(base.recording_seconds, config.recording_seconds)
            self.assertEqual(base.expected_events, config.expected_events)
            self.assertEqual(mode.value, stimulus_condition_value(config))
            self.assertEqual("SSVEP_" + mode.dataset_suffix, config.dataset_name)
        self.assertEqual("SSVEP_左眼", base.dataset_name)
        self.assertEqual("SSVEP_右眼", replace(base, eye_side="right").dataset_name)

    def test_generated_suffixes_do_not_accumulate(self):
        for mode in StimulusCondition:
            self.assertEqual("Test_" + mode.dataset_suffix,
                dataset_name_for_condition("Test_左眼_右到左_右眼_左到右", mode))

    def test_invalid_direction_and_missing_safety_ack_fail(self):
        for side, condition in (("", "left_to_right"), ("left", "right_to_left"),
                                ("left", "bounce")):
            with self.subTest(side=side, condition=condition), self.assertRaises(ValueError):
                resolve_stimulus_condition(side, condition)
        config = CaptureConfig(participant="U001", eye_side="right", stimulus_condition="right_to_left")
        with self.assertRaisesRegex(ValueError, "flicker_ack"):
            config.validate()

    def test_geometry_is_constant_size_monotonic_and_clamped(self):
        for width in (800, 801, 1920, 3840):
            for mode in (StimulusCondition.L_TO_R, StimulusCondition.R_TO_L):
                frames = [stimulus_geometry(-200, 15, width, 600, mode, i/99) for i in range(100)]
                xs = [f["x"] for f in frames]
                self.assertEqual(xs, sorted(xs, reverse=mode is StimulusCondition.R_TO_L))
                self.assertEqual(1, len({(f["width"], f["height"], f["y"]) for f in frames}))
                self.assertEqual(eye_half_geometry(-200,15,width,600,mode.start_side)["x"], xs[0])
                opposite = "right" if mode.start_side == "left" else "left"
                self.assertEqual(eye_half_geometry(-200,15,width,600,opposite)["x"], xs[-1])
                self.assertEqual(frames[0], stimulus_geometry(-200,15,width,600,mode,-2))
                self.assertEqual(frames[-1], stimulus_geometry(-200,15,width,600,mode,10))
                self.assertTrue(all(-200 <= f["x"] <= -200+width-f["width"] for f in frames))
        with self.assertRaises(ValueError):
            stimulus_geometry(0, 0, 800, 600, "left_to_right", float("nan"))

    def test_fixed_geometry_is_unchanged(self):
        for side in ("left", "right"):
            for position in (0, .5, 1, 2):
                self.assertEqual(eye_half_geometry(12, 20, 801, 600, side),
                    stimulus_geometry(12,20,801,600,side,position))

    def test_gateway_and_compatibility_cli_forward_condition_without_relaxing_gates(self):
        gateway = AcquisitionProcessGateway(protocol_path=ROOT/'configs/protocols/ssvep_four_target_v2.json',
                                             channel_config_path=ROOT/'configs/channel_config_v1_auto.json')
        for mode in (StimulusCondition.L_TO_R, StimulusCondition.R_TO_L):
            config = CaptureConfig(participant="U001", eye_side=mode.start_side,
                                   stimulus_condition=mode, acknowledge_flicker_risk=True)
            command = gateway._command(config)
            self.assertEqual(mode.value, command[command.index('--stimulus-condition')+1])
            self.assertEqual(mode.start_side, command[command.index('--eye-side')+1])
            self.assertIn('--acknowledge-flicker-risk', command)
            self.assertNotIn('--headless', command)
            self.assertNotIn('--allow-draft-channel-config', command)
        import run_ssvep_session
        with patch.object(run_ssvep_session, 'worker_main', return_value=0) as run:
            self.assertEqual(0, run_ssvep_session.main(['--eye-side','right',
                '--stimulus-condition','right_to_left','--acknowledge-flicker-risk']))
        args = run.call_args.args[0]
        self.assertEqual('right_to_left', args[args.index('--stimulus-condition')+1])

    def test_worker_rejects_headless_motion_or_mismatched_side_before_hardware(self):
        base = ['--output-root',str(ROOT),'--participant','test','--session-name','test',
                '--eye-side','left','--acknowledge-flicker-risk']
        for args in (['--stimulus-condition','left_to_right','--headless'],
                     ['--stimulus-condition','right_to_left']):
            with patch.object(worker, '_prepare_cyton_board') as hardware, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    worker.main(base+args)
            hardware.assert_not_called()

    def test_dataset_repository_preserves_explicit_direction_and_legacy_absence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, mode in enumerate(('', 'left_to_right', 'right_to_left')):
                session = root / str(i)
                session.mkdir()
                data = {'session_id':str(i), 'session_name':'capture', 'eye_side':'left',
                        'status':'completed', 'board':'cyton'}
                if mode:
                    data['stimulus_condition'] = mode
                (session/'session.json').write_text(json.dumps(data),encoding='utf-8')
                record = DatasetRepository._read_record(session/'session.json')
                self.assertIsNotNone(record)
                self.assertEqual(mode, record.stimulus_condition)
                self.assertEqual(mode or 'left', stimulus_condition_value(record))


class FakeWindow:
    def __init__(self):
        self.frame_swapped_count = 0
        self.abort_requested = False
        self.rendering_metadata = {}
        self.calls = []
        self.painted_stimulus_geometry = {}

    def show_motion_target(self, target, lit, condition, progress):
        self.calls.append((condition, progress, lit))
        self.painted_stimulus_geometry = stimulus_geometry(0,0,801,600,condition,progress)
        return len(self.calls)

    def show_target(self, target, lit):
        self.calls.append(('fixed', None, lit))
        return len(self.calls)


class MotionPresentationTests(unittest.TestCase):
    def present(self, mode, *, dropped=False, cancel_after=None, repeats=1):
        window = FakeWindow()
        frames, onsets, offsets = [], [], []
        clock = [100.0]
        calls = [0]
        def swap(application, candidate, previous, *, cancel_file, minimum_generation):
            self.assertIs(window, candidate)
            self.assertEqual(len(window.calls), minimum_generation)
            calls[0] += 1
            if cancel_after == calls[0]:
                raise worker.UserAbort()
            clock[0] += (3 if dropped and calls[0] == 5 else 1)/60
            window.frame_swapped_count += 1
            return clock[0]
        params = dict(application=object(),window=window,target_index=0,frequency_hz=12,
            duration_s=1,refresh_rate_hz=60,cancel_file=None,eye_side=mode.start_side,
            stimulus_condition=mode.value,screen_index=0,dataset_name_base='test',dataset_name='test',
            screen_name='test',screen_geometry='',screen_mapping='',stimulus_region='',
            frame_rows=frames,progress=lambda elapsed: None,
            on_stimulus_onset=onsets.append,on_stimulus_offset=offsets.append)
        with patch.object(worker,'_wait_for_frame_swap',side_effect=swap), \
             patch.object(worker,'_wait_phase'), patch.object(worker.time,'perf_counter',side_effect=lambda: clock[0]):
            for trial in range(repeats):
                worker._present_stimulus(trial_index=trial, **params)
        return window, frames, onsets, offsets

    def test_one_traversal_per_trial_ends_blank_before_next_start(self):
        for mode in (StimulusCondition.L_TO_R, StimulusCondition.R_TO_L):
            window, rows, onsets, offsets = self.present(mode,repeats=2)
            self.assertEqual(2,len(onsets))
            self.assertEqual(2,len(offsets))
            for trial in range(2):
                frames = [r for r in rows if r['trial_index']==trial]
                self.assertEqual(61,len(frames))
                self.assertEqual(0,frames[0]['motion_progress'])
                self.assertEqual(1,frames[-2]['motion_progress'])
                self.assertEqual('offset',frames[-1]['presentation_phase'])
                self.assertEqual(0,frames[-1]['lit'])
                positions = [json.loads(r['rendered_geometry_px'])['x'] for r in frames]
                self.assertEqual(positions,sorted(positions,reverse=mode is StimulusCondition.R_TO_L))
                self.assertTrue(all(r['condition_code']==mode.name for r in frames))
                self.assertTrue(all(r['condition_label_source']=='planned_condition_not_verified_gaze' for r in frames))
                self.assertEqual([int(ssvep_frame_is_lit(i,12,60)) for i in range(60)],
                                 [r['lit'] for r in frames[:-1]])
            self.assertEqual((mode.value,1.0,False),window.calls[60])
            self.assertEqual(0,window.calls[61][1])

    def test_dropped_frames_are_retained_not_claimed_as_valid_timing(self):
        _, frames, _, _ = self.present(StimulusCondition.L_TO_R,dropped=True)
        self.assertGreater(sum(r['dropped_since_previous'] for r in frames),0)
        self.assertEqual('review',worker._frame_timing_summary(frames,expected_refresh_rate_hz=60)['cadence_status'])
        self.assertEqual(sorted(r['motion_progress'] for r in frames),[r['motion_progress'] for r in frames])

    def test_cancellation_during_motion_propagates(self):
        with self.assertRaises(worker.UserAbort):
            self.present(StimulusCondition.R_TO_L,cancel_after=5)

    def test_fixed_presenter_still_uses_original_path(self):
        window, rows, _, _ = self.present(StimulusCondition.L_FIXED)
        self.assertTrue(all(call[0]=='fixed' for call in window.calls))
        self.assertTrue(all(row['motion_progress']=='' for row in rows))

    def test_cancel_file_is_checked_before_first_frame_is_submitted(self):
        window = FakeWindow()
        with tempfile.TemporaryDirectory() as tmp:
            cancel = Path(tmp)/'cancel'
            cancel.touch()
            with self.assertRaises(worker.UserAbort):
                worker._present_stimulus(application=object(),window=window,target_index=0,
                    frequency_hz=12,duration_s=1,refresh_rate_hz=60,cancel_file=cancel,
                    trial_index=0,eye_side='left',stimulus_condition='left_to_right',screen_index=0,
                    dataset_name_base='test',dataset_name='test',screen_name='',screen_geometry='',
                    screen_mapping='',stimulus_region='',frame_rows=[],progress=lambda _: None)
        self.assertEqual([],window.calls)

    def test_end_to_end_motion_saves_condition_with_raw_and_quality_artifacts(self):
        # Reuse the existing synthetic board fixture: no real serial/device calls.
        from tests.test_quality_main_capture import MainCaptureIntegration
        original_main = worker.main
        for mode in (StimulusCondition.L_TO_R, StimulusCondition.R_TO_L):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                window = FakeWindow()
                window.close = Mock()
                window.show_message = Mock()
                screen = Mock()
                screen.name.return_value = 'fixture'
                screen.refreshRate.return_value = 60
                screen.devicePixelRatio.return_value = 1
                rect = screen.geometry.return_value
                rect.x.return_value = rect.y.return_value = 0
                rect.width.return_value, rect.height.return_value = 801, 600
                mapping = {side: {'screen':screen,'screen_index':0,
                           'region_geometry':eye_half_geometry(0,0,801,600,side)}
                           for side in ('left','right')}
                clock = [time.perf_counter()]
                def swap(*args, **kwargs):
                    clock[0] += 1/60
                    window.frame_swapped_count += 1
                    return clock[0]
                def visual_main(args):
                    args = [arg for arg in args if arg != '--headless']
                    args[args.index('--eye-side')+1] = mode.start_side
                    return original_main(args+['--stimulus-condition',mode.value,
                                              '--acknowledge-flicker-risk'])
                with patch.object(worker,'main',side_effect=visual_main), \
                     patch.object(worker,'_create_stimulus_window',return_value=(Mock(),window,mapping)), \
                     patch.object(worker,'_wait_for_frame_swap',side_effect=swap):
                    blocks = MainCaptureIntegration().run_fake(root)
                window.close.assert_called_once()
                saved = blocks.parent
                session = json.loads((saved/'session.json').read_text(encoding='utf-8'))
                config = json.loads((saved/'ssvep_config.json').read_text(encoding='utf-8'))
                self.assertEqual('completed',session['status'])
                self.assertEqual(mode.value,session['stimulus_condition'])
                self.assertEqual(mode.value,config['runtime_overrides']['stimulus_condition'])
                self.assertEqual(mode.name,session['condition_code'])
                self.assertEqual('primary_screen_single_pass',session['display']['mode'])
                self.assertIsNone(session['display']['inactive'])
                self.assertEqual('unverified',session['display']['motion']['optical_timing'])
                self.assertEqual(4,session['completed_trials'])
                self.assertTrue((saved/'quality.json').is_file())
                self.assertTrue((saved/'event_associations_v2.json').is_file())
                with (saved/'events.tsv').open(encoding='utf-8',newline='') as stream:
                    events = list(csv.DictReader(stream,delimiter='\t'))
                self.assertEqual(4,sum(e['event_name']=='stimulus_onset' for e in events))
                self.assertEqual(4,sum(e['event_name']=='stimulus_offset' for e in events))
                self.assertTrue(all(e['condition_code']==mode.name for e in events))
                with (saved/'frame_timing.tsv').open(encoding='utf-8',newline='') as stream:
                    frames = list(csv.DictReader(stream,delimiter='\t'))
                self.assertEqual(4,sum(f['presentation_phase']=='offset' for f in frames))
                self.assertTrue(all(f['stimulus_condition']==mode.value for f in frames))
                record = DatasetRepository._read_record(saved/'session.json')
                self.assertEqual(mode.value,record.stimulus_condition)

    def test_logs_preserve_condition_and_frame_positions(self):
        _, frames, _, _ = self.present(StimulusCondition.R_TO_L)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'frames.tsv'
            worker._write_frame_log(path,frames)
            with path.open(encoding='utf-8',newline='') as stream:
                read = list(csv.DictReader(stream,delimiter='\t'))
            self.assertEqual('R_TO_L',read[0]['condition_code'])
            self.assertEqual('1.0',read[-1]['motion_progress'])
            event = {k:frames[0][k] for k in ('eye_side','stimulus_condition','condition_code','condition_label_source')}
            event.update(event_id=1,event_name='stimulus_onset',marker_code=1)
            write_events(Path(tmp)/'events.tsv',[event])
            with (Path(tmp)/'events.tsv').open(encoding='utf-8',newline='') as stream:
                self.assertEqual('right_to_left',next(csv.DictReader(stream,delimiter='\t'))['stimulus_condition'])


if __name__ == '__main__':
    unittest.main()
