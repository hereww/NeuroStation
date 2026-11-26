"""Regression cases for marker-blind inference, event ownership and shared live replay."""
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from eeg_tools.continuous_ssvep import Decision, Window, Gate, HOP
from eeg_tools.live_ssvep import LiveSSVEP, load_personal_gate
from eeg_tools.ssvep_replay import Interval, Signal, compare_sessions, process_session, score, totals


def fixture(seed=71):
    idx = np.arange(5000)
    t = idx / 250
    eeg = np.random.default_rng(seed).normal(0, .15, (len(t), 8))
    eeg[:, 0] += np.sin(2 * np.pi * 12 * t)
    return Signal(idx, idx % 256, t, eeg)


def session(root, hour, seed=71):
    root.mkdir()
    sig = fixture(seed)
    def save(name, value):
        (root / name).write_text(json.dumps(value), encoding='utf-8')
    save('session.json', dict(session_id=root.name, participant_id='fixture', started_at=f'2026-11-21T{hour}:00:00+08:00'))
    save('protocol.json', dict(sampling_rate_hz=250, channel_count=8, targets=[[str(f),f] for f in (10,12,15,20)],
                               onset_marker_base=100, offset_marker_base=200, post_trial_rest_s=5))
    events = [('stimulus_onset',0,12,1000,112), ('stimulus_offset',0,12,2250,212),
              ('stimulus_onset',1,10,3500,110), ('stimulus_offset',1,10,4750,210)]
    (root/'events.tsv').write_text('event_name\ttrial_index\tfrequency_hz\tsample_index\tmarker_code\tpresented_target_id\n'+
        ''.join(f'{name}\t{trial}\t{hz}\t{pos}\t{code}\t{hz}\n' for name,trial,hz,pos,code in events), encoding='utf-8')
    (root/'frame_timing.tsv').write_text('trial_index\tpresentation_phase\tlit\n0\toffset\t0\n1\toffset\t0\n', encoding='utf-8')
    mark = np.zeros(len(sig.index))
    for _,_,_,pos,code in events:
        mark[pos] = code
    data = np.column_stack((sig.index, sig.packet, sig.clock + int(hour)*3600, sig.eeg, mark))
    header = '\t'.join(['sample_index','package_num','timestamp_s',*(f'eeg_ch{i}' for i in range(1,9)),'marker'])
    np.savetxt(root/'raw_brainflow.tsv', data, delimiter='\t', header=header, comments='')


class ContinuousSSVEPTests(unittest.TestCase):
    def test_first_target_then_correction_rest_boundary_and_miss(self):
        emitted = [None,10,12,12,None,None,None]
        stream = [Decision(Window((i+1)*HOP-1,'ok'), hz, 'target' if hz else 'no_target') for i,hz in enumerate(emitted)]
        parts = [Interval(0,'stimulus',0,199,12), Interval(0,'rest',199,250,None), Interval(1,'stimulus',250,350,20)]
        a, b, c = score(parts, stream)
        self.assertEqual((a['result'], a['first_sample'], a['first_correct_sample']),('wrong',99,149))
        self.assertEqual((a['onset_to_correct_s'],a['wrong_to_correct_s']), (149/250,.2))
        self.assertEqual((b['result'],c['result']), ('false_alarm','miss'))
        self.assertIsNone(totals([a,c])['false_alarm_rate'])
        self.assertEqual(score(parts,stream[:-1])[-1]['result'],'excluded')

    def test_quiet_noise_and_all_four_targets_after_switch_off(self):
        from scripts.audit_continuous_ssvep_synthetic import case
        for amplitude in (None, 1.2):
            idx, eeg, segments = case(3149, amplitude)
            sig = Signal(idx, idx % 256, idx / 250, eeg)
            result = totals(score(segments, list(sig.replay(Gate(), 127))))
            self.assertLess(result['false_alarm'], result['rest_n'] // 2)
            if amplitude is not None:
                self.assertEqual(result['correct'], 8)

    def test_stationary_line_and_superimposed_targets(self):
        from scripts.audit_continuous_ssvep_synthetic import case
        for hz in (10, 12, 15, 20):
            idx, eeg, segments = case(3149, 1.2, background=hz, random_phase=True)
            sig = Signal(idx, idx % 256, idx / 250, eeg)
            metrics = totals(score(segments, list(sig.replay(Gate(), 127))))
            self.assertEqual((metrics['correct'], metrics['stimulus_n'], metrics['rest_n']), (8, 8, 8))
        sig = fixture()
        stream = list(sig.replay(Gate(), 17))
        self.assertTrue(all(d.hz is None for d in stream))
        self.assertTrue(all(d.state == 'no_target' for d in stream if d.window.end_sample >= 499))

    def test_weak_evidence_and_cached_calibration_match_stream(self):
        from scripts.audit_continuous_ssvep_synthetic import case
        from eeg_tools.continuous_ssvep import DecisionGate
        found = 0
        for seed in range(8293, 8298):
            idx, eeg, segments = case(seed, .15, random_phase=True)
            sig = Signal(idx, idx % 256, idx / 250, eeg)
            baseline = list(sig.replay(Gate(), len(idx)))
            found += totals(score(segments, baseline))['correct']
            custom = Gate(.45, .08, 3)
            cached = DecisionGate(custom)
            self.assertEqual([cached.apply(d.window) for d in baseline], list(sig.replay(custom, 37)))
        self.assertGreater(found, 0)

    def test_causal_chunking_live_persistence_and_gap(self):
        sig = fixture()
        # Leave a stationary 12 Hz background and add a genuine post-start change.
        sig.eeg[700:1850, 1] += np.sin(2 * np.pi * 12 * sig.clock[700:1850])
        sig.index[1900:] += 75  # Missed interval between acquisition callbacks.
        sig.packet[1900:] = (sig.packet[1900:] + 75) % 256
        sig.eeg[3500,2] = np.nan
        expected = list(sig.replay(Gate(),len(sig.index)))
        self.assertEqual(expected,list(sig.replay(Gate(),13)))
        self.assertTrue(any(d.window.quality=='missing_sample_index' for d in expected))
        self.assertTrue(any(d.window.quality=='nonfinite_eeg' for d in expected))
        self.assertIn(12,[d.hz for d in expected])
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sink = LiveSSVEP(root, range(2,10), 0, 1, Gate(), root/'preview.json')
            block = np.vstack((sig.packet, sig.clock, sig.eeg.T))
            try:
                for lo, hi in ((0,1900),(1900,len(sig.index))):
                    for a in range(lo,hi,17):
                        b = min(a+17,hi)
                        sink.consume(block[:,a:b],dict(start=int(sig.index[a]),stop=int(sig.index[b-1])+1))
            finally:
                sink.file.close()
            with sink.path.open(encoding='utf-8-sig') as f:
                rows=list(csv.DictReader(f, delimiter='\t'))
            self.assertEqual([(int(r['end_sample']),r['state'],r['frequency_hz']) for r in rows],
                             [(d.window.end_sample,d.state,str(d.hz or '')) for d in expected])
            self.assertEqual(sink.metadata(int(sig.index[-1])+1)['unprocessed_samples'],75)
            self.assertEqual(json.loads((root/'preview.json').read_text())['end_sample'],expected[-1].window.end_sample)
        sig.eeg[2000:] = 17.0
        altered=list(sig.replay(Gate(),128))
        self.assertEqual([d for d in expected if d.window.end_sample<sig.index[2000]],
                         [d for d in altered if d.window.end_sample<sig.index[2000]])

    def test_reexport_and_overlapping_calibration_sessions_are_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cal, test = root / 'cal', root / 'test'
            session(cal, '11')
            session(test, '12')  # same EEG with a different clock, file hash and ID
            with self.assertRaisesRegex(ValueError, 'Duplicated EEG'):
                compare_sessions(test, cal, root / 'duplicate')
            self.assertFalse((root / 'duplicate').exists())
            meta = cal / 'session.json'
            value = json.loads(meta.read_text())
            value['ended_at'] = '2026-11-21T12:03:00+08:00'
            meta.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'must finish before'):
                compare_sessions(test, cal, root / 'overlap')

    def test_completed_trials_without_events_remain_in_coverage(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            test = root / 'test'
            session(test, '12')
            markers = test / 'events.tsv'
            markers.write_text('\n'.join(markers.read_text().splitlines()[:3]) + '\n')
            (test / 'frame_timing.tsv').unlink()
            meta = test / 'session.json'
            value = json.loads(meta.read_text())
            for status, planned in [('completed', 2), ('aborted', 8)]:
                value.update(status=status, trial_count=planned, completed_trials=2)
                meta.write_text(json.dumps(value))
                summary = compare_sessions(test, None, root / status)
                m = summary['metrics']['baseline']
                self.assertEqual((m['stimulus_n'], m['stimulus_total'], m['rest_n']), (1, 2, 0))
                self.assertEqual(summary['exclusions']['missing_or_duplicate_event'], 1)
            value.update(headless=True)
            meta.write_text(json.dumps(value))
            results = compare_sessions(test, None, root / 'headless')
            self.assertIsNone(results['metrics']['baseline']['recognition'])
            self.assertTrue((root / 'headless' / 'baseline_predictions.tsv').is_file())

    def test_single_recording_needs_no_personal_model(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            test, output = root / 'only', root / 'baseline'
            session(test, '12')
            result = compare_sessions(test, None, output)
            self.assertEqual(set(result['metrics']), {'baseline'})
            self.assertIsNone(result['calibration'])
            metrics = result['metrics']['baseline']
            self.assertEqual((metrics['stimulus_n'], metrics['rest_n']), (2, 1))
            self.assertTrue((output / 'baseline_predictions.tsv').is_file())
            for name in ('personal_predictions.tsv', 'calibration_grid.json'):
                self.assertFalse((output / name).exists())
                self.assertNotIn(name, (output / 'report.md').read_text(encoding='utf-8'))
            with self.assertRaisesRegex(ValueError, 'personal calibration'):
                load_personal_gate(output / 'frozen_models.json', 'fixture')

    def test_unlabelled_export_matches_live_and_keeps_presentation_logs_separate(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cal, test = root / 'history', root / 'recording'
            session(cal, '11', seed=19)
            session(test, '12')
            for name in ('events.tsv', 'frame_timing.tsv'):
                (test / name).unlink()
            output = root / 'export'
            manifest = process_session(test, cal, output, chunk=37)
            self.assertEqual(json.loads((output / 'processing_manifest.json').read_bytes()), manifest)
            self.assertEqual(manifest['annotations'], [])
            self.assertEqual(set(manifest['predictions']), {'baseline', 'personal'})
            self.assertFalse(any((output / name).exists() for name in
                                 ('report.md', 'metrics.json', 'intervals.tsv', 'calibration_grid.json')))
            signal = Signal.load(test)
            block = np.vstack((signal.packet, signal.clock, signal.eeg.T))
            for name, gate in [('baseline', Gate()), ('personal', load_personal_gate(output / 'frozen_models.json', 'fixture'))]:
                live = root / name
                live.mkdir()
                sink = LiveSSVEP(live, range(2, 10), 0, 1, gate)
                try:
                    sink.consume(block, {'start': 0, 'stop': block.shape[1]})
                finally:
                    sink.file.close()
                self.assertEqual((output / manifest['predictions'][name]).read_bytes(), sink.path.read_bytes())
            # Even a log the scorer cannot parse is copied verbatim, never used for inference.
            marker_file = test / 'events.tsv'
            marker_file.write_bytes(b'malformed\r\n')
            second = root / 'second'
            self.assertEqual(process_session(test, None, second)['annotations'], ['events.tsv'])
            self.assertEqual(marker_file.read_bytes(), (second / marker_file.name).read_bytes())
            self.assertEqual((second / 'baseline_predictions.tsv').read_bytes(), (output / 'baseline_predictions.tsv').read_bytes())

    def test_label_isolation_and_exclusions(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cal, test = root/'first', root/'second'
            session(cal,'11',seed=19)
            session(test,'12')
            first=compare_sessions(test,cal,root/'run1',chunk=247)
            self.assertEqual(first['metrics']['baseline']['stimulus_n'],2)
            self.assertEqual(first['metrics']['baseline']['rest_n'],1)
            frames = test / 'frame_timing.tsv'
            current_frames = frames.read_bytes()
            frames.write_text('trial_index\tlit\n0\t1\n0\t0\n1\t0\n', encoding='utf-8')
            legacy = compare_sessions(test, cal, root / 'legacy')
            for metrics in legacy['metrics'].values():
                self.assertEqual((metrics['stimulus_n'], metrics['rest_total'], metrics['rest_n']), (2, 1, 0))
                self.assertIsNone(metrics['false_alarm_rate'])
            self.assertEqual(legacy['exclusions'], {'unverified_rest_boundaries_or_offset_frame': 1})
            report = (root / 'legacy' / 'report.md').read_text(encoding='utf-8')
            self.assertIn('| 静息段误报 | 无法评估 | 无法评估 |', report)
            frames.write_bytes(current_frames)
            (test/'events.tsv').write_text((test/'events.tsv').read_text().replace('\t12\t1000','\t15\t1000'))
            second=compare_sessions(test,cal,root/'run2',chunk=13)
            self.assertEqual(second['metrics']['baseline']['stimulus_n'],1)
            self.assertIsNone(second['metrics']['baseline']['false_alarm_rate'])
            for f in ('baseline_predictions.tsv','personal_predictions.tsv','frozen_models.json'):
                self.assertEqual((root/'run1'/f).read_bytes(),(root/'run2'/f).read_bytes())
            path = root/'run1'/'frozen_models.json'
            self.assertIsInstance(load_personal_gate(path,'fixture'), Gate)
            with self.assertRaises(ValueError):
                load_personal_gate(path,'unrelated')
            old = root / 'v2.json'
            old.write_text(json.dumps({**json.loads(path.read_text()), 'method': 'continuous_cca_spectral_v2'}))
            with self.assertRaisesRegex(ValueError, 'Re-run personal calibration'):
                load_personal_gate(old, 'fixture')
            with self.assertRaises(ValueError):
                compare_sessions(cal,test,root/'reversed')
