"""Both main-workstation and standalone contracts use injected data, never devices."""
import contextlib,csv,io,json,tempfile,threading,time,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from brainflow.board_shim import BoardShim,BoardIds
from eeg_tools.workstation import acquisition_worker as worker
from eeg_tools.streaming import recover_blocks
ROOT=Path(__file__).resolve().parents[1]

class MainCaptureIntegration(unittest.TestCase):
    def run_fake(self,root,quality_failure=False, *, raw=True, realtime=False, expected_exit=0):
        board_id=BoardIds.CYTON_BOARD; rows=BoardShim.get_num_rows(board_id)
        eeg=BoardShim.get_eeg_channels(board_id);stamp=BoardShim.get_timestamp_channel(board_id);marker=BoardShim.get_marker_channel(board_id)
        class Board:
            def __init__(self):self.n=0;self.read_threads=set();self.markers=[];self.lock=threading.Lock()
            def start_stream(self):pass
            def stop_stream(self):pass
            def release_session(self):pass
            def insert_marker(self,m):
                with self.lock:self.markers.append(m)
            def get_current_board_data(self,*_):raise AssertionError('second reader forbidden')
            def get_board_data(self):
                with self.lock:
                    self.read_threads.add(threading.get_ident());n=25;i=np.arange(self.n,self.n+n)
                    x=np.zeros((rows,n));x[0]=i%256;x[stamp]=100+i/250
                    for c,row in enumerate(eeg):x[row]=np.sin(i/8+c)
                    if self.markers:x[marker,0]=self.markers.pop(0)
                    self.n+=n;return x
        board=Board()
        def wait(seconds,**kw):
            time.sleep(.03)
            if kw.get('update'):kw['update'](seconds)
        args=['--output-root',str(root),'--participant','synthetic_fixture','--session-name','quality_test',
              '--eye-side','left','--headless','--repetitions','1','--countdown-seconds','0',
              '--protocol',str(ROOT/'configs/protocols/ssvep_four_target_v2.json'),
              '--channel-config',str(ROOT/'configs/channel_config_v1_auto.json'),
              '--status-file',str(root/'status.json')]
        if not raw:
            args.append('--no-raw-eeg')
        if realtime:
            args.append('--continuous-ssvep')
        with contextlib.ExitStack() as stack:
            if realtime:
                # This fixture tests recording, independent of the host GPU. GPU
                # dispatch and its no-fallback contract have separate coverage.
                from eeg_tools.continuous_ssvep import ContinuousDetector
                stack.enter_context(patch('eeg_tools.ssvep_gpu.load_detector',
                    side_effect=lambda key, gate: (ContinuousDetector(gate), {'id':key,'device':'fixture'})))
            stack.enter_context(patch.object(worker, '_prepare_cyton_board', return_value=(board, 'FIXTURE', [])))
            stack.enter_context(patch.object(worker,'_wait_phase',side_effect=wait))
            if quality_failure:stack.enter_context(patch('eeg_tools.quality.engine.QualityEngine.process',side_effect=ValueError('fixture_quality_failure')))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            code=worker.main(args)
        self.assertEqual(code, expected_exit)
        if not raw:
            saved = next(root.rglob('session.json')).parent
            self.assertEqual(json.loads((saved/'session.json').read_text())['recorded_samples_per_channel'], board.n)
            return saved/'signal_processing'
        sessions=list(root.glob('**/raw_blocks'))
        self.assertEqual(len(sessions),1, (root/"status.json").read_text(encoding="utf-8"))
        report=recover_blocks(sessions[0])
        self.assertEqual(report['verified_samples'],board.n)
        self.assertTrue(report['complete'])
        with (sessions[0].parent/'raw_brainflow.tsv').open(encoding='utf-8') as f:
            raw=list(csv.reader(f,delimiter='\t'))
        self.assertEqual(len(raw)-1,board.n)
        return sessions[0]
    def test_signal_selection_files_tags_and_sample_alignment(self):
        from eeg_tools.workstation.dataset import DatasetRepository
        from eeg_tools.continuous_ssvep import HOP

        for raw, realtime in ((True, False), (False, True), (True, True)):
            with self.subTest(raw=raw, realtime=realtime), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                session = self.run_fake(root, raw=raw, realtime=realtime).parent
                meta = json.loads((session/'session.json').read_text(encoding='utf-8'))
                record, = DatasetRepository(root).list_records()
                tags = tuple(key for key, enabled in [('raw_eeg', raw), ('realtime_ssvep', realtime)] if enabled)
                self.assertEqual(record.signal_types, tags)
                self.assertEqual(meta['signal_types'], list(tags))
                self.assertEqual((session/'raw_brainflow.tsv').is_file(), raw)
                self.assertEqual((session/'raw_columns.tsv').is_file(), raw)
                self.assertEqual((session/'raw_blocks').exists(), raw)
                self.assertEqual((root/'live_waveform.tsv').exists(), raw)
                self.assertEqual(bool(list(session.rglob('*.npy'))), raw)
                self.assertEqual('raw_samples' in json.loads((root/'quality-live.json').read_text()), raw)
                path = session/'continuous_ssvep_predictions.tsv'
                self.assertEqual(path.is_file(), realtime)
                report = json.loads(next((session/'derived').rglob('report.json')).read_text(encoding='utf-8'))
                self.assertGreater(report['signal_quality_updates']['all'], 0)
                self.assertTrue(report['signal_quality_updates']['seal_evidence_match'])
                with (session/'manifest.csv').open(encoding='utf-8') as f:
                    names = {row['file_name'] for row in csv.DictReader(f)}
                self.assertEqual('raw_brainflow.tsv' in names, raw)
                self.assertEqual(path.name in names, realtime)
                n = record.recorded_samples_per_channel
                quality = json.loads((session/'quality.json').read_text())
                self.assertEqual(quality['samples_per_channel'], n)
                self.assertAlmostEqual(quality['channels']['CH1']['rms'], np.mean(np.sin(np.arange(n)/8)**2)**.5)
                self.assertAlmostEqual(quality['effective_rate_hz'], 250)
                with (session/'events.tsv').open(encoding='utf-8') as f:
                    events = [r for r in csv.DictReader(f, delimiter='\t') if r['sample_index']]
                self.assertTrue(events)
                for row in events:
                    self.assertAlmostEqual(float(row['sample_time_s']), 100 + int(row['sample_index'])/250)
                if realtime:
                    self.assertTrue(meta['continuous_ssvep']['complete'])
                    with path.open(encoding='utf-8-sig') as f:
                        ends = [int(r['end_sample']) for r in csv.DictReader(f, delimiter='\t')]
                    self.assertEqual(ends, list(range(HOP - 1, n, HOP)))
                    self.assertTrue(ends)
                    if raw:
                        with (session/'raw_brainflow.tsv').open(encoding='utf-8') as f:
                            samples = list(csv.DictReader(f, delimiter='\t'))
                        self.assertEqual([int(samples[i]['sample_index']) for i in ends], ends)

    def test_realtime_failure_records_error_and_preserves_raw(self):
        with tempfile.TemporaryDirectory() as tmp, patch('eeg_tools.live_ssvep.LiveSSVEP.consume', side_effect=OSError('fixture')):
            session = self.run_fake(Path(tmp), realtime=True, expected_exit=1).parent
            meta = json.loads((session/'session.json').read_text(encoding='utf-8'))
            self.assertFalse(meta['continuous_ssvep']['complete'])
            self.assertEqual(meta['status'], 'error')
            self.assertEqual(meta['signal_types'], ['raw_eeg'])

    def test_gpu_failure_before_acquisition(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), \
                patch('eeg_tools.ssvep_gpu.load_detector', side_effect=RuntimeError('fixture GPU unavailable')) as load, \
                patch.object(worker, '_prepare_cyton_board') as prepare:
            code = worker.main(['--participant','fixture','--session-name','test','--eye-side','left',
                                '--headless','--output-root',tmp,'--continuous-ssvep','--realtime-model-id','fixture'])
            self.assertEqual(code, 1)
            self.assertEqual(load.call_args.args[0], 'fixture')
            prepare.assert_not_called()
            meta = json.loads(next(Path(tmp).rglob('session.json')).read_text(encoding='utf-8'))
            self.assertEqual(meta['status'], 'error')
            self.assertEqual(meta['requested_realtime_model_id'], 'fixture')
            self.assertIsNone(meta['realtime_model'])

    def test_empty_selection_rejected_before_hardware(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()), patch.object(worker, '_prepare_cyton_board') as prepare:
            with self.assertRaises(SystemExit):
                worker.main(['--participant','fixture','--session-name','test','--eye-side','left',
                             '--headless','--output-root',tmp,'--no-raw-eeg'])
            prepare.assert_not_called()
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_historical_signal_tags_use_saved_rows_without_rewriting_metadata(self):
        from eeg_tools.workstation.dataset import DatasetRepository
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta = root/'session.json'
            meta.write_text('{"status":"aborted"}')
            raw = root/'raw_brainflow.tsv'
            pred = root/'continuous_ssvep_predictions.tsv'
            raw.write_text('sample_index\teeg_ch1\n')
            pred.write_text('end_sample\tstate\n49\tunavailable\n')
            self.assertEqual(DatasetRepository._read_record(meta).signal_types, ('realtime_ssvep',))
            with raw.open('a') as f: f.write('0\t1.0\n')
            self.assertEqual(DatasetRepository._read_record(meta).signal_types, ('raw_eeg','realtime_ssvep'))
            self.assertEqual(meta.read_text(), '{"status":"aborted"}')

    def test_main_entry_uses_incremental_raw_and_preview_same_indices(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);blocks=self.run_fake(root)
            with (root/'live_waveform.tsv').open(encoding='utf-8') as f:preview=list(csv.DictReader(f,delimiter='\t'))
            self.assertEqual([int(r['sample_index']) for r in preview],list(range(recover_blocks(blocks)['verified_samples'])))
            self.assertTrue((blocks.parent/'event_associations_v2.json').is_file())
    def test_quality_failure_preserves_raw_and_marks_assessment_unknown(self):
        with tempfile.TemporaryDirectory() as temp:
            blocks=self.run_fake(Path(temp),True)
            seal=json.loads((blocks/'seal.json').read_text(encoding='utf-8'))
            self.assertTrue(seal['consumer_errors']);self.assertEqual(seal['watermarks']['assessed'],0)
