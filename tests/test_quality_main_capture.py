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
    def run_fake(self,root,quality_failure=False):
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
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(worker,'_prepare_cyton_board',return_value=(board,'FIXTURE',[])))
            stack.enter_context(patch.object(worker,'_wait_phase',side_effect=wait))
            if quality_failure:stack.enter_context(patch('eeg_tools.quality.engine.QualityEngine.process',side_effect=ValueError('fixture_quality_failure')))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            code=worker.main(args)
        sessions=list(root.glob('**/raw_blocks'))
        self.assertEqual(len(sessions),1, (root/"status.json").read_text(encoding="utf-8"))
        report=recover_blocks(sessions[0])
        self.assertEqual(report['verified_samples'],board.n)
        self.assertTrue(report['complete'])
        with (sessions[0].parent/'raw_brainflow.tsv').open(encoding='utf-8') as f:
            raw=list(csv.reader(f,delimiter='\t'))
        self.assertEqual(len(raw)-1,board.n)
        self.assertEqual(code,0)
        return sessions[0]
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
