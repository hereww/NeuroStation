"""Synthetic, accelerated pipeline/UI benchmark; never opens EEG hardware."""
import argparse,json,os,sys,time
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase, QFont
from eeg_tools.streaming import BlockRecorder, AsyncConsumer
from eeg_tools.quality.recording import QualitySink
from eeg_tools.quality.replay import replay_quality
from apps.workstation_ui.quality_panel import QualityPanel


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=int,default=90);p.add_argument('--block-samples',type=int,default=13)
    a=p.parse_args()
    if not 10<=a.seconds<=7200 or not 1<=a.block_samples<=250:p.error('invalid synthetic duration/block size')
    a.output.mkdir(parents=True,exist_ok=False)
    app=QApplication.instance() or QApplication([])
    font=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts/msyh.ttc'
    if font.is_file():
        fid=QFontDatabase.addApplicationFont(str(font))
        families=QFontDatabase.applicationFontFamilies(fid)
        if families:app.setFont(QFont(families[0],10))
    panel=QualityPanel();panel.resize(1120,660);panel.show()
    channels=['Fp1','Fp2','C3','C4','P7','P8','O1','O2'];r=BlockRecorder(a.output/'raw_blocks',session_id='synthetic-benchmark-only',channel_names=channels,eeg_rows=list(range(1,9)),package_row=0,timestamp_row=9)
    sink=QualitySink(r.directory,list(range(1,9)),channels)
    consumer=AsyncConsumer(r,sink)
    display=[];processing=[];save=[];began=time.monotonic()
    for start in range(0,a.seconds*250,a.block_samples):
        n=min(a.block_samples,a.seconds*250-start);t=(start+np.arange(n))/250
        data=np.vstack(((start+np.arange(n))%256,*[30*np.sin(2*np.pi*10*t+c*.1)+3*np.sin(2*np.pi*50*t) for c in range(8)],100+t))
        begin=time.monotonic();r.append(data,consumer)
        while r.assessed<r.durable:
            app.processEvents()
            if time.monotonic()-begin>5:raise RuntimeError('benchmark quality deadline exceeded')
            time.sleep(.001)
        saved=time.monotonic()
        panel.update_quality(sink.latest);app.processEvents();panel.grab()
        display.append((time.monotonic()-begin)*1000);processing.append(sink.latest['compute_ms']);save.append((saved-begin)*1000)
    consumer.close();sink.close();r.close();panel.grab().save(str(a.output/'operator-quality-preview.png'));panel.close()
    elapsed=time.monotonic()-began
    replay=replay_quality(r.directory)
    report={'schema_version':2,'fixture_only':True,'hardware_opened':False,'stimulus_started':False,
            'simulated_stream_seconds':a.seconds,'block_samples':a.block_samples,'wall_elapsed_s':elapsed,
            'quality_execution':'bounded_async_worker','quality_queue':consumer.status,'frame_count':len(display),'offscreen_sample_to_paint_p95_ms':float(np.percentile(display,95)),
            'quality_compute_p95_ms':float(np.percentile(processing,95)),
            'record_and_quality_ready_p95_ms':float(np.percentile(save,95)),
            'first_window_accumulation_s':{'amplitude':2,'psd':8,'drift':10,'trend':30},
            'targets':{'offscreen_paint_ms':250,'quality_ready_ms':1000,'compute_ms':100},
            'replay':replay,'real_30min_load':'not_run','real_2h_soak':'not_run',
            'physical_display_latency':'not_measured','software_only':True}
    report['passed']=bool(report['offscreen_sample_to_paint_p95_ms']<=250 and report['record_and_quality_ready_p95_ms']<=1000 and report['quality_compute_p95_ms']<100 and replay['online_replay_match'])
    (a.output/'performance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2));return 0 if report['passed'] else 2
if __name__=='__main__':raise SystemExit(main())
