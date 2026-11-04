"""Create immutable offline examples from already materialized development subject5.

No D: drive or outer/heldout waveform reads. Recorded reference plus artificial
pollution is NOT a clean EEG ground-truth pair. Produces a new directory only.
"""
from pathlib import Path
import io
import sys
sys.path.insert(0,str(Path(__file__).absolute().parents[1]))
import numpy as np
from eeg_tools import ssvep_test_runtime as rt
from eeg_tools.ssvep_v9_delivery import audit_report

def npy_bytes(x):
    s=io.BytesIO();np.save(s,x,allow_pickle=False);return s.getvalue()

def main():
    cfg,lock,manifest=rt.preflight()
    output=rt.checked_output_dir('analysis_outputs/ssvep_v9_examples_20261008')
    # Guarded old reference loader explicitly restricts records to subject5.
    from eeg_tools.ssvep_denoise_v5 import load_reference_records,corrupt_window
    from eeg_tools.ssvep_v8_internal_eval import case_seed,STARTS,CONDITIONS
    base_cfg=rt.json_read(rt.read_bytes(rt.safe_local_path(rt.BASE_CONFIG)))
    report=rt.json_read(rt.read_bytes(rt.safe_local_path(rt.REPORT)));audit_report(report)
    records=load_reference_records(rt.ROOT,subjects=(5,))
    payloads={};meta=[]
    for record in records:
        d=record.dataset
        ref=np.ascontiguousarray(record.signal[0,STARTS[d]:STARTS[d]+256,:].T,dtype=np.float64)
        windows=[]
        for condition in CONDITIONS:
            c=dict(base_cfg['corruption']);c.update(identity_probability=0.,combine_probability=0.,families=[condition])
            noisy=ref.copy() if condition=='identity' else corrupt_window(ref.copy(),record.sample_rate_hz,{'corruption':c},case_seed(d,5,0,condition,base_cfg))[0]
            windows.append(noisy.T)
        windows.append(ref.T.copy())  # explicitly invalid entire sixth block
        flat=ref.T.copy();flat[:,0]=0.;windows.append(flat)
        windows.append(windows[3].copy())
        signal=np.concatenate([*windows,ref.T[:31]],axis=0)
        mask=np.ones(len(signal),dtype=bool);mask[5*256+100]=False
        payloads[d+'_input.npy']=npy_bytes(signal);payloads[d+'_valid_mask.npy']=npy_bytes(mask)
        meta.append({'domain':d,'sample_rate_hz':record.sample_rate_hz,'shape':list(signal.shape),
                     'recorded_reference_subject':5,'recorded_reference_trial':0,'window_conditions':[*CONDITIONS,'invalid_mask','flat_channel','high_frequency_burst','incomplete_tail'],
                     'expected_processed_samples':1536,'reference_is_clean_ground_truth':False,'channel_anatomy_verified':False,
                     'source_npz_sha256':record.npz_sha256})
    rt.preflight();output.mkdir()
    for name,bytes_ in payloads.items():rt.exclusive_write(output/name,bytes_)
    rt.exclusive_write(output/'manifest.json',rt.json_bytes({'status':'formal','created_at_utc':rt.now(),
        'purpose':'offline inference demonstration, not independent validation','D_drive_accessed':False,
        'holdout_waveforms_loaded':False,'examples':meta,'files':{k:rt.sha(v) for k,v in payloads.items()}}))
    print(output)
    return 0
if __name__=='__main__':raise SystemExit(main())
