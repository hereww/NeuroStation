"""NeuroStation v9 fixed-window offline test; raw data and hardware untouched."""
from __future__ import annotations
import argparse
import io
import json
import sys
from pathlib import Path
import time
import zipfile
sys.path.insert(0,str(Path(__file__).absolute().parents[1]))
import numpy as np
from eeg_tools import ssvep_test_runtime as rt

def positive_int(text):
    value=int(text)
    if not 1<=value<=256:raise argparse.ArgumentTypeError('must be 1..256')
    return value

def npy_bytes(arr):
    f=io.BytesIO();np.save(f,arr,allow_pickle=False);return f.getvalue()

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--input',type=Path);p.add_argument('--signal-key',default='signal')
    p.add_argument('--domain',choices=rt.DOMAINS);p.add_argument('--sample-rate-hz',type=int)
    p.add_argument('--channel-order',help='eight unique source channel labels in input order (not guessed electrode locations)')
    m=p.add_mutually_exclusive_group();m.add_argument('--valid-mask',type=Path);m.add_argument('--all-finite-valid',action='store_true')
    p.add_argument('--acknowledge-offline-test',action='store_true')
    p.add_argument('--acknowledge-unverified-channel-mapping',action='store_true')
    p.add_argument('--device',choices=['cpu','cuda','cuda:0'],default='cpu')
    p.add_argument('--batch-size',type=positive_int,default=64)
    p.add_argument('--output',type=Path,help='NEW directory underneath analysis_outputs')
    a=p.parse_args(argv);start=time.perf_counter()
    try:
        cfg,lock,manifest=rt.preflight()
        report={'schema_version':1,'status':'formal','model_version':cfg['model_version'],'created_at_utc':rt.now(),
                'check_only':a.check_only,'inference_executed':False,'hardware_opened':False,'runtime_enabled':False,
                'production_validated':False,'weights_retrained_this_run':False,'requested_device':a.device,
                'domain_internal_gates':manifest['domain_internal_gates'],'independent_validation':False}
        if a.check_only and a.input is None:
            rt.require(a.output is None and a.valid_mask is None and not a.all_finite_valid,'source options require input')
            print(json.dumps({**report,'readiness':'ready','checkpoint_deserialized':False,'cuda_initialized':False},ensure_ascii=False,indent=2));return 0
        rt.require(a.input is not None and a.domain is not None,'--input and --domain required')
        rt.require(a.sample_rate_hz==rt.RATES[a.domain], 'native sample rate mismatch; no automatic resampling')
        labels=[s.strip() for s in (a.channel_order or '').split(',')]
        rt.require(len(labels)==8 and all(labels) and len(set(labels))==8,'--channel-order requires eight unique nonempty labels')
        rt.require(a.all_finite_valid or a.valid_mask is not None,'explicit --valid-mask or --all-finite-valid required')
        source=rt.safe_local_path(a.input)
        rt.require(source.is_file() and source.suffix.lower() in ('.npy','.npz','.csv'),'supported source file required')
        sources=[source];mask_path=rt.safe_local_path(a.valid_mask) if a.valid_mask else None
        if mask_path:
            rt.require(mask_path.is_file() and mask_path.suffix.lower()=='.npy','mask must be NPY');sources.append(mask_path)
        output=rt.checked_output_dir(a.output,sources=sources) if a.output else None
        if a.check_only:
            print(json.dumps({**report,'readiness':'ready','input_values_loaded':False,'checkpoint_deserialized':False,'cuda_initialized':False},ensure_ascii=False,indent=2));return 0
        rt.require(a.acknowledge_offline_test and a.acknowledge_unverified_channel_mapping,
                   'explicit offline-test AND unverified-channel-mapping acknowledgements required')
        rt.require(output is not None,'new --output required')
        payload=rt.read_bytes(source);input_sha=rt.sha(payload)
        if source.suffix.lower()=='.npz':
            with zipfile.ZipFile(io.BytesIO(payload)) as z:
                rt.require(sum(i.file_size for i in z.infolist())<=512*1024*1024,'NPZ expanded size limit')
        signal=rt.load_signal(payload,source.suffix.lower(),a.signal_key)
        mask_bytes=rt.read_bytes(mask_path) if mask_path else None
        valid=np.load(io.BytesIO(mask_bytes),allow_pickle=False) if mask_bytes else np.isfinite(signal).all(axis=1)
        rt.require(isinstance(valid,np.ndarray) and valid.dtype==np.bool_ and valid.shape==(len(signal),),'valid_mask must be bool [time]')
        models,gpu=rt.load_models(device=a.device)
        out,processed,stats=rt.process_signal(signal,valid,models[a.domain],sample_rate_hz=a.sample_rate_hz,batch_size=a.batch_size)
        rt.require(rt.sha(rt.read_bytes(source))==input_sha,'input file changed during inference')
        if mask_path:rt.require(rt.sha(rt.read_bytes(mask_path))==rt.sha(mask_bytes),'mask changed during inference')
        rt.preflight()  # exact code/model/evidence binding immediately before publication
        output=rt.checked_output_dir(output,sources=sources)
        result_payloads={'denoised.npy':npy_bytes(out),'processed_mask.npy':npy_bytes(processed)}
        report.update(inference_executed=True,input_path=str(source),input_sha256=input_sha,
            valid_mask_path=str(mask_path) if mask_path else None,valid_mask_sha256=rt.sha(mask_bytes) if mask_bytes else None,
            valid_policy='explicit mask' if mask_path else 'user assertion: all finite rows eligible (not quality acceptance)',
            input_shape=list(signal.shape),input_dtype=str(signal.dtype),sample_rate_hz=a.sample_rate_hz,domain=a.domain,
            channel_labels=labels,anatomical_channel_equivalence_verified=False,offline_testing_acknowledged=True,
            model_sha256=manifest['files'][rt.MODEL],gpu=gpu,processing=stats,
            output_sha256={k:rt.sha(v) for k,v in result_payloads.items()},elapsed_seconds=time.perf_counter()-start)
        output.mkdir()  # exclusive, race-safe; leave partial bundle on interruption
        for name,data in result_payloads.items():rt.exclusive_write(output/name,data)
        rt.exclusive_write(output/'report.json',rt.json_bytes(report))  # complete marker LAST
        print(json.dumps({'status':'complete','output':str(output),'processed_samples':stats['processed_samples'],
                          'preserved_samples':stats['preserved_samples'],'production_validated':False},ensure_ascii=False,indent=2))
        return 0
    except (ValueError,OSError,RuntimeError,TypeError,KeyError,zipfile.BadZipFile) as exc:
        print(json.dumps({'status':'error','error':str(exc),'inference_success':False},ensure_ascii=False),file=sys.stderr);return 1

if __name__=='__main__':raise SystemExit(main())
