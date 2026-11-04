"""Read-only, explicit offline testing runtime. No torch/CUDA at import/check-only.

Never opens hardware, changes acquisition config, or writes raw signals. The
learned core and its historical failed gate remain unmodified. Loading checks
SHA256, CPU weights_only and the original strict v8 checkpoint schema.
"""
from __future__ import annotations
import hashlib
import io
import json
import os
from pathlib import Path, PureWindowsPath
import stat
from datetime import datetime, timezone
import importlib
import numpy as np

ROOT = Path(__file__).absolute().parents[1]
CONFIG = 'configs/learned_denoiser_v9_ssvep.json'
PREREG = 'analysis_outputs/ssvep_v9_inference_preregistration_20261008.json'
LOCK = 'analysis_outputs/ssvep_v9_inference_code_lock_20261008.json'
REPORT = 'analysis_outputs/ssvep_v9_internal_diagnostic_20261008.json'
AUDIT = 'analysis_outputs/ssvep_v9_saved_audit_20261008.json'
MODEL = 'models/denoiser_v9_ssvep.pt'
MANIFEST = 'models/denoiser_v9_ssvep.manifest.json'
BASE_MODEL = 'models/denoiser_v8_ssvep.pt'
BASE_CONFIG = 'configs/learned_denoiser_v8_ssvep.json'
BASE_LOCK = 'analysis_outputs/ssvep_v8_training_lock_20261008.json'
BASE_SHA = '08170e1f9eb139233acdbf16dfb4f27e3405ecb81dcaf9ae4bb1e833c4ca92f8'
BASE_CONFIG_SHA = '484950220254e0b2c256d8cea2bf4f5d8f9084fe7aa4616f406d5e240553a16a'
BASE_LOCK_SHA = 'c229dbcfda94c9f01779238ef0823db5e0b33b938d3b1760bc1274467678fceb'
DOMAINS = ('Nakanishi2015', 'Wearable')
RATES = {'Nakanishi2015':256, 'Wearable':250}
EXPECTED_CONFIG = {
 'schema_version':1, 'status':'formal', 'model_version':'learned_denoise_v9_ssvep',
 'model_type':'frozen_v8_core_with_fixed_normalized_residual_trust_region',
 'input_channels':8, 'window_samples':256, 'hop_samples':256, 'sample_rates_hz':RATES,
 'change_budget_normalized_mse':0.00045, 'rounding_margin':1e-5,
 'protected_band_hz':[8.0,40.0], 'base_checkpoint_sha256':BASE_SHA,
 'training':'none_this_version; unchanged pretrained v8 weights',
 'normalization':'noisy_per_channel_mean_rms_eps1e-8; float32_forward_float64_residual_restore',
 'window_policy':'fixed_nonoverlap; incomplete_or_invalid_or_flat_window_preserved',
 'channel_order':'source indices only; anatomical correspondence unavailable in frozen manifests',
 'native_units':'unknown; output retains input numeric units without unit conversion',
 'runtime':{'enabled':False,'production_validated':False,'deployment_scope':'explicit_offline_test_only'},
 'clean_ground_truth_used':False,
}

class TestModelError(ValueError):
    pass

def require(ok, message):
    if not ok: raise TestModelError(message)

def now(): return datetime.now(timezone.utc).isoformat()

def sha(data): return hashlib.sha256(data).hexdigest()

def safe_local_path(value, root=ROOT):
    text = str(value)
    require(bool(text) and '\x00' not in text and '..' not in text.replace('\\','/').split('/'), 'unsafe/traversal path')
    w=PureWindowsPath(text)
    require(not any(p.endswith((' ','.')) for p in w.parts if p not in ('.','\\','/')), 'ambiguous Windows path refused')
    require(not w.drive or Path(text).is_absolute(), 'relative drive path refused')
    require(not text.startswith(('\\\\','//')) and not w.drive.startswith('\\\\'), 'UNC/device paths refused')
    require(not any(':' in p for p in w.parts[1:]) and (w.drive or ':' not in text), 'alternate stream refused')
    require(not any(p.rstrip(' .').split('.')[0].upper() in {'CON','NUL','PRN','AUX',*(f'COM{i}' for i in range(1,10)),*(f'LPT{i}' for i in range(1,10))} for p in w.parts), 'device path refused')
    path=Path(text)
    if not path.is_absolute(): path=Path(root)/path
    path=Path(os.path.abspath(path))
    for p in (*reversed(path.parents),path):
        try: s=p.lstat()
        except FileNotFoundError: continue
        require(not stat.S_ISLNK(s.st_mode) and not (getattr(s,'st_file_attributes',0)&0x400), 'symlink/reparse path refused')
        if p!=path: require(stat.S_ISDIR(s.st_mode), 'ancestor must be directory')
    return path

def read_bytes(path, *, limit=1024*1024*1024):
    path=safe_local_path(path)
    info=path.stat()
    require(stat.S_ISREG(info.st_mode) and 0<info.st_size<=limit, 'regular nonempty file size limit')
    data=path.read_bytes()
    require(len(data)==info.st_size, 'file changed while reading')
    return data

def _unique(pairs):
    out={}
    for k,v in pairs:
        require(k not in out,'duplicate JSON key')
        out[k]=v
    return out

def json_read(data):
    return json.loads(data.decode('utf-8'),object_pairs_hook=_unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(TestModelError('nonfinite JSON')))

def json_bytes(obj): return (json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')

def exclusive_write(path, data):
    path=safe_local_path(path)
    require(type(data) is bytes and bool(data), 'nonempty bytes required')
    # Exclusive creation: no replace/overwrite of any historical artifact.
    with path.open('xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())

def checked_output_dir(path, *, sources=(), root=ROOT):
    path=safe_local_path(path,root)
    analysis=safe_local_path(Path(root)/'analysis_outputs')
    require(path!=analysis and path.is_relative_to(analysis), 'output must be a NEW subdirectory of analysis_outputs')
    require(not os.path.lexists(path), 'output already exists; no-clobber')
    for source in sources:
        source=safe_local_path(source,root)
        require(source!=path and not source.is_relative_to(path), 'output contains protected input')
    require(path.parent.is_dir(), 'output parent missing')
    return path

def validate_config(cfg):
    # JSON representation gives strict scalar typing (including bool vs number).
    require(json.dumps(cfg,sort_keys=True,allow_nan=False)==json.dumps(EXPECTED_CONFIG,sort_keys=True), 'v9 fixed formal config mismatch')

def verify_files(files, root=ROOT):
    require(type(files) is dict and bool(files), 'frozen file bindings required')
    for rel,digest in files.items():
        path=safe_local_path(rel,root)
        require(path.is_relative_to(safe_local_path(root)), 'binding outside workspace')
        require(type(digest) is str and len(digest)==64 and sha(read_bytes(path))==digest, 'SHA256 mismatch: '+rel)

def locked_config(root=ROOT):
    lock=json_read(read_bytes(safe_local_path(LOCK,root)))
    require(lock['status']=='formal' and lock['stage']=='v9_preexecution_inference_lock', 'wrong code lock')
    verify_files(lock['files'],root)
    cfg=json_read(read_bytes(safe_local_path(CONFIG,root))); validate_config(cfg)
    return cfg,lock

def preflight(root=ROOT, *, packaged=True):
    cfg,lock=locked_config(root)
    manifest=None
    if packaged:
        manifest=json_read(read_bytes(safe_local_path(MANIFEST,root)))
        require(manifest['status']=='formal' and manifest['model_version']==cfg['model_version'], 'invalid package manifest')
        require(manifest['runtime_enabled'] is False and manifest['production_validated'] is False, 'unsafe runtime flags')
        verify_files(manifest['files'],root)
        report=json_read(read_bytes(safe_local_path(REPORT,root)))
        require(manifest['domain_internal_gates']=={d:report['dataset_gates'][d]['all_passed'] for d in DOMAINS}, 'manifest gate mismatch')
        audit=json_read(read_bytes(safe_local_path(AUDIT,root)))
        require(audit['audit_passed'] is True and audit['report_sha256']==manifest['files'][REPORT], 'saved audit mismatch')
    return cfg,lock,manifest

def load_models(root=ROOT, *, packaged=True, device='cpu'):
    cfg,lock,manifest=preflight(root,packaged=packaged)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    v8=importlib.import_module('eeg_tools.ssvep_v8_experiment')
    adapter=importlib.import_module('eeg_tools.ssvep_denoise_v9')
    torch=adapter.torch
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False; torch.use_deterministic_algorithms(True)
    if packaged:
        payload=read_bytes(safe_local_path(MODEL,root))
        ck=torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
        require(type(ck) is dict and set(ck)=={'schema_version','status','model_version','config','base_checkpoint_bytes','base_checkpoint_sha256','inference_lock_sha256'}, 'v9 checkpoint schema')
        require(ck['schema_version']==1 and ck['status']=='formal' and ck['model_version']==cfg['model_version'], 'v9 checkpoint metadata')
        validate_config(ck['config'])
        require(ck['inference_lock_sha256']==sha(read_bytes(safe_local_path(LOCK,root))), 'v9 lock binding mismatch')
        base=ck['base_checkpoint_bytes']
        require(type(base) is bytes and ck['base_checkpoint_sha256']==BASE_SHA and sha(base)==BASE_SHA, 'base checkpoint mismatch')
    else:
        base=read_bytes(safe_local_path(BASE_MODEL,root))
        require(sha(base)==BASE_SHA,'base checkpoint mismatch')
    base_cfg=json_read(read_bytes(safe_local_path(BASE_CONFIG,root)))
    require(sha(read_bytes(safe_local_path(BASE_CONFIG,root)))==BASE_CONFIG_SHA and sha(read_bytes(safe_local_path(BASE_LOCK,root)))==BASE_LOCK_SHA, 'v8 config/lock mismatch')
    _,models=v8.load_checkpoint_cpu(base,base_cfg,BASE_LOCK_SHA)
    require(device in ('cpu','cuda','cuda:0'), 'unsupported device')
    if device.startswith('cuda'): require(torch.cuda.is_available(), 'CUDA unavailable; choose --device cpu explicitly')
    models={d:adapter.SSVEPDenoiserV9(m).to(device=device,dtype=torch.float32).eval() for d,m in models.items()}
    return models, {'device':device,'torch_version':str(torch.__version__),'cuda_version':torch.version.cuda,
                    'device_name':torch.cuda.get_device_name(0) if device.startswith('cuda') else 'CPU',
                    'amp':False,'allow_tf32':False,'deterministic_algorithms':True}

def load_signal(data, suffix, signal_key='signal'):
    stream=io.BytesIO(data)
    if suffix=='.npy': arr=np.load(stream,allow_pickle=False)
    elif suffix=='.npz':
        with np.load(stream,allow_pickle=False) as z:
            require(signal_key in z.files,'NPZ signal key missing')
            arr=z[signal_key]
    elif suffix=='.csv': arr=np.loadtxt(stream,delimiter=',',ndmin=2)
    else: raise TestModelError('only NPY/NPZ/numeric headerless CSV accepted')
    require(isinstance(arr,np.ndarray) and arr.ndim==2 and arr.shape[1]==8 and arr.shape[0]>=256, 'signal must be [time,8], at least 256 samples')
    require(arr.dtype in (np.dtype('float32'),np.dtype('float64')), 'signal requires real float32/float64; no pickle/object/integer')
    require(arr.size<=64_000_000,'signal sample count limit')
    return arr

def process_signal(signal, valid_mask, model, *, sample_rate_hz, batch_size=64):
    require(sample_rate_hz==model.sample_rate_hz, 'native sample rate mismatch; no implicit resampling')
    require(isinstance(signal,np.ndarray) and signal.ndim==2 and signal.shape[1]==8 and signal.shape[0]>=256 and signal.dtype in (np.dtype('float32'),np.dtype('float64')), 'signal shape/dtype')
    require(isinstance(valid_mask,np.ndarray) and valid_mask.dtype==np.bool_ and valid_mask.shape==(len(signal),), 'valid_mask must be bool [time]')
    require(type(batch_size) is int and 1<=batch_size<=256,'batch size must be 1..256')
    from eeg_tools.ssvep_v6_contract import normalize_window, restore_window
    torch=importlib.import_module('torch')
    out=signal.astype(np.float64,copy=True); processed=np.zeros(len(signal),dtype=bool)
    rows=[]; batch=[]
    def flush():
        if not batch:return
        x=torch.from_numpy(np.stack([r[1] for r in batch])).to(next(model.parameters()).device)
        with torch.inference_mode(): pred,aux=model.forward_with_aux(x)
        y=pred.cpu().numpy(); gains=aux['gain'].cpu().numpy().reshape(-1)
        for i,(start,norm,rms) in enumerate(batch):
            raw=np.ascontiguousarray(signal[start:start+256].T,dtype=np.float64)
            restored=restore_window(raw,norm,y[i],rms)
            require(np.isfinite(restored).all(),'nonfinite restored prediction')
            change=float(np.mean(((restored-raw)/rms)**2))
            require(change<=0.00045*1.0001,'v9 normalized change budget violated')
            out[start:start+256]=restored.T; processed[start:start+256]=True
            rows.append({'start_sample':start,'end_sample_exclusive':start+256,'status':'processed',
                         'normalized_change':change,'trust_region_gain':float(gains[i])})
        batch.clear()
    for start in range(0,len(signal)-255,256):
        raw=np.ascontiguousarray(signal[start:start+256].T,dtype=np.float64)
        valid=valid_mask[start:start+256]
        reason=None
        if not valid.all():reason='invalid_mask_window_preserved'
        elif not np.isfinite(raw).all():reason='nonfinite_window_preserved'
        elif np.any(np.sqrt(np.mean((raw-raw.mean(axis=1,keepdims=True))**2,axis=1))<=1e-8):reason='flat_channel_window_preserved'
        if reason:
            rows.append({'start_sample':start,'end_sample_exclusive':start+256,'status':reason});continue
        norm,_,rms=normalize_window(raw); batch.append((start,norm,rms))
        if len(batch)>=batch_size:flush()
    flush(); rows.sort(key=lambda r:r['start_sample'])
    tail=len(signal)%256
    if tail:rows.append({'start_sample':len(signal)-tail,'end_sample_exclusive':len(signal),'status':'incomplete_tail_preserved'})
    # Block boundary differences are reported, not hidden by unvalidated OLA.
    boundaries=list(range(256,len(signal),256))
    jumps=[float(np.max(np.abs((out[b]-out[b-1])-(signal[b].astype(np.float64)-signal[b-1].astype(np.float64)))))
           for b in boundaries if np.isfinite(out[b-1:b+1]).all()]
    return out,processed,{'windows':rows,'processed_samples':int(processed.sum()),'preserved_samples':int((~processed).sum()),
                        'max_added_boundary_jump_input_units':max(jumps,default=0.0),
                        'boundary_and_full_record_spectral_preservation_validated':False,
                        'protected_band_claim':'per fixed window only; concatenated spectra NOT guaranteed',
                        'output_dtype':'float64','output_units':'same numeric units as input; no conversion'}
