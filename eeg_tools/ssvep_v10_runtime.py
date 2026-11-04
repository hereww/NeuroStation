"""Explicit offline V10 inference. No scientific imports before loading a model.

Independent model/manifest, not a replacement of frozen v8/v9. Source numeric
units and channel order are preserved. Runtime acceptance remains disabled.
"""
from __future__ import annotations
import importlib
import json
from pathlib import Path
from eeg_tools.ssvep_v10_data import ROOT, load_config, require, sha, canonical_sha, condition_line

MODEL = "models/denoiser_v10_personal.pt"
MANIFEST = "models/denoiser_v10_personal.manifest.json"


def load_model(root=ROOT, device="cuda:0"):
    require(device in ('cpu','cuda:0'), 'choose cpu or cuda:0 explicitly')
    root=Path(root);cfg=load_config(root)
    manifest=json.loads((root/MANIFEST).read_text(encoding='utf-8'))
    require(manifest.get('status')=='formal' and manifest.get('model_version')==cfg['model_version']
        and manifest.get('runtime_enabled') is False and manifest.get('production_validated') is False,
        'invalid/unvalidated model manifest')
    require(sha(root/MODEL)==manifest.get('checkpoint_sha256')
        and canonical_sha(cfg)==manifest.get('config_sha256'), 'model/config fingerprint mismatch')
    torch=importlib.import_module('torch')
    require(device=='cpu' or torch.cuda.is_available(),'CUDA not available; explicitly choose cpu')
    checkpoint=torch.load(root/MODEL,map_location='cpu',weights_only=True)
    require(checkpoint.get('model_version')==cfg['model_version'] and checkpoint.get('config')==cfg
        and checkpoint.get('config_sha256')==canonical_sha(cfg)
        and checkpoint.get('runtime_enabled') is False and checkpoint.get('production_validated') is False,
        'checkpoint metadata mismatch')
    require(checkpoint.get('training_lock_sha256')==manifest.get('training_lock_sha256'),'training identity mismatch')
    model=importlib.import_module('eeg_tools.ssvep_denoise_v10').build_model(cfg)
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    torch.set_num_threads(cfg['training']['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model=model.to(device=device,dtype=torch.float32).eval()
    return model, {'device':device,'device_name':torch.cuda.get_device_name(0) if device=='cuda:0' else 'CPU',
        'torch_version':str(torch.__version__),'model_version':cfg['model_version'],
        'checkpoint_sha256':manifest['checkpoint_sha256'],'selected_epoch':checkpoint['selected_epoch'],
        'production_validated':False,'runtime_enabled':False,'source_domain_validated':False}


def process_recording(signal, model, batch_size=32):
    import numpy as np
    import torch
    raw=np.asarray(signal)
    require(raw.ndim==2 and raw.shape[1]==8 and len(raw)>=512 and raw.dtype.kind=='f',
        'input must be floating [time>=512,8] native 250Hz')
    require(np.isfinite(raw).all(),'nonfinite input; no interpolation')
    require(type(batch_size) is int and 1<=batch_size<=256,'invalid batch size')
    require(getattr(model,'sample_rate_hz',None)==250,'native model sampling rate mismatch')
    # Full-record flat and railing checks must not silently repair physical channels.
    require(np.all(np.std(raw,axis=0)>1e-8),'flat input channel; no channel replacement')
    notch=condition_line(raw)
    starts=list(range(0,len(raw)-511,128))
    if starts[-1] != len(raw)-512:
        starts.append(len(raw)-512)
    window=np.hanning(514)[1:-1]  # strictly positive endpoint weights; exact full coverage
    accumulated=np.zeros(raw.shape,dtype=np.float64)
    weight=np.zeros(len(raw),dtype=np.float64)
    processed=np.zeros(len(raw),dtype=bool)
    device=next(model.parameters()).device
    max_protected=0.0;changes=[]
    for offset in range(0,len(starts),batch_size):
        batch_starts=starts[offset:offset+batch_size]
        segments=np.stack([notch[start:start+512].T for start in batch_starts])
        means=segments.mean(axis=-1,keepdims=True)
        rms=np.sqrt(np.mean((segments-means)**2,axis=-1,keepdims=True))
        require(np.isfinite(rms).all() and np.all(rms>1e-8), 'flat or invalid input window; no fabricated output')
        tensor=torch.as_tensor((segments-means)/rms,dtype=torch.float32,device=device)
        with torch.inference_mode():
            prediction, aux=model.forward_with_aux(tensor)
        pred=prediction.cpu().numpy().astype(np.float64)
        require(np.isfinite(pred).all(),'nonfinite model prediction')
        max_protected=max(max_protected,float(aux['protected_complex_energy_ratio'].max().cpu()))
        restored=pred*rms+means
        for i,start in enumerate(batch_starts):
            accumulated[start:start+512]+=restored[i].T*window[:,None]
            weight[start:start+512]+=window
            processed[start:start+512]=True
            changes.append(float(np.mean(((restored[i]-segments[i])/rms[i])**2)))
    require(np.all(weight>0),'incomplete overlap-add coverage')
    output=accumulated/weight[:,None]
    require(np.isfinite(output).all() and output.shape==raw.shape,'invalid restored output')
    receipt={'window_samples':512,'hop_samples':128,'window_count':len(starts),
        'processed_samples':int(processed.sum()),'preserved_samples':int((~processed).sum()),
        'max_window_protected_complex_energy_ratio':max_protected,
        'mean_window_normalized_change_from_notch':float(np.mean(changes)),
        'overlap_add':'positive Hann(514)[1:-1] normalized weights; final start appended for full coverage',
        'phase':'offline noncausal','unit_conversion_performed':False,'source_channel_order_preserved':True,
        'whole_record_protected_spectrum_guaranteed':False,'production_validated':False}
    return output, notch, receipt
