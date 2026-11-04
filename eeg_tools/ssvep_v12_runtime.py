"""V12 shared noisy-stat normalization and explicit offline inference.

Training, selection, window evaluation and record inference share these functions.
No scientific packages are imported by the GUI-facing metadata APIs.
"""
from __future__ import annotations
import importlib
import json
import os
from pathlib import Path
from eeg_tools.ssvep_v10_data import ROOT,sha,canonical_sha,require,condition_line
CONFIG='configs/learned_denoiser_v12_personal.json'
LOCK='analysis_outputs/ssvep_v12_training_lock_20261013.json'
MODEL='models/denoiser_v12_personal.pt'
MANIFEST='models/denoiser_v12_personal.manifest.json'
NORMALIZATION='noisy_float64_channel_mean_rms_float32_forward_float64_residual_restore'


def load_config(root=ROOT):
    cfg=json.loads((Path(root)/CONFIG).read_text(encoding='utf-8'))
    require(cfg['status']=='formal' and cfg['model_version']=='learned_denoise_v12_personal','invalid v12 config')
    require((cfg['sample_rate_hz'],cfg['channel_count'],cfg['window_samples'],cfg['hop_samples'])==(250,8,512,128),'invalid v12 native input contract')
    require(cfg['training']['seed']==20261015 and cfg['training']['epochs']==24
            and cfg['training']['identity_loss_weight']==100.0 and cfg['training']['learning_rate']==0.0003,'v12 training plan changed')
    require(cfg['normalization']['input_statistics_only'] is True and cfg['normalization']['rms_floor']==1e-8,'normalization contract changed')
    require(cfg['runtime']['enabled'] is False and cfg['runtime']['production_validated'] is False,'unvalidated v12 cannot be enabled')
    return cfg


def preflight(root=ROOT):
    root=Path(root);cfg=load_config(root)
    lock=json.loads((root/LOCK).read_text(encoding='utf-8'))
    require(lock['status']=='formal' and lock['model_version']==cfg['model_version']
            and lock['config_sha256']==canonical_sha(cfg),'v12 configuration lock mismatch')
    required={CONFIG,'eeg_tools/ssvep_v12_runtime.py','scripts/train_ssvep_denoiser_v12.py',
        'eeg_tools/ssvep_denoise_v10.py','eeg_tools/ssvep_v10_data.py','scripts/train_ssvep_denoiser_v10.py',
        'analysis_outputs/ssvep_v10_training_lock_20261013.json'}
    for subject in range(1,7):
        name=f'{cfg["data"]["reference_directory"]}/subject_{subject:03d}.npz'
        required.update((name,name+'.manifest.json'))
    require(required.issubset(lock['files']),'v12 required bindings missing')
    require(not any('subject_007' in name or 'subject_008' in name for name in lock['files']),'evaluation data included in training')
    for name,digest in lock['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and sha(root/name)==digest,
                'v12 locked source changed: '+name)
    return {'status':'ready','model_version':cfg['model_version'],'config_sha256':canonical_sha(cfg),
        'training_lock_sha256':sha(root/LOCK),'checked_files':len(lock['files']),
        'fit_subjects':[1,2,3,4],'selection_subjects':[5,6],'waveforms_loaded':False,
        'cuda_initialized':False,'runtime_enabled':False,'production_validated':False,
        'normalization_contract':NORMALIZATION}


def build_model(config=None):
    cfg=load_config() if config is None else config
    architecture={**cfg,'model_version':'learned_denoise_v10_personal',
        'model_type':'fixed_line_notch_then_learned_protected_spectral_shrinkage'}
    return importlib.import_module('eeg_tools.ssvep_denoise_v10').build_model(architecture)


def _window_array(value,name):
    import numpy as np
    array=np.asarray(value)
    require(array.dtype.kind=='f' and array.ndim==3 and array.shape[0]>0 and array.shape[1:]==(8,512),
            name+' must be floating [B,8,512]')
    require(np.isfinite(array).all(),name+' contains nonfinite samples')
    return array.astype(np.float64,copy=False)


def normalize_noisy(noisy):
    import numpy as np
    value=_window_array(noisy,'noisy')
    means=value.mean(axis=-1,keepdims=True)
    rms=np.sqrt(np.mean((value-means)**2,axis=-1,keepdims=True))
    require(np.isfinite(rms).all() and np.all(rms>1e-8),'flat/nonfinite noisy window; no channel replacement')
    normalized=np.asarray((value-means)/rms,dtype=np.float32)
    require(np.isfinite(normalized).all(),'normalization produced nonfinite samples')
    return normalized,means,rms


def _stats(value,means,rms):
    import numpy as np
    mean=np.asarray(means,dtype=np.float64);scale=np.asarray(rms,dtype=np.float64)
    require(mean.shape==scale.shape==(len(value),8,1) and np.isfinite(mean).all()
            and np.isfinite(scale).all() and np.all(scale>1e-8),'invalid shared noisy statistics')
    return mean,scale


def target_in_noisy_coordinates(reference,means,rms):
    import numpy as np
    value=_window_array(reference,'reference');mean,scale=_stats(value,means,rms)
    normalized=np.asarray((value-mean)/scale,dtype=np.float32)
    require(np.isfinite(normalized).all(),'nonfinite target coordinates')
    return normalized


def restore_residual(noisy,normalized,prediction,rms):
    import numpy as np
    value=_window_array(noisy,'noisy');norm=_window_array(normalized,'normalized')
    pred=_window_array(prediction,'prediction');scale=np.asarray(rms,dtype=np.float64)
    require(value.shape==norm.shape==pred.shape and scale.shape==(len(value),8,1)
            and np.isfinite(scale).all() and np.all(scale>1e-8),'invalid residual restore layout/statistics')
    output=value+(pred-norm)*scale
    require(np.isfinite(output).all(),'nonfinite restored output')
    return output


def process_windows(noisy,model,batch_size=32):
    import numpy as np
    import torch
    values=_window_array(noisy,'noisy')
    require(type(batch_size) is int and 1<=batch_size<=256,'invalid batch size')
    require(getattr(model,'sample_rate_hz',None)==250,'native model rate mismatch')
    device=next(model.parameters()).device;outputs=[];ratios=[];defined=[]
    for start in range(0,len(values),batch_size):
        segment=values[start:start+batch_size]
        normalized,means,rms=normalize_noisy(segment)
        tensor=torch.as_tensor(normalized,device=device,dtype=torch.float32)
        with torch.inference_mode():prediction,aux=model.forward_with_aux(tensor)
        pred=prediction.detach().cpu().numpy()
        outputs.append(restore_residual(segment,normalized,pred,rms))
        ratios.append(aux['protected_complex_energy_ratio'].detach().cpu().numpy())
        defined.append(aux['protected_energy_defined'].detach().cpu().numpy())
    return np.concatenate(outputs),{'protected_complex_energy_ratio':np.concatenate(ratios),
        'protected_energy_defined':np.concatenate(defined),'normalization_contract':NORMALIZATION}


def process_recording(signal,model,batch_size=32):
    import numpy as np
    raw=np.asarray(signal)
    require(raw.dtype.kind=='f' and raw.ndim==2 and raw.shape[1]==8 and len(raw)>=512
            and np.isfinite(raw).all(),'finite floating native 250Hz [T>=512,8] required')
    require(np.all(np.std(raw,axis=0)>1e-8),'flat channel; no channel replacement')
    notch=condition_line(raw)
    starts=list(range(0,len(raw)-511,128))
    if starts[-1]!=len(raw)-512:starts.append(len(raw)-512)
    hann=np.hanning(514)[1:-1];total_delta=np.zeros_like(notch);weights=np.zeros(len(raw))
    maximum=0.0;defined_count=0;window_changes=[]
    for offset in range(0,len(starts),batch_size):
        selected=starts[offset:offset+batch_size]
        original=np.stack([notch[start:start+512].T for start in selected])
        restored,aux=process_windows(original,model,batch_size=batch_size)
        mask=aux['protected_energy_defined'];defined_count+=int(mask.sum())
        if mask.any():maximum=max(maximum,float(aux['protected_complex_energy_ratio'][mask].max()))
        _,_,rms=normalize_noisy(original)
        for index,start in enumerate(selected):
            delta=(restored[index]-original[index]).T
            total_delta[start:start+512]+=delta*hann[:,None]
            weights[start:start+512]+=hann
            window_changes.append(float(np.mean(((restored[index]-original[index])/rms[index])**2)))
    require(np.all(weights>0),'incomplete overlap-add coverage')
    output=notch+total_delta/weights[:,None]
    require(output.shape==raw.shape and np.isfinite(output).all(),'invalid full-record output')
    return output,notch,{'window_samples':512,'hop_samples':128,'window_count':len(starts),
        'processed_samples':len(raw),'preserved_samples':0,'max_window_protected_complex_energy_ratio':maximum,
        'protected_defined_window_count':defined_count,'normalization_contract':NORMALIZATION,
        'mean_window_normalized_change_from_notch':float(np.mean(window_changes)),
        'overlap_add':'positive Hann weighted residual; final window appended; exact identity retains notch',
        'whole_record_protected_spectrum_guaranteed':False,'unit_conversion_performed':False,
        'source_channel_order_preserved':True,'phase':'offline noncausal','production_validated':False}


def load_model(root=ROOT,device='cuda:0'):
    require(device in ('cpu','cuda:0'),'choose explicit cpu or cuda:0')
    root=Path(root);cfg=load_config(root);state=preflight(root)
    manifest=json.loads((root/MANIFEST).read_text(encoding='utf-8'))
    require(manifest['status']=='formal' and manifest['model_version']==cfg['model_version']
        and manifest['config_sha256']==canonical_sha(cfg) and manifest['checkpoint_sha256']==sha(root/MODEL)
        and manifest['runtime_enabled'] is False and manifest['production_validated'] is False,'invalid v12 manifest/fingerprint')
    require(manifest['training_lock_sha256']==state['training_lock_sha256'],'v12 manifest lock mismatch')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch=importlib.import_module('torch')
    require(device=='cpu' or torch.cuda.is_available(),'CUDA unavailable; choose cpu explicitly')
    cp=torch.load(root/MODEL,map_location='cpu',weights_only=True)
    require(cp['schema_version']==1 and cp['model_version']==cfg['model_version'] and cp['config']==cfg
        and cp['config_sha256']==canonical_sha(cfg) and cp['training_lock_sha256']==state['training_lock_sha256']
        and cp['runtime_enabled'] is False and cp['production_validated'] is False,'v12 checkpoint metadata mismatch')
    model=build_model(cfg);model.load_state_dict(cp['state_dict'],strict=True)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=model.to(device=device,dtype=torch.float32).eval()
    return model,{'device':device,'device_name':torch.cuda.get_device_name(0) if device=='cuda:0' else 'CPU',
        'torch_version':str(torch.__version__),'model_version':cfg['model_version'],
        'checkpoint_sha256':manifest['checkpoint_sha256'],'selected_epoch':cp['selected_epoch'],
        'selection_gate_passed':cp['selection_gate_passed'],'normalization_contract':NORMALIZATION,
        'runtime_enabled':False,'production_validated':False,'source_domain_validated':False}
