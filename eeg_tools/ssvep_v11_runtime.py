"""Separate v11 experiment: stronger identity training, unchanged frozen v10 core.

Standard-library-only module import; scientific imports are explicit worker work.
"""
from __future__ import annotations
import importlib
import json
from pathlib import Path
from eeg_tools.ssvep_v10_data import ROOT, sha, canonical_sha, require

CONFIG='configs/learned_denoiser_v11_personal.json'
LOCK='analysis_outputs/ssvep_v11_training_lock_20261013.json'
MODEL='models/denoiser_v11_personal.pt'
MANIFEST='models/denoiser_v11_personal.manifest.json'


def load_config(root=ROOT):
    cfg=json.loads((Path(root)/CONFIG).read_text(encoding='utf-8'))
    require(cfg['status']=='formal' and cfg['model_version']=='learned_denoise_v11_personal', 'invalid v11 config')
    require(cfg['sample_rate_hz']==250 and cfg['channel_count']==8 and cfg['window_samples']==512,
            'invalid v11 native input contract')
    require(cfg['training']['identity_loss_weight']==100.0 and cfg['training']['learning_rate']==0.0003
            and cfg['training']['seed']==20261014 and cfg['training']['epochs']==24,'v11 training plan changed')
    require(cfg['runtime']['enabled'] is False and cfg['runtime']['production_validated'] is False,
            'unvalidated v11 runtime cannot be enabled')
    return cfg


def preflight(root=ROOT):
    root=Path(root);cfg=load_config(root)
    lock=json.loads((root/LOCK).read_text(encoding='utf-8'))
    require(lock['status']=='formal' and lock['config_sha256']==canonical_sha(cfg)
        and lock['model_version']==cfg['model_version'],'v11 configuration lock mismatch')
    needed={CONFIG,'eeg_tools/ssvep_v11_runtime.py','scripts/train_ssvep_denoiser_v11.py',
        'eeg_tools/ssvep_denoise_v10.py','eeg_tools/ssvep_v10_data.py','scripts/train_ssvep_denoiser_v10.py'}
    for subject in range(1,7):
        name=f'{cfg["data"]["reference_directory"]}/subject_{subject:03d}.npz'
        needed.update((name,name+'.manifest.json'))
    require(needed.issubset(lock['files']),'v11 lock missing required code/data binding')
    require(not any('subject_007' in name or 'subject_008' in name for name in lock['files']),
            'evaluation data in training lock')
    for name,digest in lock['files'].items():
        path=root/name
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and path.is_file()
            and sha(path)==digest,'v11 locked file changed: '+name)
    return {'status':'ready','model_version':cfg['model_version'],'config_sha256':canonical_sha(cfg),
        'training_lock_sha256':sha(root/LOCK),'checked_files':len(lock['files']),
        'fit_subjects':[1,2,3,4],'selection_subjects':[5,6],'waveforms_loaded':False,
        'cuda_initialized':False,'runtime_enabled':False,'production_validated':False}


def build_model(config=None):
    cfg=load_config() if config is None else config
    # The shared numerical architecture is exactly the locked v10 core;
    # v11 changes the objective, not the signal-only forward signature.
    architecture={**cfg,'model_version':'learned_denoise_v10_personal',
        'model_type':'fixed_line_notch_then_learned_protected_spectral_shrinkage'}
    return importlib.import_module('eeg_tools.ssvep_denoise_v10').build_model(architecture)


def load_model(root=ROOT,device='cuda:0'):
    require(device in ('cpu','cuda:0'),'choose explicit CPU or CUDA device')
    root=Path(root);cfg=load_config(root);preflight(root)
    manifest=json.loads((root/MANIFEST).read_text(encoding='utf-8'))
    require(manifest['model_version']==cfg['model_version'] and manifest['status']=='formal'
        and manifest['runtime_enabled'] is False and manifest['production_validated'] is False,
        'v11 invalid manifest')
    require(sha(root/MODEL)==manifest['checkpoint_sha256'] and canonical_sha(cfg)==manifest['config_sha256'],
        'v11 checkpoint/config hash mismatch')
    torch=importlib.import_module('torch')
    require(device=='cpu' or torch.cuda.is_available(),'CUDA unavailable; choose CPU explicitly')
    cp=torch.load(root/MODEL,map_location='cpu',weights_only=True)
    require(cp['model_version']==cfg['model_version'] and cp['config']==cfg
        and cp['config_sha256']==canonical_sha(cfg) and cp['training_lock_sha256']==sha(root/LOCK)
        and cp['runtime_enabled'] is False and cp['production_validated'] is False,'v11 checkpoint metadata mismatch')
    model=build_model(cfg);model.load_state_dict(cp['state_dict'],strict=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=model.to(device=device,dtype=torch.float32).eval()
    return model,{'device':device,'device_name':torch.cuda.get_device_name(0) if device=='cuda:0' else 'CPU',
        'torch_version':str(torch.__version__),'model_version':cfg['model_version'],
        'checkpoint_sha256':manifest['checkpoint_sha256'],'selected_epoch':cp['selected_epoch'],
        'selection_gate_passed':cp['selection_gate_passed'],'runtime_enabled':False,'production_validated':False,
        'source_domain_validated':False}


def process_recording(signal,model,batch_size=32):
    return importlib.import_module('eeg_tools.ssvep_v10_runtime').process_recording(signal,model,batch_size)
