"""Fixed v13 post-training research assessment; no training or epoch search.

Default loading rejects failed selection. Explicit diagnostics assess only the
fully bound failed final state, and always fail the overall research gate.
Import and --help use only the Python standard library.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PLAN = 'analysis_outputs/ssvep_v13_experiment_plan_20261013.json'
PRIOR_SYNTHETIC = 'analysis_outputs/v12_evaluation_20261013/synthetic.json'
PUBLIC_SEED = 2026101603
PUBLIC_CONDITIONS = ('identity', 'drift', 'high_burst', 'gaussian', 'combined')
SYNTHETIC_CONDITIONS = ('identity', 'line50', 'drift.3', 'gaussian', 'combined')
FREQUENCIES = (10, 12, 15, 20)
TRACKS = ('raw', 'notch', 'traditional', 'v13')
REGIONS = {'full': (0, 4096), 'interior': (512, 3584)}
DIAGNOSTIC_ROLE = 'diagnostic_last_epoch_no_eligible_checkpoint'
FALSE_FLAGS = dict(runtime_enabled=False, production_validated=False,
                   source_domain_validated=False, independent_personal_validation=False,
                   clean_ground_truth_available=False, presented_matches_are_gaze_accuracy=False)


def require(condition, message):
    if not condition:
        raise ValueError('v13 evaluation: ' + message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: ' + key)
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError('nonfinite JSON constant: ' + value)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=_unique,
                      parse_constant=_invalid_constant)


def write_json(path, value, *, atomic=False):
    path = Path(path)
    require(not os.path.lexists(path), 'refusing existing artifact: ' + str(path))
    temporary = path.with_name(path.name + '.pending') if atomic else path
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    if atomic:
        require(not os.path.lexists(path), 'completion marker already exists')
        os.replace(temporary, path)


def validate_paths(output, personal=(), *, root=ROOT):
    from eeg_tools.workstation import v9_offline as baseline
    output = baseline._path(output, root)
    analysis = baseline._path(Path(root) / 'analysis_outputs')
    require(output != analysis and output.is_relative_to(analysis),
            'output must be a new directory under analysis_outputs')
    require(not os.path.lexists(output), 'output already exists; no overwrite')
    sources = []
    for value in personal:
        source = baseline._path(value, root)
        require(source.is_dir(), 'personal source missing')
        require(not source.is_relative_to(output) and not output.is_relative_to(source), 'source/output overlap')
        require(all(not source.is_relative_to(other) and not other.is_relative_to(source)
                    for other in sources), 'duplicate or overlapping personal sources')
        require(all((source / name).is_file() for name in baseline.NATIVE_FILES), 'incomplete native source')
        sources.append(source)
    return output, sources


def load_evaluation_plan(root=ROOT):
    from eeg_tools.ssvep_v13_runtime import safe_path
    plan = read_json(Path(root) / PLAN)
    require(plan['status'] == 'formal' and plan['runtime_enabled'] is False
            and plan['production_validated'] is False
            and plan['research_checks_authorize_default_deployment'] is False, 'invalid frozen research plan')
    for name, expected in plan['prior_evidence'].items():
        require(sha(safe_path(root, name)) == expected, 'prior evidence changed: ' + name)
    require(isinstance(plan['fixed_research_checks'], dict), 'missing fixed research checks')
    return plan


def _matches_common(artifact, common):
    return all(artifact.get(key) == value and
               (type(artifact.get(key)) is type(value) if value is None or isinstance(value, bool) else True)
               for key, value in common.items())


def inference_settings():
    return {'deterministic_algorithms': True, 'cublas_workspace_config': ':4096:8',
            'allow_tf32': False, 'amp': False, 'cpu_threads': 4,
            'cudnn_deterministic': True, 'cudnn_benchmark': False}


def configure_inference():
    # Set before importing Torch or touching CUDA, including model.to(cuda).
    previous = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    loaded = sys.modules.get('torch')
    require(loaded is None or not loaded.cuda.is_initialized() or previous == ':4096:8',
            'CUDA already initialized without the frozen CUBLAS workspace setting')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    require(not torch.is_autocast_enabled('cuda') and not torch.is_autocast_enabled('cpu'),
            'AMP context is forbidden for fixed evaluation')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_evaluation_model(root=ROOT, device='cpu', *, allow_failed_selection_diagnostic=False):
    """No fallback on rejection. Diagnostic loading is an explicit separate path."""
    from eeg_tools import ssvep_v13_runtime as api
    require(device in ('cpu', 'cuda:0'), 'explicit cpu or cuda:0 required')
    state, cfg = api.preflight(root), api.load_config(root)
    if not allow_failed_selection_diagnostic:
        # The runtime loader must reject ineligible artifacts before any inference.
        manifest = read_json(api.safe_path(root, api.MANIFEST))
        if manifest.get('selection_gate_passed') is True:
            configure_inference()
        return api.load_model(root, device)
    # preflight checks frozen sources, fit/selection data, lock and prior evidence.
    # The only exception to the eligible loader is the explicitly requested role.
    paths = {name: api.safe_path(root, getattr(api, name))
             for name in ('MODEL', 'MANIFEST', 'REPORT', 'CONFIG', 'LOCK', 'PLAN')}
    before = {name: sha(path) for name, path in paths.items()}
    manifest, report = read_json(paths['MANIFEST']), read_json(paths['REPORT'])
    common = dict(schema_version=1, model_version=cfg['model_version'],
                  config_sha256=api.canonical_sha(cfg), training_lock_sha256=before['LOCK'],
                  research_plan_sha256=before['PLAN'], selected_epoch=None,
                  selection_gate_passed=False, runtime_enabled=False, production_validated=False)
    require(state['config_sha256'] == common['config_sha256']
            and state['training_lock_sha256'] == common['training_lock_sha256'], 'diagnostic preflight/config mismatch')
    for artifact in (manifest, report):
        require(_matches_common(artifact, common) and 'selected_epoch' in artifact, 'diagnostic artifact binding/selection mismatch')
        require(artifact.get('checkpoint_role') == DIAGNOSTIC_ROLE
                and artifact.get('normalization_contract') == api.NORMALIZATION, 'diagnostic role/normalization mismatch')
    require(manifest.get('status') == 'formal' and report.get('status') == 'training_completed',
            'incomplete diagnostic training artifacts')
    require(report.get('epochs') == cfg['training']['epochs'] == 16
            and type(report.get('diagnostic_epoch')) is int and report['diagnostic_epoch'] == 16
            and report.get('selected_metrics') is None, 'diagnostic must be final epoch16, unselected')
    require(manifest.get('checkpoint_sha256') == before['MODEL']
            and manifest.get('training_report_sha256') == before['REPORT'], 'diagnostic checkpoint/report SHA mismatch')
    require(len(report.get('history', [])) == 16
            and [row.get('epoch') for row in report['history']] == list(range(1, 17)), 'incomplete diagnostic history')
    require(all(row['selection']['selection_gate_passed'] is False for row in report['history']),
            'diagnostic history contains an eligible epoch')
    configure_inference()
    import torch
    require(device == 'cpu' or torch.cuda.is_available(), 'CUDA unavailable')
    checkpoint = torch.load(paths['MODEL'], map_location='cpu', weights_only=True)
    require(isinstance(checkpoint, dict) and checkpoint.get('config') == cfg, 'diagnostic checkpoint config mismatch')
    require(_matches_common(checkpoint, common) and 'selected_epoch' in checkpoint, 'diagnostic checkpoint binding/selection mismatch')
    require(checkpoint.get('status') == 'formal', 'diagnostic checkpoint status mismatch')
    # Frozen checkpoint carries common bindings. Role/final epoch are in its
    # hash-bound manifest/report, so no missing checkpoint fields are invented.
    model = api.build_model(cfg)
    require(isinstance(checkpoint.get('state_dict'), dict)
            and all(isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all())
                    for value in checkpoint['state_dict'].values()), 'invalid diagnostic weights')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    count = sum(p.numel() for p in model.parameters())
    require(count == manifest.get('parameter_count') == report.get('parameter_count'), 'diagnostic parameter count mismatch')
    require({name: sha(path) for name, path in paths.items()} == before
            and api.preflight(root) == state, 'diagnostic inputs changed while loading')
    metadata = {**manifest, 'diagnostic_epoch': 16, 'selected_epoch': None,
                'selection_gate_passed': False, 'checkpoint_role': DIAGNOSTIC_ROLE,
                'allow_failed_selection_diagnostic': True,
                'load_mode': 'explicit_failed_selection_final_state_diagnostic', **FALSE_FLAGS}
    return model.to(device).eval(), metadata


def checked_array(value, shape, name):
    import numpy as np
    array = np.asarray(value)
    require(array.dtype.kind == 'f' and array.shape == shape and bool(np.isfinite(array).all()),
            'invalid floating array: ' + name)
    return array.astype(np.float64, copy=False)


def fixed_notch(signal):
    """Independent 50Hz Q30 zero-phase stage; [..,T,8], no resampling."""
    import numpy as np
    from scipy.signal import filtfilt, iirnotch
    value = np.asarray(signal, dtype=np.float64)
    require(value.ndim in (2, 3) and value.shape[-1] == 8
            and value.shape[-2] >= 512 and bool(np.isfinite(value).all()), 'invalid notch input')
    means = value.mean(axis=-2, keepdims=True)
    b, a = iirnotch(50, 30, fs=250)
    return filtfilt(b, a, value - means, axis=-2) + means


def window_metrics(reference, raw, notch, output):
    import numpy as np
    reference = np.asarray(reference, dtype=np.float64)
    require(reference.ndim == 3 and len(reference) > 0 and reference.shape[1:] == (8, 512), 'invalid public reference')
    values = {name: checked_array(value, reference.shape, name)
              for name, value in (('raw', raw), ('notch', notch), ('model', output))}
    rms = np.std(values['notch'], axis=-1, keepdims=True, dtype=np.float64)
    require(bool(np.all(rms > 1e-8)), 'undefined public normalization')
    errors = {name: np.mean((value-reference)**2, axis=(1, 2)) for name, value in values.items()}
    total = np.mean(((values['model']-reference)/rms)**2, axis=(1, 2))
    learned = np.mean(((values['model']-values['notch'])/rms)**2, axis=(1, 2))
    rows = [{**{name+'_mse': float(err[i]) for name, err in errors.items()},
             'output_reference_normalized_change': float(total[i]),
             'learned_only_normalized_change': float(learned[i])} for i in range(len(reference))]
    means = {name+'_mse': float(err.mean()) for name, err in errors.items()}
    return {**means, 'window_count': len(reference), 'windows': rows,
            'fixed_notch_mse_reduction': means['raw_mse']-means['notch_mse'],
            'learned_only_mse_reduction': means['notch_mse']-means['model_mse'],
            'output_reference_normalized_change': float(total.mean()),
            'max_window_output_reference_normalized_change': float(total.max()),
            'learned_only_normalized_change': float(learned.mean()),
            'normalization_denominator': 'float64 per-channel RMS of notch input, squared after division'}


def evaluate_public(model, cfg, *, root=ROOT):
    import numpy as np
    from eeg_tools import ssvep_v10_data as data, ssvep_v13_runtime as api
    from scripts.train_ssvep_denoiser_v13 import corrupt
    reference, identities = data.load_reference_windows([7, 8], root=root)
    require(isinstance(reference, np.ndarray) and reference.dtype == np.float32
            and reference.ndim == 3 and reference.shape[1:] == (8, 512)
            and len(reference) > 0 and bool(np.isfinite(reference).all()), 'invalid recorded512proxy')
    require(len(identities) == len(reference)
            and {row['subject_id'] for row in identities} == {7, 8}, 'public source identities mismatch')
    original = reference.copy()
    conditions = {}
    for index, name in enumerate(PUBLIC_CONDITIONS):
        seed = PUBLIC_SEED+index
        raw = checked_array(corrupt(reference.copy(), name, seed, cfg), reference.shape, name)
        if name == 'identity':
            require(np.array_equal(raw, reference), 'identity corruption altered reference')
        notch = fixed_notch(raw.transpose(0, 2, 1)).transpose(0, 2, 1).copy()
        input_copy = notch.copy()
        output = api.process_windows(notch, model, batch_size=32)
        require(np.array_equal(notch, input_copy), 'process_windows mutated notch input')
        metrics = window_metrics(reference, raw, notch, output)
        for row, identity in zip(metrics['windows'], identities):
            row['source_identity'] = identity
        conditions[name] = {'corruption_seed': seed, **metrics}
        print(json.dumps({'stage': 'public', 'condition': name, 'seed': seed}), flush=True)
    require(np.array_equal(reference, original), 'recorded reference changed')
    return {'status': 'formal', 'subjects': [7, 8], 'reference_shape': list(reference.shape),
            'reference_dtype': str(reference.dtype), 'source_identities': identities,
            'reference_definition': 'notch-conditioned recorded512proxy, per-reference RMS units; not clean EEG',
            'evaluation_is_pristine_project_holdout': False, 'reference_is_clean_truth': False,
            'corruption_generator': 'scripts.train_ssvep_denoiser_v13.corrupt',
            'seed_definition': '2026101603 + condition index in recorded order',
            'condition_order': list(PUBLIC_CONDITIONS), 'conditions': conditions,
            'cross_v12_fair_comparison_available': False,
            'cross_v12_comparison_limitation': 'v13 augmentation differs from v12',
            'personal_data_used': False, 'labels_enter_model_forward': False, **FALSE_FLAGS}


def wrap_phase(angle):
    import numpy as np
    return (np.asarray(angle, dtype=np.float64)+np.pi)%(2*np.pi)-np.pi


def joint_sine_fit(signal, frequencies, *, sample_rate_hz=250.0, start_sample=0):
    import numpy as np
    value=np.asarray(signal,dtype=np.float64); frequencies=tuple(float(f) for f in frequencies)
    require(value.ndim==2 and value.shape[0]>2+2*len(frequencies) and value.shape[1]>0 and bool(np.isfinite(value).all()),'invalid LS signal')
    require(math.isfinite(sample_rate_hz) and sample_rate_hz>0 and frequencies and len(set(frequencies))==len(frequencies) and all(0<f<sample_rate_hz/2 for f in frequencies),'invalid LS frequencies')
    time=(np.arange(len(value),dtype=np.float64)+start_sample)/sample_rate_hz
    design=[np.ones_like(time),time-time.mean()]
    for f in frequencies: design.extend((np.sin(2*np.pi*f*time),np.cos(2*np.pi*f*time)))
    design=np.column_stack(design); coeff,_,rank,s=np.linalg.lstsq(design,value,rcond=None); require(rank==design.shape[1] and bool(np.isfinite(coeff).all()) and bool(np.isfinite(s).all()),'invalid/rank-deficient LS fit')
    components=[]
    for i,f in enumerate(frequencies):
        sine,cosine=coeff[2+2*i:4+2*i]; amp=np.hypot(sine,cosine); phase=wrap_phase(np.arctan2(cosine,sine))
        components.append({'frequency_hz':f,'amplitude_per_channel':amp.tolist(),'phase_radians_per_channel':[float(p) if a>1e-12 else None for a,p in zip(amp,phase)]})
    return {'components':components,'intercept_at_center':coeff[0].tolist(),'trend_per_second':coeff[1].tolist(),'start_sample':start_sample,'sample_count':len(value),'rank':int(rank),'condition_number':float(s[0]/s[-1]),'residual_mse_per_channel':np.mean((value-design@coeff)**2,axis=0).tolist()}


def compare_fits(reference, measured):
    require(len(reference['components'])==len(measured['components']),'LS component count mismatch'); result=[]
    for a,b in zip(reference['components'],measured['components']):
        require(a['frequency_hz']==b['frequency_hz'],'LS frequency mismatch')
        ratios=[]; errors=[]
        for amplitude,value,phase,actual in zip(a['amplitude_per_channel'],b['amplitude_per_channel'],a['phase_radians_per_channel'],b['phase_radians_per_channel']):
            ratios.append(value/amplitude if amplitude>1e-12 else None); errors.append(float(math.degrees(wrap_phase(actual-phase))) if phase is not None and actual is not None else None)
        result.append({'frequency_hz':a['frequency_hz'],'amplitude_ratio_per_channel':ratios,'wrapped_phase_error_degrees_per_channel':errors})
    return result


def waveform_errors(clean, signal):
    import numpy as np
    truth=np.asarray(clean,dtype=np.float64); value=checked_array(signal,truth.shape,'waveform'); require(truth.ndim==2 and len(truth)>0 and bool(np.isfinite(truth).all()),'invalid artificial clean')
    centered=truth-truth.mean(axis=0,keepdims=True); error=(value-value.mean(axis=0,keepdims=True))-centered; energy=float(np.mean(centered**2)); require(energy>0,'undefined centered energy'); mse=float(np.mean(error**2)); mean_error=value.mean(axis=0)-truth.mean(axis=0)
    return {'samples':len(truth),'scalar_count':int(truth.size),'clean_centered_mse':mse,'clean_centered_energy':energy,'clean_centered_normalized_mse':mse/energy,'uncentered_mse':float(np.mean((value-truth)**2)),'mean_error_per_channel':mean_error.tolist(),'mean_offset_mse':float(np.mean(mean_error**2))}


def contribution(before,after):
    x,y=before['clean_centered_mse'],after['clean_centered_mse']; return {'centered_mse_reduction':x-y,'relative_centered_error_reduction':1-y/x if x>0 else None}


def measure_synthetic_case(stored,clean,tracks,receipt):
    import numpy as np
    require(set(tracks)==set(TRACKS) and np.asarray(clean).shape==(4096,8),'exact synthetic case required'); hz=stored['frequency_hz']; regions={}
    for region,(start,end) in REGIONS.items():
        truth=clean[start:end]; ref_fit=joint_sine_fit(truth,(hz,2*hz,7),start_sample=start); measured={}
        for name in TRACKS:
            signal=checked_array(tracks[name],(4096,8),name)[start:end]; fit=joint_sine_fit(signal,(hz,2*hz,7),start_sample=start)
            measured[name]={'errors':waveform_errors(truth,signal),'joint_fit':fit,'harmonics':compare_fits(ref_fit,fit)}
        errors={name:row['errors'] for name,row in measured.items()}; regions[region]={'start_sample':start,'end_sample_exclusive':end,'artificial_clean_joint_fit':ref_fit,'tracks':measured,'contributions':{'fixed_notch_raw_to_notch':contribution(errors['raw'],errors['notch']),'learned_notch_to_v13':contribution(errors['notch'],errors['v13']),'traditional_notch_to_traditional':contribution(errors['notch'],errors['traditional'])}}
    rms=np.std(tracks['notch'],axis=0,keepdims=True,dtype=np.float64); require(bool(np.all(rms>1e-8)),'undefined long normalization')
    return {'frequency_hz':hz,'condition':stored['condition'],'seed':stored['seed'],'regions':regions,'receipt':receipt,'output_reference_normalized_change':float(np.mean(((tracks['v13']-clean)/rms)**2)),'learned_only_normalized_change':float(np.mean(((tracks['v13']-tracks['notch'])/rms)**2))}


def aggregate_cases(cases):
    require(cases,'cannot aggregate empty cases'); regions={}
    for region in REGIONS:
        tracks={}
        for name in TRACKS:
            rows=[case['regions'][region]['tracks'][name]['errors'] for case in cases]; avg={key:math.fsum(row[key] for row in rows)/len(rows) for key in ('clean_centered_mse','clean_centered_energy','uncentered_mse','mean_offset_mse')}; avg['clean_centered_normalized_mse']=avg['clean_centered_mse']/avg['clean_centered_energy']; tracks[name]=avg
        regions[region]={'tracks':tracks,'contributions':{'fixed_notch_raw_to_notch':contribution(tracks['raw'],tracks['notch']),'learned_notch_to_v13':contribution(tracks['notch'],tracks['v13']),'traditional_notch_to_traditional':contribution(tracks['notch'],tracks['traditional'])},'v13_minus_traditional_centered_nmse':tracks['v13']['clean_centered_normalized_mse']-tracks['traditional']['clean_centered_normalized_mse']}
    return {'case_count':len(cases),'regions':regions,'output_reference_normalized_change':math.fsum(case['output_reference_normalized_change'] for case in cases)/len(cases),'learned_only_normalized_change':math.fsum(case['learned_only_normalized_change'] for case in cases)/len(cases)}


def load_known_cases(plan,root=ROOT):
    path=Path(root)/PRIOR_SYNTHETIC; require(sha(path)==plan['prior_evidence'][PRIOR_SYNTHETIC],'v12 synthetic artifact SHA mismatch'); payload=read_json(path); cases=payload['cases']; require(payload['case_count']==len(cases)==20,'fixed20 case count mismatch'); require(set((row['frequency_hz'],row['condition']) for row in cases)=={(f,c) for f in FREQUENCIES for c in SYNTHETIC_CONDITIONS},'fixed20 cases changed'); require(all(type(row['seed']) is int and row['seed']>=0 for row in cases),'invalid stored seeds'); return cases


def evaluate_synthetic(model,plan,*,root=ROOT):
    import numpy as np
    from scripts.evaluate_ssvep_denoiser_v10 import synthetic_case
    from eeg_tools import ssvep_v13_runtime as api
    from eeg_tools.workstation.v9_offline import _traditional
    measured=[]
    for stored in load_known_cases(plan,root):
        clean,raw=synthetic_case(stored['frequency_hz'],stored['condition'],stored['seed']); raw_before=raw.copy(); output,notch,receipt=api.process_recording(raw,model); validate_receipt(receipt,len(raw)); require(np.array_equal(raw,raw_before),'synthetic raw input mutated')
        tracks={'raw':raw,'notch':checked_array(notch,raw.shape,'notch'),'traditional':_traditional(raw),'v13':checked_array(output,raw.shape,'v13')}; require(np.allclose(notch,fixed_notch(raw),rtol=0,atol=1e-10),'recording notch mismatch')
        row=measure_synthetic_case(stored,clean,tracks,receipt); baseline=[]
        for name in ('raw','notch','traditional'):
            for region in REGIONS:
                for key in ('clean_centered_mse','clean_centered_normalized_mse','mean_offset_mse'):
                    old=float(stored['errors'][name][region][key]); actual=row['regions'][region]['tracks'][name]['errors'][key]; ok=math.isclose(old,actual,rel_tol=1e-8,abs_tol=1e-10); baseline.append({'track':name,'region':region,'metric':key,'stored':old,'actual':actual,'matches':ok})
        require(all(item['matches'] for item in baseline),'known case baseline differs from frozen v12 artifact'); row['known_case_baseline_checks']=baseline; measured.append(row)
    return {'status':'formal','case_count':len(measured),'cases':measured,'condition_order':list(SYNTHETIC_CONDITIONS),'frequencies':list(FREQUENCIES),'known_case_source':PRIOR_SYNTHETIC,'known_case_source_sha256':sha(Path(root)/PRIOR_SYNTHETIC),'conditions':{c:aggregate_cases([x for x in measured if x['condition']==c]) for c in SYNTHETIC_CONDITIONS},'pooled_corrupted':aggregate_cases([x for x in measured if x['condition']!='identity']),'balanced_drift_and_combined':aggregate_cases([x for x in measured if x['condition'] in ('drift.3','combined')]),'known_clean_is_artificial_only':True,'benchmark_is_reused_development_data':True,'phase_convention':'joint LS intercept, centered time trend, fundamental, second harmonic, 7Hz; absolute sample time','research_error_region':'full','interior_definition':'[512,3584)','cross_v12_fair_comparison_available':False,**FALSE_FLAGS}


def validate_receipt(receipt, length):
    from eeg_tools import ssvep_v13_runtime as api
    starts = list(range(0, length-4096+1, 1024))
    if starts[-1] != length-4096:
        starts.append(length-4096)
    expected = dict(window_samples=4096, hop_samples=1024, window_count=len(starts),
                    final_window_start=starts[-1], processed_samples=length)
    require(all(type(receipt.get(k)) is int and receipt[k] == v for k, v in expected.items()),
            'invalid recording receipt dimensions')
    for key, value in dict(runtime_enabled=False, production_validated=False,
                           unit_conversion_performed=False, resampled=False,
                           source_channel_order_preserved=True,
                           whole_record_protected_spectrum_guaranteed=False).items():
        require(receipt.get(key) is value, 'invalid recording receipt flag: '+key)
    require(receipt.get('normalization_contract') == api.NORMALIZATION, 'receipt normalization mismatch')


def normalization_contract(model):
    import numpy as np
    import torch
    from eeg_tools import ssvep_v13_runtime as api
    rng = np.random.default_rng(PUBLIC_SEED)
    results = []
    device = next(model.parameters(), torch.empty(0)).device
    for length in (512, 4096):
        probe = rng.normal(size=(3, 8, length)).astype(np.float64)
        probe *= np.arange(1, 9)[None, :, None]
        probe += np.arange(8)[None, :, None]*1000
        normalized, _, rms = api.normalize_noisy(probe)
        with torch.inference_mode():
            predicted, aux = model.forward_with_aux(torch.from_numpy(normalized).to(device))
        prediction = predicted.cpu().numpy()
        expected_freq = np.fft.rfftfreq(length, 1/250)
        expected_mask = (expected_freq == 0) | ((expected_freq >= 8) & (expected_freq <= 40))
        gain = aux['gain'].cpu().numpy()
        mask = aux['protected_mask'].cpu().numpy()
        freq = aux['frequencies_hz'].cpu().numpy()
        require(mask.dtype == np.bool_ and np.array_equal(mask, expected_mask), 'invalid protected mask')
        require(freq.shape == expected_freq.shape and bool(np.isfinite(freq).all())
                and np.array_equal(freq, expected_freq), 'invalid actual-N frequencies')
        require(gain.shape == (3, 8, len(freq)) and bool(np.isfinite(gain).all())
                and bool(np.all((gain >= 0) & (gain <= 1))) and bool(np.all(gain[..., mask] == 1)),
                'invalid/nonfinite gain or protected gain')
        require(prediction.dtype == np.float32 and prediction.shape == normalized.shape
                and bool(np.isfinite(prediction).all()), 'invalid explicit prediction')
        explicit = api.restore_residual(probe, normalized, prediction, rms)
        actual = api.process_windows(probe, model, batch_size=32)
        results.append({'window_samples': length, 'protected_mask_verified': True,
                        'finite_gain_verified': True, 'unit_protected_gain_verified': True,
                        'max_absolute_difference': float(np.max(np.abs(
                            checked_array(actual, probe.shape, 'contract')-explicit)))})
    return {'max_absolute_difference': max(x['max_absolute_difference'] for x in results),
            'seed': PUBLIC_SEED, 'cases': results,
            'definition': 'explicit normalize_noisy -> signal-only forward -> restore_residual vs process_windows'}


def personal_eval(source,folder,model,metadata=None):
    import numpy as np
    from eeg_tools import ssvep_v13_runtime as api
    from eeg_tools.personal_ssvep_decoder import decode_window,config_summary
    from eeg_tools.workstation import v9_offline as baseline
    from scripts.diagnose_neurostation_v9_output import validate_metadata,parse_raw
    before=baseline._source_hashes(source); validate_metadata(read_json(source/'session.json'),read_json(source/'channel_config.json')); raw,header=parse_raw((source/'raw_brainflow.tsv').read_bytes()); raw_before=raw.copy(); output,notch,receipt=api.process_recording(raw,model); validate_receipt(receipt,len(raw)); require(np.array_equal(raw,raw_before),'personal raw input mutated')
    tracks={'raw':raw,'notch':checked_array(notch,raw.shape,'personal notch'),'traditional':checked_array(baseline._traditional(raw),raw.shape,'personal traditional'),'v13':checked_array(output,raw.shape,'personal v13')}; require(np.allclose(notch,fixed_notch(raw),rtol=0,atol=1e-10),'personal notch mismatch')
    targets,trials,metadata_reason=baseline._classification_inputs(source,raw); rows=[]; target_ids={x[0] for x in targets}
    for trial in trials:
        row=dict(trial); row['predictions']={}
        for name,value in tracks.items():
            reason=metadata_reason or trial.get('unavailable_reason')
            if not reason:
                try: prediction=decode_window(value[trial['window_start_sample_index']:trial['offset_sample_index']].T,250.0,targets)
                except (ValueError,ArithmeticError) as exc: prediction={'available':False,'target_id':None,'reason':str(exc)}
            else: prediction={'available':False,'target_id':None,'reason':reason}
            row['predictions'][name]=prediction
        rows.append(row)
    summaries={}
    for name in TRACKS:
        available=[row for row in rows if row['predictions'][name].get('available') and row.get('presented_target_id') in target_ids]; margins=[row['predictions'][name]['confidence'] for row in available]
        summaries[name]={'recorded_trial_count':len(rows),'available_presented_trial_count':len(available),'presented_match_count':sum(row['predictions'][name]['target_id']==row['presented_target_id'] for row in available),'mean_top_two_margin':math.fsum(margins)/len(margins) if margins else None,'margin_is_probability':False}
    folder.mkdir(); artifacts={}
    for name,value in tracks.items():
        path=folder/(name+'.npy')
        with path.open('xb') as stream: np.save(stream,value,allow_pickle=False); stream.flush(); os.fsync(stream.fileno())
        artifacts[path.name]={'sha256':sha(path),'shape':list(value.shape),'dtype':str(value.dtype)}
    after=baseline._source_hashes(source); require(before==after,'personal source files changed')
    report={'status':'formal','source_session':str(source),'source_header':header,'source_sha256_before':before,'source_sha256_after':after,'source_hashes_match':True,'targets':targets,'trials':rows,'classification_unavailable_reason':metadata_reason,'tracks':summaries,'decoder':config_summary(),'cca_available_means':'calculation succeeded; forced argmax, not verified SSVEP detection','margin_definition':'relative top-two CCA score margin; not probability','presented_matching_definition':'matches to recorded presentation, not independent gaze accuracy','artifacts':artifacts,'receipt':receipt,**FALSE_FLAGS}; report.update(model_context(metadata or {})); write_json(folder/'report.json',report); return report


def _finite(value): return type(value) in (int,float) and math.isfinite(value)

def research_gate(public,synthetic,personal,contract,plan,metadata):
    thresholds=plan['fixed_research_checks']; checks={}
    def put(name,value,threshold,passed,available=True,detail=None): checks[name]={'available':bool(available),'passed':bool(available and passed),'value':value,'threshold':threshold,'detail':detail}
    put('selection_gate_passed_required',metadata.get('selection_gate_passed'),True,metadata.get('selection_gate_passed') is True,detail='Required by evaluator; explicit failed-selection diagnostic is never a selected model.')
    for name,value in (('public_identity_normalized_change_max',public['conditions']['identity']['output_reference_normalized_change']),('long_synthetic_identity_normalized_change_max',synthetic['conditions']['identity']['output_reference_normalized_change'])): put(name,value,thresholds[name],_finite(value) and 0<=value<=thresholds[name],_finite(value))
    pooled=synthetic['balanced_drift_and_combined']['regions']['full']; reduction=pooled['contributions']['learned_notch_to_v13']['relative_centered_error_reduction']; key='pooled_long_drift_and_combined_error_reduction_from_notch_min'; put(key,reduction,thresholds[key],_finite(reduction) and reduction>=thresholds[key],_finite(reduction))
    details={}
    for condition in SYNTHETIC_CONDITIONS[1:]:
        t=synthetic['conditions'][condition]['regions']['full']['tracks']; details[condition]={'notch_mse':t['notch']['clean_centered_mse'],'v13_mse':t['v13']['clean_centered_mse']}
    t=synthetic['pooled_corrupted']['regions']['full']['tracks']; details['pooled_corrupted']={'notch_mse':t['notch']['clean_centered_mse'],'v13_mse':t['v13']['clean_centered_mse']}; valid=all(_finite(x[k]) and x[k]>=0 for x in details.values() for k in ('notch_mse','v13_mse')); key='all_long_corrupted_conditions_nonworse_than_notch'; put(key,details,thresholds[key],valid and all(x['v13_mse']<=x['notch_mse'] for x in details.values()),valid)
    protected=[row for row in synthetic['cases'] if row['condition'] in ('identity','line50')]; complete=len(protected)==8 and {(x['frequency_hz'],x['condition']) for x in protected}=={(f,c) for f in FREQUENCIES for c in ('identity','line50')}
    for index,harmonic in enumerate(('fundamental','second_harmonic')):
        for metric,field in (('amplitude_relative_error','amplitude_ratio_per_channel'),('phase_error_degrees','wrapped_phase_error_degrees_per_channel')):
            key='noisefree_and_line50_'+harmonic+'_'+metric+'_max'; errors=[]; valid=complete
            for case in protected:
                for region in REGIONS:
                    values=case['regions'][region]['tracks']['v13']['harmonics'][index][field]; valid=valid and len(values)==8 and all(_finite(x) for x in values); errors.extend(abs(x-1) if metric=='amplitude_relative_error' else abs(x) for x in values if _finite(x))
            maximum=max(errors) if errors else None; put(key,maximum,thresholds[key],valid and maximum<=thresholds[key],valid)
    value=contract.get('max_absolute_difference'); valid=_finite(value) and {x.get('window_samples') for x in contract.get('cases',[])}=={512,4096}; put('normalization_contract_max_absolute_difference',value,thresholds['normalization_contract_max_absolute_difference'],valid and 0<=value<=thresholds['normalization_contract_max_absolute_difference'],valid)
    personal_values=[]; personal_valid=len(personal)>=2
    for row in personal:
        tracks=row['tracks']; trials=tracks['raw']['recorded_trial_count']; available=trials>0 and all(tracks[name]['available_presented_trial_count']==trials for name in TRACKS); personal_valid=personal_valid and available and row['source_hashes_match'] is True; personal_values.append({'source_session':row['source_session'],'available':available,'raw_presented_match_count':tracks['raw']['presented_match_count'],'v13_presented_match_count':tracks['v13']['presented_match_count']})
    key='personal_presented_match_count_not_lower_than_raw'; put(key,personal_values,thresholds[key],personal_valid and all(x['v13_presented_match_count']>=x['raw_presented_match_count'] for x in personal_values),personal_valid,detail='Presentation proxy only; missing personal evidence is unavailable/false, never vacuous true.')
    require(set(thresholds)==set(x for x in checks if x!='selection_gate_passed_required'),'unhandled fixed plan check')
    return {'passed':all(x['passed'] for x in checks.values()),'checks':checks,'threshold_source':PLAN,'selection_gate_passed_required':True,'clinical_validation':False,'clean_physiological_validation':False,'authorizes_default_deployment':False,**FALSE_FLAGS}


def deployment_comparison(synthetic,metadata):
    return {'per_condition_v13_minus_traditional_centered_nmse':{c:{r:v['v13_minus_traditional_centered_nmse'] for r,v in x['regions'].items()} for c,x in synthetic['conditions'].items()},'harmonic_comparison_location':'synthetic.json cases[].regions.*.tracks.{traditional,v13}.harmonics','recommendation':('Failed selection: diagnostic final state only; do not enable default runtime.' if metadata.get('selection_gate_passed') is not True else 'Offline research only; inspect full/interior errors and harmonics against traditional filtering before further validation.'),'automatic_deployment_performed':False,**FALSE_FLAGS}


def model_context(metadata):
    return {key: metadata.get(key) for key in
            ('checkpoint_role', 'selected_epoch', 'diagnostic_epoch', 'selection_gate_passed')}


def input_hashes(cfg, root=ROOT):
    from eeg_tools import ssvep_v13_runtime as api, ssvep_v10_data as data
    names = {api.CONFIG, api.LOCK, api.MODEL, api.MANIFEST, api.REPORT, PLAN,
             'scripts/evaluate_ssvep_denoiser_v13.py', 'tests/test_evaluate_ssvep_denoiser_v13.py',
             'eeg_tools/ssvep_v13_runtime.py', 'scripts/train_ssvep_denoiser_v13.py',
             'eeg_tools/ssvep_v10_data.py', data.CONFIG, data.LOCK,
             'scripts/evaluate_ssvep_denoiser_v10.py', 'eeg_tools/personal_ssvep_decoder.py',
             'configs/personal_ssvep_decoder_v1.json', 'eeg_tools/workstation/v9_offline.py',
             'scripts/diagnose_neurostation_v9_output.py', 'eeg_tools/offline_analysis.py'}
    names.update(read_json(Path(root)/PLAN)['prior_evidence'])
    names.update(read_json(Path(root)/api.LOCK)['files'])
    for subject in (7, 8):
        name = cfg['data']['reference_directory']+f'/subject_{subject:03d}.npz'
        names.update((name, name+'.manifest.json'))
    return {name: sha(api.safe_path(root, name)) for name in sorted(names)}


def verify_output_artifacts(output):
    """One SHA table, including every subreport and npy; verify nested bindings."""
    artifacts = {}
    for path in sorted(output.rglob('*')):
        if path.is_file():
            artifacts[path.relative_to(output).as_posix()] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    for folder in sorted(output.glob('personal_*')):
        report = read_json(folder/'report.json')
        for name, metadata in report['artifacts'].items():
            key = (folder/name).relative_to(output).as_posix()
            require(key in artifacts and artifacts[key]['sha256'] == metadata['sha256'],
                    'personal/summary artifact SHA disagreement: '+key)
    return artifacts


def evaluate(output_directory, personal=(), device='cpu', *, root=ROOT,
             allow_failed_selection_diagnostic=False):
    output, sources = validate_paths(output_directory, personal, root=root)
    require(device in ('cpu', 'cuda:0'), 'invalid device')
    from eeg_tools import ssvep_v13_runtime as api
    from eeg_tools.workstation.v9_offline import _source_hashes
    plan = load_evaluation_plan(root)
    cfg = api.load_config(root)
    preflight_before = api.preflight(root)
    before = input_hashes(cfg, root)
    source_before = {str(source): _source_hashes(source) for source in sources}
    model, metadata = load_evaluation_model(root, device,
        allow_failed_selection_diagnostic=allow_failed_selection_diagnostic)
    output.mkdir(parents=True, exist_ok=False)
    try:
        public = evaluate_public(model, cfg, root=root)
        synthetic = evaluate_synthetic(model, plan, root=root)
        contract = normalization_contract(model)
        public.update(model_context(metadata))
        synthetic.update(model_context(metadata))
        write_json(output/'public.json', public)
        write_json(output/'synthetic.json', synthetic)
        write_json(output/'normalization.json', {'status':'formal', **contract, **model_context(metadata), **FALSE_FLAGS})
        reports = [personal_eval(source, output/f'personal_{i:02d}', model, metadata)
                   for i, source in enumerate(sources)]
        for source, report in zip(sources, reports):
            require(source_before[str(source)] == report['source_sha256_before'] == _source_hashes(source),
                    'personal source changed before bundle commit')
        gate = research_gate(public, synthetic, reports, contract, plan, metadata)
        artifacts = verify_output_artifacts(output)
        source_after = input_hashes(cfg, root)
        require(source_after == before, 'frozen evaluation inputs changed before commit')
        require(api.preflight(root) == preflight_before, 'preflight changed before commit')
        summary = {
            'schema_version': 1, 'status': 'formal', 'completion_status': 'complete',
            'created_at_utc': datetime.now(timezone.utc).isoformat(), 'device': device,
            'model': metadata, **model_context(metadata),
            'allow_failed_selection_diagnostic': allow_failed_selection_diagnostic,
            'public': {'report': 'public.json', 'subjects': [7, 8]},
            'synthetic': {'report': 'synthetic.json', 'case_count': synthetic['case_count']},
            'normalization_contract': contract, 'inference_settings': inference_settings(),
            'personal': [{'source_session': row['source_session'], 'report': f'personal_{i:02d}/report.json',
                          'tracks': row['tracks'], 'source_sha256': row['source_sha256_before']}
                         for i, row in enumerate(reports)],
            'personal_sources_sha256_before': source_before,
            'personal_sources_sha256_after': {str(source): _source_hashes(source) for source in sources},
            'source_input_sha256_before': before, 'source_input_sha256_after': source_after,
            'preflight_before': preflight_before, 'preflight_after': api.preflight(root),
            'artifacts': artifacts, 'research_gate': gate,
            'traditional_comparison': deployment_comparison(synthetic, metadata),
            'training_performed': False, 'checkpoint_selection_performed': False,
            'evaluation_is_pristine_project_holdout': False, 'cross_v12_fair_comparison_available': False,
            'clinical_validation': False, 'clean_physiological_validation': False,
            'completion_marker_definition': 'Only summary.json commits this bundle; its SHA table excludes itself.',
            'limitations': ['Public references are recorded proxies, not clean EEG.',
                            'v13/v12 augmentation differs; no fair cross-version comparison.',
                            'Long numerical references are artificial, and fixed20 cases were used historically.',
                            'Presentation matching is not independent gaze accuracy.',
                            'Physical electrode mapping and calibration remain unverified.',
                            'Failed selection diagnostics cannot authorize default runtime.'], **FALSE_FLAGS}
        require(summary['personal_sources_sha256_after'] == source_before, 'personal source final SHA mismatch')
        require(summary['preflight_after'] == preflight_before, 'preflight final mismatch')
        require(input_hashes(cfg, root) == before, 'source input final SHA mismatch')
        require(verify_output_artifacts(output) == artifacts, 'output changed before commit')
        write_json(output/'summary.json', summary, atomic=True)
        return summary
    except Exception as exc:
        try:
            write_json(output/'incomplete.json', {'status': 'formal', 'completion_status': 'incomplete',
                'error': str(exc), 'error_type': type(exc).__name__, **FALSE_FLAGS}, atomic=True)
        except Exception:
            pass
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-directory',required=True); parser.add_argument('--personal-session',action='append',default=[]); parser.add_argument('--device',choices=('cpu','cuda:0'),default='cpu'); parser.add_argument('--allow-failed-selection-diagnostic',action='store_true',help='Explicitly assess fully verified failed-selection final state; research gate necessarily fails'); args=parser.parse_args(argv)
    try: result=evaluate(args.output_directory,args.personal_session,args.device,allow_failed_selection_diagnostic=args.allow_failed_selection_diagnostic)
    except Exception as exc: print(json.dumps({'status':'formal','completion_status':'incomplete','error':str(exc),'error_type':type(exc).__name__,**FALSE_FLAGS},ensure_ascii=False),file=sys.stderr); return 1
    print(json.dumps({'status':'formal','completion_status':'complete','summary':str(Path(args.output_directory)/'summary.json'),'research_gate_passed':result['research_gate']['passed'],'checkpoint_role':result['checkpoint_role'],'selection_gate_passed':result['selection_gate_passed'],**FALSE_FLAGS},ensure_ascii=False)); return 0

if __name__=='__main__': raise SystemExit(main())
