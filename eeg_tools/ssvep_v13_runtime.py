"""Independent v13 offline research runtime. Metadata checks import no scientific stack.

Only the normalized signal enters the network. Reference units, channel order and
native time axis are retained. Per-window protected bins do not imply protected
whole-record spectra after Hann overlap-add. See the frozen config for the design.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'configs/learned_denoiser_v13_personal.json'
SOURCES = ('eeg_tools/ssvep_v13_runtime.py', 'scripts/train_ssvep_denoiser_v13.py',
           'tests/test_ssvep_v13.py')
LOCK = 'analysis_outputs/ssvep_v13_training_lock_20261013.json'
MODEL = 'models/denoiser_v13_personal.pt'
MANIFEST = 'models/denoiser_v13_personal.manifest.json'
REPORT = 'analysis_outputs/ssvep_v13_training_20261013.json'
PLAN = 'analysis_outputs/ssvep_v13_experiment_plan_20261013.json'
NORMALIZATION = 'noisy_float64_channel_mean_rms_float32_forward_float64_residual_restore'
CONDITIONS = ('identity', 'drift', 'high_burst', 'gaussian', 'combined')


def require(ok, message):
    if not ok:
        raise ValueError('v13: ' + message)


def sha(filename):
    digest = hashlib.sha256()
    with Path(filename).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def safe_path(root, name, *, new=False):
    """Allow only normalized relative paths; reject links/reparse points in parents.

    Exclusive creation is still required at the final write (new is only a check).
    This protects against accidental redirection, not a hostile concurrent OS user.
    """
    require(isinstance(name, str) and '\\' not in name and ':' not in name,
            'unsafe relative path')
    parts = PurePosixPath(name).parts
    require(parts and not name.startswith('/') and '..' not in parts
            and '/'.join(parts) == name, 'unsafe relative path')
    base = Path(root).absolute()
    for item in (base, *base.parents):
        if item.exists():
            stat = item.lstat()
            require(not item.is_symlink() and not (getattr(stat, 'st_file_attributes', 0) & 0x400),
                    'linked/reparse workspace parent')
    current = base
    for part in parts:
        current = current / part
        if current.exists() or current.is_symlink():
            stat = current.lstat()
            require(not current.is_symlink() and not (getattr(stat, 'st_file_attributes', 0) & 0x400),
                    'linked/reparse path: ' + name)
            if current.is_file():
                require(stat.st_nlink == 1, 'hard-linked file: ' + name)
    require(current.resolve().is_relative_to(base.resolve()), 'path escapes workspace')
    if new:
        require(not current.exists(), 'refusing existing output: ' + name)
    return current


def load_config(root=ROOT):
    cfg = json.loads(safe_path(root, CONFIG).read_text(encoding='utf-8'))
    require(cfg['schema_version'] == 1 and cfg['status'] == 'formal'
            and cfg['model_version'] == 'learned_denoise_v13_personal', 'invalid config identity')
    require((cfg['sample_rate_hz'], cfg['channel_count'], cfg['window_samples'], cfg['hop_samples'])
            == (250, 8, 4096, 1024) and cfg['supported_window_samples'] == [512, 4096],
            'native context contract changed')
    require(cfg['model']['channels'] == [4, 24, 24, 1] and cfg['model']['kernels'] == [7, 5, 1]
            and cfg['model']['protected_band_hz'] == [8, 40]
            and cfg['model']['gain_bounds'] == [0, 1]
            and cfg['model']['dc_preserved'] is True
            and cfg['model']['forward_inputs'] == ['signal'], 'architecture contract changed')
    require(cfg['normalization']['definition'] == NORMALIZATION
            and cfg['normalization']['rms_floor'] == 1e-8, 'normalization changed')
    require(cfg['preprocessing'] == dict(notch_hz=50, notch_q=30, phase='offline_zero_phase',
                                        resample=False, unit_conversion=False), 'notch contract changed')
    data = cfg['data']
    require(data['fit_subjects'] == [1, 2, 3, 4] and data['selection_subjects'] == [5, 6]
            and data['evaluation_subjects'] == [7, 8] and data['personal_sessions_used'] is False
            and data['public_evaluation_used_for_selection'] is False
            and data['public_evaluation_is_pristine'] is False
            and (data['window_start_sample'], data['window_end_sample_exclusive']) == (125, 637)
            and data['reference_directory'] == 'datasets/processed/ssvep_reference_v5_20261007/Wearable',
            'data roles/layout changed')
    synth = cfg['synthetic']
    require((synth['fit_count'], synth['selection_count'], synth['fit_seed'], synth['selection_seed'])
            == (96, 32, 2026101601, 2026101602)
            and synth['fundamental_hz'] == [8, 20] and synth['second_harmonic_max_hz'] == 40
            and synth['background_hz'] == 7 and synth['continuous'] is True
            and synth['background_1_over_f'] is True and synth['dc'] is True, 'synthetic plan changed')
    aug = cfg['augmentation']
    require(aug == dict(conditions=list(CONDITIONS), drift_hz=[0.1, 0.8], drift_rms=[0.2, 1.5],
                        trend_rms=[0.1, 0.8], burst_hz=[45, 110], burst_rms=[0.2, 1.5],
                        burst_duration_fraction=[0.05, 0.3], gaussian_rms=[0.2, 0.8]),
            'corruption plan changed')
    train = cfg['training']
    require((train['seed'], train['epochs'], train['batch_size'], train['device'])
            == (20261016, 16, 32, 'cuda:0') and train['require_cuda'] is True
            and train['deterministic'] is True and train['amp'] is False
            and train['allow_tf32'] is False and train['identity_loss_weight'] == 100
            and train['learning_rate'] == 0.0003 and train['weight_decay'] == 0.0001
            and train['gradient_clip'] == 1 and train['cpu_threads'] == 4, 'training plan changed')
    require(cfg['selection']['identity_normalized_change_max'] == 0.002
            and cfg['selection']['require_both_domain_identity_gates'] is True, 'selection gate changed')
    require(cfg['runtime']['enabled'] is False and cfg['runtime']['production_validated'] is False,
            'unvalidated model cannot be enabled')
    require(cfg['artifacts'] == dict(lock=LOCK, checkpoint=MODEL, manifest=MANIFEST, report=REPORT),
            'artifact paths changed')
    require(cfg['research_plan'] == PLAN, 'research plan path changed')
    return cfg


def validate_plan(root, cfg):
    plan = json.loads(safe_path(root, PLAN).read_text(encoding='utf-8'))
    require(plan['schema_version'] == 1 and plan['status'] == 'formal'
            and plan['runtime_enabled'] is False and plan['production_validated'] is False
            and plan['research_checks_authorize_default_deployment'] is False, 'invalid research plan')
    design = plan['training_design']
    expected = dict(long_window_samples=4096, sample_rate_hz=250, channel_count=8, hop_samples=1024,
                    public_fit_subjects=[1, 2, 3, 4], public_selection_subjects=[5, 6],
                    public_evaluation_subjects=[7, 8], personal_sessions_for_fit_or_selection=False,
                    labels_enter_model_forward=False, same_noisy_statistics_for_reference_and_input=True,
                    epochs=16, independent_personal_validation=False)
    require(all(design.get(key) == value for key, value in expected.items()), 'plan/config design mismatch')
    checks = plan['fixed_research_checks']
    require(checks['public_identity_normalized_change_max'] == cfg['selection']['identity_normalized_change_max']
            and checks['long_synthetic_identity_normalized_change_max'] == cfg['selection']['identity_normalized_change_max'],
            'plan/config identity gate mismatch')
    expected_evidence = {'analysis_outputs/v12_evaluation_20261013/summary.json',
                         'analysis_outputs/v12_evaluation_20261013/synthetic.json',
                         'analysis_outputs/v12_architecture_diagnosis_20261013.json'}
    require(set(plan['prior_evidence']) == expected_evidence, 'plan prior evidence set mismatch')
    for name, digest in plan['prior_evidence'].items():
        require(sha(safe_path(root, name)) == digest, 'plan prior evidence changed: ' + name)
    return plan


def bindings(cfg):
    files = [CONFIG, *SOURCES, PLAN]
    for subject in cfg['data']['fit_subjects'] + cfg['data']['selection_subjects']:
        name = cfg['data']['reference_directory'] + f'/subject_{subject:03d}.npz'
        files.extend((name, name + '.manifest.json'))
    return files


def preflight(root=ROOT):
    cfg = load_config(root)
    validate_plan(root, cfg)
    lock = json.loads(safe_path(root, LOCK).read_text(encoding='utf-8'))
    require(lock['schema_version'] == 1 and lock['status'] == 'formal'
            and lock['model_version'] == cfg['model_version']
            and lock['config_sha256'] == canonical_sha(cfg)
            and lock['research_plan_sha256'] == sha(safe_path(root, PLAN))
            and lock['normalization_contract'] == NORMALIZATION, 'configuration/plan lock mismatch')
    require(lock['roles'] == {'fit': [1, 2, 3, 4], 'select': [5, 6], 'personal': [], 'evaluation': []}
            and lock['synthetic'] == cfg['synthetic'] and lock['runtime_enabled'] is False
            and lock['production_validated'] is False, 'training lock role/plan mismatch')
    require(set(lock['files']) == set(bindings(cfg)), 'unexpected/missing lock bindings')
    for name, digest in lock['files'].items():
        require(isinstance(digest, str) and len(digest) == 64 and sha(safe_path(root, name)) == digest,
                'locked file changed: ' + name)
    return dict(status='ready', config_sha256=canonical_sha(cfg), training_lock_sha256=sha(safe_path(root, LOCK)),
                checked_files=len(lock['files']), waveforms_loaded=False, torch_imported=False,
                cuda_initialized=False, runtime_enabled=False, production_validated=False)


def _windows(value, name='signal'):
    import numpy as np
    arr = np.asarray(value)
    require(arr.dtype.kind == 'f' and arr.ndim == 3 and arr.shape[0] > 0
            and arr.shape[1] == 8 and arr.shape[2] in (512, 4096) and np.isfinite(arr).all(),
            name + ' must be finite floating [B,8,512|4096]')
    return arr.astype(np.float64, copy=False)


def normalize_noisy(noisy):
    import numpy as np
    x = _windows(noisy)
    means = x.mean(axis=-1, keepdims=True, dtype=np.float64)
    rms = np.sqrt(np.mean((x - means)**2, axis=-1, keepdims=True, dtype=np.float64))
    require(np.isfinite(means).all() and np.isfinite(rms).all() and np.all(rms > 1e-8),
            'flat/overflowed noisy channel; no channel replacement')
    normalized = np.asarray((x - means) / rms, dtype=np.float32)
    require(np.isfinite(normalized).all(), 'normalization overflow')
    return normalized, means, rms


def _stats(x, means, rms):
    import numpy as np
    mean, scale = np.asarray(means), np.asarray(rms)
    require(mean.dtype == scale.dtype == np.float64 and mean.shape == scale.shape == (len(x), 8, 1)
            and np.isfinite(mean).all() and np.isfinite(scale).all() and np.all(scale > 1e-8),
            'invalid float64 noisy statistics')
    return mean, scale


def target_in_noisy_coordinates(reference, means, rms):
    import numpy as np
    x = _windows(reference, 'reference')
    mean, scale = _stats(x, means, rms)
    target = np.asarray((x - mean) / scale, dtype=np.float32)
    require(np.isfinite(target).all(), 'target coordinate overflow')
    return target


def restore_residual(noisy, normalized, prediction, rms):
    import numpy as np
    x = _windows(noisy)
    a, b, scale = np.asarray(normalized), np.asarray(prediction), np.asarray(rms)
    require(a.dtype == b.dtype == np.float32 and a.shape == b.shape == x.shape
            and np.isfinite(a).all() and np.isfinite(b).all() and scale.dtype == np.float64
            and scale.shape == (len(x), 8, 1) and np.isfinite(scale).all() and np.all(scale > 1e-8),
            'invalid residual restoration inputs')
    result = x + (b.astype(np.float64) - a.astype(np.float64)) * scale
    require(np.isfinite(result).all(), 'restoration overflow')
    return result


def restored_mse_torch(prediction, normalized, noisy, rms, reference):
    """Autograd through EXACT runtime float64 residual restoration, no rounded target loss."""
    import torch
    require(prediction.dtype == normalized.dtype == torch.float32
            and prediction.shape == normalized.shape == noisy.shape == reference.shape
            and noisy.dtype == reference.dtype == rms.dtype == torch.float64
            and rms.shape == (*noisy.shape[:2], 1), 'loss coordinate/shape mismatch')
    restored = noisy + (prediction.double() - normalized.double()) * rms
    loss = (restored - reference).square().mean(dim=(1, 2))
    require(bool(torch.isfinite(loss).all()), 'nonfinite restored-unit loss')
    return loss


def condition_line(signal):
    import numpy as np
    from scipy.signal import iirnotch, filtfilt
    arr = np.asarray(signal)
    require(arr.dtype.kind == 'f' and arr.ndim in (2, 3) and arr.shape[-1] == 8
            and arr.shape[-2] >= 512 and np.isfinite(arr).all(), 'invalid native notch input')
    x = arr.astype(np.float64, copy=False)
    mean = x.mean(axis=-2, keepdims=True)
    b, a = iirnotch(50, 30, fs=250)
    result = filtfilt(b, a, x - mean, axis=-2) + mean
    require(np.isfinite(result).all(), 'notch output overflow')
    return result


def build_model(config=None):
    """Lazy torch factory. N determines features, masks and inverse FFT length."""
    import torch
    from torch import nn
    from torch.nn import functional as F
    cfg = load_config() if config is None else config
    require(cfg['model_version'] == 'learned_denoise_v13_personal'
            and cfg['model']['channels'] == [4, 24, 24, 1], 'v13 config required')

    class SpectralDenoiserV13(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Conv1d(4, 24, 7, padding=3), nn.SiLU(),
                                         nn.Conv1d(24, 24, 5, padding=2), nn.SiLU(), nn.Conv1d(24, 1, 1))
            nn.init.zeros_(self.encoder[-1].weight)
            nn.init.constant_(self.encoder[-1].bias, cfg['model']['initial_gate_bias'])

        def forward_with_aux(self, signal):
            require(isinstance(signal, torch.Tensor) and signal.dtype == torch.float32
                    and signal.ndim == 3 and signal.shape[0] > 0 and signal.shape[1] == 8
                    and signal.shape[-1] in (512, 4096) and bool(torch.isfinite(signal).all()),
                    'forward requires finite float32 [B,8,512|4096] signal only')
            n = signal.shape[-1]
            spectrum = torch.fft.rfft(signal, n=n, dim=-1)
            require(bool(torch.isfinite(spectrum).all()), 'FFT overflow')
            freq = torch.fft.rfftfreq(n, d=1 / 250, device=signal.device)
            protected = (freq == 0) | ((freq >= 8) & (freq <= 40))
            local = torch.log1p(spectrum.abs() / math.sqrt(n))
            across = local.mean(dim=1, keepdim=True).expand_as(local)
            b, c, bins = local.shape
            smooth = F.avg_pool1d(F.pad(local.reshape(b*c, 1, bins), (2, 2), mode='replicate'), 5, stride=1)
            feature = torch.stack((local, across, local - smooth.reshape(b, c, bins),
                                   (freq / 125).reshape(1, 1, bins).expand_as(local)), dim=2)
            logits = self.encoder(feature.reshape(b*c, 4, bins)).reshape(b, c, bins)
            require(bool(torch.isfinite(logits).all()), 'encoder overflow')
            gain = 1 - torch.sigmoid(logits).masked_fill(protected, 0)
            output = torch.fft.irfft(spectrum * gain, n=n, dim=-1)
            require(bool(torch.isfinite(output).all()), 'output overflow')
            return output, {'gain': gain, 'frequencies_hz': freq, 'protected_mask': protected}

        def forward(self, signal):
            return self.forward_with_aux(signal)[0]

    return SpectralDenoiserV13()


def process_windows(noisy, model, batch_size=32):
    """Learned stage only. Caller supplies separately notch-conditioned input."""
    import numpy as np
    import torch
    require(type(batch_size) is int and batch_size > 0, 'positive integer batch size required')
    x = _windows(noisy)
    device = next(model.parameters(), torch.empty(0)).device
    require(not model.training, 'inference requires model.eval()')
    output = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            chunk = x[start:start + batch_size]
            normalized, _, rms = normalize_noisy(chunk)
            prediction = model(torch.from_numpy(normalized).to(device))
            require(isinstance(prediction, torch.Tensor) and prediction.dtype == torch.float32
                    and tuple(prediction.shape) == normalized.shape and prediction.device == device,
                    'invalid model output contract')
            output.append(restore_residual(chunk, normalized, prediction.cpu().numpy(), rms))
    return np.concatenate(output)


def window_starts(length):
    require(type(length) is int and length >= 4096, 'recording requires >=4096 native samples')
    starts = list(range(0, length - 4096 + 1, 1024))
    if starts[-1] != length - 4096:
        starts.append(length - 4096)
    return starts


def process_recording(signal, model, batch_size=32):
    import numpy as np
    x = np.asarray(signal)
    require(x.dtype.kind == 'f' and x.ndim == 2 and x.shape[1] == 8 and len(x) >= 4096
            and np.isfinite(x).all(), 'finite floating native [T>=4096,8] recording required')
    require(type(batch_size) is int and batch_size > 0, 'positive integer batch size required')
    notch = condition_line(x)
    starts = window_starts(len(x))
    hann = np.hanning(4098)[1:-1]
    residual = np.zeros_like(notch, dtype=np.float64)
    weights = np.zeros(len(x), dtype=np.float64)
    for offset in range(0, len(starts), batch_size):
        selected = starts[offset:offset + batch_size]
        chunk = np.stack([notch[s:s+4096].T for s in selected])
        restored = process_windows(chunk, model, batch_size)
        for i, start in enumerate(selected):
            residual[start:start+4096] += (restored[i] - chunk[i]).T * hann[:, None]
            weights[start:start+4096] += hann
    require(np.all(weights > 0), 'OLA coverage failure')
    output = notch + residual / weights[:, None]
    require(np.isfinite(output).all() and output.shape == x.shape, 'invalid OLA output')
    return output, notch, dict(window_samples=4096, hop_samples=1024, window_count=len(starts),
                              final_window_start=starts[-1], processed_samples=len(x),
                              normalization_contract=NORMALIZATION, unit_conversion_performed=False,
                              resampled=False, source_channel_order_preserved=True,
                              fixed_notch='50Hz Q30 offline zero phase; report separately',
                              overlap_add='positive Hann weighted residual including final window',
                              whole_record_protected_spectrum_guaranteed=False,
                              runtime_enabled=False, production_validated=False)


def load_model(root=ROOT, device='cpu'):
    require(device in ('cpu', 'cuda:0'), 'explicit cpu or cuda:0 required')
    state, cfg = preflight(root), load_config(root)
    manifest = json.loads(safe_path(root, MANIFEST).read_text(encoding='utf-8'))
    report = json.loads(safe_path(root, REPORT).read_text(encoding='utf-8'))
    for value in (manifest, report):
        require(value['model_version'] == cfg['model_version']
                and value['config_sha256'] == state['config_sha256']
                and value['training_lock_sha256'] == state['training_lock_sha256']
                and value['research_plan_sha256'] == sha(safe_path(root, PLAN))
                and value['runtime_enabled'] is False and value['production_validated'] is False
                and value['selection_gate_passed'] is True, 'unbound/ineligible v13 artifact')
    require(manifest['checkpoint_sha256'] == sha(safe_path(root, MODEL))
            and manifest['training_report_sha256'] == sha(safe_path(root, REPORT)), 'artifact hash mismatch')
    import torch
    require(device == 'cpu' or torch.cuda.is_available(), 'CUDA unavailable')
    checkpoint = torch.load(safe_path(root, MODEL), map_location='cpu', weights_only=True)
    require(checkpoint['config'] == cfg and checkpoint['config_sha256'] == state['config_sha256']
            and checkpoint['training_lock_sha256'] == state['training_lock_sha256']
            and checkpoint['research_plan_sha256'] == sha(safe_path(root, PLAN))
            and checkpoint['model_version'] == cfg['model_version']
            and checkpoint['selected_epoch'] == manifest['selected_epoch'] == report['selected_epoch']
            and checkpoint['selection_gate_passed'] is True
            and checkpoint['runtime_enabled'] is False and checkpoint['production_validated'] is False,
            'checkpoint binding mismatch')
    model = build_model(cfg)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.to(device).eval(), manifest
