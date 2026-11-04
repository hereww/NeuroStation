"""V13 frozen, review-before-training experiment. Default/check-only is metadata only.

--freeze-only exclusively creates the lock. --run requires that lock, CUDA:0,
and empty artifact destinations. No personal or public7/8 data is opened here.
The checkpoint remains offline research only, even when selection gates pass.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from eeg_tools import ssvep_v13_runtime as api


def exclusive_json(root, name, value):
    encoded = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    target = api.safe_path(root, name, new=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    target = api.safe_path(root, name, new=True)
    with target.open('xb') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return target


def assert_new_outputs(root):
    for name in (api.MODEL, api.MANIFEST, api.REPORT):
        api.safe_path(root, name, new=True)


def freeze_only(root=ROOT):
    cfg = api.load_config(root)
    api.validate_plan(root, cfg)
    api.safe_path(root, api.LOCK, new=True)
    assert_new_outputs(root)
    files = {name: api.sha(api.safe_path(root, name)) for name in api.bindings(cfg)}
    lock = dict(schema_version=1, status='formal', model_version=cfg['model_version'],
                config_sha256=api.canonical_sha(cfg), research_plan_sha256=files[api.PLAN], files=files,
                roles={'fit': [1, 2, 3, 4], 'select': [5, 6], 'personal': [], 'evaluation': []},
                synthetic=cfg['synthetic'], normalization_contract=api.NORMALIZATION,
                runtime_enabled=False, production_validated=False,
                reference_disclosure='not physiological clean truth; synthetic numerical benchmark; public7/8 not pristine')
    exclusive_json(root, api.LOCK, lock)
    return api.preflight(root)


def check_only(root=ROOT):
    cfg = api.load_config(root)
    api.validate_plan(root, cfg)
    api.require(set(api.bindings(cfg)).isdisjoint((api.LOCK, api.MODEL, api.REPORT, api.MANIFEST)),
                'source/output collision')
    for name in api.bindings(cfg):
        api.require(api.safe_path(root, name).is_file(), 'missing source: ' + name)
    if api.safe_path(root, api.LOCK).exists():
        result = api.preflight(root)
    else:
        result = dict(status='ready_to_freeze', config_sha256=api.canonical_sha(cfg),
                      waveforms_loaded=False, torch_imported=False, cuda_initialized=False,
                      runtime_enabled=False, production_validated=False)
    assert_new_outputs(root)
    return result


def load_public_references(subjects, cfg, root=ROOT):
    """Read the existing Wearable schema, but use ONLY v13 config and lock.

    Reference scaling is performed once here; that reference coordinate system is
    then fixed. No noisy-window re-scaling of losses or target-side stats occurs.
    """
    import numpy as np
    api.require(subjects in (cfg['data']['fit_subjects'], cfg['data']['selection_subjects']),
                'only declared fit/select cohorts may be read by trainer')
    lock = json.loads(api.safe_path(root, api.LOCK).read_text(encoding='utf-8'))
    refs, identities = [], []
    for subject in subjects:
        name = cfg['data']['reference_directory'] + f'/subject_{subject:03d}.npz'
        filename = api.safe_path(root, name)
        api.require(api.sha(filename) == lock['files'][name]
                    and api.sha(api.safe_path(root, name + '.manifest.json')) == lock['files'][name + '.manifest.json'],
                    'public reference changed')
        manifest = json.loads(api.safe_path(root, name + '.manifest.json').read_text(encoding='utf-8'))
        api.require(manifest['subject_id'] == subject and manifest['dataset'] == 'Wearable'
                    and manifest['native_sample_rate_hz'] == 250
                    and manifest['raw_reference_is_clean_ground_truth'] is False, 'public provenance mismatch')
        with np.load(filename, allow_pickle=False) as archive:
            raw = np.asarray(archive['reference'], dtype=np.float64)
        api.require(raw.shape == (240, 710, 8) and np.isfinite(raw).all(), 'public reference layout mismatch')
        ref = api.condition_line(raw)[:, cfg['data']['window_start_sample']:cfg['data']['window_end_sample_exclusive'], :]
        ref = ref.transpose(0, 2, 1).copy()
        ref -= ref.mean(axis=-1, keepdims=True)
        scale = np.sqrt(np.mean(ref**2, axis=-1, keepdims=True))
        api.require(np.isfinite(scale).all() and np.all(scale > 1e-8), 'flat public reference channel')
        refs.append(ref / scale)
        identities.append(dict(subject_id=subject, count=len(ref), source_sha256=lock['files'][name],
                               coordinate_system='fixed per-reference RMS units; no physical microvolt claim'))
    return np.concatenate(refs), identities


def synthetic_references(count, seed, cfg):
    """Independent continuous 16.384s references, no labels/templates exported."""
    import numpy as np
    api.require(type(count) is int and count > 0 and type(seed) is int and seed >= 0,
                'positive synthetic count/nonnegative seed required')
    rng = np.random.default_rng(seed)
    n, fs = cfg['window_samples'], cfg['sample_rate_hz']
    t = np.arange(n, dtype=np.float64) / fs
    freq = np.fft.rfftfreq(n, 1 / fs)
    values = []
    for _ in range(count):
        fundamental = rng.uniform(*cfg['synthetic']['fundamental_hz'])
        phase = rng.uniform(0, 2 * np.pi, (8, 1))
        amplitude = rng.uniform(0.8, 1.2, (8, 1))
        reference = amplitude * np.sin(2 * np.pi * fundamental * t + phase)
        reference += rng.uniform(0.1, 0.3, (8, 1)) * np.sin(4 * np.pi * fundamental * t + 2 * phase)
        reference += 0.03 * np.sin(2 * np.pi * cfg['synthetic']['background_hz'] * t + rng.uniform(0, 2*np.pi, (8, 1)))
        # Power spectrum proportional to 1/f, low amplitude; DC added separately.
        spectrum = rng.normal(size=(8, len(freq))) + 1j * rng.normal(size=(8, len(freq)))
        spectrum[:, 0] = 0
        spectrum[:, 1:] /= np.sqrt(freq[None, 1:])
        background = np.fft.irfft(spectrum, n=n, axis=-1)
        background /= np.sqrt(np.mean(background**2, axis=-1, keepdims=True))
        reference += 0.02 * background + rng.uniform(-0.3, 0.3, (8, 1))
        values.append(reference)
    return api._windows(np.stack(values), 'synthetic reference')


def corrupt(reference, condition, seed, cfg):
    """Corruption is created OUTSIDE forward; only resulting samples reach model.

    Added artifacts have zero sample mean because protected DC is unidentifiable.
    Burst RMS is specified over its active segment, not over the entire window.
    """
    import numpy as np
    ref = api._windows(reference, 'reference')
    api.require(condition in api.CONDITIONS and type(seed) is int and seed >= 0, 'invalid corruption request')
    if condition == 'identity':
        return ref.copy()
    rng = np.random.default_rng(seed)
    count, channels, n = ref.shape
    t = np.arange(n, dtype=np.float64) / cfg['sample_rate_hz']
    rms = np.sqrt(np.mean((ref - ref.mean(-1, keepdims=True))**2, axis=-1, keepdims=True))
    api.require(np.all(rms > 1e-8), 'flat corruption reference')
    aug = cfg['augmentation']
    noise = np.zeros_like(ref)

    def centered_scaled(wave, amplitude):
        wave = wave - wave.mean(-1, keepdims=True)
        scale = np.sqrt(np.mean(wave**2, axis=-1, keepdims=True))
        api.require(np.all(scale > 1e-12), 'degenerate synthetic artifact')
        return wave / scale * rms * amplitude

    if condition in ('drift', 'combined'):
        frequency = rng.uniform(*aug['drift_hz'], (count, channels, 1))
        drift = np.sin(2*np.pi*frequency*t + rng.uniform(0, 2*np.pi, (count, channels, 1)))
        noise += centered_scaled(drift, rng.uniform(*aug['drift_rms'], (count, channels, 1)))
        trend = np.linspace(-1, 1, n)[None, None, :] + rng.uniform(-0.4, 0.4, (count, channels, 1)) * np.linspace(-1, 1, n)[None, None, :]**2
        noise += centered_scaled(trend, rng.uniform(*aug['trend_rms'], (count, channels, 1)))
    if condition in ('high_burst', 'combined'):
        for i in range(count):
            duration = max(4, int(n * rng.uniform(*aug['burst_duration_fraction'])))
            start = int(rng.integers(0, n - duration + 1))
            wave = np.sin(2*np.pi*rng.uniform(*aug['burst_hz'], (channels, 1))*t[start:start+duration]
                          + rng.uniform(0, 2*np.pi, (channels, 1))) * np.hanning(duration)[None, :]
            wave -= wave.mean(-1, keepdims=True)
            wave /= np.sqrt(np.mean(wave**2, axis=-1, keepdims=True))
            noise[i, :, start:start+duration] += wave * rms[i] * rng.uniform(*aug['burst_rms'], (channels, 1))
    if condition in ('gaussian', 'combined'):
        noise += centered_scaled(rng.normal(size=ref.shape), rng.uniform(*aug['gaussian_rms'], (count, channels, 1)))
    out = ref + noise
    api.require(np.isfinite(out).all(), 'corruption overflow')
    return out


def cases_for_domain(reference, seed, cfg):
    """Fixed-case selection or epoch-specific fit; seeds independent by role/domain."""
    import numpy as np
    child_seeds = np.random.SeedSequence(seed).generate_state(len(api.CONDITIONS))
    cases = {}
    for condition, child in zip(api.CONDITIONS, child_seeds):
        raw = corrupt(reference, condition, int(child), cfg)
        notch = api.condition_line(raw.transpose(0, 2, 1)).transpose(0, 2, 1).copy()
        cases[condition] = (raw, notch)
    return cases


def selection_summary(domains, cfg):
    """Mean over conditions WITHIN domain, then mean over the TWO domains."""
    api.require(set(domains) == {'public', 'long'}, 'both selection domains required')
    objective, gates = [], {}
    for name, metrics in domains.items():
        api.require(set(metrics['conditions']) == set(api.CONDITIONS), 'missing selection condition')
        identity = metrics['identity_normalized_change']
        api.require(math.isfinite(identity) and identity >= 0, 'invalid identity metric')
        gates[name] = identity <= cfg['selection']['identity_normalized_change_max']
        errors = [metrics['conditions'][c]['model_mse'] for c in api.CONDITIONS[1:]]
        api.require(all(math.isfinite(v) and v >= 0 for v in errors), 'invalid selection MSE')
        objective.append(sum(errors) / len(errors))
    return dict(balanced_corrupted_mse=sum(objective) / 2,
                identity_gates=gates, selection_gate_passed=all(gates.values()), domains=domains)


def preferred_candidate(metrics, previous_score):
    # Ineligible low-error epochs never displace a gated candidate; ties keep earliest.
    return metrics['selection_gate_passed'] and (previous_score is None
            or metrics['balanced_corrupted_mse'] < previous_score)


def evaluate_selection(model, references, cases, cfg):
    import numpy as np
    model.eval()
    domains = {}
    for name in ('public', 'long'):
        ref, conditions = references[name], {}
        identity_change = None
        for condition in api.CONDITIONS:
            raw, notch = cases[name][condition]
            output = api.process_windows(notch, model, cfg['training']['batch_size'])
            conditions[condition] = dict(raw_mse=float(np.mean((raw-ref)**2)),
                                          notch_mse=float(np.mean((notch-ref)**2)),
                                          model_mse=float(np.mean((output-ref)**2)),
                                          fixed_notch_mse_reduction=float(np.mean((raw-ref)**2)-np.mean((notch-ref)**2)),
                                          learned_mse_reduction=float(np.mean((notch-ref)**2)-np.mean((output-ref)**2)))
            if condition == 'identity':
                _, _, rms = api.normalize_noisy(notch)
                # Gate includes total output-to-reference change. Learned-only is also reported.
                identity_change = float(np.mean(((output-ref)/rms)**2))
                conditions[condition]['learned_only_normalized_change'] = float(np.mean(((output-notch)/rms)**2))
        domains[name] = dict(count=len(ref), window_samples=ref.shape[-1], conditions=conditions,
                             identity_normalized_change=identity_change)
    return selection_summary(domains, cfg)


def fit_epoch(model, optimizer, references, cases, cfg, epoch):
    import numpy as np
    import torch
    model.train()
    batch = cfg['training']['batch_size']
    rng = np.random.default_rng(np.random.SeedSequence([cfg['training']['seed'], epoch, 91]))
    orders = {name: rng.permutation(len(ref)*len(api.CONDITIONS)) for name, ref in references.items()}
    # Equal update counts, no tiling of waveforms: independently shuffle/cycle long CASES.
    steps = max(math.ceil(len(order) / batch) for order in orders.values())
    cursors = {name: 0 for name in references}
    total, updates = 0.0, 0
    for _ in range(steps):
        for name in ('public', 'long'):
            order, cursor = orders[name], cursors[name]
            if cursor >= len(order):
                order = orders[name] = rng.permutation(len(order))
                cursor = 0
            indices = order[cursor:cursor+batch]
            cursors[name] = cursor + len(indices)
            ref = references[name]
            which, rows = indices // len(ref), indices % len(ref)
            noisy = np.stack([cases[name][api.CONDITIONS[int(c)]][1][r] for c, r in zip(which, rows)])
            target = ref[rows]
            normalized, _, rms = api.normalize_noisy(noisy)
            x = torch.from_numpy(normalized).to('cuda:0')
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)  # signal ONLY, never metadata, condition, label, reference or gaze
            losses = api.restored_mse_torch(prediction, x, torch.from_numpy(noisy).to('cuda:0'),
                                             torch.from_numpy(rms).to('cuda:0'), torch.from_numpy(target).to('cuda:0'))
            weights = torch.as_tensor(np.where(which == 0, cfg['training']['identity_loss_weight'], 1.0),
                                      dtype=torch.float64, device='cuda:0')
            loss = (losses * weights).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())
            updates += 1
    return dict(weighted_restored_mse=total/updates, optimizer_updates=updates,
                equal_domain_updates=True)


def run_training(root=ROOT):
    cfg, metadata = api.load_config(root), api.preflight(root)
    assert_new_outputs(root)
    # This function is reachable only by explicit --run. No AMP, TF32 or CPU fallback.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import random
    import numpy as np
    import torch
    api.require(torch.cuda.is_available(), 'CUDA required; no implicit CPU training')
    torch.set_num_threads(cfg['training']['cpu_threads'])
    random.seed(cfg['training']['seed'])
    np.random.seed(cfg['training']['seed'])
    torch.manual_seed(cfg['training']['seed'])
    torch.cuda.manual_seed_all(cfg['training']['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = api.build_model(cfg).to(device='cuda:0', dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'],
                                  weight_decay=cfg['training']['weight_decay'])
    public_fit, fit_sources = load_public_references(cfg['data']['fit_subjects'], cfg, root)
    public_select, select_sources = load_public_references(cfg['data']['selection_subjects'], cfg, root)
    fit = {'public': public_fit, 'long': synthetic_references(cfg['synthetic']['fit_count'], cfg['synthetic']['fit_seed'], cfg)}
    selection = {'public': public_select, 'long': synthetic_references(cfg['synthetic']['selection_count'], cfg['synthetic']['selection_seed'], cfg)}
    select_cases = {name: cases_for_domain(ref, [cfg['training']['seed'], 2, index], cfg)
                    for index, (name, ref) in enumerate(selection.items())}
    history, best_state, best_metric, selected_epoch = [], None, None, None
    for epoch in range(1, cfg['training']['epochs'] + 1):
        fit_cases = {name: cases_for_domain(ref, [cfg['training']['seed'], 1, index, epoch], cfg)
                     for index, (name, ref) in enumerate(fit.items())}
        fit_metrics = fit_epoch(model, optimizer, fit, fit_cases, cfg, epoch)
        metrics = evaluate_selection(model, selection, select_cases, cfg)
        row = dict(epoch=epoch, fit=fit_metrics, selection=metrics)
        print(json.dumps(row, allow_nan=False), flush=True)
        history.append(row)
        if preferred_candidate(metrics, best_metric):
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_metric, selected_epoch = metrics['balanced_corrupted_mse'], epoch
    final_metadata = api.preflight(root)
    api.require(final_metadata == metadata, 'frozen sources changed during training')
    eligible = best_state is not None
    if not eligible:
        best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    for value in best_state.values():
        api.require(bool(torch.isfinite(value).all()), 'nonfinite saved parameter')
    common = dict(schema_version=1, status='formal', model_version=cfg['model_version'],
                  config_sha256=metadata['config_sha256'], training_lock_sha256=metadata['training_lock_sha256'],
                  research_plan_sha256=api.sha(api.safe_path(root, api.PLAN)),
                  selected_epoch=selected_epoch, selection_gate_passed=eligible,
                  runtime_enabled=False, production_validated=False)
    report = {**common, 'status': 'training_completed', 'parameter_count': sum(p.numel() for p in model.parameters()),
              'training_device': 'cuda:0', 'device_name': torch.cuda.get_device_name(0),
              'torch_version': str(torch.__version__), 'epochs': cfg['training']['epochs'],
              'normalization_contract': api.NORMALIZATION, 'amp': False, 'allow_tf32': False,
              'deterministic_algorithms': True, 'reference_disclosure': cfg['design'],
              'public_sources': {'fit': fit_sources, 'select': select_sources},
              'synthetic': cfg['synthetic'], 'history': history,
              'selected_metrics': history[selected_epoch-1]['selection'] if eligible else None,
              'checkpoint_role': 'selected_research' if eligible else 'diagnostic_last_epoch_no_eligible_checkpoint',
              'diagnostic_epoch': None if eligible else cfg['training']['epochs'],
              'personal_data_used': False, 'public7_8_used': False,
              'public7_8_is_pristine': False, 'physiological_clean_truth': False,
              'long_synthetic_is_artificial_numerical_benchmark': True,
              'post_training_research_checks_evaluated': False,
              'post_training_research_checks': api.validate_plan(root, cfg)['fixed_research_checks'],
              'selection_gate_authorizes_default_deployment': False}
    assert_new_outputs(root)
    # Exclusive files and fsync: failures leave reviewable incomplete artifacts;
    # reruns refuse partial artifacts instead of silently overwriting them.
    model_path = api.safe_path(root, api.MODEL, new=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path = api.safe_path(root, api.MODEL, new=True)
    with model_path.open('xb') as stream:
        torch.save({**common, 'config': cfg, 'state_dict': best_state}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    report_path = exclusive_json(root, api.REPORT, report)
    manifest = {**common, 'checkpoint_sha256': api.sha(model_path),
                'training_report_sha256': api.sha(report_path), 'parameter_count': report['parameter_count'],
                'training_device': report['training_device'], 'device_name': report['device_name'],
                'checkpoint_role': report['checkpoint_role'], 'normalization_contract': api.NORMALIZATION}
    exclusive_json(root, api.MANIFEST, manifest)
    return dict(status='training_completed', selected_epoch=selected_epoch, selection_gate_passed=eligible,
                checkpoint=api.MODEL, manifest=api.MANIFEST, report=api.REPORT,
                runtime_enabled=False, production_validated=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--check-only', action='store_true')
    group.add_argument('--freeze-only', action='store_true')
    group.add_argument('--run', action='store_true')
    args = parser.parse_args(argv)
    try:
        result = run_training() if args.run else freeze_only() if args.freeze_only else check_only()
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({'status': 'rejected', 'error': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
