"""Read-only, pre-clip loss gradients at the SHA-bound failed v13 final state.

No training, optimizer trajectory, epoch search, gate or physiological validation.
Only frozen fit1-4/select5-6 and artificial long references are used. Import/help
require only the standard library. A completed report is the last atomic write.
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
SCRIPT = 'scripts/diagnose_ssvep_v13_loss_conflict.py'
TEST = 'tests/test_diagnose_ssvep_v13_loss_conflict.py'
EVALUATOR = 'scripts/evaluate_ssvep_denoiser_v13.py'
TARGETS = ('public_identity', 'public_drift', 'long_identity', 'long_drift')
FALSE_FLAGS = dict(runtime_enabled=False, production_validated=False,
                   source_domain_validated=False, independent_personal_validation=False,
                   physiological_validation=False, physiological_clean_truth=False,
                   clean_ground_truth_available=False, clinical_validated=False,
                   training_performed=False, epoch_selection_performed=False,
                   optimizer_trajectory_reproduced=False, actual_adamw_update_measured=False,
                   public7_8_used=False, personal_data_used=False,
                   gate_evaluated=False, diagnostic_authorizes_deployment=False)


def require(ok, message):
    if not ok:
        raise ValueError('v13 loss diagnostic: ' + message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def uniform_indices(count, selected=32):
    """Integer floor of linspace, fixed before looking at any waveforms/losses."""
    require(type(count) is int and type(selected) is int and count >= selected >= 2,
            'invalid sampling counts')
    return [i * (count - 1) // (selected - 1) for i in range(selected)]


def sampling_recipe(cfg):
    require(cfg['data']['fit_subjects'] == [1, 2, 3, 4]
            and cfg['data']['selection_subjects'] == [5, 6], 'fit/select cohorts changed')
    require(cfg['training']['epochs'] == 16
            and cfg['training']['identity_loss_weight'] == 100, 'frozen epoch/weight changed')
    require(cfg['synthetic']['fit_count'] == 96
            and cfg['synthetic']['selection_count'] == 32, 'long counts changed')
    result = {}
    for role, subjects, role_index, count_key, seed_key in (
            ('fit', [1, 2, 3, 4], 1, 'fit_count', 'fit_seed'),
            ('select', [5, 6], 2, 'selection_count', 'selection_seed')):
        local = uniform_indices(240)
        public = [subject_index * 240 + row
                  for subject_index in range(len(subjects)) for row in local]
        seed = cfg['training']['seed']
        def case_seed(domain_index):
            return [seed, role_index, domain_index, 16] if role == 'fit' else [seed, role_index, domain_index]
        result[role] = {
            'diagnostic_role': 'last_epoch_fit_cases' if role == 'fit' else 'frozen_selection_cases',
            'public': {'subjects': subjects, 'full_count': 240 * len(subjects),
                       'indices_per_subject': local, 'indices': public, 'selected_count': len(public),
                       'window_samples': 512, 'case_seed': case_seed(0)},
            'long': {'full_count': cfg['synthetic'][count_key],
                     'reference_seed': cfg['synthetic'][seed_key],
                     'indices': uniform_indices(cfg['synthetic'][count_key]),
                     'selected_count': 32, 'window_samples': 4096, 'case_seed': case_seed(1)},
        }
    return result


def create_output_directory(output, root=ROOT):
    """Reject reuse, traversal and reparse links before scientific imports."""
    from eeg_tools import ssvep_v13_runtime as api
    base = Path(root).absolute()
    candidate = Path(output)
    if candidate.is_absolute():
        require(candidate.is_relative_to(base), 'output outside workspace')
        name = candidate.relative_to(base).as_posix()
    else:
        name = candidate.as_posix()
    path = api.safe_path(base, name, new=True)
    analysis = api.safe_path(base, 'analysis_outputs')
    require(path != analysis and path.is_relative_to(analysis),
            'output must be a new directory under analysis_outputs')
    require(path.parent.is_dir(), 'output parent must already exist')
    path.mkdir(exist_ok=False)
    return path


def atomic_report(path, report):
    path = Path(path)
    require(not os.path.lexists(path), 'report already exists')
    pending = path.with_name(path.name + '.pending')
    with pending.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    require(not os.path.lexists(path), 'report already exists at commit')
    os.replace(pending, path)


def source_hashes(root, cfg):
    from eeg_tools import ssvep_v13_runtime as api
    names = set(api.bindings(cfg)) | {api.LOCK, api.MODEL, api.MANIFEST, api.REPORT,
                                     EVALUATOR, SCRIPT, TEST}
    plan = json.loads(api.safe_path(root, api.PLAN).read_text(encoding='utf-8'))
    names.update(plan['prior_evidence'])
    return {name: sha(api.safe_path(root, name)) for name in sorted(names)}


def array_sha(array):
    import numpy as np
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(json.dumps({'dtype': value.dtype.str, 'shape': list(value.shape)},
                             sort_keys=True).encode('utf-8'))
    digest.update(memoryview(value).cast('B'))
    return digest.hexdigest()


def tensor_sha(tensor):
    import torch
    value = tensor.detach().cpu().contiguous()
    header = json.dumps({'dtype': str(value.dtype), 'shape': list(value.shape)}, sort_keys=True)
    digest = hashlib.sha256(header.encode('utf-8'))
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def snapshot_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def assert_read_only_state(model, before):
    import torch
    after = model.state_dict()
    require(list(after) == list(before), 'model state keys changed')
    for name, old in before.items():
        new = after[name].detach().cpu()
        require(old.dtype == new.dtype and old.shape == new.shape
                and torch.equal(old, new) and tensor_sha(old) == tensor_sha(new),
                'model state tensor changed: ' + name)
    require(all(not module.training for module in model.modules()), 'model left eval mode')
    require(all(parameter.grad is None for parameter in model.parameters()), 'parameter .grad mutated')
    return {name: tensor_sha(value) for name, value in after.items()}


def mean_loss_gradients(model, batches, per_sample_loss):
    """Accumulate batch SUM gradients in float64, divide by actual sample count.

    autograd.grad leaves .grad untouched. The callback receives a batch and must
    return one loss per sample; model remains eval with autograd enabled.
    """
    import torch
    require(all(not module.training for module in model.modules()), 'eval model required')
    named = list(model.named_parameters())
    require(named and all(parameter.grad is None for _, parameter in named), 'fresh parameter grads required')
    active = [(name, parameter) for name, parameter in named if parameter.requires_grad]
    totals = {name: torch.zeros_like(parameter, device='cpu', dtype=torch.float64)
              for name, parameter in named}
    count, loss_sum, records, unused = 0, 0.0, [], set()
    with torch.enable_grad():
        for batch in batches:
            losses = per_sample_loss(batch)
            require(isinstance(losses, torch.Tensor) and losses.ndim == 1 and len(losses) > 0
                    and bool(torch.isfinite(losses).all()), 'finite per-sample losses required')
            size = len(losses)
            gradients = torch.autograd.grad(losses.sum(), [p for _, p in active], allow_unused=True)
            for (name, _), gradient in zip(active, gradients):
                if gradient is None:
                    unused.add(name)
                    continue
                require(bool(torch.isfinite(gradient).all()), 'nonfinite gradient: ' + name)
                totals[name].add_(gradient.detach().cpu().double())
            value = float(losses.detach().double().sum().cpu())
            count += size
            loss_sum += value
            records.append({'sample_count': size, 'original_unit_mse': value / size})
    require(count > 0, 'no samples')
    result = {name: gradient / count for name, gradient in totals.items()}
    require(math.isfinite(loss_sum) and all(bool(torch.isfinite(g).all()) for g in result.values()),
            'aggregate overflow')
    return result, {'sample_count': count, 'batch_count': len(records),
                    'original_unit_mse': loss_sum / count, 'batches': records,
                    'unused_in_at_least_one_batch': sorted(unused),
                    'non_differentiable_parameters': [name for name, p in named if not p.requires_grad]}


def gradient_relation(lhs, rhs):
    """Zero vectors have undefined cosine (null), never fabricated alignment."""
    import numpy as np
    a, b = np.asarray(lhs, dtype=np.float64).reshape(-1), np.asarray(rhs, dtype=np.float64).reshape(-1)
    require(a.shape == b.shape and a.size > 0, 'gradient layout mismatch')
    result = dict(finite=False, status='nonfinite', dot_product=None, cosine=None,
                  lhs_norm=None, rhs_norm=None)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return result
    am, bm = float(np.max(np.abs(a))), float(np.max(np.abs(b)))
    na = 0.0 if am == 0 else am * float(np.linalg.norm(a / am))
    nb = 0.0 if bm == 0 else bm * float(np.linalg.norm(b / bm))
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        dot = float(np.dot(a, b))
    result.update(lhs_norm=na if math.isfinite(na) else None,
                  rhs_norm=nb if math.isfinite(nb) else None,
                  dot_product=dot if math.isfinite(dot) else None)
    if not all(math.isfinite(v) for v in (na, nb, dot)):
        result['status'] = 'numeric_overflow'
        return result
    result['finite'] = True
    if na == 0 or nb == 0:
        result['status'] = 'both_zero' if na == nb == 0 else 'lhs_zero' if na == 0 else 'rhs_zero'
        return result
    unit_a, unit_b = a / am, b / bm
    cosine = float(np.dot(unit_a / np.linalg.norm(unit_a), unit_b / np.linalg.norm(unit_b)))
    result.update(status='ok', cosine=max(-1.0, min(1.0, cosine)))
    return result


def flatten_gradients(gradients):
    import numpy as np
    return np.concatenate([value.numpy().reshape(-1) for value in gradients.values()])


def gradient_summary(gradients, weight):
    import numpy as np
    flat = flatten_gradients(gradients)
    relation = gradient_relation(flat, flat)
    require(relation['finite'], 'nonfinite aggregate gradient norm')
    parameters, layers = {}, {}
    for name, value in gradients.items():
        vector = value.numpy().reshape(-1)
        norm = gradient_relation(vector, vector)['lhs_norm']
        require(norm is not None and math.isfinite(norm), 'nonfinite parameter norm')
        parameters[name] = dict(shape=list(value.shape), count=value.numel(),
                                plain_norm=norm, weighted_norm=weight * norm,
                                gradient_sha256=tensor_sha(value))
        layer = name.rsplit('.', 1)[0] if '.' in name else '(root)'
        layers.setdefault(layer, []).append(vector)
    layer_norms = {}
    for name, vectors in layers.items():
        vector = np.concatenate(vectors)
        norm = gradient_relation(vector, vector)['lhs_norm']
        layer_norms[name] = dict(plain_norm=norm, weighted_norm=weight * norm)
    return dict(plain_gradient_norm=relation['lhs_norm'], loss_weight=weight,
                weighted_gradient_norm=weight * relation['lhs_norm'],
                parameters=parameters, layers=layer_norms)


def band_energy(noisy, restored, sample_rate=250):
    """One-sided Parseval MSE, channel/window mean, original reference units."""
    import numpy as np
    x, y = np.asarray(noisy, dtype=np.float64), np.asarray(restored, dtype=np.float64)
    require(x.shape == y.shape and x.ndim == 3 and x.shape[-1] >= 2
            and np.isfinite(x).all() and np.isfinite(y).all(), 'invalid spectrum inputs')
    n = x.shape[-1]
    frequency = np.fft.rfftfreq(n, 1 / sample_rate)
    # Boundary 40 belongs to protected 8-40, and DC is separate from 0-1.
    masks = {'dc': frequency == 0, '0_1': (frequency > 0) & (frequency < 1),
             '1_8': (frequency >= 1) & (frequency < 8),
             '8_40': (frequency >= 8) & (frequency <= 40),
             '40_125': (frequency > 40) & (frequency <= 125)}
    require(bool(np.all(sum(mask.astype(int) for mask in masks.values()) == 1)),
            'bands must cover every bin once')
    weights = np.full(len(frequency), 2.0)
    weights[0] = 1.0
    if n % 2 == 0:
        weights[-1] = 1.0
    delta = np.fft.rfft(y, axis=-1) - np.fft.rfft(x, axis=-1)
    energies = np.mean(np.abs(delta)**2, axis=(0, 1)) * weights / n**2
    result = {name: float(energies[mask].sum()) for name, mask in masks.items()}
    result['outside_protected'] = result['0_1'] + result['1_8'] + result['40_125']
    result['total'] = sum(result[name] for name in masks)
    require(all(math.isfinite(value) for value in result.values()), 'spectral energy overflow')
    return result


def domain_cases(root, cfg, role, domain, recipe):
    """Always corrupt the FULL original domain before fixed subsampling."""
    import numpy as np
    from scripts import train_ssvep_denoiser_v13 as trainer
    spec = recipe[role][domain]
    if domain == 'public':
        reference, sources = trainer.load_public_references(spec['subjects'], cfg, root)
    else:
        reference = trainer.synthetic_references(spec['full_count'], spec['reference_seed'], cfg)
        sources = []
    require(reference.shape == (spec['full_count'], 8, spec['window_samples']), 'reference shape mismatch')
    complete = trainer.cases_for_domain(reference, spec['case_seed'], cfg)
    # Record child seeds too, in the frozen CONDITIONS order, without changing generation.
    from eeg_tools import ssvep_v13_runtime as api
    children = np.random.SeedSequence(spec['case_seed']).generate_state(len(api.CONDITIONS))
    hashes = {'full_reference_sha256': array_sha(reference), 'public_sources': sources,
              'child_seeds': {name: int(seed) for name, seed in zip(api.CONDITIONS, children)},
              'full_cases': {name: {'raw_sha256': array_sha(complete[name][0]),
                                    'notch_sha256': array_sha(complete[name][1])}
                             for name in ('identity', 'drift')}}
    indices = spec['indices']
    selected = reference[indices].copy()
    cases = {name: complete[name][1][indices].copy() for name in ('identity', 'drift')}
    hashes['selected_reference_sha256'] = array_sha(selected)
    hashes['selected_notch_sha256'] = {name: array_sha(value) for name, value in cases.items()}
    return selected, cases, hashes


def measure_target(model, noisy, reference, batch_size, cfg):
    import numpy as np
    import torch
    from eeg_tools import ssvep_v13_runtime as api
    require(type(batch_size) is int and batch_size > 0, 'positive batch size required')
    require(noisy.shape == reference.shape, 'case/reference mismatch')
    device = next(model.parameters()).device
    spectrum_sum = {}
    def batches():
        for start in range(0, len(reference), batch_size):
            yield noisy[start:start + batch_size], reference[start:start + batch_size]
    def losses(batch):
        chunk, target = batch
        normalized, _, rms = api.normalize_noisy(chunk)
        x = torch.from_numpy(normalized).to(device)
        prediction = model(x)  # Signal only. Reference never selects or changes output.
        values = api.restored_mse_torch(prediction, x, torch.from_numpy(chunk).to(device),
                                        torch.from_numpy(rms).to(device), torch.from_numpy(target).to(device))
        output = api.restore_residual(chunk, normalized, prediction.detach().cpu().numpy(), rms)
        energy = band_energy(chunk, output, cfg['sample_rate_hz'])
        for name, value in energy.items():
            spectrum_sum[name] = spectrum_sum.get(name, 0.0) + len(chunk) * value
        return values
    gradients, metrics = mean_loss_gradients(model, batches(), losses)
    metrics['spectral_input_output_squared_difference'] = {
        name: value / metrics['sample_count'] for name, value in spectrum_sum.items()}
    metrics['selected_input_sha256'] = array_sha(noisy)
    return gradients, metrics


def role_relations(gradients, summaries):
    relations = {}
    for index, lhs in enumerate(TARGETS):
        for rhs in TARGETS[index + 1:]:
            measured = gradient_relation(flatten_gradients(gradients[lhs]), flatten_gradients(gradients[rhs]))
            weight = summaries[lhs]['loss_weight'] * summaries[rhs]['loss_weight']
            measured['weighted_dot_product'] = (None if measured['dot_product'] is None
                                                  else weight * measured['dot_product'])
            measured['cosine_unchanged_by_positive_identity_weight'] = True
            relations[lhs + '__' + rhs] = measured
    pi, ld = summaries['public_identity'], summaries['long_drift']
    denominator = ld['plain_gradient_norm']
    return {'pairs': relations,
            'public_identity_to_long_drift_plain_norm_ratio': None if denominator == 0 else pi['plain_gradient_norm'] / denominator,
            'public_identity_to_long_drift_weighted_norm_ratio': None if denominator == 0 else pi['weighted_gradient_norm'] / denominator,
            'public_identity_vs_long_drift': relations['public_identity__long_drift']}


def save_gradient_archive(path, gradients):
    import numpy as np
    arrays, layout = {}, {}
    for role, objectives in gradients.items():
        for target, parameters in objectives.items():
            for name, gradient in parameters.items():
                key = role + '__' + target + '__' + name
                arrays[key] = gradient.numpy()
                layout[key] = {'role': role, 'target': target, 'parameter': name,
                               'shape': list(gradient.shape), 'dtype': 'float64',
                               'gradient_kind': 'plain_sample_mean_pre_clip',
                               'array_sha256': array_sha(arrays[key])}
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    return {'filename': Path(path).name, 'sha256': sha(path), 'layout': layout}


def diagnose(output_directory, *, device='cpu', batch_size=8, save_gradients=False,
             allow_failed_selection_diagnostic=False, root=ROOT):
    require(allow_failed_selection_diagnostic is True, 'explicit --allow-failed-selection-diagnostic required')
    require(device in ('cpu', 'cuda:0'), 'explicit cpu or cuda:0 required')
    require(type(batch_size) is int and batch_size > 0, 'positive batch size required')
    from eeg_tools import ssvep_v13_runtime as api
    from scripts import evaluate_ssvep_denoiser_v13 as evaluator
    output = create_output_directory(output_directory, root)
    cfg = api.load_config(root)
    recipe = sampling_recipe(cfg)  # Fixed before loading model or numerical data.
    before = source_hashes(root, cfg)
    model, metadata = evaluator.load_evaluation_model(
        root, device, allow_failed_selection_diagnostic=True)
    require(metadata.get('checkpoint_role') == evaluator.DIAGNOSTIC_ROLE
            and metadata.get('diagnostic_epoch') == 16 and metadata.get('selected_epoch') is None
            and metadata.get('selection_gate_passed') is False, 'failed final state required')
    state = snapshot_state(model)
    state_before = assert_read_only_state(model, state)
    import torch
    import numpy as np
    require(not any(isinstance(module, torch.nn.modules.dropout._DropoutNd) for module in model.modules()),
            'frozen architecture unexpectedly contains dropout')
    reports, all_gradients = {}, {}
    for role in ('fit', 'select'):
        objectives, metrics, data_hashes = {}, {}, {}
        for domain in ('public', 'long'):
            reference, cases, hashes = domain_cases(root, cfg, role, domain, recipe)
            data_hashes[domain] = hashes
            for condition in ('identity', 'drift'):
                target = domain + '_' + condition
                gradients, values = measure_target(model, cases[condition], reference, batch_size, cfg)
                weight = cfg['training']['identity_loss_weight'] if condition == 'identity' else 1
                values.update(gradient_summary(gradients, weight))
                values['weighted_original_unit_mse'] = weight * values['original_unit_mse']
                objectives[target], metrics[target] = gradients, values
            del reference, cases
        reports[role] = {'diagnostic_role': recipe[role]['diagnostic_role'], 'targets': metrics,
                         'relations': role_relations(objectives, metrics), 'data_sha256': data_hashes}
        all_gradients[role] = objectives
    archive = save_gradient_archive(output / 'gradients.npz', all_gradients) if save_gradients else None
    state_after = assert_read_only_state(model, state)
    after = source_hashes(root, cfg)
    require(before == after, 'source files changed during diagnostic')
    require(state_before == state_after, 'model state hashes changed')
    report = {
        'schema_version': 1, 'status': 'formal', 'diagnostic_status': 'completed_read_only',
        'diagnostic_kind': 'local_pre_clip_plain_loss_gradient_at_failed_final_checkpoint',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'checkpoint': metadata,
        'device': device, 'batch_size': batch_size, 'optimizer_steps': 0,
        'normalization_contract': api.NORMALIZATION,
        'units': 'original fixed reference coordinates; public per-reference RMS units, not microvolts',
        'gradient_contract': {'loss': 'api.restored_mse_torch per-window channel/time mean',
                              'aggregation': 'sum batch loss gradients / total samples; float64 accumulation',
                              'parameter_order': list(next(iter(all_gradients['fit'].values()))),
                              'plain_gradient': 'pre-clip; no weight decay, optimizer, Adam moments or updates',
                              'identity_weight': 100, 'drift_weight': 1,
                              'weighted_norms_and_dots_only': True,
                              'positive_weight_does_not_change_cosine': True,
                              'model_mode': 'eval', 'dropout_present': False,
                              'train_eval_forward_equivalence': 'frozen Conv1d/SiLU/FFT architecture has no dropout or running-state normalization; eval retained'},
        'subsampling': {'diagnostic_only_not_gate': True,
                        'indices_fixed_before_waveforms_or_losses': True,
                        'rule': 'floor(i*(N-1)/31), i=0..31, includes both endpoints; public applied independently per subject',
                        'full_case_generation_before_subsampling': True,
                        'fit_noise_epoch': 16, 'selection_noise_epoch': None, 'recipe': recipe},
        'spectrum_contract': {'input': 'notch-conditioned noisy model input',
                              'output': 'same restored residual output as runtime',
                              'quantity': 'one-sided Parseval mean squared input-output difference in original units',
                              'bands_hz': {'dc': '{0}', '0_1': '(0,1)', '1_8': '[1,8)',
                                           '8_40': '[8,40]', '40_125': '(40,125]'},
                              'non_overlapping': True, 'protected': 'DC and [8,40]',
                              'outside_protected': '0_1 + 1_8 + 40_125',
                              'window_only_not_whole_record_ola': True},
        'roles': reports, 'gradient_archive': archive,
        'source_files_sha256_before': before, 'source_files_sha256_after': after,
        'model_state_tensor_sha256_before': state_before, 'model_state_tensor_sha256_after': state_after,
        'model_state_dict_unchanged_per_tensor': True, 'source_files_unchanged': True,
        'software': {'torch': str(torch.__version__), 'numpy': str(np.__version__),
                     'inference_settings': evaluator.inference_settings()},
        'interpretation_limits': [
            'Negative cosine is a local first-order conflict between these sampled numerical losses at this fixed state.',
            'Positive/zero cosine is not proof of global compatibility; cancellation and parameter scale affect aggregate geometry.',
            'Public recorded proxies are not clean physiological truth; long drift/reference pairs are artificial.',
            'No gradient relation, MSE or protected-band preservation establishes physiological or clinical correctness.',
            'Multiple batches form one fixed subset mean; these are not independent replicates or evidence for an entire cohort.',
            'This diagnostic does not reproduce optimizer history, clipped gradients or actual AdamW updates, and cannot promote a failed checkpoint.',
        ], **FALSE_FLAGS,
    }
    atomic_report(output / 'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', required=True, help='New directory under analysis_outputs; parent must exist')
    parser.add_argument('--allow-failed-selection-diagnostic', action='store_true',
                        help='Required explicit consent to measure the fully SHA-verified failed final state')
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cpu')
    parser.add_argument('--batch-size', type=int, default=8, help='Memory batching only; does not change indices/seeds')
    parser.add_argument('--save-gradients', action='store_true', help='Save complete plain mean gradients as SHA-bound NPZ')
    args = parser.parse_args(argv)
    if not args.allow_failed_selection_diagnostic:
        parser.error('--allow-failed-selection-diagnostic is required')
    if args.batch_size <= 0:
        parser.error('--batch-size must be positive')
    try:
        result = diagnose(args.output_directory, device=args.device, batch_size=args.batch_size,
                          save_gradients=args.save_gradients,
                          allow_failed_selection_diagnostic=args.allow_failed_selection_diagnostic)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, str(exc) + '\n')
    print(json.dumps({'report': str(Path(args.output_directory).absolute() / 'report.json'),
                      'diagnostic_status': result['diagnostic_status'], **FALSE_FLAGS}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
