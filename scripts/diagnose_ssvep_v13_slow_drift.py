"""Read-only artificial v13 slow-drift structure diagnostic, with a frozen recipe.

Fixed attenuation counterfactuals are not an optimal oracle or a bound on all
possible models. Whole-record FFT is neither a deployable pipeline nor evidence
of physiological clean recovery. Import and --help need only the stdlib.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RECIPE_JSON = r'''{
  "schema_version": 1,
  "status": "formal",
  "name": "v13_slow_drift_fixed_attenuation_counterfactual",
  "seed": 2026101301,
  "randomness": "none; seed reserved and recorded, no stochastic draws",
  "sample_rate_hz": 250,
  "samples": 24371,
  "channels": 8,
  "units": "artificial arbitrary units (AU); not microvolts or physiological clean EEG",
  "clean": {
    "fundamental_hz": 10,
    "fundamental_amplitude_au": 1,
    "second_harmonic_hz": 20,
    "second_harmonic_amplitude_au": 0.25,
    "phase_per_channel_radians": [
      0,
      0.27,
      0.54,
      0.81,
      1.08,
      1.35,
      1.62,
      1.89
    ],
    "phase_applies_to": "both harmonics",
    "dc_offset_default_au": [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    "optional_dc_offsets_au": [
      0,
      10,
      20,
      30,
      40,
      50,
      60,
      70
    ]
  },
  "cases": [
    "identity",
    "linear_trend",
    "sin.01",
    "sin.03",
    "sin.1",
    "sin.3",
    "step"
  ],
  "artifact": {
    "rms_au": 3,
    "center": "per channel over entire record before RMS scaling",
    "linear_trend": "sample_index",
    "sin_frequencies_hz": {
      "sin.01": 0.01,
      "sin.03": 0.03,
      "sin.1": 0.1,
      "sin.3": 0.3
    },
    "sin_phase_radians": 0,
    "step": "0 before floor(N/2), 1 from floor(N/2); then center and RMS scale",
    "channels": "same deterministic artifact in all eight channels",
    "identity": "exactly zero artifact; RMS scaling not applied"
  },
  "notch": {
    "frequency_hz": 50,
    "q": 30,
    "phase": "offline zero phase",
    "implementation": "ssvep_v13_runtime.condition_line; float64 channel mean removed and restored"
  },
  "attenuation": {
    "zero_gain_band_hz": "0 < f < 8",
    "dc_gain": 1,
    "gain_at_or_above_8hz": 1,
    "mean": "float64 per channel removed and restored",
    "fft": "unwindowed rfft/irfft with actual transform length; no taper inside FFT",
    "block_samples": 4096,
    "hop_samples": 1024,
    "starts": "ssvep_v13_runtime.window_starts including original final tail window",
    "synthesis": "notch plus positive-Hann weighted residual OLA",
    "hann": "np.hanning(4098)[1:-1]",
    "whole_record": "same fixed gain at real N=24371 after same notch; structural offline comparison only",
    "arithmetic": "float64 fixed counterfactual; actual optional v13 retains its own float32 forward"
  },
  "regions": {
    "full": [
      0,
      24371
    ],
    "interior": [
      4096,
      20275
    ]
  },
  "tracks": [
    "raw",
    "fixed50Hznotch",
    "traditional",
    "fixed_sub8_block_OLA",
    "fixed_sub8_whole_record"
  ],
  "traditional": "eeg_tools.workstation.v9_offline._traditional(signal)",
  "joint_ls": {
    "frequencies_hz": [
      10,
      20
    ],
    "nuisance_terms": [
      "intercept",
      "linear trend"
    ],
    "phase_reference": "absolute sample time"
  },
  "selection": "no threshold tuning, candidate search, training, epoch selection or deployment"
}'''
RECIPE_SHA256 = '5725ab469e1552268ba2df2399557805d6b8757ee87e4e07ca6cf73aad86e68a'
SCRIPT = 'scripts/diagnose_ssvep_v13_slow_drift.py'
TEST = 'tests/test_diagnose_ssvep_v13_slow_drift.py'
FALSE_FLAGS = dict(runtime_enabled=False, production_validated=False,
                   source_domain_validated=False, independent_personal_validation=False,
                   clean_ground_truth_available=False, presented_matches_are_gaze_accuracy=False,
                   training_performed=False, checkpoint_selection_performed=False,
                   default_deployment_authorized=False, personal_data_used=False,
                   physiological_clean_ground_truth_available=False,
                   all_possible_models_ruled_out=False, optimal_oracle=False)
LIMITATIONS = [
    'Fixed attenuation counterfactual only; no optimal oracle, candidate selection or bound on all possible models.',
    'Per-window DC preservation and residual OLA can retain very slow backgrounds; results describe only this fixed recipe.',
    'Whole-record FFT is an offline structural comparison, not a deployable pipeline or physiological clean evidence.',
    'Artificial reference is known analytically in arbitrary units; there are no recorded EEG or personal data here.',
    'LS amplitude/phase estimates include an intercept and linear trend; this does not remove drift from waveform MSE.',
    'Float64 fixed attenuation differs from optional actual v13 float32 neural forward; no claim of numerical equivalence.',
    'No thresholds are fitted and no production, hardware, session-quality or clinical validation is granted.',
]


def require(ok, message):
    if not ok:
        raise ValueError('v13 slow drift: ' + message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def frozen_recipe():
    recipe = json.loads(RECIPE_JSON)
    canonical = json.dumps(recipe, sort_keys=True, separators=(',', ':'),
                           ensure_ascii=False, allow_nan=False).encode('utf-8')
    require(hashlib.sha256(canonical).hexdigest() == RECIPE_SHA256, 'embedded recipe hash mismatch')
    return recipe


def checked_signal(signal, *, minimum=4096):
    import numpy as np
    value = np.asarray(signal)
    require(value.dtype.kind == 'f' and value.ndim == 2 and value.shape[1] == 8
            and len(value) >= minimum and bool(np.isfinite(value).all()),
            'finite floating [T,8] input required')
    return value.astype(np.float64, copy=False)


def generate_case(case, *, channel_dc_offsets=False):
    """Clean is used here for artificial construction, and later only for scoring."""
    import numpy as np
    recipe = frozen_recipe()
    require(case in recipe['cases'], 'case outside frozen recipe')
    time = np.arange(recipe['samples'], dtype=np.float64) / recipe['sample_rate_hz']
    phases = np.asarray(recipe['clean']['phase_per_channel_radians'])
    clean = np.sin(2*np.pi*10*time[:, None] + phases)
    clean += 0.25*np.sin(2*np.pi*20*time[:, None] + phases)
    offsets = recipe['clean']['optional_dc_offsets_au' if channel_dc_offsets else 'dc_offset_default_au']
    clean += np.asarray(offsets)[None, :]
    if case == 'identity':
        artifact = np.zeros_like(time)
    else:
        if case == 'linear_trend':
            artifact = np.arange(len(time), dtype=np.float64)
        elif case == 'step':
            artifact = (np.arange(len(time)) >= len(time)//2).astype(np.float64)
        else:
            artifact = np.sin(2*np.pi*recipe['artifact']['sin_frequencies_hz'][case]*time)
        artifact -= artifact.mean(dtype=np.float64)
        artifact *= recipe['artifact']['rms_au'] / np.sqrt(np.mean(artifact**2))
    artifact = np.broadcast_to(artifact[:, None], clean.shape).copy()
    return clean, clean + artifact, artifact


def _fixed_sub8(signal):
    """Signal-only fixed gain: restore the float64 window mean, including DC."""
    import numpy as np
    value = checked_signal(signal)
    means = value.mean(axis=0, keepdims=True, dtype=np.float64)
    spectrum = np.fft.rfft(value - means, axis=0)
    frequencies = np.fft.rfftfreq(len(value), d=1/250)
    spectrum[(frequencies > 0) & (frequencies < 8)] = 0
    return np.fft.irfft(spectrum, n=len(value), axis=0) + means


def fixed_sub8_block_ola(signal):
    """Actual v13 notch and coverage; fixed signal-only attenuation per window."""
    import numpy as np
    from eeg_tools import ssvep_v13_runtime as api
    notch = api.condition_line(checked_signal(signal))
    starts = api.window_starts(len(notch))
    hann = np.hanning(4098)[1:-1]
    residual = np.zeros_like(notch, dtype=np.float64)
    weights = np.zeros(len(notch), dtype=np.float64)
    for start in starts:
        chunk = notch[start:start+4096]
        residual[start:start+4096] += (_fixed_sub8(chunk)-chunk)*hann[:, None]
        weights[start:start+4096] += hann
    require(bool(np.all(weights > 0)), 'Hann OLA has uncovered samples')
    output = notch + residual / weights[:, None]
    checked_signal(output)
    return output, dict(window_starts=starts, window_samples=4096, hop_samples=1024,
                        window_count=len(starts), final_window_start=starts[-1],
                        processed_samples=len(notch), minimum_coverage_weight=float(weights.min()),
                        all_samples_covered=True, arithmetic='float64',
                        synthesis='positive Hann residual OLA', **FALSE_FLAGS)


def fixed_sub8_whole_record(signal):
    from eeg_tools import ssvep_v13_runtime as api
    return _fixed_sub8(api.condition_line(checked_signal(signal)))


def load_optional_model(include_failed_v13, device, *, root=ROOT):
    require(device in ('cpu', 'cuda:0'), 'explicit cpu or cuda:0 required')
    require(include_failed_v13 or device == 'cpu', 'CUDA requires explicit --include-failed-v13')
    if not include_failed_v13:
        return None, dict(included=False, actual_device='cpu', checkpoint_role=None,
                          selected_epoch=None, **FALSE_FLAGS)
    from scripts import evaluate_ssvep_denoiser_v13 as ev
    model, metadata = ev.load_evaluation_model(root, device,
                                               allow_failed_selection_diagnostic=True)
    require(not model.training, 'model must already be in eval mode')
    actual = str(next(model.parameters()).device)
    require(actual == device, 'actual model device differs from request')
    require(metadata.get('checkpoint_role') == ev.DIAGNOSTIC_ROLE
            and metadata.get('selected_epoch') is None
            and metadata.get('selection_gate_passed') is False, 'unexpected model role')
    require(all(metadata.get(key) is False for key in ev.FALSE_FLAGS), 'model safety flags changed')
    return model, dict(included=True, actual_device=actual, metadata=metadata, **FALSE_FLAGS)


def signal_tracks(signal, model=None):
    """No clean reference, labels, scores or adaptive filter parameters enter here."""
    import numpy as np
    from eeg_tools import ssvep_v13_runtime as api
    from eeg_tools.workstation import v9_offline as baseline
    value = checked_signal(signal)
    original = value.copy()
    block, receipt = fixed_sub8_block_ola(value.copy())
    tracks = dict(raw=value.copy(), fixed50Hznotch=api.condition_line(value.copy()),
                  traditional=baseline._traditional(value.copy()),
                  fixed_sub8_block_OLA=block,
                  fixed_sub8_whole_record=fixed_sub8_whole_record(value.copy()))
    receipts = {'fixed_sub8_block_OLA': receipt}
    if model is not None:
        require(not model.training, 'model must be eval-only')
        output, actual_notch, actual_receipt = api.process_recording(value.copy(), model, batch_size=32)
        require(np.array_equal(actual_notch, tracks['fixed50Hznotch']), 'actual v13 notch mismatch')
        require(actual_receipt.get('runtime_enabled') is False
                and actual_receipt.get('production_validated') is False, 'actual runtime flags changed')
        tracks['failed_v13'] = output
        receipts['failed_v13'] = actual_receipt
    require(np.array_equal(value, original), 'signal input mutated')
    for output in tracks.values():
        require(checked_signal(output).shape == value.shape, 'output shape changed')
    return tracks, receipts


def measure_case(clean, tracks):
    from scripts import evaluate_ssvep_denoiser_v13 as ev
    recipe = frozen_recipe()
    require(checked_signal(clean).shape == (recipe['samples'], 8), 'reference outside recipe')
    require(set(tracks) in (set(recipe['tracks']), set(recipe['tracks']) | {'failed_v13'}),
            'track set outside recipe')
    regions = {}
    for region, (start, end) in recipe['regions'].items():
        truth = clean[start:end]
        reference_fit = ev.joint_sine_fit(truth, (10, 20), start_sample=start)
        measured = {}
        for name, value in tracks.items():
            require(checked_signal(value).shape == clean.shape, 'measurement shape mismatch')
            segment = value[start:end]
            errors = ev.waveform_errors(truth, segment)
            # Explicit names keep both NMSE definitions and the independent DC term visible.
            errors['centered_nmse'] = errors['clean_centered_normalized_mse']
            errors['uncentered_nmse'] = errors['uncentered_mse'] / errors['clean_centered_energy']
            fit = ev.joint_sine_fit(segment, (10, 20), start_sample=start)
            measured[name] = dict(errors=errors, joint_fit=fit,
                                  harmonics=ev.compare_fits(reference_fit, fit))
        notch = measured['fixed50Hznotch']['errors']
        for row in measured.values():
            row['relative_notch_mse_reduction'] = {
                key: 1-row['errors'][key]/notch[key] if notch[key] > 0 else None
                for key in ('clean_centered_mse', 'uncentered_mse', 'mean_offset_mse')}
        regions[region] = dict(start_sample=start, end_sample_exclusive=end,
                               artificial_clean_joint_fit=reference_fit, tracks=measured)
    return regions


def source_snapshot(*, root=ROOT, include_failed_v13=False):
    from eeg_tools import ssvep_v13_runtime as api
    names = {SCRIPT, TEST, 'AGENTS.md', 'eeg_tools/ssvep_v13_runtime.py',
             'scripts/evaluate_ssvep_denoiser_v13.py', 'eeg_tools/workstation/v9_offline.py',
             api.CONFIG, api.LOCK, api.PLAN}
    if include_failed_v13:
        cfg = api.load_config(root)
        names.update(api.bindings(cfg))
        names.update((api.MODEL, api.MANIFEST, api.REPORT))
        plan = json.loads(api.safe_path(root, api.PLAN).read_text(encoding='utf-8'))
        names.update(plan['prior_evidence'])
    return {name: sha(api.safe_path(root, name)) for name in sorted(names)}


def validate_output(output, *, root=ROOT):
    from eeg_tools import ssvep_v13_runtime as api
    base = Path(root).absolute()
    candidate = Path(output)
    candidate = candidate.absolute() if candidate.is_absolute() else base / candidate
    require(candidate.is_relative_to(base), 'output must be under workspace analysis_outputs')
    relative = candidate.relative_to(base).as_posix()
    require(relative.startswith('analysis_outputs/'), 'output must be a new analysis_outputs subdirectory')
    candidate = api.safe_path(base, relative, new=True)
    require(candidate.parent.is_dir(), 'output parent must already exist')
    return candidate


def write_json(path, value, *, atomic=False):
    path = Path(path)
    require(not os.path.lexists(path), 'refusing existing artifact: ' + str(path))
    pending = path.with_name(path.name + '.pending') if atomic else path
    with pending.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    if atomic:
        require(not os.path.lexists(path), 'report already exists')
        os.replace(pending, path)


def diagnose(output, *, include_failed_v13=False, device='cpu', save_arrays=False,
             channel_dc_offsets=False, root=ROOT):
    """Exclusive output ownership. report.json is the last, atomic completion marker."""
    recipe = frozen_recipe()  # Validate the preregistered hash before numerical work.
    folder = validate_output(output, root=root)
    before = source_snapshot(root=root, include_failed_v13=include_failed_v13)
    require(device == 'cpu' or include_failed_v13, 'CUDA requires explicit failed-model diagnostic')
    from eeg_tools import ssvep_v13_runtime as api
    cfg = api.load_config(root)
    require((cfg['sample_rate_hz'], cfg['channel_count'], cfg['window_samples'], cfg['hop_samples'])
            == (250, 8, 4096, 1024), 'frozen runtime dimensions differ')
    model, model_info = load_optional_model(include_failed_v13, device, root=root)
    folder.mkdir(exist_ok=False)
    options = dict(include_failed_v13=include_failed_v13, requested_device=device,
                   save_arrays=save_arrays, channel_dc_offsets=channel_dc_offsets)
    write_json(folder / 'recipe.json', dict(recipe=recipe, recipe_sha256=RECIPE_SHA256,
                                            run_options=options, **FALSE_FLAGS))
    import numpy as np
    import scipy
    cases, artifacts = [], {}
    for name in recipe['cases']:
        clean, signal, artifact = generate_case(name, channel_dc_offsets=channel_dc_offsets)
        tracks, receipts = signal_tracks(signal, model)
        regions = measure_case(clean, tracks)
        cases.append(dict(case=name, artifact_mean_per_channel_au=artifact.mean(axis=0).tolist(),
                          artifact_rms_per_channel_au=np.sqrt(np.mean(artifact**2, axis=0)).tolist(),
                          regions=regions, receipts=receipts))
        if save_arrays:
            filename = name + '.npz'
            with (folder / filename).open('xb') as stream:
                np.savez_compressed(stream, artificial_clean=clean, artifact=artifact, **tracks)
                stream.flush()
                os.fsync(stream.fileno())
            artifacts[filename] = sha(folder / filename)
    artifacts['recipe.json'] = sha(folder / 'recipe.json')
    after = source_snapshot(root=root, include_failed_v13=include_failed_v13)
    require(after == before, 'source/config/lock/weight SHA changed; report will not be committed')
    report = dict(schema_version=1, status='formal', diagnostic_kind='artificial_structure_only',
                  recipe=recipe, recipe_sha256=RECIPE_SHA256, run_options=options,
                  actual_device=model_info['actual_device'], model=model_info,
                  runtime_config_flags=dict(enabled=cfg['runtime']['enabled'],
                                            production_validated=cfg['runtime']['production_validated']),
                  cases=cases, units=recipe['units'], limitations=LIMITATIONS,
                  nmse_denominator='regional artificial clean centered energy for both centered and uncentered NMSE',
                  relative_reduction='1 - track MSE / fixed50Hznotch MSE in the same region; undefined denominator gives null',
                  provenance=dict(before_sha256=before, after_sha256=after, all_matched=True),
                  artifacts_sha256=artifacts,
                  environment=dict(python=platform.python_version(), numpy=np.__version__,
                                   scipy=scipy.__version__, torch_imported='torch' in sys.modules),
                  **FALSE_FLAGS)
    write_json(folder / 'report.json', report, atomic=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', required=True,
                        help='new exclusive subdirectory of analysis_outputs; parent must exist')
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cpu',
                        help='default CPU; CUDA only with explicit failed v13 track')
    parser.add_argument('--include-failed-v13', action='store_true',
                        help='explicit load_evaluation_model failed-selection diagnostic; eval only')
    parser.add_argument('--channel-dc-offsets', action='store_true',
                        help='use frozen per-channel offsets [0,10,...,70] AU')
    parser.add_argument('--save-arrays', action='store_true', help='also save one compressed NPZ per case')
    args = parser.parse_args(argv)
    try:
        report = diagnose(args.output_directory, include_failed_v13=args.include_failed_v13,
                          device=args.device, save_arrays=args.save_arrays,
                          channel_dc_offsets=args.channel_dc_offsets)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, str(exc) + '\n')
    print(json.dumps(dict(report=str(Path(args.output_directory) / 'report.json'),
                          recipe_sha256=report['recipe_sha256'], actual_device=report['actual_device'],
                          **FALSE_FLAGS), ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
