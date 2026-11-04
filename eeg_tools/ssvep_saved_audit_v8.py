"""Independent standard-library audit of saved v8 INTERNAL diagnostic JSON.

No torch, numpy, production metrics/gates, model inference, NPZ or PT input.
Recomputes saved scalar arithmetic only; does NOT prove original phase values,
model correctness, source binary provenance or independent research acceptance.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import stat
import statistics
import tempfile
from typing import Any

ROOT = Path(__file__).absolute().parents[1]
MODEL_VERSION = "learned_denoise_v8_ssvep"
CONFIG_RELATIVE = "configs/learned_denoiser_v8_ssvep.json"
CONFIG_SHA256 = "484950220254e0b2c256d8cea2bf4f5d8f9084fe7aa4616f406d5e240553a16a"
PREREG_SHA256 = "eb15e295c70d7c845d4e8f27b069e923d925fca64da6a9a24f1fa61dc5ff2bb3"
REPORT_RELATIVE = "analysis_outputs/ssvep_v8_internal_diagnostic_20261008.json"
DEFAULT_OUTPUT_PREFIX = "analysis_outputs/ssvep_v8_saved_internal_audit_20261008"
AUDIT_VERSION = "ssvep_saved_internal_json_audit_v8"
AUDIT_SOURCE_PATHS = (
    "eeg_tools/ssvep_saved_audit_v8.py", "scripts/audit_ssvep_saved_internal_v8.py",
    "tests/test_ssvep_saved_audit_v8.py",
)
DATASETS = ("Nakanishi2015", "Wearable")
SUBJECTS = (5, 6)
CONDITIONS = ("identity", "transient", "low_frequency_drift", "high_frequency_burst", "common_mode")
TRIALS = {"Nakanishi2015": 180, "Wearable": 240}
RATES = {"Nakanishi2015": 256, "Wearable": 250}
STARTS = {"Nakanishi2015": 38, "Wearable": 125}
CASE_COUNT, REFERENCE_TRIAL_COUNT = 4200, 840
SEED, WINDOW = 13026100808, 256
# Arithmetic comparison tolerance only. Never applied to acceptance thresholds.
REL_TOL, ABS_TOL = 1e-12, 1e-15
MAX_JSON_BYTES = 64 * 1024 * 1024
CASE_FIELDS = frozenset((
    "dataset", "subject_id", "trial_index", "condition", "seed", "window_samples",
    "window_start_sample", "sample_rate_hz", "input_normalized_mse", "output_normalized_mse",
    "relative_reduction", "identity_normalized_change", "outcome", "protected_complex_energy_ratio",
    "protected_weighted_abs_phase_radians", "undefined_phase_bin_count",
))
COHORT_FIELDS = ("complete_cohort", "case_count", "reference_trial_count", "phase_case_count",
                 "undefined_phase_case_count", "undefined_phase_bin_count", "phase_verifiable")
REPORT_FIELDS = frozenset((
    "schema_version", "status", "model_version", "created_at_utc", "stage", "diagnostic_seed",
    "diagnostic_epoch", "frozen", "preflight", "scope", "deployment", "limitations",
    "check_only", "inference_performed", "gpu", "model_state_sha256_before",
    "model_state_sha256_after", "aggregates", "dataset_gates", "all_internal_gates_passed", "cases",
    *COHORT_FIELDS,
))
SCOPE = {
    "training_performed_this_diagnostic": False, "checkpoint_prior_training": True,
    "epoch_selection_performed": False, "parameter_selection_performed": False,
    "outer_validation_waveforms_loaded": False, "holdout_waveforms_loaded": False,
    "outer_validation_used_for_training_or_selection": False, "clean_ground_truth_used": False,
    "independent_acceptance": False, "runtime_enabled": False, "production_validated": False,
}

LIMITATIONS = (
    "JSON-only scalar reaggregation; no model inference, EEG/NPZ/PT read or GPU initialization.",
    "Original normalized errors and protected-band phase/complex values are saved claims, not independently recomputed from waveforms.",
    "Source checkpoint, NPZ and frozen training/evaluation lock hashes in the diagnostic are recorded claims, not byte-verified by this audit.",
    "Subjects 5-6 and synthetic corruption are internal development/selection; engineering gates are not independent research acceptance.",
    "Mixed global/condition summaries pool only dimensionless saved metrics, never native raw power; gates remain per dataset.",
    "No streaming/Hann-OLA preservation, task decoding, clinical validity, outer/holdout acceptance or deployment is established.",
)


class AuditError(ValueError):
    """Fail-closed saved-JSON input/publication violation."""


def _require(condition, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _finite_tree(value: Any) -> bool:
    if value is None or type(value) in (bool, str):
        return True
    if type(value) in (int, float):
        try:
            return math.isfinite(value)
        except OverflowError:
            return False
    if type(value) is list:
        return all(_finite_tree(v) for v in value)
    if type(value) is dict:
        return all(type(k) is str and _finite_tree(v) for k, v in value.items())
    return False


def _exact(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return set(actual) == set(expected) and all(_exact(actual[k], v) for k, v in expected.items())
    if type(expected) is list:
        return len(actual) == len(expected) and all(_exact(a, b) for a, b in zip(actual, expected))
    return actual == expected


_FROZEN_CONFIG_JSON = r'''
{
  "schema_version": 1,
  "status": "formal",
  "model_version": "learned_denoise_v8_ssvep",
  "model_type": "zero_initialized_per_bin_high_attenuation_spectral_tcn",
  "input_channels": 8,
  "window_samples": 256,
  "start_samples": {
    "Nakanishi2015": 38,
    "Wearable": 125
  },
  "sample_rates_hz": {
    "Nakanishi2015": 256,
    "Wearable": 250
  },
  "normalization": "per_channel_noisy_window_mean_rms_eps1e-8_float32_forward_float64_residual_restore",
  "base_reference_contract": {
    "config_path": "configs/learned_denoiser_v7_ssvep.json",
    "config_sha256": "ac578c9f2343510aeb7f6eeb843497b6e7a8320028fc409d8afd5cdfc660e243",
    "training_lock_path": "analysis_outputs/ssvep_v7_training_lock_20261007.json",
    "training_lock_sha256": "09d0db827f7c19836aa33797a8407f3ba0be7fdd53400bb6cc8b24d0be18ab25"
  },
  "model": {
    "hidden_channels": 32,
    "kernel_size": 7,
    "dilations": [
      1,
      2,
      4,
      8,
      16,
      32
    ],
    "group_norm_groups": 8,
    "dropout": 0.0,
    "protected_band_hz": [
      8.0,
      40.0
    ],
    "low_residual_limit_normalized": 0.5,
    "low_band_projection": "rfft_keep_frequency_lt8_then_irfft_n256",
    "high_band_projection": "rfft_keep_frequency_gt40_then_irfft_n256",
    "high_attenuation_max": 0.8,
    "high_attenuation_mapping": "clamp(raw,0,0.8) per channel and per rFFT bin strictly above 40 Hz",
    "high_attenuation_scope": "one_real_scalar_per_channel_per_high_bin",
    "zero_initialize_low_output": true,
    "zero_initialize_high_gate": true,
    "post_projection_clip": false,
    "high_spectral_feature_channels": 16,
    "high_spectral_kernel_size": 5,
    "high_spectral_input": "log1p(abs(rfft(x))/sqrt(window_samples)) per high-frequency bin"
  },
  "conditions": [
    "identity",
    "transient",
    "low_frequency_drift",
    "high_frequency_burst",
    "common_mode"
  ],
  "sampling": "one_fixed_window_per_trial_all_five_conditions_balanced_no_combinations",
  "corruption": {
    "seed": 20261007,
    "identity_probability": 0.2,
    "combine_probability": 0.0,
    "families": [
      "transient",
      "low_frequency_drift",
      "high_frequency_burst",
      "common_mode"
    ],
    "transient": {
      "amplitude_rms_range": [
        0.5,
        3.0
      ],
      "width_samples_range": [
        3,
        25
      ],
      "channel_count_range": [
        1,
        3
      ]
    },
    "low_frequency_drift": {
      "frequency_hz_range": [
        0.25,
        4.0
      ],
      "amplitude_rms_range": [
        0.2,
        1.5
      ],
      "channel_count_range": [
        1,
        8
      ]
    },
    "high_frequency_burst": {
      "frequency_hz_range": [
        45.0,
        100.0
      ],
      "amplitude_rms_range": [
        0.2,
        1.5
      ],
      "duration_samples_range": [
        20,
        120
      ],
      "channel_count_range": [
        1,
        4
      ]
    },
    "common_mode": {
      "amplitude_rms_range": [
        0.2,
        1.0
      ],
      "width_samples_range": [
        10,
        80
      ]
    },
    "target_definition": "recorded_reference_reconstruction; reference is not clean ground truth"
  },
  "seeds": {
    "training": 8026100808,
    "internal_selection": 11026100808,
    "diagnostic": 13026100808,
    "formula": "base + dataset_index*10000000 + subject_id*100000 + trial_index*10 + condition_index + epoch*100000000"
  },
  "loss": {
    "relative_mse_input_floor": 0.01,
    "high_frequency_case_weight": 4.0,
    "other_corrupted_case_weight": 1.0,
    "identity_normalized_mse_weight": 60.0,
    "high_frequency_low_branch_change_weight": 20.0,
    "normalized_change_penalty_weight": 0.001,
    "target_channel_variance_epsilon": 1e-08,
    "case_definition": "mean((pred-target)^2/(demeaned_target_channel_variance+1e-8)) in noisy-normalized coordinates; same for input baseline; identity has no input-ratio division",
    "aggregation": "sum(weighted_corrupted_relative_mse)/sum_corrupted_weights +60*mean_identity_mse +20*mean_HF_low_branch_normalized_change +0.001*mean_all_normalized_change; condition masks training/loss only never model.forward"
  },
  "training": {
    "fit_subjects": [
      1,
      2,
      3,
      4
    ],
    "internal_validation_subjects": [
      5,
      6
    ],
    "epochs": 18,
    "batch_size": 64,
    "learning_rate": 0.0005,
    "weight_decay": 0.0001,
    "optimizer": "AdamW",
    "seed": 2026100808,
    "workers": 0,
    "amp": false,
    "device": "cuda:0",
    "require_cuda": true,
    "deterministic_algorithms": true,
    "cublas_workspace_config": ":4096:8",
    "allow_tf32": false,
    "cpu_threads": 4,
    "gradient_clip_norm": 1.0,
    "selection": "per_domain: eligible-first, min_worst_condition_ratio_of_mean_error, min_identity_error, min_internal_total_loss, earliest_epoch; if none eligible still save best with gates_failed"
  },
  "internal_research_gates": {
    "high_frequency_burst_mean_error_reduction_min": 0.05,
    "other_corrupted_condition_mean_error_reduction_min": 0.0,
    "identity_mean_normalized_change_max": 0.0005,
    "identity_protected_band_complex_energy_ratio_max": 1e-06,
    "identity_protected_band_weighted_abs_phase_radians_max": 0.02,
    "all_gates_required_per_dataset": true,
    "interpretation": "internal engineering screen only, not independent task preservation or production acceptance"
  },
  "runtime": {
    "enabled": false,
    "production_validated": false,
    "deployment_scope": "offline_v8_train_internal_research_only"
  },
  "safety": {
    "clean_ground_truth_used": false,
    "holdout_waveforms_loaded": false,
    "outer_validation_waveforms_loaded": false,
    "outer_validation_used_for_training_or_selection": false,
    "native_units_assumed": false,
    "cross_dataset_participant_identity_known": false
  }
}
'''


def validate_config(cfg: dict) -> None:
    _require(_finite_tree(cfg) and _exact(cfg, json.loads(_FROZEN_CONFIG_JSON)),
             'fixed formal v8 configuration/type mismatch')


def safe_path(root: Path, value: str | Path) -> Path:
    """Reject external/D:/UNC/traversal/device/ADS paths BEFORE stat; no links."""
    for text in (str(root), str(value)):
        _require(bool(text) and '\x00' not in text and '..' not in text.replace('\\', '/').split('/'),
                 'empty/NUL/traversal path forbidden')
        win = PureWindowsPath(text)
        _require(win.drive.casefold() != 'd:', 'D: access forbidden')
        _require(not win.drive or (Path(text).is_absolute() and not win.drive.startswith('\\\\')),
                 'foreign/relative drive/UNC forbidden')
        _require(not text.startswith(('//', '\\\\')), 'UNC path forbidden')
        _require(not any(':' in part for part in win.parts[1:]) and (bool(win.drive) or ':' not in text),
                 'alternate data stream forbidden')
        for part in win.parts:
            if part in ('.', '\\', '/'):
                continue
            _require(not part.endswith((' ', '.')), 'ambiguous Windows path forbidden')
            _require(part.split('.')[0].upper() not in {'CON', 'PRN', 'AUX', 'NUL',
                         *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))},
                     'device path forbidden')
    base = Path(os.path.abspath(root))
    path = Path(value) if Path(value).is_absolute() else base / value
    path = Path(os.path.abspath(path))
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise AuditError('outside-workspace path forbidden') from exc
    for entry in (*reversed(path.parents), path):
        try:
            info = entry.lstat()
        except FileNotFoundError:
            continue
        _require(not stat.S_ISLNK(info.st_mode) and not (getattr(info, 'st_file_attributes', 0) & 0x400),
                 'symlink/reparse path forbidden')
        if entry != path:
            _require(stat.S_ISDIR(info.st_mode), 'path ancestor must be directory')
    return path


def _stamp(info) -> tuple:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _json_path(root: Path, rel: str | Path) -> Path:
    _require(str(rel).endswith('.json'), 'audit input must be JSON, never NPZ/PT')
    return safe_path(root, rel)


def _read_bytes(root: Path, rel: str | Path, *, source: bool = False) -> tuple[bytes, tuple]:
    if source:
        _require(str(rel) in AUDIT_SOURCE_PATHS, 'only this audit three source files may be hashed')
        path = safe_path(root, rel)
    else:
        path = _json_path(root, rel)
    try:
        info = path.lstat()
        _require(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= MAX_JSON_BYTES, 'regular nonempty bounded file required')
        before = _stamp(info)
        descriptor = os.open(safe_path(root, path), os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0))
        with os.fdopen(descriptor, 'rb') as handle:
            descriptor_before = _stamp(os.fstat(handle.fileno()))
            # Windows fstat/path ctime may differ: compare identity/size/mtime,
            # then separately require each full stat to remain unchanged.
            _require(descriptor_before[:4] == before[:4], 'input replaced before JSON/source read')
            data = handle.read(MAX_JSON_BYTES + 1)
            _require(_stamp(os.fstat(handle.fileno())) == descriptor_before, 'input changed during JSON/source read')
        _require(len(data) <= MAX_JSON_BYTES and _stamp(safe_path(root, path).lstat()) == before,
                 'input changed/oversized during JSON/source read')
    except OSError as exc:
        raise AuditError(f'inaccessible audit JSON/source: {rel}') from exc
    return data, before


def strict_json_loads(data: str | bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, f'duplicate JSON key: {key}')
            result[key] = value
        return result
    def constant(token):
        raise AuditError(f'nonfinite JSON token: {token}')
    try:
        if type(data) is bytes:
            data = data.decode('utf-8-sig')
        value = json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
        _require(type(value) is dict and _finite_tree(value), 'finite JSON object required')
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise AuditError('invalid strict JSON') from exc
    return value


def read_json(root: Path, rel: str | Path) -> dict:
    data, _ = _read_bytes(root, rel)
    return strict_json_loads(data)


def _number(value, context: str, minimum=None):
    _require(type(value) in (int, float) and _finite_tree(value), f'{context}: finite number, not bool, required')
    _require(minimum is None or value >= minimum, f'{context}: number below minimum')
    return value


def _digest(value, context: str) -> None:
    _require(type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) is not None, f'{context}: lowercase SHA256 required')


def _compare(actual, expected, path: str, mismatches: list, *, exact_number: bool = False) -> None:
    """Compare full trees including extra/missing keys; tolerance only arithmetic."""
    if type(expected) is dict:
        if type(actual) is not dict:
            mismatches.append({'path': path, 'reason': 'object required', 'saved': actual, 'expected': expected})
            return
        for key in sorted(set(actual) - set(expected)):
            mismatches.append({'path': f'{path}.{key}', 'reason': 'unexpected field', 'saved': actual[key], 'expected': None})
        for key, value in expected.items():
            if key not in actual:
                mismatches.append({'path': f'{path}.{key}', 'reason': 'missing field', 'saved': None, 'expected': value})
            else:
                _compare(actual[key], value, f'{path}.{key}', mismatches,
                         exact_number=exact_number or key == 'threshold')
        return
    if type(expected) is bool or type(expected) is int:
        equal = type(actual) is type(expected) and actual == expected
    elif type(expected) is float:
        equal = type(actual) in (int, float) and (actual == expected if exact_number else
                  math.isclose(actual, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL))
    else:
        equal = type(actual) is type(expected) and actual == expected
    if not equal:
        mismatches.append({'path': path, 'reason': 'saved/recomputed mismatch', 'saved': actual, 'expected': expected})


def _report_header(report: dict, cfg: dict) -> None:
    _require(type(report) is dict and _finite_tree(report), 'report must be finite JSON object')
    _require(set(report) == REPORT_FIELDS, 'executed v8 diagnostic report schema mismatch (not check-only/v6)')
    expected = {'schema_version': 1, 'status': 'formal', 'model_version': MODEL_VERSION,
        'stage': 'v8_fixed_internal_diagnostic', 'diagnostic_seed': SEED, 'diagnostic_epoch': 0,
        'check_only': False, 'inference_performed': True, 'scope': SCOPE,
        'deployment': {'status': 'blocked', 'runtime_enabled': False, 'production_validated': False}}
    _require(all(key in report and _exact(report[key], value) for key, value in expected.items()),
             'unsafe/type-invalid or unsupported executed v8 diagnostic header')
    _require(type(report['created_at_utc']) is str and bool(report['created_at_utc']), 'missing report timestamp')
    frozen = report['frozen']
    fields = {'configuration_sha256', 'preregistration_sha256', 'training_lock_sha256', 'evaluation_lock_sha256',
              'checkpoint_sha256', 'training_report_sha256', 'code_lock_sha256'}
    _require(type(frozen) is dict and set(frozen) == fields, 'frozen provenance schema mismatch')
    for key in fields:
        _digest(frozen[key], f'frozen.{key}')
    _require(frozen['configuration_sha256'] == CONFIG_SHA256 and frozen['preregistration_sha256'] == PREREG_SHA256,
             'saved frozen configuration/preregistration drift')
    for key in ('model_state_sha256_before', 'model_state_sha256_after'):
        _digest(report[key], key)
    preflight = report['preflight']
    preflight_fields = {'stage', 'check_only', 'verified_hashes', 'deferred', 'lock_sha256',
                        'code_lock_sha256', 'waveforms_loaded_this_call', 'checkpoint_loaded_this_call',
                        'cuda_initialized_this_call'}
    _require(type(preflight) is dict and set(preflight) == preflight_fields, 'invalid saved preflight schema')
    expected_preflight = {'stage': 'v8_internal_diagnostic', 'check_only': False,
                          'waveforms_loaded_this_call': False, 'checkpoint_loaded_this_call': False,
                          'cuda_initialized_this_call': False}
    _require(all(_exact(preflight.get(k), v) for k, v in expected_preflight.items()), 'invalid preflight call scope')
    for key in ('lock_sha256', 'code_lock_sha256'):
        _digest(preflight[key], 'preflight.' + key)
    _require(type(preflight['verified_hashes']) is dict and type(preflight['deferred']) is list and
             all(type(v) is str for v in preflight['deferred']), 'invalid preflight hash/deferred metadata')
    for value in preflight['verified_hashes'].values():
        _digest(value, 'preflight.verified_hashes')
    gpu = report['gpu']
    _require(type(gpu) is dict and set(gpu) == {'device', 'device_name', 'torch_version', 'cuda_version',
             'amp', 'allow_tf32', 'deterministic_algorithms'}, 'invalid saved GPU metadata schema')
    _require(gpu['device'] == 'cuda:0' and gpu['amp'] is False and gpu['allow_tf32'] is False and
             gpu['deterministic_algorithms'] is True and type(gpu['device_name']) is str and
             type(gpu['torch_version']) is str and type(gpu['cuda_version']) is str, 'unsafe saved GPU metadata')
    _require(type(report['limitations']) is list and all(type(v) is str for v in report['limitations']), 'invalid limitations')
    _require(type(report['cases']) is list, 'cases must be list')
    # Do not resolve saved paths or verify checkpoint/EEG bytes: those are
    # recorded metadata claims, not newly accessed evidence in a scalar audit.
    _require(cfg['runtime']['enabled'] is False and cfg['runtime']['production_validated'] is False,
             'audit never enables production')


def _validate_rows(rows: list, mismatches: list) -> tuple[list[dict], dict, dict]:
    seen, seeds = set(), set()
    normalized = []
    for index, saved in enumerate(rows):
        where = f'cases[{index}]'
        _require(type(saved) is dict and set(saved) == CASE_FIELDS and _finite_tree(saved),
                 f'{where}: exact finite case schema required; no waveforms/extra fields')
        dataset, subject, trial, condition = (saved[k] for k in ('dataset', 'subject_id', 'trial_index', 'condition'))
        _require(type(dataset) is str and dataset in DATASETS and type(condition) is str and condition in CONDITIONS,
                 f'{where}: illegal dataset/condition')
        _require(type(subject) is int and subject in SUBJECTS and type(trial) is int and 0 <= trial < TRIALS[dataset],
                 f'{where}: only native diagnostic subjects/trials permitted')
        key = dataset, subject, trial, condition
        _require(key not in seen, f'{where}: duplicate case identity')
        expected_seed = SEED + DATASETS.index(dataset) * 10000000 + subject * 100000 + trial * 10 + CONDITIONS.index(condition)
        for field, expected in (('seed', expected_seed), ('window_samples', WINDOW),
                                ('sample_rate_hz', RATES[dataset]), ('window_start_sample', STARTS[dataset])):
            _require(type(saved[field]) is int and saved[field] == expected, f'{where}.{field}: fixed provenance mismatch')
        _require(saved['seed'] not in seeds, f'{where}: duplicate case seed')
        seen.add(key)
        seeds.add(saved['seed'])
        _require(type(saved['input_normalized_mse']) is float and type(saved['output_normalized_mse']) is float,
                 f'{where}: v8 MSE scalars must be float, not int/bool')
        a = _number(saved['input_normalized_mse'], where + '.input_normalized_mse', 0)
        b = _number(saved['output_normalized_mse'], where + '.output_normalized_mse', 0)
        identity = condition == 'identity'
        reduction = None if identity or a == 0 else (a - b) / a
        _require(reduction is None or math.isfinite(reduction), f'{where}: recomputed reduction not finite')
        _require(saved['relative_reduction'] is None or type(saved['relative_reduction']) in (int, float),
                 f'{where}: reduction type invalid')
        outcome = 'win' if b < a else 'loss' if b > a else 'tie'
        _compare(saved['relative_reduction'], reduction, where + '.relative_reduction', mismatches)
        _compare(saved['outcome'], outcome, where + '.outcome', mismatches)
        bins = saved['undefined_phase_bin_count']
        _require(type(bins) is int and bins >= 0, f'{where}: invalid undefined phase bin count')
        spectral = ('protected_complex_energy_ratio', 'protected_weighted_abs_phase_radians')
        if identity:
            _compare(a, 0.0, where + '.identity_input_mse', mismatches, exact_number=True)
            _number(saved['identity_normalized_change'], where + '.identity_normalized_change', 0)
            _compare(saved['identity_normalized_change'], float(b), where + '.identity_normalized_change', mismatches)
            for field in spectral:
                if saved[field] is not None:
                    _number(saved[field], where + '.' + field, 0)
            phase = saved['protected_weighted_abs_phase_radians']
            _require(phase is None or phase <= math.pi, f'{where}: phase exceeds pi')
            _require(bins == 0 or phase is None, f'{where}: undefined phase must be null, never numeric/pass')
            _require(saved['protected_complex_energy_ratio'] is not None or phase is None,
                     f'{where}: phase cannot be defined when reference complex support is undefined')
        else:
            _require(saved['identity_normalized_change'] is None and bins == 0 and
                     all(saved[field] is None for field in spectral), f'{where}: nonidentity phase/change must be null/count zero')
        row = dict(saved)
        row.update(relative_reduction=reduction, identity_normalized_change=float(b) if identity else None,
                   outcome=outcome)
        normalized.append(row)
    expected = {(d, s, t, k) for d in DATASETS for s in SUBJECTS for t in range(TRIALS[d]) for k in CONDITIONS}
    missing = sorted(expected - seen)
    if missing:
        mismatches.append({'path': 'cases.cohort', 'reason': 'missing diagnostic cases',
                           'saved': {'case_count': len(rows)}, 'expected': {'case_count': CASE_COUNT, 'missing': [list(k) for k in missing]}})
    identities = [r for r in normalized if r['condition'] == 'identity']
    undefined_cases = sum(r['undefined_phase_bin_count'] > 0 or any(r[k] is None for k in
                         ('protected_complex_energy_ratio', 'protected_weighted_abs_phase_radians')) for r in identities)
    unique_trials = {(r['dataset'], r['subject_id'], r['trial_index']) for r in normalized}
    cohort = {'complete_cohort': seen == expected and len(rows) == CASE_COUNT, 'case_count': len(rows),
        'reference_trial_count': len(unique_trials), 'phase_case_count': len(identities),
        'undefined_phase_case_count': undefined_cases,
        'undefined_phase_bin_count': sum(r['undefined_phase_bin_count'] for r in identities),
        'phase_verifiable': len(identities) == REFERENCE_TRIAL_COUNT and undefined_cases == 0}
    counts = {'global': len(normalized), 'dataset': {}, 'condition': {}, 'dataset_condition': {},
              'subject': {}, 'subject_condition': {}}
    for r in normalized:
        d, s, k = r['dataset'], r['subject_id'], r['condition']
        for name, key in (('dataset', d), ('condition', k), ('dataset_condition', f'{d}/{k}'),
                          ('subject', f'{d}/{s}'), ('subject_condition', f'{d}/{s}/{k}')):
            counts[name][key] = counts[name].get(key, 0) + 1
    return normalized, cohort, counts


def _mean(values: list):
    try:
        value = statistics.fmean(values)
    except (OverflowError, ValueError) as exc:
        raise AuditError('nonfinite/empty saved-metric mean') from exc
    _require(math.isfinite(value), 'nonfinite recomputed mean')
    return value


def _conditions(rows: list[dict]) -> dict:
    """Reaggregate saved dimensionless scalars; never raw power/FFT/gates here."""
    result = {}
    for condition in CONDITIONS:
        part = [r for r in rows if r['condition'] == condition]
        if not part:
            continue
        a = _mean([r['input_normalized_mse'] for r in part])
        b = _mean([r['output_normalized_mse'] for r in part])
        def complete_mean(field):
            values = [r[field] for r in part]
            return None if any(v is None for v in values) else _mean(values)
        ratio = (a - b) / a if a > 0 and condition != 'identity' else None
        _require(ratio is None or math.isfinite(ratio), 'nonfinite ratio of mean errors')
        result[condition] = {'case_count': len(part), 'mean_input_normalized_mse': a,
            'mean_output_normalized_mse': b, 'ratio_of_mean_reduction': ratio,
            'mean_identity_normalized_change': complete_mean('identity_normalized_change'),
            'mean_protected_complex_energy_ratio': complete_mean('protected_complex_energy_ratio'),
            'mean_protected_weighted_abs_phase_radians': complete_mean('protected_weighted_abs_phase_radians'),
            'undefined_phase_case_count': sum(condition == 'identity' and
                 (r['undefined_phase_bin_count'] > 0 or r['protected_complex_energy_ratio'] is None or
                  r['protected_weighted_abs_phase_radians'] is None) for r in part)}
    return result


def _dataset_gate(summary: dict, cfg: dict) -> dict:
    """Independent frozen per-dataset screen; no cross-dataset gate pooling."""
    thresholds = cfg['internal_research_gates']
    gates = {}
    def compare(condition, field, threshold, minimum=False):
        group = summary.get(condition, {})
        value, count = group.get(field), group.get('case_count', 0)
        eligible = type(count) is int and count > 0 and type(value) in (int, float) and math.isfinite(value)
        passed = eligible and (value >= threshold if minimum else value <= threshold)
        gates[f'{condition}.{field}'] = {'passed': bool(passed), 'value': value, 'threshold': threshold,
                                        'comparison': '>=' if minimum else '<='}
    for condition in CONDITIONS[1:]:
        key = 'high_frequency_burst_mean_error_reduction_min' if condition == 'high_frequency_burst' else 'other_corrupted_condition_mean_error_reduction_min'
        compare(condition, 'ratio_of_mean_reduction', thresholds[key], True)
    compare('identity', 'mean_identity_normalized_change', thresholds['identity_mean_normalized_change_max'])
    compare('identity', 'mean_protected_complex_energy_ratio', thresholds['identity_protected_band_complex_energy_ratio_max'])
    compare('identity', 'mean_protected_weighted_abs_phase_radians', thresholds['identity_protected_band_weighted_abs_phase_radians_max'])
    undefined = summary.get('identity', {}).get('undefined_phase_case_count')
    gates['identity.phase_defined'] = {'passed': type(undefined) is int and undefined == 0,
                                      'value': undefined, 'threshold': 0, 'comparison': '=='}
    return {'all_passed': all(r['passed'] for r in gates.values()), 'gates': gates}


def audit_report(report: dict, cfg: dict) -> dict:
    """Pure JSON-object audit; schema violations raise, saved drift is mismatch.

    audit_passed means consistent saved scalar bookkeeping ONLY. Even if every
    internal engineering gate passes, research/runtime/production remain false.
    """
    validate_config(cfg)
    _report_header(report, cfg)
    mismatches = []
    rows, cohort, counts = _validate_rows(report['cases'], mismatches)
    groups = {name: defaultdict(list) for name in ('dataset', 'condition', 'dataset_condition', 'subject', 'subject_condition')}
    for row in rows:
        d, s, k = row['dataset'], row['subject_id'], row['condition']
        for name, key in (('dataset', d), ('condition', k), ('dataset_condition', f'{d}/{k}'),
                          ('subject', f'{d}/{s}'), ('subject_condition', f'{d}/{s}/{k}')):
            groups[name][key].append(row)
    aggregates = {'global': _conditions(rows)}
    for name, buckets in groups.items():
        aggregates[name] = {key: _conditions(bucket) for key, bucket in sorted(buckets.items())}
    dataset_gates = {d: _dataset_gate(aggregates['dataset'].get(d, {}), cfg) for d in DATASETS}
    all_passed = (all(r['all_passed'] for r in dataset_gates.values()) and
                  cohort['complete_cohort'] and cohort['phase_verifiable'])
    for field, expected in cohort.items():
        _compare(report[field], expected, field, mismatches)
    _compare(report['aggregates'], aggregates, 'aggregates', mismatches)
    _compare(report['dataset_gates'], dataset_gates, 'dataset_gates', mismatches)
    _compare(report['all_internal_gates_passed'], all_passed, 'all_internal_gates_passed', mismatches)
    _compare(report['model_state_sha256_after'], report['model_state_sha256_before'], 'model_state_sha256_after', mismatches)
    _compare(report['preflight']['lock_sha256'], report['frozen']['evaluation_lock_sha256'],
             'preflight.lock_sha256', mismatches)
    _compare(report['preflight']['code_lock_sha256'], report['frozen']['code_lock_sha256'],
             'preflight.code_lock_sha256', mismatches)
    return {'schema_version': 1, 'status': 'formal', 'audit_version': AUDIT_VERSION,
        'stage': 'saved_v8_internal_json_reaggregation', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'check_only': False, 'audit_performed': True, 'audit_passed': not mismatches,
        'mismatch_count': len(mismatches), 'mismatches': mismatches,
        'recomputed': {**cohort, 'group_case_counts': counts, 'aggregates': aggregates,
                       'dataset_gates': dataset_gates, 'all_internal_gates_passed': all_passed},
        'saved_binary_hash_claims': dict(report['frozen']),
        'comparison_policy': {'mean_relative_tolerance': REL_TOL, 'mean_absolute_tolerance': ABS_TOL,
                              'gate_threshold_tolerance': 0.0, 'counts_booleans_nulls_exact': True},
        'research_acceptance': False, 'independent_acceptance': False, 'runtime_enabled': False,
        'production_validated': False, 'waveforms_loaded': False, 'checkpoint_loaded': False,
        'cuda_initialized': False, 'model_inference_performed': False, 'binary_hashes_verified': False,
        'original_phase_numeric_values_verified': False, 'limitations': list(LIMITATIONS)}


def output_paths(root: Path, prefix: str = DEFAULT_OUTPUT_PREFIX) -> tuple[str, str]:
    _require(type(prefix) is str and re.fullmatch(r'analysis_outputs/ssvep_v8_saved_internal_audit[A-Za-z0-9_-]*', prefix) is not None,
             'output prefix must be workspace analysis_outputs/ssvep_v8_saved_internal_audit...')
    paths = prefix + '.json', prefix + '.md'
    for rel in paths:
        path = safe_path(root, rel)
        _require(not os.path.lexists(path), f'output collision: {rel}')
    return paths


def _unlink_owned(root: Path, path: Path, inode: tuple) -> None:
    try:
        path = safe_path(root, path)
        info = path.lstat()
    except (OSError, AuditError):
        return
    if stat.S_ISREG(info.st_mode) and (info.st_dev, info.st_ino) == inode:
        path.unlink()


def publish_audit(root: Path, prefix: str, audit: dict) -> tuple[Path, Path]:
    """Temp + fsync + exclusive hardlinks; rollback only this writer's inodes."""
    _require(type(audit) is dict and _finite_tree(audit), 'finite audit object required')
    relative = output_paths(root, prefix)
    targets = tuple(safe_path(root, rel) for rel in relative)
    contents = (json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8') + b'\n',
                render_markdown(audit))
    pending, published = [], []
    try:
        for target, content in zip(targets, contents):
            safe_path(root, target.parent).mkdir(parents=True, exist_ok=True)
            safe_path(root, target)
            descriptor, name = tempfile.mkstemp(prefix=target.name + '.', suffix='.pending', dir=target.parent)
            inode_info = os.fstat(descriptor)
            inode = inode_info.st_dev, inode_info.st_ino
            temporary = Path(name)
            pending.append((temporary, inode))
            with os.fdopen(descriptor, 'wb') as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        for (temporary, inode), target in zip(pending, targets):
            info = safe_path(root, temporary).lstat()
            _require((info.st_dev, info.st_ino) == inode, 'pending audit inode changed')
            safe_path(root, target)
            os.link(temporary, target)  # Exclusive, never replace or rename.
            published.append((target, inode))
            info = safe_path(root, target).lstat()
            _require((info.st_dev, info.st_ino) == inode, 'published audit inode changed')
    except BaseException:
        for target, inode in published:
            _unlink_owned(root, target, inode)
        raise
    finally:
        for temporary, inode in pending:
            _unlink_owned(root, temporary, inode)
    return targets


def render_markdown(audit: dict) -> bytes:
    lines = ['# SSVEP v8 saved internal diagnostic JSON audit', '',
        '**Scalar bookkeeping audit only; NOT independent acceptance or deployment.**', '',
        f"- Audit passed: {audit['audit_passed']}", f"- Mismatches: {audit['mismatch_count']}",
        '- Research acceptance: false; runtime enabled: false; production validated: false.',
        '- Original phase numeric values verified from EEG: false.',
        '- NPZ/PT byte hashes or inference re-executed: false.', '', '## Independent recomputation', '']
    cohort = audit['recomputed']
    lines.extend(f'- {k}: {cohort[k]}' for k in (*COHORT_FIELDS, 'all_internal_gates_passed'))
    lines.extend(['', '## Per-dataset frozen internal engineering gates', '', '```json',
        json.dumps(cohort['dataset_gates'], ensure_ascii=False, indent=2, allow_nan=False), '```',
        '', '## JSON/source evidence', ''])
    lines.extend(f'- {key}: `{value}`' for key, value in sorted(audit.get('inputs', {}).items()) if type(value) is str)
    lines.extend(f'- Audit source `{key}`: `{value}`' for key, value in sorted(audit.get('audit_source_sha256', {}).items()))
    lines.extend(['', '## Mismatches', ''])
    if not audit['mismatches']:
        lines.append('None within the stated scalar arithmetic comparison policy.')
    else:
        lines.extend(f"- `{m['path']}`: {m['reason']}" for m in audit['mismatches'])
    lines.extend(['', 'Full expected/saved values and all group summaries/counts are in the audit JSON.',
                  '', '## Limitations', ''])
    lines.extend('- ' + item for item in audit['limitations'])
    return ('\n'.join(lines) + '\n').encode('utf-8')


def audit_saved_report(root: Path = ROOT, *, report_relative: str = REPORT_RELATIVE,
                       output_prefix: str = DEFAULT_OUTPUT_PREFIX, check_only: bool = False) -> dict:
    """Audit JSON+config only. Check-only never opens the report or publishes.

    Audit-source hashes are separate evidence, not changes to frozen v8
    training/diagnostic bindings. Input collisions are refused before any read.
    """
    _require(type(check_only) is bool, 'check_only must be bool')
    outputs = output_paths(root, output_prefix)  # Before config/report reads.
    report_path = _json_path(root, report_relative)
    _require(report_path not in tuple(safe_path(root, p) for p in outputs), 'input/output collision')
    cfg_data, cfg_stamp = _read_bytes(root, CONFIG_RELATIVE)
    cfg_sha = hashlib.sha256(cfg_data).hexdigest()
    _require(cfg_sha == CONFIG_SHA256, 'frozen config bytes SHA drift')
    cfg = strict_json_loads(cfg_data)
    validate_config(cfg)
    if check_only:
        return {'schema_version': 1, 'status': 'formal', 'audit_version': AUDIT_VERSION,
            'check_only': True, 'audit_performed': False, 'audit_passed': False,
            'report_read': False, 'report_read_deferred': report_relative, 'would_publish': list(outputs),
            'config_sha256': cfg_sha, 'research_acceptance': False, 'runtime_enabled': False,
            'production_validated': False, 'waveforms_loaded': False, 'checkpoint_loaded': False,
            'cuda_initialized': False, 'limitations': list(LIMITATIONS)}
    report_data, report_stamp = _read_bytes(root, report_relative)
    report_sha = hashlib.sha256(report_data).hexdigest()
    report = strict_json_loads(report_data)
    result = audit_report(report, cfg)
    snapshots = {CONFIG_RELATIVE: (cfg_sha, cfg_stamp), report_relative: (report_sha, report_stamp)}
    source_hashes = {}
    for rel in AUDIT_SOURCE_PATHS:
        data, stamp = _read_bytes(root, rel, source=True)
        digest = hashlib.sha256(data).hexdigest()
        source_hashes[rel] = digest
        snapshots[rel] = digest, stamp
    result['inputs'] = {'report_path': report_relative, 'report_sha256': report_sha,
                        'config_path': CONFIG_RELATIVE, 'config_sha256': cfg_sha}
    result['audit_source_sha256'] = source_hashes
    result['audit_source_binding_scope'] = 'separate_v8_saved_json_audit_evidence_not_training_or_diagnostic_bindings'
    # Rehash JSON/source + re-stat just before exclusive publication. No paths
    # from the saved diagnostic are resolved, statted, hashed or followed.
    for rel, (digest, stamp) in snapshots.items():
        data, current = _read_bytes(root, rel, source=rel in AUDIT_SOURCE_PATHS)
        _require(current == stamp and hashlib.sha256(data).hexdigest() == digest, f'audit input/source changed: {rel}')
    publish_audit(root, output_prefix, result)
    return result
