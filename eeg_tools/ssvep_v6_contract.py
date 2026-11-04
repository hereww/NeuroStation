"""Frozen v6 offline safety and numerical contract. No binary I/O at import.

Preflight never creates a lock. Check-only defers ALL NPZ/PT byte reads.
Diagnostic execution hashes/loads only subjects 5-6, never fit/outer/holdout.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any
import copy
import hashlib
import json
import math
import os
import stat
import tempfile

import numpy as np

ROOT = Path(__file__).absolute().parents[1]
MODEL_VERSION = "learned_denoise_v6_ssvep"
CONFIG_RELATIVE = "configs/learned_denoiser_v6_ssvep.json"
PREREG_RELATIVE = "analysis_outputs/ssvep_v6_training_preregistration_20261007.json"
TRAIN_LOCK_RELATIVE = "analysis_outputs/ssvep_v6_training_lock_20261007.json"
EVAL_LOCK_RELATIVE = "analysis_outputs/ssvep_v6_internal_evaluation_lock_20261007.json"
CHECKPOINT_RELATIVE = "models/denoiser_v6_ssvep.pt"
TRAIN_REPORT_RELATIVE = "analysis_outputs/ssvep_v6_training_20261007.json"
CONFIG_SHA256 = "f1deaf6f28add5f14f24719675fb071f1c27c62befc7bafafc47c3e8c66eb8f0"
PREREG_SHA256 = "6d98383db26f7afb92302fac4371eb40c3dd31539f282ef1a37fe7440d5c07ec"
REFERENCE_DIR = "datasets/processed/ssvep_reference_v5_20261007"
MANIFEST_RELATIVE = REFERENCE_DIR + "/manifest.json"
PARTITION_RELATIVE = "analysis_outputs/ssvep_reference_v5_partition_lock_20261007.json"
DATASETS = ("Nakanishi2015", "Wearable")
CONDITIONS = ("identity", "transient", "low_frequency_drift", "high_frequency_burst", "common_mode")
SHAPES = {"Nakanishi2015": (180, 1114, 8), "Wearable": (240, 710, 8)}
RATES = {"Nakanishi2015": 256, "Wearable": 250}
STARTS = {"Nakanishi2015": 38, "Wearable": 125}
NEW_SOURCE_PATHS = (
    "eeg_tools/ssvep_v6_contract.py", "tests/test_ssvep_v6_contract.py",
    "eeg_tools/ssvep_denoise_v6.py", "scripts/train_ssvep_denoiser_v6.py",
    "tests/test_ssvep_denoise_v6.py", "eeg_tools/ssvep_v6_internal_eval.py",
    "scripts/evaluate_ssvep_denoiser_v6_internal.py", "tests/test_ssvep_v6_internal_eval.py",
)
NONBINARY_INPUTS = (
    "eeg_tools/ssvep_denoise_v5.py", "eeg_tools/ssvep_reference_v5.py",
    "eeg_tools/ssvep_internal_benchmark_v5.py", "configs/learned_denoiser_v5_ssvep.json",
    MANIFEST_RELATIVE, PARTITION_RELATIVE, CONFIG_RELATIVE,
)
INPUT_PATHS = NONBINARY_INPUTS + tuple(
    path for dataset in DATASETS for subject in range(1, 7)
    for path in (f"{REFERENCE_DIR}/{dataset}/subject_{subject:03d}.npz",
                 f"{REFERENCE_DIR}/{dataset}/subject_{subject:03d}.npz.manifest.json")
)
SCOPE_FALSE_FLAGS = (
    "runtime_enabled", "production_validated", "holdout_waveforms_loaded",
    "outer_validation_waveforms_loaded", "outer_validation_used_for_training_or_selection",
)

class V6Error(ValueError):
    """Fail-closed contract violation."""


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise V6Error(message)


def _plain(value: Any) -> bool:
    if value is None or type(value) in (bool, int, str):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(_plain(v) for v in value)
    if type(value) is dict:
        return all(type(k) is str and _plain(v) for k, v in value.items())
    return False


def _exact(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return set(actual) == set(expected) and all(_exact(actual[k], v) for k, v in expected.items())
    if type(expected) is list:
        return len(actual) == len(expected) and all(_exact(a, b) for a, b in zip(actual, expected))
    return actual == expected


def _required(obj: Any, expected: dict, context: str) -> None:
    _require(type(obj) is dict and all(k in obj and _exact(obj[k], v) for k, v in expected.items()),
             f"{context}: required fields/types mismatch")


# Literal frozen configuration, not a mutable reference to the caller's config.
_EXPECTED_CONFIG_JSON = r'''
{
  "schema_version": 1,
  "status": "formal",
  "model_version": "learned_denoise_v6_ssvep",
  "model_type": "zero_initialized_band_projected_residual_tcn",
  "reference_version": "ssvep_reference_v5",
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
    "pre_projection_residual_limit_normalized": 0.5,
    "zero_initialize_output": true,
    "protected_band_hz": [
      8.0,
      40.0
    ],
    "projection": "rfft_remove_inclusive_protected_band_then_irfft_n256",
    "post_projection_clip": false
  },
  "corruption": {
    "seed": 20261007,
    "identity_probability": 0.2,
    "combine_probability": 0.2,
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
  "conditions": [
    "identity",
    "transient",
    "low_frequency_drift",
    "high_frequency_burst",
    "common_mode"
  ],
  "sampling": "one_fixed_window_per_trial_all_five_conditions_balanced_no_combinations",
  "seeds": {
    "training": 2026107601,
    "internal_selection": 2026107602,
    "diagnostic": 2026107603,
    "formula": "base + dataset_index*10000000 + subject_id*100000 + trial_index*10 + condition_index + epoch*100000000"
  },
  "loss": {
    "corrupted_case_relative_mse_weight": 1.0,
    "relative_mse_input_floor": 0.01,
    "identity_normalized_mse_weight": 20.0,
    "normalized_change_penalty_weight": 0.001,
    "target_channel_variance_epsilon": 1e-08,
    "case_definition": "mean((pred-target)^2/(demeaned_target_channel_variance+1e-8)) in noisy-normalized coordinates; same for input baseline; identity has no input-ratio division"
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
    "seed": 2026100706,
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
    "deployment_scope": "offline_train_internal_research_only"
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
    expected = json.loads(_EXPECTED_CONFIG_JSON)
    _require(_plain(cfg) and _exact(cfg, expected), 'fixed formal v6 configuration mismatch')
    for dataset, fs in cfg['sample_rates_hz'].items():
        _require(cfg['model']['protected_band_hz'][1] < fs / 2, 'protected band exceeds Nyquist')
        _require(cfg['corruption']['high_frequency_burst']['frequency_hz_range'][1] < fs / 2, 'corruption exceeds Nyquist')
        _require(cfg['start_samples'][dataset] + cfg['window_samples'] <= SHAPES[dataset][1], 'window exceeds record')
    for family in ('transient', 'common_mode'):
        _require(cfg['corruption'][family]['width_samples_range'][1] <= cfg['window_samples'], 'width exceeds window')
    _require(cfg['corruption']['high_frequency_burst']['duration_samples_range'][1] <= cfg['window_samples'], 'duration exceeds window')


def safe_path(root: Path, value: str | Path) -> Path:
    """Lexically reject foreign/traversal paths BEFORE touching filesystem."""
    text, root_text = str(value), str(root)
    for item in (text, root_text):
        _require(bool(item) and '\x00' not in item and '..' not in item.replace('\\', '/').split('/'),
                 'empty/NUL/traversal path forbidden')
        win = PureWindowsPath(item)
        _require(win.drive.casefold() != 'd:', 'D: access forbidden')
        _require(not win.drive or (Path(item).is_absolute() and not win.drive.startswith('\\\\')),
                 'foreign/relative drive or UNC forbidden')
        _require(not any(':' in part for part in win.parts[1:]) and
                 (bool(win.drive) or ':' not in item), 'alternate stream forbidden')
        _require(not any(part.endswith((' ', '.')) for part in win.parts if part not in ('.', '\\', '/')),
                 'ambiguous Windows path forbidden')
        _require(not any(part.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL',
                     *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}
                     for part in win.parts), 'device path forbidden')
    base = Path(os.path.abspath(root))
    path = Path(text) if Path(text).is_absolute() else base / text
    path = Path(os.path.abspath(path))
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise V6Error('external workspace path forbidden') from exc
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


def _stat_tuple(info) -> tuple:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _stamp(root: Path, rel: str | Path) -> tuple:
    try:
        info = safe_path(root, rel).lstat()
    except OSError as exc:
        raise V6Error(f'missing/inaccessible frozen file: {rel}') from exc
    _require(stat.S_ISREG(info.st_mode) and info.st_size > 0, f'nonempty regular file required: {rel}')
    return _stat_tuple(info)


def _read_bytes(root: Path, rel: str | Path) -> bytes:
    before = _stamp(root, rel)
    try:
        # O_NOFOLLOW adds descriptor-level protection on platforms supporting it.
        descriptor = os.open(safe_path(root, rel), os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0))
        with os.fdopen(descriptor, 'rb') as handle:
            descriptor_before = _stat_tuple(os.fstat(handle.fileno()))
            # Windows descriptor/path ctime semantics can differ; identity, size
            # and mtime must match, while both ctimes stay stable independently.
            _require(descriptor_before[:4] == before[:4], 'file replaced before read')
            data = handle.read()
            _require(_stat_tuple(os.fstat(handle.fileno())) == descriptor_before, 'file changed during read')
    except OSError as exc:
        raise V6Error(f'cannot read frozen file: {rel}') from exc
    _require(before == _stamp(root, rel), f'file changed during read: {rel}')
    return data


def file_sha256(root: Path, rel: str | Path) -> str:
    """Streaming SHA with path/stat/descriptor stability, no deserialization."""
    before = _stamp(root, rel)
    try:
        descriptor = os.open(safe_path(root, rel), os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0))
        with os.fdopen(descriptor, 'rb') as handle:
            descriptor_before = _stat_tuple(os.fstat(handle.fileno()))
            _require(descriptor_before[:4] == before[:4], 'file replaced before hash')
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
            _require(_stat_tuple(os.fstat(handle.fileno())) == descriptor_before, 'file changed during hash')
    except OSError as exc:
        raise V6Error(f'cannot hash frozen file: {rel}') from exc
    _require(before == _stamp(root, rel), f'file changed during hash: {rel}')
    return digest


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f'duplicate JSON key: {key}')
        result[key] = value
    return result


def read_json(root: Path, rel: str | Path) -> dict:
    _require(str(rel).endswith('.json'), 'metadata path must end with .json')
    try:
        value = json.loads(_read_bytes(root, rel).decode('utf-8-sig'), object_pairs_hook=_unique_pairs,
                           parse_constant=lambda token: (_ for _ in ()).throw(V6Error(f'nonfinite JSON: {token}')))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise V6Error(f'invalid strict JSON: {rel}') from exc
    _require(type(value) is dict and _plain(value), 'JSON must be finite object')
    return value


def load_config(root: Path = ROOT) -> dict:
    cfg = read_json(root, CONFIG_RELATIVE)
    validate_config(cfg)
    return cfg


def _bindings(files: Any, paths, context: str) -> None:
    _require(type(files) is dict and set(files) == set(paths), f'{context}: exact file set required')
    _require(all(type(h) is str and len(h) == 64 and all(c in '0123456789abcdef' for c in h)
                 for h in files.values()), f'{context}: lowercase SHA256 required')


def _validate_prereg(protocol: dict, cfg: dict) -> None:
    _required(protocol, {'schema_version': 1, 'status': 'formal',
                         'experiment': 'learned_denoise_v6_ssvep_train_internal_only',
                         'configuration': {'path': CONFIG_RELATIVE, 'sha256': CONFIG_SHA256},
                         'model_design': cfg['model'], 'training_schedule': cfg['training'],
                         'internal_gates': cfg['internal_research_gates'],
                         'scope': {'fit_subjects': [1, 2, 3, 4], 'internal_selection_subjects': [5, 6],
                                   'diagnostic_subjects': [5, 6], 'outer_validation_subjects_not_loaded': [7, 8],
                                   'holdout_subjects_not_loaded': [9, 10], 'recorded_reference_is_clean_ground_truth': False,
                                   'runtime_enabled': False, 'production_validated': False}}, 'preregistration')
    _bindings(protocol.get('bound_files'), INPUT_PATHS, 'preregistration')
    _require(protocol['bound_files'][CONFIG_RELATIVE] == CONFIG_SHA256, 'preregistered config hash mismatch')


def _lock_relative(stage: str) -> str:
    _require(stage in ('training', 'internal_evaluation'), 'illegal preflight stage')
    return TRAIN_LOCK_RELATIVE if stage == 'training' else EVAL_LOCK_RELATIVE


def _validate_lock(lock: dict, protocol: dict, stage: str) -> dict:
    marker = 'frozen_before_v6_training_waveform_read' if stage == 'training' else 'frozen_before_v6_internal_diagnostic'
    _required(lock, {'schema_version': 1, 'status': 'formal', 'stage': marker}, 'freeze lock')
    paths = INPUT_PATHS + (PREREG_RELATIVE,) + NEW_SOURCE_PATHS
    if stage == 'internal_evaluation':
        paths += (CHECKPOINT_RELATIVE, TRAIN_REPORT_RELATIVE)
    _bindings(lock.get('files'), paths, 'freeze lock')
    _require(lock['files'][PREREG_RELATIVE] == PREREG_SHA256, 'lock/preregistration hash mismatch')
    for path, digest in protocol['bound_files'].items():
        _require(lock['files'][path] == digest, f'lock/preregistration binding mismatch: {path}')
    _required(lock.get('scope'), dict.fromkeys(SCOPE_FALSE_FLAGS, False), 'lock safety scope')
    unsafe = set(SCOPE_FALSE_FLAGS) | {'clean_ground_truth_used', 'runtime', 'production', 'holdout', 'outer',
        'outer_validation_used_for_selection', 'holdout_waveform_loaded', 'outer_validation_waveform_loaded',
        'recorded_reference_is_clean_ground_truth', 'native_units_assumed'}
    if stage == 'training':
        _required(lock, {'checkpoint_loaded': False, 'waveforms_loaded': False}, 'training lock unread markers')
        unsafe |= {'checkpoint_loaded', 'waveforms_loaded', 'cuda_initialized'}
    def inspect(value):
        if type(value) is dict:
            for key, item in value.items():
                if key in unsafe:
                    if type(item) is dict:
                        _require(all(type(v) is bool and v is False for v in item.values()), f'unsafe lock flag: {key}')
                    else:
                        _require(item is False, f'unsafe lock flag: {key}')
                inspect(item)
        elif type(value) is list:
            for item in value:
                inspect(item)
    inspect(lock)
    def scope_flags_false(value):
        if type(value) is bool:
            return value is False
        if type(value) is dict:
            return all(scope_flags_false(item) for item in value.values())
        if type(value) is list:
            return all(scope_flags_false(item) for item in value)
        return True
    _require(scope_flags_false(lock['scope']), 'lock scope contains a true safety/access flag')
    return dict(lock['files'])


def _catalog(root: Path, protocol: dict, stage: str) -> tuple[dict, ...]:
    manifest = read_json(root, MANIFEST_RELATIVE)
    partition = read_json(root, PARTITION_RELATIVE)
    partition_sha = protocol['bound_files'][PARTITION_RELATIVE]
    _required(manifest, {'schema_version': 1, 'status': 'formal', 'reference_version': 'ssvep_reference_v5',
                        'build_completed': True, 'source_lock_file_sha256': partition_sha,
                        'raw_reference_is_clean_ground_truth': False,
                        'runtime': {'enabled': False, 'production_validated': False}}, 'catalog')
    _required(partition, {'schema_version': 1, 'status': 'formal', 'stage': 'reference_identity_freeze',
                         'partitions': {'train': [1, 2, 3, 4, 5, 6], 'validation': [7, 8], 'holdout': [9, 10]},
                         'raw_reference_is_clean_ground_truth': False,
                         'runtime': {'enabled': False, 'production_validated': False}}, 'partition')
    rows = manifest.get('materialized_subjects')
    _require(type(rows) is list and len(rows) == 16, 'complete 16-record catalog required')
    identities, selected = set(), {}
    for row in rows:
        _require(type(row) is dict and row.get('dataset') in DATASETS and type(row.get('subject_id')) is int,
                 'illegal catalog identity')
        dataset, subject = row['dataset'], row['subject_id']
        _require(1 <= subject <= 8 and (dataset, subject) not in identities, 'duplicate/illegal catalog identity')
        identities.add((dataset, subject))
        _require(row.get('split') == ('train' if subject <= 6 else 'validation'), 'catalog split mismatch')
        if subject > 6:
            continue  # Never resolve/stat a path or open a sidecar for 7-8/9-10.
        rel = f'{REFERENCE_DIR}/{dataset}/subject_{subject:03d}.npz'
        _required(row, {'relative_path': f'{dataset}/subject_{subject:03d}.npz', 'trials': SHAPES[dataset][0],
                        'npz_sha256': protocol['bound_files'][rel],
                        'manifest_sha256': protocol['bound_files'][rel + '.manifest.json']}, 'train catalog row')
        side = read_json(root, rel + '.manifest.json')
        _required(side, {'schema_version': 1, 'status': 'formal', 'reference_version': 'ssvep_reference_v5',
                         'dataset': dataset, 'subject_id': subject, 'split': 'train',
                         'reference_shape': list(SHAPES[dataset]), 'reference_dtype': 'float64',
                         'native_sample_rate_hz': RATES[dataset], 'onset_sample_zero_based': STARTS[dataset],
                         'native_units': None, 'source_lock_file_sha256': partition_sha,
                         'raw_reference_is_clean_ground_truth': False, 'runtime_enabled': False,
                         'npz_sha256': row['npz_sha256'], 'materialized_trials': SHAPES[dataset][0]}, 'sidecar')
        if stage == 'training' or subject in (5, 6):
            selected[(dataset, subject)] = {'dataset': dataset, 'subject_id': subject,
                'npz_sha256': row['npz_sha256'], 'sidecar_sha256': row['manifest_sha256'],
                'relative_path': rel, 'sample_rate_hz': RATES[dataset], 'onset_sample_zero_based': STARTS[dataset]}
    _require(identities == {(d, s) for d in DATASETS for s in range(1, 9)}, 'catalog cohort incomplete')
    return tuple(selected[key] for key in sorted(selected))


@dataclass(frozen=True)
class Preflight:
    root: Path
    config: dict
    lock_sha256: str
    files: dict[str, str]
    stamps: dict[str, tuple]
    selected: tuple[dict, ...]
    check_only: bool
    stage: str
    deferred: tuple[str, ...]
    verified_hashes: dict[str, str]

    @property
    def deferred_hashes(self) -> tuple[str, ...]:
        return self.deferred

    def report(self) -> dict:
        """CLI-safe metadata; unread markers describe this call, NOT history."""
        return {'schema_version': 1, 'status': 'formal', 'stage': self.stage, 'check_only': self.check_only,
                'lock_sha256': self.lock_sha256, 'selected': list(self.selected),
                'verified_hashes': dict(self.verified_hashes), 'deferred': list(self.deferred),
                'binary_hashes_complete': not self.deferred,
                'read_state_scope': 'this_preflight_call_only_not_global_experiment_history',
                'waveforms_loaded': False,
                'checkpoint_loaded': False, 'cuda_initialized': False}


def _defer(rel: str, stage: str, check_only: bool) -> bool:
    if rel == CHECKPOINT_RELATIVE:
        return check_only
    if rel.endswith('.npz'):
        return check_only or (stage == 'internal_evaluation' and
                             PureWindowsPath(rel).name not in ('subject_005.npz', 'subject_006.npz'))
    return False


def preflight(root: Path = ROOT, stage: str = 'training', check_only: bool = True) -> Preflight:
    _require(type(check_only) is bool, 'check_only must be bool')
    lock_rel = _lock_relative(stage)
    base = safe_path(root, '.')
    lock = read_json(base, lock_rel)
    lock_sha = file_sha256(base, lock_rel)
    cfg = load_config(base)
    _require(file_sha256(base, CONFIG_RELATIVE) == CONFIG_SHA256, 'config SHA mismatch')
    _require(file_sha256(base, PREREG_RELATIVE) == PREREG_SHA256, 'preregistration SHA mismatch')
    protocol = read_json(base, PREREG_RELATIVE)
    _validate_prereg(protocol, cfg)
    files = _validate_lock(lock, protocol, stage)
    stamps, hashes, deferred = {lock_rel: _stamp(base, lock_rel)}, {}, []
    for rel, expected in files.items():
        stamps[rel] = _stamp(base, rel)
        if _defer(rel, stage, check_only):
            deferred.append(rel)
        else:
            hashes[rel] = file_sha256(base, rel)
            _require(hashes[rel] == expected, f'frozen SHA mismatch: {rel}')
    selected = _catalog(base, protocol, stage)
    state = Preflight(base, cfg, lock_sha, files, stamps, selected, check_only, stage,
                      tuple(deferred), hashes)
    verify_frozen(state)
    return state


def verify_frozen(state: Preflight) -> None:
    _require(type(state) is Preflight and type(state.check_only) is bool, 'invalid preflight state')
    lock_rel = _lock_relative(state.stage)
    _require(file_sha256(state.root, lock_rel) == state.lock_sha256, 'freeze lock changed')
    lock = read_json(state.root, lock_rel)
    protocol = read_json(state.root, PREREG_RELATIVE)
    _validate_prereg(protocol, state.config)
    files = _validate_lock(lock, protocol, state.stage)
    _require(files == state.files and _exact(state.config, load_config(state.root)), 'frozen state bindings/config changed')
    deferred = tuple(rel for rel in files if _defer(rel, state.stage, state.check_only))
    _require(state.deferred == deferred and set(state.verified_hashes) == set(files) - set(deferred),
             'frozen verification coverage changed')
    _require(set(state.stamps) == set(files) | {lock_rel}, 'frozen stamp coverage changed')
    for rel, before in state.stamps.items():
        _require(_stamp(state.root, rel) == before, f'frozen file stat changed: {rel}')
        if rel != lock_rel and rel not in deferred:
            digest = file_sha256(state.root, rel)
            _require(digest == files[rel] == state.verified_hashes[rel], f'frozen file hash changed: {rel}')
    _require(_catalog(state.root, protocol, state.stage) == state.selected, 'frozen selected cohort changed')


def load_training_records(state: Preflight):
    _require(type(state) is Preflight and not state.check_only, 'waveform loading forbidden in check-only')
    verify_frozen(state)
    subjects = (1, 2, 3, 4, 5, 6) if state.stage == 'training' else (5, 6)
    from eeg_tools import ssvep_denoise_v5 as core
    try:
        records = core.load_reference_records(state.root, subjects=subjects)
    except (ValueError, OSError) as exc:
        raise V6Error('frozen training/internal reference load failed') from exc
    expected = {(d, s) for d in DATASETS for s in subjects}
    _require(len(records) == len(expected), 'loaded cohort count mismatch')
    observed = set()
    for record in records:
        key = record.dataset, record.subject_id
        _require(type(record.subject_id) is int and key in expected and key not in observed, 'loaded cohort identity mismatch')
        observed.add(key)
        rel = f'{REFERENCE_DIR}/{record.dataset}/subject_{record.subject_id:03d}.npz'
        _require(record.sample_rate_hz == RATES[record.dataset] and record.npz_sha256 == state.files[rel] and
                 record.sidecar_sha256 == state.files[rel + '.manifest.json'], 'loaded metadata mismatch')
        arr = np.asarray(record.signal)
        _require(arr.shape == SHAPES[record.dataset] and arr.dtype == np.float64 and np.isfinite(arr).all(),
                 'loaded waveform contract mismatch')
    _require(observed == expected, 'loaded cohort incomplete')
    verify_frozen(state)
    return records


def _window(value: Any, name: str, dtype=None) -> np.ndarray:
    arr = np.asarray(value)
    _require(arr.shape == (8, 256) and arr.dtype.kind in 'fi' and np.isfinite(arr).all(),
             f'{name} must be finite real [8,256]')
    if dtype is not None:
        _require(arr.dtype == np.dtype(dtype), f'{name} dtype must be {np.dtype(dtype)}')
    return arr


def _case_seed(dataset: str, subject_id: int, trial_index: int, condition: str,
               purpose: str, epoch: int, cfg: dict) -> int:
    _require(dataset in DATASETS and condition in CONDITIONS, 'illegal dataset/condition')
    _require(purpose in ('training', 'internal_selection', 'diagnostic'), 'illegal case purpose')
    subjects = (1, 2, 3, 4) if purpose == 'training' else (5, 6)
    _require(type(subject_id) is int and subject_id in subjects, 'purpose subject allowlist violation')
    _require(type(trial_index) is int and 0 <= trial_index < SHAPES[dataset][0], 'illegal trial index')
    _require(type(epoch) is int and (1 <= epoch <= 18 if purpose == 'training' else epoch == 0),
             'illegal purpose epoch')
    return (cfg['seeds'][purpose] + DATASETS.index(dataset) * 10000000 + subject_id * 100000 +
            trial_index * 10 + CONDITIONS.index(condition) + epoch * 100000000)


def corrupt_case(reference, dataset, subject_id, trial_index, condition, cfg, *, purpose='training', epoch=0):
    validate_config(cfg)
    seed = _case_seed(dataset, subject_id, trial_index, condition, purpose, epoch, cfg)
    ref = _window(reference, 'reference', np.float64)
    if condition == 'identity':
        return ref.copy(), seed
    override = copy.deepcopy(cfg['corruption'])
    override.update(identity_probability=0.0, combine_probability=0.0, families=[condition])
    from eeg_tools import ssvep_denoise_v5 as core
    try:
        noisy, identity, chosen = core.corrupt_window(ref.copy(), RATES[dataset], {'corruption': override}, seed)
    except (ValueError, OSError) as exc:
        raise V6Error('fixed corruption failed') from exc
    _require(identity is False and chosen == [condition], 'fixed single corruption condition violated')
    return _window(noisy, 'corrupted output', np.float64), seed


def normalize_window(noisy):
    from eeg_tools import ssvep_denoise_v5 as core
    ref = _window(noisy, 'noisy', np.float64)
    with np.errstate(over='ignore', invalid='ignore'):
        normalized, mean, rms = core.normalize_window(ref)
    _window(normalized, 'normalized noisy', np.float32)
    _require(mean.shape == rms.shape == (8, 1) and mean.dtype == rms.dtype == np.float64 and
             np.isfinite(mean).all() and np.isfinite(rms).all() and (rms > 0).all(), 'invalid normalization statistics')
    return normalized, mean, rms


def restore_window(noisy, normalized_noisy, prediction, noisy_rms):
    from eeg_tools import ssvep_internal_benchmark_v5 as benchmark
    try:
        return benchmark.restore_window(noisy, normalized_noisy, prediction, noisy_rms)
    except (ValueError, TypeError) as exc:
        raise V6Error('invalid residual restore inputs/output') from exc


def case_metrics(ref, noisy, out, *, identity=False, sample_rate_hz: int) -> dict:
    """Rectangular demeaned rFFT; explicit native sample rate is mandatory.

    A vanished positive-reference bin makes phase undefined, never a fake zero.
    Spectral fields are null for nonidentity cases; no Hann/OLA proof is claimed.
    """
    _require(type(identity) is bool, 'identity must be bool')
    _require(type(sample_rate_hz) is int and sample_rate_hz in (250, 256), 'illegal sample rate')
    ref = _window(ref, 'reference', np.float64)
    noisy, out = _window(noisy, 'noisy', np.float64), _window(out, 'output', np.float64)
    _require(not identity or np.array_equal(ref, noisy), 'identity input must equal reference')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        demeaned = ref - ref.mean(axis=1, keepdims=True)
        denominator = np.mean(demeaned ** 2, axis=1, keepdims=True) + 1e-8
        a = float(np.mean((noisy - ref) ** 2 / denominator))
        b = float(np.mean((out - ref) ** 2 / denominator))
    _require(np.isfinite(denominator).all() and math.isfinite(a) and math.isfinite(b), 'nonfinite metric')
    reduction = (a - b) / a if a > 0 and not identity else None
    _require(reduction is None or math.isfinite(reduction), 'nonfinite reduction')
    complex_ratio, phase, undefined = None, None, 0
    if identity:
        frequencies = np.fft.rfftfreq(256, 1.0 / sample_rate_hz)
        mask = (frequencies >= 8.0) & (frequencies <= 40.0)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            f_ref = np.fft.rfft(demeaned, axis=1)[:, mask]
            f_out = np.fft.rfft(out - out.mean(axis=1, keepdims=True), axis=1)[:, mask]
            pr, po = np.abs(f_ref) ** 2, np.abs(f_out) ** 2
            total = float(pr.sum())
            energy = float(np.sum(np.abs(f_out - f_ref) ** 2))
        _require(np.isfinite(pr).all() and np.isfinite(po).all() and math.isfinite(total) and math.isfinite(energy),
                 'nonfinite protected spectrum')
        positive_reference = pr > 1e-12
        support = positive_reference & (po > 1e-12)
        undefined = int(np.count_nonzero(positive_reference & ~support))
        if total > 1e-12:
            complex_ratio = energy / total
            _require(math.isfinite(complex_ratio), 'nonfinite complex energy ratio')
        if undefined == 0 and np.any(support):
            weights = pr[support]
            angles = np.abs(np.angle(f_out[support] * np.conj(f_ref[support])))
            # Identical supported complex bins have exactly zero phase; avoid
            # roundoff from z*conj(z), without ever filling undefined support.
            angles = np.where(f_out[support] == f_ref[support], 0.0, angles)
            phase = float(np.sum(weights * angles) / np.sum(weights))
            _require(math.isfinite(phase), 'nonfinite phase metric')
    return {'input_normalized_mse': a, 'output_normalized_mse': b, 'relative_reduction': reduction,
            'identity_normalized_change': b if identity else None,
            'outcome': 'win' if b < a else 'loss' if b > a else 'tie',
            'protected_complex_energy_ratio': complex_ratio,
            'protected_weighted_abs_phase_radians': phase, 'undefined_phase_bin_count': undefined}


def _validate_case(row: dict) -> tuple:
    required = {'dataset', 'subject_id', 'trial_index', 'condition', 'seed', 'input_normalized_mse',
        'output_normalized_mse', 'relative_reduction', 'identity_normalized_change', 'outcome',
        'protected_complex_energy_ratio', 'protected_weighted_abs_phase_radians', 'undefined_phase_bin_count'}
    _require(type(row) is dict and required <= set(row) and _plain(row), 'invalid case schema/nonfinite')
    dataset, subject, trial, condition = (row[k] for k in ('dataset', 'subject_id', 'trial_index', 'condition'))
    _require(dataset in DATASETS and condition in CONDITIONS and type(subject) is int and subject in range(1, 7),
             'illegal case identity/subject')
    _require(type(trial) is int and 0 <= trial < SHAPES[dataset][0], 'illegal case trial')
    # The validated dataset defines native rate even for legacy rows omitting
    # this optional provenance field; a declared rate must match it exactly.
    if 'sample_rate_hz' in row:
        _require(type(row['sample_rate_hz']) is int and row['sample_rate_hz'] == RATES[dataset],
                 'case sample rate must match native dataset rate')
    _require(type(row['seed']) is int, 'case seed must be int')
    cfg = json.loads(_EXPECTED_CONFIG_JSON)
    # Rows may omit purpose/epoch, but the seed still must encode a legal case.
    candidates = [('training', e) for e in range(1, 19)] if subject <= 4 else [('internal_selection', 0), ('diagnostic', 0)]
    if 'purpose' in row:
        candidates = [(p, e) for p, e in candidates if p == row['purpose']]
    if 'epoch' in row:
        _require(type(row['epoch']) is int, 'row epoch must be int')
        candidates = [(p, e) for p, e in candidates if e == row['epoch']]
    _require(any(row['seed'] == _case_seed(dataset, subject, trial, condition, p, e, cfg) for p, e in candidates),
             'illegal case seed/purpose/epoch')
    a, b = row['input_normalized_mse'], row['output_normalized_mse']
    _require(type(a) is float and type(b) is float and a >= 0 and b >= 0, 'invalid normalized MSE')
    reduction = (a - b) / a if a > 0 and condition != 'identity' else None
    _require(_exact(row['relative_reduction'], reduction), 'inconsistent case reduction')
    _require(_exact(row['identity_normalized_change'], b if condition == 'identity' else None), 'inconsistent identity change')
    _require(row['outcome'] == ('win' if b < a else 'loss' if b > a else 'tie'), 'inconsistent case outcome')
    spectral = ('protected_complex_energy_ratio', 'protected_weighted_abs_phase_radians')
    if condition == 'identity':
        _require(a == 0.0 and type(row['undefined_phase_bin_count']) is int and row['undefined_phase_bin_count'] >= 0,
                 'invalid identity input/undefined bin count')
        for key in spectral:
            _require(row[key] is None or (type(row[key]) is float and row[key] >= 0), 'invalid identity spectral metric')
        _require(row['protected_weighted_abs_phase_radians'] is None or
                 row['protected_weighted_abs_phase_radians'] <= math.pi, 'phase exceeds pi')
        _require(row['undefined_phase_bin_count'] == 0 or row['protected_weighted_abs_phase_radians'] is None,
                 'undefined phase cannot pass as numeric')
    else:
        _require(all(row[key] is None for key in spectral) and type(row['undefined_phase_bin_count']) is int and row['undefined_phase_bin_count'] == 0, 'nonidentity spectral metrics must be null/count zero')
    return dataset, subject, trial, condition


def summarize_cases(rows, *, allow_mixed_datasets=False) -> dict:
    """Return condition mapping, defaulting to a single dataset/subject subset.

    Explicit allow_mixed_datasets=True enables DESCRIPTIVE-only cross-dataset
    means of existing dimensionless normalized errors, complex-energy ratios
    and phase angles. Never aggregate native raw power or recompute spectra:
    each case's metrics already use its native sample rate. Partial condition
    groups are legal. Identity/seed/rate/finite checks apply in both modes.

    B/C must use the default single-dataset scope for gate_summary. Mixed/global
    descriptive summaries MUST NOT be passed as per-dataset gate acceptance.
    Missing or undefined identity support remains null, never a fake zero.
    """
    _require(type(allow_mixed_datasets) is bool, 'allow_mixed_datasets must be bool')
    rows = list(rows)
    _require(bool(rows), 'no cases to summarize')
    identities, seeds, domains = set(), set(), set()
    for row in rows:
        key = _validate_case(row)
        _require(key not in identities and row['seed'] not in seeds, 'duplicate case identity/seed')
        identities.add(key)
        seeds.add(row['seed'])
        domains.add(row['dataset'])
    _require(allow_mixed_datasets or len(domains) == 1,
             'per-dataset summary required; cross-domain pooling needs descriptive-only opt-in')
    result = {}
    for condition in CONDITIONS:
        subset = [r for r in rows if r['condition'] == condition]
        if not subset:
            continue
        a = float(np.mean([r['input_normalized_mse'] for r in subset], dtype=np.float64))
        b = float(np.mean([r['output_normalized_mse'] for r in subset], dtype=np.float64))
        _require(math.isfinite(a) and math.isfinite(b), 'nonfinite mean metric')
        def mean_complete(field):
            values = [r[field] for r in subset]
            if any(v is None for v in values):
                return None
            value = float(np.mean(values, dtype=np.float64))
            _require(math.isfinite(value), 'nonfinite aggregate metric')
            return value
        reduction = (a - b) / a if a > 0 and condition != 'identity' else None
        _require(reduction is None or math.isfinite(reduction), 'nonfinite aggregate reduction')
        result[condition] = {'case_count': len(subset), 'mean_input_normalized_mse': a,
            'mean_output_normalized_mse': b, 'ratio_of_mean_reduction': reduction,
            'mean_identity_normalized_change': mean_complete('identity_normalized_change'),
            'mean_protected_complex_energy_ratio': mean_complete('protected_complex_energy_ratio'),
            'mean_protected_weighted_abs_phase_radians': mean_complete('protected_weighted_abs_phase_radians'),
            'undefined_phase_case_count': sum(r['condition'] == 'identity' and
                (r['undefined_phase_bin_count'] > 0 or r['protected_complex_energy_ratio'] is None or
                 r['protected_weighted_abs_phase_radians'] is None) for r in subset)}
    return result


def gate_summary(summary, cfg) -> dict:
    """DATASET SCOPE ONLY: invoke independently for each dataset, never pool.

    The caller must group by dataset and use summarize_cases(rows) with the
    default allow_mixed_datasets=False. Mixed/global summaries made with the
    descriptive opt-in MUST NOT be supplied or interpreted as per-dataset
    acceptance. A plain condition mapping carries no dataset provenance, so
    passing such a mapping cannot establish its scope here. Thresholds remain
    the frozen per-dataset internal engineering screen, not deployment approval.
    """
    # Scope is a caller precondition; this entry point never authorizes pooled
    # global/condition descriptive output as dataset-specific acceptance.
    validate_config(cfg)
    _require(type(summary) is dict and set(summary) <= set(CONDITIONS) and _plain(summary), 'invalid condition summary')
    fields = {'case_count', 'mean_input_normalized_mse', 'mean_output_normalized_mse',
              'ratio_of_mean_reduction', 'mean_identity_normalized_change',
              'mean_protected_complex_energy_ratio', 'mean_protected_weighted_abs_phase_radians',
              'undefined_phase_case_count'}
    for condition, section in summary.items():
        _require(type(section) is dict and set(section) == fields, 'invalid summary fields')
        _require(type(section['case_count']) is int and section['case_count'] > 0 and
                 type(section['undefined_phase_case_count']) is int and
                 0 <= section['undefined_phase_case_count'] <= section['case_count'], 'invalid summary counts')
        for field in fields - {'case_count', 'undefined_phase_case_count', 'ratio_of_mean_reduction'}:
            value = section[field]
            _require(value is None or (type(value) is float and value >= 0), 'invalid summary error/spectral metric')
        _require(type(section['mean_input_normalized_mse']) is float and
                 type(section['mean_output_normalized_mse']) is float, 'summary MSE must be numeric')
        reduction = section['ratio_of_mean_reduction']
        _require(reduction is None or (type(reduction) is float and reduction <= 1), 'invalid summary reduction')
        if condition == 'identity':
            _require(section['mean_input_normalized_mse'] == 0.0 and reduction is None and
                     section['mean_identity_normalized_change'] == section['mean_output_normalized_mse'],
                     'inconsistent identity summary')
            phase = section['mean_protected_weighted_abs_phase_radians']
            _require(phase is None or phase <= math.pi, 'summary phase exceeds pi')
        else:
            _require(all(section[field] is None for field in ('mean_identity_normalized_change',
                        'mean_protected_complex_energy_ratio', 'mean_protected_weighted_abs_phase_radians')) and
                     section['undefined_phase_case_count'] == 0, 'nonidentity summary has protected metrics')
    thresholds = cfg['internal_research_gates']
    gates = {}
    def compare(condition, field, threshold, minimum=False):
        section = summary.get(condition, {})
        value = section.get(field)
        count = section.get('case_count')
        eligible = type(count) is int and count > 0 and type(value) in (float, int) and math.isfinite(value)
        passed = eligible and (value >= threshold if minimum else value <= threshold)
        gates[f'{condition}.{field}'] = {'passed': bool(passed), 'value': value, 'threshold': threshold,
                                        'comparison': '>=' if minimum else '<='}
    for condition in CONDITIONS[1:]:
        threshold = thresholds['high_frequency_burst_mean_error_reduction_min'] if condition == 'high_frequency_burst' else thresholds['other_corrupted_condition_mean_error_reduction_min']
        compare(condition, 'ratio_of_mean_reduction', threshold, True)
    compare('identity', 'mean_identity_normalized_change', thresholds['identity_mean_normalized_change_max'])
    compare('identity', 'mean_protected_complex_energy_ratio', thresholds['identity_protected_band_complex_energy_ratio_max'])
    compare('identity', 'mean_protected_weighted_abs_phase_radians', thresholds['identity_protected_band_weighted_abs_phase_radians_max'])
    undefined = summary.get('identity', {}).get('undefined_phase_case_count')
    gates['identity.phase_defined'] = {'passed': type(undefined) is int and undefined == 0,
                                      'value': undefined, 'threshold': 0, 'comparison': '=='}
    return {'all_passed': all(g['passed'] for g in gates.values()), 'gates': gates}


def _output_path(root: Path, value: str | Path) -> Path:
    path = safe_path(root, value)
    rel = path.relative_to(safe_path(root, '.')).as_posix()
    _require(rel == CHECKPOINT_RELATIVE or
             (path.parent == safe_path(root, 'analysis_outputs') and path.name.startswith('ssvep_v6') and
              path.name != 'ssvep_v6'), 'output must be new v6 checkpoint or ssvep_v6-prefixed report')
    return path


def assert_new_outputs(root: Path, paths) -> tuple[Path, ...]:
    _require(not isinstance(paths, (str, bytes, Path)), 'outputs must be path collection')
    targets = tuple(_output_path(root, rel) for rel in paths)
    _require(len(set(targets)) == len(targets), 'duplicate output target')
    for path in targets:
        _require(not os.path.lexists(path), f'refuse existing output: {path}')
    return targets


def _unlink_owned(root: Path, path: Path, identity: tuple) -> None:
    """Never follow a swapped parent or remove a competing writer's inode."""
    try:
        path = safe_path(root, path)
        info = path.lstat()
    except (OSError, V6Error):
        return
    if stat.S_ISREG(info.st_mode) and (info.st_dev, info.st_ino) == identity:
        path.unlink()


def publish_bundle(root: Path, payloads: dict[str, bytes]) -> tuple[Path, ...]:
    """Exclusive hardlink publication; on partial failure remove only own inodes.

    Not an all-file transaction for concurrent readers. Never replace/rename an
    existing target, and never retry a failed exclusive link as an overwrite.
    """
    _require(type(payloads) is dict and bool(payloads) and
             all(type(k) is str and type(v) is bytes and bool(v) for k, v in payloads.items()),
             'payloads must be nonempty dict[str, nonempty bytes]')
    targets = assert_new_outputs(root, payloads)
    pending, published = [], []
    try:
        for target, content in zip(targets, payloads.values()):
            safe_path(root, target.parent).mkdir(parents=True, exist_ok=True)
            safe_path(root, target)
            descriptor, name = tempfile.mkstemp(prefix=target.name + '.', suffix='.pending', dir=target.parent)
            temporary = Path(name)
            info = os.fstat(descriptor)
            inode = info.st_dev, info.st_ino
            pending.append((temporary, inode))
            with os.fdopen(descriptor, 'wb') as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        for (temporary, inode), target in zip(pending, targets):
            safe_path(root, temporary)
            safe_path(root, target)
            _require((temporary.lstat().st_dev, temporary.lstat().st_ino) == inode, 'temporary publication inode changed')
            os.link(temporary, target)  # Exclusive: EEXIST must fail, never replace.
            published.append((target, inode))
            _require((safe_path(root, target).lstat().st_dev, target.lstat().st_ino) == inode, 'published inode changed')
    except BaseException:
        for target, inode in published:
            _unlink_owned(root, target, inode)
        raise
    finally:
        for temporary, inode in pending:
            _unlink_owned(root, temporary, inode)
    return targets
