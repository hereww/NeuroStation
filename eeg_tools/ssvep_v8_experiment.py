"""Frozen v8 offline experiment runner; importing this module does no data I/O.

The main session MUST independently freeze the 62-entry training lock. This
module cannot create/repair locks, override the experiment, deploy, or resume.
Check-only reads just config/preregistration/training-lock JSON; NPZ hashes and
v6 loader invocation are deferred until an explicitly authorized full run.

Model adapter: build_model(config, sample_rate_hz), forward_with_aux(noisy)
-> (prediction, {low_delta, high_attenuation}); forward(noisy) -> prediction. No condition,
subject, target, reference, or corruption-family labels enter forward. The
selected epoch's complete native-restored C case rows are saved alongside all
18 epoch summaries; selection and diagnostic seeds are purpose-separated.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any
import copy
import hashlib
import importlib
import io
import json
import math
import os
import random
import stat
import tempfile

ROOT = Path(__file__).absolute().parents[1]
MODEL_VERSION = "learned_denoise_v8_ssvep"
CONFIG_RELATIVE = "configs/learned_denoiser_v8_ssvep.json"
PREREG_RELATIVE = "analysis_outputs/ssvep_v8_training_preregistration_20261008.json"
TRAIN_LOCK_RELATIVE = "analysis_outputs/ssvep_v8_training_lock_20261008.json"
CHECKPOINT_RELATIVE = "models/denoiser_v8_ssvep.pt"
TRAIN_REPORT_RELATIVE = "analysis_outputs/ssvep_v8_training_20261008.json"
CONFIG_SHA256 = '484950220254e0b2c256d8cea2bf4f5d8f9084fe7aa4616f406d5e240553a16a'
PREREG_SHA256 = 'eb15e295c70d7c845d4e8f27b069e923d925fca64da6a9a24f1fa61dc5ff2bb3'
REFERENCE_DIR = "datasets/processed/ssvep_reference_v5_20261007"
MANIFEST_RELATIVE = REFERENCE_DIR + "/manifest.json"
PARTITION_RELATIVE = "analysis_outputs/ssvep_reference_v5_partition_lock_20261007.json"
REFERENCE_MANIFEST_SHA256 = "607617315889697a5bfcea986bf055bf4ed31182e8a69776842c736f481604b6"
PARTITION_LOCK_SHA256 = "cb5e05b9d33fb3034f178392f65817185226756cf4c609dcfec53b4c2d4b8a8f"
DOMAINS = ("Nakanishi2015", "Wearable")
CONDITIONS = ("identity", "transient", "low_frequency_drift", "high_frequency_burst", "common_mode")
SHAPES = {"Nakanishi2015": (180, 1114, 8), "Wearable": (240, 710, 8)}
RATES = {"Nakanishi2015": 256, "Wearable": 250}
NEW_SOURCE_PATHS = (
    "eeg_tools/ssvep_denoise_v8.py", "tests/test_ssvep_denoise_v8.py",
    "eeg_tools/ssvep_v8_experiment.py", "scripts/train_ssvep_denoiser_v8.py",
    "tests/test_ssvep_v8_experiment.py",
)
V6_SOURCE_PATHS = (
    "eeg_tools/ssvep_v6_contract.py", "tests/test_ssvep_v6_contract.py",
    "eeg_tools/ssvep_denoise_v6.py", "scripts/train_ssvep_denoiser_v6.py",
    "tests/test_ssvep_denoise_v6.py", "eeg_tools/ssvep_v6_internal_eval.py",
    "scripts/evaluate_ssvep_denoiser_v6_internal.py", "tests/test_ssvep_v6_internal_eval.py",
)
BOUND_PATHS = ('eeg_tools/ssvep_denoise_v5.py', 'eeg_tools/ssvep_reference_v5.py', 'eeg_tools/ssvep_internal_benchmark_v5.py', 'configs/learned_denoiser_v5_ssvep.json', 'datasets/processed/ssvep_reference_v5_20261007/manifest.json', 'analysis_outputs/ssvep_reference_v5_partition_lock_20261007.json', 'configs/learned_denoiser_v6_ssvep.json', 'datasets/processed/ssvep_reference_v5_20261007/Nakanishi2015/subject_001.npz', 'datasets/processed/ssvep_reference_v5_20261007/Nakanishi2015/subject_001.npz.manifest.json', 'datasets/processed/ssvep_reference_v5_20261007/Nakanishi2015/subject_002.npz', 'datasets/processed/ssvep_reference_v5_20261007/Nakanishi2015/subject_002.npz.manifest.json', 'datasets/processed/ssvep_reference_v5_20261007/Nakanishi2015/subject_003.npz', 'datasets/processed/ssvep_reference_v5_20261007/Nakanishi2015/subject_003.npz.manifest.json', 'datasets/processed/ssvep_reference_v5_20261007/Nakanishi2015/subject_004.npz', 'datasets/processed/ssvep_reference_v5_20261007/Nakanishi2015/subject_004.npz.manifest.json', 'datasets/processed/ssvep_reference_v5_20261007/Nakanishi2015/subject_005.npz', 'datasets/processed/ssvep_reference_v5_20261007/Nakanishi2015/subject_005.npz.manifest.json', 'datasets/processed/ssvep_reference_v5_20261007/Nakanishi2015/subject_006.npz', 'datasets/processed/ssvep_reference_v5_20261007/Nakanishi2015/subject_006.npz.manifest.json', 'datasets/processed/ssvep_reference_v5_20261007/Wearable/subject_001.npz', 'datasets/processed/ssvep_reference_v5_20261007/Wearable/subject_001.npz.manifest.json', 'datasets/processed/ssvep_reference_v5_20261007/Wearable/subject_002.npz', 'datasets/processed/ssvep_reference_v5_20261007/Wearable/subject_002.npz.manifest.json', 'datasets/processed/ssvep_reference_v5_20261007/Wearable/subject_003.npz', 'datasets/processed/ssvep_reference_v5_20261007/Wearable/subject_003.npz.manifest.json', 'datasets/processed/ssvep_reference_v5_20261007/Wearable/subject_004.npz', 'datasets/processed/ssvep_reference_v5_20261007/Wearable/subject_004.npz.manifest.json', 'datasets/processed/ssvep_reference_v5_20261007/Wearable/subject_005.npz', 'datasets/processed/ssvep_reference_v5_20261007/Wearable/subject_005.npz.manifest.json', 'datasets/processed/ssvep_reference_v5_20261007/Wearable/subject_006.npz', 'datasets/processed/ssvep_reference_v5_20261007/Wearable/subject_006.npz.manifest.json', 'analysis_outputs/ssvep_v6_training_preregistration_20261007.json', 'eeg_tools/ssvep_v6_contract.py', 'tests/test_ssvep_v6_contract.py', 'eeg_tools/ssvep_denoise_v6.py', 'scripts/train_ssvep_denoiser_v6.py', 'tests/test_ssvep_denoise_v6.py', 'eeg_tools/ssvep_v6_internal_eval.py', 'scripts/evaluate_ssvep_denoiser_v6_internal.py', 'tests/test_ssvep_v6_internal_eval.py', 'analysis_outputs/ssvep_v6_training_lock_20261007.json', 'analysis_outputs/ssvep_v6_training_20261007.json', 'analysis_outputs/ssvep_v6_internal_diagnostic_20261007.json', 'configs/learned_denoiser_v8_ssvep.json', 'configs/learned_denoiser_v7_ssvep.json', 'analysis_outputs/ssvep_v7_training_preregistration_20261007.json', 'analysis_outputs/ssvep_v7_training_lock_20261007.json', 'analysis_outputs/ssvep_v7_training_20261007.json', 'analysis_outputs/ssvep_v7_internal_diagnostic_patch_20261008.json', 'analysis_outputs/ssvep_v7_saved_internal_audit_patch_20261008.json', 'eeg_tools/ssvep_denoise_v7.py', 'tests/test_ssvep_denoise_v7.py', 'eeg_tools/ssvep_v7_experiment.py', 'scripts/train_ssvep_denoiser_v7.py', 'tests/test_ssvep_v7_experiment.py', 'models/denoiser_v7_ssvep.pt')
TRAIN_LOCK_PATHS = BOUND_PATHS + (PREREG_RELATIVE,) + NEW_SOURCE_PATHS
PROVENANCE = {
    "runtime_enabled": False, "production_validated": False,
    "clean_ground_truth_used": False, "recorded_reference_is_clean_ground_truth": False,
    "outer_validation_waveforms_loaded": False, "holdout_waveforms_loaded": False,
    "outer_validation_used_for_training_or_selection": False,
    "native_units_assumed": False, "cross_dataset_participant_identity_known": False,
    "internal_subjects_are_development_data": True,
}
_SCOPE = {
    "fit_subjects": [1, 2, 3, 4], "internal_selection_subjects": [5, 6],
    "diagnostic_subjects": [5, 6], "outer_validation_subjects_not_loaded": [7, 8],
    "holdout_subjects_not_loaded": [9, 10], "recorded_reference_is_clean_ground_truth": False,
    "runtime_enabled": False, "production_validated": False,
}
_EXPECTED_CONFIG_JSON = r'''
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

_EXPECTED_PREREG_JSON = r'''
{
  "schema_version": 1,
  "status": "formal",
  "experiment": "learned_denoise_v8_ssvep_train_internal_only",
  "created_at_utc": "2026-10-08T03:03:53.631789+00:00",
  "configuration": {
    "path": "configs/learned_denoiser_v8_ssvep.json",
    "sha256": "484950220254e0b2c256d8cea2bf4f5d8f9084fe7aa4616f406d5e240553a16a"
  },
  "base_v7_training_lock": {
    "config_path": "configs/learned_denoiser_v7_ssvep.json",
    "config_sha256": "ac578c9f2343510aeb7f6eeb843497b6e7a8320028fc409d8afd5cdfc660e243",
    "training_lock_path": "analysis_outputs/ssvep_v7_training_lock_20261007.json",
    "training_lock_sha256": "09d0db827f7c19836aa33797a8407f3ba0be7fdd53400bb6cc8b24d0be18ab25"
  },
  "bound_files": {
    "eeg_tools/ssvep_denoise_v5.py": "e912e3fef500eee389e9a36c98f4d64d93a8506b9c0f6635d0909a69108fa5ac",
    "eeg_tools/ssvep_reference_v5.py": "25cdd661870df17a0cb632923912a9bfb95eca923e2181036fcf360513169eea",
    "eeg_tools/ssvep_internal_benchmark_v5.py": "3ee16b4c6ba88e51f50fa757ac8792c9a53eaaef437b120c718b37ec5740f05a",
    "configs/learned_denoiser_v5_ssvep.json": "5c5ea64a992bda5aed30351e572a361d4361269b4d3bdf33ed2e3a7dbed92838",
    "datasets/processed/ssvep_reference_v5_20261007/manifest.json": "607617315889697a5bfcea986bf055bf4ed31182e8a69776842c736f481604b6",
    "analysis_outputs/ssvep_reference_v5_partition_lock_20261007.json": "cb5e05b9d33fb3034f178392f65817185226756cf4c609dcfec53b4c2d4b8a8f",
    "configs/learned_denoiser_v6_ssvep.json": "f1deaf6f28add5f14f24719675fb071f1c27c62befc7bafafc47c3e8c66eb8f0",
    "datasets/processed/ssvep_reference_v5_20261007/Nakanishi2015/subject_001.npz": "9b2539f3294628340016735471470c7923186b9c9a90c44812b7b6b3360856fd",
    "datasets/processed/ssvep_reference_v5_20261007/Nakanishi2015/subject_001.npz.manifest.json": "c9c0258828ef1f4ab16dacf90f2105fa0c6a76515bd44d2225448c24968fe907",
    "datasets/processed/ssvep_reference_v5_20261007/Nakanishi2015/subject_002.npz": "fadc9b5951070fd4328d6cac9aba23a326e437f8c0ad7ff667f37f4d37d9d1a5",
    "datasets/processed/ssvep_reference_v5_20261007/Nakanishi2015/subject_002.npz.manifest.json": "8975f8c3776796d3b4344aaf82f9d3879883c930f7e563de979c56c5820a6c22",
    "datasets/processed/ssvep_reference_v5_20261007/Nakanishi2015/subject_003.npz": "fba3d62fce3ac1ad025bc7679083d0adb4e4e9925164df948332a25ba67b8508",
    "datasets/processed/ssvep_reference_v5_20261007/Nakanishi2015/subject_003.npz.manifest.json": "1c21995cc1bfa2305105b1865cfd781121f1e65982e1360c58a85951827c7e32",
    "datasets/processed/ssvep_reference_v5_20261007/Nakanishi2015/subject_004.npz": "d37f3c8a109665ca61c1a84103359c0a42cf2b9c31ea7e22791742bbc8605c41",
    "datasets/processed/ssvep_reference_v5_20261007/Nakanishi2015/subject_004.npz.manifest.json": "a86c8d135b1b35bb997450c6b08e9e1f12080a7d7bb8fbeebea44254851ce1dd",
    "datasets/processed/ssvep_reference_v5_20261007/Nakanishi2015/subject_005.npz": "71dcf309da1aa2c43d7ff9529f8e4ba7e5e968b454a1c8a8edfe3423fdc9eeec",
    "datasets/processed/ssvep_reference_v5_20261007/Nakanishi2015/subject_005.npz.manifest.json": "ef7ed46e989009e1714739751af2d4bec7c4a854f7236c1f2c633300d54031c0",
    "datasets/processed/ssvep_reference_v5_20261007/Nakanishi2015/subject_006.npz": "d1e7ec1f1cb5dbad51d41b7efa34afe264b637b7030d9a1f245413484b12c172",
    "datasets/processed/ssvep_reference_v5_20261007/Nakanishi2015/subject_006.npz.manifest.json": "536c006853b703550bc11113637321c6a3d5f32bc4bc214aed896aa8ced29825",
    "datasets/processed/ssvep_reference_v5_20261007/Wearable/subject_001.npz": "566b5d62307e748810bc69be3c7102324110ebfb3261d7172e6e1c3baacf2a3e",
    "datasets/processed/ssvep_reference_v5_20261007/Wearable/subject_001.npz.manifest.json": "3e5efcb8633335638b25cd96bb1262b28d6f919af9c93589466a52b9ca282831",
    "datasets/processed/ssvep_reference_v5_20261007/Wearable/subject_002.npz": "4801f424a6537818966c05c2068344ef6fa2531bce0dc1b28bdb9544f5e668c3",
    "datasets/processed/ssvep_reference_v5_20261007/Wearable/subject_002.npz.manifest.json": "376ebcc331b3f80ef4f45a530d92b4c20bcf65ca351ca4d8c573c6ba6d7870ea",
    "datasets/processed/ssvep_reference_v5_20261007/Wearable/subject_003.npz": "b810ac66fca593ebebf21694429b70740a8763dbbb1b271c351de39bf8c8d1c7",
    "datasets/processed/ssvep_reference_v5_20261007/Wearable/subject_003.npz.manifest.json": "57ba3a827e3ca34f3b75a1d1d9bd6d3a3baa4fc5ffe2d4de514d631f1e23fbd9",
    "datasets/processed/ssvep_reference_v5_20261007/Wearable/subject_004.npz": "f315d012a552a2b1de0af0d9fba14e3406e5c74a2096d1ee848565369bc29037",
    "datasets/processed/ssvep_reference_v5_20261007/Wearable/subject_004.npz.manifest.json": "fc7f5704b23c2c2c47d803ee898c06416af48eec4e27226d070c7d7aa99acd2c",
    "datasets/processed/ssvep_reference_v5_20261007/Wearable/subject_005.npz": "f06d8c6ab127d4397e39ffd0a0f38371d6257cc1d7e587e6947c8952a93e6085",
    "datasets/processed/ssvep_reference_v5_20261007/Wearable/subject_005.npz.manifest.json": "c2bcafd8114328525bb158260407965c6b2c3d1c9d0443097c0c561fa9dd26dc",
    "datasets/processed/ssvep_reference_v5_20261007/Wearable/subject_006.npz": "6bbb0426d9c4ed0c429c5c37abe175b781ef263b9686258454d53a056155390d",
    "datasets/processed/ssvep_reference_v5_20261007/Wearable/subject_006.npz.manifest.json": "e29d0f3157b18a9a974e53172ce53f71f76378fe3aeb273cc285144301c2c647",
    "analysis_outputs/ssvep_v6_training_preregistration_20261007.json": "6d98383db26f7afb92302fac4371eb40c3dd31539f282ef1a37fe7440d5c07ec",
    "eeg_tools/ssvep_v6_contract.py": "7907014e9f2ea559a6e88f0ea416d44834e0a2a71aa285f1b5cd75d8a67f9c3a",
    "tests/test_ssvep_v6_contract.py": "9a8152c2d10eeb3f6965da64445a500bd4ceeefc9d99b98da0fabf3262a7aaee",
    "eeg_tools/ssvep_denoise_v6.py": "4172b56005a36ae1be90f923a01e49b1669d6dc5c1123608d59278bed5eb5c75",
    "scripts/train_ssvep_denoiser_v6.py": "4d8d5bba56a63c932fa482da547b075cd6255448bc1c50bbca87f24fbd2ee846",
    "tests/test_ssvep_denoise_v6.py": "301f3a0b6a0159f160aa5e7ca4e2d55d55feb44273c87f38d72973d3df11c956",
    "eeg_tools/ssvep_v6_internal_eval.py": "bf18506f90f06abb3548e9e8f989f7d93494ef894546e33313375bab58bc8d23",
    "scripts/evaluate_ssvep_denoiser_v6_internal.py": "5dde8ad7a283e76e7510c0b0f9b1e122f1490e2696f1c80b1922fd9346c1d511",
    "tests/test_ssvep_v6_internal_eval.py": "2358872379f0a46c1bb10389719b732bbdb7a791c39a8a37338cc095fc458efe",
    "analysis_outputs/ssvep_v6_training_lock_20261007.json": "8512d498efe0263ddc4726e7f7800dcde5979eca3cb603a6fcd2d03f364177d9",
    "analysis_outputs/ssvep_v6_training_20261007.json": "b69d3d45408f858962b0aa90d0751dea2d3cd46a6365c99a68eaafb2464caf68",
    "analysis_outputs/ssvep_v6_internal_diagnostic_20261007.json": "71bd4372d3719afc1d84648fe54c71d5f47b3086e93f1bbd36d051aded460760",
    "configs/learned_denoiser_v8_ssvep.json": "484950220254e0b2c256d8cea2bf4f5d8f9084fe7aa4616f406d5e240553a16a",
    "configs/learned_denoiser_v7_ssvep.json": "ac578c9f2343510aeb7f6eeb843497b6e7a8320028fc409d8afd5cdfc660e243",
    "analysis_outputs/ssvep_v7_training_preregistration_20261007.json": "2e8c2b123a9d292b2d37c36b9aa21020dcafbe97a769a50ab8fafb2e06be23cd",
    "analysis_outputs/ssvep_v7_training_lock_20261007.json": "09d0db827f7c19836aa33797a8407f3ba0be7fdd53400bb6cc8b24d0be18ab25",
    "analysis_outputs/ssvep_v7_training_20261007.json": "78935a3255bcf3b7ee577a16f804fc1e11ca80bc79b99d0e3dc5992629bb4235",
    "analysis_outputs/ssvep_v7_internal_diagnostic_patch_20261008.json": "7a1b3c1bde7f7029e8c92a7b48b6eaec5b6301f5cfca3ce76186717f291ea7a2",
    "analysis_outputs/ssvep_v7_saved_internal_audit_patch_20261008.json": "fe2cfc9749ab5b839ae1a9611dc5e149f490cbe6cdefea723ec25dcd4df7dd57",
    "eeg_tools/ssvep_denoise_v7.py": "e7d351e894b8f309cb443384624447590869b5964a1e5de3735e79a925cbc444",
    "tests/test_ssvep_denoise_v7.py": "febe88f1eb9e745233fb769b386cbda800a79718c241a4408e4b41913e330886",
    "eeg_tools/ssvep_v7_experiment.py": "44ca201b235fac796743e9c6870c4dffe6a131303ad624140d72db2c60265605",
    "scripts/train_ssvep_denoiser_v7.py": "1a05f477a572ff4537ae88bec4b4ee819fd678cbebd3d75b4717c0061e4a836d",
    "tests/test_ssvep_v7_experiment.py": "5250f2dbd9464a5739122e660eaaa5f76c9475c6388a95988ccf6c42a30af3a1",
    "models/denoiser_v7_ssvep.pt": "a79cc4665ff8723d797054a0b81ea00aacda09b80621f1150c3918931a5a3bfe"
  },
  "model_design": {
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
  "loss_design": {
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
  "training_schedule": {
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
  "internal_gates": {
    "high_frequency_burst_mean_error_reduction_min": 0.05,
    "other_corrupted_condition_mean_error_reduction_min": 0.0,
    "identity_mean_normalized_change_max": 0.0005,
    "identity_protected_band_complex_energy_ratio_max": 1e-06,
    "identity_protected_band_weighted_abs_phase_radians_max": 0.02,
    "all_gates_required_per_dataset": true,
    "interpretation": "internal engineering screen only, not independent task preservation or production acceptance"
  },
  "scope": {
    "fit_subjects": [
      1,
      2,
      3,
      4
    ],
    "internal_selection_subjects": [
      5,
      6
    ],
    "diagnostic_subjects": [
      5,
      6
    ],
    "outer_validation_subjects_not_loaded": [
      7,
      8
    ],
    "holdout_subjects_not_loaded": [
      9,
      10
    ],
    "recorded_reference_is_clean_ground_truth": false,
    "runtime_enabled": false,
    "production_validated": false
  },
  "development_history": "v5/v6/v7 outcomes and reused internal subjects5-6 inform this v8 engineering hypothesis; these are development data, not independent acceptance",
  "rationale": "Hypothesis only: one high-band scalar per channel/window in v7 may attenuate unaffected high-frequency bins together with a localized burst. V8 predicts one non-amplifying real attenuation per channel/bin from log-magnitude spectral features. Initial identity, low branch, loss and all gates remain fixed; gain floor changes from0.5 to0.2 as an explicit architecture change, not a gate relaxation. No causal explanation of v7 failure has been established from scalar reports.",
  "limits": [
    "Noncausal fixed256-point rectangularFFT; no continuous-stream/HannOLA/task/clinical acceptance",
    "Raw high-gate clamp can saturate; no success guarantee",
    "Zero-init exact identity; gain in[0.2,1] in nominal >40Hz FFT bins",
    "Only signal magnitude in high branch; no subject/condition/family/reference/target labels in forward",
    "Fit subjects1-4, selection5-6; no outer7-8/holdout9-10; no D drive read",
    "18 epochs, one architecture, no extension/sweep/resume after failure; deployment blocked regardless of internal pass"
  ],
  "new_source_files": [
    "eeg_tools/ssvep_denoise_v8.py",
    "tests/test_ssvep_denoise_v8.py",
    "eeg_tools/ssvep_v8_experiment.py",
    "scripts/train_ssvep_denoiser_v8.py",
    "tests/test_ssvep_v8_experiment.py"
  ],
  "expected_training_lock_file_count": 62,
  "expected_diagnostic_lock_file_count": 65,
  "reference_npz_byte_reads_for_v8_before_freeze": 0
}
'''


class V8Error(ValueError):
    """Fail-closed offline experiment violation."""


def _require(ok, message):
    if not ok:
        raise V8Error(message)


def _plain(value):
    if value is None or type(value) in (bool, int, str):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(_plain(v) for v in value)
    if type(value) is dict:
        return all(type(k) is str and _plain(v) for k, v in value.items())
    return False


def _exact(a, b):
    if type(a) is not type(b):
        return False
    if type(b) is dict:
        return set(a) == set(b) and all(_exact(a[k], v) for k, v in b.items())
    if type(b) is list:
        return len(a) == len(b) and all(_exact(x, y) for x, y in zip(a, b))
    return a == b


def _required(value, expected, context):
    _require(type(value) is dict and all(k in value and _exact(value[k], v)
             for k, v in expected.items()), context + ': required values/types mismatch')


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_config(cfg):
    _require(_plain(cfg) and _exact(cfg, json.loads(_EXPECTED_CONFIG_JSON)),
             'fixed formal v8 configuration mismatch; no overrides')


def _contract():
    return importlib.import_module('eeg_tools.ssvep_v6_contract')


def _torch():
    return importlib.import_module('torch')


def _model_api():
    return importlib.import_module('eeg_tools.ssvep_denoise_v8')


def _protocol():
    return json.loads(_EXPECTED_PREREG_JSON)


def safe_path(root, value):
    """Reject lexical escape/D:/UNC/ADS before stat; reject all reparse ancestors."""
    for text in (str(root), str(value)):
        _require(bool(text) and '\x00' not in text and '..' not in text.replace('\\', '/').split('/'),
                 'empty/NUL/traversal path forbidden')
        win = PureWindowsPath(text)
        _require(win.drive.casefold() != 'd:', 'D: access forbidden')
        _require(not win.drive or (Path(text).is_absolute() and not win.drive.startswith('\\\\')),
                 'foreign/relative drive or UNC forbidden')
        _require(not any(':' in p for p in win.parts[1:]) and (bool(win.drive) or ':' not in text),
                 'alternate stream forbidden')
        _require(not any(p.endswith((' ', '.')) for p in win.parts if p not in ('.', '\\', '/')),
                 'ambiguous Windows path forbidden')
        _require(not any(p.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL',
                 *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}
                 for p in win.parts), 'device path forbidden')
    base = Path(os.path.abspath(root))
    path = Path(value) if Path(value).is_absolute() else base / value
    path = Path(os.path.abspath(path))
    try:
        rel = path.relative_to(base)
    except ValueError as exc:
        raise V8Error('external workspace path forbidden') from exc
    _require(not any(p.casefold() in {'outer', 'holdout'} for p in rel.parts), 'outer/holdout path forbidden')
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


def _stamp(root, rel):
    try:
        info = safe_path(root, rel).lstat()
    except OSError as exc:
        raise V8Error(f'missing/inaccessible frozen file: {rel}') from exc
    _require(stat.S_ISREG(info.st_mode) and info.st_size > 0, f'nonempty regular file required: {rel}')
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read_bytes(root, rel):
    """Text inputs only here; NPZ/PT must go through the guarded v6 contract."""
    _require(str(rel) in set(TRAIN_LOCK_PATHS) | {TRAIN_LOCK_RELATIVE}, 'unlisted input path')
    _require(str(rel).endswith(('.json', '.py')), 'binary reads forbidden in metadata reader')
    before = _stamp(root, rel)
    try:
        fd = os.open(safe_path(root, rel), os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0))
        with os.fdopen(fd, 'rb') as handle:
            info = os.fstat(handle.fileno())
            descriptor = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
            _require(descriptor[:4] == before[:4], 'file replaced before read')
            data = handle.read()
            after = os.fstat(handle.fileno())
            _require(descriptor == (after.st_dev, after.st_ino, after.st_size,
                     after.st_mtime_ns, after.st_ctime_ns), 'file changed during read')
    except OSError as exc:
        raise V8Error(f'cannot read frozen file: {rel}') from exc
    _require(_stamp(root, rel) == before and len(data) == before[2], 'frozen path changed during read')
    return data


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'duplicate JSON key: ' + key)
        result[key] = value
    return result


def _parse_json(data):
    try:
        obj = json.loads(data.decode('utf-8-sig'), object_pairs_hook=_unique_pairs,
                         parse_constant=lambda token: (_ for _ in ()).throw(V8Error('nonfinite JSON: ' + token)))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise V8Error('invalid strict JSON') from exc
    _require(type(obj) is dict and _plain(obj), 'metadata must be finite plain JSON object')
    return obj


def _metadata(root, rel, expected_sha=None):
    data = _read_bytes(root, rel)
    digest = hashlib.sha256(data).hexdigest()
    _require(expected_sha is None or digest == expected_sha, 'frozen SHA mismatch: ' + rel)
    return _parse_json(data), digest


def load_config(root=ROOT):
    cfg, _ = _metadata(root, CONFIG_RELATIVE, CONFIG_SHA256)
    validate_config(cfg)
    return cfg


def _bindings(files, paths, context):
    _require(type(files) is dict and set(files) == set(paths) and len(files) == len(paths),
             context + ': exact file set required')
    _require(all(_sha(h) for h in files.values()), context + ': lowercase SHA256 required')


def validate_preregistration(protocol, cfg):
    validate_config(cfg)
    _require(_exact(protocol, _protocol()), 'fixed v8 preregistration mismatch')
    _required(protocol, {
        'schema_version': 1, 'status': 'formal',
        'experiment': 'learned_denoise_v8_ssvep_train_internal_only',
        'configuration': {'path': CONFIG_RELATIVE, 'sha256': CONFIG_SHA256},
        'base_v7_training_lock': cfg['base_reference_contract'],
        'model_design': cfg['model'], 'loss_design': cfg['loss'],
        'training_schedule': cfg['training'], 'internal_gates': cfg['internal_research_gates'],
        'scope': _SCOPE, 'new_source_files': list(NEW_SOURCE_PATHS),
        'expected_training_lock_file_count': 62, 'expected_diagnostic_lock_file_count': 65,
        'reference_npz_byte_reads_for_v8_before_freeze': 0,
    }, 'frozen preregistration')
    _bindings(protocol.get('bound_files'), BOUND_PATHS, '56 preregistration bindings')
    _require(protocol['bound_files'][CONFIG_RELATIVE] == CONFIG_SHA256, 'config binding mismatch')
    for k in ('config_path', 'training_lock_path'):
        base = cfg['base_reference_contract']
        _require(protocol['bound_files'][base[k]] == base[k.replace('_path', '_sha256')], 'base v6 binding mismatch')


def validate_training_lock(lock, protocol):
    """Only the pre-read training lock; never accept diagnostic or v6 locks."""
    _require(_plain(lock), 'lock must be plain finite metadata')
    _require(_exact(protocol, _protocol()), 'lock requires frozen v8 preregistration')
    _required(lock, {'schema_version': 1, 'status': 'formal',
        'stage': 'frozen_before_v8_training_waveform_read',
        'checkpoint_loaded': False, 'waveforms_loaded': False}, 'v8 training lock')
    _bindings(lock.get('files'), TRAIN_LOCK_PATHS, '62-entry v8 training lock')
    _require(lock['files'][PREREG_RELATIVE] == PREREG_SHA256, 'training lock/preregistration SHA mismatch')
    for p, digest in protocol['bound_files'].items():
        _require(lock['files'][p] == digest, 'lock/preregistration binding mismatch: ' + p)
    unsafe = set(k for k, v in PROVENANCE.items() if v is False) | {
        'runtime', 'production', 'outer', 'holdout', 'enabled', 'checkpoint_loaded',
        'waveforms_loaded', 'cuda_initialized', 'outer_validation_used_for_selection',
        'holdout_waveform_loaded', 'outer_validation_waveform_loaded',
    }
    scope = lock.get('scope')
    _required(scope, {k: False for k in ('runtime_enabled', 'production_validated',
        'holdout_waveforms_loaded', 'outer_validation_waveforms_loaded',
        'outer_validation_used_for_training_or_selection')}, 'lock safety scope')
    def inspect(value):
        if type(value) is dict:
            for key, item in value.items():
                if key in unsafe:
                    _require(item is False, 'unsafe lock flag: ' + key)
                inspect(item)
        elif type(value) is list:
            for item in value:
                inspect(item)
    inspect(lock)
    def scope_false(value):
        if type(value) is bool:
            return value is False
        if type(value) is dict:
            return all(scope_false(v) for v in value.values())
        return False
    _require(scope_false(scope), 'lock scope must contain only explicitly false safety flags')
    return copy.deepcopy(lock['files'])


@dataclass(frozen=True)
class FrozenExperiment:
    root: Path
    config: dict
    preregistration: dict
    files: dict
    lock_sha256: str | None
    check_only: bool
    stage: str = 'training'

    def report(self):
        return {'schema_version': 1, 'status': 'formal', 'stage': self.stage,
            'check_only': self.check_only, 'lock_sha256': self.lock_sha256,
            'training_lock_present': self.lock_sha256 is not None,
            'readiness': 'metadata_valid_binary_verification_deferred' if self.lock_sha256 else 'awaiting_62_entry_training_lock',
            'expected_training_lock_file_count': 62, 'binary_hashes_complete': False,
            'metadata_only': True, 'waveforms_loaded': False, 'checkpoint_loaded': False,
            'cuda_initialized': False, 'read_state_scope': 'this_call_only_not_experiment_history', **PROVENANCE}


def preflight(root=ROOT, *, check_only=True, stage='training'):
    _require(type(check_only) is bool and stage == 'training', 'training-only preflight; no runtime/prod/outer/holdout')
    base = safe_path(root, '.')
    cfg = load_config(base)
    protocol, _ = _metadata(base, PREREG_RELATIVE, PREREG_SHA256)
    validate_preregistration(protocol, cfg)
    lock_path = safe_path(base, TRAIN_LOCK_RELATIVE)
    if not os.path.lexists(lock_path):
        _require(check_only, 'v8 62-entry training lock required BEFORE any true data read')
        return FrozenExperiment(base, cfg, protocol, {}, None, True)
    lock, digest = _metadata(base, TRAIN_LOCK_RELATIVE)
    files = validate_training_lock(lock, protocol)
    state = FrozenExperiment(base, cfg, protocol, files, digest, check_only)
    validate_frozen(state)
    return state


def _require_training_preflight(state):
    _require(type(state) is FrozenExperiment and state.stage == 'training' and state.check_only is False
             and _sha(state.lock_sha256), 'full frozen v8 training preflight required')


def validate_frozen(state):
    """Recheck bindings. Check-only opens ONLY three metadata JSONs.

    Full-run checks hash text/source entries, never waveform/checkpoint bytes.
    The guarded loader delegates binary verification and record loading to v6
    after this v8 62-entry freeze has passed. No lock writer exists here.
    """
    _require(type(state) is FrozenExperiment and type(state.check_only) is bool and state.stage == 'training',
             'invalid frozen v8 state')
    cfg = load_config(state.root)
    protocol, _ = _metadata(state.root, PREREG_RELATIVE, PREREG_SHA256)
    validate_preregistration(protocol, cfg)
    _require(_exact(cfg, state.config) and _exact(protocol, state.preregistration), 'frozen config/preregistration changed')
    if state.lock_sha256 is None:
        _require(state.check_only and state.files == {}, 'training lock required')
        _require(not os.path.lexists(safe_path(state.root, TRAIN_LOCK_RELATIVE)), 'training lock appeared; redo preflight')
        return
    lock, digest = _metadata(state.root, TRAIN_LOCK_RELATIVE, state.lock_sha256)
    files = validate_training_lock(lock, protocol)
    _require(digest == state.lock_sha256 and _exact(files, state.files), 'frozen lock bindings changed')
    if not state.check_only:
        for rel, expected in files.items():
            # Binary inputs are verified by their guarded contracts at the
            # first true read boundary.  The v8 metadata reader is deliberately
            # text-only; this includes historical checkpoints retained solely
            # as provenance bindings.
            if rel.endswith(('.npz', '.pt')):
                continue
            _require(hashlib.sha256(_read_bytes(state.root, rel)).hexdigest() == expected,
                     'frozen text/source SHA mismatch: ' + rel)


verify_frozen = validate_frozen


def load_training_records(state):
    """First true read boundary: verify v8, then v6 hashes, then v6 loader."""
    _require_training_preflight(state)
    validate_frozen(state)
    api = _contract()
    base = api.preflight(root=state.root, stage='training', check_only=False)
    _require(all(state.files.get(p) == h for p, h in base.files.items()), 'v6/v8 reference contract divergence')
    validate_frozen(state)
    api.verify_frozen(base)
    records = api.load_training_records(base)
    validate_records(records, state.config, files=state.files)
    api.verify_frozen(base)
    validate_frozen(state)
    return records


def validate_records(records, cfg, *, files=None):
    validate_config(cfg)
    np = importlib.import_module('numpy')
    expected = {(d, s) for d in DOMAINS for s in range(1, 7)}
    _require(type(records) in (list, tuple) and len(records) == 12, 'exactly 12 subjects1-6 records required')
    seen = set()
    for record in records:
        d, s = record.dataset, record.subject_id
        _require(type(d) is str and type(s) is int and (d, s) in expected and (d, s) not in seen,
                 'missing/duplicate/forbidden record identity')
        seen.add((d, s))
        arr = record.signal
        _require(type(record.sample_rate_hz) is int and record.sample_rate_hz == RATES[d]
                 and isinstance(arr, np.ndarray) and arr.shape == SHAPES[d]
                 and arr.dtype == np.float64 and np.isfinite(arr).all(), 'frozen record shape/rate/dtype mismatch')
        if files is not None:
            rel = f'{REFERENCE_DIR}/{d}/subject_{s:03d}.npz'
            _require(record.npz_sha256 == files[rel] and record.sidecar_sha256 == files[rel + '.manifest.json'],
                     'record provenance differs from v8 lock')
    _require(seen == expected, 'incomplete reference cohort')


def case_seed(dataset, subject_id, trial_index, condition, cfg, *, purpose, epoch):
    validate_config(cfg)
    _require(dataset in DOMAINS and condition in CONDITIONS and purpose in ('training', 'internal_selection', 'diagnostic'),
             'invalid dataset/condition/purpose')
    subjects = (1, 2, 3, 4) if purpose == 'training' else (5, 6)
    _require(type(subject_id) is int and subject_id in subjects, 'purpose subject allowlist violation')
    _require(type(trial_index) is int and 0 <= trial_index < SHAPES[dataset][0], 'invalid trial index')
    _require(type(epoch) is int and (1 <= epoch <= 18 if purpose == 'training' else epoch == 0), 'invalid purpose epoch')
    return (cfg['seeds'][purpose] + DOMAINS.index(dataset) * 10000000 + subject_id * 100000
            + trial_index * 10 + CONDITIONS.index(condition) + epoch * 100000000)


def corrupt_case(reference, dataset, subject_id, trial_index, condition, cfg, *, purpose, epoch):
    seed = case_seed(dataset, subject_id, trial_index, condition, cfg, purpose=purpose, epoch=epoch)
    api = _contract()
    ref = api._window(reference, 'reference', 'float64')
    if condition == 'identity':
        return ref.copy(), seed
    override = copy.deepcopy(cfg['corruption'])
    override.update(identity_probability=0.0, combine_probability=0.0, families=[condition])
    core = importlib.import_module('eeg_tools.ssvep_denoise_v5')
    noisy, identity, families = core.corrupt_window(ref.copy(), RATES[dataset], {'corruption': override}, seed)
    _require(identity is False and families == [condition], 'single-condition corruption contract violated')
    return api._window(noisy, 'corrupted output', 'float64'), seed


class ReferenceWindowDataset:
    """In-memory windows only; no loader or I/O. Training/selection cohort exact."""
    def __init__(self, records, domain, cfg, *, purpose, epoch):
        validate_records(records, cfg)
        _require(domain in DOMAINS and purpose in ('training', 'internal_selection'), 'invalid training dataset/purpose')
        self.cfg, self.domain, self.purpose = copy.deepcopy(cfg), domain, purpose
        subjects = (1, 2, 3, 4) if purpose == 'training' else (5, 6)
        self.records = sorted((r for r in records if r.dataset == domain and r.subject_id in subjects), key=lambda r: r.subject_id)
        self.entries = [(i, t, c) for i, r in enumerate(self.records) for t in range(SHAPES[domain][0]) for c in CONDITIONS]
        self.set_epoch(epoch)

    def set_epoch(self, epoch):
        case_seed(self.domain, self.records[0].subject_id, 0, 'identity', self.cfg, purpose=self.purpose, epoch=epoch)
        self.epoch = epoch

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        np = importlib.import_module('numpy')
        ri, trial, condition = self.entries[index]
        record = self.records[ri]
        start = self.cfg['start_samples'][self.domain]
        ref = np.ascontiguousarray(record.signal[trial, start:start + 256, :].T, dtype=np.float64)
        noisy, seed = corrupt_case(ref, self.domain, record.subject_id, trial, condition, self.cfg,
                                  purpose=self.purpose, epoch=self.epoch)
        norm, mean, rms = _contract().normalize_window(noisy)
        target = ((ref - mean) / rms).astype(np.float32)
        _require(np.isfinite(target).all(), 'nonfinite normalized target')
        return {'noisy': norm, 'target': target, 'identity': condition == 'identity',
            'ref64': ref, 'noisy64': noisy, 'rms64': rms, 'dataset': self.domain,
            'subject_id': record.subject_id, 'trial': trial, 'condition': condition,
            'seed': seed, 'sample_rate_hz': record.sample_rate_hz, 'purpose': self.purpose, 'epoch': self.epoch}



def compute_loss(pred, target, noisy, identity, aux, cfg, *, conditions):
    """Canonical model loss, with runner-only string-label and aux schema guards.

    Labels are consumed ONLY here; forward_with_aux receives x alone. Delegate
    numeric equations and sufficient statistics to the frozen v8 model module
    so model/runner cannot silently use different weighting or denominators.
    """
    validate_config(cfg)
    _require(type(conditions) in (list, tuple) and all(type(c) is str and c in CONDITIONS for c in conditions),
             'loss conditions must be the fixed string-label list, no combinations')
    _require(type(aux) is dict and set(aux) == {'low_delta', 'high_attenuation'}, 'invalid model auxiliary schema')
    try:
        return _model_api().compute_loss(pred, target, noisy, identity, aux, cfg, conditions=conditions)
    except (ValueError, TypeError) as exc:
        raise V8Error('invalid v8 loss input: ' + str(exc)) from exc


class LossAccumulator:
    """Canonical model accumulator, case/weight-correct global epoch means."""
    def __init__(self, cfg):
        validate_config(cfg)
        self._accumulator = _model_api().LossAccumulator()

    def update(self, losses):
        try:
            self._accumulator.update(losses)
        except (ValueError, TypeError, KeyError) as exc:
            raise V8Error('invalid loss sufficient statistics: ' + str(exc)) from exc

    def result(self):
        try:
            result = self._accumulator.result()
        except ValueError as exc:
            raise V8Error(str(exc)) from exc
        _require(_plain(result), 'nonfinite aggregated loss')
        return result


def _validated_v6_rows(rows, cfg, *, purpose, domain, complete):
    """Validate v8 provenance first, translate ONLY seed for v6 numeric validator.

    v6 has its own frozen seed bases: pass copies to v6 without mutating saved
    v8 rows or modifying v6. Saved rows retain actual v8 purpose/epoch/seed.
    """
    validate_config(cfg)
    _require(purpose in ('internal_selection', 'diagnostic') and domain in DOMAINS,
             'C rows limited to one domain/subjects5-6')
    _require(type(rows) is list and bool(rows) and all(type(r) is dict and _plain(r) for r in rows), 'invalid C rows')
    api = _contract()
    base_cfg = json.loads(api._EXPECTED_CONFIG_JSON)
    seen, mapped = set(), []
    fields = {'dataset', 'subject_id', 'trial_index', 'condition', 'seed', 'purpose', 'epoch', 'sample_rate_hz',
        'input_normalized_mse', 'output_normalized_mse', 'relative_reduction', 'identity_normalized_change',
        'outcome', 'protected_complex_energy_ratio', 'protected_weighted_abs_phase_radians', 'undefined_phase_bin_count'}
    for row in rows:
        _require(set(row) == fields, 'exact C case schema required; no hidden runtime/acceptance flags')
        d, sid, trial, cond = (row.get(k) for k in ('dataset', 'subject_id', 'trial_index', 'condition'))
        _require(d == domain and _exact(row.get('sample_rate_hz'), RATES[domain])
                 and row.get('purpose') == purpose and _exact(row.get('epoch'), 0), 'C row scope/rate/purpose mismatch')
        seed = case_seed(d, sid, trial, cond, cfg, purpose=purpose, epoch=0)
        _require(_exact(row.get('seed'), seed), 'C row v8 seed mismatch')
        key = (sid, trial, cond)
        _require(key not in seen, 'duplicate C case')
        seen.add(key)
        translated = dict(row)
        translated['seed'] = api._case_seed(d, sid, trial, cond, purpose, 0, base_cfg)
        api._validate_case(translated)
        mapped.append(translated)
    if complete:
        expected = {(s, t, c) for s in (5, 6) for t in range(SHAPES[domain][0]) for c in CONDITIONS}
        _require(seen == expected, 'complete internal subjects5-6 x trials x five conditions required')
    return mapped


def summarize_cases(rows, cfg, *, purpose='internal_selection', domain, complete=True):
    _require(type(complete) is bool, 'complete must be bool')
    mapped = _validated_v6_rows(rows, cfg, purpose=purpose, domain=domain, complete=complete)
    return _contract().summarize_cases(mapped)  # single domain, never pooling


def gate_summary(summary, cfg):
    validate_config(cfg)
    api = _contract()
    base_cfg = json.loads(api._EXPECTED_CONFIG_JSON)
    _require(_exact(base_cfg['internal_research_gates'], cfg['internal_research_gates']), 'v6/v8 gate thresholds differ')
    return api.gate_summary(summary, base_cfg)


def selection_key(summary, gates, internal_total_loss, epoch):
    _require(type(epoch) is int and 1 <= epoch <= 18, 'selection candidates only epochs1-18')
    _require(type(internal_total_loss) is float and math.isfinite(internal_total_loss)
             and internal_total_loss >= 0, 'invalid selection loss')
    _require(type(summary) is dict and set(summary) == set(CONDITIONS), 'five selection conditions required')
    _require(type(gates) is dict and type(gates.get('all_passed')) is bool, 'invalid gate result')
    ratios = []
    for c in CONDITIONS[1:]:
        a, b = summary[c]['mean_input_normalized_mse'], summary[c]['mean_output_normalized_mse']
        _require(type(a) is float and a > 0 and math.isfinite(a)
                 and type(b) is float and b >= 0 and math.isfinite(b), 'invalid condition selection means')
        ratios.append(b / a)
    identity = summary['identity']['mean_output_normalized_mse']
    _require(type(identity) is float and math.isfinite(identity) and identity >= 0, 'invalid identity selection error')
    key = (0 if gates['all_passed'] else 1, max(ratios), identity, internal_total_loss, epoch)
    _require(all(math.isfinite(x) for x in key), 'nonfinite selection key')
    return key


def cublas_environment():
    value = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    _require(value is None or value == ':4096:8', 'conflicting CUBLAS_WORKSPACE_CONFIG')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def set_seed(seed):
    _require(type(seed) is int and seed >= 0, 'seed must be nonnegative int')
    np, torch = importlib.import_module('numpy'), _torch()
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))  # ONLY legacy overall NumPy RNG needs truncation.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_training(cfg):
    validate_config(cfg)
    cublas_environment()  # BEFORE importing torch/model or initializing CUDA
    torch = _torch()
    _require(not torch.cuda.is_initialized(), 'CUDA already initialized; use a fresh frozen run process')
    _require(torch.cuda.is_available(), 'CUDA:0 required; no CPU fallback')
    torch.set_num_threads(cfg['training']['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(0)
    set_seed(cfg['training']['seed'])
    return 'cuda:0'


def _batch_integer(value, name):
    if type(value) is int:
        return value
    torch = _torch()
    _require(isinstance(value, torch.Tensor) and value.ndim == 0
             and value.dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8),
             'batch ' + name + ' must be integer scalar, never bool/float')
    return int(value.item())


def run_epoch(model, loader, cfg, device, optimizer=None):
    """CPU ONLY for synthetic tests; full train_domain refuses CPU fallback."""
    validate_config(cfg)
    torch, api = _torch(), _contract()
    training = optimizer is not None
    model.train(training)
    accumulator, rows, seen, epoch_scope = LossAccumulator(cfg), [], set(), None
    with (torch.enable_grad() if training else torch.inference_mode()):
        for batch in loader:
            noisy = batch['noisy'].to(device=device, dtype=torch.float32)
            target = batch['target'].to(device=device, dtype=torch.float32)
            identity = batch['identity'].to(device=device, dtype=torch.bool)
            n, conditions = noisy.shape[0], batch['condition']
            # Guard BEFORE a forward/optimizer step, not just at save time.
            _require(type(conditions) in (list, tuple) and len(conditions) == n, 'invalid batch conditions')
            for i in range(n):
                d, sid, t = batch['dataset'][i], _batch_integer(batch['subject_id'][i], 'subject'), _batch_integer(batch['trial'][i], 'trial')
                purpose, epoch = batch['purpose'][i], _batch_integer(batch['epoch'][i], 'epoch')
                _require(purpose == ('training' if training else 'internal_selection'), 'batch purpose mismatch')
                _require(_batch_integer(batch['sample_rate_hz'][i], 'rate') == RATES.get(d)
                         and _batch_integer(batch['seed'][i], 'seed') == case_seed(d, sid, t, conditions[i], cfg, purpose=purpose, epoch=epoch),
                         'batch provenance/rate/seed mismatch')
                scope = (d, purpose, epoch)
                _require(epoch_scope is None or scope == epoch_scope, 'epoch domain/purpose/seed epoch must remain fixed')
                epoch_scope = scope
                identity_key = (d, sid, t, conditions[i])
                _require(identity_key not in seen, 'duplicate epoch case identity')
                seen.add(identity_key)
            _require(len(set(batch['dataset'])) == 1, 'no cross-domain optimization/selection pooling')
            if training:
                optimizer.zero_grad(set_to_none=True)
            pred, aux = model.forward_with_aux(noisy)  # x ONLY, absolutely no labels
            losses = compute_loss(pred, target, noisy, identity, aux, cfg, conditions=conditions)
            accumulator.update(losses)
            if training:
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['gradient_clip_norm'], error_if_nonfinite=True)
                optimizer.step()
                _require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'nonfinite model after optimizer step')
                _require(all(bool(torch.isfinite(v).all()) for state in optimizer.state.values()
                         for v in state.values() if isinstance(v, torch.Tensor)), 'nonfinite optimizer state')
            else:
                predictions, normalized = pred.cpu().numpy(), noisy.cpu().numpy()
                for i in range(n):
                    ref = batch['ref64'][i].cpu().numpy()
                    corrupted, rms = batch['noisy64'][i].cpu().numpy(), batch['rms64'][i].cpu().numpy()
                    out = api.restore_window(corrupted, normalized[i], predictions[i], rms)
                    metrics = api.case_metrics(ref, corrupted, out, identity=bool(identity[i]),
                                               sample_rate_hz=int(batch['sample_rate_hz'][i]))
                    rows.append({**metrics, 'dataset': batch['dataset'][i], 'subject_id': int(batch['subject_id'][i]),
                        'trial_index': int(batch['trial'][i]), 'condition': conditions[i], 'seed': int(batch['seed'][i]),
                        'sample_rate_hz': int(batch['sample_rate_hz'][i]), 'purpose': 'internal_selection', 'epoch': 0})
    return accumulator.result(), rows


def train_domain(records, domain, cfg, device, *, frozen_state):
    _require_training_preflight(frozen_state)
    validate_config(cfg)
    _require(_exact(cfg, frozen_state.config), 'trainer config differs from frozen state')
    _require(device == 'cuda:0' and domain in DOMAINS, 'actual domain training requires CUDA:0/native domain')
    validate_frozen(frozen_state)
    torch = _torch()
    _require(torch.cuda.is_available() and torch.cuda.current_device() == 0
             and torch.are_deterministic_algorithms_enabled()
             and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
             and not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
             and not torch.backends.cudnn.benchmark and torch.backends.cudnn.deterministic,
             'fixed deterministic CUDA:0 environment required')
    validate_records(records, cfg, files=frozen_state.files)
    set_seed(cfg['training']['seed'])  # independent fresh domain models, never warm-start
    fit = ReferenceWindowDataset(records, domain, cfg, purpose='training', epoch=1)
    internal = ReferenceWindowDataset(records, domain, cfg, purpose='internal_selection', epoch=0)
    expected_fit, expected_internal = SHAPES[domain][0] * 20, SHAPES[domain][0] * 10
    _require(len(fit) == expected_fit and len(internal) == expected_internal, 'frozen case counts mismatch')
    generator = torch.Generator(device='cpu').manual_seed(cfg['training']['seed'])
    common = {'batch_size': cfg['training']['batch_size'], 'num_workers': 0, 'drop_last': False}
    fit_loader = torch.utils.data.DataLoader(fit, shuffle=True, generator=generator, **common)
    internal_loader = torch.utils.data.DataLoader(internal, shuffle=False, **common)
    model = _model_api().build_model(cfg, RATES[domain]).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'],
        weight_decay=cfg['training']['weight_decay'], foreach=False, fused=False)
    history, best_key, best_state, best_rows = [], None, None, None
    for epoch in range(1, cfg['training']['epochs'] + 1):
        validate_frozen(frozen_state)
        fit.set_epoch(epoch)
        train_loss, _ = run_epoch(model, fit_loader, cfg, device, optimizer)
        internal_loss, rows = run_epoch(model, internal_loader, cfg, device)
        summary = summarize_cases(rows, cfg, domain=domain)
        gates = gate_summary(summary, cfg)
        key = selection_key(summary, gates, internal_loss['total'], epoch)
        if best_key is None or key < best_key:
            best_key = key
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_rows = copy.deepcopy(rows)
        row = {'epoch': epoch, 'train_loss': train_loss, 'internal_loss': internal_loss,
            'conditions': summary, 'gates': gates, 'selection_key': list(key), 'best_epoch': best_key[-1]}
        _require(_plain(row), 'nonfinite epoch report')
        history.append(row)
        print(json.dumps({'dataset': domain, **row}, ensure_ascii=False, allow_nan=False), flush=True)
    validate_frozen(frozen_state)
    return best_state, {'dataset': domain, 'sample_rate_hz': RATES[domain],
        'fit_subjects': [1, 2, 3, 4], 'internal_validation_subjects': [5, 6],
        'fit_case_count': expected_fit, 'internal_case_count': expected_internal,
        'best_epoch': best_key[-1], 'best_selection_key': list(best_key), 'internal_gates_passed': best_key[0] == 0,
        'selection_status': 'internal_gates_passed' if best_key[0] == 0 else 'gates_failed',
        'history': history, 'best_case_rows': best_rows}


def source_binding(records):
    return [{'dataset': r.dataset, 'subject_id': r.subject_id, 'sample_rate_hz': r.sample_rate_hz,
             'npz_sha256': r.npz_sha256, 'sidecar_sha256': r.sidecar_sha256}
            for r in sorted(records, key=lambda r: (DOMAINS.index(r.dataset), r.subject_id))]


def _expected_source_binding(files):
    return [{'dataset': d, 'subject_id': s, 'sample_rate_hz': RATES[d],
        'npz_sha256': files[f'{REFERENCE_DIR}/{d}/subject_{s:03d}.npz'],
        'sidecar_sha256': files[f'{REFERENCE_DIR}/{d}/subject_{s:03d}.npz.manifest.json']}
        for d in DOMAINS for s in range(1, 7)]


def make_training_report(cfg, state, records, domain_reports):
    _require_training_preflight(state)
    validate_frozen(state)
    _require(_exact(cfg, state.config), 'report config differs from frozen state')
    validate_records(records, cfg, files=state.files)
    report = {'schema_version': 1, 'status': 'formal', 'model_version': MODEL_VERSION,
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'device': 'cuda:0',
        'config_sha256': CONFIG_SHA256, 'preregistration_sha256': PREREG_SHA256,
        'training_lock_sha256': state.lock_sha256,
        'reference_manifest_sha256': REFERENCE_MANIFEST_SHA256, 'partition_lock_sha256': PARTITION_LOCK_SHA256,
        'source_binding': source_binding(records), 'domain_reports': domain_reports,
        'sample_rates_hz': dict(RATES), 'fit_subjects': [1, 2, 3, 4],
        'internal_validation_subjects': [5, 6], 'target_definition': cfg['corruption']['target_definition'],
        'amp': False, 'selection_case_rows_scope': 'selected_epoch_complete_C_rows_per_native_domain',
        'development_history': state.preregistration['development_history'], **PROVENANCE}
    validate_training_report(report, cfg, state.lock_sha256)
    return report


LOSS_FIELDS = {'total', 'corrupted', 'identity', 'hf_low_branch_change', 'change',
               'corrupted_weight_sum', 'case_count', 'identity_count', 'corrupted_count', 'hf_count'}
DOMAIN_REPORT_FIELDS = {'dataset', 'sample_rate_hz', 'fit_subjects', 'internal_validation_subjects',
    'fit_case_count', 'internal_case_count', 'best_epoch', 'best_selection_key', 'internal_gates_passed',
    'selection_status', 'history', 'best_case_rows'}
REPORT_FIXED = {'schema_version': 1, 'status': 'formal', 'model_version': MODEL_VERSION,
    'device': 'cuda:0', 'config_sha256': CONFIG_SHA256, 'preregistration_sha256': PREREG_SHA256,
    'reference_manifest_sha256': REFERENCE_MANIFEST_SHA256, 'partition_lock_sha256': PARTITION_LOCK_SHA256,
    'sample_rates_hz': dict(RATES), 'fit_subjects': [1, 2, 3, 4], 'internal_validation_subjects': [5, 6],
    'amp': False, 'selection_case_rows_scope': 'selected_epoch_complete_C_rows_per_native_domain', **PROVENANCE}
REPORT_FIELDS = set(REPORT_FIXED) | {'created_at_utc', 'training_lock_sha256', 'source_binding',
    'domain_reports', 'target_definition', 'development_history'}
STATE_SCHEMA = {'version': 1, 'domains': list(DOMAINS), 'tensor_dtype': 'float32',
    'tensor_device': 'cpu', 'state_dict_keys_shapes': 'exact_build_model_native_rate',
    'optimizer_state_saved': False, 'resume_allowed': False}
CHECKPOINT_FIELDS = {'schema_version', 'status', 'model_version', 'model_type', 'config',
    'config_sha256', 'preregistration_sha256', 'training_lock_sha256', 'reference_manifest_sha256',
    'partition_lock_sha256', 'state_schema', 'training_report', 'models'} | set(PROVENANCE)


def _validate_aggregated_loss(loss, cases, cfg):
    _require(type(loss) is dict and set(loss) == LOSS_FIELDS and _plain(loss), 'invalid aggregated loss schema')
    _required(loss, {'case_count': cases, 'identity_count': cases // 5, 'corrupted_count': 4 * cases // 5,
        'hf_count': cases // 5, 'corrupted_weight_sum': float(cases // 5 * 7)}, 'balanced epoch loss counts/weights')
    for k in ('total', 'corrupted', 'identity', 'hf_low_branch_change', 'change'):
        _require(type(loss[k]) is float and loss[k] >= 0, 'invalid aggregate loss value')
    l = cfg['loss']
    expected = (loss['corrupted'] + l['identity_normalized_mse_weight'] * loss['identity']
        + l['high_frequency_low_branch_change_weight'] * loss['hf_low_branch_change']
        + l['normalized_change_penalty_weight'] * loss['change'])
    _require(math.isclose(loss['total'], expected, rel_tol=1e-12, abs_tol=1e-12), 'invalid global loss aggregation')


def validate_training_report(report, cfg, training_lock_sha256):
    validate_config(cfg)
    protocol = _protocol()
    _require(_sha(training_lock_sha256), 'invalid training lock SHA')
    _require(type(report) is dict and set(report) == REPORT_FIELDS and _plain(report), 'invalid training report schema')
    _required(report, {**REPORT_FIXED, 'training_lock_sha256': training_lock_sha256,
        'target_definition': cfg['corruption']['target_definition'], 'development_history': protocol['development_history'],
        'source_binding': _expected_source_binding(protocol['bound_files'])}, 'report metadata/provenance')
    try:
        _require(type(report['created_at_utc']) is str, 'invalid report timestamp type')
        dt = datetime.fromisoformat(report['created_at_utc'])
        _require(dt.utcoffset() is not None and dt.utcoffset().total_seconds() == 0, 'UTC timestamp required')
    except (TypeError, ValueError) as exc:
        raise V8Error('invalid report UTC timestamp') from exc
    domains = report['domain_reports']
    _require(type(domains) is dict and set(domains) == set(DOMAINS), 'exactly two independent domain reports required')
    for d, domain in domains.items():
        _require(type(domain) is dict and set(domain) == DOMAIN_REPORT_FIELDS, 'invalid domain report schema')
        fit_cases, c_cases = SHAPES[d][0] * 20, SHAPES[d][0] * 10
        _required(domain, {'dataset': d, 'sample_rate_hz': RATES[d], 'fit_subjects': [1, 2, 3, 4],
            'internal_validation_subjects': [5, 6], 'fit_case_count': fit_cases, 'internal_case_count': c_cases}, 'domain contract')
        history = domain['history']
        _require(type(history) is list and len(history) == 18, 'exactly18 epochs; no extensions/epoch-zero candidate')
        keys = []
        for epoch, row in enumerate(history, 1):
            _require(type(row) is dict and set(row) == {'epoch', 'train_loss', 'internal_loss', 'conditions', 'gates', 'selection_key', 'best_epoch'},
                     'invalid history row schema')
            _require(_exact(row['epoch'], epoch), 'history epochs must be1..18 in order')
            _validate_aggregated_loss(row['train_loss'], fit_cases, cfg)
            _validate_aggregated_loss(row['internal_loss'], c_cases, cfg)
            summary = row['conditions']
            gates = gate_summary(summary, cfg)
            _require(set(summary) == set(CONDITIONS) and all(_exact(s['case_count'], c_cases // 5) for s in summary.values()),
                     'incomplete per-condition summary')
            _require(_exact(row['gates'], gates), 'recorded gates mismatch')
            key = selection_key(summary, gates, row['internal_loss']['total'], epoch)
            keys.append(key)
            _require(_exact(row['selection_key'], list(key)) and _exact(row['best_epoch'], min(keys)[-1]), 'invalid selection history')
        best = min(keys)
        eligible = best[0] == 0
        _required(domain, {'best_epoch': best[-1], 'best_selection_key': list(best), 'internal_gates_passed': eligible,
            'selection_status': 'internal_gates_passed' if eligible else 'gates_failed'}, 'selected epoch/gates_failed preservation')
        selected_summary = summarize_cases(domain['best_case_rows'], cfg, domain=d)
        saved_summary = history[best[-1] - 1]['conditions']
        _require(_exact(selected_summary, saved_summary), 'selected C rows/summary mismatch')


def checkpoint_metadata(cfg, training_report, training_lock_sha256):
    validate_training_report(training_report, cfg, training_lock_sha256)
    return {'schema_version': 1, 'status': 'formal', 'model_version': MODEL_VERSION,
        'model_type': cfg['model_type'], 'config': copy.deepcopy(cfg), 'config_sha256': CONFIG_SHA256,
        'preregistration_sha256': PREREG_SHA256, 'training_lock_sha256': training_lock_sha256,
        'reference_manifest_sha256': REFERENCE_MANIFEST_SHA256, 'partition_lock_sha256': PARTITION_LOCK_SHA256,
        'state_schema': copy.deepcopy(STATE_SCHEMA), 'training_report': copy.deepcopy(training_report), **PROVENANCE}


def validate_checkpoint(ck, cfg, training_lock_sha256):
    """Strict metadata and exact CPU float32 finite state; no CUDA/path reads."""
    _require(type(ck) is dict and set(ck) == CHECKPOINT_FIELDS, 'invalid v8 checkpoint keys')
    expected = checkpoint_metadata(cfg, ck['training_report'], training_lock_sha256)
    for k, v in expected.items():
        _require(_exact(ck[k], v), 'checkpoint metadata mismatch: ' + k)
    _require(type(ck['models']) is dict and set(ck['models']) == set(DOMAINS), 'exactly two independent model states required')
    torch, models = _torch(), {}
    with torch.random.fork_rng(devices=[]):  # building validators MUST NOT disturb training RNG
        for d in DOMAINS:
            model = _model_api().build_model(cfg, RATES[d]).cpu()
            expected_state = model.state_dict()
            state = ck['models'][d]
            _require(isinstance(state, dict) and set(state) == set(expected_state), 'invalid state keys: ' + d)
            for name, tensor in state.items():
                _require(isinstance(tensor, torch.Tensor) and tensor.layout == torch.strided
                    and tensor.device.type == 'cpu' and tensor.dtype == torch.float32
                    and tensor.shape == expected_state[name].shape and not tensor.requires_grad
                    and bool(torch.isfinite(tensor).all()), 'invalid CPU state tensor: ' + d + '/' + name)
            model.load_state_dict(state, strict=True)
            model.eval()
            models[d] = model
    return models


def load_checkpoint_cpu(payload, cfg, training_lock_sha256):
    """Buffers ONLY; caller owns diagnostic freeze/path. Always weights_only/CPU."""
    _require(type(payload) is bytes or type(payload) is io.BytesIO, 'checkpoint loader accepts only bytes/BytesIO, never paths')
    stream = io.BytesIO(payload) if type(payload) is bytes else io.BytesIO(payload.getvalue())
    ck = _torch().load(stream, map_location='cpu', weights_only=True)
    return ck, validate_checkpoint(ck, cfg, training_lock_sha256)


def _output_path(root, rel):
    path = safe_path(root, rel)
    relative = path.relative_to(safe_path(root, '.')).as_posix()
    report = (path.parent == safe_path(root, 'analysis_outputs') and path.name.startswith('ssvep_v8')
              and path.suffix in ('.json', '.md') and 'lock' not in path.name.casefold()
              and 'prereg' not in path.name.casefold())
    _require(relative == CHECKPOINT_RELATIVE or report, 'only v8 checkpoint and v8 analysis reports may be published')
    return path


def assert_new_outputs(root, paths=None):
    paths = (CHECKPOINT_RELATIVE, TRAIN_REPORT_RELATIVE) if paths is None else paths
    for rel in paths:
        _require(not os.path.lexists(_output_path(root, rel)), 'no-clobber output already exists: ' + str(rel))


def publish_bundle(root, payloads):
    """Own exclusive hard-link publisher; NEVER replace existing outputs.

    V8 analysis JSON/Markdown reports and unique v8 checkpoint ONLY. Locks and
    preregistration are never publishable. Failed second publication rolls back
    ONLY a link to our staged inode. A crash may leave a partial bundle, which
    the next run rejects, never repairs or overwrites.
    """
    _require(type(payloads) is dict and bool(payloads) and all(type(k) is str and type(v) is bytes and v
             for k, v in payloads.items()), 'nonempty relative-path byte bundle required')
    destinations = [_output_path(root, rel) for rel in payloads]
    _require(len({str(p).casefold() for p in destinations}) == len(destinations), 'duplicate/aliased output paths')
    assert_new_outputs(root, payloads)
    staged, published = {}, []
    try:
        for rel, data in payloads.items():
            dest = _output_path(root, rel)
            _require(dest.parent.is_dir(), 'output parent directory missing')
            fd, temp = tempfile.mkstemp(prefix='.ssvep-v8-stage-', dir=dest.parent)
            temp = Path(temp)
            staged[rel] = temp
            with os.fdopen(fd, 'wb') as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        for rel, temp in staged.items():
            dest = _output_path(root, rel)
            _require(not os.path.lexists(dest), 'output collision during publication')
            try:
                os.link(temp, dest)  # atomic exclusive; EEXIST even for raced dangling symlink
            except OSError as exc:
                raise V8Error('exclusive output publication failed: ' + rel) from exc
            published.append((dest, temp))
    except BaseException:
        for dest, temp in reversed(published):
            try:
                if os.path.samestat(dest.lstat(), temp.lstat()):
                    dest.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for temp in staged.values():
            try:
                temp.unlink()
            except FileNotFoundError:
                pass


def publish_training_bundle(state, models, report):
    _require_training_preflight(state)
    validate_frozen(state)
    assert_new_outputs(state.root)
    ck = checkpoint_metadata(state.config, report, state.lock_sha256)
    ck['models'] = models
    validate_checkpoint(ck, state.config, state.lock_sha256)
    stream = io.BytesIO()
    _torch().save(ck, stream)
    load_checkpoint_cpu(stream, state.config, state.lock_sha256)
    report_bytes = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    validate_frozen(state)
    publish_bundle(state.root, {CHECKPOINT_RELATIVE: stream.getvalue(), TRAIN_REPORT_RELATIVE: report_bytes})


def run_training(state):
    """Authorized full-run API; never called by check-only or imports."""
    _require_training_preflight(state)
    assert_new_outputs(state.root)
    validate_frozen(state)
    device = configure_training(state.config)
    validate_frozen(state)  # immediately BEFORE first real binary verification/load
    records = load_training_records(state)
    models, reports = {}, {}
    for domain in DOMAINS:
        models[domain], reports[domain] = train_domain(records, domain, state.config, device, frozen_state=state)
    report = make_training_report(state.config, state, records, reports)
    publish_training_bundle(state, models, report)
    return report
