"""Numerical/provenance tests for the standalone artificial-waveform audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts import audit_ssvep_v12_long_waveform as audit


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


class EvaluationFixture:
    """Tiny bound files, never a trainable checkpoint or personal recording."""
    def __init__(self, root, *, measured=False):
        self.root = Path(root)
        self.evaluation = self.root / "analysis_outputs/evaluation"
        self.output = self.root / "analysis_outputs/new_audit"
        self.synthetic = {"case_count": 20, "frequencies": list(audit.FREQUENCIES),
                          "conditions": list(audit.CONDITIONS),
                          "known_clean_is_artificial_only": True, "cases": []}
        for index, (frequency, condition) in enumerate(
                (f, c) for f in audit.FREQUENCIES for c in audit.CONDITIONS):
            # Deliberately not the evaluator's seed formula: exact saved seeds must win.
            row = {"frequency_hz": frequency, "condition": condition, "seed": 501 + index * 7,
                   "errors": {name: {"interior": {"samples": 3072, "clean_centered_mse": 0.0}}
                              for name in audit.TRACKS}}
            if measured:
                from eeg_tools.ssvep_v12_runtime import condition_line
                from eeg_tools.workstation.v9_offline import _traditional
                from scripts.evaluate_ssvep_denoiser_v10 import synthetic_case
                clean, raw = synthetic_case(frequency, condition, row["seed"])
                notch = condition_line(raw)
                tracks = {"raw": raw, "notch": notch, "traditional": _traditional(raw), "v12": notch}
                for name, signal in tracks.items():
                    row["errors"][name]["interior"] = audit.waveform_errors(clean[512:-512], signal[512:-512])
            self.synthetic["cases"].append(row)
        self.public = {"reference_is_clean_truth": False, "conditions": {
            "identity": {"window_count": 2, "input_mse": 0, "output_mse": 0.001}}}
        config = {"runtime": {"enabled": False, "production_validated": False}}
        dump(self.root / audit.CONFIG, config)
        for name in (audit.MODEL, audit.LOCK, audit.TRAINING_REPORT, audit.RUNTIME,
                     "scripts/evaluate_ssvep_denoiser_v10.py", "eeg_tools/workstation/v9_offline.py",
                     "eeg_tools/ssvep_v10_data.py", "eeg_tools/ssvep_denoise_v10.py"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"frozen fixture bytes: " + name.encode())
        self.manifest = {"status": "formal", "model_version": "learned_denoise_v12_personal",
                         "selected_epoch": 21, "runtime_enabled": False, "production_validated": False,
                         "config_sha256": audit._canonical_sha(config),
                         "checkpoint_sha256": audit.sha256(self.root / audit.MODEL),
                         "training_lock_sha256": audit.sha256(self.root / audit.LOCK),
                         "training_report_sha256": audit.sha256(self.root / audit.TRAINING_REPORT),
                         "files": {name: audit.sha256(self.root / name) for name in
                                   (audit.MODEL, audit.LOCK, audit.TRAINING_REPORT, audit.RUNTIME, audit.CONFIG)}}
        dump(self.root / audit.MANIFEST, self.manifest)
        self.summary = {"status": "formal", "completion_status": "complete",
                        "runtime_enabled": False, "production_validated": False,
                        "independent_personal_validation": False,
                        "public": {"report": "public.json"},
                        "synthetic": {"report": "synthetic.json", "case_count": 20},
                        "research_gate": {"passed": True},
                        "personal": [{"source_session": str(self.root / "DO_NOT_OPEN_PERSONAL_DATA")}],
                        "config_sha256": self.manifest["config_sha256"],
                        "model": {key: self.manifest[key] for key in
                                  ("checkpoint_sha256", "selected_epoch", "model_version")}}
        self.save_evaluation()

    def save_evaluation(self):
        dump(self.evaluation / "public.json", self.public)
        dump(self.evaluation / "synthetic.json", self.synthetic)
        self.summary["artifacts"] = {name: {"sha256": audit.sha256(self.evaluation / name)}
                                     for name in ("public.json", "synthetic.json")}
        dump(self.evaluation / "summary.json", self.summary)


class JointLeastSquaresTests(unittest.TestCase):
    def test_known_amplitude_phase_intercept_and_trend_recovered_jointly(self):
        start, samples = 512, 3072
        time = (np.arange(samples) + start) / 250
        frequencies = (12, 24, 7)
        amplitudes = np.array([[1.2, 0.7, 2.1], [0.31, 0.24, 0.8], [0.22, 0.5, 0.15]])
        phases = np.array([[np.pi - 0.02, -np.pi + 0.03, 0.8],
                           [-1.1, 1.6, 0.4], [2.8, -2.9, -0.9]])
        slopes = np.array([0.11, -0.09, 0.04])
        intercepts = np.array([100.0, -7.0, 30.0])
        signal = intercepts + (time - time.mean())[:, None] * slopes
        for index, frequency in enumerate(frequencies):
            signal += amplitudes[index] * np.sin(2 * np.pi * frequency * time[:, None] + phases[index])
        result = audit.joint_sine_fit(signal, frequencies, start_sample=start)
        self.assertEqual(result["design_rank"], 8)
        np.testing.assert_allclose(result["intercept_at_time_origin"], intercepts, atol=1e-10)
        np.testing.assert_allclose(result["linear_trend_units_per_second"], slopes, atol=1e-11)
        self.assertLess(max(result["residual_mse_per_channel"]), 1e-22)
        for index, component in enumerate(result["components"]):
            np.testing.assert_allclose(component["amplitude"], amplitudes[index], atol=1e-11)
            np.testing.assert_allclose(audit.wrap_phase(np.array(component["phase_radians"]) - phases[index]),
                                       0, atol=1e-11)

    def test_fit_comparison_wraps_phase_cut_and_recovers_gain(self):
        time = np.arange(4096) / 250
        frequencies = (10, 20, 7)
        truth = np.zeros((len(time), 2))
        measured = truth.copy()
        phases = (np.pi - 0.02, -np.pi + 0.01)
        for index, frequency in enumerate(frequencies):
            for channel, phase in enumerate(phases):
                truth[:, channel] += (0.7 ** index) * np.sin(2 * np.pi * frequency * time + phase)
                measured[:, channel] += (0.7 ** index) * 1.6 * np.sin(
                    2 * np.pi * frequency * time + phase + 0.05)
        comparisons = audit.compare_fits(audit.joint_sine_fit(truth, frequencies),
                                        audit.joint_sine_fit(measured, frequencies))
        for component in comparisons:
            np.testing.assert_allclose(component["amplitude_ratio_per_channel"], 1.6, atol=1e-11)
            np.testing.assert_allclose(component["wrapped_phase_error_radians_per_channel"], 0.05, atol=1e-11)
        np.testing.assert_allclose(audit.wrap_phase([2 * np.pi - 0.1, -2 * np.pi + 0.1, np.pi]),
                                   [-0.1, 0.1, -np.pi], atol=1e-14)

    def test_centering_does_not_hide_phase_error(self):
        time = np.arange(1000) / 250
        clean = (np.sin(2 * np.pi * 10 * time) + 100)[:, None]
        shifted = (np.sin(2 * np.pi * 10 * time + np.pi / 2) - 30)[:, None]
        errors = audit.waveform_errors(clean, shifted)
        self.assertAlmostEqual(errors["clean_centered_mse"], 1.0, places=12)
        self.assertAlmostEqual(errors["clean_centered_normalized_mse"], 2.0, places=12)
        self.assertAlmostEqual(errors["mean_offset_mse"], 130 ** 2, places=9)
        result = audit.compare_fits(audit.joint_sine_fit(clean, (10, 20, 7)),
                                   audit.joint_sine_fit(shifted, (10, 20, 7)))
        self.assertAlmostEqual(result[0]["amplitude_ratio_per_channel"][0], 1.0, places=12)
        self.assertAlmostEqual(result[0]["wrapped_phase_error_radians_per_channel"][0], np.pi / 2, places=12)

    def test_dc_offset_is_reported_separately_and_zero_amplitude_phase_is_null(self):
        time = np.arange(1000) / 250
        clean = np.sin(2 * np.pi * 10 * time)[:, None]
        errors = audit.waveform_errors(clean, clean + 17)
        self.assertLess(errors["clean_centered_mse"], 1e-25)
        self.assertAlmostEqual(errors["uncentered_mse"], 289, places=10)
        absent = audit.joint_sine_fit(np.zeros_like(clean), (10, 20, 7))
        self.assertEqual(absent["components"][0]["phase_radians"], [None])
        compared = audit.compare_fits(audit.joint_sine_fit(clean, (10, 20, 7)), absent)
        self.assertIsNone(compared[0]["wrapped_phase_error_radians_per_channel"][0])

    def test_invalid_ls_and_error_inputs_rejected(self):
        for frequencies in ((10, 10, 7), (0, 20, 7), (125, 20, 7)):
            with self.subTest(frequencies=frequencies), self.assertRaises(ValueError):
                audit.joint_sine_fit(np.ones((100, 2)), frequencies)
        with self.assertRaisesRegex(ValueError, "flat clean"):
            audit.waveform_errors(np.ones((100, 2)), np.ones((100, 2)))


class ProvenanceAndPathTests(unittest.TestCase):
    def test_tampered_public_and_synthetic_sha_rejected_before_output(self):
        for filename in ("public.json", "synthetic.json"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temporary:
                fixture = EvaluationFixture(temporary)
                path = fixture.evaluation / filename
                path.write_bytes(path.read_bytes() + b" ")  # Valid JSON, different bytes.
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    audit.preflight(fixture.evaluation, fixture.output, "cuda:0", root=fixture.root)
                self.assertFalse(fixture.output.exists())

    def test_existing_output_root_overlap_escape_and_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = EvaluationFixture(temporary)
            fixture.output.mkdir()
            for destination in (fixture.output, fixture.root / "analysis_outputs", fixture.root / "outside",
                                fixture.evaluation / "nested", fixture.root / "analysis_outputs/../escape"):
                with self.subTest(destination=destination), self.assertRaises(ValueError):
                    audit.validate_paths(fixture.evaluation, destination, root=fixture.root)

    def test_frozen_checkpoint_epoch_and_source_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = EvaluationFixture(temporary)
            fixture.summary["model"]["selected_epoch"] = 20
            fixture.save_evaluation()
            with self.assertRaisesRegex(ValueError, "epoch"):
                audit.preflight(fixture.evaluation, fixture.output, root=fixture.root)
            fixture.summary["model"]["selected_epoch"] = 21
            fixture.save_evaluation()
            (fixture.root / audit.MODEL).write_bytes(b"tampered checkpoint")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                audit.preflight(fixture.evaluation, fixture.output, root=fixture.root)

    def test_exact_seeds_and_complete_grid_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = EvaluationFixture(temporary)
            fixture.synthetic["cases"].reverse()
            fixture.save_evaluation()
            state = audit.preflight(fixture.evaluation, fixture.output, root=fixture.root)
            self.assertEqual([row["seed"] for row in state["rows"]], [501 + 7 * i for i in range(20)])
            fixture.synthetic["cases"][0] = fixture.synthetic["cases"][1]
            fixture.save_evaluation()
            with self.assertRaisesRegex(ValueError, "duplicate synthetic case"):
                audit.preflight(fixture.evaluation, fixture.output, root=fixture.root)

    def test_import_and_full_readonly_preflight_never_import_torch_or_initialize_cuda(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = EvaluationFixture(temporary)
            code = """
import builtins, sys
original_import = builtins.__import__
def forbid_scientific(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy', 'scipy', 'torch', 'matplotlib'}:
        raise AssertionError('scientific/CUDA import during preflight: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = forbid_scientific
from scripts import audit_ssvep_v12_long_waveform as a
from pathlib import Path
root = Path(sys.argv[1])
a.preflight(root/'analysis_outputs/evaluation', root/'analysis_outputs/new_audit', 'cuda:0', root=root)
assert not {'numpy', 'scipy', 'torch', 'matplotlib'} & sys.modules.keys()
assert not (root/'analysis_outputs/new_audit').exists()
try:
    a.validate_paths(root/'analysis_outputs/evaluation', root/'analysis_outputs/evaluation', root=root)
except ValueError:
    pass
else:
    raise AssertionError('existing path accepted')
"""
            result = subprocess.run([sys.executable, "-B", "-c", code, str(fixture.root)],
                                    cwd=audit.ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_exclusive_json_output_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            audit.write_json(path, {"original": True})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                audit.write_json(path, {"overwritten": True})
            self.assertEqual(path.read_bytes(), before)


class AuditIntegrationTests(unittest.TestCase):
    def test_twenty_seeded_cases_eighty_comparisons_flags_and_marker_written_last(self):
        from eeg_tools import ssvep_v12_runtime as runtime
        from scripts import evaluate_ssvep_denoiser_v10 as generator
        with tempfile.TemporaryDirectory() as temporary:
            fixture = EvaluationFixture(temporary, measured=True)
            before = {p: audit.sha256(p) for p in fixture.evaluation.glob("*.json")}
            metadata = {**fixture.summary["model"], "device": "cpu"}
            def infer(raw, model):
                notch = runtime.condition_line(raw)
                return notch.copy(), notch, {"production_validated": False}
            writes = []
            original_write = audit.write_json
            def record_write(path, value):
                writes.append(Path(path).name)
                original_write(path, value)
            with mock.patch.object(runtime, "load_model", return_value=(object(), metadata)) as load, \
                 mock.patch.object(runtime, "process_recording", side_effect=infer) as process, \
                 mock.patch.object(generator, "synthetic_case", wraps=generator.synthetic_case) as regenerate, \
                 mock.patch.object(audit, "write_json", side_effect=record_write):
                report = audit.audit(fixture.evaluation, output=fixture.output, root=fixture.root)
            load.assert_called_once_with(fixture.root, "cpu")
            self.assertEqual(process.call_count, 20)
            self.assertEqual([call.args[2] for call in regenerate.call_args_list], [501 + 7 * i for i in range(20)])
            self.assertEqual(report["reproduction"]["comparison_count"], 80)
            self.assertTrue(report["reproduction"]["verified"])
            self.assertEqual(writes[-1], "report.json")
            self.assertTrue(report["existing_research_gate_passed"])
            for flag in ("production_validated", "runtime_enabled", "independent_personal_validation",
                         "clean_ground_truth_available", "real_eeg_clean_ground_truth_available",
                         "deployment_gate_evaluated", "existing_research_gate_is_deployment_acceptance",
                         "training_performed", "checkpoint_selection_performed", "personal_raw_data_accessed"):
                self.assertIs(report[flag], False, flag)
            for name, artifact in report["artifacts"].items():
                self.assertEqual(audit.sha256(fixture.output / name), artifact["sha256"])
            self.assertEqual({p: audit.sha256(p) for p in before}, before)
            interior = report["nmse_by_region_and_condition"]["interior"]
            for condition in audit.CONDITIONS:
                entry = interior["conditions"][condition]
                self.assertEqual(entry["case_count"], 4)
                stages = entry["stage_contributions"]
                self.assertEqual(stages["neural_notch_to_v12"]["centered_nmse_reduction"], 0)
                self.assertAlmostEqual(stages["fixed_notch_raw_to_notch"]["centered_nmse_reduction"]
                                       + stages["neural_notch_to_v12"]["centered_nmse_reduction"],
                                       stages["combined_raw_to_v12"]["centered_nmse_reduction"])
            detail = json.loads((fixture.output / "case_measurements.json").read_text())
            self.assertEqual(len(detail["cases"]), 20)
            self.assertEqual(len(detail["cases"][0]["regions"]["interior"]["tracks"]["v12"]
                                 ["comparison_to_artificial_clean"]), 3)
            with mock.patch.object(runtime, "load_model", side_effect=AssertionError("must not load")):
                with self.assertRaisesRegex(ValueError, "already exists"):
                    audit.audit(fixture.evaluation, output=fixture.output, device="cuda:0", root=fixture.root)

    def test_saved_mse_mismatch_fails_reproduction_even_if_old_gate_passed(self):
        from scripts.evaluate_ssvep_denoiser_v10 import synthetic_case
        clean, raw = synthetic_case(10, "identity", 51)
        tracks = {name: raw.copy() for name in audit.TRACKS}
        row = {"frequency_hz": 10, "condition": "identity", "seed": 51,
               "errors": {name: {"interior": {"clean_centered_mse": 0}} for name in audit.TRACKS}}
        row["errors"]["v12"]["interior"]["clean_centered_mse"] = 0.1
        measured = audit.measure_case(row, clean, tracks)
        checks = {check["track"]: check for check in measured["stored_mse_checks"]}
        self.assertFalse(checks["v12"]["matches"])
        self.assertTrue(all(checks[name]["matches"] for name in ("raw", "notch", "traditional")))

    def test_inference_failure_leaves_no_completion_marker(self):
        from eeg_tools import ssvep_v12_runtime as runtime
        with tempfile.TemporaryDirectory() as temporary:
            fixture = EvaluationFixture(temporary)
            with mock.patch.object(runtime, "load_model", return_value=(object(), fixture.summary["model"])), \
                 mock.patch.object(runtime, "process_recording", side_effect=RuntimeError("interrupted")):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    audit.audit(fixture.evaluation, output=fixture.output, root=fixture.root)
            self.assertTrue(fixture.output.is_dir())
            self.assertFalse((fixture.output / "report.json").exists())


if __name__ == "__main__":
    unittest.main()
