"""Synthetic fixture tests for the explicit v3 offline diagnostics launcher.

These tests exercise only parser/preflight/fixture inference behavior.  They do
not train, inspect the v2 test split, load S4/S5, open hardware, or start a GUI.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from eeg_tools import learned_denoise_v2 as v2
from eeg_tools import learned_denoise_v3 as v3

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "start_denoiser_v3_diagnostics.ps1"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
PWSH = shutil.which("pwsh")


def json_reports(stdout: str) -> list[dict]:
    """Extract standalone JSON objects from validator/preflight/CLI output."""
    decoder = json.JSONDecoder()
    reports: list[dict] = []
    cursor = 0
    while cursor < len(stdout):
        start = stdout.find("{", cursor)
        if start < 0:
            break
        try:
            value, consumed = decoder.raw_decode(stdout[start:])
        except ValueError:
            cursor = start + 1
            continue
        if isinstance(value, dict):
            reports.append(value)
        cursor = start + consumed
    return reports


def snapshot(path: Path) -> dict[str, str]:
    paths = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    return {str(item.resolve()): hashlib.sha256(item.read_bytes()).hexdigest() for item in paths}


@unittest.skipUnless(PWSH and PYTHON.is_file() and v2.torch is not None,
                     "pwsh, project .venv and optional PyTorch are required")
class DenoiserV3LauncherTests(unittest.TestCase):
    """Use valid temporary weights, while keeping all data synthetic."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="v3 launcher 中文 [fixture] ")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.config = deepcopy(v3.DEFAULT_CONFIG)
        self.config_path = self.directory / "formal v3 config.json"
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        self.v2_path = self.directory / "frozen v2.pt"
        self.v3_path = self.directory / "domain gate v3.pt"
        self._write_fixture_checkpoints()

        self.input_npz = self.directory / "synthetic input.npz"
        samples = np.arange(500, dtype=np.float64)
        raw = (2.0 + 0.5 * np.sin(2.0 * math.pi * 10.0 * samples / 250.0))[None, :]
        np.savez(self.input_npz, signal=raw, sample_rate_hz=np.asarray(250.0),
                 timestamp_s=samples / 250.0)
        self.input_before = snapshot(self.input_npz)
        self.output = self.directory / "new diagnostic output"

    def _write_fixture_checkpoints(self) -> None:
        """Write strict weights-only-compatible v2 and v3 fixture checkpoints."""
        torch = v2.torch
        torch.manual_seed(20261005)
        denoiser = v2.build_model(v2.DEFAULT_CONFIG)
        v2_checkpoint = {
            "checkpoint_schema_version": 1,
            "model_version": v2.MODEL_VERSION,
            "config": deepcopy(v2.DEFAULT_CONFIG),
            "architecture": deepcopy(v2.DEFAULT_CONFIG["model"]),
            "model_state": denoiser.state_dict(),
            "calibrated_threshold": 0.5,
            "validation_threshold": 0.5,
            "threshold_calibration": {
                "scope": "validation_only",
                "comparison": ">",
                "production_calibrated": False,
                "method": "clean_score_order_statistic",
                "score_reduction": "per_channel_window_mean",
                "empirical_only": True,
                "clean_window_count": 1,
                "observed_clean_false_replacement_fraction": 0.0,
                "max_clean_false_replacement_fraction": 0.01,
            },
        }
        torch.save(v2_checkpoint, self.v2_path)
        self.assertEqual(
            v2.validate_checkpoint_metadata(v2_checkpoint, v2.DEFAULT_CONFIG)["model_version"],
            v2.MODEL_VERSION,
        )

        gate = v3.build_gate(self.config)
        v3_checkpoint = {
            "checkpoint_schema_version": 1,
            "model_version": v3.MODEL_VERSION,
            "config": deepcopy(self.config),
            "gate_state": gate.state_dict(),
            "support_threshold": 0.5,
            "threshold_calibration": {
                "scope": "validation_only",
                "method": "unsupported_score_order_statistic",
                "comparison": ">",
                "empirical_only": True,
                "production_calibrated": False,
                "unsupported_window_count": 1,
                "observed_unsupported_false_acceptance_fraction": 0.0,
                "max_unsupported_false_acceptance_fraction": 0.01,
            },
            "denoiser_sha256": hashlib.sha256(self.v2_path.read_bytes()).hexdigest(),
            "dataset_sha256": "0" * 64,
            "metadata": {"scope": "synthetic_fixture_train_validation_only"},
        }
        torch.save(v3_checkpoint, self.v3_path)
        self.assertEqual(
            v3.validate_checkpoint_metadata(v3_checkpoint, self.config)["model_version"],
            v3.MODEL_VERSION,
        )

    def run_launcher(self, *options: object, root: Path = ROOT) -> subprocess.CompletedProcess[str]:
        command = [
            PWSH, "-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
            "-File", str(LAUNCHER), "-Root", str(root), "-PythonPath", str(PYTHON),
            "-Checkpoint", str(self.v3_path), "-DenoiserCheckpoint", str(self.v2_path),
            "-Config", str(self.config_path), *map(str, options),
        ]
        return subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=120)

    def preflight_report(self, result: subprocess.CompletedProcess[str]) -> dict:
        reports = [value for value in json_reports(result.stdout) if "diagnostic_ready" in value]
        self.assertEqual(1, len(reports), result.stdout + result.stderr)
        return reports[0]

    def assert_safe(self, report: dict) -> None:
        self.assertTrue(report["diagnostic_only"])
        for field in ("automatic_processing_enabled", "release_gate_passed", "production_validated",
                      "gui_started", "hardware_opened", "acquisition_started", "visual_stimulus_started",
                      "test_evaluation_executed"):
            self.assertFalse(report[field], field)
        self.assertTrue(report["raw_is_immutable"])
        self.assertTrue(report["original_quality_unchanged"])
        self.assertIn("NOT production approval", report["notice"])

    def test_powershell_parser_exposes_only_explicit_safe_entrypoints(self) -> None:
        env = os.environ.copy()
        env["NEUROSTATION_V3_LAUNCHER"] = str(LAUNCHER)
        code = r'''
$tokens = $null; $parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    $env:NEUROSTATION_V3_LAUNCHER, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count -gt 0) { $parseErrors | ForEach-Object { $_.Message }; exit 1 }
$ast.ParamBlock.Parameters.Name.VariablePath.UserPath | ConvertTo-Json -Compress
'''
        result = subprocess.run([PWSH, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", code],
                                cwd=ROOT, env=env, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=30)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        names = set(json.loads(result.stdout))
        self.assertTrue({"CheckOnly", "SessionDir", "InputNPZ", "OutputDir", "Checkpoint",
                         "DenoiserCheckpoint", "Config", "Device", "Threads", "Root",
                         "PythonPath"} <= names)
        self.assertNotIn("StartHardware", names)
        self.assertNotIn("EnableAutomatic", names)

    def test_formal_release_preflight_fails_before_model_loading(self) -> None:
        empty_root = self.directory / "not a project root"
        empty_root.mkdir()
        result = self.run_launcher("-CheckOnly", root=empty_root)
        self.assertNotEqual(0, result.returncode)
        self.assertIn("Formal release configuration preflight failed", result.stdout)
        self.assertNotIn("loaded_v3_and_bound_v2_weights_only", result.stdout)

    def test_check_only_loads_real_fixture_weights_and_never_reads_source(self) -> None:
        result = self.run_launcher("-CheckOnly")
        report = self.preflight_report(result)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assert_safe(report)
        self.assertTrue(report["diagnostic_ready"])
        self.assertEqual("loaded_v3_and_bound_v2_weights_only", report["model_load_status"])
        self.assertEqual("check_only", report["execution"])
        self.assertFalse(self.output.exists())
        self.assertEqual(self.input_before, snapshot(self.input_npz))

    def test_no_source_is_readiness_only_and_output_requires_source(self) -> None:
        result = self.run_launcher("-OutputDir", self.output)
        self.assertNotEqual(0, result.returncode)
        report = self.preflight_report(result)
        self.assertFalse(report["diagnostic_ready"])
        self.assertIn("requires explicit SessionDir or InputNPZ", report["error"])
        self.assertFalse(self.output.exists())

        result = self.run_launcher()
        report = self.preflight_report(result)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assert_safe(report)
        self.assertEqual("readiness_only", report["execution"])
        self.assertFalse(self.output.exists())

    def test_explicit_input_runs_offline_without_mutating_input_or_claiming_repair(self) -> None:
        result = self.run_launcher("-InputNPZ", self.input_npz, "-OutputDir", self.output, "-Threads", "2")
        reports = json_reports(result.stdout)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        readiness = [value for value in reports if "diagnostic_ready" in value][0]
        self.assert_safe(readiness)
        self.assertEqual(self.input_before, snapshot(self.input_npz))
        self.assertTrue((self.output / "denoised_signal.npz").is_file())
        self.assertTrue((self.output / "domain_scores.csv").is_file())
        report = json.loads((self.output / "quality_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual("completed", report["status"])
        self.assertTrue(report["diagnostic_only"])
        self.assertFalse(report["automatic_processing_enabled"])
        self.assertFalse(report["release_gate_passed"])
        self.assertFalse(report["production_validated"])
        self.assertEqual(2, report["threads"])
        with np.load(self.output / "denoised_signal.npz", allow_pickle=False) as archive:
            for field in ("denoised", "model_denoised", "artifact_probability", "confidence",
                          "replacement_mask", "valid_mask", "rejected_mask",
                          "support_probability", "domain_rejected_mask"):
                self.assertIn(field, archive.files)
            self.assertEqual((1, 500), archive["support_probability"].shape)

    def test_existing_output_is_rejected_before_inference(self) -> None:
        self.output.mkdir()
        marker = self.output / "must remain.txt"
        marker.write_text("do not replace", encoding="utf-8")
        result = self.run_launcher("-InputNPZ", self.input_npz, "-OutputDir", self.output)
        self.assertNotEqual(0, result.returncode)
        self.assertIn("OutputDir already exists", result.stdout)
        self.assertEqual("do not replace", marker.read_text(encoding="utf-8"))

    def test_cuda_request_never_falls_back_to_cpu(self) -> None:
        result = self.run_launcher("-CheckOnly", "-Device", "cuda")
        if result.returncode == 0:
            # The explicitly selected .venv is CPU-only in the supported fixture.
            self.fail("CPU fixture unexpectedly accepted explicit CUDA request")
        self.assertIn("CUDA requested but unavailable; no CPU fallback", result.stdout)
        self.assertNotIn("loaded_v3_and_bound_v2_weights_only", result.stdout)


if __name__ == "__main__":
    unittest.main()
