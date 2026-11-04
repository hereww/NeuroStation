"""Fixture-only tests for the explicit, non-production v2 PowerShell entry point."""
from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from eeg_tools import learned_denoise_v2 as v2

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "start_denoiser_v2_diagnostics.ps1"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def json_reports(stdout: str) -> list[dict]:
    """The release validator and diagnostics each print a standalone JSON object."""
    decoder = json.JSONDecoder()
    reports = []
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


def hashes(directory: Path) -> dict[str, str]:
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file()}


@unittest.skipUnless(shutil.which("pwsh") and PYTHON.is_file(), "pwsh and project .venv are required")
class DenoiserV2LauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="v2 launcher 中文 [fixture] ")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.config = v2.load_config(ROOT / "configs" / "learned_denoiser_v2.json")
        self.config_path = self.directory / "formal config [v2].json"
        self.write_config(self.config)
        self.model_path = self.directory / "missing model [v2].pt"
        self.session = self.directory / "source session [raw]"
        self.session.mkdir()
        (self.session / "session.json").write_text(
            json.dumps({"status": "formal", "sampling_rate_hz": 250}), encoding="utf-8")
        (self.session / "quality_metrics.json").write_text(
            '{"official_quality":"unchanged-fixture"}', encoding="utf-8")
        with (self.session / "raw_brainflow.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["sample_index", "package_num", "timestamp_s", "eeg_ch1"])
            for sample in range(500):
                writer.writerow([sample, sample % 256, 1000 + sample / 250,
                                 12 + 3 * math.sin(2 * math.pi * 10 * sample / 250)])
        self.output = self.directory / "new output [diagnostic]"
        self.source_before = hashes(self.session)

    def write_config(self, config):
        self.config_path.write_text(json.dumps(config), encoding="utf-8")

    def run_launcher(self, *options, env=None):
        command = [shutil.which("pwsh"), "-NoLogo", "-NoProfile", "-NonInteractive",
                   "-ExecutionPolicy", "Bypass", "-File", str(LAUNCHER),
                   "-Model", str(self.model_path), "-Config", str(self.config_path), *map(str, options)]
        return subprocess.run(command, cwd=self.directory, env=env, capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=90)

    def readiness(self, result):
        reports = [value for value in json_reports(result.stdout) if "diagnostic_ready" in value]
        self.assertEqual(1, len(reports), result.stdout + result.stderr)
        return reports[0]

    def assert_safe(self, report):
        self.assertIs(True, report["diagnostic_only"])
        for field in ("automatic_processing_enabled", "release_gate_passed", "production_validated",
                      "gui_started", "hardware_opened", "acquisition_started", "visual_stimulus_started"):
            self.assertIs(False, report[field], field)
        self.assertIs(True, report["raw_is_immutable"])
        self.assertIs(True, report["original_quality_unchanged"])
        self.assertIs(False, report["test_evaluation_executed"])
        self.assertEqual(self.source_before, hashes(self.session))
        self.assertIn("NOT production approval", report["notice"])

    def test_powershell_parser_and_safe_entrypoints(self):
        env = os.environ.copy()
        env["NEUROSTATION_V2_LAUNCHER"] = str(LAUNCHER)
        code = r'''
$tokens = $null; $parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    $env:NEUROSTATION_V2_LAUNCHER, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count -gt 0) {
    $parseErrors | ForEach-Object { $_.Message }; exit 1
}
$ast.ParamBlock.Parameters.Name.VariablePath.UserPath | ConvertTo-Json -Compress
'''
        parsed = subprocess.run([shutil.which("pwsh"), "-NoLogo", "-NoProfile", "-NonInteractive",
                                 "-Command", code], env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(0, parsed.returncode, parsed.stdout + parsed.stderr)
        self.assertEqual(["CheckOnly", "SessionDir", "OutputDir", "Model", "Config", "Threads"],
                         json.loads(parsed.stdout))
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("validate_release_configs.py", source)
        self.assertIn(".venv\\Scripts\\python.exe", source)
        self.assertIn("-LiteralPath", source)
        self.assertIn("& $Python @RunArguments", source)
        self.assertIn("load_model(model_path, config, device=", source)
        for forbidden in ("Invoke-Expression", "Start-Process", "Remove-Item", "workstation.py",
                          "train_denoiser_v2.py", "evaluate_denoiser_v2.py", "weights_only=False",
                          "pip install", "--port", "--stimulus", "--acquire"):
            self.assertNotIn(forbidden, source)

    def test_missing_checkpoint_check_only_is_not_ready_and_read_only(self):
        config_hash = hashlib.sha256(self.config_path.read_bytes()).hexdigest()
        result = self.run_launcher("-CheckOnly", "-Threads", "2")
        report = self.readiness(result)
        self.assertNotEqual(0, result.returncode, result.stdout + result.stderr)
        self.assert_safe(report)
        self.assertIs(False, report["model_available"])
        self.assertIs(False, report["diagnostic_ready"])
        self.assertEqual("missing_checkpoint", report["model_load_status"])
        self.assertEqual("formal", report["formal_config_status"])
        self.assertEqual(self.config["runtime"]["validation_status"], report["validation_status"])
        self.assertEqual(self.config["runtime"], report["runtime"])
        self.assertEqual(str(ROOT), report["project_root"])
        self.assertEqual("check_only", report["execution"])
        self.assertIn("checkpoint missing", report["error"])
        self.assertEqual(config_hash, hashlib.sha256(self.config_path.read_bytes()).hexdigest())
        self.assertFalse(self.output.exists())

    def test_no_session_missing_model_prints_manual_usage_and_never_processes(self):
        result = self.run_launcher()
        report = self.readiness(result)
        self.assertNotEqual(0, result.returncode)
        self.assert_safe(report)
        self.assertEqual("readiness_only", report["execution"])
        self.assertIn("-SessionDir '<existing-session>'", result.stdout)
        self.assertFalse(self.output.exists())

    def test_nonformal_or_automatically_enabled_config_fails_closed(self):
        for mutation in ("nonformal", "enabled"):
            config = deepcopy(self.config)
            if mutation == "nonformal":
                config["status"] = "template"
            else:
                config["runtime"]["enabled"] = True
            self.write_config(config)
            with self.subTest(mutation=mutation):
                result = self.run_launcher("-CheckOnly")
                report = self.readiness(result)
                self.assertNotEqual(0, result.returncode)
                self.assert_safe(report)
                self.assertEqual("invalid", report["formal_config_status"])
                self.assertFalse(report["diagnostic_ready"])

    def test_session_requires_new_external_output_even_in_check_only(self):
        existing = self.directory / "existing output"
        existing.mkdir()
        for destination in (None, self.session, self.session / "derived", existing, self.directory):
            options = ["-CheckOnly", "-SessionDir", self.session]
            if destination is not None:
                options.extend(["-OutputDir", destination])
            with self.subTest(destination=destination):
                result = self.run_launcher(*options)
                report = self.readiness(result)
                self.assertNotEqual(0, result.returncode)
                self.assert_safe(report)
                self.assertEqual("invalid", report["output_validation_status"])
                self.assertIn("Session/output error", report["error"])
                self.assertFalse((self.session / "derived").exists())

    def test_missing_torch_rejects_existing_checkpoint_without_loading(self):
        self.model_path.write_bytes(b"never loaded without torch")
        stub = self.directory / "dependency stub"
        stub.mkdir()
        (stub / "torch.py").write_text('raise ImportError("fixture torch unavailable")\n', encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(stub) + os.pathsep + env.get("PYTHONPATH", "")
        result = self.run_launcher("-CheckOnly", env=env)
        report = self.readiness(result)
        self.assertNotEqual(0, result.returncode)
        self.assert_safe(report)
        self.assertTrue(report["model_available"])
        self.assertFalse(report["diagnostic_ready"])
        self.assertEqual("unavailable", report["dependencies"]["torch"])
        self.assertEqual("dependency_unavailable", report["model_load_status"])
        self.assertIn("fixture torch unavailable", report["error"])

    def test_saved_research_summary_is_read_only_and_cannot_enable_processing(self):
        # Run the launcher's same embedded preflight against an isolated fixture root.
        # Never alter or evaluate the real saved test report in the shared workspace.
        source = LAUNCHER.read_text(encoding="utf-8")
        preflight = source.split("$Preflight = @'\n", 1)[1].split("\n'@", 1)[0]
        report_path = self.directory / "analysis_outputs" / "learned_denoiser_v2_test.json"
        report_path.parent.mkdir()
        saved = {"status": "completed", "test_window_count": 7,
                 "semi_synthetic_research_gate_passed": True,
                 "release_gate_passed": True, "production_validated": True,
                 "automatic_processing_enabled": True,
                 "gate": {"scope": "semi_synthetic_epoch_holdout_only"}}
        report_path.write_text(json.dumps(saved), encoding="utf-8")
        before = report_path.read_bytes()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [str(PYTHON), "-c", preflight, "--root", str(self.directory),
             "--model", str(self.model_path), "--config", str(self.config_path),
             "--threads", "2", "--check-only"], cwd=self.directory, env=env,
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        report = self.readiness(result)
        self.assertNotEqual(0, result.returncode)  # A saved research pass cannot replace a model.
        self.assert_safe(report)
        summary = report["research_acceptance_summary"]
        self.assertTrue(summary["semi_synthetic_research_gate_passed"])
        self.assertEqual(7, summary["test_window_count"])
        self.assertEqual("NOT production approval", summary["notice"])
        self.assertEqual("not_assessed", summary["applies_to_selected_model"])
        self.assertEqual(before, report_path.read_bytes())
        self.assertFalse(self.output.exists())

    def fixture_checkpoint(self):
        """Random finite fixture weights and explicit primitive provenance; no training/data access."""
        checkpoint = {
            "checkpoint_schema_version": 1, "model_version": v2.MODEL_VERSION,
            "config": deepcopy(self.config), "architecture": deepcopy(self.config["model"]),
            "model_state": v2.build_model(self.config).state_dict(),
            "calibrated_threshold": 1.0, "validation_threshold": 1.0,
            "threshold_calibration": {
                "scope": "validation_only", "comparison": ">", "production_calibrated": False,
                "method": "clean_score_order_statistic", "score_reduction": "per_channel_window_mean",
                "empirical_only": True, "clean_window_count": 1,
                "observed_clean_false_replacement_fraction": 0.0,
                "max_clean_false_replacement_fraction": 0.01,
            },
        }
        v2.torch.save(checkpoint, self.model_path)

    @unittest.skipIf(v2.torch is None, "optional PyTorch unavailable")
    def test_valid_fixture_without_session_only_reports_diagnostic_readiness(self):
        self.fixture_checkpoint()
        model_hash = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        result = self.run_launcher()
        report = self.readiness(result)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assert_safe(report)
        self.assertTrue(report["diagnostic_ready"])
        self.assertEqual("loaded_weights_only", report["model_load_status"])
        self.assertEqual("readiness_only", report["execution"])
        self.assertIn("-CheckOnly", result.stdout)
        self.assertFalse(self.output.exists())
        self.assertEqual(model_hash, hashlib.sha256(self.model_path.read_bytes()).hexdigest())

    @unittest.skipIf(v2.torch is None, "optional PyTorch unavailable")
    def test_check_only_with_explicit_session_validates_but_does_not_process(self):
        self.fixture_checkpoint()
        result = self.run_launcher("-CheckOnly", "-SessionDir", self.session, "-OutputDir", self.output)
        report = self.readiness(result)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assert_safe(report)
        self.assertEqual("valid_new_external_directory", report["output_validation_status"])
        self.assertTrue(report["diagnostic_ready"])
        self.assertFalse(self.output.exists())
        self.assertNotIn("completed", [item.get("status") for item in json_reports(result.stdout)])

    @unittest.skipIf(v2.torch is None, "optional PyTorch unavailable")
    def test_checkpoint_requiring_pickle_objects_is_rejected_without_unsafe_fallback(self):
        v2.torch.save({"unsupported_object": self.session}, self.model_path)
        result = self.run_launcher("-CheckOnly")
        report = self.readiness(result)
        self.assertNotEqual(0, result.returncode)
        self.assert_safe(report)
        self.assertEqual("invalid_checkpoint", report["model_load_status"])
        self.assertFalse(report["diagnostic_ready"])
        self.assertIn("cannot safely load", report["error"])

    @unittest.skipIf(v2.torch is None, "optional PyTorch unavailable")
    def test_explicit_fixture_session_runs_offline_without_mutating_raw_or_official_quality(self):
        self.fixture_checkpoint()
        result = self.run_launcher("-SessionDir", self.session, "-OutputDir", self.output, "-Threads", "2")
        report = self.readiness(result)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assert_safe(report)
        self.assertEqual("explicit_offline_session", report["execution"])
        derived = json.loads((self.output / "quality_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual("completed", derived["status"])
        self.assertEqual(str(self.session), derived["source"])
        self.assertEqual(str(self.output), derived["output_dir"])
        self.assertEqual(2, derived["threads"])
        self.assertTrue(derived["source_hashes_unchanged"])
        self.assertTrue(derived["diagnostic_only"])
        for key in ("automatic_processing_enabled", "release_gate_passed", "production_validated"):
            self.assertIs(False, derived[key])
        self.assertEqual(self.source_before, hashes(self.session))
        self.assertTrue((self.output / "denoised_signal.npz").is_file())


if __name__ == "__main__":
    unittest.main()
