"""Synthetic tests for the v7 diagnostic compatibility patch."""
import json
import subprocess
import sys
import unittest

from eeg_tools import ssvep_v7_internal_eval as base
from eeg_tools import ssvep_v7_internal_eval_patch as patch


from tests.historical_evidence import historical_evidence

class TorchVersionLike(str):
    pass


class PatchTests(unittest.TestCase):
    def test_normalize_gpu_versions_only_changes_string_subclasses(self):
        report = {
            "gpu": {
                "torch_version": TorchVersionLike("2.14.1+cu130"),
                "cuda_version": "13.0",
                "device": "cuda:0",
            },
            "case_count": 4200,
        }
        self.assertFalse(base.finite_tree(report))
        patch.normalize_gpu_versions(report)
        self.assertIs(type(report["gpu"]["torch_version"]), str)
        self.assertIs(type(report["gpu"]["cuda_version"]), str)
        self.assertTrue(base.finite_tree(report))
        self.assertEqual(report["case_count"], 4200)

    def test_deviation_is_explicit_and_formal(self):
        self.assertEqual(patch.DEVIATION["type"], "finite_tree_type_compatibility")
        self.assertIn("serialization metadata", patch.DEVIATION["scope"])

    def test_patch_cold_import_does_not_load_torch(self):
        cmd = [
            sys.executable,
            "-B",
            "-c",
            "import sys; import eeg_tools.ssvep_v7_internal_eval_patch; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules",
        ]
        p = subprocess.run(cmd, cwd=patch.ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)

    @historical_evidence
    def test_check_only_publishes_a_patch_scoped_report(self):
        report = patch.evaluate(check_only=True, prefix="analysis_outputs/ssvep_v7_internal_diagnostic_patch_test")
        self.assertTrue(report["check_only"])
        self.assertEqual(report["stage"], "v7_fixed_internal_diagnostic_patch")
        self.assertFalse(report["all_internal_gates_passed"])
        self.assertTrue(report["patch"]["applied"])
        for suffix in (".json", ".md"):
            (patch.ROOT / ("analysis_outputs/ssvep_v7_internal_diagnostic_patch_test_check_only" + suffix)).unlink(missing_ok=True)


class SyntheticPatchPublicationTests(unittest.TestCase):
    """Exercise real freeze verification/publication without private records."""
    def test_synthetic_check_only_publishes_and_rejects_changed_code(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch as mock_patch
        from tests.test_ssvep_v7_internal_eval import PreflightFixture
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            fixture = PreflightFixture(root)
            with mock_patch.object(base, "CONFIG_SHA256", fixture.config_sha), mock_patch.object(base, "PREREG_SHA256", fixture.prereg_sha):
                state = base.preflight(root, check_only=True)
                for rel in patch.PATCH_CODE_PATHS:
                    fixture.put(rel, ("SYNTHETIC code only " + rel).encode())
                code = {"schema_version": 1, "status": "formal",
                        "stage": "frozen_before_v7_internal_diagnostic_patch_code_execution",
                        "files": {rel: fixture.digest(rel) for rel in patch.PATCH_CODE_PATHS}}
                fixture.put(patch.PATCH_CODE_LOCK, json.dumps(code).encode())
                lock = {"schema_version": 1, "status": "formal",
                        "stage": "frozen_before_v7_internal_diagnostic_patch_execution",
                        "deviation": patch.DEVIATION,
                        "scope": dict.fromkeys(base.FALSE_FLAGS, False),
                        "files": patch._bound_hashes(state, fixture.digest(patch.PATCH_CODE_LOCK)),
                        "base_evaluation_lock_sha256": state.lock_sha256,
                        "base_code_lock_sha256": state.code_sha256}
                fixture.put(patch.PATCH_EVAL_LOCK, json.dumps(lock).encode())
                report = patch.evaluate(root, check_only=True, prefix="analysis_outputs/ssvep_v7_internal_diagnostic_patch_synthetic")
                self.assertTrue(report["check_only"])
                self.assertTrue(report["patch"]["applied"])
                for flag in ("runtime_enabled", "production_validated", "waveforms_loaded", "checkpoint_loaded", "cuda_initialized", "all_internal_gates_passed"):
                    self.assertIs(report[flag], False)
                output = root / "analysis_outputs/ssvep_v7_internal_diagnostic_patch_synthetic_check_only.json"
                self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)
                fixture.put(patch.PATCH_CODE_PATHS[0], b"CHANGED SYNTHETIC SOURCE")
                with self.assertRaisesRegex(ValueError, "patch code changed"):
                    patch.preflight(root, check_only=True)


if __name__ == "__main__":
    unittest.main()
