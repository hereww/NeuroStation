"""Synthetic tests for the v7 diagnostic compatibility patch."""
import json
import subprocess
import sys
import unittest

from eeg_tools import ssvep_v7_internal_eval as base
from eeg_tools import ssvep_v7_internal_eval_patch as patch


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

    def test_check_only_publishes_a_patch_scoped_report(self):
        report = patch.evaluate(check_only=True, prefix="analysis_outputs/ssvep_v7_internal_diagnostic_patch_test")
        self.assertTrue(report["check_only"])
        self.assertEqual(report["stage"], "v7_fixed_internal_diagnostic_patch")
        self.assertFalse(report["all_internal_gates_passed"])
        self.assertTrue(report["patch"]["applied"])
        for suffix in (".json", ".md"):
            (patch.ROOT / ("analysis_outputs/ssvep_v7_internal_diagnostic_patch_test_check_only" + suffix)).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
