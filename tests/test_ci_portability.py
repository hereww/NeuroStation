"""Regression checks for clean CI checkout, never hardware acceptance."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from scripts.verify_denoiser_v4_artifacts import checked_path

ROOT = Path(__file__).resolve().parents[1]


class PortableContractTests(unittest.TestCase):
    def test_frozen_configs_retain_original_exact_bytes(self):
        expected = {
            "v6": "f1deaf6f28add5f14f24719675fb071f1c27c62befc7bafafc47c3e8c66eb8f0",
            "v8": "484950220254e0b2c256d8cea2bf4f5d8f9084fe7aa4616f406d5e240553a16a",
        }
        attrs = (ROOT / ".gitattributes").read_text()
        for version, digest in expected.items():
            path = f"configs/learned_denoiser_{version}_ssvep.json"
            with self.subTest(version=version):
                self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest)
                self.assertIn(path + " -text", attrs)

    def test_cross_platform_escape_and_ads_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            for relative in ("../outside", r"..\outside", r"a\..\outside", r"C:\outside", r"\server\share", "report.json:stream"):
                with self.subTest(relative=relative), self.assertRaises(ValueError):
                    checked_path(root, relative)
            self.assertEqual(checked_path(root, "nested/report.json"), root / "nested/report.json")

    def test_workflows_keep_tests_and_desktop_dependencies_separate(self):
        for name in ("desktop.yml", "release.yml"):
            text = (ROOT / ".github/workflows" / name).read_text()
            with self.subTest(workflow=name):
                self.assertIn("C:/NeuroStation-ci", text)
                self.assertIn("PYTHONUTF8:", text)
                self.assertIn("Canonical temporary paths", text)
                self.assertIn(".venv-ci-tests", text)
                self.assertIn('"$TEST_PYTHON" -m unittest discover -s tests -v', text)
                self.assertNotIn("NEUROSTATION_VERIFY_HISTORICAL_EVIDENCE: ", text)
        self.assertNotIn("torch", (ROOT / "requirements.txt").read_text().lower())
