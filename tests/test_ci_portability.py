"""Regression checks for clean CI checkout, never hardware acceptance."""
import ast
import hashlib
import re
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
                self.assertIn("Nuitka==4.2.1", text)
                self.assertIn('"$TEST_PYTHON" -m pip check', text)
                self.assertIn('"numpy==1.26.4" "scipy==1.14.1"', text)
                self.assertIn('"$TEST_PYTHON" -m unittest discover -s tests -v', text)
                self.assertNotIn("NEUROSTATION_VERIFY_HISTORICAL_EVIDENCE: ", text)
        self.assertNotIn("torch", (ROOT / "requirements.txt").read_text().lower())

    def test_windows_stage_contains_every_imported_root_runtime_module(self):
        build = (ROOT / "scripts/build_desktop.ps1").read_text(encoding="utf-8")
        manifest = build.split("$sourceItems = @(", 1)[1].split("\n)", 1)[0]
        staged = set(re.findall(r'"([^"\n]+)"', manifest))
        project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        entries = project.split("[tool.pyside6-project]", 1)[1].split("files = [", 1)[1].split("]", 1)[0]
        declared = set(ast.literal_eval("[" + entries + "]"))
        sources = [ROOT / "workstation.py"]
        for folder in ("apps", "eeg_tools"):
            sources.extend(p for p in (ROOT / folder).rglob("*.py") if "tests" not in p.parts)
        # Include all shared root modules too, to check their transitive imports.
        sources.extend(ROOT.glob("neurostation_*.py"))
        required = set()
        for source in sources:
            for node in ast.walk(ast.parse(source.read_text(encoding="utf-8-sig"))):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                    names = [node.module]
                else:
                    continue
                for name in names:
                    root_module = name.split(".")[0] + ".py"
                    if (ROOT / root_module).is_file():
                        required.add(root_module)
        self.assertTrue(required)
        self.assertFalse(required - staged, f"Missing staged root modules: {required - staged}")
        self.assertFalse(required - declared, f"Missing declared root modules: {required - declared}")
        self.assertLess(build.index("Staged desktop runtime imports failed"), build.index("& $buildPython @arguments"))
