"""Keep the desktop, package, and release versions in sync."""
from __future__ import annotations

import configparser
import json
from pathlib import Path
import re
import unittest

from neurostation_contract import PRODUCT_NAME, PRODUCT_SEMVER, PRODUCT_VERSION, RELEASE_DATE
from scripts.generate_sbom import build_sbom


ROOT = Path(__file__).resolve().parents[1]


class ReleaseMetadataTests(unittest.TestCase):
    def test_project_and_executable_metadata_match_contract(self) -> None:
        project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        version = re.search(r'^version = "([^"]+)"$', project, re.MULTILINE)
        self.assertIsNotNone(version)
        self.assertEqual(version.group(1), PRODUCT_SEMVER)
        spec = configparser.ConfigParser(interpolation=None)
        spec.read(ROOT / "pysidedeploy.spec", encoding="utf-8")
        arguments = spec["nuitka"]["extra_args"].split()
        self.assertIn(f"--file-version={PRODUCT_SEMVER}", arguments)
        self.assertIn(f"--product-version={PRODUCT_SEMVER}", arguments)
        self.assertIn(f"--file-description={PRODUCT_NAME}_{PRODUCT_VERSION}_EEG_workstation", arguments)

    def test_localized_titles_match_contract(self) -> None:
        for language in ("zh-CN", "en-US"):
            with self.subTest(language=language):
                catalog = json.loads((ROOT / "apps/workstation_ui/locales" / f"{language}.json").read_text(encoding="utf-8"))
                self.assertTrue(catalog["app.title"].startswith(f"{PRODUCT_NAME} {PRODUCT_VERSION} ·"))

    def test_build_and_release_workflow_match_contract(self) -> None:
        build = (ROOT / "scripts/build_desktop.ps1").read_text(encoding="utf-8")
        self.assertIn(f'$releaseVersion = "{PRODUCT_VERSION}"', build)
        self.assertIn(f'$releaseSemver = "{PRODUCT_SEMVER}"', build)
        self.assertIn(f'$releaseDate = "{RELEASE_DATE}"', build)
        notes_name = f"NeuroStation-{PRODUCT_VERSION}-Windows-x64.md"
        archive_name = f"NeuroStation-{PRODUCT_VERSION}-Windows-x64.zip"
        self.assertIn(notes_name, build)
        self.assertIn(archive_name, build)
        workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
        self.assertIn(f'default: "v{PRODUCT_SEMVER}"', workflow)
        self.assertIn(f"name: NeuroStation {PRODUCT_VERSION}", workflow)
        self.assertIn(f"body_path: release/{notes_name}", workflow)
        self.assertIn(f"files: dist/{archive_name}", workflow)
        notes = (ROOT / "release" / notes_name).read_text(encoding="utf-8")
        for field in (f"Product version: `{PRODUCT_VERSION}`", f"Semantic version: `{PRODUCT_SEMVER}`", f"Build date: `{RELEASE_DATE}`", f"Intended release tag: `v{PRODUCT_SEMVER}`"):
            self.assertIn(field, notes)

    def test_sbom_version_matches_contract(self) -> None:
        sbom = build_sbom()
        self.assertEqual(sbom["metadata"]["component"]["version"], PRODUCT_SEMVER)
        self.assertEqual(sbom["serialNumber"], f"urn:neurostation:sbom:{PRODUCT_SEMVER}")


if __name__ == "__main__":
    unittest.main()
