"""Formal defaults are a permanent release requirement, not a draft bypass."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from eeg_tools.config import ConfigError
from scripts.validate_release_configs import validate_release_configs


ROOT = Path(__file__).resolve().parents[1]


class ReleaseConfigPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        shutil.copytree(ROOT / "configs", self.root / "configs")

    def update(self, relative: str, transform) -> None:
        path = self.root / "configs" / relative
        value = json.loads(path.read_text(encoding="utf-8"))
        transform(value)
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_bundled_release_defaults_are_formal(self) -> None:
        report = validate_release_configs(self.root)
        self.assertEqual("ready", report["status"])
        self.assertEqual("formal", report["protocol_status"])
        self.assertEqual([], report["warnings"])

    def test_draft_or_candidate_protocol_cannot_be_released(self) -> None:
        for status in ("draft_for_workstation_mvp", "draft_for_technical_validation", "formal_candidate", ""):
            with self.subTest(status=status):
                self.update("protocols/ssvep_four_target_v2.json", lambda value: value.update(status=status))
                with self.assertRaisesRegex(ConfigError, "status must be formal"):
                    validate_release_configs(self.root)

    def test_legacy_stimulus_resource_must_also_be_formal(self) -> None:
        self.update("ssvep_config_v1.json", lambda value: value.update(status="draft"))
        with self.assertRaisesRegex(ConfigError, "status must be formal"):
            validate_release_configs(self.root)

    def test_mvp_version_marker_cannot_be_released(self) -> None:
        self.update("channel_config_v1_auto.json", lambda value: value.update(config_version="CHANNEL-V1-MVP"))
        with self.assertRaisesRegex(ConfigError, "must not contain MVP/DRAFT"):
            validate_release_configs(self.root)

    def test_incomplete_default_map_cannot_be_released(self) -> None:
        self.update("channel_config_v1_auto.json", lambda value: value["channels"][0].update(electrode_position=None))
        with self.assertRaisesRegex(ConfigError, "configuration is incomplete"):
            validate_release_configs(self.root)

    def test_new_protocol_resource_must_also_be_formal(self) -> None:
        source = self.root / "configs/protocols/ssvep_four_target_v2.json"
        new_protocol = source.with_name("future_protocol.json")
        value = json.loads(source.read_text(encoding="utf-8"))
        value["status"] = "draft"
        new_protocol.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ConfigError, "status must be formal"):
            validate_release_configs(self.root)

    def test_all_desktop_build_entries_enforce_formal_configs(self) -> None:
        for script in ("build_desktop.ps1", "build_desktop.sh"):
            with self.subTest(script=script):
                text = (ROOT / "scripts" / script).read_text(encoding="utf-8")
                self.assertIn("validate_release_configs.py", text)


if __name__ == "__main__":
    unittest.main()
