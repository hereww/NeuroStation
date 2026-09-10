from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eeg_tools.workstation.openbci_workspace import OpenBCIWorkspaceManager


class OpenBCIWorkspaceManagerTests(unittest.TestCase):
    def make_project(self, root: Path) -> OpenBCIWorkspaceManager:
        integration = root / "integrations" / "openbci_gui"
        integration.mkdir(parents=True)
        (integration / "upstream.lock.json").write_text(
            json.dumps(
                {
                    "repository": "https://github.com/OpenBCI/OpenBCI_GUI.git",
                    "revision": "abc",
                    "source_directory": ".vendor/OpenBCI_GUI",
                }
            ),
            encoding="utf-8",
        )
        return OpenBCIWorkspaceManager(root)

    def test_missing_source_is_reported_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = self.make_project(Path(directory)).status()
            self.assertFalse(state.source_ready)
            self.assertFalse(state.overlay_ready)
            self.assertFalse(state.executable_ready)

    def test_explicit_executable_is_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = self.make_project(root)
            executable = root / ("OpenBCI_GUI.exe" if os.name == "nt" else "OpenBCI_GUI")
            executable.write_bytes(b"placeholder")
            with patch.dict(
                os.environ,
                {"NEUROSTATION_OPENBCI_GUI_EXECUTABLE": str(executable)},
            ):
                state = manager.status()
            self.assertTrue(state.executable_ready)
            self.assertEqual(executable.resolve(), state.executable)

    def test_overlay_requires_matching_catalogs_and_translated_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            data = source / "OpenBCI_GUI" / "data" / "workstation-i18n"
            data.mkdir(parents=True)
            (source / "OpenBCI_GUI" / "WorkstationI18n.pde").write_text(
                "String tr(String key) { return key; }", encoding="utf-8"
            )
            (source / "OpenBCI_GUI" / "TopNav.pde").write_text(
                'createControlPanelCollapser(tr("nav.system_control_panel"), 0);',
                encoding="utf-8",
            )
            for name, value in (("en-US.json", "System"), ("zh-CN.json", "系统")):
                (data / name).write_text(
                    json.dumps({"nav.system_control_panel": value}, ensure_ascii=False),
                    encoding="utf-8",
                )
            self.assertTrue(OpenBCIWorkspaceManager._overlay_ready(source))
            (data / "zh-CN.json").write_text("{}", encoding="utf-8")
            self.assertFalse(OpenBCIWorkspaceManager._overlay_ready(source))


if __name__ == "__main__":
    unittest.main()
