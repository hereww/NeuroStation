from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eeg_tools.workstation.openbci_workspace import (
    OpenBCIWorkspaceManager,
    OpenBCIWorkspaceState,
    _WINDOWS_OPENBCI_JARS,
)


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

    def test_windows_command_matches_portable_openbci_launcher(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            executable = runtime / "OpenBCI_GUI.exe"
            executable.write_bytes(b"placeholder")
            (runtime / "java" / "bin").mkdir(parents=True)
            (runtime / "java" / "bin" / "javaw.exe").write_bytes(b"placeholder")
            (runtime / "lib").mkdir()
            for name in _WINDOWS_OPENBCI_JARS:
                (runtime / "lib" / name).write_bytes(b"placeholder")

            with patch("eeg_tools.workstation.openbci_workspace.sys.platform", "win32"):
                command = OpenBCIWorkspaceManager._launch_command(executable)

            self.assertEqual(runtime / "java" / "bin" / "javaw.exe", Path(command[0]))
            self.assertIn("-Djava.net.useSystemProxies=false", command)
            self.assertIn("-Dhttp.proxyHost=127.0.0.1", command)
            self.assertIn("-Dhttp.proxyPort=1", command)
            self.assertIn("-Djava.library.path=" + str(runtime / "lib"), command)
            self.assertEqual("OpenBCI_GUI", command[-1])
            classpath = command[command.index("-classpath") + 1]
            self.assertIn("OpenBCI_GUI.jar", classpath)
            self.assertIn(";", classpath)

    def test_windows_command_falls_back_when_portable_runtime_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "OpenBCI_GUI.exe"
            executable.write_bytes(b"placeholder")
            with patch("eeg_tools.workstation.openbci_workspace.sys.platform", "win32"):
                command = OpenBCIWorkspaceManager._launch_command(executable)
            self.assertEqual([str(executable)], command)

    def test_launch_detaches_gui_stdio_and_passes_dataset_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "OpenBCI_GUI.exe"
            executable.write_bytes(b"placeholder")
            manager = self.make_project(root)
            state = OpenBCIWorkspaceState(
                source_ready=True,
                overlay_ready=True,
                executable_ready=True,
                revision="abc",
                executable=executable,
            )
            with (
                patch.object(manager, "status", return_value=state),
                patch("eeg_tools.workstation.openbci_workspace.sys.platform", "win32"),
                patch("eeg_tools.workstation.openbci_workspace.subprocess.Popen") as popen,
            ):
                popen.return_value.pid = 1234
                process_id = manager.launch(
                    locale="zh-CN",
                    dataset_root=root / "Datasets",
                )

            self.assertEqual(1234, process_id)
            options = popen.call_args.kwargs
            self.assertIs(options["stdin"], subprocess.DEVNULL)
            self.assertIs(options["stdout"], subprocess.DEVNULL)
            self.assertIs(options["stderr"], subprocess.DEVNULL)
            self.assertEqual("zh-CN", options["env"]["NEUROSTATION_LANGUAGE"])
            self.assertEqual(
                str((root / "Datasets").resolve()),
                options["env"]["NEUROSTATION_DATASETS"],
            )
            self.assertTrue(options.get("creationflags", 0))


if __name__ == "__main__":
    unittest.main()
