"""Discover and launch the source-rebuilt OpenBCI GUI as a managed process."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys


class OpenBCIWorkspaceError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenBCIWorkspaceState:
    source_ready: bool
    overlay_ready: bool
    executable_ready: bool
    revision: str
    executable: Path | None


class OpenBCIWorkspaceManager:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.lock_path = self.project_root / "integrations" / "openbci_gui" / "upstream.lock.json"

    def status(self) -> OpenBCIWorkspaceState:
        lock = json.loads(self.lock_path.read_text(encoding="utf-8"))
        source = (self.project_root / lock["source_directory"]).resolve()
        source_ready = (source / ".git").exists()
        revision = ""
        if source_ready:
            result = subprocess.run(
                ["git", "-C", str(source), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
            )
            if result.returncode == 0:
                revision = result.stdout.strip()
            source_ready = revision == lock["revision"]
        overlay_ready = source_ready and self._overlay_ready(source)
        executable = self._find_executable(source)
        return OpenBCIWorkspaceState(
            source_ready=source_ready,
            overlay_ready=overlay_ready,
            executable_ready=executable is not None,
            revision=revision,
            executable=executable,
        )

    def launch(self, *, locale: str, dataset_root: Path) -> int:
        state = self.status()
        if state.executable is None:
            raise OpenBCIWorkspaceError("validation.openbci_unavailable")
        environment = os.environ.copy()
        environment["NEUROSTATION_LANGUAGE"] = "zh-CN" if locale.lower().startswith("zh") else "en-US"
        environment["NEUROSTATION_DATASETS"] = str(dataset_root.resolve())
        options: dict = {
            "cwd": str(state.executable.parent),
            "env": environment,
        }
        if sys.platform == "win32":
            options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        process = subprocess.Popen([str(state.executable)], **options)
        return process.pid

    @staticmethod
    def _overlay_ready(source: Path) -> bool:
        i18n = source / "OpenBCI_GUI" / "WorkstationI18n.pde"
        english_path = source / "OpenBCI_GUI" / "data" / "workstation-i18n" / "en-US.json"
        chinese_path = source / "OpenBCI_GUI" / "data" / "workstation-i18n" / "zh-CN.json"
        if not all(path.is_file() for path in (i18n, english_path, chinese_path)):
            return False
        try:
            english = json.loads(english_path.read_text(encoding="utf-8"))
            chinese = json.loads(chinese_path.read_text(encoding="utf-8"))
            top_nav = (source / "OpenBCI_GUI" / "TopNav.pde").read_text(encoding="utf-8")
        except (OSError, json.JSONDecodeError):
            return False
        return set(english) == set(chinese) and 'tr("nav.system_control_panel")' in top_nav

    def _find_executable(self, source: Path) -> Path | None:
        override = os.environ.get("NEUROSTATION_OPENBCI_GUI_EXECUTABLE")
        candidates: list[Path] = []
        if override:
            candidates.append(Path(override).expanduser())
        runtime = self.project_root / "integrations" / "openbci_gui" / "runtime"
        if sys.platform == "win32":
            candidates.extend(
                (
                    runtime / "windows" / "OpenBCI_GUI.exe",
                    source / "application.windows64" / "OpenBCI_GUI.exe",
                    source / "application.windows64" / "OpenBCI_GUI" / "OpenBCI_GUI.exe",
                )
            )
        elif sys.platform == "darwin":
            candidates.extend(
                (
                    runtime / "macos" / "OpenBCI_GUI.app" / "Contents" / "MacOS" / "OpenBCI_GUI",
                    source / "application.macosx" / "OpenBCI_GUI.app" / "Contents" / "MacOS" / "OpenBCI_GUI",
                )
            )
        else:
            candidates.extend(
                (
                    runtime / "linux" / "OpenBCI_GUI",
                    source / "application.linux64" / "OpenBCI_GUI",
                )
            )
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.is_file():
                return resolved
        return None
