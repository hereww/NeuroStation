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


_WINDOWS_OPENBCI_JARS = (
    "LSLLink.jar",
    "OpenBCI_GUI.jar",
    "core.jar",
    "gluegen-rt.jar",
    "jogl-all.jar",
    "jna-platform.jar",
    "jna.jar",
    "jl1.0.1.jar",
    "jsminim.jar",
    "minim.jar",
    "mp3spi1.9.5.jar",
    "tritonus_aos.jar",
    "tritonus_share.jar",
    "jssc.jar",
    "serial.jar",
    "net.jar",
    "grafica.jar",
    "GifAnimation.jar",
    "oscP5.jar",
    "udp.jar",
    "jSerialComm.jar",
    "brainflow.jar",
    "commons-codec-1.4.jar",
    "commons-logging-1.1.1.jar",
    "httpclient-4.1.2.jar",
    "httpclient-cache-4.1.2.jar",
    "httpcore-4.1.2.jar",
    "httpmime-4.1.2.jar",
    "httprequests_processing.jar",
    "controlP5.jar",
    "openbci_gui_helpers.jar",
    "ssdp_client.jar",
)


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
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if sys.platform == "win32":
            creation_flags = (
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                | getattr(subprocess, "DETACHED_PROCESS", 0)
            )
            if creation_flags:
                options["creationflags"] = creation_flags
        process = subprocess.Popen(self._launch_command(state.executable), **options)
        return process.pid

    @classmethod
    def _launch_command(cls, executable: Path) -> list[str]:
        """Build the detached command used by the portable Windows launcher.

        The exported Processing executable works on its own, but the portable
        launcher shipped with OpenBCI GUI starts the bundled JVM explicitly.
        That path also disables system proxies, which prevents startup from
        waiting on an unreachable proxy in the workstation environment.
        """

        if sys.platform != "win32":
            return [str(executable)]

        runtime_root = executable.parent
        javaw = runtime_root / "java" / "bin" / "javaw.exe"
        library_root = runtime_root / "lib"
        jars = [library_root / name for name in _WINDOWS_OPENBCI_JARS]
        if not javaw.is_file() or not all(path.is_file() for path in jars):
            return [str(executable)]

        return [
            str(javaw),
            "-Djna.nosys=true",
            "-Djava.net.useSystemProxies=false",
            "-Dhttp.proxyHost=127.0.0.1",
            "-Dhttp.proxyPort=1",
            "-Dhttps.proxyHost=127.0.0.1",
            "-Dhttps.proxyPort=1",
            "-DsocksProxyHost=127.0.0.1",
            "-DsocksProxyPort=1",
            f"-Djava.library.path={library_root}",
            "-classpath",
            ";".join(str(path) for path in jars),
            "OpenBCI_GUI",
        ]

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
