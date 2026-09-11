"""Launch the integrated NeuroStation desktop MVP."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import platform
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def _dependency_status(module_name: str) -> dict[str, Any]:
    """Return import status without making diagnostics depend on optional packages."""

    try:
        module = __import__(module_name)
    except Exception as error:  # pragma: no cover - platform/package specific
        return {"available": False, "error": f"{type(error).__name__}: {error}"}
    installed_version = getattr(module, "__version__", None)
    if not installed_version and module_name == "brainflow":
        try:
            from brainflow.board_shim import BoardShim

            installed_version = BoardShim.get_version()
        except Exception:  # pragma: no cover - native runtime specific
            pass
    if not installed_version:
        try:
            installed_version = package_version(module_name)
        except PackageNotFoundError:
            pass
    return {"available": True, "version": str(installed_version) if installed_version else None}


def build_diagnostics() -> dict[str, Any]:
    """Build a JSON-safe preflight report for support and field testing.

    This intentionally reports configuration readiness separately from hardware
    connectivity: a valid Cyton channel map is necessary for a session, but it
    cannot prove that a USB dongle is plugged in or that electrodes are attached.
    """

    protocol_path = ROOT / "configs" / "protocols" / "ssvep_four_target_v2.json"
    stimulus_path = ROOT / "configs" / "ssvep_config_v1.json"
    final_channel_path = ROOT / "configs" / "channel_config_v1.json"
    auto_channel_path = ROOT / "configs" / "channel_config_v1_auto.json"
    template_channel_path = ROOT / "configs" / "channel_config_v1_template.json"
    channel_path = (
        final_channel_path
        if final_channel_path.is_file()
        else (auto_channel_path if auto_channel_path.is_file() else template_channel_path)
    )

    config_report: dict[str, Any] = {
        "protocol_path": str(protocol_path),
        "stimulus_path": str(stimulus_path),
        "channel_config_path": str(channel_path),
        "final_channel_config_present": final_channel_path.is_file(),
        "status": "invalid",
        "warnings": [],
        "error": None,
    }
    try:
        from eeg_tools.config import load_and_validate_configs

        _, _, warnings = load_and_validate_configs(stimulus_path, channel_path)
        config_report["warnings"] = warnings
        config_report["status"] = "ready" if not warnings else "draft"
    except Exception as error:
        config_report["error"] = f"{type(error).__name__}: {error}"

    from eeg_tools.workstation.desktop_gateway import DesktopGateway
    from eeg_tools.workstation.device_discovery import discover_serial_ports

    gateway = DesktopGateway(
        protocol_path=protocol_path,
        channel_config_path=channel_path,
    )
    openbci = gateway.openbci_status
    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "resources": {
            "protocol_ready": protocol_path.is_file(),
            "locales_ready": all(
                (ROOT / "apps" / "workstation_ui" / "locales" / name).is_file()
                for name in ("zh-CN.json", "en-US.json")
            ),
        },
        "dependencies": {
            "PySide6": _dependency_status("PySide6"),
            "brainflow": _dependency_status("brainflow"),
        },
        "configuration": config_report,
        "openbci_gui": {
            "source_ready": openbci.source_ready,
            "overlay_ready": openbci.overlay_ready,
            "executable_ready": openbci.executable_ready,
            "revision": openbci.revision,
        },
        "hardware": {
            "cyton_connected": False,
            "serial_ports": [item.as_dict() for item in discover_serial_ports()],
            "note": "Diagnostics discovers serial endpoints but does not open the serial port; Cyton readiness is verified by BrainFlow at task start.",
        },
    }


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--acquisition-worker":
        from eeg_tools.workstation.acquisition_worker import main as worker_main

        return worker_main(sys.argv[2:])
    parser = argparse.ArgumentParser(description="NeuroStation EEG workstation MVP")
    parser.add_argument("--language", choices=("zh-CN", "en-US"))
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Directory for session metadata and BrainFlow acquisition datasets",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Open the desktop window briefly, then exit (packaging verification)",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Print packaged resource and OpenBCI integration status, then exit",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Launch SSVEP in explicit full-screen visual preview mode",
    )
    arguments = parser.parse_args()
    from eeg_tools.workstation.desktop_gateway import DesktopGateway

    channel_config = ROOT / "configs" / "channel_config_v1.json"
    if not channel_config.is_file():
        channel_config = ROOT / "configs" / "channel_config_v1_auto.json"
    if not channel_config.is_file():
        channel_config = ROOT / "configs" / "channel_config_v1_template.json"
    gateway = DesktopGateway(
        protocol_path=ROOT / "configs" / "protocols" / "ssvep_four_target_v2.json",
        channel_config_path=channel_config,
        dataset_root=arguments.dataset_root,
    )
    if arguments.diagnostics:
        report = build_diagnostics()
        report["root"] = str(ROOT)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication
        from apps.workstation_ui.app import MainWindow
    except ModuleNotFoundError as error:
        if error.name and error.name.startswith("PySide6"):
            print(
                "缺少 PySide6 / PySide6 is required. Install: "
                "python -m pip install -r requirements.txt",
                file=sys.stderr,
            )
            return 2
        raise

    application = QApplication(sys.argv[:1])
    application.setApplicationName("NeuroStation")
    application.setOrganizationName("NeuroStation")
    window = MainWindow(
        gateway=gateway,
        locale=arguments.language,
        persist_settings=True,
        save_directory_override=arguments.dataset_root,
        preview_mode=arguments.preview,
    )
    window.show()
    if arguments.smoke_test:
        QTimer.singleShot(500, application.quit)
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
