"""Launch the integrated NeuroStation desktop workstation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from neurostation_contract import (
    PRODUCT_DESCRIPTION,
    PRODUCT_NAME,
    PRODUCT_SEMVER,
    PRODUCT_VERSION,
    RELEASE_DATE,
    CaptureMode,
    default_save_directory,
    default_user_channel_config_path,
    default_user_protocol_path,
)
from neurostation_diagnostics import DiagnosticStore


ROOT = Path(__file__).resolve().parent


def _configure_bundled_native_libraries() -> None:
    """Expose bundled macOS dylibs before a BrainFlow worker is started."""

    if sys.platform != "darwin":
        return
    library_root = ROOT / "brainflow" / "lib"
    if not library_root.is_dir():
        return
    path = str(library_root)
    for variable in ("DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH"):
        existing = os.environ.get(variable, "")
        os.environ[variable] = os.pathsep.join(
            value for value in (path, existing) if value
        )


def build_diagnostics(gateway=None) -> dict:
    """Build the same metadata-only report shown by the desktop diagnostics page."""

    bundled_protocol_path = ROOT / "configs" / "protocols" / "ssvep_four_target_v2.json"
    protocol_path = (
        default_user_protocol_path()
        if default_user_protocol_path().is_file()
        else bundled_protocol_path
    )
    final_channel_path = default_user_channel_config_path()
    bundled_final_channel_path = ROOT / "configs" / "channel_config_v1.json"
    auto_channel_path = ROOT / "configs" / "channel_config_v1_auto.json"
    template_channel_path = ROOT / "configs" / "channel_config_v1_template.json"
    channel_path = (
        final_channel_path
        if final_channel_path.is_file()
        else (
            bundled_final_channel_path
            if bundled_final_channel_path.is_file()
            else (auto_channel_path if auto_channel_path.is_file() else template_channel_path)
        )
    )

    if gateway is None:
        from eeg_tools.workstation.desktop_gateway import DesktopGateway

        gateway = DesktopGateway(
            protocol_path=protocol_path,
            channel_config_path=channel_path,
        )
    report = DiagnosticStore(persist=False).build_report(gateway)
    report["application"].update(
        {
            "release_date": RELEASE_DATE,
            "description": PRODUCT_DESCRIPTION,
            "release_note": (
                "Windows standalone release for research and teaching validation; "
                "not a medical device."
            ),
        }
    )
    openbci = gateway.openbci_status
    report["openbci_gui"] = {
        "source_ready": openbci.source_ready,
        "overlay_ready": openbci.overlay_ready,
        "executable_ready": openbci.executable_ready,
        "revision": openbci.revision,
    }
    device = report.get("hardware", {}).get("device", {})
    report["hardware"]["cyton_connected"] = bool(
        device.get("connected") and not device.get("simulated")
    )
    report["hardware"]["note"] = (
        "Diagnostics discovers serial endpoints but does not open the serial port; "
        "Cyton readiness is verified by BrainFlow at task start."
    )
    return report


def configure_storage_defaults() -> bool:
    """Keep installed-user storage unless an isolated runtime is explicitly selected.

    Existing environment overrides always win. No data or settings are migrated.
    """
    from eeg_tools.runtime_bindings import runtime_config

    isolated_root = runtime_config().get("output_root") or os.environ.get("NEUROSTATION_QUALITY_ROOT")
    if isolated_root:
        root = Path(isolated_root)
        os.environ.setdefault("NEUROSTATION_DATASETS", str(root / "recordings"))
        os.environ.setdefault("NEUROSTATION_SETTINGS", str(root / "settings"))
        os.environ.setdefault("NEUROSTATION_DIAGNOSTICS", str(root / "diagnostics"))
    return bool(isolated_root or os.environ.get("NEUROSTATION_RUNTIME_CONFIG"))


def main() -> int:
    isolated_runtime = configure_storage_defaults()
    _configure_bundled_native_libraries()
    if len(sys.argv) > 1 and sys.argv[1] == "--realtime-commands":
        from apps.realtime_commands import main as realtime_main
        return realtime_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "--realtime-capture-worker":
        from eeg_tools.realtime_capture import main as capture_main
        return capture_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "--acquisition-worker":
        from eeg_tools.workstation.acquisition_worker import main as worker_main

        return worker_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "--learned-monitor":
        from eeg_tools.workstation.learned_monitor import main as monitor_main

        return monitor_main(sys.argv[2:])
    parser = argparse.ArgumentParser(
        description=f"{PRODUCT_NAME} {PRODUCT_VERSION} EEG acquisition workstation"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"{PRODUCT_NAME} {PRODUCT_VERSION} ({PRODUCT_SEMVER})",
    )
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
        "--port",
        default="AUTO",
        help="Cyton serial port (default: AUTO)",
    )
    parser.add_argument("--quality-page", action="store_true", help="Open the signal and quality workspace without starting acquisition")
    parser.add_argument("--quality-workflow-smoke", type=Path, help="Run the synthetic demo/replay/report GUI workflow into a fresh output folder, then exit")
    arguments = parser.parse_args()
    from eeg_tools.workstation.desktop_gateway import DesktopGateway

    channel_config = default_user_channel_config_path()
    if not channel_config.is_file():
        channel_config = ROOT / "configs" / "channel_config_v1.json"
    if not channel_config.is_file():
        channel_config = ROOT / "configs" / "channel_config_v1_auto.json"
    if not channel_config.is_file():
        channel_config = ROOT / "configs" / "channel_config_v1_template.json"
    protocol_path = default_user_protocol_path()
    if not protocol_path.is_file():
        protocol_path = ROOT / "configs" / "protocols" / "ssvep_four_target_v2.json"
    gateway = DesktopGateway(
        protocol_path=protocol_path,
        channel_config_path=channel_config,
        dataset_root=arguments.dataset_root or default_save_directory(),
    )
    gateway.auto_import_default = False  # Never automatically copy personal recordings at startup.
    if arguments.diagnostics:
        report = build_diagnostics(gateway)
        report["root"] = str(ROOT)
        # Keep diagnostics machine-readable on Windows consoles and pipes
        # whose active code page cannot encode the localized report.
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return 0
    try:
        from PySide6.QtCore import QTimer, QSettings
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
    application.setApplicationName(PRODUCT_NAME)
    application.setApplicationVersion(PRODUCT_SEMVER)
    application.setOrganizationName(PRODUCT_NAME)
    window = MainWindow(
        gateway=gateway,
        locale=arguments.language,
        persist_settings=not (arguments.smoke_test or arguments.quality_workflow_smoke),
        settings=(QSettings("NeuroStationQualityUpgrade", "NeuroStationQualityUpgrade")
                  if isolated_runtime else QSettings("NeuroStation", "NeuroStation")),
        save_directory_override=arguments.dataset_root,
        initial_mode=CaptureMode.CYTON,
        initial_port=arguments.port,
    )
    if arguments.quality_page or arguments.quality_workflow_smoke:
        window.navigate("quality")
    window.show()
    if arguments.quality_workflow_smoke:
        from apps.workstation_ui.quality_smoke import run_quality_workflow_smoke
        result = run_quality_workflow_smoke(window, arguments.quality_workflow_smoke)
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0 if result["all_passed"] else 2
    if arguments.smoke_test:
        QTimer.singleShot(500, application.quit)
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
