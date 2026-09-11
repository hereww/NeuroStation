"""Run with python -m apps.workstation_ui.main or python apps/workstation_ui/main.py."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys
import tempfile


def main() -> int:
    parser = argparse.ArgumentParser(
        description="NeuroStation desktop visual-preview workstation"
    )
    parser.add_argument("--language", choices=("zh-CN", "en-US"))
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Directory for full-screen preview session files (default: system temporary folder)",
    )
    arguments = parser.parse_args()
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    try:
        from PySide6.QtWidgets import QApplication
        from apps.workstation_ui.app import MainWindow
        DesktopGateway = importlib.import_module(
            "eeg_tools.workstation.desktop_gateway"
        ).DesktopGateway
    except ModuleNotFoundError as error:
        if error.name and error.name.startswith("PySide6"):
            print("PySide6 is required. Install: python -m pip install -r apps/workstation_ui/requirements.txt", file=sys.stderr)
            return 2
        raise
    application = QApplication(sys.argv[:1])
    application.setApplicationName("NeuroStation")
    application.setOrganizationName("NeuroStation")
    root = Path(__file__).resolve().parents[2]
    preview_root = (
        arguments.dataset_root.expanduser().resolve()
        if arguments.dataset_root is not None
        else Path(tempfile.gettempdir()) / "NeuroStationPreview"
    )
    gateway = DesktopGateway(
        protocol_path=root / "configs" / "protocols" / "ssvep_four_target_v2.json",
        channel_config_path=root / "configs" / "channel_config_v1_auto.json",
        dataset_root=preview_root,
    )
    window = MainWindow(
        gateway=gateway,
        locale=arguments.language,
        persist_settings=True,
        save_directory_override=preview_root,
        preview_mode=True,
    )
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
