"""Run with python -m apps.workstation_ui.main or python apps/workstation_ui/main.py."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys

from neurostation_contract import (
    CaptureMode,
    PRODUCT_NAME,
    PRODUCT_SEMVER,
    PRODUCT_VERSION,
    default_user_channel_config_path,
    default_user_protocol_path,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"{PRODUCT_NAME} {PRODUCT_VERSION} desktop acquisition workstation"
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
        help="Directory for session files (default: Documents/NeuroStation/Datasets)",
    )
    parser.add_argument(
        "--port",
        default="AUTO",
        help="Cyton serial port (default: AUTO)",
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
    application.setApplicationName(PRODUCT_NAME)
    application.setApplicationVersion(PRODUCT_SEMVER)
    application.setOrganizationName(PRODUCT_NAME)
    root = Path(__file__).resolve().parents[2]
    if arguments.dataset_root is not None:
        session_root = arguments.dataset_root.expanduser().resolve()
    else:
        from neurostation_contract import default_save_directory

        session_root = default_save_directory()
    bundled_protocol_path = root / "configs" / "protocols" / "ssvep_four_target_v2.json"
    bundled_channel_path = root / "configs" / "channel_config_v1_auto.json"
    protocol_path = (
        default_user_protocol_path()
        if default_user_protocol_path().is_file()
        else bundled_protocol_path
    )
    channel_config_path = (
        default_user_channel_config_path()
        if default_user_channel_config_path().is_file()
        else bundled_channel_path
    )
    gateway = DesktopGateway(
        protocol_path=protocol_path,
        channel_config_path=channel_config_path,
        dataset_root=session_root,
    )
    window = MainWindow(
        gateway=gateway,
        locale=arguments.language,
        persist_settings=True,
        save_directory_override=session_root,
        initial_mode=CaptureMode.CYTON,
        initial_port=arguments.port,
    )
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
