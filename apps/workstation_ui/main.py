"""Run with python -m apps.workstation_ui.main or python apps/workstation_ui/main.py."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="NeuroStation independent desktop UI (simulation only)")
    parser.add_argument("--language", choices=("zh-CN", "en-US"))
    arguments = parser.parse_args()
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    try:
        from PySide6.QtWidgets import QApplication
        from apps.workstation_ui.app import MainWindow
    except ModuleNotFoundError as error:
        if error.name and error.name.startswith("PySide6"):
            print("PySide6 is required. Install: python -m pip install -r apps/workstation_ui/requirements.txt", file=sys.stderr)
            return 2
        raise
    application = QApplication(sys.argv[:1])
    application.setApplicationName("NeuroStation")
    application.setOrganizationName("NeuroStation")
    window = MainWindow(locale=arguments.language, persist_settings=True)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
