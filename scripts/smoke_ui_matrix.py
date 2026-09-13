"""Offscreen UI smoke checks across common Windows scale factors."""
from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    for scale in ("1.0", "1.25", "1.5"):
        environment = os.environ.copy()
        environment["QT_QPA_PLATFORM"] = "offscreen"
        environment["QT_SCALE_FACTOR"] = scale
        code = (
            "from PySide6.QtWidgets import QApplication; "
            "from apps.workstation_ui.app import MainWindow; "
            "from apps.workstation_ui.gateway import MockGateway; "
            "app=QApplication([]); w=MainWindow(MockGateway(), timer_enabled=False); "
            "w.show(); app.processEvents(); "
            "assert w.width() >= 320 and w.height() >= 560; "
            "w.navigate('ssvep'); app.processEvents(); assert not w.grab().isNull(); "
            "w.navigate('datasets'); app.processEvents(); assert not w.grab().isNull(); "
            "w.close()"
        )
        result = subprocess.run([sys.executable, "-c", code], env=environment, check=False)
        if result.returncode:
            print(f"UI smoke failed at scale {scale}", file=sys.stderr)
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
