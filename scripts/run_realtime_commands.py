"""Launch the standalone realtime commands UI after formal config validation."""
from __future__ import annotations

from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    from scripts.validate_release_configs import validate_release_configs
    try:
        with redirect_stdout(sys.stderr):
            validate_release_configs()
    except (OSError, ValueError) as exc:
        print(json.dumps({"type": "error", "source": "realtime_eeg_model", "status": "formal",
                          "error": "release_config_invalid", "message": str(exc),
                          "experiment_only": True, "personal_validated": False,
                          "runtime_enabled": False, "production_validated": False,
                          "default_deployment_authorized": False, "executed": False},
                         ensure_ascii=True, allow_nan=False), file=sys.stderr)
        return 2
    from apps.realtime_commands import main as ui_main
    return ui_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
