"""Fixed v6 internal diagnostic CLI; no input, selection, device or fs overrides."""
from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

from eeg_tools.ssvep_v6_internal_eval import main

if __name__ == "__main__":
    raise SystemExit(main())
