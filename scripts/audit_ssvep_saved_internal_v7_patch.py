"""Run the independent JSON-only audit for the v7 patched diagnostic."""
from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
from eeg_tools.ssvep_saved_audit_v7_patch import main

if __name__ == "__main__":
    raise SystemExit(main())
