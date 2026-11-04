"""Fixed v8 CUDA internal diagnostic entrypoint."""
from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
from eeg_tools.ssvep_v8_internal_eval import main
if __name__ == "__main__":
    raise SystemExit(main())
