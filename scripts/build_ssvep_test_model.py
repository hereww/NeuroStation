"""Explicit v9 diagnostic/packaging entrypoint, never modifies v8 artifacts."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).absolute().parents[1]))
from eeg_tools.ssvep_v9_delivery import main
if __name__=='__main__':raise SystemExit(main())
