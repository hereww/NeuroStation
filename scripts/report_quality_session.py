"""Generate a new immutable quality report. Never modifies raw/history files."""
from pathlib import Path
import argparse,json,sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eeg_tools.quality.reporting import build_session_report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session',type=Path);parser.add_argument('--output',type=Path)
    a=parser.parse_args()
    result=build_session_report(a.session,a.output)
    print(json.dumps({'output_dir':result['output_dir']},ensure_ascii=False));return 0
if __name__=='__main__':raise SystemExit(main())
