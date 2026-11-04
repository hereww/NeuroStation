"""Replay saved quality blocks; does not connect to a device or modify the source."""
import argparse,json,sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eeg_tools.quality.replay import replay_quality

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('blocks',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();r=replay_quality(a.blocks);text=json.dumps(r,ensure_ascii=False,indent=2)
    if a.output:
        with a.output.open('x',encoding='utf-8') as f:f.write(text+'\n')
    print(text);return 0 if r['online_replay_match'] and r['raw_hashes_unchanged'] else 2
if __name__=='__main__':raise SystemExit(main())
