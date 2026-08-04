#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--settings',required=True,type=Path); args=ap.parse_args()
    cfg=json.loads(args.settings.read_text(encoding='utf-8'))
    expected=Path(cfg['training']['b1_direct_standalone_merged_output'])
    entry=cfg.get('models',{}).get('rs-eot-b1-direct')
    if not isinstance(entry,dict) or Path(entry.get('path','')) != expected:
        print('[FAIL] settings model entry rs-eot-b1-direct is missing or inconsistent')
        return 2
    if not (expected/'config.json').is_file():
        print('[FAIL] B1 Direct merged model is incomplete:',expected)
        return 2
    print('[PASS] B1 Direct model registered:',expected)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
