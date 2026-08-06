#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--settings',required=True,type=Path); args=ap.parse_args()
    cfg=json.loads(args.settings.read_text(encoding='utf-8'))
    expected=Path(cfg['training']['b2_merged_output'])
    entry=cfg.get('models',{}).get('rs_eot_b2_socratic')
    if not isinstance(entry,dict) or Path(str(entry.get('path',''))) != expected:
        print('[FAIL] settings model entry rs_eot_b2_socratic is missing or inconsistent')
        return 2
    if not (expected/'config.json').is_file():
        print('[FAIL] B2 merged model is incomplete:',expected)
        return 2
    print('[PASS] B2 Socratic model registered:',expected)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
