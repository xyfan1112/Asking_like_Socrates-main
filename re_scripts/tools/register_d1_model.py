#!/usr/bin/env python3
"""Verify that the Chinese settings already point to the merged D1 model."""
from __future__ import annotations
import argparse, json
from pathlib import Path


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--settings',required=True,type=Path); args=ap.parse_args()
    cfg=json.loads(args.settings.read_text(encoding='utf-8'))
    expected=Path(cfg['training']['d1_merged_output'])
    entry=cfg.get('models',{}).get('rs_eot_d1_direct_only')
    if not isinstance(entry,dict) or Path(entry.get('path','')) != expected:
        print('[FAIL] settings model entry rs_eot_d1_direct_only is missing or inconsistent')
        return 2
    if not (expected/'config.json').is_file():
        print('[FAIL] D1 merged model is not complete:',expected)
        return 2
    print('[PASS] D1 model registered in settings:',expected)
    return 0

if __name__=='__main__': raise SystemExit(main())
