#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

def deep_merge(base, old):
    if isinstance(base, dict) and isinstance(old, dict):
        out=dict(base)
        for k,v in old.items():
            out[k]=deep_merge(out[k],v) if k in out else v
        return out
    return old

ap=argparse.ArgumentParser()
ap.add_argument('--old',required=True)
ap.add_argument('--template',default='settings.example.json')
ap.add_argument('--output',default='settings.json')
a=ap.parse_args()
base=json.loads(Path(a.template).read_text(encoding='utf-8'))
old=json.loads(Path(a.old).read_text(encoding='utf-8'))
out=deep_merge(base,old)
t=out.setdefault('training',{})
t.setdefault('overwrite_cache',True)
t.setdefault('require_agents_stopped',True)
t.setdefault('auto_stop_agents_before_sft',True)
t['dataset_registry_mode']='self_contained_dataset_dir'
Path(a.output).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(f'[SETTINGS MIGRATE] PASS -> {a.output}')
