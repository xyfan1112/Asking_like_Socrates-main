#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, platform, subprocess, sys
from pathlib import Path
from common import load_settings, settings_from_cli, write_json


def cmd(command):
    try: return subprocess.run(command, shell=True, text=True, capture_output=True, timeout=30).stdout.strip()
    except Exception as e: return f"ERROR:{e}"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--settings'); args=ap.parse_args()
    sp=settings_from_cli(__file__,args.settings); s=load_settings(sp); p=s['paths']; proj=s['project']
    report={
      'settings':str(sp),'python':sys.version,'platform':platform.platform(),
      'nvidia_smi_L':cmd('nvidia-smi -L'),'nvidia_smi':cmd('nvidia-smi'),
      'git_head':cmd(f"cd {proj['project_root']} 2>/dev/null && git rev-parse HEAD"),
      'git_status':cmd(f"cd {proj['project_root']} 2>/dev/null && git status --short"),
      'paths':{},'models':{},'agents':{}
    }
    for k,v in p.items(): report['paths'][k]={'path':v,'exists':Path(v).expanduser().exists()}
    for k,v in s['models'].items(): report['models'][k]={'path':v['path'],'config_exists':(Path(v['path'])/'config.json').is_file()}
    for k,v in s['agents'].items(): report['agents'][k]={'path':v['model_path'],'config_exists':(Path(v['model_path'])/'config.json').is_file(),'gpu':v['gpu'],'url':v['base_url']}
    out=Path(p['pipeline_work_root'])/'reports'/'preflight.json'; write_json(out,report)
    print(json.dumps(report,ensure_ascii=False,indent=2)); print(f'保存: {out}')
if __name__=='__main__': main()
