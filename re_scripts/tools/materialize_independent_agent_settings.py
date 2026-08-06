#!/usr/bin/env python3
from __future__ import annotations
import argparse,copy,json
from pathlib import Path

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--settings',required=True,type=Path); ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--reasoner-model',required=True); ap.add_argument('--perceiver-model',required=True); ap.add_argument('--verifier-model',required=True)
    args=ap.parse_args(); cfg=json.loads(args.settings.read_text(encoding='utf-8'))
    specs={
      'reasoner':(args.reasoner_model,'local-reasoner',8001,False),
      'perceiver':(args.perceiver_model,'local-perceiver',8002,True),
      'verifier':(args.verifier_model,'local-verifier',8003,False),
    }
    for role,(path,name,port,mm) in specs.items():
        p=Path(path).expanduser().resolve()
        if not (p/'config.json').is_file(): raise SystemExit(f'{role} model config missing: {p}/config.json')
        r=cfg.setdefault('agents',{}).setdefault(role,{})
        r.update({'model_path':str(p),'served_name':name,'base_url':f'http://127.0.0.1:{port}/v1','host':'127.0.0.1','port':port,'is_multimodal':mm,'deployment_mode':'independent_three_services'})
    cfg['independent_agents_v1_2_3']={'enabled':True,'physical_servers':3,'logical_roles':3,'contexts_separate':True,'models':{k:v[0] for k,v in specs.items()}}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('[PASS] independent three-agent settings:',args.output); return 0
if __name__=='__main__': raise SystemExit(main())
