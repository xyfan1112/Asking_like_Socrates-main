#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,socket,subprocess
from pathlib import Path
from typing import Any


def exists(path: str | Path | None) -> dict[str,Any]:
    if not path:
        return {"path": None, "exists": False}
    p=Path(path)
    return {"path":str(p),"exists":p.exists(),"is_file":p.is_file(),"is_dir":p.is_dir()}


def port_open(port:int)->bool:
    with socket.socket() as s:
        s.settimeout(.2)
        return s.connect_ex(('127.0.0.1',port))==0


def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--settings',required=True,type=Path); args=ap.parse_args()
    if not args.settings.is_file():
        print(json.dumps({"settings":str(args.settings),"exists":False,"hint":"run prepare"},ensure_ascii=False,indent=2)); return 1
    s=json.loads(args.settings.read_text(encoding='utf-8'))
    p=s['paths']; t=s['training']; o=s.get('official_socratic',{}); route=s.get('model_routing',{})
    b0_key=str(route.get('b0_model_key') or 'rs_eot'); b1_key=str(route.get('b1_model_key') or 'rs_eot_b1_direct'); b2_key=str(route.get('b2_model_key') or 'rs_eot_b2_socratic')
    artifacts={
      'data_gate':exists(Path(p['pipeline_work_root'])/'reports'/'data_layer_final_report.json'),
      'direct_train':exists(Path(p['dota128_llamafactory_root'])/'dota128_direct_train_official.json'),
      'agent_inputs':exists(Path(p['pipeline_work_root'])/'agent_inputs'/'train_agent_inputs.jsonl'),
      'socratic_raw':exists(Path(o.get('raw_output_dir',''))/'dota128_train_official.jsonl'),
      'b1_adapter':exists(t.get('b1_direct_standalone_adapter_output')),
      'b1_merged':exists(t.get('b1_direct_standalone_merged_output')),
      'b2_adapter':exists(t.get('b2_adapter_output')),
      'b2_merged':exists(t.get('b2_merged_output')),
      'b0_matrix':exists(Path(p['test_run_root'])/'matrix'/b0_key/'test_matrix_status.json'),
      'b1_matrix':exists(Path(p['test_run_root'])/'matrix'/b1_key/'test_matrix_status.json'),
      'b2_matrix':exists(Path(p['test_run_root'])/'matrix'/b2_key/'test_matrix_status.json'),
    }
    report={
      'schema_version':'status_v1_2_3',
      'settings':str(args.settings),
      'version':s.get('v1_2_3',{}).get('version'),
      'language':s.get('taxonomy',{}).get('qa_language'),
      'dataset':s.get('paths',{}).get('dota128_root'),
      'classes':s.get('taxonomy',{}).get('classes_file'),
      'direct_qa_mode':s.get('data_conversion',{}).get('direct_qa_mode'),
      'model_routing':route,
      'sft':s.get('v1_2_3',{}).get('sft'),
      'evaluation':s.get('v1_2_3',{}).get('evaluation'),
      'ports':{str(port):port_open(port) for port in [8001,8002,8003,8010,8011,8012,8013,8091]},
      'artifacts':artifacts,
    }
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0
if __name__=='__main__': raise SystemExit(main())
