#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'main_layer'))
from common import *

def to_socratic(x):
 content=x['trace'].strip()+"\n</think>\n"+str(x['teacher_forced_final']).strip()
 return {'messages':[{'role':'user','content':x['query']},{'role':'assistant','content':content}],'images':[x['image_path']],'metadata':{'task':x['task'],'trajectory_bucket':x['bucket'],'id':x['id']}}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--settings'); ap.add_argument('--include-single-round',action='store_true'); args=ap.parse_args(); s=load_settings(settings_from_cli(__file__,args.settings)); audit=Path(s['paths']['trajectory_audit_dir']); agent=Path(s['paths']['pipeline_work_root'])/'agent_inputs'; out=Path(s['paths']['dota128_llamafactory_root']);out.mkdir(parents=True,exist_ok=True)
 strict=json.loads((audit/'accepted_strict.json').read_text(encoding='utf-8')) if (audit/'accepted_strict.json').exists() else []
 relaxed=json.loads((audit/'accepted_single_round.json').read_text(encoding='utf-8')) if args.include_single_round and (audit/'accepted_single_round.json').exists() else []
 soc=[to_socratic(x) for x in strict+relaxed]
 direct=json.loads((agent/'train_direct.json').read_text(encoding='utf-8')); val=json.loads((agent/'val_direct.json').read_text(encoding='utf-8'))
 (out/'dota128_socratic_train.json').write_text(json.dumps(soc,ensure_ascii=False,indent=2),encoding='utf-8')
 (out/'dota128_direct_train.json').write_text(json.dumps(direct,ensure_ascii=False,indent=2),encoding='utf-8')
 (out/'dota128_mixed_train.json').write_text(json.dumps(direct+soc,ensure_ascii=False,indent=2),encoding='utf-8')
 (out/'dota128_direct_val.json').write_text(json.dumps(val,ensure_ascii=False,indent=2),encoding='utf-8')
 info={
  'dota128_direct_train':{'file_name':'dota128_direct_train.json','formatting':'sharegpt','columns':{'messages':'messages','images':'images'},'tags':{'role_tag':'role','content_tag':'content','user_tag':'user','assistant_tag':'assistant'}},
  'dota128_mixed_train':{'file_name':'dota128_mixed_train.json','formatting':'sharegpt','columns':{'messages':'messages','images':'images'},'tags':{'role_tag':'role','content_tag':'content','user_tag':'user','assistant_tag':'assistant'}},
  'dota128_direct_val':{'file_name':'dota128_direct_val.json','formatting':'sharegpt','columns':{'messages':'messages','images':'images'},'tags':{'role_tag':'role','content_tag':'content','user_tag':'user','assistant_tag':'assistant'}}}
 write_json(out/'dataset_info.json',info);report={'direct_train':len(direct),'socratic_train':len(soc),'mixed_train':len(direct)+len(soc),'val':len(val),'include_single_round':args.include_single_round};write_json(out/'build_report.json',report);print(json.dumps(report,ensure_ascii=False,indent=2));return 2 if not soc else 0
if __name__=='__main__':raise SystemExit(main())
