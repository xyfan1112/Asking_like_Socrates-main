#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math,re,sys
from collections import Counter,defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'main_layer'))
from common import read_jsonl,write_json

def norm(x):return re.sub(r'[^a-z0-9]+',' ',str(x or '').lower()).strip()
def token_f1(a,b):
 ca,cb=Counter(norm(a).split()),Counter(norm(b).split());inter=sum((ca&cb).values());p=inter/max(sum(ca.values()),1);r=inter/max(sum(cb.values()),1);return 2*p*r/max(p+r,1e-9)
def correct(pred,gt):
 a,b=norm(pred),norm(gt)
 if a==b:return True,'exact'
 na=re.findall(r'-?\d+(?:\.\d+)?',a);nb=re.findall(r'-?\d+(?:\.\d+)?',b)
 if na and nb and len(na)==len(nb) and all(abs(float(x)-float(y))<1e-6 for x,y in zip(na,nb)):return True,'numeric'
 if token_f1(a,b)>=.8:return True,'token_f1'
 return False,'wrong'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--pred',required=True);ap.add_argument('--output');ap.add_argument('--fail-on-incomplete',action='store_true');args=ap.parse_args();path=Path(args.pred);rows=read_jsonl(path,skip_bad=True);manifest_path=path.with_suffix('.manifest.json');manifest=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {};expected=int(manifest.get('expected_runs',len(rows)));k=int(manifest.get('k',0) or 0);by=defaultdict(list);per=[]
 for r in rows:
  if r.get('target_type') not in {None,'vqa'}:continue
  ok,method=correct(r.get('prediction'),r.get('gt'));x={**r,'correct':ok,'judge_method':method};per.append(x);by[r['id']].append(x)
 incomplete=len(rows)!=expected or (k>0 and any(len(g)!=k for g in by.values()));errors=sum(bool(r.get('error')) for r in rows);truncated=sum(str(r.get('finish_reason') or '').lower()=='length' for r in rows);invalid=incomplete or errors>0 or truncated>0
 qvals=[[int(x['correct']) for x in g] for g in by.values()];summary={'status':'INVALID' if invalid else 'VALID','runs':len(per),'all_file_rows':len(rows),'expected_runs':expected,'questions':len(by),'expected_k':k,'incomplete':incomplete,'error_rows':errors,'truncated_rows':truncated,'avg_at_k':sum(sum(v)/len(v) for v in qvals)/max(len(qvals),1),'conv_at_k':sum(sum(v)>=math.ceil(len(v)/2) for v in qvals)/max(len(qvals),1),'pass_at_k':sum(any(v) for v in qvals)/max(len(qvals),1),'note':'Local rule score; retain a separate semantic-judge score for synonym-heavy VQA.'}
 out=Path(args.output) if args.output else path.with_name(path.stem+'_vqa_metrics.json');write_json(out,{'summary':summary,'manifest':manifest,'per_run':per});print(json.dumps(summary,ensure_ascii=False,indent=2))
 if invalid and args.fail_on_incomplete:raise SystemExit(3)
if __name__=='__main__':main()
